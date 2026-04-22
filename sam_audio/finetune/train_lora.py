import argparse
import json
import random
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio.finetune.json_dataset import (
    PromptedAudioSeparationJsonDataset,
    collate_json_separation_samples,
)
from sam_audio.finetune.lora import (
    apply_lora,
    count_trainable_parameters,
    freeze_module_parameters,
    save_lora_adapter,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for SAM-Audio using JSON/JSONL prompt+audio records."
    )
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path)
    parser.add_argument(
        "--checkpoint-path", type=str, default="facebook/sam-audio-small"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-prefixes",
        type=str,
        default="transformer",
        help="Comma-separated module prefixes to adapt. Default: transformer",
    )
    parser.add_argument(
        "--lora-targets",
        type=str,
        default="",
        help="Optional comma-separated leaf module names (for example: wq,wk,wv,wo). Empty means all Linear layers under the selected prefixes.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--device", type=str)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def choose_autocast(device: torch.device, dtype_name: str):
    if device.type != "cuda":
        return nullcontext()
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_name]
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def set_frozen_modules_eval(model: SAMAudio):
    for module_name in (
        "audio_codec",
        "text_encoder",
        "vision_encoder",
        "span_predictor",
        "visual_ranker",
        "text_ranker",
    ):
        module = getattr(model, module_name, None)
        if module is not None:
            module.eval()


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(prediction.dtype).unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0) * prediction.size(-1)
    return ((prediction - target).pow(2) * mask).sum() / denom


def compute_flow_matching_loss(
    model: SAMAudio,
    batch_dict: dict,
    device: torch.device,
) -> torch.Tensor:
    input_batch = batch_dict["input_batch"].to(device)
    target_audio = batch_dict["target_audio"].to(device)
    residual_audio = batch_dict["residual_audio"].to(device)

    with torch.no_grad():
        forward_args = model._get_forward_args(input_batch)
        target_features = model.audio_codec(target_audio)
        residual_features = model.audio_codec(residual_audio)

    clean_features = torch.cat([target_features, residual_features], dim=1).transpose(
        1, 2
    )
    noise = torch.randn_like(clean_features)
    time = torch.rand(clean_features.size(0), device=device, dtype=clean_features.dtype)
    noisy_audio = (
        (1.0 - time[:, None, None]) * noise + time[:, None, None] * clean_features
    )
    velocity_target = clean_features - noise

    prediction = model.forward(
        noisy_audio=noisy_audio,
        time=time,
        **forward_args,
    )
    return masked_mse_loss(
        prediction=prediction,
        target=velocity_target,
        mask=input_batch.audio_pad_mask,
    )


def evaluate(
    model: SAMAudio,
    dataloader: DataLoader,
    device: torch.device,
    dtype_name: str,
) -> float:
    model.eval()
    set_frozen_modules_eval(model)

    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch_dict in dataloader:
            with choose_autocast(device, dtype_name):
                loss = compute_flow_matching_loss(model, batch_dict, device)
            total_loss += float(loss.item())
            total_batches += 1
    return total_loss / max(total_batches, 1)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)

    processor = SAMAudioProcessor.from_pretrained(args.checkpoint_path)
    if args.sample_rate != processor.audio_sampling_rate:
        raise ValueError(
            f"--sample-rate must match the checkpoint sample rate ({processor.audio_sampling_rate})"
        )
    train_dataset = PromptedAudioSeparationJsonDataset(
        args.train_json, sample_rate=args.sample_rate
    )
    if len(train_dataset) == 0:
        raise ValueError(f"No training records found in {args.train_json}")
    eval_dataset = (
        PromptedAudioSeparationJsonDataset(args.eval_json, sample_rate=args.sample_rate)
        if args.eval_json is not None
        else None
    )

    collate_fn = partial(collate_json_separation_samples, processor=processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    eval_loader = (
        DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        if eval_dataset is not None
        else None
    )

    model = SAMAudio.from_pretrained(args.checkpoint_path).to(device)
    freeze_module_parameters(model)

    prefixes = [item.strip() for item in args.lora_prefixes.split(",") if item.strip()]
    targets = [item.strip() for item in args.lora_targets.split(",") if item.strip()] or None
    adapted_modules = apply_lora(
        model,
        rank=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        prefixes=prefixes,
        target_names=targets,
    )
    if len(adapted_modules) == 0:
        raise ValueError(
            "No Linear modules matched the requested LoRA configuration. "
            f"prefixes={prefixes}, targets={targets}"
        )

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and args.dtype == "float16"
    )

    trainable_count, total_count = count_trainable_parameters(model)
    with open(args.output_dir / "train_args.json", "w") as fout:
        json.dump(vars(args), fout, indent=2, sort_keys=True, default=str)

    print(
        json.dumps(
            {
                "device": str(device),
                "train_records": len(train_dataset),
                "eval_records": len(eval_dataset) if eval_dataset is not None else 0,
                "adapted_modules": len(adapted_modules),
                "trainable_parameters": trainable_count,
                "total_parameters": total_count,
            },
            indent=2,
        )
    )

    global_step = 0
    accumulated_batches = 0
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        set_frozen_modules_eval(model)

        running_loss = 0.0
        running_batches = 0
        for batch_idx, batch_dict in enumerate(train_loader, start=1):
            with choose_autocast(device, args.dtype):
                loss = compute_flow_matching_loss(model, batch_dict, device)
                scaled_loss = loss / args.grad_accum_steps

            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.item())
            running_batches += 1
            accumulated_batches += 1

            if accumulated_batches % args.grad_accum_steps == 0:
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, max_norm=args.grad_clip_norm
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accumulated_batches = 0

                if global_step % args.log_every == 0:
                    elapsed = time.time() - start_time
                    avg_loss = running_loss / max(running_batches, 1)
                    print(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "step": global_step,
                                "train_loss": round(avg_loss, 6),
                                "elapsed_sec": round(elapsed, 1),
                            }
                        )
                    )
                    running_loss = 0.0
                    running_batches = 0

                if args.eval_every > 0 and eval_loader is not None:
                    if global_step % args.eval_every == 0:
                        eval_loss = evaluate(model, eval_loader, device, args.dtype)
                        print(
                            json.dumps(
                                {
                                    "step": global_step,
                                    "eval_loss": round(eval_loss, 6),
                                }
                            )
                        )
                        model.train()
                        set_frozen_modules_eval(model)

                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_lora_adapter(
                        model,
                        args.output_dir / f"step-{global_step}",
                        checkpoint_path=args.checkpoint_path,
                        module_names=adapted_modules,
                        rank=args.lora_r,
                        alpha=args.lora_alpha,
                        dropout=args.lora_dropout,
                    )

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if accumulated_batches > 0 and (args.max_steps <= 0 or global_step < args.max_steps):
        if args.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    final_eval = None
    if eval_loader is not None:
        final_eval = evaluate(model, eval_loader, device, args.dtype)

    save_lora_adapter(
        model,
        args.output_dir / "final",
        checkpoint_path=args.checkpoint_path,
        module_names=adapted_modules,
        rank=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    summary = {
        "final_step": global_step,
        "train_seconds": round(time.time() - start_time, 1),
        "final_eval_loss": None if final_eval is None else round(final_eval, 6),
        "adapter_dir": str(args.output_dir / "final"),
    }
    with open(args.output_dir / "summary.json", "w") as fout:
        json.dump(summary, fout, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
