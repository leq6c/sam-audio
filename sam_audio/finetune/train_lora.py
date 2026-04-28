import argparse
import json
import os
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="Validation batch size. Defaults to --batch-size when set to 0.",
    )
    parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=-1,
        help="Validation worker count. Defaults to --num-workers when set to -1.",
    )
    parser.add_argument(
        "--eval-cache",
        choices=["none", "cpu", "gpu"],
        default="cpu",
        help=(
            "Cache frozen validation features once and reuse them for later evals. "
            "'cpu' is the safest default."
        ),
    )
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
    parser.add_argument(
        "--use-rslora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use rank-stabilized LoRA scaling (alpha / sqrt(r)) instead of alpha / r.",
    )
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


def append_jsonl_line(file_obj, payload: dict):
    file_obj.write(json.dumps(payload, sort_keys=True) + "\n")
    file_obj.flush()


def make_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    collate_fn,
    pin_memory: bool,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def cache_tensor(value: Optional[torch.Tensor], cache_device: torch.device):
    if value is None:
        return None
    value = value.detach()
    if cache_device.type == "cpu":
        return value.cpu()
    return value.to(cache_device)


def move_tensor(value: Optional[torch.Tensor], device: torch.device):
    if value is None:
        return None
    return value.to(device, non_blocking=True)


def eval_cache_supported(adapted_modules: list[str]) -> bool:
    forbidden_prefixes = ("audio_codec", "text_encoder", "vision_encoder")
    return not any(
        module_name.startswith(forbidden_prefixes) for module_name in adapted_modules
    )


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


def compute_cached_flow_matching_loss(
    model: SAMAudio,
    cached_batch: dict,
    device: torch.device,
) -> torch.Tensor:
    forward_args = {
        key: move_tensor(value, device)
        for key, value in cached_batch["forward_args"].items()
    }
    clean_features = move_tensor(cached_batch["clean_features"], device)
    audio_pad_mask = move_tensor(cached_batch["audio_pad_mask"], device)

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
        mask=audio_pad_mask,
    )


def build_eval_cache(
    model: SAMAudio,
    dataloader: DataLoader,
    device: torch.device,
    cache_mode: str,
) -> list[dict]:
    cache_device = torch.device("cpu" if cache_mode == "cpu" else device.type)
    model.eval()
    set_frozen_modules_eval(model)

    cached_batches = []
    with torch.no_grad():
        for batch_dict in dataloader:
            input_batch = batch_dict["input_batch"].to(device)
            target_audio = batch_dict["target_audio"].to(device, non_blocking=True)
            residual_audio = batch_dict["residual_audio"].to(device, non_blocking=True)

            forward_args = {
                key: cache_tensor(value, cache_device)
                for key, value in model._get_forward_args(input_batch).items()
            }
            target_features = model.audio_codec(target_audio)
            residual_features = model.audio_codec(residual_audio)
            clean_features = torch.cat(
                [target_features, residual_features], dim=1
            ).transpose(1, 2)

            cached_batches.append(
                {
                    "forward_args": forward_args,
                    "clean_features": cache_tensor(clean_features, cache_device),
                    "audio_pad_mask": cache_tensor(
                        input_batch.audio_pad_mask, cache_device
                    ),
                }
            )
    return cached_batches


def evaluate(
    model: SAMAudio,
    dataloader: Optional[DataLoader],
    cached_batches: Optional[list[dict]],
    device: torch.device,
    dtype_name: str,
) -> float:
    model.eval()
    set_frozen_modules_eval(model)

    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        if cached_batches is not None:
            for cached_batch in cached_batches:
                with choose_autocast(device, dtype_name):
                    loss = compute_cached_flow_matching_loss(
                        model, cached_batch, device
                    )
                total_loss += float(loss.item())
                total_batches += 1
        else:
            assert dataloader is not None
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
    eval_batch_size = args.batch_size if args.eval_batch_size <= 0 else args.eval_batch_size
    eval_num_workers = args.num_workers if args.eval_num_workers < 0 else args.eval_num_workers

    collate_fn = partial(collate_json_separation_samples, processor=processor)
    pin_memory = device.type == "cuda"
    train_loader = make_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    eval_loader = (
        make_dataloader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=eval_num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
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
        use_rslora=args.use_rslora,
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
    eval_cache_reason = None
    use_eval_cache = (
        eval_loader is not None
        and args.eval_cache != "none"
        and eval_cache_supported(adapted_modules)
    )
    if eval_loader is None:
        eval_cache_reason = "no_eval_dataset"
    elif args.eval_cache == "none":
        eval_cache_reason = "disabled"
    elif not eval_cache_supported(adapted_modules):
        eval_cache_reason = "unsupported_adapted_modules"
    else:
        eval_cache_reason = args.eval_cache
    eval_cached_batches = (
        build_eval_cache(model, eval_loader, device, args.eval_cache)
        if use_eval_cache
        else None
    )
    with open(args.output_dir / "train_args.json", "w") as fout:
        json.dump(vars(args), fout, indent=2, sort_keys=True, default=str)

    print(
        json.dumps(
            {
                "device": str(device),
                "train_records": len(train_dataset),
                "eval_records": len(eval_dataset) if eval_dataset is not None else 0,
                "eval_batch_size": eval_batch_size,
                "eval_num_workers": eval_num_workers,
                "eval_cache": args.eval_cache if use_eval_cache else "none",
                "eval_cache_reason": eval_cache_reason,
                "use_rslora": args.use_rslora,
                "adapted_modules": len(adapted_modules),
                "trainable_parameters": trainable_count,
                "total_parameters": total_count,
            },
            indent=2,
        )
    )

    train_metrics_path = args.output_dir / "train_metrics.jsonl"
    eval_metrics_path = args.output_dir / "eval_metrics.jsonl"

    train_metrics_fout = open(train_metrics_path, "w")
    eval_metrics_fout = open(eval_metrics_path, "w") if eval_loader is not None else None
    try:
        global_step = 0
        accumulated_batches = 0
        optimizer.zero_grad(set_to_none=True)
        start_time = time.time()
        step_loss_sum = 0.0
        step_loss_batches = 0
        best_eval_loss = None

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
                loss_value = float(loss.item())
                running_loss += loss_value
                running_batches += 1
                step_loss_sum += loss_value
                step_loss_batches += 1
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

                    elapsed = time.time() - start_time
                    step_loss = step_loss_sum / max(step_loss_batches, 1)
                    append_jsonl_line(
                        train_metrics_fout,
                        {
                            "event": "train_step",
                            "epoch": epoch + 1,
                            "step": global_step,
                            "batch_idx": batch_idx,
                            "grad_accum_batches": step_loss_batches,
                            "train_loss": round(step_loss, 6),
                            "lr": optimizer.param_groups[0]["lr"],
                            "elapsed_sec": round(elapsed, 1),
                        },
                    )
                    step_loss_sum = 0.0
                    step_loss_batches = 0

                    if global_step % args.log_every == 0:
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
                            eval_loss = evaluate(
                                model,
                                eval_loader,
                                eval_cached_batches,
                                device,
                                args.dtype,
                            )
                            eval_payload = {
                                "event": "eval_step",
                                "epoch": epoch + 1,
                                "step": global_step,
                                "eval_loss": round(eval_loss, 6),
                                "elapsed_sec": round(time.time() - start_time, 1),
                            }
                            print(json.dumps({"step": global_step, "eval_loss": round(eval_loss, 6)}))
                            append_jsonl_line(eval_metrics_fout, eval_payload)
                            if best_eval_loss is None or eval_loss < best_eval_loss:
                                best_eval_loss = eval_loss
                                save_lora_adapter(
                                    model,
                                    args.output_dir / "best",
                                    checkpoint_path=args.checkpoint_path,
                                    module_names=adapted_modules,
                                    rank=args.lora_r,
                                    alpha=args.lora_alpha,
                                    dropout=args.lora_dropout,
                                    use_rslora=args.use_rslora,
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
                            use_rslora=args.use_rslora,
                        )

                    if args.max_steps > 0 and global_step >= args.max_steps:
                        break

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if accumulated_batches > 0 and (
            args.max_steps <= 0 or global_step < args.max_steps
        ):
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=args.grad_clip_norm
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            elapsed = time.time() - start_time
            step_loss = step_loss_sum / max(step_loss_batches, 1)
            append_jsonl_line(
                train_metrics_fout,
                {
                    "event": "train_step",
                    "epoch": args.epochs,
                    "step": global_step,
                    "batch_idx": None,
                    "grad_accum_batches": step_loss_batches,
                    "train_loss": round(step_loss, 6),
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_sec": round(elapsed, 1),
                },
            )

        final_eval = None
        if eval_loader is not None:
            final_eval = evaluate(
                model,
                eval_loader,
                eval_cached_batches,
                device,
                args.dtype,
            )
            append_jsonl_line(
                eval_metrics_fout,
                {
                    "event": "eval_final",
                    "step": global_step,
                    "eval_loss": round(final_eval, 6),
                    "elapsed_sec": round(time.time() - start_time, 1),
                },
            )
            if best_eval_loss is None or final_eval < best_eval_loss:
                best_eval_loss = final_eval
                save_lora_adapter(
                    model,
                    args.output_dir / "best",
                    checkpoint_path=args.checkpoint_path,
                    module_names=adapted_modules,
                    rank=args.lora_r,
                    alpha=args.lora_alpha,
                    dropout=args.lora_dropout,
                    use_rslora=args.use_rslora,
                )
    finally:
        train_metrics_fout.close()
        if eval_metrics_fout is not None:
            eval_metrics_fout.close()

    save_lora_adapter(
        model,
        args.output_dir / "final",
        checkpoint_path=args.checkpoint_path,
        module_names=adapted_modules,
        rank=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        use_rslora=args.use_rslora,
    )

    summary = {
        "final_step": global_step,
        "train_seconds": round(time.time() - start_time, 1),
        "final_eval_loss": None if final_eval is None else round(final_eval, 6),
        "best_eval_loss": None if best_eval_loss is None else round(best_eval_loss, 6),
        "adapter_dir": str(args.output_dir / "final"),
    }
    with open(args.output_dir / "summary.json", "w") as fout:
        json.dump(summary, fout, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
