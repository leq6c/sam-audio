#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio.finetune.json_dataset import (
    PromptedAudioSeparationJsonDataset,
    collate_json_separation_samples,
)
from sam_audio.finetune.lora import freeze_module_parameters, load_lora_adapter
from sam_audio.finetune.train_lora import masked_mse_loss, set_frozen_modules_eval


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark validation/eval speed for SAM-Audio LoRA fine-tuning."
    )
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--device", type=str)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def choose_autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


class InMemorySampleDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def load_model(checkpoint_path: str, adapter_dir: Path | None, device: torch.device):
    model = SAMAudio.from_pretrained(checkpoint_path).to(device)
    freeze_module_parameters(model)
    if adapter_dir is not None:
        load_lora_adapter(model, adapter_dir, map_location="cpu")
    model.eval()
    set_frozen_modules_eval(model)
    return model


def prepare_subset(dataset_path: Path, sample_rate: int, max_samples: int):
    dataset = PromptedAudioSeparationJsonDataset(dataset_path, sample_rate=sample_rate)
    count = min(len(dataset), max_samples)
    subset = Subset(dataset, range(count))
    samples = [subset[i] for i in range(count)]
    return subset, samples


def current_eval_pass(
    model: SAMAudio,
    processor: SAMAudioProcessor,
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    device: torch.device,
) -> float:
    collate_fn = partial(collate_json_separation_samples, processor=processor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch_dict in loader:
            input_batch = batch_dict["input_batch"].to(device)
            target_audio = batch_dict["target_audio"].to(device)
            residual_audio = batch_dict["residual_audio"].to(device)
            with choose_autocast(device):
                with torch.no_grad():
                    forward_args = model._get_forward_args(input_batch)
                    target_features = model.audio_codec(target_audio)
                    residual_features = model.audio_codec(residual_audio)
                clean_features = torch.cat(
                    [target_features, residual_features], dim=1
                ).transpose(1, 2)
                noise = torch.randn_like(clean_features)
                time_t = torch.rand(
                    clean_features.size(0),
                    device=device,
                    dtype=clean_features.dtype,
                )
                noisy_audio = (
                    (1.0 - time_t[:, None, None]) * noise
                    + time_t[:, None, None] * clean_features
                )
                velocity_target = clean_features - noise
                prediction = model.forward(
                    noisy_audio=noisy_audio,
                    time=time_t,
                    **forward_args,
                )
                loss = masked_mse_loss(
                    prediction=prediction,
                    target=velocity_target,
                    mask=input_batch.audio_pad_mask,
                )
            total_loss += float(loss.item())
            total_batches += 1
    return total_loss / max(total_batches, 1)


def build_frozen_cache(
    model: SAMAudio,
    processor: SAMAudioProcessor,
    samples: list[dict],
    *,
    batch_size: int,
    device: torch.device,
) -> list[dict]:
    dataset = InMemorySampleDataset(samples)
    collate_fn = partial(collate_json_separation_samples, processor=processor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    cached = []
    with torch.no_grad():
        for batch_dict in loader:
            input_batch = batch_dict["input_batch"].to(device)
            target_audio = batch_dict["target_audio"].to(device)
            residual_audio = batch_dict["residual_audio"].to(device)
            forward_args = {
                key: (None if value is None else value.detach())
                for key, value in model._get_forward_args(input_batch).items()
            }
            clean_features = torch.cat(
                [model.audio_codec(target_audio), model.audio_codec(residual_audio)], dim=1
            ).transpose(1, 2)
            cached.append(
                {
                    "forward_args": forward_args,
                    "clean_features": clean_features.detach(),
                    "audio_pad_mask": input_batch.audio_pad_mask.detach(),
                }
            )
    return cached


def cached_eval_pass(model: SAMAudio, cached_batches: list[dict], device: torch.device) -> float:
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in cached_batches:
            clean_features = batch["clean_features"]
            forward_args = batch["forward_args"]
            with choose_autocast(device):
                noise = torch.randn_like(clean_features)
                time_t = torch.rand(
                    clean_features.size(0),
                    device=device,
                    dtype=clean_features.dtype,
                )
                noisy_audio = (
                    (1.0 - time_t[:, None, None]) * noise
                    + time_t[:, None, None] * clean_features
                )
                velocity_target = clean_features - noise
                prediction = model.forward(
                    noisy_audio=noisy_audio,
                    time=time_t,
                    **forward_args,
                )
                loss = masked_mse_loss(
                    prediction=prediction,
                    target=velocity_target,
                    mask=batch["audio_pad_mask"],
                )
            total_loss += float(loss.item())
            total_batches += 1
    return total_loss / max(total_batches, 1)


def benchmark(name: str, fn, repeats: int, device: torch.device):
    durations = []
    losses = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        loss = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        losses.append(loss)
    mean_sec = sum(durations) / len(durations)
    return {
        "name": name,
        "repeats": repeats,
        "mean_sec": round(mean_sec, 4),
        "min_sec": round(min(durations), 4),
        "max_sec": round(max(durations), 4),
        "samples_per_sec": round(math.inf if mean_sec == 0 else (len(losses) * 0 + 1), 4),
        "loss_mean": round(sum(losses) / len(losses), 6),
        "durations_sec": [round(x, 4) for x in durations],
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    processor = SAMAudioProcessor.from_pretrained(args.checkpoint_path)
    raw_subset, samples = prepare_subset(args.json, args.sample_rate, args.max_samples)
    in_memory_dataset = InMemorySampleDataset(samples)
    model = load_model(args.checkpoint_path, args.adapter_dir, device)

    current_bs1 = benchmark(
        "current_bs1_nw4",
        lambda: current_eval_pass(
            model,
            processor,
            raw_subset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            device=device,
        ),
        args.repeats,
        device,
    )
    current_bs4 = benchmark(
        "current_bs4_nw4",
        lambda: current_eval_pass(
            model,
            processor,
            raw_subset,
            batch_size=4,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            device=device,
        ),
        args.repeats,
        device,
    )
    current_bs8 = benchmark(
        "current_bs8_nw8",
        lambda: current_eval_pass(
            model,
            processor,
            raw_subset,
            batch_size=8,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            device=device,
        ),
        args.repeats,
        device,
    )
    in_memory_bs8 = benchmark(
        "in_memory_bs8_nw0",
        lambda: current_eval_pass(
            model,
            processor,
            in_memory_dataset,
            batch_size=8,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            device=device,
        ),
        args.repeats,
        device,
    )

    cached_batches = build_frozen_cache(
        model,
        processor,
        samples,
        batch_size=8,
        device=device,
    )
    cached_gpu = benchmark(
        "cached_frozen_gpu_bs8",
        lambda: cached_eval_pass(model, cached_batches, device),
        args.repeats,
        device,
    )

    results = {
        "device": str(device),
        "max_samples": len(samples),
        "results": [current_bs1, current_bs4, current_bs8, in_memory_bs8, cached_gpu],
    }

    baseline = current_bs1["mean_sec"]
    for item in results["results"]:
        item["samples_per_sec"] = round(len(samples) / item["mean_sec"], 4)
        item["speedup_vs_current_bs1"] = round(baseline / item["mean_sec"], 4)

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
