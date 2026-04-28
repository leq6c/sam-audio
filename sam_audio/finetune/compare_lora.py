import argparse
import json
import statistics
from pathlib import Path

import soundfile as sf
import torch

from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio.finetune.json_dataset import PromptedAudioSeparationJsonDataset
from sam_audio.finetune.lora import freeze_module_parameters, load_lora_adapter


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare base SAM-Audio and a LoRA adapter on a JSON/JSONL dataset, "
            "and export the first sample plus the best-improved sample."
        )
    )
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N scanned samples. Set 0 to disable.",
    )
    return parser.parse_args()


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_batch(processor: SAMAudioProcessor, sample: dict):
    anchors = sample["anchors"] if len(sample["anchors"]) > 0 else None
    return processor(
        descriptions=[sample["prompt"]],
        audios=[sample["mixture"].unsqueeze(0)],
        anchors=[anchors] if anchors is not None else None,
    )


def si_sdr(reference: torch.Tensor, estimation: torch.Tensor) -> float:
    reference = reference.detach().float().cpu()
    estimation = estimation.detach().float().cpu()
    n = min(reference.numel(), estimation.numel())
    reference = reference[:n]
    estimation = estimation[:n]
    reference = reference - reference.mean()
    estimation = estimation - estimation.mean()
    ref_energy = torch.sum(reference**2).clamp_min(1e-8)
    projection = torch.sum(estimation * reference) * reference / ref_energy
    noise = estimation - projection
    ratio = torch.sum(projection**2).clamp_min(1e-8) / torch.sum(noise**2).clamp_min(1e-8)
    return float(10.0 * torch.log10(ratio))


def save_wav(path: Path, wav: torch.Tensor, sample_rate: int):
    wav = wav.detach().float().cpu().clamp(-1.0, 1.0)
    if wav.ndim > 1:
        wav = wav.squeeze(0)
    sf.write(str(path), wav.numpy(), sample_rate)


def load_models(checkpoint_path: str, adapter_dir: Path, device: torch.device):
    base = SAMAudio.from_pretrained(checkpoint_path).to(device)
    freeze_module_parameters(base)
    adapted = SAMAudio.from_pretrained(checkpoint_path).to(device)
    freeze_module_parameters(adapted)
    load_lora_adapter(adapted, adapter_dir, map_location="cpu")
    return base, adapted


def compare_sample(
    *,
    processor: SAMAudioProcessor,
    base: SAMAudio,
    adapted: SAMAudio,
    sample: dict,
    idx: int,
    seed: int,
    device: torch.device,
):
    batch = build_batch(processor, sample).to(device)
    forward_args = base._get_forward_args(batch)
    torch.manual_seed(seed + idx)
    noise = torch.randn_like(forward_args["audio_features"])
    base_result = base.separate(batch, noise=noise.clone())
    adapted_result = adapted.separate(batch, noise=noise.clone())

    mixture = sample["mixture"]
    target = sample["target"]
    residual = sample["residual"]
    base_target = base_result.target[0]
    base_residual = base_result.residual[0]
    lora_target = adapted_result.target[0]
    lora_residual = adapted_result.residual[0]

    base_target_score = si_sdr(target, base_target)
    lora_target_score = si_sdr(target, lora_target)
    base_residual_score = si_sdr(residual, base_residual)
    lora_residual_score = si_sdr(residual, lora_residual)

    report = {
        "idx": idx,
        "sample_id": sample["id"],
        "prompt": sample["prompt"],
        "audio_url": sample["audio_url"],
        "groundtruth_audio_url": sample["groundtruth_audio_url"],
        "metadata": sample.get("metadata", {}),
        "metrics": {
            "mixture_vs_target_si_sdr": round(si_sdr(target, mixture), 4),
            "base_target_si_sdr": round(base_target_score, 4),
            "lora_target_si_sdr": round(lora_target_score, 4),
            "delta_target_si_sdr": round(lora_target_score - base_target_score, 4),
            "base_residual_vs_other_si_sdr": round(base_residual_score, 4),
            "lora_residual_vs_other_si_sdr": round(lora_residual_score, 4),
        },
    }
    audio = {
        "mixture": mixture,
        "groundtruth_target": target,
        "groundtruth_residual": residual,
        "base_target": base_target,
        "base_residual": base_residual,
        "lora_target": lora_target,
        "lora_residual": lora_residual,
    }
    return report, audio


def write_comparison(
    output_dir: Path,
    report: dict,
    audio: dict[str, torch.Tensor],
    sample_rate: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, wav in audio.items():
        save_wav(output_dir / f"{key}.wav", wav, sample_rate)

    report = dict(report)
    report["files"] = {
        "mixture": "mixture.wav",
        "groundtruth_target": "groundtruth_target.wav",
        "groundtruth_residual": "groundtruth_residual.wav",
        "base_target": "base_target.wav",
        "base_residual": "base_residual.wav",
        "lora_target": "lora_target.wav",
        "lora_residual": "lora_residual.wav",
    }
    with open(output_dir / "report.json", "w") as fout:
        json.dump(report, fout, indent=2, sort_keys=True)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    processor = SAMAudioProcessor.from_pretrained(args.checkpoint_path)
    if args.sample_rate != processor.audio_sampling_rate:
        raise ValueError(
            f"--sample-rate must match the checkpoint sample rate ({processor.audio_sampling_rate})"
        )

    dataset = PromptedAudioSeparationJsonDataset(args.json, sample_rate=args.sample_rate)
    if len(dataset) == 0:
        raise ValueError(f"No records found in {args.json}")

    base, adapted = load_models(args.checkpoint_path, args.adapter_dir, device)

    rows = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        report, _ = compare_sample(
            processor=processor,
            base=base,
            adapted=adapted,
            sample=sample,
            idx=idx,
            seed=args.seed,
            device=device,
        )
        rows.append(
            {
                "idx": idx,
                "sample_id": report["sample_id"],
                "metrics": report["metrics"],
            }
        )
        if args.progress_every > 0 and (
            (idx + 1) % args.progress_every == 0 or idx == len(dataset) - 1
        ):
            print(
                json.dumps(
                    {
                        "processed": idx + 1,
                        "total": len(dataset),
                        "last_sample_id": report["sample_id"],
                    }
                )
            )

    first_row = rows[0]
    best_row = max(rows, key=lambda row: row["metrics"]["delta_target_si_sdr"])
    selections = {"first": first_row, "best": best_row}

    deltas = [row["metrics"]["delta_target_si_sdr"] for row in rows]
    base_scores = [row["metrics"]["base_target_si_sdr"] for row in rows]
    lora_scores = [row["metrics"]["lora_target_si_sdr"] for row in rows]
    summary = {
        "count": len(rows),
        "mean_base_target_si_sdr": round(sum(base_scores) / len(base_scores), 4),
        "mean_lora_target_si_sdr": round(sum(lora_scores) / len(lora_scores), 4),
        "mean_delta_target_si_sdr": round(sum(deltas) / len(deltas), 4),
        "median_delta_target_si_sdr": round(statistics.median(deltas), 4),
        "improved_count": sum(1 for value in deltas if value > 0),
        "non_improved_count": sum(1 for value in deltas if value <= 0),
        "best_delta_target_si_sdr": round(max(deltas), 4),
        "worst_delta_target_si_sdr": round(min(deltas), 4),
    }

    selected_reports = {}
    for label, row in selections.items():
        sample = dataset[row["idx"]]
        report, audio = compare_sample(
            processor=processor,
            base=base,
            adapted=adapted,
            sample=sample,
            idx=row["idx"],
            seed=args.seed,
            device=device,
        )
        selected_reports[label] = report
        write_comparison(
            args.output_dir / label,
            report=report,
            audio=audio,
            sample_rate=processor.audio_sampling_rate,
        )

    index_payload = {
        "device": str(device),
        "seed": args.seed,
        "count": len(rows),
        "summary": summary,
        "first": selected_reports["first"],
        "best": selected_reports["best"],
        "rows": rows,
    }
    with open(args.output_dir / "index.json", "w") as fout:
        json.dump(index_payload, fout, indent=2, sort_keys=True)

    print(
        json.dumps(
            {
                "count": len(rows),
                "mean_delta_target_si_sdr": summary["mean_delta_target_si_sdr"],
                "improved_count": summary["improved_count"],
                "first_sample_id": selected_reports["first"]["sample_id"],
                "best_sample_id": selected_reports["best"]["sample_id"],
                "best_delta_target_si_sdr": selected_reports["best"]["metrics"][
                    "delta_target_si_sdr"
                ],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
