import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

from sam_audio.finetune.data import (
    compute_peak_scale,
    generate_activity_spans,
    load_stereo_stems,
)


GENDER_TO_DESCRIPTION = {
    "female": "woman speaking",
    "woman": "woman speaking",
    "f": "woman speaking",
    "male": "man speaking",
    "man": "man speaking",
    "m": "man speaking",
    "child": "child speaking",
    "kid": "child speaking",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a SAM-Audio-style fine-tuning manifest from stereo files where "
            "left/right channels are already separated speakers."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--glob", type=str, default="*.wav")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--default-description", type=str, default="single speaker")
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--min-silence-ms", type=int, default=250)
    parser.add_argument("--min-sounding-ms", type=int, default=250)
    parser.add_argument("--silence-thresh-db", type=float, default=-40.0)
    parser.add_argument("--max-peak", type=float, default=0.99)
    parser.add_argument("--add-negative-spans", action="store_true")
    return parser.parse_args()


def normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip().lower()
    return value or None


def load_metadata(metadata_csv: Optional[Path]) -> dict[str, dict[str, str]]:
    if metadata_csv is None:
        return {}

    result = {}
    with open(metadata_csv) as fin:
        reader = csv.DictReader(fin)
        if "path" not in (reader.fieldnames or []):
            raise ValueError("metadata csv must include a 'path' column")
        for row in reader:
            path = row["path"].strip()
            if not path:
                continue
            result[path] = row
            result[Path(path).as_posix()] = row
            result[Path(path).name] = row
    return result


def lookup_metadata(
    metadata_by_path: dict[str, dict[str, str]],
    relpath: Path,
) -> dict[str, str]:
    for key in (relpath.as_posix(), str(relpath), relpath.name):
        if key in metadata_by_path:
            return metadata_by_path[key]
    return {}


def infer_description(
    default_description: str,
    metadata: dict[str, str],
    speaker_idx: int,
) -> str:
    prompt = normalize_text(metadata.get(f"prompt_{speaker_idx}"))
    if prompt is not None:
        return prompt

    gender = normalize_text(metadata.get(f"gender_{speaker_idx}"))
    if gender is not None and gender in GENDER_TO_DESCRIPTION:
        return GENDER_TO_DESCRIPTION[gender]

    return default_description


def infer_split(relpath: Path, metadata: dict[str, str], val_ratio: float) -> str:
    split = normalize_text(metadata.get("split"))
    if split is not None:
        return split
    if val_ratio <= 0.0:
        return "train"

    digest = hashlib.sha1(relpath.as_posix().encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def build_record(
    relpath: Path,
    description: str,
    split: str,
    sample_rate: int,
    target_channel: int,
    residual_channel: int,
    duration_sec: float,
    scale: float,
    anchors: list[list[str | float]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    record_id = relpath.with_suffix("").as_posix().replace("/", "__")
    return {
        "id": f"{record_id}-spk{target_channel}",
        "task": "speaker_extraction",
        "split": split,
        "source_format": "stereo_stems",
        "source_path": relpath.as_posix(),
        "sample_rate": sample_rate,
        "target_channel": target_channel,
        "residual_channel": residual_channel,
        "description": description,
        "anchors": anchors,
        "duration_sec": round(duration_sec, 4),
        "scale": round(scale, 8),
        "metadata": metadata,
    }


def main():
    args = parse_args()
    metadata_by_path = load_metadata(args.metadata_csv)
    files = sorted(args.input_dir.rglob(args.glob))
    files = [path for path in files if path.is_file()]
    if len(files) == 0:
        raise ValueError(f"No audio files matched {args.glob} under {args.input_dir}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.output_jsonl.resolve().parent

    record_count = 0
    with open(args.output_jsonl, "w") as fout:
        for path in files:
            input_relpath = path.relative_to(args.input_dir)
            relpath = Path(os.path.relpath(path.resolve(), manifest_dir))
            metadata = lookup_metadata(metadata_by_path, input_relpath)
            split = infer_split(input_relpath, metadata, args.val_ratio)

            wav = load_stereo_stems(path, sample_rate=args.sample_rate)
            speaker0 = wav[0]
            speaker1 = wav[1]
            scale = compute_peak_scale(
                speaker0,
                speaker1,
                max_peak=args.max_peak,
            )
            duration_sec = speaker0.numel() / args.sample_rate

            speaker_wavs = [speaker0, speaker1]
            for target_channel in range(2):
                residual_channel = 1 - target_channel
                description = infer_description(
                    default_description=args.default_description,
                    metadata=metadata,
                    speaker_idx=target_channel,
                )
                anchors = generate_activity_spans(
                    speaker_wavs[target_channel],
                    sample_rate=args.sample_rate,
                    silence_thresh_db=args.silence_thresh_db,
                    min_silence_ms=args.min_silence_ms,
                    min_sounding_ms=args.min_sounding_ms,
                    token="+",
                )
                if args.add_negative_spans:
                    anchors.extend(
                        generate_activity_spans(
                            speaker_wavs[residual_channel],
                            sample_rate=args.sample_rate,
                            silence_thresh_db=args.silence_thresh_db,
                            min_silence_ms=args.min_silence_ms,
                            min_sounding_ms=args.min_sounding_ms,
                            token="-",
                        )
                    )
                anchors.sort(key=lambda x: (float(x[1]), float(x[2]), str(x[0])))
                record = build_record(
                    relpath=relpath,
                    description=description,
                    split=split,
                    sample_rate=args.sample_rate,
                    target_channel=target_channel,
                    residual_channel=residual_channel,
                    duration_sec=duration_sec,
                    scale=scale,
                    anchors=anchors,
                    metadata={
                        "speaker_index": target_channel,
                        "raw_metadata_path": metadata.get("path"),
                        "gender": normalize_text(metadata.get(f"gender_{target_channel}")),
                    },
                )
                print(json.dumps(record), file=fout)
                record_count += 1

    print(
        json.dumps(
            {
                "files": len(files),
                "records": record_count,
                "manifest": str(args.output_jsonl),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
