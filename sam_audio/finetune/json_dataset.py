import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from sam_audio.finetune.data import (
    align_audio_pair,
    load_audio_mono,
    resolve_audio_reference,
)


JsonRecord = dict[str, Any]


def load_json_records(path: str | Path) -> list[JsonRecord]:
    path = Path(path)
    suffix = path.suffix.lower()
    with open(path) as fin:
        if suffix == ".jsonl":
            records = [json.loads(line) for line in fin if line.strip()]
        else:
            payload = json.load(fin)
            if isinstance(payload, list):
                records = payload
            elif isinstance(payload, dict):
                if "data" in payload and isinstance(payload["data"], list):
                    records = payload["data"]
                elif "records" in payload and isinstance(payload["records"], list):
                    records = payload["records"]
                else:
                    raise ValueError(
                        f"Expected a JSON array or a dict with 'data'/'records' in {path}"
                    )
            else:
                raise ValueError(f"Unsupported JSON payload in {path}")
    return records


def _get_required(record: JsonRecord, key: str) -> Any:
    if key not in record:
        raise ValueError(f"Missing required field {key!r} in record: {record}")
    return record[key]


def parse_span_field(span_field: Any) -> list[tuple[str, float, float]]:
    if span_field in (None, "", []):
        return []

    if isinstance(span_field, dict):
        anchors = []
        for key, token in (("positive", "+"), ("negative", "-")):
            for start, end in span_field.get(key, []):
                anchors.append((token, float(start), float(end)))
        anchors.sort(key=lambda item: (item[1], item[2], item[0]))
        return anchors

    if not isinstance(span_field, list):
        raise ValueError(
            f"Unsupported span field {span_field!r}. Expected list or dict."
        )

    anchors = []
    for item in span_field:
        if isinstance(item, dict):
            token = str(item.get("token", "+"))
            start = float(item["start"])
            end = float(item["end"])
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            token = "+"
            start, end = item
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            token, start, end = item
        else:
            raise ValueError(
                f"Unsupported span item {item!r}. Expected [start, end] or [token, start, end]."
            )
        anchors.append((str(token), float(start), float(end)))

    anchors.sort(key=lambda item: (item[1], item[2], item[0]))
    return anchors


class PromptedAudioSeparationJsonDataset(torch.utils.data.Dataset):
    """Dataset for JSON/JSONL records with:
    - audio-url
    - span
    - prompt
    - groundtruth-audio-url
    """

    def __init__(
        self,
        manifest_path: str | Path,
        sample_rate: int = 48_000,
    ):
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.records = load_json_records(self.manifest_path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        mixture_ref = resolve_audio_reference(
            self.manifest_path, _get_required(record, "audio-url")
        )
        target_ref = resolve_audio_reference(
            self.manifest_path, _get_required(record, "groundtruth-audio-url")
        )
        mixture = load_audio_mono(mixture_ref, sample_rate=self.sample_rate)
        target = load_audio_mono(target_ref, sample_rate=self.sample_rate)
        mixture, target = align_audio_pair(mixture, target)
        residual = mixture - target

        return {
            "id": record.get("id", f"sample-{idx}"),
            "prompt": str(_get_required(record, "prompt")),
            "anchors": parse_span_field(record.get("span")),
            "mixture": mixture,
            "target": target,
            "residual": residual,
            "audio_url": str(mixture_ref),
            "groundtruth_audio_url": str(target_ref),
            "metadata": record.get("metadata", {}),
        }


def collate_json_separation_samples(
    samples: list[dict[str, Any]],
    processor,
) -> dict[str, Any]:
    descriptions = [sample["prompt"] for sample in samples]
    mixtures = [sample["mixture"].unsqueeze(0) for sample in samples]
    anchors = [sample["anchors"] for sample in samples]
    has_any_anchor = any(len(anchor_list) > 0 for anchor_list in anchors)

    batch = processor(
        descriptions=descriptions,
        audios=mixtures,
        anchors=anchors if has_any_anchor else None,
    )

    target_sizes = torch.tensor([sample["target"].numel() for sample in samples])
    residual_sizes = torch.tensor([sample["residual"].numel() for sample in samples])
    targets = pad_sequence([sample["target"] for sample in samples], batch_first=True)
    residuals = pad_sequence(
        [sample["residual"] for sample in samples], batch_first=True
    )

    return {
        "input_batch": batch,
        "target_audio": targets.unsqueeze(1),
        "target_sizes": target_sizes,
        "residual_audio": residuals.unsqueeze(1),
        "residual_sizes": residual_sizes,
        "records": samples,
    }
