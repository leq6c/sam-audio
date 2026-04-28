import json
import wave
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from torch.nn.utils.rnn import pad_sequence


ManifestRecord = dict[str, Any]


def load_manifest_records(manifest_path: str | Path) -> list[ManifestRecord]:
    records = []
    with open(manifest_path) as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL record at line {line_no} in {manifest_path}"
                ) from exc
    return records


def resolve_manifest_path(manifest_path: str | Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(manifest_path).resolve().parent / path


def resolve_audio_reference(manifest_path: str | Path, audio_url: str) -> Path:
    parsed = urlparse(audio_url)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file":
            return Path(unquote(parsed.path))
        return resolve_manifest_path(manifest_path, audio_url)
    raise ValueError(
        f"Unsupported audio reference {audio_url!r}. Only local paths and file:// URLs are supported."
    )


def load_stereo_stems(
    audio_path: str | Path,
    sample_rate: int,
) -> torch.Tensor:
    wav, sr = load_audio(audio_path, sample_rate=None)
    if wav.size(0) != 2:
        raise ValueError(
            f"Expected stereo stems at {audio_path}, but found {wav.size(0)} channels"
        )
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


def _load_pcm_wav(audio_path: str | Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(audio_path), "rb") as fin:
        sample_width = fin.getsampwidth()
        if sample_width not in {2, 4}:
            raise ValueError(
                f"Unsupported WAV sample width {sample_width} bytes at {audio_path}"
            )
        frame_rate = fin.getframerate()
        num_channels = fin.getnchannels()
        frames = fin.readframes(fin.getnframes())

    dtype = np.int16 if sample_width == 2 else np.int32
    scale = float(2 ** (8 * sample_width - 1))
    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    audio = audio.reshape(-1, num_channels).T / scale
    return torch.from_numpy(audio), frame_rate


def _load_audio_with_pydub(audio_path: str | Path) -> tuple[torch.Tensor, int]:
    segment = AudioSegment.from_file(audio_path)
    sample_width = segment.sample_width
    if sample_width not in {1, 2, 4}:
        raise ValueError(
            f"Unsupported sample width {sample_width} bytes at {audio_path}"
        )

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
    scale = float(2 ** (8 * sample_width - 1))
    audio = np.array(segment.get_array_of_samples(), dtype=dtype).astype(np.float32)
    audio = audio.reshape(-1, segment.channels).T / scale
    return torch.from_numpy(audio), segment.frame_rate


def load_audio(
    audio_path: str | Path,
    sample_rate: Optional[int] = None,
    mono: bool = False,
) -> tuple[torch.Tensor, int]:
    try:
        wav, sr = torchaudio.load(audio_path)
    except Exception:
        suffix = Path(audio_path).suffix.lower()
        if suffix == ".wav":
            wav, sr = _load_pcm_wav(audio_path)
        else:
            wav, sr = _load_audio_with_pydub(audio_path)

    if mono and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sample_rate is not None and sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        sr = sample_rate

    return wav, sr


def load_audio_mono(
    audio_path: str | Path,
    sample_rate: int,
) -> torch.Tensor:
    wav, _ = load_audio(audio_path, sample_rate=sample_rate, mono=True)
    if wav.ndim == 1:
        return wav
    return wav.squeeze(0)


def align_audio_pair(
    mixture: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mixture_length = mixture.numel()
    if target.numel() > mixture_length:
        target = target[:mixture_length]
    elif target.numel() < mixture_length:
        target = torch.nn.functional.pad(target, (0, mixture_length - target.numel()))
    return mixture, target


def compute_peak_scale(
    target: torch.Tensor,
    residual: torch.Tensor,
    max_peak: float = 0.99,
) -> float:
    mixture = target + residual
    peak = max(
        target.abs().max().item(),
        residual.abs().max().item(),
        mixture.abs().max().item(),
    )
    if peak <= 0:
        return 1.0
    return min(1.0, max_peak / peak)


def _to_audio_segment(waveform: torch.Tensor, sample_rate: int) -> AudioSegment:
    waveform = waveform.detach().float().cpu().clamp(-1.0, 1.0)
    pcm16 = (waveform * 32767.0).round().short().numpy().tobytes()
    return AudioSegment(
        data=pcm16,
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )


def generate_activity_spans(
    waveform: torch.Tensor,
    sample_rate: int,
    silence_thresh_db: float = -40.0,
    min_silence_ms: int = 250,
    min_sounding_ms: int = 250,
    token: str = "+",
) -> list[list[str | float]]:
    segment = _to_audio_segment(waveform, sample_rate)
    regions = detect_nonsilent(
        segment,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_db,
    )
    spans = []
    for start_ms, end_ms in regions:
        if end_ms - start_ms < min_sounding_ms:
            continue
        spans.append(
            [
                token,
                round(start_ms / 1000.0, 4),
                round(end_ms / 1000.0, 4),
            ]
        )
    return spans


class PromptedSpeechFineTuneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: Optional[str] = None,
        task: Optional[str] = None,
        sample_rate: Optional[int] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.records = load_manifest_records(self.manifest_path)
        if split is not None:
            self.records = [x for x in self.records if x.get("split") == split]
        if task is not None:
            self.records = [x for x in self.records if x.get("task") == task]
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        if record.get("source_format") != "stereo_stems":
            raise ValueError(
                f"Unsupported source_format={record.get('source_format')} in record {record.get('id')}"
            )

        sample_rate = self.sample_rate or int(record["sample_rate"])
        source_path = resolve_manifest_path(self.manifest_path, record["source_path"])
        wav = load_stereo_stems(source_path, sample_rate=sample_rate)

        target = wav[int(record["target_channel"])]
        residual = wav[int(record["residual_channel"])]
        scale = float(record.get("scale", 1.0))
        if scale != 1.0:
            target = target * scale
            residual = residual * scale
        mixture = target + residual

        anchors = [
            (anchor[0], float(anchor[1]), float(anchor[2]))
            for anchor in record.get("anchors", [])
        ]
        return {
            "id": record["id"],
            "task": record["task"],
            "split": record.get("split"),
            "source_path": str(source_path),
            "sample_rate": sample_rate,
            "description": record["description"],
            "anchors": anchors,
            "mixture": mixture,
            "target": target,
            "residual": residual,
            "metadata": record.get("metadata", {}),
        }


def collate_prompted_separation_samples(
    samples: list[dict[str, Any]],
    processor,
) -> dict[str, Any]:
    descriptions = [sample["description"] for sample in samples]
    mixtures = [sample["mixture"].unsqueeze(0) for sample in samples]
    has_anchors = any(sample["anchors"] for sample in samples)
    anchors = [sample["anchors"] for sample in samples] if has_anchors else None
    batch = processor(descriptions=descriptions, audios=mixtures, anchors=anchors)

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
