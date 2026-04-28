# Speech Fine-Tuning Data Prep

This repository does not ship an official SAM-Audio training script, but the files in
`sam_audio/finetune` make it possible to convert a 2-speaker, channel-separated corpus
into a manifest that is close to the paper's speech setup.

## Design

For a stereo file where channel 0 and channel 1 are already separated speakers:

- `mixture = ch0 + ch1`
- `target = ch{i}`
- `residual = ch{1-i}`
- one training record is emitted for each target speaker

This matches the paper's `speaker extraction` setup more closely than treating the
mixture as a generic speech/noise task.

When speaker attributes are available, use them as the text prompt:

- `female -> woman speaking`
- `male -> man speaking`
- `child -> child speaking`

When no attributes are available, the fallback prompt is `single speaker`. Since the
same text can map to both speakers, the manifest also carries span prompts derived from
VAD on the target channel. This makes the conditioning much closer to the original
`text + span` training regime than pure text-only fine-tuning on ambiguous prompts.

## Manifest Schema

Each JSONL record contains:

- `source_path`: path to the original stereo stem file
- `target_channel` / `residual_channel`: which speaker to extract
- `description`: text prompt
- `anchors`: `+` span prompts derived from the target speech activity
- `scale`: deterministic peak normalization factor applied on load
- `task`: currently `speaker_extraction`

`PromptedSpeechFineTuneDataset` reconstructs `mixture`, `target`, and `residual` on the
fly from the stereo file, so data does not need to be duplicated on disk.

## CLI

After `pip install .`, build a manifest with:

```bash
sam-audio-prepare-speech-manifest \
  --input-dir /path/to/stereo-speaker-wavs \
  --output-jsonl /path/to/manifests/speaker_text_span.jsonl \
  --default-description "single speaker" \
  --val-ratio 0.02
```

Optional metadata CSV:

```text
path,gender_0,gender_1,prompt_0,prompt_1,split
session001.wav,female,male,,,
session002.wav,male,male,host speaking,guest speaking,train
```

If you want explicit anti-target spans from the non-target speaker as well:

```bash
sam-audio-prepare-speech-manifest \
  --input-dir /path/to/stereo-speaker-wavs \
  --output-jsonl /path/to/manifests/speaker_text_span_neg.jsonl \
  --add-negative-spans
```

## Python Usage

```python
from sam_audio import SAMAudioProcessor
from sam_audio.finetune import (
    PromptedSpeechFineTuneDataset,
    collate_prompted_separation_samples,
)

processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
dataset = PromptedSpeechFineTuneDataset("speaker_text_span.jsonl", split="train")

samples = [dataset[0], dataset[1]]
batch = collate_prompted_separation_samples(samples, processor)
```

`batch["input_batch"]` is the processor batch for SAM-Audio, while `target_audio` and
`residual_audio` provide the supervision tensors for a future training loop.
