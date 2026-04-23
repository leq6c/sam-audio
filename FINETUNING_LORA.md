# LoRA Fine-Tuning

This repository now includes a minimal LoRA fine-tuning path for SAM-Audio.

## Dataset Format

Training data is read from `.json` or `.jsonl`.

Each record must contain:

- `audio-url`: mixture audio path
- `span`: optional span prompt
- `prompt`: text prompt
- `groundtruth-audio-url`: target source audio path

Example:

```json
[
  {
    "audio-url": "mixtures/sample_001.wav",
    "span": [[0.2, 0.8], [1.1, 1.5]],
    "prompt": "speech, single speaker",
    "groundtruth-audio-url": "targets/sample_001.wav"
  }
]
```

Notes:

- Relative paths are resolved relative to the JSON file.
- `file://` URLs are also supported.
- `span` may be omitted, `null`, `[[start, end], ...]`, or `[[token, start, end], ...]`.
- The residual target is reconstructed on the fly as `mixture - target`.

## Training

After installing the package and its dependencies, run:

```bash
sam-audio-train-lora \
  --train-json /path/to/train.json \
  --eval-json /path/to/valid.json \
  --checkpoint-path facebook/sam-audio-small \
  --output-dir /path/to/output \
  --batch-size 1 \
  --epochs 1 \
  --learning-rate 1e-4
```

By default, LoRA is applied to all `nn.Linear` layers under `transformer`.

You can narrow this with:

```bash
sam-audio-train-lora \
  --train-json /path/to/train.json \
  --checkpoint-path facebook/sam-audio-small \
  --output-dir /path/to/output \
  --lora-prefixes transformer \
  --lora-targets wq,wk,wv,wo
```

The trainer saves:

- `train_args.json`
- `summary.json`
- `final/adapter_config.json`
- `final/adapter_model.pt`

`adapter_model.pt` contains only LoRA weights.

## Included Sample Dataset

The workspace includes a minimal smoke-test dataset in
[../sample_dataset](../sample_dataset/README.md).

Use it with:

```bash
sam-audio-train-lora \
  --train-json ../sample_dataset/train.json \
  --eval-json ../sample_dataset/eval.json \
  --checkpoint-path facebook/sam-audio-small \
  --output-dir /tmp/sam-audio-sample-lora \
  --batch-size 1 \
  --epochs 1 \
  --max-steps 1 \
  --log-every 1
```
