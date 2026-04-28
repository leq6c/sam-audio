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
You can enable `rsLoRA` with `--use-rslora`, which switches the adapter scaling
from `alpha / r` to `alpha / sqrt(r)`.

Validation can now be configured independently from training:

- `--eval-batch-size`: validation batch size
- `--eval-num-workers`: validation dataloader workers
- `--eval-cache {none,cpu,gpu}`: cache frozen validation features once and reuse them

For large validation sets, this is the recommended pattern:

```bash
sam-audio-train-lora \
  --train-json /path/to/train.json \
  --eval-json /path/to/valid.json \
  --checkpoint-path facebook/sam-audio-small \
  --output-dir /path/to/output \
  --batch-size 1 \
  --eval-batch-size 4 \
  --num-workers 4 \
  --eval-num-workers 8 \
  --eval-cache cpu \
  --epochs 3 \
  --learning-rate 1e-4
```

You can narrow this with:

```bash
sam-audio-train-lora \
  --train-json /path/to/train.json \
  --checkpoint-path facebook/sam-audio-small \
  --output-dir /path/to/output \
  --lora-prefixes transformer \
  --lora-targets wq,wk,wv,wo
```

To train with rank-stabilized LoRA:

```bash
sam-audio-train-lora \
  --train-json /path/to/train.json \
  --checkpoint-path facebook/sam-audio-small \
  --output-dir /path/to/output \
  --lora-r 64 \
  --lora-alpha 32 \
  --use-rslora
```

The trainer saves:

- `train_args.json`
- `train_metrics.jsonl`
- `eval_metrics.jsonl` (when `--eval-json` is provided)
- `summary.json`
- `best/adapter_config.json` and `best/adapter_model.pt` when eval is enabled
- `final/adapter_config.json`
- `final/adapter_model.pt`

`adapter_model.pt` contains only LoRA weights.
`train_metrics.jsonl` stores one JSON record per optimizer step, which is useful for
plotting loss curves after training.

## Comparison Export

To compare the base checkpoint and a trained LoRA adapter, run:

```bash
sam-audio-compare-lora \
  --json /path/to/eval.jsonl \
  --checkpoint-path /path/to/checkpoint \
  --adapter-dir /path/to/output/final \
  --output-dir /path/to/comparison
```

This command scans the full dataset to score every sample, then exports two
comparison directories:

- `first/`
- `best/`

Each directory contains:

- `mixture.wav`
- `groundtruth_target.wav`
- `groundtruth_residual.wav`
- `base_target.wav`
- `base_residual.wav`
- `lora_target.wav`
- `lora_residual.wav`
- `report.json`

`index.json` stores the full metric table plus the selected `first` and `best`
sample reports.

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
