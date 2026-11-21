#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
F4_PYTHON="${F4_PYTHON:-.venv/bin/python}"
PAI_RAW_ROOT="${PAI_RAW_ROOT:-/mnt/raid/dataset/PAI/mini}"
PAI_SMOKE_DATA="${PAI_SMOKE_DATA:-/mnt/raid/dataset/PAI/mini_stream4d_smoke}"
PAI_SMOKE_OUTPUT="${PAI_SMOKE_OUTPUT:-outputs/pai_smoke}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

"$F4_PYTHON" tools/data_process/prepare_pai.py \
  --raw-root "$PAI_RAW_ROOT" --output "$PAI_SMOKE_DATA" \
  --max-clips 2 --max-frames 8 --workers 2
"$F4_PYTHON" -m stream4d.cli train --config configs/stream4d/pai.yaml \
  --manifest "$PAI_SMOKE_DATA/train.jsonl" --stage perception --steps 16 \
  --output "$PAI_SMOKE_OUTPUT/perception"
"$F4_PYTHON" -m stream4d.cli train --checkpoint "$PAI_SMOKE_OUTPUT/perception/adapter.pt" \
  --manifest "$PAI_SMOKE_DATA/train.jsonl" --stage gaussian --steps 4 \
  --output "$PAI_SMOKE_OUTPUT/gaussian"
"$F4_PYTHON" -m stream4d.cli train --checkpoint "$PAI_SMOKE_OUTPUT/gaussian/adapter.pt" \
  --manifest "$PAI_SMOKE_DATA/train.jsonl" --stage joint --steps 4 \
  --output "$PAI_SMOKE_OUTPUT/joint"
"$F4_PYTHON" -m stream4d.cli infer --checkpoint "$PAI_SMOKE_OUTPUT/joint/adapter.pt" \
  --manifest "$PAI_SMOKE_DATA/test.jsonl" --output "$PAI_SMOKE_OUTPUT/test" \
  --render --render-size 96 160
"$F4_PYTHON" tools/stream4d/validate_pai.py \
  --train-manifest "$PAI_SMOKE_DATA/train.jsonl" --test-manifest "$PAI_SMOKE_DATA/test.jsonl" \
  --output "$PAI_SMOKE_OUTPUT/finetune_validation" --steps 5 --finetune-blocks 1
