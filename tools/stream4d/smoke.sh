#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
py="${F4_PYTHON:-.venv/bin/python}"
"$py" -m unittest discover -s tests/stream4d -v
"$py" -m stream4d.cli synthetic --config configs/stream4d/tiny.yaml --output data/smoke --frames 12
"$py" -m stream4d.cli train --config configs/stream4d/tiny.yaml --manifest data/smoke/frames.jsonl --output outputs/smoke_perception --steps 12 --allow-synthetic
"$py" -m stream4d.cli train --checkpoint outputs/smoke_perception/adapter.pt --manifest data/smoke/frames.jsonl --output outputs/smoke_joint --stage joint --steps 4 --allow-synthetic
"$py" -m stream4d.cli infer --checkpoint outputs/smoke_joint/adapter.pt --manifest data/smoke/frames.jsonl --output outputs/smoke_stream --allow-synthetic --render
