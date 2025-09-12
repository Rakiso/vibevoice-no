#!/usr/bin/env bash
set -euo pipefail

python inference_generate.py --model_dir ./vibevoice_no_7b --text "$1" --seconds 3.0 --out out.wav
python privacy/voice_anonymity_check.py --train_jsonl data/train_24k.jsonl --gen out.wav --threshold 0.8 || echo "⚠️ Similarity over threshold"


