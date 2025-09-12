#!/usr/bin/env bash
set -euo pipefail

TEXT=${1:-"Hei, dette er en test."}

python vibevoice_no/inference_generate.py --model_dir vibevoice_no/vibevoice_no_7b --text "$TEXT" --seconds 3.0 --out out.wav
python vibevoice_no/privacy/voice_anonymity_check.py --train_jsonl vibevoice_no/data/train_24k.jsonl --gen out.wav --threshold 0.8 || echo "⚠️ Similarity over threshold"


