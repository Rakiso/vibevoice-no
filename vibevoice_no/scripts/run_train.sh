#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip
pip install -r requirements.txt

# Validate
python vibevoice_no/dataset_tools/validate_dataset.py --jsonl vibevoice_no/data/train.jsonl --sample 200 || true

# Preprocess to 24k
python vibevoice_no/dataset_tools/preprocess_resample_trim.py \
  --in_jsonl  vibevoice_no/data/train.rel.jsonl \
  --out_jsonl vibevoice_no/data/train_24k.jsonl \
  --out_root  vibevoice_no/data_processed --trim_db 30 --normalize

python vibevoice_no/dataset_tools/preprocess_resample_trim.py \
  --in_jsonl  vibevoice_no/data/valid.rel.jsonl \
  --out_jsonl vibevoice_no/data/valid_24k.jsonl \
  --out_root  vibevoice_no/data_processed --trim_db 30 --normalize

# Train (full) by default
python vibevoice_no/train_norwegian_vibevoice.py \
  --config vibevoice_no/configs/training.example.yaml \
  --use_balanced_sampler --use_eval_balanced_sampler


