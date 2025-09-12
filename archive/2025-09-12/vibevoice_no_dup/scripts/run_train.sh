#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt
# Optional: pip install -e .  # if local VibeVoice repo present

python dataset_tools/validate_dataset.py --jsonl data/train.jsonl --sample 100 || true
python dataset_tools/preprocess_resample_trim.py --in_jsonl data/train.jsonl --out_jsonl data/train_24k.jsonl --out_root data_processed --trim_db 30 --normalize
python dataset_tools/preprocess_resample_trim.py --in_jsonl data/valid.jsonl --out_jsonl data/valid_24k.jsonl --out_root data_processed --trim_db 30 --normalize

python train_norwegian_vibevoice.py --config configs/training.example.yaml --use_balanced_sampler --use_eval_balanced_sampler


