#!/usr/bin/env bash
set -euo pipefail

# Local CPU-only smoke test on macOS/Linux without CUDA
# - Creates venv
# - Installs editable vendor + vibevoice_no deps
# - Validates dataset JSONL and first N wavs
# - Preprocesses a tiny subset to 24k
# - Runs privacy check against the generated wav

JSONL_TRAIN=${1:-vibevoice_no/data/train.jsonl}
JSONL_VALID=${2:-vibevoice_no/data/valid.jsonl}
SUBSET_N=${SUBSET_N:-50}

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

echo "[validate] ${JSONL_TRAIN}"
python vibevoice_no/dataset_tools/validate_dataset.py --jsonl "${JSONL_TRAIN}" --sample ${SUBSET_N} || true

echo "[subset]"
python vibevoice_no/dataset_tools/jsonl_subset.py \
  --in_jsonl  "${JSONL_TRAIN}" \
  --out_jsonl vibevoice_no/data/train.head.jsonl \
  --n ${SUBSET_N}

echo "[preprocess]"
python vibevoice_no/dataset_tools/preprocess_resample_trim.py \
  --in_jsonl  vibevoice_no/data/train.head.jsonl \
  --out_jsonl vibevoice_no/data/train.head_24k.jsonl \
  --out_root  vibevoice_no/data_processed --trim_db 30 --normalize

echo "[privacy]"
# Generate a dummy 1s tone to exercise the pipeline without the big model
python - <<'PY'
import numpy as np, soundfile as sf
sr=24000
t=np.linspace(0,1,sr,endpoint=False)
sf.write('out_dummy.wav', 0.1*np.sin(2*np.pi*440*t).astype(np.float32), sr)
print('Wrote out_dummy.wav')
PY

python vibevoice_no/privacy/voice_anonymity_check.py \
  --train_jsonl vibevoice_no/data/train.head_24k.jsonl \
  --gen out_dummy.wav --threshold 0.8 || echo "privacy check signaled high similarity (expected low)"

echo "[tokenizer/processor]"
python vibevoice_no/scripts/tokenizer_processor_smoke.py --model_name_or_path vibevoice_no/vibevoice_no_7b --seconds 0.1

echo "[done] Local smoke complete"


