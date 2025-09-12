#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/prepare_4090.sh [/absolute/path/to/wavs]
#
# Steps:
# - Installs ffmpeg/git (Debian/Ubuntu)
# - Creates .venv
# - Installs CUDA 12.4 wheels for PyTorch 2.8 (matches 4090, CUDA 12.x)
# - Installs project requirements
# - Optionally symlinks your dataset into vibevoice_no/data_raw/wavs
# - Runs a quick CUDA and inference smoke test

DATA_WAVS_PATH="${1:-}"

echo "[setup] Updating system packages (if apt available)"
if command -v apt >/dev/null 2>&1; then
  apt update -y && apt install -y ffmpeg git
else
  echo "[setup] 'apt' not found; skipping system package install"
fi

echo "[setup] Creating virtual environment"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

echo "[setup] Installing PyTorch 2.6.0 CUDA 12.4 wheels"
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0+cu124 torchaudio==2.6.0+cu124

echo "[setup] Installing project requirements"
pip install -r requirements.txt

if [[ -n "${DATA_WAVS_PATH}" ]]; then
  if [[ -d "${DATA_WAVS_PATH}" ]]; then
    echo "[dataset] Linking dataset -> vibevoice_no/data_raw/wavs"
    mkdir -p vibevoice_no/data_raw
    ln -sfn "${DATA_WAVS_PATH}" vibevoice_no/data_raw/wavs
  else
    echo "[dataset] WARN: Provided path does not exist: ${DATA_WAVS_PATH}" >&2
  fi
else
  echo "[dataset] No dataset path provided. You can link later with:"
  echo "         ln -s /abs/path/to/wavs vibevoice_no/data_raw/wavs"
fi

echo "[smoke] Checking CUDA availability"
python - <<'PY'
import torch
print('cuda_available=', torch.cuda.is_available())
print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

echo "[smoke] Running minimal inference init (0.5s)"
set +e
python vibevoice_no/inference_generate.py \
  --model_dir vibevoice_no/vibevoice_no_7b \
  --text "Hei" \
  --seconds 0.5 \
  --out smoke.wav
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  echo "[smoke] Inference init failed (likely missing model dir). This is expected if weights not present yet." >&2
else
  echo "[smoke] Inference succeeded: smoke.wav"
fi

echo "[done] 4090 setup complete. Suggested next steps:"
echo " - Preprocess to 24k: make preprocess (requires train.jsonl/valid.jsonl)"
echo " - Start LoRA training: make train-smoke (Ctrl-C after first logs)"


