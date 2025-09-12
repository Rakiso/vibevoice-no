## VibeVoice Norwegian Finetune (Jmica/VibeVoice7B on thor-l/nor_tts)

End-to-end scaffolding for preparing `thor-l/nor_tts`, preprocessing to 24 kHz, JSONL emission, full finetune or LoRA, inference, and a simple privacy check (ECAPA similarity).

### Quickstart

```bash
# 0) System prep
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
huggingface-cli login  # if dataset is private

# 1) Pull dataset & make JSONL
python vibevoice_no/dataset_tools/hf_to_jsonl.py \
  --repo_id thor-l/nor_tts --out_dir vibevoice_no/data/raw \
  --train_jsonl vibevoice_no/data/train.jsonl \
  --valid_jsonl vibevoice_no/data/valid.jsonl

# 2) Validate & preprocess → 24 kHz
python vibevoice_no/dataset_tools/validate_dataset.py --jsonl vibevoice_no/data/train.jsonl --sample 200 || true
python vibevoice_no/dataset_tools/preprocess_resample_trim.py \
  --in_jsonl  vibevoice_no/data/train.jsonl \
  --out_jsonl vibevoice_no/data/train_24k.jsonl \
  --out_root  vibevoice_no/data_processed --trim_db 30 --normalize

python vibevoice_no/dataset_tools/preprocess_resample_trim.py \
  --in_jsonl  vibevoice_no/data/valid.jsonl \
  --out_jsonl vibevoice_no/data/valid_24k.jsonl \
  --out_root  vibevoice_no/data_processed --trim_db 30 --normalize

# 3a) Train (full finetune)
python vibevoice_no/train_norwegian_vibevoice.py \
  --config vibevoice_no/configs/training.example.yaml \
  --use_balanced_sampler --use_eval_balanced_sampler

# 3b) Train (LoRA)
python vibevoice_no/train_norwegian_vibevoice_lora.py \
  --config vibevoice_no/configs/training.example.yaml \
  --use_balanced_sampler --use_eval_balanced_sampler

# 4) Inference + privacy
python vibevoice_no/inference_generate.py --model_dir vibevoice_no/vibevoice_no_7b --text "Hei, verden!" --seconds 3.5 --out out.wav
python vibevoice_no/privacy/voice_anonymity_check.py --train_jsonl vibevoice_no/data/train_24k.jsonl --candidate_wav out.wav --threshold 0.8 || echo "⚠️ Similarity over threshold"
```

### Notes

- Use LoRA for 1×24GB VRAM; prefer BF16 on supported GPUs.
- The pipeline assumes speaker-agnostic training; if `speaker` exists, balanced sampling can be enabled via flags.
- If `thor-l/nor_tts` is private, login or provide `--hf_token` to dataset tool.

### VRAM presets

Use these ready-made configs (adjust as needed):

```bash
# LoRA on 16 GB
python vibevoice_no/train_norwegian_vibevoice_lora.py \
  --config vibevoice_no/configs/training.lora.16gb.yaml \
  --use_balanced_sampler --use_eval_balanced_sampler

# LoRA on 24 GB (recommended)
python vibevoice_no/train_norwegian_vibevoice_lora.py \
  --config vibevoice_no/configs/training.lora.24gb.yaml \
  --use_balanced_sampler --use_eval_balanced_sampler

# LoRA on 40 GB
python vibevoice_no/train_norwegian_vibevoice_lora.py \
  --config vibevoice_no/configs/training.lora.40gb.yaml \
  --use_balanced_sampler --use_eval_balanced_sampler

# Full finetune on 80 GB
python vibevoice_no/train_norwegian_vibevoice.py \
  --config vibevoice_no/configs/training.full.80gb.yaml \
  --use_balanced_sampler --use_eval_balanced_sampler
```

