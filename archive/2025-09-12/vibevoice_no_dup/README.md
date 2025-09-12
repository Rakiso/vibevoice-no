# VibeVoice Norwegian TTS Fine-tuning (24 kHz, speaker-agnostic)

## Installasjon (monorepo)
Hvis du bruker dette som del av VibeVoice-monorepoet:
```bash
# fra repo-roten
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

This package fine-tunes `Jmica/VibeVoice7B` for Norwegian TTS with a neutral, non-identifiable voice. It trains without speaker conditioning, encourages speaker diversity per batch, and includes a privacy check to detect potential voice memorization.

Why 24 kHz? The VibeVoice acoustic preprocessor uses a compression ratio of 3200. At 24 kHz, this yields ~7.5 acoustic latents per second, which we reflect by inserting diffusion placeholders in the text sequence.

## Dataset schema
See `dataset_tools/schema.md` for the JSONL format and examples. Audio must be 24 kHz mono by training time (the provided preprocessor can resample and trim to 24 kHz mono).

## Quickstart

1) Validate and preprocess your data
```bash
python dataset_tools/validate_dataset.py --jsonl data/train.jsonl --sample 100
python dataset_tools/metadata_report.py --jsonl data/train.jsonl --probe_audio --tokenize --model_name_or_path Jmica/VibeVoice7B
# If starting from a Hugging Face dataset (e.g., thor-l/nor_tts):
python dataset_tools/convert_hf_to_jsonl.py \
  --hf_repo thor-l/nor_tts --split train --out_jsonl data/train.jsonl \
  --audio_column audio --text_column transcript --lang no --download_audio --out_audio_dir data_hf_audio
python dataset_tools/preprocess_resample_trim.py \
  --in_jsonl data/train.jsonl --out_jsonl data/train_24k.jsonl \
  --out_root data_processed --trim_db 30 --normalize
python dataset_tools/preprocess_resample_trim.py \
  --in_jsonl data/valid.jsonl --out_jsonl data/valid_24k.jsonl \
  --out_root data_processed --trim_db 30 --normalize
```

2) Train
```bash
python train_norwegian_vibevoice.py \
  --model_name_or_path Jmica/VibeVoice7B \
  --train_file data/train_24k.jsonl \
  --valid_file data/valid_24k.jsonl \
  --output_dir ./vibevoice_no_7b \
  --num_epochs 1 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --diffusion_weight 1.0 \
  --ddpm_batch_mul 4 \
  --bf16
```

3) Inference and privacy check
```bash
python inference_generate.py --model_dir ./vibevoice_no_7b --text "Hei, verden!" --seconds 3.0 --out out.wav
python privacy/voice_anonymity_check.py --train_jsonl data/train_24k.jsonl --gen out.wav --threshold 0.8 || echo "Warning: similarity over threshold"
```

## VRAM tips
- Use `--bf16` on supported GPUs.
- Increase `--gradient_accumulation_steps`.
- Use the LoRA trainer (`train_norwegian_vibevoice_lora.py`) for 1×24GB setups.

## Notes on speaker-agnostic training
- No speaker conditioning or reference voices are used.
- A balanced sampler promotes many distinct speakers per batch to reduce timbre memorization.


