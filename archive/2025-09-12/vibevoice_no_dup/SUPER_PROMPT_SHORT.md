Tren en speaker-agnostisk norsk TTS med LoRA på én RTX 4090 (24 GB) fra basemodellen Jmica/VibeVoice7B; all trening i 24 kHz mono; sett N = ceil(samples_24k/3200) placeholders i prompt; total loss = LM + diffusion (maskér diffusion-tokens i labels); LoRA kun på LM-blokkene (diffusion head trenbar); inference kan bruke nøytrale M/F-presets (pitch/formant) som IKKE påvirker trening; inkluder enkel anonymitetssjekk (ECAPA) mot treningsspeakers.

```bash
# 1) Preflight & preprocess (24 kHz)
python dataset_tools/validate_dataset.py --jsonl data/train.jsonl --sample 50
python dataset_tools/preprocess_resample_trim.py --in_jsonl data/train.jsonl --out_jsonl data/train_24k.jsonl --out_root data_processed --trim_db 30 --normalize
python dataset_tools/preprocess_resample_trim.py --in_jsonl data/valid.jsonl --out_jsonl data/valid_24k.jsonl --out_root data_processed --trim_db 30 --normalize
```

```bash
# 2) LoRA training på RTX 4090 (24 GB)
python train_norwegian_vibevoice_lora.py \
  --model_name_or_path Jmica/VibeVoice7B \
  --train_file data/train_24k.jsonl \
  --valid_file data/valid_24k.jsonl \
  --output_dir ./vibevoice_no_7b_lora \
  --num_epochs 1 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 48 \
  --learning_rate 2e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --diffusion_weight 1.0 \
  --ddpm_batch_mul 2 \
  --max_length 4096 \
  --bf16
# OOM fallback: --max_length 3072/2048, --ddpm_batch_mul 1, øk --gradient_accumulation_steps
```


