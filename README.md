# vibevoice-no

Dette repoet samler biblioteket (`vendor/vibevoice/vibevoice`) og norsk trenings-/inference-verktøyene (`vibevoice_no`). Oppsettet gir én enhetlig installasjon og tydelig separasjon av ansvar:

- `vendor/vibevoice/`: selve TTS-biblioteket og demoer
- `vibevoice_no/`: treningsskript, LoRA, datasettverktøy og enkel inference

## Installasjon (lokalt og på VAST)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
Dette installerer `VibeVoice-main` i editable mode og alle avhengigheter for `vibevoice_no`.

## Kjøring

### Trening (eksempel)
```bash
source .venv/bin/activate
cd vibevoice_no
python train_norwegian_vibevoice.py \
  --config configs/training.example.yaml
```

### Inference (eksempel)
```bash
source .venv/bin/activate
cd vibevoice_no
python inference_generate.py \
  --model_dir ./vibevoice_no_7b \
  --text "Hei, verden!" \
  --seconds 3.0 \
  --out out.wav
```

## Modeller og data
- Ikke legg modellvekter i Git. Bruk Hugging Face, S3 eller et VAST volume og pek skriptene til stien, f.eks. `/workspace/models/VibeVoice7B`.
- `.env*`-filer ignoreres i Git (eksempelfiler kan versjoneres).

## RunPod 4090 hurtigstart (anbefalt for inferens/LoRA)

```bash
# Koble til RunPod (eksempel):
ssh -p <PORT> -i ~/.ssh/<RUNPOD_KEY> root@<IP>

cd /workspace
git clone <repo-url> vibevoice-no
cd vibevoice-no

# Legg cache/data på volumet (unngå 20 GB containerlag)
export XDG_CACHE_HOME=/workspace/.cache
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export TORCH_HOME=/workspace/.cache/torch
export PIP_CACHE_DIR=/workspace/.cache/pip

# Oppsett for 4090 (CUDA 12.4) + prosjektavhengigheter
bash scripts/prepare_4090.sh
source .venv/bin/activate
pip install 'peft>=0.17.1,<0.18'
```

### LoRA-inferens med HF-basemodell + lokal adapter

Kopier LoRA-adapteren til podden (fra lokal maskin):

```bash
scp -P <PORT> -i ~/.ssh/<RUNPOD_KEY> -r \
  /path/to/checkpoint-XXXX \
  root@<IP>:/workspace/vibevoice-no/checkpoint-XXXX
```

Kjør generering (basemodell lastes fra Hugging Face):

```bash
PYTHONPATH=vendor python vibevoice_no/inference_generate.py \
  --model_dir Jmica/VibeVoice7B \
  --adapter_dir checkpoint-XXXX \
  --text "Hei, dette er en test." \
  --seconds 3.0 \
  --out out.wav \
  --device auto --dtype auto
```

Tips:
- 80 GB volum holder typisk for inferens + noen checkpoints (basemodell ~17 GB, adapter ~0.5 GB, miljø ~4–8 GB).
- Sett `HF_HOME`/`TRANSFORMERS_CACHE` før første kjøring for å unngå cache i containerlaget.

## VAST hurtigstart
```bash
ssh vast-direct
cd /workspace
# klon repoet hvis det ikke allerede finnes
# git clone <repo-url> VibeVoice
cd VibeVoice
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
Kjør deretter trening eller inference som over.
