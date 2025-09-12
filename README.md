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
