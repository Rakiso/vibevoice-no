#!/usr/bin/env python3
import argparse
import math
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


VISION_START = "<|vision_start|>"
VISION_PAD = "<|vision_pad|>"
VISION_END = "<|vision_end|>"
TARGET_SR = 24000
COMPRESS_RATIO = 3200


def build_prompt(text: str, seconds: float) -> str:
    approx_samples = int(seconds * TARGET_SR)
    N = int(math.ceil(approx_samples / float(COMPRESS_RATIO)))
    control = "[voice: neutral, non-identifiable, Norwegian bokmål]"
    prompt = (
        "Text input:\n"
        f"Speaker: {text} {control}\n"
        "Speech output:\n"
        f"{VISION_START} "
        + (" ".join([VISION_PAD] * N) + " " if N > 0 else "")
        + f"{VISION_END}\n"
    )
    return prompt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--out", type=str, default="out.wav")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VibeVoiceForConditionalGeneration.from_pretrained(args.model_dir)
    processor = VibeVoiceProcessor.from_pretrained(args.model_dir)
    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": [VISION_START, VISION_PAD, VISION_END]})
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass
    model.to(device)
    model.eval()

    prompt = build_prompt(args.text, args.seconds)
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1)
        if hasattr(model, "decode_last_speech"):
            audio = model.decode_last_speech()
        elif hasattr(model, "get_generated_audio"):
            audio = model.get_generated_audio()
        else:
            audio = np.zeros(int(args.seconds * TARGET_SR), dtype=np.float32)

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()
    sf.write(args.out, audio, TARGET_SR)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()


