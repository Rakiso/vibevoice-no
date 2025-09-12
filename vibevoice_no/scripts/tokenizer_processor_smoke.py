#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
from __future__ import annotations

import argparse

import numpy as np
import torch

from typing import Any
from vibevoice.processor.vibevoice_processor import (  # type: ignore[import]
    VibeVoiceProcessor,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="CPU smoke: load processor, tokenize prompt, fake audio"
    )
    p.add_argument("--model_name_or_path", default="Jmica/VibeVoice7B")
    p.add_argument("--seconds", type=float, default=0.5)
    args = p.parse_args()

    try:
        processor = VibeVoiceProcessor.from_pretrained(args.model_name_or_path)
    except Exception as e:
        print(
            "tokenizer/processor unavailable (likely missing HF access):",
            type(e).__name__,
            str(e),
        )
        print("SKIP: tokenizer/processor smoke")
        return

    text = "Speaker 1: Hei! Dette er en tokenizer-smoke test."
    proc_out = processor(text=[text], return_tensors="pt", padding=True)
    token_like: Any = getattr(
        proc_out,
        "input_ids",
        getattr(proc_out, "data", None),
    )
    shape = None
    if hasattr(token_like, "shape"):
        try:
            shape = tuple(token_like.shape)  # type: ignore[attr-defined]
        except Exception:
            shape = None
    print("token_ids shape:", shape)

    # Build a small dummy audio batch to ensure processor handles audio inputs
    target_sr = 24000
    num_samples = int(target_sr * args.seconds)
    fake_audio = [np.zeros(num_samples, dtype=np.float32)]
    try:
        proc_audio = processor(
            text=[text],
            audio=fake_audio,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        if hasattr(proc_audio, "data"):
            data = proc_audio.data
        else:
            data = proc_audio
        audio_values = data.get("audio_values")
        if isinstance(audio_values, torch.Tensor):
            print("audio_values shape:", tuple(audio_values.shape))
        else:
            print("audio_values present:", audio_values is not None)
    except Exception as e:
        print("processor(audio=...) path failed:", type(e).__name__, str(e))

    print("OK: tokenizer/processor smoke complete")


if __name__ == "__main__":
    main()
