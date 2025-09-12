#!/usr/bin/env python3
import argparse
import math
from typing import Optional  # noqa: F401

import numpy as np
import soundfile as sf
import torch

from vibevoice.modular.modeling_vibevoice_inference import (  # type: ignore[import]
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import (  # type: ignore[import]
    VibeVoiceProcessor,
)


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
    # Memory/placement controls
    p.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Model weights dtype (defaults to fp16 on CUDA, fp32 on CPU)",
    )
    p.add_argument(
        "--device_map",
        choices=["auto", "none"],
        default="auto",
        help=(
            "Use Accelerate device_map auto when on CUDA to reduce peak VRAM"
        ),
    )
    args = p.parse_args()

    is_cuda = torch.cuda.is_available()

    # Choose dtype
    if args.dtype == "auto":
        torch_dtype = torch.float16 if is_cuda else torch.float32
    elif args.dtype == "bf16":
        torch_dtype = getattr(torch, "bfloat16", torch.float16)
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Choose device_map
    use_device_map_auto = is_cuda and (args.device_map == "auto")

    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype,
        device_map=("auto" if use_device_map_auto else None),
        low_cpu_mem_usage=True,
    )
    processor = VibeVoiceProcessor.from_pretrained(args.model_dir)
    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [VISION_START, VISION_PAD, VISION_END]}
    )
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass
    if not use_device_map_auto:
        device = torch.device("cuda" if is_cuda else "cpu")
        model.to(device)
    else:
        # When device_map=auto, inputs should be moved by HF internals
        device = torch.device("cuda" if is_cuda else "cpu")
    model.eval()

    prompt = build_prompt(args.text, args.seconds)
    inputs = tokenizer([prompt], return_tensors="pt")
    # With device_map=auto, HF will route tensors appropriately; otherwise move
    if not use_device_map_auto:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        gen_out = model.generate(
            inputs=inputs["input_ids"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            tokenizer=tokenizer,
            max_new_tokens=1,
            show_progress_bar=False,
            return_speech=True,
        )
        audio = None
        if hasattr(gen_out, "speech_outputs"):
            speech_outputs = getattr(gen_out, "speech_outputs")
            if isinstance(speech_outputs, list) and speech_outputs:
                audio = speech_outputs[0]
        if audio is None:
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


