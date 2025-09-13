#!/usr/bin/env python3
import argparse
import math
from typing import Optional  # noqa: F401

import numpy as np
import soundfile as sf  # type: ignore[import]
import torch

from vibevoice.modular.modeling_vibevoice_inference import (  # type: ignore[import]
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import (  # type: ignore[import]
    VibeVoiceProcessor,
)
from peft import PeftModel  # type: ignore[import]


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


def _to_numpy_audio(x: object, seconds: float) -> np.ndarray:
    """Coerce various audio-like objects to 1D float32 numpy array."""
    target_len = max(1, int(seconds * TARGET_SR))
    arr: np.ndarray | None = None
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().float().numpy()
    elif isinstance(x, np.ndarray):
        arr = x.astype(np.float32, copy=False)
    elif isinstance(x, (list, tuple)):
        # Pick the first tensor/ndarray element if present
        for item in x:
            if isinstance(item, torch.Tensor):
                arr = item.detach().cpu().float().numpy()
                break
            if isinstance(item, np.ndarray):
                arr = item.astype(np.float32, copy=False)
                break
    if arr is None:
        return np.zeros(target_len, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim > 1:
        # Flatten multi-channel to mono
        arr = arr.reshape(-1)
    if arr.size == 0:
        arr = np.zeros(target_len, dtype=np.float32)
    # Remove NaN/Inf
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32, copy=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--out", type=str, default="out.wav")
    p.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help="Optional path to a PEFT/LoRA adapter checkpoint to apply",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device to run on. 'auto' prefers CUDA, then MPS, else CPU.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only load model/adapter and exit (useful to validate on low-memory Macs)",
    )
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

    # Resolve device preference
    requested = args.device
    is_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if requested == "auto":
        if is_cuda:
            device_str = "cuda"
        elif has_mps:
            device_str = "mps"
        else:
            device_str = "cpu"
    else:
        device_str = requested

    # Choose dtype (MPS requires float32 for stability)
    if args.dtype == "auto":
        if device_str == "cuda":
            torch_dtype = torch.float16
        elif device_str == "mps":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float32
    elif args.dtype == "bf16":
        torch_dtype = getattr(torch, "bfloat16", torch.float16)
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Choose device_map (only relevant on CUDA)
    use_device_map_auto = (device_str == "cuda") and (args.device_map == "auto")

    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype,
        device_map=("auto" if use_device_map_auto else None),
        low_cpu_mem_usage=True,
    )
    # Optionally apply LoRA adapter
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        # Try to merge LoRA for inference efficiency and to keep base APIs
        try:
            model = model.merge_and_unload()
        except Exception:
            pass
    processor = VibeVoiceProcessor.from_pretrained(args.model_dir)
    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens({
        "additional_special_tokens": [VISION_START, VISION_PAD, VISION_END]
    })
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass
    if not use_device_map_auto:
        device = torch.device(device_str)
        model.to(device)
    else:
        # When device_map=auto, inputs should be moved by HF internals
        device = torch.device("cuda")
    model.eval()

    # Dry-run mode: validate load and exit
    if args.dry_run:
        print(f"Loaded model from {args.model_dir} on device {device_str} with dtype {torch_dtype}")
        if args.adapter_dir:
            print(f"Applied LoRA adapter from {args.adapter_dir}")
        return

    prompt = build_prompt(args.text, args.seconds)
    inputs = tokenizer([prompt], return_tensors="pt")
    # With device_map=auto, HF will route tensors appropriately; otherwise move
    if not use_device_map_auto:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Build minimal placeholder speech inputs expected by vendor generate()
    batch_size = int(inputs["input_ids"].shape[0])
    seq_len = int(inputs["input_ids"].shape[1])
    # No speech frames to insert by default; masks are all False
    speech_input_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    # Match acoustic time steps (use 1 step for tiny dummy audio)
    speech_masks = torch.zeros((batch_size, 1), dtype=torch.bool)
    # Tiny dummy audio (silence) just to satisfy API; vendor will ignore if masks are False
    num_samples = max(1, int(TARGET_SR * min(args.seconds, 0.1)))
    speech_tensors = torch.zeros((batch_size, num_samples), dtype=torch.float32)

    with torch.no_grad():
        gen_out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            tokenizer=tokenizer,
            max_new_tokens=1,
            show_progress_bar=False,
            return_speech=True,
            speech_tensors=speech_tensors,
            speech_masks=speech_masks,
            speech_input_mask=speech_input_mask,
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

    audio_np = _to_numpy_audio(audio, args.seconds)
    sf.write(args.out, audio_np, TARGET_SR)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
