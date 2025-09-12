#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)
import librosa
import yaml

from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration


def _bf16_if_cuda() -> Optional[torch.dtype]:
    if torch.cuda.is_available():
        return torch.bfloat16
    return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


class JsonlTtsDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.items[idx]
        return rec


class TtsDataCollator:
    def __init__(
        self,
        processor: Any,
        augment_pitch_semitones: float = 0.0,
        augment_tempo_pct: float = 0.0,
        target_sr: int = 24000,
    ) -> None:
        self.processor = processor
        self.augment_pitch_semitones = float(augment_pitch_semitones)
        self.augment_tempo_pct = float(augment_tempo_pct)
        self.target_sr = int(target_sr)

    def _load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        y, sr = librosa.load(path, sr=None, mono=False)
        if y.ndim == 2:
            y = np.mean(y, axis=0)
        if sr != self.target_sr:
            try:
                y = librosa.resample(y=y, orig_sr=sr, target_sr=self.target_sr, res_type="soxr_hq")
            except Exception:
                y = librosa.resample(y=y, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        return y, sr

    def _augment(self, y: np.ndarray, sr: int) -> np.ndarray:
        if abs(self.augment_pitch_semitones) > 1e-6:
            try:
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=self.augment_pitch_semitones)
            except Exception:
                pass
        if abs(self.augment_tempo_pct) > 1e-6:
            try:
                rate = 1.0 + (self.augment_tempo_pct / 100.0)
                rate = max(0.5, min(2.0, rate))
                y = librosa.effects.time_stretch(y, rate=rate)
            except Exception:
                pass
        return y

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts: List[str] = []
        audios: List[np.ndarray] = []
        for rec in batch:
            text = str(rec.get("text", ""))
            audio_path = str(rec.get("audio"))
            y, sr = self._load_audio(audio_path)
            y = self._augment(y, sr)
            texts.append(ensure_script_format(text))
            audios.append(y.astype(np.float32))

        # Try common processor signatures
        try:
            proc_out = self.processor(
                text=texts,
                audio=audios,
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding=True,
            )
            return proc_out.data if hasattr(proc_out, "data") else proc_out
        except Exception:
            pass

        # Fallback: text-only tokenization; audio passed separately for custom models
        proc_out = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
        )
        proc_out["audio_values"] = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(a) for a in audios], batch_first=True
        )
        proc_out["sampling_rate"] = torch.tensor([self.target_sr] * len(audios))
        return proc_out


def ensure_script_format(text: str, default_idx: int = 1) -> str:
    """Ensure lines follow "Speaker N:" format required by the VibeVoice parser.
    - Convert bracket style "[N]:" to "Speaker N:"
    - If no speaker markers present, wrap each line as "Speaker 1: ..."
    """
    import re
    if not text:
        return f"Speaker {default_idx}:"
    # Normalize and split to non-empty lines
    lines = [l.strip() for l in text.replace("\\n", "\n").splitlines() if l.strip()]
    if not lines:
        return f"Speaker {default_idx}:"
    # Convert any bracket markers to Speaker markers
    converted = [re.sub(r"(?m)^\s*\[(\d+)\]\s*:\s*", r"Speaker \1: ", l) for l in lines]
    # If any line already has Speaker N:, accept as-is
    if any(re.match(r"^\s*Speaker\s+\d+\s*:\s+", l) for l in converted):
        return "\n".join(converted)
    # Otherwise, wrap every line as Speaker 1:
    return "\n".join(f"Speaker {default_idx}: {l}" for l in converted)


def _build_balanced_sampler(items: List[Dict[str, Any]]) -> Optional[WeightedRandomSampler]:
    speakers: List[Optional[str]] = [
        str(x.get("speaker")) if x.get("speaker") is not None else None for x in items
    ]
    if all(s is None for s in speakers):
        return None
    # Compute inverse-frequency weights per speaker (None gets its own bucket)
    from collections import Counter

    counts = Counter(speakers)
    weights = [1.0 / counts[s] for s in speakers]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune VibeVoice7B for Norwegian TTS")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--use_balanced_sampler", action="store_true")
    parser.add_argument("--use_eval_balanced_sampler", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = 42
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_name = cfg.get("model_name_or_path", "Jmica/VibeVoice7B")
    bf16_flag = bool(cfg.get("bf16", True)) and (torch.cuda.is_available())
    torch_dtype = torch.bfloat16 if bf16_flag else None

    processor = VibeVoiceProcessor.from_pretrained(model_name)
    model = VibeVoiceForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    train_items = _read_jsonl(Path(cfg["train_file"]))
    eval_items = _read_jsonl(Path(cfg["valid_file"]))

    train_ds = JsonlTtsDataset(train_items)
    eval_ds = JsonlTtsDataset(eval_items)

    collator = TtsDataCollator(
        processor=processor,
        augment_pitch_semitones=float(cfg.get("augment_pitch_semitones", 0.0)),
        augment_tempo_pct=float(cfg.get("augment_tempo_pct", 0.0)),
        target_sr=24000,
    )

    train_sampler = None
    eval_sampler = None
    if args.use_balanced_sampler:
        train_sampler = _build_balanced_sampler(train_items)
    if args.use_eval_balanced_sampler:
        eval_sampler = _build_balanced_sampler(eval_items)

    training_args = TrainingArguments(
        output_dir=str(cfg.get("output_dir", "./vibevoice_no_7b")),
        num_train_epochs=int(cfg.get("num_epochs", 3)),
        per_device_train_batch_size=int(cfg.get("per_device_batch_size", 1)),
        per_device_eval_batch_size=int(cfg.get("per_device_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 32)),
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.05)),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        bf16=bool(cfg.get("bf16", True)) and torch.cuda.is_available(),
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    if train_sampler is not None:
        trainer._get_train_sampler = lambda: train_sampler  # type: ignore
    if eval_sampler is not None:
        trainer._get_eval_sampler = lambda x: eval_sampler  # type: ignore

    trainer.train()
    trainer.save_model(training_args.output_dir)
    try:
        processor.save_pretrained(training_args.output_dir)
    except Exception:
        pass


if __name__ == "__main__":
    main()


