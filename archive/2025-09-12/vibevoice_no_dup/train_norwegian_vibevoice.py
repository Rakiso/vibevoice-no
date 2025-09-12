#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import soundfile as sf
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, TrainerCallback
import yaml

from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from dataset_tools.balanced_sampler import BalancedSpeakerSampler


VISION_START = "<|vision_start|>"
VISION_PAD = "<|vision_pad|>"
VISION_END = "<|vision_end|>"
TARGET_SR = 24000
COMPRESS_RATIO = 3200


def load_wave_mono_24k(path: str, start_sec: Optional[float], end_sec: Optional[float]) -> np.ndarray:
    if path.startswith("http://") or path.startswith("https://"):
        raise ValueError("Remote audio not supported at training time. Download first.")
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        # Mixdown
        data = np.mean(data, axis=1)
    if start_sec is not None or end_sec is not None:
        s = max(0.0, float(start_sec or 0.0))
        e = float(end_sec) if end_sec is not None else len(data) / float(sr)
        data = data[int(s * sr): int(e * sr)]
    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
    return data.astype(np.float32, copy=False)


class JsonlRows(Dataset):
    def __init__(self, jsonl_path: str):
        self.rows: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


@dataclass
class CollatorConfig:
    max_length: int
    augment_pitch_semitones: float = 0.0
    augment_tempo_pct: float = 0.0


class VibeVoiceCollator:
    def __init__(self, processor: VibeVoiceProcessor, model: VibeVoiceForConditionalGeneration, cfg: CollatorConfig):
        self.processor = processor
        self.model = model
        self.cfg = cfg
        self.tokenizer = processor.tokenizer
        # Ensure special tokens exist in tokenizer
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [VISION_START, VISION_PAD, VISION_END]
        })
        try:
            self.model.resize_token_embeddings(len(self.tokenizer))
        except Exception:
            pass
        self.token_id_start = self.tokenizer.convert_tokens_to_ids(VISION_START)
        self.token_id_pad = self.tokenizer.convert_tokens_to_ids(VISION_PAD)
        self.token_id_end = self.tokenizer.convert_tokens_to_ids(VISION_END)
        if any(tid in (None, self.tokenizer.unk_token_id) for tid in [self.token_id_start, self.token_id_pad, self.token_id_end]):
            raise ValueError("Special tokens for diffusion placeholders are missing from tokenizer.")

    def _maybe_augment(self, y: np.ndarray) -> np.ndarray:
        y_aug = y
        if abs(self.cfg.augment_pitch_semitones) > 1e-6:
            y_aug = librosa.effects.pitch_shift(y_aug, sr=TARGET_SR, n_steps=self.cfg.augment_pitch_semitones)
        if abs(self.cfg.augment_tempo_pct) > 1e-6:
            rate = 1.0 + (self.cfg.augment_tempo_pct / 100.0)
            # Guard against too-short arrays after stretch
            if len(y_aug) > 4:
                y_aug = librosa.effects.time_stretch(y_aug, rate)
        return y_aug.astype(np.float32, copy=False)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_texts: List[str] = []
        waveforms: List[np.ndarray] = []
        Ns: List[int] = []

        # Load audio, apply optional slicing and augmentation, compute N
        for ex in examples:
            y = load_wave_mono_24k(ex["audio"], ex.get("start_sec"), ex.get("end_sec"))
            if self.cfg.augment_pitch_semitones or self.cfg.augment_tempo_pct:
                y = self._maybe_augment(y)
            waveforms.append(y)
            num_samples = len(y)
            N = int(math.ceil(num_samples / float(COMPRESS_RATIO)))
            Ns.append(N)
            text = ex["text"].strip()
            prompt = (
                "Text input:\n"
                f"Speaker: {text}\n"
                "Speech output:\n"
                f"{VISION_START} "
                + (" ".join([VISION_PAD] * N) + " " if N > 0 else "")
                + f"{VISION_END}\n"
            )
            batch_texts.append(prompt)

        # Tokenize batch
        enc = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Masks for diffusion placeholders
        acoustic_masks = []
        for ids in input_ids:
            ids_list = ids.tolist()
            mask = [
                1 if tok in (self.token_id_start, self.token_id_pad, self.token_id_end) else 0
                for tok in ids_list
            ]
            acoustic_masks.append(mask)
        acoustic_input_mask = torch.tensor(acoustic_masks, dtype=torch.bool)
        acoustic_loss_mask = acoustic_input_mask.clone()

        # Labels with diffusion positions masked out
        labels = input_ids.clone()
        labels[acoustic_input_mask] = -100

        # Prepare speech features via processor
        speech_inputs = self.processor.prepare_speech_inputs(
            speech_waveforms=waveforms,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )
        speech_tensors = speech_inputs["speech_tensors"]  # [B, T, ...] per processor spec
        speech_masks = speech_inputs.get("speech_masks")

        # Semantic tokens (no grad)
        semantic_list: List[torch.Tensor] = []
        with torch.no_grad():
            for y in waveforms:
                sem = self.model.semantic_tokenizer(y, sampling_rate=TARGET_SR)
                if isinstance(sem, np.ndarray):
                    sem = torch.from_numpy(sem)
                if sem.dim() == 1:
                    sem = sem.unsqueeze(0)
                semantic_list.append(sem.squeeze(0).to(torch.long))
        # Pad semantic tokens to longest
        max_sem = max(s.size(0) for s in semantic_list)
        padded_sem = torch.full((len(semantic_list), max_sem), fill_value=-100, dtype=torch.long)
        for i, s in enumerate(semantic_list):
            L = s.size(0)
            padded_sem[i, :L] = s

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "acoustic_input_mask": acoustic_input_mask,
            "acoustic_loss_mask": acoustic_loss_mask,
            "speech_tensors": speech_tensors,
            "speech_masks": speech_masks,
            "speech_semantic_tensors": padded_sem,
        }
        return batch


class VibeVoiceTrainer(Trainer):
    def __init__(self, diffusion_weight: float, ddpm_batch_mul: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion_weight = diffusion_weight
        self.ddpm_batch_mul = ddpm_batch_mul

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs, ddpm_batch_mul=self.ddpm_batch_mul)
        # Expect model to return lm loss as `loss` and diffusion loss as `diffusion_loss`
        lm_loss = outputs.get("loss") if isinstance(outputs, dict) else getattr(outputs, "loss")
        diffusion_loss = (
            outputs.get("diffusion_loss") if isinstance(outputs, dict) else getattr(outputs, "diffusion_loss", None)
        )
        if diffusion_loss is None:
            total = lm_loss
        else:
            total = lm_loss + self.diffusion_weight * diffusion_loss
        return (total, outputs) if return_outputs else total


class AmpLoggingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        mode = "bf16" if args.bf16 else ("fp16" if getattr(args, "fp16", False) else "fp32")
        print(f"[AMP] Precision mode: {mode}")
        try:
            from torch.cuda.amp import GradScaler  # noqa: F401
            has_amp = True
        except Exception:
            has_amp = False
        print(f"[AMP] torch.cuda.amp available: {has_amp}")

    def on_step_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer")
        scaler_val = None
        if trainer is not None and hasattr(trainer, "scaler") and trainer.scaler is not None:
            try:
                scaler_val = float(trainer.scaler.get_scale())
            except Exception:
                scaler_val = None
        if scaler_val is not None and state.global_step % max(1, args.logging_steps) == 0:
            print(f"[AMP] GradScaler scale: {scaler_val:.2e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="YAML config with training parameters")
    p.add_argument("--model_name_or_path", type=str, default="Jmica/VibeVoice7B")
    p.add_argument("--train_file", type=str, required=False)
    p.add_argument("--valid_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=False)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--diffusion_weight", type=float, default=1.0)
    p.add_argument("--ddpm_batch_mul", type=int, default=4)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--augment_pitch_semitones", type=float, default=0.0)
    p.add_argument("--augment_tempo_pct", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_balanced_sampler", action="store_true")
    p.add_argument("--use_eval_balanced_sampler", action="store_true")
    args = p.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    # basic required checks
    if not args.train_file or not args.output_dir:
        raise SystemExit("--train_file and --output_dir are required (via CLI or YAML --config)")
    return args


def main() -> None:
    args = parse_args()

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32

    processor: VibeVoiceProcessor = VibeVoiceProcessor.from_pretrained(args.model_name_or_path)
    model: VibeVoiceForConditionalGeneration = VibeVoiceForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype
    )
    model.gradient_checkpointing_enable()
    model.to(device)

    train_ds = JsonlRows(args.train_file)
    eval_ds = JsonlRows(args.valid_file) if args.valid_file else None

    collator = VibeVoiceCollator(
        processor=processor,
        model=model,
        cfg=CollatorConfig(
            max_length=args.max_length,
            augment_pitch_semitones=args.augment_pitch_semitones,
            augment_tempo_pct=args.augment_tempo_pct,
        ),
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        bf16=args.bf16,
        logging_steps=50,
        save_steps=1000,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=1000 if eval_ds is not None else None,
        save_total_limit=2,
    )

    trainer = VibeVoiceTrainer(
        diffusion_weight=args.diffusion_weight,
        ddpm_batch_mul=args.ddpm_batch_mul,
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[AmpLoggingCallback()],
    )

    # Optional balanced sampler for speaker-agnostic emphasis
    if args.use_balanced_sampler:
        sampler = BalancedSpeakerSampler(jsonl_path=args.train_file, dataset_size=len(train_ds), seed=args.seed)
        def _custom_train_dataloader():
            return DataLoader(
                train_ds,
                batch_size=training_args.per_device_train_batch_size,
                sampler=sampler,
                collate_fn=collator,
                num_workers=0,
                pin_memory=True,
            )
        trainer.get_train_dataloader = _custom_train_dataloader

    if args.use_eval_balanced_sampler and eval_ds is not None:
        eval_sampler = BalancedSpeakerSampler(jsonl_path=args.valid_file, dataset_size=len(eval_ds), seed=args.seed)
        def _custom_eval_dataloader():
            return DataLoader(
                eval_ds,
                batch_size=training_args.per_device_eval_batch_size,
                sampler=eval_sampler,
                collate_fn=collator,
                num_workers=0,
                pin_memory=True,
            )
        trainer.get_eval_dataloader = _custom_eval_dataloader

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()


