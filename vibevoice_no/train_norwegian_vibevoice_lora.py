#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)
import yaml

from train_norwegian_vibevoice import (
    JsonlTtsDataset,
    TtsDataCollator,
    _read_jsonl,
    _build_balanced_sampler,
)

from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration

from peft import LoraConfig, get_peft_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA finetune VibeVoice7B for Norwegian TTS")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--use_balanced_sampler", action="store_true")
    parser.add_argument("--use_eval_balanced_sampler", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


def _guess_target_modules(model: torch.nn.Module) -> List[str]:
    names: List[str] = []
    for name, module in model.named_modules():
        low = name.lower()
        if any(k in low for k in ["attn", "attention", "mlp", "proj", "ff", "diffusion", "text"]):
            # Only leaf modules
            if len(list(module.children())) == 0:
                names.append(name.split(".")[-1])
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for n in names:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq or ["q_proj", "k_proj", "v_proj", "o_proj"]


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = 42
    set_seed(seed)

    model_name = cfg.get("model_name_or_path", "Jmica/VibeVoice7B")
    bf16_flag = bool(cfg.get("bf16", True)) and (torch.cuda.is_available())
    torch_dtype = torch.bfloat16 if bf16_flag else None

    processor = VibeVoiceProcessor.from_pretrained(model_name)
    base_model = VibeVoiceForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    target_modules = _guess_target_modules(base_model)
    lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=target_modules,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    try:
        model.print_trainable_parameters()
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
        evaluation_strategy="steps",
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
    # Save adapter only
    trainer.model.save_pretrained(training_args.output_dir)
    try:
        processor.save_pretrained(training_args.output_dir)
    except Exception:
        pass


if __name__ == "__main__":
    main()


