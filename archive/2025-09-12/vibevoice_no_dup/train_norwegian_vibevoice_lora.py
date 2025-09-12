#!/usr/bin/env python3
import argparse
import random
import numpy as np
import torch
from transformers import TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model
import yaml

from train_norwegian_vibevoice import (
    VibeVoiceProcessor,
    VibeVoiceForConditionalGeneration,
    JsonlRows,
    VibeVoiceCollator,
    CollatorConfig,
    VibeVoiceTrainer,
    AmpLoggingCallback,
)
from torch.utils.data import DataLoader
from dataset_tools.balanced_sampler import BalancedSpeakerSampler


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--model_name_or_path", type=str, default="Jmica/VibeVoice7B")
    p.add_argument("--train_file", type=str, required=False)
    p.add_argument("--valid_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=False)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--diffusion_weight", type=float, default=1.0)
    p.add_argument("--ddpm_batch_mul", type=int, default=4)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--augment_pitch_semitones", type=float, default=0.0)
    p.add_argument("--augment_tempo_pct", type=float, default=0.0)
    p.add_argument("--freeze_diffusion_head", action="store_true")
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
    if not args.train_file or not args.output_dir:
        raise SystemExit("--train_file and --output_dir are required (via CLI or YAML --config)")
    return args


def wrap_with_lora(model: VibeVoiceForConditionalGeneration, freeze_diffusion_head: bool) -> VibeVoiceForConditionalGeneration:
    # Freeze everything by default
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze diffusion head unless requested to freeze
    if not freeze_diffusion_head:
        if hasattr(model, "diffusion_head"):
            for p in model.diffusion_head.parameters():
                p.requires_grad = True

    # Apply LoRA to LM blocks
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules=TARGET_MODULES
    )
    model = get_peft_model(model, lora_cfg)
    return model


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
    model = wrap_with_lora(model, args.freeze_diffusion_head)
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


