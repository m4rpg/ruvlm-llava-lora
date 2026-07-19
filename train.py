"""Дообучение LLaVA-модели LoRA-адаптером на открытых данных deepvk.

Запуск:
    python -m src.train --config configs/llava_saiga_lora.yaml
"""
from __future__ import annotations

import argparse
import os

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from .config import Config, load_config, set_seed
from .data import LlavaCollator, build_train_dataset, dedup_by_image_question


def _dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(name, torch.float32)


def load_model_and_processor(cfg: Config):
    m = cfg.model
    kwargs: dict = {"torch_dtype": _dtype(m.get("dtype", "bfloat16"))}

    if m.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_dtype(m.get("dtype", "bfloat16")),
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"

    model = LlavaForConditionalGeneration.from_pretrained(m["base_id"], **kwargs)
    processor = AutoProcessor.from_pretrained(m["base_id"])
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return model, processor


def apply_lora(model, cfg: Config):
    # Vision encoder всегда заморожен.
    if hasattr(model, "vision_tower"):
        for p in model.vision_tower.parameters():
            p.requires_grad_(False)

    # Для QLoRA (4-bit) веса нужно подготовить: привести normы к fp32,
    # включить input grads и т.п. — иначе градиенты не потекут.
    if cfg.model.get("load_in_4bit"):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.train.get("gradient_checkpointing", True)
        )

    lora = LoraConfig(
        r=cfg.lora["r"],
        lora_alpha=cfg.lora["alpha"],
        lora_dropout=cfg.lora["dropout"],
        target_modules=cfg.lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    # Опционально дообучаем MLP-проектор вместе с LoRA (эксперимент E3).
    if cfg.train.get("unfreeze_projector") and hasattr(model, "multi_modal_projector"):
        for p in model.multi_modal_projector.parameters():
            p.requires_grad_(True)

    model.print_trainable_parameters()
    return model


def build_training_args(cfg: Config) -> TrainingArguments:
    t = cfg.train
    out_dir = os.path.join(t.get("output_dir", "outputs"), cfg.run_name)
    return TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=t.get("epochs", 1),
        per_device_train_batch_size=t.get("per_device_batch_size", 4),
        gradient_accumulation_steps=t.get("grad_accum", 16),
        learning_rate=float(t.get("lr", 2e-4)),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        weight_decay=t.get("weight_decay", 0.0),
        lr_scheduler_type=t.get("lr_scheduler", "cosine"),
        bf16=cfg.model.get("dtype") == "bfloat16",
        fp16=cfg.model.get("dtype") == "float16",
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        logging_steps=t.get("logging_steps", 20),
        save_steps=t.get("save_steps", 200),
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,   # важно: сохраняем 'image' для коллатора
        dataloader_num_workers=4,
        seed=cfg.seed,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-dedup", action="store_true", help="пропустить дедупликацию")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    print(">> Загрузка модели и процессора:", cfg.model["base_id"])
    model, processor = load_model_and_processor(cfg)
    model = apply_lora(model, cfg)
    if cfg.train.get("gradient_checkpointing", True):
        model.enable_input_require_grads()

    print(">> Сборка обучающей смеси (только открытые данные deepvk)")
    train_ds = build_train_dataset(cfg)
    if not args.no_dedup:
        train_ds = dedup_by_image_question(train_ds)
    print(f"   Примеров в обучении: {len(train_ds)}")

    collator = LlavaCollator(
        processor=processor,
        max_seq_len=cfg.data.get("max_seq_len", 2048),
        image_token=cfg.model.get("image_token", "<image>"),
    )

    trainer = Trainer(
        model=model,
        args=build_training_args(cfg),
        train_dataset=train_ds,
        data_collator=collator,
    )

    trainer.train()

    out_dir = os.path.join(cfg.train.get("output_dir", "outputs"), cfg.run_name)
    trainer.save_model(out_dir)          # сохраняем LoRA-адаптер
    processor.save_pretrained(out_dir)
    print(f">> Адаптер сохранён в {out_dir}")


if __name__ == "__main__":
    main()
