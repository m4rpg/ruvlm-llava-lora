"""Загрузка и подготовка открытых датасетов deepvk для обучения LLaVA-модели.

Датасеты:
  - deepvk/LLaVA-Instruct-ru : инструкции по картинкам (диалоги)
  - deepvk/GQA-ru            : короткие ответы визуального рассуждения
MMBench-ru здесь НЕ используется — это held-out для оценки.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Any

import torch
from datasets import concatenate_datasets, interleave_datasets, load_dataset
from PIL import Image

from .config import Config, MixSource

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Приведение разных схем к единому формату {image, question, answer}
# ---------------------------------------------------------------------------
def _to_pil(img: Any) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    raise ValueError(f"Неизвестный формат изображения: {type(img)}")


def _instruct_to_pairs(example: dict) -> dict:
    """LLaVA-Instruct-ru хранит поле 'conversations' (list из human/gpt).

    Разворачиваем в один сведённый диалог user→assistant (берём первую пару).
    """
    convs = example.get("conversations") or example.get("messages") or []
    question, answer = "", ""
    for turn in convs:
        role = turn.get("from") or turn.get("role")
        val = turn.get("value") or turn.get("content", "")
        if role in ("human", "user") and not question:
            question = val.replace("<image>", "").strip()
        elif role in ("gpt", "assistant") and question and not answer:
            answer = val.strip()
            break
    return {"question": question, "answer": answer, "image": example["image"]}


def _gqa_to_pairs(example: dict) -> dict:
    q = example.get("question", "")
    a = example.get("answer", "") or example.get("fullAnswer", "")
    return {"question": q, "answer": a, "image": example["image"]}


_ADAPTERS = {
    "deepvk/LLaVA-Instruct-ru": _instruct_to_pairs,
    "deepvk/GQA-ru": _gqa_to_pairs,
}


def _load_one(src: MixSource, cfg: Config):
    ds = load_dataset(src.id, split=src.split)
    adapter = _ADAPTERS.get(src.id, _gqa_to_pairs)
    keep = ds.column_names
    ds = ds.map(adapter, remove_columns=[c for c in keep if c != "image"],
                num_proc=cfg.data.get("num_proc", 4))
    ds = ds.filter(lambda e: bool(e["question"]) and bool(e["answer"]))
    return ds


def build_train_dataset(cfg: Config):
    """Собирает обучающую смесь по весам из конфига (взвешенный interleave)."""
    parts, weights = [], []
    for src in cfg.mix:
        parts.append(_load_one(src, cfg))
        weights.append(src.weight)

    if len(parts) == 1:
        mixed = parts[0]
    else:
        total = sum(weights)
        probs = [w / total for w in weights]
        mixed = interleave_datasets(parts, probabilities=probs,
                                    seed=cfg.seed, stopping_strategy="all_exhausted")

    n = cfg.data.get("max_train_samples")
    if n:
        mixed = mixed.select(range(min(n, len(mixed))))
    return mixed


def dedup_by_image_question(ds):
    """Удаляем дубликаты по (хеш изображения, нормализованный вопрос)."""
    seen = set()
    keep_idx = []
    for i, ex in enumerate(ds):
        img = _to_pil(ex["image"])
        h = hashlib.md5(img.tobytes()).hexdigest()[:16]
        key = (h, ex["question"].strip().lower())
        if key not in seen:
            seen.add(key)
            keep_idx.append(i)
    return ds.select(keep_idx)


# ---------------------------------------------------------------------------
# Коллатор: формируем chat-шаблон и маскируем лосс по вопросу
# ---------------------------------------------------------------------------
@dataclass
class LlavaCollator:
    processor: Any
    max_seq_len: int = 2048
    image_token: str = "<image>"

    def _prompt(self, question: str) -> str:
        messages = [{"role": "user", "content": f"{self.image_token}\n{question}"}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        images, prompts, full_texts = [], [], []
        for f in features:
            images.append(_to_pil(f["image"]))
            prompt = self._prompt(f["question"])
            prompts.append(prompt)
            full_texts.append(prompt + f["answer"] + self.processor.tokenizer.eos_token)

        batch = self.processor(
            images=images, text=full_texts,
            padding=True, truncation=True, max_length=self.max_seq_len,
            return_tensors="pt",
        )

        # Маскируем всё, кроме токенов ответа ассистента.
        labels = batch["input_ids"].clone()
        prompt_lens = [
            len(self.processor.tokenizer(p, add_special_tokens=False)["input_ids"])
            for p in prompts
        ]
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = IGNORE_INDEX
        labels[batch["input_ids"] == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
        batch["labels"] = labels
        return batch
