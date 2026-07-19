"""Загрузка YAML-конфигурации в удобные dataclass-объекты."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class MixSource:
    id: str
    split: str = "train"
    weight: float = 1.0


@dataclass
class Config:
    """Тонкая обёртка над dict из YAML: доступ и как к атрибутам, и как к словарю."""

    raw: dict[str, Any]

    # --- удобные геттеры верхнего уровня ---
    @property
    def run_name(self) -> str:
        return self.raw.get("run_name", "run")

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def lora(self) -> dict[str, Any]:
        return self.raw["lora"]

    @property
    def train(self) -> dict[str, Any]:
        return self.raw["train"]

    @property
    def eval(self) -> dict[str, Any]:
        return self.raw["eval"]

    @property
    def mix(self) -> list[MixSource]:
        return [MixSource(**m) for m in self.data["mix"]]


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
