"""Метрики бенчмарков.

GQA-ru      → ExactMatch по нормализованному однословному ответу.
MMBench-ru  → Accuracy по выбранной букве варианта (A/B/C/D), опц. CircularEval.
"""
from __future__ import annotations

import re
import string
import unicodedata

# ---------------------------------------------------------------------------
# GQA-ru: ExactMatch
# ---------------------------------------------------------------------------

# Служебные слова, которые не должны влиять на совпадение коротких ответов.
_STOP = {"это", "на", "в", "the", "a", "an", "is", "are", "да,", "нет,"}


def normalize_answer(text: str) -> str:
    """Нормализация для ExactMatch: регистр, пунктуация, ё→е, лишние пробелы."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).strip().lower()
    text = text.replace("ё", "е")
    # берём первую строку/предложение — модель иногда добавляет пояснение
    text = re.split(r"[.\n]", text)[0]
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t and t not in _STOP]
    return " ".join(tokens).strip()


def exact_match(pred: str, gold: str) -> bool:
    """1, если нормализованные строки совпали. Учитывает частый случай да/нет."""
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p == g and g != "":
        return True
    # да/нет: засчитываем, если ключевое слово присутствует
    if g in {"да", "нет"} and g in p.split():
        return True
    return False


def gqa_accuracy(preds: list[str], golds: list[str]) -> float:
    assert len(preds) == len(golds)
    if not preds:
        return 0.0
    hits = sum(exact_match(p, g) for p, g in zip(preds, golds))
    return 100.0 * hits / len(preds)


# ---------------------------------------------------------------------------
# MMBench-ru: выбор буквы варианта
# ---------------------------------------------------------------------------

_CHOICE_RE = re.compile(r"\b([ABCD])\b")


def parse_choice(text: str) -> str | None:
    """Достаём букву A/B/C/D из ответа модели. None — если не распарсилось."""
    if not text:
        return None
    t = text.strip().upper()
    # частый случай: ответ начинается с буквы
    if t and t[0] in "ABCD" and (len(t) == 1 or not t[1].isalpha()):
        return t[0]
    m = _CHOICE_RE.search(t)
    return m.group(1) if m else None


def mmbench_accuracy(pred_letters: list[str | None], gold_letters: list[str]) -> float:
    assert len(pred_letters) == len(gold_letters)
    if not gold_letters:
        return 0.0
    hits = sum(1 for p, g in zip(pred_letters, gold_letters) if p is not None and p == g)
    return 100.0 * hits / len(gold_letters)


def circular_variants(options: dict[str, str], answer: str) -> list[tuple[dict[str, str], str]]:
    """CircularEval: циклические перестановки непустых вариантов.

    Возвращает список (перемешанные_варианты, новая_буква_правильного_ответа).
    Пример засчитывается верным, только если модель права на ВСЕХ перестановках.
    """
    letters = [c for c in "ABCD" if options.get(c) not in (None, "", "nan")]
    texts = [options[c] for c in letters]
    gold_text = options[answer]
    variants = []
    for shift in range(len(letters)):
        rolled = texts[shift:] + texts[:shift]
        remap = {letters[i]: rolled[i] for i in range(len(letters))}
        new_answer = next(l for l, txt in remap.items() if txt == gold_text)
        variants.append((remap, new_answer))
    return variants
