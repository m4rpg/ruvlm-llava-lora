"""Оценка VLM на бенчмарках GQA-ru (ExactMatch) и MMBench-ru (Accuracy).

Примеры:
    # baseline
    python -m src.evaluate --config configs/llava_saiga_lora.yaml --benchmark gqa \
        --model deepvk/llava-saiga-8b
    # дообученный адаптер + разбивка по категориям
    python -m src.evaluate --config configs/llava_saiga_lora.yaml --benchmark mmbench \
        --adapter outputs/ruvlm-lora --per-category
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .config import load_config, set_seed
from .data import _to_pil
from .metrics import (
    circular_variants,
    exact_match,
    gqa_accuracy,
    mmbench_accuracy,
    parse_choice,
)


def load_for_inference(base_id: str, adapter: str | None, dtype=torch.bfloat16,
                       load_in_4bit: bool = False):
    kwargs: dict = {"torch_dtype": dtype, "device_map": "auto"}
    if load_in_4bit:
        # 4-bit нужен, чтобы 8B-модель влезла в одну небольшую GPU (например, T4 в Colab).
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = LlavaForConditionalGeneration.from_pretrained(base_id, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        # merge_and_unload несовместим с 4-bit — в этом случае оставляем адаптер как есть.
        if not load_in_4bit:
            model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(adapter or base_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.eval()
    return model, processor


@torch.no_grad()
def generate(model, processor, image, question: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": f"<image>\n{question}"}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=[_to_pil(image)], text=prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0, inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# GQA-ru
# ---------------------------------------------------------------------------
def eval_gqa(model, processor, cfg) -> dict:
    c = cfg.eval["gqa"]
    ds = load_dataset(c["id"], split=c["split"])
    if c.get("max_samples"):
        ds = ds.select(range(min(c["max_samples"], len(ds))))

    preds, golds = [], []
    for ex in tqdm(ds, desc="GQA-ru"):
        pred = generate(model, processor, ex["image"], ex["question"], c["max_new_tokens"])
        preds.append(pred)
        golds.append(ex["answer"])
    score = gqa_accuracy(preds, golds)
    return {"benchmark": "GQA-ru", "metric": "ExactMatch", "score": round(score, 2), "n": len(golds)}


# ---------------------------------------------------------------------------
# MMBench-ru
# ---------------------------------------------------------------------------
_MMB_PROMPT = (
    "{hint}Вопрос: {question}\n"
    "Варианты ответа:\n{options}\n"
    "Ответь ТОЛЬКО одной буквой правильного варианта (A, B, C или D)."
)


def _format_options(opts: dict[str, str]) -> str:
    return "\n".join(f"{k}. {v}" for k in "ABCD" if opts.get(k) not in (None, "", "nan"))


def eval_mmbench(model, processor, cfg, per_category: bool = False) -> dict:
    c = cfg.eval["mmbench"]
    ds = load_dataset(c["id"], split=c["split"])
    if c.get("max_samples"):
        ds = ds.select(range(min(c["max_samples"], len(ds))))

    pred_letters, gold_letters = [], []
    cat_hits: dict[str, list] = defaultdict(list)

    for ex in tqdm(ds, desc="MMBench-ru"):
        opts = {k: ex.get(k) for k in "ABCD"}
        answer = ex["answer"]
        hint = f"Контекст: {ex['hint']}\n" if ex.get("hint") else ""

        variants = (
            circular_variants(opts, answer) if c.get("circular_eval")
            else [(opts, answer)]
        )
        all_correct = True
        last_pred = None
        for remap, new_ans in variants:
            prompt = _MMB_PROMPT.format(hint=hint, question=ex["question"],
                                        options=_format_options(remap))
            raw = generate(model, processor, ex["image"], prompt, c["max_new_tokens"])
            letter = parse_choice(raw)
            last_pred = letter
            if letter != new_ans:
                all_correct = False
        # для CircularEval пример верен только если верны все перестановки
        pred_letters.append(answer if all_correct else (last_pred or "X"))
        gold_letters.append(answer)
        cat_hits[ex.get("category", "unknown")].append(int(all_correct))

    score = mmbench_accuracy(pred_letters, gold_letters)
    result = {"benchmark": "MMBench-ru", "metric": "Accuracy",
              "score": round(score, 2), "n": len(gold_letters),
              "circular_eval": bool(c.get("circular_eval"))}
    if per_category:
        result["per_category"] = {
            cat: round(100.0 * sum(v) / len(v), 2) for cat, v in sorted(cat_hits.items())
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--benchmark", choices=["gqa", "mmbench"], required=True)
    ap.add_argument("--model", default=None, help="base_id (переопределяет конфиг)")
    ap.add_argument("--adapter", default=None, help="путь к LoRA-адаптеру")
    ap.add_argument("--per-category", action="store_true")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="4-bit загрузка (для маленькой GPU, например T4 в Colab)")
    ap.add_argument("--out", default=None, help="куда сохранить JSON с результатом")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    base_id = args.model or cfg.model["base_id"]

    model, processor = load_for_inference(base_id, args.adapter, load_in_4bit=args.load_in_4bit)

    if args.benchmark == "gqa":
        res = eval_gqa(model, processor, cfg)
    else:
        res = eval_mmbench(model, processor, cfg, per_category=args.per_category)

    res["model"] = base_id
    res["adapter"] = args.adapter
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
