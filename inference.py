"""Демо-инференс: задать вопрос по картинке дообученной модели.

    python -m src.inference --adapter outputs/ruvlm-lora \
        --image assets/demo.jpg --prompt "Что изображено на картинке?"
"""
from __future__ import annotations

import argparse

import torch
from PIL import Image

from .evaluate import generate, load_for_inference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="deepvk/llava-saiga-8b")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="Опиши подробно, что изображено на картинке.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="4-bit загрузка (для маленькой GPU, например T4 в Colab)")
    args = ap.parse_args()

    model, processor = load_for_inference(args.base, args.adapter, dtype=torch.bfloat16,
                                          load_in_4bit=args.load_in_4bit)
    image = Image.open(args.image).convert("RGB")
    answer = generate(model, processor, image, args.prompt, args.max_new_tokens)

    print("Вопрос :", args.prompt)
    print("Ответ  :", answer)


if __name__ == "__main__":
    main()
