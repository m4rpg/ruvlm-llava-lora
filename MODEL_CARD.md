# Карточка модели: RuVLM (LoRA поверх LLaVA-Saiga-8B)

Здесь — описание модели, которая получается в проекте. Оформлено как обычная карточка модели с
Hugging Face, чтобы читалось привычно.

## Коротко о модели

| Поле | Значение |
|------|----------|
| Название | RuVLM-llava-saiga-8b-lora |
| Что умеет | отвечать на вопросы по картинке, описывать её, рассуждать |
| Язык | в основном русский, английский частично тянется от базы |
| Тип | LoRA-адаптер поверх `deepvk/llava-saiga-8b` |
| Языковая модель | `IlyaGusev/saiga_llama3_8b` (Llama-3-8B, докрученная под русский) |
| Энкодер картинок | CLIP ViT-L/14-336, заморожен |
| Переходник | MLP из базы (по желанию тоже дообучается) |
| Метод обучения | LoRA (r=16, alpha=32), bf16, одна GPU |
| Данные обучения | `deepvk/LLaVA-Instruct-ru` + train-часть `deepvk/GQA-ru` |
| Лицензия | наследуется от базовой модели и датасетов |

## Как пользоваться

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from PIL import Image
import torch

base = "deepvk/llava-saiga-8b"
adapter = "outputs/ruvlm-lora"     # локальная папка или repo id

processor = AutoProcessor.from_pretrained(base)
model = LlavaForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)      # накатываем LoRA поверх базы
model = model.merge_and_unload()                        # можно вплавить адаптер в веса

img = Image.open("assets/demo.jpg").convert("RGB")
messages = [{"role": "user", "content": "<image>\nЧто изображено на картинке?"}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(images=[img], text=prompt, return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(processor.decode(out[0], skip_special_tokens=True))
```

## Метрики

Считаются кодом из `../src/evaluate.py`, сам протокол описан в `SOLUTION.md` (раздел про
оценку). Цифры референсов взяты из карточки `deepvk/llava-saiga-8b`.

| Модель | GQA-ru (ExactMatch) | MMBench-ru (Accuracy) |
|--------|:---:|:---:|
| llava-1.5-7b (для сравнения) | 28.39 | 52.25 |
| deepvk/llava-saiga-8b (точка старта) | 51.44 | 56.65 |
| RuVLM | заполняется после прогона | заполняется после прогона |

Строка с моделью RuVLM заполняется реальными числами после того, как прогнаны обучение и
оценка — см. `RESULTS.md`.

## Откуда данные для обучения

- `deepvk/LLaVA-Instruct-ru` — 144k русских инструкций по картинкам.
- `deepvk/GQA-ru` — train-часть, короткие ответы на визуальные вопросы.

MMBench-ru в обучении не участвует, остаётся только на оценку.

## Где такое пригодится

- Ассистенты, которые понимают картинки и говорят по-русски (вопрос-ответ, описание сцены).
- Автоописание контента — товары, фото; помощь незрячим; поддержка модерации.

## Чего от неё не стоит ждать

- Может нафантазировать по мелким деталям, слабовата в чтении текста на картинке и в счёте
  объектов.
- Тянет за собой перекосы базовой модели и обучающих данных.
- Не для медицины, юридических решений и распознавания конкретных людей.

## Как повторить

- Конфиг: `../configs/llava_saiga_lora.yaml`
- Обучение: `python -m src.train --config configs/llava_saiga_lora.yaml`
- Оценка: `python -m src.evaluate --config ... --benchmark {gqa,mmbench} --adapter outputs/ruvlm-lora`
