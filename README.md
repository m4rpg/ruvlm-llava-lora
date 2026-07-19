# RuVLM — модель, которая «видит» картинку и отвечает по-русски

Проект на хакатон. Если совсем коротко: берётся готовая визуально-языковая модель
(VLM — это когда на вход идёт и картинка, и текст, и модель может, например, отвечать на
вопросы по изображению) и дообучается на открытых русских данных от VK, чтобы она лучше
работала именно на русском. Качество проверяется на двух русских бенчмарках — GQA-ru и
MMBench-ru — и цель в том, чтобы выбить метрики повыше.

Зачем это вообще нужно — расписано в `docs/PROJECT.md`. Как всё сделано изнутри —
в `docs/SOLUTION.md`.

## Быстрый старт в Colab

Проще всего — открыть `notebooks/RuVLM_colab.ipynb` в Google Colab (File → Upload notebook,
либо открыть файл прямо из репозитория). Ноутбук сам поставит зависимости, скачает данные
deepvk, дообучит LoRA и заполнит таблицу с метриками. Нужен только GPU-рантайм — бесплатной
T4 хватает для демо-прогона.

## Что где лежит

- `docs/PROJECT.md` — про что проект: цель, задачи, ожидаемый результат, зачем это надо.
- `docs/SOLUTION.md` — самое подробное: архитектура, какие данные и как они используются, обучение, оценка.
- `docs/MODEL_CARD.md` — описание получившейся модели (в формате карточки, как на Hugging Face).
- `docs/RESULTS.md` — таблицы с метриками и разбор, где модель ошибается.
- `docs/RuVLM_presentation.pptx` — презентация.
- `notebooks/RuVLM_colab.ipynb` — запуск обучения и оценки в Google Colab.
- `src/` — весь код: подготовка данных, обучение, оценка, инференс.
- `configs/`, `scripts/` — конфиг эксперимента и скрипты запуска.

## Какие данные используются

Всё взято из открытой коллекции deepvk на Hugging Face
(deepvk/Vision-Language-Modeling). Ничего закрытого — только то, что VK сами выложили:

- `deepvk/LLaVA-Instruct-ru` (144k) — на этом модель учится следовать инструкциям по картинкам.
- `deepvk/GQA-ru` (80.1k) — train-часть идёт в обучение, тестовая — на оценку (метрика ExactMatch).
- `deepvk/MMBench-ru` (3.91k) — только оценка, в обучение он не попадает принципиально.
- `deepvk/llava-saiga-8b` (8B) — точка старта, это уже готовая русская VLM от deepvk.

Подробнее, что именно куда идёт, — в `docs/SOLUTION.md`, раздел про данные.

## Как запустить локально

```bash
# окружение
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# сначала смотрим, что выдаёт исходная модель без дообучения (baseline)
python -m src.evaluate --config configs/llava_saiga_lora.yaml --benchmark gqa     --model deepvk/llava-saiga-8b
python -m src.evaluate --config configs/llava_saiga_lora.yaml --benchmark mmbench --model deepvk/llava-saiga-8b

# дообучение LoRA на данных deepvk
python -m src.train --config configs/llava_saiga_lora.yaml

# оценка уже дообученной модели
python -m src.evaluate --config configs/llava_saiga_lora.yaml --benchmark gqa     --adapter outputs/ruvlm-lora
python -m src.evaluate --config configs/llava_saiga_lora.yaml --benchmark mmbench --adapter outputs/ruvlm-lora

# и можно просто спросить у модели что-нибудь по картинке
python -m src.inference --adapter outputs/ruvlm-lora --image assets/demo.jpg --prompt "Что изображено на картинке?"
```

На маленькой GPU (16 ГБ) добавьте `--load-in-4bit` к командам оценки/инференса и поставьте
`load_in_4bit: true` в конфиге для обучения (QLoRA).

## Целевые метрики

| Модель | GQA-ru (ExactMatch) | MMBench-ru (Acc) |
|--------|:---:|:---:|
| llava-1.5-7b (английская, для сравнения) | 28.39 | 52.25 |
| deepvk/llava-saiga-8b (русская, точка старта) | 51.44 | 56.65 |
| RuVLM (цель) | ≥ 53 | ≥ 58 |

Цифры двух верхних моделей не выдуманы — они из карточки `deepvk/llava-saiga-8b`.

## Лицензия

Код — под MIT (см. `LICENSE`). Датасеты и базовые модели идут под своими лицензиями, их нужно
смотреть на страницах на Hugging Face.
