# 🍎 AI Fruit & Vegetable Classifier Bot

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Aiogram](https://img.shields.io/badge/aiogram-3.x-blue?logo=telegram)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Telegram-бот на базе свёрточной нейронной сети, который распознаёт **257 видов фруктов и овощей** по фотографии. Полноценное End-to-End решение: от обучения ML-модели до интерфейса в мессенджере.

---

## О проекте

Пользователь отправляет фото — бот за доли секунды определяет, что изображено, и возвращает название класса вместе с показателем уверенности модели (Confidence Score).

---

## Возможности

- **257 классов** — редкие сорта яблок, экзотические фрукты, различные виды орехов и многое другое
- **Confidence Score** — бот показывает вероятность предсказания (например: *«Манго, уверенность: 97.3%»*)
- **Асинхронная обработка** — Aiogram 3 + asyncio, бот не блокируется при одновременных запросах
- **Валидация ввода** — корректно обрабатывает текстовые сообщения и неподдерживаемые форматы файлов

---

## Стек технологий

| Область | Инструменты |
|---|---|
| ML / CV | PyTorch, Torchvision, Pillow |
| Backend | Python 3.9+, Aiogram 3, asyncio |
| Обучение | tqdm, matplotlib |

---

## Архитектура и ML-пайплайн

### Модель

Backbone — **ResNet18**, выбран за баланс скорости инференса и точности.  
Применён **Transfer Learning** (Fine-tuning): веса предобучены на ImageNet, последний полносвязный слой заменён под 257 классов.

### Датасет и обучение

- **Датасет:** [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits)
- **Препроцессинг:** изображения приводятся к размеру 100×100 пикселей и нормализуются
- **Оптимизатор:** Adam с `weight_decay=1e-4`
- **Loss:** CrossEntropyLoss
- **Scheduler:** ReduceLROnPlateau — автоматически снижает Learning Rate при стагнации метрик
- **AMP:** `torch.cuda.amp` для смешанной точности — ускоряет обучение и снижает потребление VRAM

---

## Установка и запуск

### 1. Клонируй репозиторий

```bash
git clone https://github.com/ckfdf935/Fruit-classifier
cd Fruit-classifier
```

### 2. Создай виртуальное окружение

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Установи зависимости

```bash
pip install -r requirements.txt
```

### 4. Добавь веса модели

Скачай файл `fruits_model.pth` из раздела [Releases](../../releases) и положи его в корень проекта.  
Либо обучи модель самостоятельно:

```bash
python training.py
```

### 5. Укажи токен бота

Замени `TOKEN` в файле `TelegramBot.py` на токен своего бота от [@BotFather](https://t.me/BotFather).

> Для продакшена используй переменные окружения или `.env`-файл, не хардкодь токен в коде.

### 6. Запусти бота

```bash
python TelegramBot.py
```

---

## Структура проекта

```
Fruit/
├── predictions.py        # Файл для получения предсказания 
├── TelegramBot.py        # Основной файл бота
├── training.py           # Скрипт обучения модели
├── fruits_model.pth      # Веса обученной модели (не включены в репозиторий)
├── requirements.txt
└── README.md
```

---
