# Extraneous Speech Detection in Two-Channel Telephone Recordings

Детекция и удаление посторонней речи (`own`, `leak`, `third`, `silence`) в двухканальных телефонных аудиозаписях (8 кГц, стерео).

## 1. Описание задачи
В двухканальной телефонной записи (канал A — левый, канал B — правый) на каждую дорожку попадает:
- `own` — речь владельца дорожки;
- `leak` — эхо/наводка речи собеседника со второй дорожки (задержка $\tau \in [20, 300]$ мс, затухание $6\text{--}40$ дБ, нелинейные искажения динамика);
- `third` — третий говорящий рядом с микрофоном;
- `silence` — пауза/фоновый шум.

Источники могут звучать одновременно (например, `own` и `leak` во время double talk).

## 2. Структура проекта
```text
├── src/
│   ├── __init__.py
│   ├── config.py         # Параметры сигнала, тракта, лимиты RTF и константы
│   ├── audio_utils.py    # Ввод-вывод WAV, нормализация, расчет RMS/dB энергии
│   ├── metrics.py        # Frame-level F1 с допуском 50 мс, косинусная близость, RTF
│   ├── baseline.py       # Базовое правило (+6 дБ)
│   ├── synthesizer.py    # (Этап 2) Физическая модель телефонного тракта
│   ├── features.py       # (Этап 3) Межканальные признаки (ILD, GCC-PHAT, когерентность)
│   ├── detector.py       # (Этап 3) Модель детекции
│   └── cleaner.py        # (Этап 3) Модуль очистки leak
├── tests/
│   ├── test_metrics.py   # Unit-тесты для допуска 50 мс, F1, эмбеддингов
│   └── test_baseline.py  # Unit-тесты базового правила и демонстрация его ошибок
├── scripts/
│   ├── download_data.py  # Скрипт загрузки компактных датасетов (~5-6 ГБ)
│   ├── generate_dataset.py # Генерация train/val/test с трудными случаями
│   └── evaluate.py       # Сводная таблица сравнения метрик
├── notebooks/
│   └── experiments.ipynb # Интерактивный исследовательский ноутбук
├── detect.py             # CLI: python detect.py input.wav result.json
├── clean.py              # CLI: python clean.py input.wav clean.wav
├── requirements.txt
└── DEVELOPMENT_LOG.md    # Подробный журнал этапов разработки
```

## 3. Установка и запуск тестов
```bash
pip install -r requirements.txt
python3 -m pytest -v tests/
```
