# Детекция посторонней речи

Проект обрабатывает стерео телефонную запись WAV (8 кГц, 16 бит). Для каждого 10-мс кадра каждого канала определяются `own`, `leak` и `third`, после чего просачивание удаляется спектральным клинером.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для замера производительности на одном CPU-потоке:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Использование

```bash
python detect.py input.wav predictions.json
python clean.py input.wav cleaned.wav
```

`detect.py` сохраняет интервалы по каналам `A` и `B`. `clean.py` сохраняет очищенную стереозапись; опция `--preview` дополнительно создаёт 16-кГц preview.

## Данные и обучение

Данные для синтеза берутся из `data/russian_speech/`. Дикторы `spk_1..4` используются для обучения, `spk_5..7` остаются тестовыми. RIR и шумы находятся в `data/rirs_noises/`.

Скрипты подготовки и обучения:

```bash
python scripts/download_data.py
python scripts/train_detector.py
python scripts/train_cleaner.py
```

Скрипты сохраняют веса в `models/`. Для воспроизводимого запуска всех этапов откройте [notebooks/pipeline.ipynb](notebooks/pipeline.ipynb). В notebook по умолчанию используются готовые веса; переменная `TRAIN = True` включает обучение.

## Проверка

```bash
pytest tests/
python evaluate.py --num_per_scenario 3 --duration_sec 7.0 --seed 42
```

Основные ограничения и результаты описаны в [REPORT.md](REPORT.md), машинный отчёт оценки находится в [docs/evaluation_results.md](docs/evaluation_results.md).

## Структура

- `src/` — синтез данных, признаки, детектор, клинер и метрики;
- `detect.py`, `clean.py` — публичные CLI;
- `scripts/` — загрузка данных и обучение;
- `evaluate.py`, `tests/` — валидация;
- `models/` — готовые веса;
- `samples/` — примеры и результаты, сохранённые без изменений.
