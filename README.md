# Детекция посторонней речи в двухканальной телефонной записи

Проект определяет в каждой дорожке `own`, `leak` и `third` на сетке 10 мс и удаляет `leak`, сохраняя речь владельца. Вход: стерео WAV, 8 кГц; канал 0 это A, канал 1 это B.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для воспроизводимой CPU-проверки:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## CLI

Детекция интервалов:

```bash
python detect.py input.wav result.json
```

Очистка обеих дорожек:

```bash
python clean.py input.wav clean.wav
python clean.py input.wav clean.wav --preview clean_16k.wav
```

## Данные и синтез

Диалоги синтезируются только из русской речи `data/russian_speech/`:

- `ru_spk_1..4` используются для train;
- `ru_spk_5..7` используются для test;
- VoxCeleb не используется как источник диалогов и оставлен для независимых embedding-экспериментов;
- применяется speed/pitch augmentation в диапазоне `0.88..1.12`;
- RIRS_NOISES/OpenSLR 28 используются для RIR и фонового шума.

Сгенерировать демонстрационные WAV, JSON-разметку и метрики:

```bash
python scripts/generate_demo_samples.py
```

Сгенерировать listening set из 15 записей, разделённых по измеренному качеству:

```bash
python scripts/generate_quality_samples.py
```

Результат находится в `samples/quality_listening/`: по 5 папок `bad`, `medium`, `good`. В каждой папке есть `input.wav`, `cleaned.wav`, `reference_A.wav`, `metrics.json`; общий индекс: `manifest.json`.

## Оценка и диагностика

Полная оценка на пяти сценариях:

```bash
python evaluate.py --num_per_scenario 3 --duration_sec 7.0 --seed 42
```

Диагностические спектрограммы и WAV:

```bash
python scripts/visualize_diagnostics.py
```

Исследовательский notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/experiments_and_pipeline.ipynb
```

## Архитектура

`src/features.py` вычисляет 36 признаков на 10-мс кадре: энергию, ILD, GCC-PHAT peak и задержку, NLMS residual, спектральные дескрипторы, 12 mel-полос A, 12 mel-полос B и три Speaker Anchor-признака.

Замороженный детектор `1D-TCN` использует эти признаки и выдаёт независимые вероятности `own/leak/third`. Файл `models/detector_weights.pt` и код детектора не переобучаются в финальной полировке.

Клинер выполняет Time-Delay Alignment для задержки 2..30 кадров, оценивает частотный коэффициент утечки `H_BA` по aligned B и применяет 32-полосную IRM-модель. На чистой речи gain равен 1; на pure leak применяется noise-floor gating; double-talk обрабатывается гибридной IRM/аналитической маской с сохранением комфортного шума.

## Тесты

```bash
pytest tests/
```

Текущая проверка: 20 тестов проходят. Финальный измеренный pipeline RTF на одном CPU-ядре: около `0.026`.

## Артефакты

- [REPORT.md](REPORT.md) — аналитический отчёт и ограничения метода;
- [docs/evaluation_results.md](docs/evaluation_results.md) — машинно сохранённый отчёт оценки;
- `samples/diagnostics/` — спектрограммы и эталонные WAV;
- `samples/quality_listening/` — 15 samples для прослушивания;
- `notebooks/experiments_and_pipeline.ipynb` — воспроизводимый notebook.
