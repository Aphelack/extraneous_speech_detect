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

Диалоги синтезируются из двух корпусов: `data/russian_speech/` и `data/voxceleb1_test/`. Дикторы строго разделены между Train и Test без пересечений:

- **Train**: 28 дикторов VoxCeleb1 + 4 русских (`spk_1..4`);
- **Test**: 12 дикторов VoxCeleb1 + 3 русских (`spk_5..7`).

RIR и шумы находятся в `data/rirs_noises/`.

Сценарии синтеза: `normal`, `double_talk`, `high_leak_lt_10db`, `path_drift`, `third_speaker`, `no_leak`. В 20% записей утечка отсутствует на одной или обеих дорожках; затухание утечки покрывает весь диапазон 6–40 дБ, задержка — 20–300 мс с независимым дрейфом для каждого направления.

Эталонная разметка строится не по прямоугольникам расписания, а по фактической слышимости каждого компонента: кадр помечается `own`/`leak`/`third`, только если энергия соответствующего компонента превышает шум дорожки в 1.5 раза и уровень −52 dBFS.

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
python evaluate.py --dataset-path data/russian_speech --num_per_scenario 10 --duration_sec 7.0 --seed 42
```

`evaluate.py` считает **глобальный micro-F1**: TP/FP/FN суммируются по всем кадрам всех записей и обеих дорожек, и F1 вычисляется один раз на класс. Кадры в пределах 50 мс от границы эталона полностью исключаются из подсчёта (не дают ни TP, ни FP, ни FN). Дополнительно выводится FPR для `leak` на срезе `no_leak`.

`evaluate.py` принимает путь к корпусу с папками дикторов. В каждой папке должны находиться WAV-файлы одного диктора:

```text
dataset/
├── speaker_01/
│   ├── utterance_001.wav
│   └── utterance_002.wav
├── speaker_02/
│   └── utterance_001.wav
└── speaker_03/
	└── utterance_001.wav
```

Все WAV приводятся к 8 кГц автоматически. Дикторы сортируются по имени: первая часть используется для обучения, вторая часть — для оценки. Для текущего тестового корпуса пример структуры такой:

```text
data/russian_speech/
├── spk_1/*.wav
├── spk_2/*.wav
├── spk_3/*.wav
├── spk_4/*.wav   # train
├── spk_5/*.wav
├── spk_6/*.wav
└── spk_7/*.wav   # test
```

Для другого корпуса укажите его путь:

```bash
python evaluate.py --dataset-path /path/to/dataset --num_per_scenario 30
```

Для подготовленного корпуса из пар `input.wav` и `ground_truth.json` можно обработать все записи без случайной генерации новых диалогов:

```bash
python scripts/prepare_evaluation_corpus.py \
	--validation-path data/russian_speech_validation \
	--output-path data/evaluation_corpus \
	--records-per-scenario 10 \
	--duration-sec 10
python evaluate.py --dataset-path data/evaluation_corpus --all-corpus
```

`prepare_evaluation_corpus.py` использует весь доступный пул WAV-файлов validation-дикторов и создаёт 50 записей: по 10 для каждого сценария `normal`, `double_talk`, `high_leak_lt_10db`, `path_drift` и `third_speaker`.

Ожидаемая структура подготовленного корпуса:

```text
evaluation_corpus/
├── normal/record_001/input.wav
├── normal/record_001/ground_truth.json
├── double_talk/record_001/input.wav
└── double_talk/record_001/ground_truth.json
```

В `ground_truth.json` должны быть ключи `A` и `B`, а значения — списки интервалов с полями `start`, `end` и `label`. Оценщик рекурсивно находит все пары и обрабатывает каждую запись.

Основные ограничения и результаты описаны в [REPORT.md](REPORT.md), машинный отчёт оценки находится в [docs/evaluation_results.md](docs/evaluation_results.md).

## Структура

- `src/` — синтез данных, признаки, детектор, клинер и метрики;
- `detect.py`, `clean.py` — публичные CLI;
- `scripts/` — загрузка данных и обучение;
- `evaluate.py`, `tests/` — валидация;
- `models/` — готовые веса;
- `samples/` — примеры и результаты
