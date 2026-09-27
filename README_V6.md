# LRC Extandator V6 — Alignment Engine rebuild

Это drop-in обновление для `krangras/LRC_Extandator`. Оно не заменяет редактор и интерфейс: существующие `templates/` и основной `static/script.js` продолжают работать. Меняется тяжёлое ядро синхронизации и добавляется слой качества/benchmark/API.

## Что исправлено принципиально

### 1. LRC-таймкоды больше не выбрасываются

Старая версия парсила LRC, а затем передавала `py-roller` только текст строк. В результате `[00:42.100]` исчезал, и повторяющийся припев мог быть найден вообще в другой части трека.

V6 хранит каждый LRC timestamp как `anchor_start` и после общей транскрипции повторно выравнивает строку **только в локальном окне около её LRC-якоря**. Это особенно важно для повторяющихся chorus/verse.

### 2. Используются исходные acoustic timings py-roller

У `py-roller` display-level финализация может растянуть `aligned_units` до начала следующей строки. Для обычного rolling LRC это нормально, но для точного ELRC может растягивать последнее слово через инструментальную паузу.

V6 берёт нетронутые `metadata.unit_matches` и восстанавливает из них реальные start/end/confidence до этого stretching pass.

### 3. Интерполяция больше не маскируется под alignment

Каждое слово имеет provenance:

- `aligned` — найдено акустически;
- `anchor_shifted` — акустический word shape сохранён, строка перенесена на доверенный LRC anchor;
- `interpolated` — модель слово не нашла, время оценено между соседями;
- `repaired_*` — тайминг пришлось структурно исправить;
- `whisper_asr` — fallback, когда официального текста нет.

И отдельно хранится `confidence`.

### 4. Quality gate

Результат получает отчёт:

```json
{
  "score": 0.91,
  "grade": "excellent",
  "publishable": true,
  "alignedWordRatio": 0.97,
  "interpolatedWordRatio": 0.01,
  "meanWordConfidence": 0.89,
  "anchorMaeMs": 148.3
}
```

Это позволяет будущему Better Lyrics автоматически предпочитать Enhanced ELRC только когда он действительно лучше исходного LRC.

### 5. Demucs стал кандидатом, а не религией

В режиме `auto` сначала считается mix. Если качество высокое — дорогое разделение вокала вообще не запускается. Если качество слабое, считается Demucs-кандидат и выбирается лучший по quality score.

В UI V6 добавлены варианты:

- `Авто — только если помогает`;
- `Всегда Demucs`;
- `Без Demucs`.

### 6. Повторяющиеся припевы

Количество повторов определяется автоматически и выбирается `py-roller --aligner-repetition none/few/full`. В `quality_mode=max` при слабом результате может выполняться rescue-pass с `full`.

### 7. Нет двойной Whisper-транскрипции

Если текст вообще не удалось найти, V6 делает `faster-whisper` один раз и использует его word timestamps напрямую. Старая схема сначала делала Whisper-LRC, затем py-roller снова транскрибировал тот же файл.

### 8. Persistent cache

Тяжёлый alignment кэшируется в SQLite по:

- SHA-256 аудио;
- lyrics + LRC anchors;
- языку;
- версии engine;
- версии py-roller;
- параметрам alignment.

Изменился текст/модель/настройка — это автоматически другой cache key.

### 9. Локальный API для будущего Better Lyrics

Добавлены:

- `GET /v1/health`
- `GET /v1/cache`
- `DELETE /v1/cache`
- `POST /v1/align`

API слушает только `127.0.0.1` по умолчанию и защищён локальным token в `.lrc_extandator/api-token.txt`.

## Почему нужен эталонный датасет

Не для обучения — сначала именно для **измерения**.

Например, сегодня кажется, что Demucs делает лучше. Без ground truth это субъективно. С эталонным ELRC можно получить:

```text
                    onset MAE    P95      ≤100 ms    runtime
mix                  84 ms       220 ms   71%         18 s
Demucs               57 ms       141 ms   86%         42 s
new CTC backend      49 ms       118 ms   90%         26 s
```

Тогда видно, что реально улучшает pipeline, а что только красиво звучит в описании.

Benchmark меряет не только модель, а **весь путь**:

`audio → separation → transcription → parser → aligner → anchor correction → repair`.

Именно поэтому один и тот же датасет нужен после каждого серьёзного изменения.

## Как применить

Самый простой вариант — распаковать содержимое patch-архива **поверх корня существующего `LRC_Extandator`** с заменой файлов.

Будут заменены:

- `app.py`
- `lrc_maker.py`
- `lrc_formats.py`
- `requirements.txt`
- `start.ps1`
- `install.bat`
- `launcher.bat`
- `run.bat`
- `setup.bat`
- `tests/test_lrc_formats.py`

Будут добавлены:

- `alignment_engine.py`
- `alignment_quality.py`
- `alignment_cache.py`
- `benchmark.py`
- `requirements-experimental.txt`
- `static/v6-enhancer.js`
- `static/v6-enhancer.css`
- новые unit tests
- `benchmarks/`

После этого:

```text
install.bat
launcher.bat
```

Installer использует Python 3.10–3.12, создаёт `.venv`, ставит `py-roller`, запускает его validated runtime installer/doctor и затем unit tests.

## Benchmark

Создай локально:

```text
benchmarks/
  audio/
  lyrics/
  reference/
  manifest.json
```

`reference/*.elrc` — вручную проверенные точные тайминги.

```powershell
.venv\Scripts\python.exe benchmark.py run benchmarks\manifest.json --out benchmark-results.json
```

Или сравнить два ELRC:

```powershell
.venv\Scripts\python.exe benchmark.py score reference.elrc generated.elrc
```

## Что ещё нельзя честно назвать «идеальным» без датасета

Кодовая архитектура теперь умеет измерять качество и исправляет несколько объективных ошибок старого pipeline. Но нельзя честно сказать, что конкретные thresholds (`0.84`, окна ±2 сек и т.п.) оптимальны для всех жанров, пока они не прогнаны по реальному benchmark-набору.

Следующий правильный этап — собрать 30–50 эталонных треков и **подбирать правила по метрикам, а не на глаз**.
