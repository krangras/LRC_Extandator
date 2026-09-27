# LRC Extandator — Forced Alignment v7.1 Boundary-Aware

Локальный инструмент для превращения **уже синхронизированного построчного LRC** в пословный ELRC.

Основа v7/v7.1: текст больше не распознаётся заново. Движок считает LRC известной истиной и решает только задачу **когда произнесено каждое известное слово**.

## Новый pipeline

```text
аудио
  +
[00:42.100] Give me a reason to why I'm here
[00:46.300] I've been so far from home
  ↓
локальное окно около 42.100…46.300 + overlap за следующий anchor
  ↓
MMS forced-alignment acoustic model
  ↓
CTC Viterbi: текущая строка + 1–3 служебных слова следующей строки
  ↓
проверка границы + узкий rescue-pass последнего слова
  ↓
word boundaries + confidence + origin
  ↓
ELRC
```

В основном пути **нет ASR**. Если нет LRC с таймкодом у каждой строки, программа останавливается и просит синхронизированный LRC вместо угадывания текста песни.

## Что сохранено от последней рабочей версии

- прежний веб-интерфейс и ручной ELRC-редактор;
- поиск LRC по провайдерам;
- импорт/экспорт LRC и ELRC;
- quality score, provenance и confidence;
- persistent cache;
- Demucs;
- локальный API `/v1/align`;
- benchmark-инструмент.

## Что изменилось

- `Alignment Engine 7.1.0` использует `torchaudio.pipelines.MMS_FA` как acoustic model;
- сам forced-alignment DP реализован внутри проекта (`ctc_viterbi_align`);
- каждая строка анализируется в маленьком окне с overlap за следующий LRC-якорь;
- первые 1–3 слова следующей строки добавляются только как CTC lookahead и затем удаляются из результата;
- если последнее слово прилипло к следующему anchor или имеет слабую confidence, запускается отдельный короткий boundary-rescue;
- в `max` boundary-rescue проверяет каждую межстрочную границу, а не только явно слабые строки;
- если длинный lookahead не помещается, контекст автоматически уменьшается 3 → 2 → 1 → 0 без повторного инференса;
- слабая строка получает максимум один расширенный retry;
- Demucs в режиме `auto` запускается только при низком качестве первого прохода;
- перед Demucs acoustic model выгружается из VRAM, что полезно для видеокарт на 6 GB;
- результат слова хранит `confidence`, `origin` и нормализованную форму;
- SSE отправляет heartbeat, поэтому интерфейс не выглядит зависшим во время тяжёлого этапа;
- старый full-song candidate lattice удалён из основного pipeline.

## Windows / RTX 4050

Самый простой путь:

1. Распакуй архив в отдельную папку.
2. Запусти `install.bat`.
3. После установки запусти `run.bat`.
4. Откроется `http://127.0.0.1:5000`.

Установщик принимает Python 3.10–3.14. При наличии NVIDIA GPU он ставит CUDA-сборку PyTorch; без NVIDIA — CPU-сборку.

На первом `install.bat` модель forced alignment загружается и кэшируется. Это крупный файл, поэтому первый запуск заметно тяжелее последующих.

### FFmpeg

Для MP3/M4A/AAC/OGG рекомендуется `ffmpeg` в `PATH`. Также поддерживается структура:

```text
LRC_Extandator_ForcedAlignment_v7_1/
  ffmpeg/
    bin/
      ffmpeg.exe
      ffprobe.exe
```

## Как подавать текст

Правильно:

```text
[00:31.250] I open my eyes and these lies
[00:35.600] They breed and they feed off of me
```

Недостаточно:

```text
I open my eyes and these lies
They breed and they feed off of me
```

Во втором случае неизвестно даже приблизительное положение строк, а v7.1 намеренно не занимается распознаванием песни с нуля.

## Режимы качества

- **fast** — один проход по исходному mix, минимальные окна;
- **balanced** — более широкие окна и retry слабых строк;
- **max** — самый широкий overlap, 3-word lookahead, проверка каждой границы, rescue-pass и адаптивный Demucs.

Для твоей RTX 4050 разумный дефолт — `max` + `Demucs: auto`.

## Почему Demucs не запускается сразу

Большинству строк достаточно исходного mix. Поэтому v7.1 сначала делает дешёвый CTC alignment. Если confidence/quality нормальные — на этом всё. Если нет — только тогда делается вокальный retry.

Это уменьшает общее время и пиковую VRAM-нагрузку.

## Проверка окружения

```powershell
.\.venv\Scripts\python.exe doctor.py
```

С предварительной загрузкой acoustic model:

```powershell
.\.venv\Scripts\python.exe doctor.py --preload
```

## Тесты

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

В текущей сборке есть тесты CTC Viterbi, повторяющихся символов, lookahead/backoff, boundary guard, last-word rescue, LRC-windowing, романизации известного текста, запрета plain-lyrics fallback, quality, benchmark-метрик и LRC/ELRC round-trip.

## Benchmark

Сравнить готовый ELRC с эталоном:

```powershell
.\.venv\Scripts\python.exe benchmark.py score reference.elrc candidate.elrc
```

Для batch benchmark `lyrics` в manifest теперь обязан быть **timed LRC**.

## Форматы экспорта

- точный ELRC с миллисекундами;
- совместимый ELRC;
- обычный LRC.

Пример:

```text
[00:42.100]<00:42.100>Give <00:42.390>me <00:42.620>a <00:42.780>reason <00:43.410>to <00:43.620>why <00:43.910>I'm <00:44.250>here
```

## Ключевые файлы

```text
alignment_engine.py   — MMS + lookahead CTC Viterbi + boundary rescue + Demucs retry
alignment_quality.py  — confidence/quality gate
alignment_cache.py    — persistent alignment cache
lrc_maker.py          — orchestration и LRC providers
lrc_formats.py        — LRC/ELRC parser/export
app.py                — web UI + local API + SSE
 doctor.py             — диагностика runtime
```
