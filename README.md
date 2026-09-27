# LRC Extandator — Forced Alignment v7.2 Adaptive Rescue

Локальный инструмент для превращения **уже синхронизированного построчного LRC** в пословный ELRC.

V7.2 не распознаёт текст песни заново. Текст известен заранее, а движок решает только задачу **когда именно прозвучало каждое известное слово**.

## Pipeline v7.2

```text
аудио + timed LRC
        ↓
локальный CTC forced alignment каждой строки
        ↓
lookahead 1–3 слов следующей строки
        ↓
boundary-rescue последнего слова
        ↓
поиск подозрительных фраз внутри строки
        ↓
Adaptive Rescue:
  1) tight re-anchor pass
  2) wide-context pass
  3) no-lookahead pass (max)
        ↓
word-level consensus между независимыми проходами
        ↓
только нестабильные слова:
локальный темп + pronunciation mass + energy-onset fallback
        ↓
при тяжёлых участках в max/auto:
Demucs vocals как независимый акустический кандидат
        ↓
сверка границы N → N+1 с обеих сторон
        ↓
отдельный display_start для корректного переключения строки
        ↓
ELRC
```

В основном пути **нет ASR**. Если нет LRC с таймкодом у каждой строки, программа останавливается и просит обычный синхронизированный LRC.

## Главное в v7.2

### 1. Строка больше не обязана переключаться в момент первого подозрительного onset

Word timestamps остаются акустическими, но для строки дополнительно вычисляется `display_start`.

Он учитывает:

- конец последнего слова предыдущей строки;
- lookahead предыдущей строки;
- первое слово следующей строки;
- confidence обоих наблюдений.

Поэтому в ELRC может быть, например:

```text
[00:33.420]<00:33.180>They ...
```

Первое слово акустически началось в `33.180`, но одноактивный lyrics UI переключит строку в `33.420`, когда предыдущая фраза уже закончилась. Word timing при этом не уничтожается.

### 2. Плохая средняя confidence больше не единственный триггер

Строка отправляется в Adaptive Rescue, если обнаружено одно из событий:

- очень слабое отдельное слово;
- несколько слабых слов подряд;
- интерполированное слово;
- физически подозрительный локальный темп;
- проблемная межстрочная граница.

Одна плохая фраза больше не прячется за восемью хорошими словами.

### 3. Несколько проходов действительно разные

Детерминированный CTC нет смысла запускать три раза одинаково. Поэтому rescue использует разные условия:

- **tight re-anchor** — узкое окно от уже найденного начала;
- **wide context** — расширенное окно и больше контекста;
- **no lookahead** — в `max`, чтобы проверить, не тянет ли следующая строка текущую фразу.

После этого выбирается не «последний запуск», а word-level consensus.

### 4. Темп — последний fallback, а не основной aligner

Если слово стабильно найдено CTC, темп его не трогает.

Если несколько проходов расходятся или confidence очень низкая, используются:

- локальный темп соседних хороших строк;
- длина романизованного слова как pronunciation mass;
- уверенные соседние слова как anchors;
- ближайший разумный energy onset в аудио.

Такой результат получает `origin = tempo_energy_rescue` и пониженную confidence — система не делает вид, что это точное акустическое совпадение.

### 5. Demucs теперь особенно нужен именно для тяжёлых участков

В `max + auto`, если пришлось использовать tempo fallback или осталась реально слабая adaptive-фраза, запускается Demucs даже если средний score всей песни высокий.

То есть одна сломанная строка больше не игнорируется только потому, что остальные 39 хорошие.

## Режимы

- **fast** — базовый forced alignment, минимум дополнительных проходов;
- **balanced** — expanded retry + rescue явно плохих фраз;
- **max** — boundary verification, 3 adaptive passes, consensus, tempo fallback и selective Demucs.

Для RTX 4050 рекомендуется:

```text
Качество: max
Demucs: auto
```

## Windows

1. Распакуй архив в отдельную папку.
2. Запусти `install.bat`.
3. Запусти `run.bat`.
4. Откроется `http://127.0.0.1:5000`.

Python: 3.10–3.14. При наличии NVIDIA установщик использует CUDA PyTorch.

### FFmpeg

Рекомендуется `ffmpeg` в PATH. Также поддерживается:

```text
LRC_Extandator_ForcedAlignment_v7_2/
  ffmpeg/
    bin/
      ffmpeg.exe
      ffprobe.exe
```

## Входной текст

Нужен timed LRC:

```text
[00:31.250] I open my eyes and these lies
[00:35.600] They breed and they feed off of me
```

Plain lyrics без timestamps намеренно не запускают распознавание текста.

## Quality diagnostics

V7.2 дополнительно возвращает:

- `adaptiveRescues`;
- `tempoRescuedWords`;
- `lineSwitchReconciliations`;
- `lineSwitchDelays`;
- `crossLineDisagreementMs`;
- `adaptiveRescueReasons`;
- `consensus_spread_ms` на слове;
- `origin` и `confidence`.

В интерфейсе эти значения показываются рядом с обычным alignment score.

## Тесты

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Покрываются CTC Viterbi, repeated tokens, LRC anchors, lookahead/backoff, boundary rescue, adaptive consensus, tempo fallback, display-start reconciliation, ELRC round-trip и запрет ASR fallback.

## Benchmark

```powershell
.\.venv\Scripts\python.exe benchmark.py score reference.elrc candidate.elrc
```

## Основные файлы

```text
alignment_engine.py   — CTC + boundary + adaptive multi-pass + tempo/energy fallback
alignment_quality.py  — quality gate и diagnostics
alignment_cache.py    — отдельный cache v7.2
lrc_maker.py          — orchestration и providers
lrc_formats.py        — parser/export + display_start
app.py                — web UI / SSE / local API
```
