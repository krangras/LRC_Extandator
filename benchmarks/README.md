# Эталонный датасет Forced Alignment v7.1

Benchmark измеряет весь LRC → ELRC pipeline на фиксированном наборе песен.

Для каждого кейса нужны:

- `audio/...` — исходное аудио;
- `lyrics/...` — **синхронизированный построчный LRC**;
- `reference/...` — вручную доведённый ELRC ground truth.

Plain lyrics здесь больше не используются: задача v7.1 — forced alignment известного текста внутри известных LRC-якорей.

Метрики:

- MAE/P50/P95 начала слова;
- MAE/P95 конца слова;
- доля слов в пределах 50/100/200 мс;
- покрытие слов;
- ошибка начала строк;
- runtime.

Дефолтная matrix сравнивает `fast`, `balanced`, `max/auto` и принудительный Demucs.

```powershell
python benchmark.py run benchmarks/manifest.json --out benchmark-results.json
python benchmark.py score reference.elrc generated.elrc
```

Для нормального набора полезно 30–50 песен: pop, rock/metal, scream, rap, растянутый вокал, duet/backing vocals, live, повторы, русский/английский и другие языки.
