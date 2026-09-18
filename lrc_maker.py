import os
import re
import json
import time
import logging
import sys
import shutil
import subprocess
import requests
import queue
import threading
from pathlib import Path
from lrc_formats import build_outputs, format_time, parse_lyrics, repair_lines

logger = logging.getLogger(__name__)

FFMPEG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin")
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROLLER_DIR = os.path.join(BASE_DIR, "temp", "roller")
os.makedirs(ROLLER_DIR, exist_ok=True)

# ============================================================
# py-roller: локальный пайплайн выравнивания (субпроцесс)
# s=деммукс-сплит вокала, f=фильтр, t=транскрипция,
# p=парсинг текста, a=выравнивание, w=запись LRC
# ============================================================

ROLLER_STAGE_RANGES = {
    "run": (30, 32),
    "splitter": (15, 30),
    "filter": (30, 31),
    "transcriber": (30, 65),
    "parser": (65, 72),
    "aligner": (72, 92),
    "writer": (92, 98),
}

_APP_TO_ROLLER_LANG = {
    "rus": "mul", "eng": "en", "ukr": "mul", "kaz": "mul",
    "deu": "mul", "fra": "mul", "spa": "mul", "ita": "mul",
    "pol": "mul", "tur": "mul", "ara": "mul", "por": "mul",
    "bel": "mul", "ron": "mul",
}


def roller_language(language):
    """Маппинг кода языка приложения на язык py-roller (zh/en/mul)."""
    return _APP_TO_ROLLER_LANG.get((language or "rus").strip().lower(), "mul")


def _py_roller_exe():
    exe = shutil.which("py-roller")
    if not exe:
        for cand in (
            os.path.join(os.path.dirname(sys.executable), "Scripts", "py-roller.exe"),
            os.path.join(sys.prefix, "Scripts", "py-roller.exe"),
        ):
            if os.path.exists(cand):
                exe = cand
                break
    if not exe:
        raise RuntimeError('py-roller не найден. Установи: pip install "py-roller>=0.8.3,<0.9"')
    return exe


def _resolve_large_v3_path():
    """Возвращает локальный путь к faster-whisper large-v3 из HF-кэша (без скачивания)."""
    configured = os.environ.get("LRC_WHISPER_MODEL_PATH", "").strip()
    if configured:
        configured = os.path.abspath(os.path.expanduser(configured))
        if os.path.isfile(os.path.join(configured, "model.bin")):
            return configured
        logger.warning("LRC_WHISPER_MODEL_PATH не содержит model.bin: %s", configured)

    snap_root = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "models--Systran--faster-whisper-large-v3", "snapshots",
    )
    if not os.path.isdir(snap_root):
        return None
    try:
        snap = sorted(os.listdir(snap_root))[-1]
        candidate = os.path.join(snap_root, snap)
        if os.path.isfile(os.path.join(candidate, "model.bin")):
            return candidate
    except Exception:
        pass
    return None


def _handle_roller_event(evt, progress_callback):
    if not progress_callback:
        return
    ev_type = evt.get("type")
    stage = evt.get("stage") or ""
    message = (evt.get("message") or "").strip()
    lo, hi = ROLLER_STAGE_RANGES.get(stage, (30, 98))
    if ev_type == "stage_started":
        progress_callback(lo, f"🔄 {message}" if message else "🔄 Working...")
    elif ev_type == "stage_progress":
        pct = evt.get("progress")
        if isinstance(pct, (int, float)):
            progress_callback(int(round(lo + pct * (hi - lo))), message or "⏳ Working...")
    elif ev_type == "stage_completed":
        progress_callback(hi, f"✅ {message}" if message else "✅ Done")
    elif ev_type == "download_progress":
        pct = evt.get("progress")
        if isinstance(pct, (int, float)):
            progress_callback(int(round(5 + pct * 25)), message or "⬇️ Downloading model...")
    elif ev_type == "download_completed":
        progress_callback(30, message or "✅ Model ready")


def _nvidia_lib_paths():
    """Пути к CUDA runtime/cublas из pip-пакетов nvidia-* (если установлены)."""
    paths = []
    try:
        import nvidia.cublas.lib
        import nvidia.cuda_runtime.lib
        import pathlib
        for mod in (nvidia.cublas.lib, nvidia.cuda_runtime.lib):
            if mod.__file__:
                paths.append(str(pathlib.Path(mod.__file__).parent))
    except Exception:
        pass
    return paths


def run_roller_process(cmd, progress_callback=None, timeout=3600):
    """Запуск py-roller как субпроцесса; прогресс берётся из PYROLLER_EVENT JSONL."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    nvidia_paths = _nvidia_lib_paths()
    if nvidia_paths:
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(nvidia_paths) + (":" + current if current else "")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    last_pct = [0]
    def clamped(pct, msg):
        if pct < last_pct[0]:
            pct = last_pct[0]
        last_pct[0] = pct
        if progress_callback:
            progress_callback(pct, msg)

    lines_queue = queue.Queue()
    stream_done = object()

    def read_stdout():
        try:
            for stdout_line in proc.stdout:
                lines_queue.put(stdout_line)
        finally:
            lines_queue.put(stream_done)

    threading.Thread(target=read_stdout, name="py-roller-output", daemon=True).start()
    output_lines = []
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            proc.kill()
            proc.wait()
            raise RuntimeError(f"py-roller превысил таймаут {timeout // 60} мин")
        try:
            line = lines_queue.get(timeout=min(0.5, remaining))
        except queue.Empty:
            if proc.poll() is not None:
                continue
            continue
        if line is stream_done:
            break
        line = line.rstrip("\n")
        if line.startswith("PYROLLER_EVENT "):
            try:
                evt = json.loads(line[len("PYROLLER_EVENT "):])
            except Exception:
                continue
            try:
                _handle_roller_event(evt, clamped)
            except Exception as e:
                logger.warning(f"⚠️ roller progress handler: {e}")
        else:
            output_lines.append(line)
    proc.wait(timeout=10)
    if proc.returncode != 0:
        tail = "\n".join(output_lines[-30:])
        raise RuntimeError(f"py-roller завершился с ошибкой (код {proc.returncode}):\n{tail}")
    return True


def build_roller_command(stages, audio_path, lyrics_path, language, job_dir):
    cmd = [_py_roller_exe(), "run", "--stages", ",".join(stages)]
    if audio_path:
        cmd += ["--audio", audio_path]
    if lyrics_path:
        cmd += ["--lyrics", lyrics_path]
    cmd += ["--language", roller_language(language)]
    model = _resolve_large_v3_path()
    model = model or os.environ.get("LRC_ROLLER_MODEL", "").strip() or "large-v2"
    if model:
        cmd += ["--transcriber-model-name", model]
        logger.info("🧠 Using whisper model: %s", model)
    alignment_path = os.path.join(job_dir, "alignment.json")
    roller_path = os.path.join(job_dir, "out.lrc")
    cmd += [
        "--output-alignment-result", alignment_path,
        "--output-roller", roller_path,
        "--intermediate", os.path.join(job_dir, "intermediate"),
        "--cleanup", "never",
        "--log-level", "WARNING",
        "--progress-format", "jsonl",
        "--output-format", "json",
    ]
    return cmd, alignment_path, roller_path


def write_lyrics_file(lines):
    os.makedirs(ROLLER_DIR, exist_ok=True)
    path = os.path.join(ROLLER_DIR, f"lyrics_{int(time.time() * 1000)}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ============================================================
# Из alignment_result py-roller -> строки с пословными таймингами
# ============================================================

def _norm_word(w):
    return re.sub(r"[\W_]", "", w, flags=re.UNICODE).lower()


def _group_units_by_word(units):
    """Сгруппировать фонемы по source_word: word -> (start, end)."""
    groups = []
    cur = None
    for u in units:
        sw = (u.get("metadata") or {}).get("source_word")
        s = u.get("start_time")
        e = u.get("end_time")
        if s is None or e is None:
            continue
        if cur is None or sw != cur:
            if cur is not None:
                groups.append((cur, s0, e0))
            cur = sw
            s0, e0 = s, e
        else:
            s0 = min(s0, s)
            e0 = max(e0, e)
    if cur is not None:
        groups.append((cur, s0, e0))
    return groups


def _interpolate_line_words(entries, line_start, line_end):
    """Слова без таймингов распределяются между опорными словами и границами строки."""
    n = len(entries)
    if n == 0:
        return entries
    anchor_pos = [-1]
    anchor_times = [line_start]
    for j, e in enumerate(entries):
        if e.get("aligned"):
            anchor_pos.append(j)
            anchor_times.append(e["start"])
    anchor_pos.append(n)
    anchor_times.append(max(line_end, line_start + 0.01))
    for j, e in enumerate(entries):
        if e.get("aligned"):
            continue
        lo = max(i for i, p in enumerate(anchor_pos) if p < j)
        hi = min(i for i, p in enumerate(anchor_pos) if p > j)
        p_lo, p_hi = anchor_pos[lo], anchor_pos[hi]
        t_lo, t_hi = anchor_times[lo], anchor_times[hi]
        span = (p_hi - p_lo) if p_hi > p_lo else 1
        f1 = (j - p_lo) / span
        f2 = (j + 1 - p_lo) / span
        e["start"] = round(t_lo + f1 * (t_hi - t_lo), 3)
        e["end"] = round(t_lo + f2 * (t_hi - t_lo), 3)
        e["aligned"] = True
    return entries


def _words_from_alignment_line(line):
    raw = (line.get("raw_text") or "").strip()
    tokens = raw.split()
    start = line.get("assigned_time")
    if start is None:
        start = line.get("start_time", 0.0)
    end = line.get("end_time")
    groups = _group_units_by_word(line.get("aligned_units") or [])

    entries = []
    gi = 0
    for tok in tokens:
        ntok = _norm_word(tok)
        entry = {"word": tok, "start": None, "end": None, "aligned": False}
        if ntok:
            for k in range(gi, len(groups)):
                sw, s, e = groups[k]
                if _norm_word(sw) == ntok:
                    entry["start"], entry["end"], entry["aligned"] = s, e, True
                    gi = k + 1
                    break
        entries.append(entry)

    line_start = float(start) if start is not None else 0.0
    line_end = float(end) if end is not None else line_start + 4.0
    matched = sum(1 for entry in entries if entry.get("aligned"))
    entries = _interpolate_line_words(entries, line_start, line_end)
    words = [
        {"word": e["word"], "start": round(e["start"], 3), "end": round(e["end"], 3)}
        for e in entries
    ]
    return words, {
        "matched_words": matched,
        "total_words": len(entries),
        "interpolated_words": len(entries) - matched,
    }


def lines_from_alignment(payload):
    """alignment_result py-roller -> [{start, text, words:[{word,start,end}]}]."""
    result = []
    for line in payload.get("lines") or []:
        raw = (line.get("raw_text") or "").strip()
        if not raw:
            continue
        start = line.get("assigned_time")
        if start is None:
            start = line.get("start_time", 0.0)
        words, quality = _words_from_alignment_line(line)
        matched = quality["matched_words"]
        total = quality["total_words"]
        result.append({
            "start": round(float(start), 3),
            "text": raw,
            "words": words,
            "alignment_quality": round(matched / total, 3) if total else 0.0,
            "matched_words": matched,
            "interpolated_words": quality["interpolated_words"],
        })
    return result


def _alignment_quality(lines):
    """Weighted fraction of words matched directly by the aligner."""
    total = sum(len(line.get("words") or []) for line in lines)
    matched = sum(int(line.get("matched_words") or 0) for line in lines)
    return round(matched / total, 3) if total else 0.0


def _sanitize_word_times(words, line_start, line_end):
    """Гарантирует, что у каждого слова есть корректные тайминги (start >= 0, end > start)."""
    n = len(words)
    if n == 0:
        return words
    for w in words:
        s = w.get("start")
        e = w.get("end")
        if not isinstance(s, (int, float)):
            s = line_start
        if not isinstance(e, (int, float)) or e <= s:
            e = s + 0.05
        if s < 0:
            s = 0.0
        if e < 0:
            e = s + 0.05
        w["start"] = round(s, 3)
        w["end"] = round(e, 3)
    for i in range(1, n):
        if words[i]["start"] < words[i - 1]["start"]:
            words[i]["start"] = round(words[i - 1]["start"] + 0.01, 3)
        if words[i]["end"] <= words[i]["start"]:
            words[i]["end"] = round(words[i]["start"] + 0.05, 3)
    return words


def _finalize_lines(lines):
    fixed, _ = repair_lines(lines)
    return fixed


# ============================================================
# Базовые утилиты LRC
# ============================================================

def format_seconds_lrc(seconds):
    return format_time(seconds, precision=2)


def format_lrc_time(seconds):
    return format_time(seconds, precision=2)


def parse_lrc(lrc_text):
    """Compatibility wrapper returning normalized timed lines.

    Unlike the old regex this accepts millisecond timestamps, metadata,
    multiple line tags and existing Enhanced LRC without leaving ``<...>``
    word tags inside the lyric text.
    """
    return parse_lyrics(lrc_text).get("lines", [])


def split_plain_text(lrc_text):
    """Plain-текст в строки: переносы строк как есть, предложения — только если нет переносов."""
    if not lrc_text:
        return []
    lines = [ln.strip() for ln in lrc_text.splitlines() if ln.strip()]
    if len(lines) > 1:
        return lines
    if len(lines) == 1:
        parts = [p.strip() for p in re.split(r'(?<=[.!?…])\s+', lines[0]) if p.strip()]
        return parts if len(parts) > 1 else lines
    return lines


def _detect_instrumental_breaks_from_words(result, min_gap=4.0):
    """Инструментальные проигрыши по реальным позициям слов в выравнивании."""
    breaks = []
    for i in range(len(result) - 1):
        cur = result[i]
        nxt = result[i + 1]
        if not cur.get('words') or not nxt.get('words'):
            continue
        cur_starts = [w['start'] for w in cur['words']]
        cur_ends = [w['end'] for w in cur['words']]
        if len(cur_starts) >= 2 and (max(cur_ends) - min(cur_starts)) < 0.5:
            continue
        v_hi = max(cur_ends)
        v_lo = min(w['start'] for w in nxt['words'])
        gap = v_lo - v_hi
        if gap >= min_gap:
            breaks.append({'start': round(v_hi, 3), 'end': round(v_lo, 3)})
    return breaks


def detect_instrumental_breaks(audio_path, elrc_lines=None, min_gap=4.0):
    """Проигрыши (только для логов, в ELRC не попадают)."""
    breaks = _detect_instrumental_breaks_from_words(elrc_lines or [], min_gap)

    try:
        import librosa
        from faster_whisper.vad import get_speech_timestamps, VadOptions
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        chunks = get_speech_timestamps(
            y,
            VadOptions(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=1000,
                speech_pad_ms=400,
            ),
            sampling_rate=sr,
        )
        if chunks:
            speech = [(int(c['start']) / sr, int(c['end']) / sr) for c in chunks]
            vad_gaps = []
            for i in range(len(speech) - 1):
                gap = speech[i + 1][0] - speech[i][1]
                if gap >= min_gap:
                    vad_gaps.append((round(speech[i][1], 2), round(speech[i + 1][0], 2)))
            if vad_gaps:
                gap_str = ", ".join(
                    f"{format_seconds_lrc(g[0])}-{format_seconds_lrc(g[1])}" for g in vad_gaps
                )
                logger.info(f"🎵 VAD (full audio, только для справки): речевые промежутки -> {gap_str}")
    except Exception as e:
        logger.warning(f"⚠️ VAD (full audio) не сработал: {e}")

    return breaks


# ============================================================
# Поиск текста
# ============================================================

def _fetch_json(url, params=None, headers=None, timeout=10):
    try:
        response = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"⚠️ {url} failed: {e}")
        return None


def _search_lrclib(artist, title, album=None):
    headers = {"User-Agent": "LRCStudio/4.0 (lyrics)"}
    base = {"artist_name": artist, "track_name": title}
    param_sets = [dict(base, album_name=album)] if album else []
    param_sets.append(base)
    for params in param_sets:
        data = _fetch_json("https://lrclib.net/api/search", params=params, headers=headers)
        if not data:
            continue
        for item in data:
            synced = (item.get("syncedLyrics") or "").strip()
            if synced:
                return synced
    return ""


def _search_musixmatch(artist, title):
    api_key = os.environ.get("MUSIXMATCH_API_KEY", "").strip()
    if not api_key:
        logger.info("ℹ️ Musixmatch пропущен: не задан MUSIXMATCH_API_KEY")
        return ""
    data = _fetch_json(
        "https://api.musixmatch.com/ws/1.1/matcher.lyrics.get",
        params={"q_track": title, "q_artist": artist, "apikey": api_key},
    )
    if not data:
        return ""
    if data.get("message", {}).get("header", {}).get("status_code") != 200:
        return ""
    body = data.get("message", {}).get("body", {})
    lyrics = (body.get("lyrics") or {}).get("lyrics_body") or ""
    if not lyrics:
        return ""
    lyrics = lyrics.split("\n\n*******")[0].split("\n\n(Source:")[0].strip()
    return lyrics


def _search_betterlyrics(artist, title):
    data = _fetch_json(
        "https://api.betterlirics.xyz/lyrics",
        params={"track": title, "artist": artist},
        timeout=8,
    )
    if not data:
        return ""
    if isinstance(data, dict):
        lyrics = (
            data.get("lyrics") or data.get("syncedLyrics")
            or data.get("lrc") or data.get("text") or ""
        )
    else:
        lyrics = ""
    return str(lyrics).strip() if lyrics else ""


def _search_netease(artist, title):
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://music.163.com"}
    data = _fetch_json(
        "https://music.163.com/api/search/get",
        params={"s": f"{artist} {title}", "type": 1, "offset": 0, "limit": 10},
        headers=headers,
    )
    if not data:
        return ""
    songs = (data.get("result") or {}).get("songs") or []
    for song in songs:
        song_id = song.get("id")
        if not song_id:
            continue
        lyric_data = _fetch_json(
            "https://music.163.com/api/song/lyric",
            params={"id": song_id, "lv": -1, "kv": -1, "tv": -1},
            headers=headers,
        )
        if not lyric_data:
            continue
        lrc = ((lyric_data.get("lrc") or {}).get("lyric") or "").strip()
        if lrc:
            return lrc
    return ""


def _lyric_providers(artist, title, album=None):
    return [
        ("LRCLIB", _search_lrclib, (artist, title, album)),
        ("BetterLyrics", _search_betterlyrics, (artist, title)),
        ("Musixmatch", _search_musixmatch, (artist, title)),
        ("NetEase", _search_netease, (artist, title)),
    ]


def iter_search_providers(artist, title, album=None):
    """Поочерёдно опрашивает источники текста (для SSE-интерфейса).

    Yields (name, status, lyrics): status = searching|found|empty|error."""
    for name, fn, args in _lyric_providers(artist, title, album):
        yield (name, "searching", "")
        try:
            raw = fn(*args)
        except Exception as e:
            logger.warning(f"⚠️ {name} failed: {e}")
            yield (name, "error", str(e))
            continue
        if raw and raw.strip():
            yield (name, "found", raw)
        else:
            yield (name, "empty", "")


def search_lrc(artist, title, album=None):
    """Ищет существующий LRC/текст по провайдерам. Возвращает первый найденный
    синхронизированный текст, либо plain-текст как fallback."""
    plain_fallback = ""
    for name, status, raw in iter_search_providers(artist, title, album):
        if status != "found" or not raw:
            continue
        lines = parse_lrc(raw)
        if lines:
            logger.info(f"✅ Synced LRC from {name}: {len(lines)} lines")
            return raw
        if not plain_fallback:
            plain_fallback = raw
    if plain_fallback:
        logger.info("ℹ️ Только текст без таймингов — будет выровнен по аудио")
        return plain_fallback
    return ""


_LANG_MAP = {
    "rus": "ru", "eng": "en", "ukr": "uk", "kaz": "kk",
    "deu": "de", "fra": "fr", "spa": "es", "ita": "it",
    "pol": "pl", "tur": "tr", "ara": "ar", "por": "pt",
    "bel": "be", "ron": "ro",
}


def generate_lrc_with_whisper(audio_path, language=None, progress_callback=None):
    """Создаёт построчный LRC из транскрипции (только когда текст не найден)."""
    try:
        from faster_whisper import WhisperModel
        if progress_callback:
            progress_callback(20, "🎙️ Генерация текста через Whisper...")
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel("large-v3", device=device, compute_type=compute_type)
        whisper_lang = _LANG_MAP.get(language, language) if language else None
        segments_iter, info = model.transcribe(
            audio_path, language=whisper_lang, word_timestamps=True
        )
        lrc_lines = []
        for seg in segments_iter:
            timestamp = seg.start
            minutes = int(timestamp) // 60
            secs = timestamp - 60 * minutes
            lrc_lines.append(f"[{minutes:02d}:{secs:05.2f}] {seg.text}")
        del model
        _free_gpu_memory()
        return "\n".join(lrc_lines)
    except Exception as e:
        logger.error(f"Whisper LRC generation failed: {e}")
    try:
        import whisper
        if progress_callback:
            progress_callback(20, "🎙️ Генерация текста через Whisper (fallback)...")
        model = whisper.load_model("large-v3")
        result = model.transcribe(audio_path, language=_LANG_MAP.get(language, language) if language else None)
        lrc_lines = []
        for seg in result['segments']:
            timestamp = seg['start']
            minutes = int(timestamp) // 60
            secs = timestamp - 60 * minutes
            lrc_lines.append(f"[{minutes:02d}:{secs:05.2f}] {seg['text']}")
        del model
        _free_gpu_memory()
        return "\n".join(lrc_lines)
    except Exception as e:
        logger.error(f"Fallback Whisper LRC failed: {e}")
    return ""


def _free_gpu_memory():
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ============================================================
# Главный пайплайн
# ============================================================

def _extract_metadata(audio_path):
    """Теги: mutagen (mp3) -> ffprobe -> дефолты."""
    tags = {}
    try:
        from mutagen.mp3 import MP3
        from mutagen.easyid3 import EasyID3
        audio = MP3(audio_path, ID3=EasyID3)
        if audio:
            for key in ("artist", "title", "album"):
                vals = audio.get(key)
                if vals:
                    tags[key] = vals[0] if isinstance(vals, list) else vals
        if tags:
            return tags
    except Exception as e:
        logger.warning(f"⚠️ mutagen metadata failed: {e}")
    try:
        bundled_ffprobe = os.path.join(FFMPEG_DIR, "ffprobe.exe")
        ffprobe = bundled_ffprobe if os.path.exists(bundled_ffprobe) else shutil.which("ffprobe")
        if not ffprobe:
            return tags
        result = subprocess.run(
            [ffprobe, "-v", "error",
             "-show_entries", "format_tags", "-of", "json", audio_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=20,
        )
        data = json.loads(result.stdout or "{}")
        ftags = (data.get("format") or {}).get("tags") or {}
        for src, dst in (("artist", "artist"), ("ARTIST", "artist"),
                         ("title", "title"), ("TITLE", "title"),
                         ("album", "album"), ("ALBUM", "album")):
            if dst not in tags and ftags.get(src):
                tags[dst] = ftags[src]
    except Exception as e:
        logger.warning(f"⚠️ ffprobe metadata failed: {e}")
    return tags


def generate_elrc(audio_path, artist=None, title=None, album=None, progress_callback=None, lrc_text=None, use_demucs=True, language="rus"):
    if progress_callback is None:
        progress_callback = lambda *a: None

    lyrics_path = None
    job_dir = None
    try:
        progress_callback(5, "🔍 Starting pipeline...")

        if not artist or not title or not album:
            progress_callback(8, "📋 Extracting metadata...")
            meta = _extract_metadata(str(audio_path))
            artist = artist or meta.get("artist") or "Unknown Artist"
            title = title or meta.get("title") or "Unknown Title"
            album = album or meta.get("album") or ""

        progress_callback(10, f"🎤 {artist} — {title}")

        # ===== определяем текст =====
        existing_lrc_lines = []
        if lrc_text:
            parsed_input = parse_lyrics(lrc_text)
            existing_lrc_lines = parsed_input["lines"]
            if existing_lrc_lines:
                logger.info(f"✅ Using provided LRC: {len(existing_lrc_lines)} lines")
            else:
                plain_lines = parsed_input["plain_lines"] or split_plain_text(lrc_text)
                if plain_lines:
                    existing_lrc_lines = [{'start': None, 'text': t} for t in plain_lines]
                    logger.info(f"✅ Using provided text without timestamps: {len(existing_lrc_lines)} lines")

        if not existing_lrc_lines:
            progress_callback(15, "🔎 Searching LRCLIB, BetterLyrics, Musixmatch, NetEase...")
            found_lrc = search_lrc(artist, title, album)
            if found_lrc:
                parsed_found = parse_lyrics(found_lrc)
                existing_lrc_lines = parsed_found["lines"]
                if existing_lrc_lines:
                    logger.info(f"✅ Found synced LRC: {len(existing_lrc_lines)} lines")
                else:
                    plain_lines = parsed_found["plain_lines"] or split_plain_text(found_lrc)
                    if plain_lines:
                        existing_lrc_lines = [{'start': None, 'text': t} for t in plain_lines]
                        logger.info(f"✅ Found plain lyrics text: {len(existing_lrc_lines)} lines")

        if not existing_lrc_lines:
            raw_lrc = generate_lrc_with_whisper(audio_path, language, progress_callback)
            if raw_lrc:
                existing_lrc_lines = parse_lrc(raw_lrc)
                logger.info(f"✅ Generated LRC via Whisper: {len(existing_lrc_lines)} lines")

        if not existing_lrc_lines:
            progress_callback(100, "❌ No lyrics found")
            return {
                'success': False,
                'error': 'No lyrics found for this track',
                'lines': []
            }

        # ===== py-roller: транскрипция + выравнивание =====
        lyrics_path = write_lyrics_file([l['text'] for l in existing_lrc_lines])

        job_dir = os.path.join(ROLLER_DIR, f"job_{int(time.time() * 1000)}")
        os.makedirs(job_dir, exist_ok=True)
        stages = ['s', 'f', 't', 'p', 'a', 'w'] if use_demucs else ['t', 'p', 'a', 'w']
        cmd, alignment_path, roller_path = build_roller_command(
            stages, str(audio_path), lyrics_path, language, job_dir
        )
        logger.info("🚀 py-roller: " + " ".join(cmd))

        progress_callback(12, f"🚀 py-roller: {'деммукс-сплит + ' if use_demucs else ''}транскрипция + выравнивание...")
        run_roller_process(cmd, progress_callback)

        if not os.path.exists(alignment_path):
            raise RuntimeError("py-roller не создал alignment.json")

        with open(alignment_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        payload = data.get('payload') or data
        elrc_lines = lines_from_alignment(payload)

        if not elrc_lines:
            raise RuntimeError("py-roller не вернул ни одной выровненной строки")

        elrc_lines = _finalize_lines(elrc_lines)

        # ===== проигрыши (только для логов) =====
        breaks = []
        try:
            breaks = detect_instrumental_breaks(audio_path, elrc_lines)
            if breaks:
                log_times = ", ".join(
                    f"{format_seconds_lrc(b['start'])}-{format_seconds_lrc(b['end'])}"
                    for b in breaks
                )
                logger.info(f"🎶 Instrumental breaks (проигрыши, только для логов): {log_times}")
        except Exception as e:
            logger.warning(f"⚠️ Instrumental break detection failed: {e}")

        metadata = {
            "ar": artist,
            "ti": title,
            "al": album,
            "by": "LRC Studio",
            "re": "LRC Studio",
            "ve": "5.0",
        }
        outputs = build_outputs(elrc_lines, metadata)
        elrc_lines = outputs["lines"]

        progress_callback(100, f"✅ Done: {len(elrc_lines)} lines")

        return {
            'success': True,
            'artist': artist,
            'title': title,
            'album': album,
            'lrc': outputs["lrc"],
            'elrc': outputs["elrc"],
            'elrc_compatible': outputs["elrc_compatible"],
            'lines': elrc_lines,
            'line_count': len(elrc_lines),
            'word_count': sum(len(l.get('words', [])) for l in elrc_lines),
            'alignment_quality': _alignment_quality(elrc_lines),
            'breaks': breaks,
            'validation_issues': outputs["issues"],
        }

    except Exception as e:
        logger.error(f"❌ generate_elrc failed: {e}", exc_info=True)
        progress_callback(100, f"❌ Error: {e}")
        return {
            'success': False,
            'error': str(e),
            'lines': []
        }
    finally:
        if lyrics_path:
            try:
                os.remove(lyrics_path)
            except OSError:
                pass
        keep_artifacts = os.environ.get("LRC_KEEP_ARTIFACTS", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if job_dir and not keep_artifacts:
            shutil.rmtree(job_dir, ignore_errors=True)
