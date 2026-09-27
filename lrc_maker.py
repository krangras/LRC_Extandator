"""LRC Studio / Extandator V7 core.

V7 is intentionally a *forced-alignment* tool, not an ASR tool.  It preserves
the last working provider/UI/cache surface, but the heavy runtime accepts
known timed LRC text and only estimates exact word boundaries.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import unicodedata
from typing import Any, Iterable

import requests

from alignment_engine import ALIGNMENT_ENGINE_VERSION, AlignmentEngine, AlignmentOptions
from alignment_quality import quality_report
from lrc_formats import build_outputs, format_time, parse_lyrics, repair_lines

logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG_DIR = os.path.join(BASE_DIR, "ffmpeg", "bin")
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

_ENGINE: AlignmentEngine | None = None


def _engine() -> AlignmentEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = AlignmentEngine()
    return _ENGINE


def format_seconds_lrc(seconds: float) -> str:
    return format_time(seconds, precision=2)


def format_lrc_time(seconds: float) -> str:
    return format_time(seconds, precision=2)


def parse_lrc(lrc_text: str) -> list[dict[str, Any]]:
    return parse_lyrics(lrc_text).get("lines", [])


def split_plain_text(text: str) -> list[str]:
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    if len(lines) == 1:
        parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", lines[0]) if p.strip()]
        return parts if len(parts) > 1 else lines
    return []


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _extract_metadata(audio_path: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    try:
        from mutagen import File as MutagenFile

        audio = MutagenFile(audio_path, easy=True)
        if audio and getattr(audio, "tags", None):
            for key in ("artist", "title", "album"):
                value = audio.tags.get(key)
                if value:
                    tags[key] = value[0] if isinstance(value, (list, tuple)) else str(value)
        if tags.get("artist") or tags.get("title"):
            return tags
    except Exception as exc:
        logger.debug("mutagen metadata failed: %s", exc)

    try:
        bundled = os.path.join(FFMPEG_DIR, "ffprobe.exe")
        ffprobe = bundled if os.path.isfile(bundled) else shutil.which("ffprobe")
        if not ffprobe:
            return tags
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format_tags", "-of", "json", audio_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        data = json.loads(proc.stdout or "{}")
        raw = (data.get("format") or {}).get("tags") or {}
        lowered = {str(k).lower(): str(v) for k, v in raw.items()}
        for key in ("artist", "title", "album"):
            if key not in tags and lowered.get(key):
                tags[key] = lowered[key]
    except Exception as exc:
        logger.debug("ffprobe metadata failed: %s", exc)
    return tags


# ---------------------------------------------------------------------------
# Lyrics provider search
# ---------------------------------------------------------------------------

def _normalize_search_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("’", "'").replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _deobfuscate_title(value: str) -> str:
    """Undo common title censorship without global leetspeak corruption."""
    replacements = {
        r"\bsh[!\*#]+t\b": "shit",
        r"\bf[!\*#]+ck\b": "fuck",
        r"\bf[!\*#]+k\b": "fuck",
        r"\bb[!\*#]+tch\b": "bitch",
        r"\ba\$\$\b": "ass",
        r"\bd[@4]mn\b": "damn",
    }
    out = value
    for pattern, repl in replacements.items():
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def _strip_video_suffixes(value: str) -> str:
    patterns = [
        r"\s*[\[(](?:official\s+)?(?:music\s+)?video(?:\s+clip)?[\])]\s*$",
        r"\s*[\[(](?:official\s+)?audio[\])]\s*$",
        r"\s*[\[(]lyrics?(?:\s+video)?[\])]\s*$",
        r"\s*[\[(]visuali[sz]er[\])]\s*$",
        r"\s*[\[(](?:4k|hd|hq|remaster(?:ed)?(?:\s+\d{4})?)[\])]\s*$",
    ]
    out = value
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            new = re.sub(pattern, "", out, flags=re.IGNORECASE).strip()
            if new != out:
                out = new
                changed = True
    return out


def title_variants(title: str) -> list[str]:
    base = _normalize_search_text(title)
    cleaned = _strip_video_suffixes(base)
    uncensored = _deobfuscate_title(cleaned)
    result: list[str] = []
    for item in (base, cleaned, uncensored):
        item = item.strip(" -–—|•")
        if item and item.casefold() not in {x.casefold() for x in result}:
            result.append(item)
    return result


def _fetch_json(url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, timeout: int = 10) -> Any:
    try:
        response = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.info("provider request failed %s: %s", url, exc)
        return None


def _similarity(a: str, b: str) -> float:
    na = re.sub(r"[^\w]+", " ", _deobfuscate_title(_normalize_search_text(a)).casefold()).strip()
    nb = re.sub(r"[^\w]+", " ", _deobfuscate_title(_normalize_search_text(b)).casefold()).strip()
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _search_lrclib(artist: str, title: str, album: str | None = None) -> str:
    headers = {"Lrclib-Client": "LRC-Extandator/6.0 (https://github.com/krangras/LRC_Extandator)"}
    candidates: list[tuple[float, str]] = []
    for variant in title_variants(title):
        params: dict[str, Any] = {"artist_name": artist, "track_name": variant}
        if album:
            params["album_name"] = album
        data = _fetch_json("https://lrclib.net/api/search", params=params, headers=headers, timeout=12)
        if not isinstance(data, list):
            continue
        for item in data:
            synced = str(item.get("syncedLyrics") or "").strip()
            plain = str(item.get("plainLyrics") or "").strip()
            raw = synced or plain
            if not raw:
                continue
            title_score = _similarity(variant, str(item.get("trackName") or ""))
            artist_score = _similarity(artist, str(item.get("artistName") or "")) if artist else 0.75
            album_score = _similarity(album or "", str(item.get("albumName") or "")) if album else 0.75
            sync_bonus = 0.10 if synced else 0.0
            score = 0.58 * title_score + 0.30 * artist_score + 0.12 * album_score + sync_bonus
            candidates.append((score, raw))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1] if candidates[0][0] >= 0.52 else ""


def _search_musixmatch(artist: str, title: str) -> str:
    api_key = os.environ.get("MUSIXMATCH_API_KEY", "").strip()
    if not api_key:
        return ""
    for variant in title_variants(title):
        data = _fetch_json(
            "https://api.musixmatch.com/ws/1.1/matcher.lyrics.get",
            params={"q_track": variant, "q_artist": artist, "apikey": api_key},
            timeout=10,
        )
        if not data or data.get("message", {}).get("header", {}).get("status_code") != 200:
            continue
        lyrics = ((data.get("message", {}).get("body", {}).get("lyrics") or {}).get("lyrics_body") or "").strip()
        if lyrics:
            return lyrics.split("\n\n*******", 1)[0].split("\n\n(Source:", 1)[0].strip()
    return ""


def _search_betterlyrics_legacy(artist: str, title: str) -> str:
    """Legacy public endpoint kept as a best-effort provider.

    The official Better Lyrics extension now uses an authenticated unified API;
    this standalone tool intentionally does not scrape credentials from it.
    """
    for variant in title_variants(title):
        data = _fetch_json(
            "https://api.betterlirics.xyz/lyrics",
            params={"track": variant, "artist": artist},
            timeout=8,
        )
        if isinstance(data, dict):
            lyrics = data.get("lyrics") or data.get("syncedLyrics") or data.get("lrc") or data.get("text")
            if lyrics:
                return str(lyrics).strip()
    return ""


def _search_netease(artist: str, title: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://music.163.com"}
    for variant in title_variants(title):
        data = _fetch_json(
            "https://music.163.com/api/search/get",
            params={"s": f"{artist} {variant}".strip(), "type": 1, "offset": 0, "limit": 10},
            headers=headers,
            timeout=10,
        )
        songs = ((data or {}).get("result") or {}).get("songs") or []
        for song in songs:
            if artist and _similarity(artist, " / ".join(a.get("name", "") for a in song.get("artists", []))) < 0.38:
                continue
            song_id = song.get("id")
            if not song_id:
                continue
            lyric_data = _fetch_json(
                "https://music.163.com/api/song/lyric",
                params={"id": song_id, "lv": -1, "kv": -1, "tv": -1},
                headers=headers,
                timeout=10,
            )
            lrc = (((lyric_data or {}).get("lrc") or {}).get("lyric") or "").strip()
            if lrc:
                return lrc
    return ""


def _lyric_providers(artist: str, title: str, album: str | None = None):
    return [
        ("LRCLIB", _search_lrclib, (artist, title, album)),
        ("NetEase", _search_netease, (artist, title)),
        ("Musixmatch", _search_musixmatch, (artist, title)),
        ("BetterLyrics", _search_betterlyrics_legacy, (artist, title)),
    ]


def iter_search_providers(artist: str, title: str, album: str | None = None):
    """Search providers concurrently while keeping the old SSE tuple contract."""
    providers = _lyric_providers(artist, title, album)
    for name, _fn, _args in providers:
        yield name, "searching", ""

    with ThreadPoolExecutor(max_workers=min(4, len(providers))) as pool:
        future_map = {pool.submit(fn, *args): name for name, fn, args in providers}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                raw = future.result()
            except Exception as exc:
                logger.warning("%s failed: %s", name, exc)
                yield name, "error", str(exc)
                continue
            if raw and str(raw).strip():
                yield name, "found", str(raw)
            else:
                yield name, "empty", ""


def search_lrc(artist: str, title: str, album: str | None = None) -> str:
    # Completion order is nondeterministic because providers run concurrently.
    # Rank by timing richness first, then by source reliability.
    provider_rank = {"LRCLIB": 4, "NetEase": 3, "Musixmatch": 2, "BetterLyrics": 1}
    candidates: list[tuple[int, int, str]] = []
    for name, status, raw in iter_search_providers(artist, title, album):
        if status != "found" or not raw:
            continue
        parsed = parse_lyrics(raw)
        lines = parsed.get("lines") or []
        if lines:
            word_count = sum(len(line.get("words") or []) for line in lines)
            richness = 3 if word_count else 2
        else:
            richness = 1
        candidates.append((richness, provider_rank.get(name, 0), raw))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


# ---------------------------------------------------------------------------
# Final forced-alignment orchestration (no ASR fallback)
# ---------------------------------------------------------------------------

def _detect_instrumental_breaks_from_words(lines: list[dict[str, Any]], min_gap: float = 4.0) -> list[dict[str, float]]:
    breaks: list[dict[str, float]] = []
    for current, nxt in zip(lines, lines[1:]):
        cw, nw = current.get("words") or [], nxt.get("words") or []
        if not cw or not nw:
            continue
        current_ends = [float(w["end"]) for w in cw if isinstance(w.get("end"), (int, float))]
        next_starts = [float(w["start"]) for w in nw if isinstance(w.get("start"), (int, float))]
        if not current_ends or not next_starts:
            continue
        current_end = max(current_ends)
        next_start = min(next_starts)
        if next_start - current_end >= min_gap:
            breaks.append({"start": round(current_end, 3), "end": round(next_start, 3)})
    return breaks


def _payload_from_lines(
    *,
    lines: list[dict[str, Any]],
    artist: str,
    title: str,
    album: str,
    quality: dict[str, Any],
    alignment_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "ar": artist,
        "ti": title,
        "al": album,
        "by": "LRC Extandator",
        "re": "LRC Extandator Forced Alignment",
        "ve": "7.0",
    }
    outputs = build_outputs(lines, metadata)
    breaks = _detect_instrumental_breaks_from_words(outputs["lines"])
    return {
        "success": True,
        "artist": artist,
        "title": title,
        "album": album,
        "lrc": outputs["lrc"],
        "elrc": outputs["elrc"],
        "elrc_compatible": outputs["elrc_compatible"],
        "lines": outputs["lines"],
        "line_count": len(outputs["lines"]),
        "word_count": sum(len(line.get("words") or []) for line in outputs["lines"]),
        "breaks": breaks,
        "validation_issues": outputs["issues"],
        "quality": quality,
        "alignment": alignment_meta or {},
        "engine_version": ALIGNMENT_ENGINE_VERSION,
    }


def _has_line_anchors(lines: list[dict[str, Any]]) -> bool:
    if not lines:
        return False
    return all(
        isinstance(line.get("anchor_start"), (int, float)) or isinstance(line.get("start"), (int, float))
        for line in lines
    )


def generate_elrc(
    audio_path,
    artist=None,
    title=None,
    album=None,
    progress_callback=None,
    lrc_text=None,
    use_demucs="auto",
    language="rus",
    *,
    quality_mode="max",
    force_realign=False,
    use_cache=True,
):
    """Convert a known, timed LRC to word-timed ELRC by forced alignment.

    Important V7.2 contract: if no timed LRC can be obtained, the function stops
    with a clear error.  It never guesses lyrics with an ASR model.
    """
    progress_callback = progress_callback or (lambda *_args: None)
    try:
        audio_path = str(Path(audio_path).resolve())
        progress_callback(2, "Forced Alignment v7.2: проверяю аудио и LRC…")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(audio_path)

        meta = _extract_metadata(audio_path) if not artist or not title or not album else {}
        artist = str(artist or meta.get("artist") or "Unknown Artist").strip()
        title = str(title or meta.get("title") or "Unknown Title").strip()
        album = str(album or meta.get("album") or "").strip()
        progress_callback(5, f"{artist} — {title}")

        parsed_input = parse_lyrics(lrc_text or "") if lrc_text else {"lines": [], "plain_lines": [], "metadata": {}}
        source_lines: list[dict[str, Any]] = list(parsed_input.get("lines") or [])
        supplied_plain = list(parsed_input.get("plain_lines") or [])

        # Existing ELRC already contains word timings; preserve it unless the
        # caller explicitly wants a fresh acoustic forced alignment.
        imported_word_count = sum(len(line.get("words") or []) for line in source_lines)
        if source_lines and imported_word_count and not force_realign:
            fixed, issues = repair_lines(source_lines)
            quality = quality_report(fixed)
            quality["source"] = "imported_elrc"
            progress_callback(100, "Готово: существующий ELRC проверен без повторного выравнивания")
            payload = _payload_from_lines(
                lines=fixed,
                artist=artist,
                title=title,
                album=album,
                quality=quality,
                alignment_meta={"mode": "imported", "cacheHit": False},
            )
            payload["validation_issues"] = list(dict.fromkeys(payload["validation_issues"] + issues))
            return payload

        # If the user did not provide a timed LRC, keep the existing provider
        # Keep the existing provider search; synced lyrics rank above plain.
        if not source_lines:
            progress_callback(8, "Ищу синхронизированный LRC по библиотекам…")
            found = search_lrc(artist, title, album)
            if found:
                parsed = parse_lyrics(found)
                source_lines = list(parsed.get("lines") or [])
                if not source_lines:
                    supplied_plain = list(parsed.get("plain_lines") or split_plain_text(found))

        if not source_lines:
            detail = " Найден только обычный текст без таймкодов." if supplied_plain else ""
            raise ValueError(
                "Forced Alignment v7.2 не распознаёт текст песни с нуля. "
                "Нужен обычный синхронизированный LRC вида [00:42.100] строка." + detail
            )

        source_lines = [line for line in source_lines if str(line.get("text") or "").strip()]
        if not source_lines:
            raise ValueError("LRC не содержит строк для выравнивания")
        if not _has_line_anchors(source_lines):
            raise ValueError(
                "Для forced alignment нужны LRC timestamps у каждой строки. "
                "Plain lyrics без [mm:ss.xxx] намеренно не отправляются в ASR."
            )

        if isinstance(use_demucs, str):
            value = use_demucs.strip().lower()
            demucs_mode: bool | str = "auto" if value in {"auto", "adaptive"} else value in {"1", "true", "yes", "on"}
        else:
            demucs_mode = bool(use_demucs)

        options = AlignmentOptions(
            use_demucs=demucs_mode,
            quality_mode=quality_mode if quality_mode in {"fast", "balanced", "max"} else "max",
            use_cache=bool(use_cache),
            retry_weak_lines=True,
        )
        progress_callback(12, "LRC anchors готовы — запускаю adaptive multi-pass CTC forced alignment…")
        aligned = _engine().align(
            audio_path,
            source_lines,
            language=language,
            options=options,
            progress_callback=lambda pct, msg: progress_callback(12 + round(pct * 0.87), msg),
        )
        quality = dict(aligned["quality"])
        quality["source"] = "forced_alignment"
        payload = _payload_from_lines(
            lines=aligned["lines"],
            artist=artist,
            title=title,
            album=album,
            quality=quality,
            alignment_meta={
                "mode": "local_ctc_forced_alignment",
                "backend": aligned.get("backend"),
                "device": aligned.get("device"),
                "gpuName": aligned.get("gpuName"),
                "runtimeSec": aligned.get("runtimeSec"),
                "cacheHit": aligned.get("cacheHit", False),
                "selectedCandidate": aligned.get("selectedCandidate"),
                "candidates": aligned.get("candidates", []),
            },
        )
        progress_callback(100, f"Готово: {quality.get('grade')} / {quality.get('score', 0):.3f}")
        return payload

    except Exception as exc:
        logger.exception("generate_elrc failed")
        progress_callback(100, f"Ошибка: {exc}")
        return {"success": False, "error": str(exc), "lines": []}
