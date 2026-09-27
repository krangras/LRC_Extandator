"""Forced-alignment engine for LRC -> ELRC.

V7 deliberately does *not* perform speech recognition.  The text is treated as
known ground truth and the acoustic model is asked only one question: where in
a small audio window did the known characters occur?

Pipeline:
    timed LRC anchors
        -> per-line local audio window
        -> multilingual MMS forced-alignment acoustic model
        -> custom CTC Viterbi path
        -> character spans
        -> word spans
        -> optional one-shot expanded-window retry
        -> optional Demucs retry only for weak lines

There is no ASR or external legacy-aligner fallback in this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from typing import Any, Callable, Iterable, Literal

import numpy as np

from alignment_cache import AlignmentCache
from alignment_quality import compare_quality, quality_report
from lrc_formats import repair_lines

ALIGNMENT_ENGINE_VERSION = "7.0.0"
BACKEND_NAME = "MMS_FA/custom-CTC-viterbi"
ProgressCallback = Callable[[int, str], None]

# Application/UI language codes -> ISO-639-3 codes understood by uroman.
_LANGUAGE_TO_ISO3 = {
    "rus": "rus", "ru": "rus",
    "eng": "eng", "en": "eng",
    "ukr": "ukr", "uk": "ukr",
    "kaz": "kaz", "kk": "kaz",
    "deu": "deu", "ger": "deu", "de": "deu",
    "fra": "fra", "fre": "fra", "fr": "fra",
    "spa": "spa", "es": "spa",
    "ita": "ita", "it": "ita",
    "pol": "pol", "pl": "pol",
    "tur": "tur", "tr": "tur",
    "ara": "ara", "ar": "ara",
    "por": "por", "pt": "por",
    "bel": "bel", "be": "bel",
    "ron": "ron", "rum": "ron", "ro": "ron",
    "jpn": "jpn", "ja": "jpn",
    "kor": "kor", "ko": "kor",
    "zho": "zho", "chi": "zho", "zh": "zho",
}

_COMMON_CENSORED = {
    "sh!t": "shit", "sh*t": "shit", "s#!t": "shit",
    "f*ck": "fuck", "f**k": "fuck", "f#ck": "fuck",
    "b!tch": "bitch", "b*tch": "bitch",
    "a$$": "ass", "d@mn": "damn", "h3ll": "hell",
}


@dataclass(frozen=True, slots=True)
class AlignmentProfile:
    name: str
    pre_margin: float
    post_margin: float
    last_line_window: float
    retry_pre_margin: float
    retry_post_margin: float
    weak_line_threshold: float
    anchor_retry_delta: float
    demucs_trigger_score: float
    demucs_trigger_weak_ratio: float


PROFILES: dict[str, AlignmentProfile] = {
    "fast": AlignmentProfile(
        name="fast",
        pre_margin=0.65,
        post_margin=0.20,
        last_line_window=9.0,
        retry_pre_margin=0.65,
        retry_post_margin=0.20,
        weak_line_threshold=0.42,
        anchor_retry_delta=1.10,
        demucs_trigger_score=0.0,
        demucs_trigger_weak_ratio=2.0,
    ),
    "balanced": AlignmentProfile(
        name="balanced",
        pre_margin=0.95,
        post_margin=0.35,
        last_line_window=11.0,
        retry_pre_margin=1.65,
        retry_post_margin=0.65,
        weak_line_threshold=0.50,
        anchor_retry_delta=0.95,
        demucs_trigger_score=0.78,
        demucs_trigger_weak_ratio=0.22,
    ),
    "max": AlignmentProfile(
        name="max",
        pre_margin=1.25,
        post_margin=0.50,
        last_line_window=13.0,
        retry_pre_margin=2.20,
        retry_post_margin=0.90,
        weak_line_threshold=0.56,
        anchor_retry_delta=0.80,
        demucs_trigger_score=0.84,
        demucs_trigger_weak_ratio=0.14,
    ),
}


@dataclass(slots=True)
class AlignmentOptions:
    use_demucs: bool | Literal["auto"] = "auto"
    quality_mode: Literal["fast", "balanced", "max"] = "max"
    use_cache: bool = True
    retry_weak_lines: bool = True
    demucs_weak_lines_only: bool = True
    demucs_timeout_seconds: int = 1200


@dataclass(frozen=True, slots=True)
class AccelerationInfo:
    device: str
    gpu_name: str | None
    vram_gb: float | None
    torch_version: str | None
    torchaudio_version: str | None


@dataclass(frozen=True, slots=True)
class NormalizedWord:
    display: str
    chars: str
    token_start: int | None
    token_end: int | None


_RUNTIME_LOCK = threading.RLock()
_INFERENCE_LOCK = threading.RLock()
_ROMANIZER_LOCK = threading.RLock()
_MODEL: Any | None = None
_BUNDLE: Any | None = None
_TOKEN_DICT: dict[str, int] | None = None
_DEVICE = "cpu"
_ROMANIZER: Any | None = None


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def get_profile(name: str | None) -> AlignmentProfile:
    return PROFILES.get(str(name or "max").strip().lower(), PROFILES["max"])


def detect_acceleration() -> AccelerationInfo:
    try:
        import torch
        import torchaudio

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return AccelerationInfo(
                device="cuda",
                gpu_name=torch.cuda.get_device_name(0),
                vram_gb=round(float(props.total_memory) / (1024**3), 2),
                torch_version=getattr(torch, "__version__", None),
                torchaudio_version=getattr(torchaudio, "__version__", None),
            )
        return AccelerationInfo(
            device="cpu",
            gpu_name=None,
            vram_gb=None,
            torch_version=getattr(torch, "__version__", None),
            torchaudio_version=getattr(torchaudio, "__version__", None),
        )
    except Exception:
        return AccelerationInfo("unavailable", None, None, None, None)


def runtime_info() -> dict[str, Any]:
    accel = detect_acceleration()
    return {
        "engineVersion": ALIGNMENT_ENGINE_VERSION,
        "backend": BACKEND_NAME,
        "device": accel.device,
        "gpuName": accel.gpu_name,
        "vramGb": accel.vram_gb,
        "torchVersion": accel.torch_version,
        "torchaudioVersion": accel.torchaudio_version,
        "uromanVersion": _package_version("uroman"),
        "modelLoaded": _MODEL is not None,
    }


def _load_runtime(progress_callback: ProgressCallback | None = None) -> tuple[Any, dict[str, int], str]:
    """Load the MMS forced-alignment acoustic model lazily.

    torchaudio supplies only the pretrained acoustic network and its alphabet;
    the actual alignment DP is implemented below and does not rely on
    torchaudio.functional.forced_align.
    """
    global _MODEL, _BUNDLE, _TOKEN_DICT, _DEVICE
    with _RUNTIME_LOCK:
        if _MODEL is not None and _TOKEN_DICT is not None:
            return _MODEL, _TOKEN_DICT, _DEVICE

        try:
            import torch
            import torchaudio
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "Не установлены PyTorch/torchaudio. Запусти install.bat."
            ) from exc

        if not hasattr(torchaudio.pipelines, "MMS_FA"):
            raise RuntimeError(
                "В этой версии torchaudio нет MMS_FA. Переустанови окружение через install.bat."
            )

        if progress_callback:
            progress_callback(7, "Загружаю MMS forced-alignment model (на первом запуске ~1.2 GB)…")

        bundle = torchaudio.pipelines.MMS_FA
        model = bundle.get_model(with_star=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        if device == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        token_dict = dict(bundle.get_dict(star=None))
        if token_dict.get("-") != 0:
            raise RuntimeError("Неожиданный MMS_FA словарь: blank token должен иметь id=0")

        _BUNDLE = bundle
        _MODEL = model
        _TOKEN_DICT = token_dict
        _DEVICE = device
        return model, token_dict, device


def release_acoustic_model() -> None:
    global _MODEL, _BUNDLE, _TOKEN_DICT
    with _RUNTIME_LOCK:
        _MODEL = None
        _BUNDLE = None
        _TOKEN_DICT = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _get_romanizer() -> Any:
    global _ROMANIZER
    with _ROMANIZER_LOCK:
        if _ROMANIZER is not None:
            return _ROMANIZER
        try:
            import uroman as ur
        except Exception as exc:
            raise RuntimeError("Не установлен uroman. Запусти install.bat.") from exc
        _ROMANIZER = ur.Uroman()
        return _ROMANIZER


def _iso3(language: str | None) -> str | None:
    value = str(language or "").strip().lower()
    return _LANGUAGE_TO_ISO3.get(value, value if len(value) == 3 else None)


def _deobfuscate_word(value: str) -> str:
    lowered = unicodedata.normalize("NFKC", value).casefold()
    if lowered in _COMMON_CENSORED:
        return _COMMON_CENSORED[lowered]
    if any(ch.isalpha() for ch in lowered) and any(ch in "!@$#*" for ch in lowered):
        lowered = lowered.replace("@", "a").replace("$", "s").replace("!", "i")
        lowered = lowered.replace("#", "").replace("*", "")
    return lowered


def _clean_romanized(value: str) -> str:
    value = value.replace("’", "'").replace("`", "'").casefold()
    # uroman is normally ASCII-ish already, but NFKD makes accents safe for the
    # 28-character MMS alignment alphabet.
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return "".join(ch for ch in value if ("a" <= ch <= "z") or ch == "'")


def romanize_word(word: str, language: str | None = None, romanizer: Any | None = None) -> str:
    raw = _deobfuscate_word(str(word or "").strip())
    if not raw:
        return ""
    romanizer = romanizer or _get_romanizer()
    lcode = _iso3(language)
    try:
        value = romanizer.romanize_string(raw, lcode=lcode) if lcode else romanizer.romanize_string(raw)
    except TypeError:  # compatibility with older uroman wrappers
        value = romanizer.romanize_string(raw)
    return _clean_romanized(str(value))


def normalize_words(text: str, language: str | None = None, romanizer: Any | None = None) -> tuple[list[NormalizedWord], str]:
    display_words = [part for part in re.findall(r"\S+", str(text or "")) if part.strip()]
    romanizer = romanizer or _get_romanizer()
    normalized: list[NormalizedWord] = []
    target_parts: list[str] = []
    cursor = 0
    for display in display_words:
        chars = romanize_word(display, language, romanizer)
        if chars:
            start = cursor
            cursor += len(chars)
            normalized.append(NormalizedWord(display, chars, start, cursor))
            target_parts.append(chars)
        else:
            normalized.append(NormalizedWord(display, "", None, None))
    return normalized, "".join(target_parts)


def _ffmpeg_executable() -> str | None:
    root = Path(__file__).resolve().parent
    candidates = [
        root / "ffmpeg" / "bin" / "ffmpeg.exe",
        root / "ffmpeg" / "bin" / "ffmpeg",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return shutil.which("ffmpeg")


def decode_audio_16k_mono(audio_path: str, progress_callback: ProgressCallback | None = None) -> np.ndarray:
    """Decode the whole song once to mono float32/16 kHz.

    A 4-minute song is only ~15 MB at this format, so keeping it in RAM is far
    cheaper than repeatedly invoking a decoder for every LRC line.
    """
    ffmpeg = _ffmpeg_executable()
    if ffmpeg:
        if progress_callback:
            progress_callback(3, "Декодирую аудио один раз: mono 16 kHz…")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(audio_path),
            "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=240, creationflags=flags)
        if proc.returncode == 0 and proc.stdout:
            audio = np.frombuffer(proc.stdout, dtype="<f4").copy()
            if audio.size:
                return audio
        err = proc.stderr.decode("utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"ffmpeg не смог декодировать аудио: {err}")

    # Fallback for WAV/FLAC/MP3 builds supported by libsndfile.
    try:
        import soundfile as sf
        import torch
        import torchaudio.functional as AF

        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1, dtype=np.float32)
        if int(sr) != 16000:
            tensor = torch.from_numpy(mono).unsqueeze(0)
            mono = AF.resample(tensor, int(sr), 16000)[0].cpu().numpy()
        return np.asarray(mono, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(
            "Не найден ffmpeg и резервный декодер не смог открыть файл. "
            "Установи ffmpeg или положи его в ffmpeg\\bin."
        ) from exc


def _window_bounds(
    anchor: float,
    next_anchor: float | None,
    audio_duration: float,
    *,
    pre_margin: float,
    post_margin: float,
    last_line_window: float,
) -> tuple[float, float]:
    start = max(0.0, anchor - pre_margin)
    if next_anchor is not None and next_anchor > anchor + 0.05:
        end = min(audio_duration, next_anchor + post_margin)
    else:
        end = min(audio_duration, anchor + last_line_window)
    if end <= start + 0.20:
        end = min(audio_duration, start + 0.20)
    return start, end


def _emission_for_window(
    waveform: np.ndarray,
    start: float,
    end: float,
    *,
    model: Any,
    device: str,
) -> np.ndarray:
    import torch

    s0 = max(0, int(math.floor(start * 16000)))
    s1 = min(len(waveform), int(math.ceil(end * 16000)))
    if s1 - s0 < 1600:
        raise RuntimeError("Слишком короткое окно для forced alignment")
    clip = torch.from_numpy(np.ascontiguousarray(waveform[s0:s1])).unsqueeze(0)
    clip = clip.to(device, non_blocking=(device == "cuda"))

    with _INFERENCE_LOCK, torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                emission, _ = model(clip)
        else:
            emission, _ = model(clip)
        log_probs = torch.log_softmax(emission[0].float(), dim=-1).cpu().numpy()
    del clip, emission
    return np.asarray(log_probs, dtype=np.float32)


def ctc_viterbi_align(log_probs: np.ndarray, target_ids: list[int], blank_id: int = 0) -> list[dict[str, Any]]:
    """Align a known CTC token sequence to frame log-probabilities.

    Returns one span per target token.  This is the core forced-alignment
    algorithm and intentionally lives in this project instead of another CLI.
    """
    if log_probs.ndim != 2:
        raise ValueError("log_probs must have shape [frames, vocab]")
    frames, vocab = log_probs.shape
    if not target_ids:
        return []
    if any(t < 0 or t >= vocab for t in target_ids):
        raise ValueError("target token outside model vocabulary")

    # Standard CTC expanded sequence: blank, t0, blank, t1, ..., blank.
    states = 2 * len(target_ids) + 1
    ext = np.full(states, int(blank_id), dtype=np.int32)
    ext[1::2] = np.asarray(target_ids, dtype=np.int32)

    neg_inf = np.float32(-1e30)
    previous = np.full(states, neg_inf, dtype=np.float32)
    back = np.full((frames, states), -1, dtype=np.int8)

    previous[0] = log_probs[0, blank_id]
    if states > 1:
        previous[1] = log_probs[0, ext[1]]
        back[0, 1] = 1
    back[0, 0] = 0

    for t in range(1, frames):
        current = np.full(states, neg_inf, dtype=np.float32)
        for s in range(states):
            best_score = previous[s]
            transition = 0  # stay
            if s > 0 and previous[s - 1] > best_score:
                best_score = previous[s - 1]
                transition = 1
            if (
                s > 1
                and ext[s] != blank_id
                and ext[s] != ext[s - 2]
                and previous[s - 2] > best_score
            ):
                best_score = previous[s - 2]
                transition = 2
            if best_score <= neg_inf / 2:
                continue
            current[s] = best_score + log_probs[t, ext[s]]
            back[t, s] = transition
        previous = current

    final_candidates = [states - 1]
    if states >= 2:
        final_candidates.append(states - 2)
    final_state = max(final_candidates, key=lambda s: float(previous[s]))
    if previous[final_state] <= neg_inf / 2:
        raise RuntimeError("CTC path не найден: текст не помещается в выбранное аудио-окно")

    path = np.empty(frames, dtype=np.int32)
    state = final_state
    for t in range(frames - 1, -1, -1):
        path[t] = state
        if t == 0:
            break
        transition = int(back[t, state])
        if transition < 0:
            raise RuntimeError("Повреждён CTC backtrace")
        state -= transition

    spans: list[dict[str, Any]] = []
    for token_index, token_id in enumerate(target_ids):
        target_state = 2 * token_index + 1
        indices = np.flatnonzero(path == target_state)
        if indices.size == 0:
            raise RuntimeError(f"CTC path пропустил token #{token_index}")
        first, last = int(indices[0]), int(indices[-1])
        probs = np.exp(log_probs[indices, token_id].astype(np.float64))
        confidence = float(np.mean(np.clip(probs, 0.0, 1.0)))
        spans.append({
            "token_index": token_index,
            "token_id": int(token_id),
            "start_frame": first,
            "end_frame": last + 1,
            "confidence": max(0.0, min(1.0, confidence)),
        })
    return spans


def _interpolate_missing_words(words: list[dict[str, Any]], line_start: float, line_end: float) -> None:
    """Fill only words that cannot be represented by the acoustic alphabet."""
    missing = [i for i, word in enumerate(words) if word.get("start") is None]
    if not missing:
        return

    for idx in missing:
        left = idx - 1
        while left >= 0 and words[left].get("end") is None:
            left -= 1
        right = idx + 1
        while right < len(words) and words[right].get("start") is None:
            right += 1
        left_time = float(words[left]["end"]) if left >= 0 and words[left].get("end") is not None else line_start
        right_time = float(words[right]["start"]) if right < len(words) and words[right].get("start") is not None else line_end

        block_start = left + 1
        block_end = right
        count = max(1, block_end - block_start)
        span = max(0.03 * count, right_time - left_time)
        step = span / count
        for offset, j in enumerate(range(block_start, block_end)):
            start = left_time + step * offset
            end = left_time + step * (offset + 1)
            words[j]["start"] = round(start, 3)
            words[j]["end"] = round(max(start + 0.01, end), 3)
            words[j]["confidence"] = 0.10
            words[j]["origin"] = "interpolated_unsupported_token"


def _alignment_candidate_score(line: dict[str, Any]) -> float:
    confidence = float(line.get("confidence") or 0.0)
    delta = abs(float(line.get("anchor_delta_ms") or 0.0)) / 1000.0
    # Acoustic evidence dominates; anchor proximity acts as a conservative prior.
    return confidence - min(0.22, delta * 0.055)


def _align_line_once(
    waveform: np.ndarray,
    source_line: dict[str, Any],
    next_anchor: float | None,
    *,
    language: str,
    model: Any,
    token_dict: dict[str, int],
    device: str,
    pre_margin: float,
    post_margin: float,
    last_line_window: float,
    method_suffix: str,
    romanizer: Any,
) -> dict[str, Any]:
    anchor_raw = source_line.get("anchor_start")
    if not isinstance(anchor_raw, (int, float)):
        anchor_raw = source_line.get("start")
    if not isinstance(anchor_raw, (int, float)):
        raise ValueError("Forced alignment требует LRC timestamp для каждой строки")
    anchor = max(0.0, float(anchor_raw))
    duration = len(waveform) / 16000.0
    window_start, window_end = _window_bounds(
        anchor,
        next_anchor,
        duration,
        pre_margin=pre_margin,
        post_margin=post_margin,
        last_line_window=last_line_window,
    )

    text = str(source_line.get("text") or "").strip()
    normalized_words, target = normalize_words(text, language, romanizer)
    if not normalized_words:
        raise ValueError("Пустая строка текста")
    if not target:
        # No alignable characters: preserve anchor and distribute conservatively.
        synthetic_end = next_anchor if next_anchor and next_anchor > anchor else min(duration, anchor + 1.5)
        words = [
            {"word": item.display, "start": None, "end": None, "confidence": 0.0, "origin": "missing"}
            for item in normalized_words
        ]
        _interpolate_missing_words(words, anchor, max(anchor + 0.05, synthetic_end))
        return {
            **source_line,
            "start": round(anchor, 3),
            "end": round(max(float(words[-1]["end"]), anchor + 0.05), 3),
            "text": text,
            "words": words,
            "confidence": 0.10,
            "method": f"{BACKEND_NAME}/{method_suffix}",
            "anchor_start": round(anchor, 3),
            "anchor_delta_ms": 0.0,
            "window_start": round(window_start, 3),
            "window_end": round(window_end, 3),
        }

    try:
        target_ids = [token_dict[ch] for ch in target]
    except KeyError as exc:
        raise RuntimeError(f"MMS_FA alphabet unexpectedly lacks {exc.args[0]!r}") from exc

    log_probs = _emission_for_window(waveform, window_start, window_end, model=model, device=device)
    token_spans = ctc_viterbi_align(log_probs, target_ids, blank_id=token_dict["-"])
    frame_sec = (window_end - window_start) / max(1, log_probs.shape[0])

    words: list[dict[str, Any]] = []
    aligned_confidences: list[float] = []
    for item in normalized_words:
        if item.token_start is None or item.token_end is None:
            words.append({
                "word": item.display,
                "start": None,
                "end": None,
                "confidence": 0.0,
                "origin": "missing",
                "normalized": item.chars,
            })
            continue
        spans = token_spans[item.token_start:item.token_end]
        first = spans[0]
        last = spans[-1]
        start = window_start + float(first["start_frame"]) * frame_sec
        end = window_start + float(last["end_frame"]) * frame_sec
        conf = float(np.mean([float(span["confidence"]) for span in spans]))
        aligned_confidences.append(conf)
        words.append({
            "word": item.display,
            "start": round(start, 3),
            "end": round(max(start + 0.01, end), 3),
            "confidence": round(conf, 4),
            "origin": "aligned" if conf >= 0.25 else "aligned_low_confidence",
            "normalized": item.chars,
        })

    first_aligned = next((w for w in words if w.get("start") is not None), None)
    last_aligned = next((w for w in reversed(words) if w.get("end") is not None), None)
    aligned_start = float(first_aligned["start"]) if first_aligned else anchor
    aligned_end = float(last_aligned["end"]) if last_aligned else max(anchor + 0.1, window_end)
    _interpolate_missing_words(words, aligned_start, aligned_end)

    line_start = float(words[0]["start"]) if words else aligned_start
    line_end = max(float(word["end"]) for word in words) if words else aligned_end
    if next_anchor is not None and next_anchor > anchor:
        line_end = min(line_end, next_anchor + 0.35)
        for word in words:
            if float(word["end"]) > next_anchor + 0.35:
                word["end"] = round(max(float(word["start"]) + 0.01, next_anchor + 0.35), 3)
                word["origin"] = "repaired_next_anchor_cap"
                word["confidence"] = min(float(word.get("confidence") or 0.0), 0.25)

    line_conf = float(np.mean(aligned_confidences)) if aligned_confidences else 0.10
    anchor_delta_ms = (line_start - anchor) * 1000.0
    return {
        **source_line,
        "start": round(line_start, 3),
        "end": round(max(line_start + 0.05, line_end), 3),
        "text": text,
        "words": words,
        "confidence": round(max(0.0, min(1.0, line_conf)), 4),
        "method": f"{BACKEND_NAME}/{method_suffix}",
        "anchor_start": round(anchor, 3),
        "anchor_delta_ms": round(anchor_delta_ms, 1),
        "window_start": round(window_start, 3),
        "window_end": round(window_end, 3),
    }


def _next_anchor(source_lines: list[dict[str, Any]], index: int) -> float | None:
    for future in source_lines[index + 1:]:
        value = future.get("anchor_start")
        if not isinstance(value, (int, float)):
            value = future.get("start")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _align_lines(
    waveform: np.ndarray,
    source_lines: list[dict[str, Any]],
    *,
    language: str,
    profile: AlignmentProfile,
    model: Any,
    token_dict: dict[str, int],
    device: str,
    progress_callback: ProgressCallback | None,
    retry_weak_lines: bool,
    only_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    romanizer = _get_romanizer()
    indices = [i for i in range(len(source_lines)) if only_indices is None or i in only_indices]
    result_map: dict[int, dict[str, Any]] = {}
    total = max(1, len(indices))

    for order, index in enumerate(indices):
        source = source_lines[index]
        next_anchor = _next_anchor(source_lines, index)
        if progress_callback:
            pct = 4 + round((order / total) * 90)
            progress_callback(pct, f"Forced alignment: строка {index + 1}/{len(source_lines)}")

        primary = _align_line_once(
            waveform, source, next_anchor,
            language=language,
            model=model,
            token_dict=token_dict,
            device=device,
            pre_margin=profile.pre_margin,
            post_margin=profile.post_margin,
            last_line_window=profile.last_line_window,
            method_suffix="local",
            romanizer=romanizer,
        )
        best = primary
        delta_sec = abs(float(primary.get("anchor_delta_ms") or 0.0)) / 1000.0
        should_retry = (
            retry_weak_lines
            and profile.retry_pre_margin > profile.pre_margin
            and (
                float(primary.get("confidence") or 0.0) < profile.weak_line_threshold
                or delta_sec > profile.anchor_retry_delta
            )
        )
        if should_retry:
            try:
                retry = _align_line_once(
                    waveform, source, next_anchor,
                    language=language,
                    model=model,
                    token_dict=token_dict,
                    device=device,
                    pre_margin=profile.retry_pre_margin,
                    post_margin=profile.retry_post_margin,
                    last_line_window=profile.last_line_window + 3.0,
                    method_suffix="expanded-retry",
                    romanizer=romanizer,
                )
                if _alignment_candidate_score(retry) > _alignment_candidate_score(primary) + 0.005:
                    best = retry
            except Exception:
                # A failed retry must never destroy a valid first pass.
                pass
        result_map[index] = best

    return [result_map[i] for i in indices]


def _demucs_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("demucs") is not None
    except Exception:
        return False


def _separate_vocals(
    audio_path: str,
    *,
    use_cuda: bool,
    timeout: int,
    progress_callback: ProgressCallback | None,
) -> tuple[str, tempfile.TemporaryDirectory[str]]:
    if not _demucs_available():
        raise RuntimeError("Demucs не установлен (это необязательный fallback)")
    temp_dir = tempfile.TemporaryDirectory(prefix="lrc_extandator_v7_demucs_")
    output = Path(temp_dir.name)
    device = "cuda" if use_cuda else "cpu"
    if progress_callback:
        progress_callback(5, f"Demucs: отделяю вокал на {device.upper()} только для слабых строк…")
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "--segment", "7",
        "-d", device,
        "-o", str(output),
        str(audio_path),
    ]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=flags)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2200:]
        temp_dir.cleanup()
        raise RuntimeError(f"Demucs завершился с ошибкой: {tail}")
    stem = Path(audio_path).stem
    expected = output / "htdemucs" / stem / "vocals.wav"
    if not expected.exists():
        matches = list(output.glob("**/vocals.wav"))
        if not matches:
            temp_dir.cleanup()
            raise RuntimeError("Demucs завершился, но vocals.wav не найден")
        expected = matches[0]
    return str(expected), temp_dir


def _sha256_file(path: str | os.PathLike[str], chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _lyrics_fingerprint(lines: Iterable[dict[str, Any]]) -> str:
    payload = []
    for line in lines:
        anchor = line.get("anchor_start")
        if not isinstance(anchor, (int, float)):
            anchor = line.get("start")
        payload.append({
            "text": str(line.get("text") or ""),
            "anchor": round(float(anchor), 3) if isinstance(anchor, (int, float)) else None,
        })
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cache_key(audio_path: str, lines: list[dict[str, Any]], language: str, options: AlignmentOptions) -> str:
    payload = {
        "engine": ALIGNMENT_ENGINE_VERSION,
        "backend": BACKEND_NAME,
        "audio": _sha256_file(audio_path),
        "lyrics": _lyrics_fingerprint(lines),
        "language": _iso3(language),
        "options": asdict(options),
        "torch": _package_version("torch"),
        "torchaudio": _package_version("torchaudio"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_timed_lines(source_lines: list[dict[str, Any]]) -> None:
    missing: list[int] = []
    for i, line in enumerate(source_lines):
        anchor = line.get("anchor_start")
        if not isinstance(anchor, (int, float)):
            anchor = line.get("start")
        if not isinstance(anchor, (int, float)):
            missing.append(i + 1)
    if missing:
        preview = ", ".join(map(str, missing[:8])) + ("…" if len(missing) > 8 else "")
        raise ValueError(
            "Forced Alignment v7 требует синхронизированный LRC с таймкодом каждой строки. "
            f"Нет таймкода у строк: {preview}. ASR намеренно не используется."
        )


class AlignmentEngine:
    def __init__(self, cache: AlignmentCache | None = None):
        self.cache = cache or AlignmentCache()

    def align(
        self,
        audio_path: str | os.PathLike[str],
        source_lines: list[dict[str, Any]],
        *,
        language: str = "eng",
        options: AlignmentOptions | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        options = options or AlignmentOptions()
        profile = get_profile(options.quality_mode)
        audio_path = str(Path(audio_path).resolve())
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(audio_path)
        if not source_lines:
            raise ValueError("Нет строк текста для выравнивания")
        _validate_timed_lines(source_lines)

        key = _cache_key(audio_path, source_lines, language, options)
        if options.use_cache:
            cached = self.cache.get(key)
            if cached:
                cached["cacheHit"] = True
                if progress_callback:
                    progress_callback(100, "Готово: forced alignment взят из кэша")
                return cached

        started = time.perf_counter()
        if progress_callback:
            progress_callback(1, "Forced Alignment v7: LRC anchors → local CTC windows")
        waveform = decode_audio_16k_mono(audio_path, progress_callback)
        model, token_dict, device = _load_runtime(progress_callback)
        accel = detect_acceleration()
        if progress_callback:
            name = accel.gpu_name or "CPU"
            progress_callback(9, f"MMS forced alignment: {name}; распознавание текста отключено")

        # First pass: original mix for all lines, unless Demucs was explicitly forced.
        mix_lines: list[dict[str, Any]] | None = None
        mix_quality: dict[str, Any] | None = None
        candidates: list[dict[str, Any]] = []

        if options.use_demucs is not True:
            mix_lines = _align_lines(
                waveform,
                source_lines,
                language=language,
                profile=profile,
                model=model,
                token_dict=token_dict,
                device=device,
                progress_callback=(lambda p, m: progress_callback(10 + round(p * 0.68), m)) if progress_callback else None,
                retry_weak_lines=options.retry_weak_lines,
            )
            mix_lines, _issues = repair_lines(mix_lines, duration=len(waveform) / 16000.0)
            mix_quality = quality_report(mix_lines)
            mix_quality["runtimeSec"] = round(time.perf_counter() - started, 3)
            candidates.append({
                "config": {"label": "mix/local-ctc", "use_demucs": False},
                "lines": mix_lines,
                "quality": mix_quality,
            })

        should_demucs = options.use_demucs is True
        weak_indices: set[int] = set()
        if mix_lines is not None and mix_quality is not None:
            weak_indices = {
                i for i, line in enumerate(mix_lines)
                if float(line.get("confidence") or 0.0) < profile.weak_line_threshold
                or abs(float(line.get("anchor_delta_ms") or 0.0)) > profile.anchor_retry_delta * 1000.0
            }
            weak_ratio = len(weak_indices) / max(1, len(mix_lines))
            if options.use_demucs == "auto" and profile.name != "fast":
                should_demucs = (
                    float(mix_quality.get("score") or 0.0) < profile.demucs_trigger_score
                    or weak_ratio >= profile.demucs_trigger_weak_ratio
                )

        if should_demucs:
            if progress_callback:
                progress_callback(80, "Низкая уверенность: один Demucs retry, без ASR")
            temp_handle: tempfile.TemporaryDirectory[str] | None = None
            try:
                # The acoustic model is ~1.2 GB. Release it before Demucs on 6 GB cards.
                release_acoustic_model()
                vocals_path, temp_handle = _separate_vocals(
                    audio_path,
                    use_cuda=(accel.device == "cuda"),
                    timeout=options.demucs_timeout_seconds,
                    progress_callback=(lambda p, m: progress_callback(80 + round(p * 0.06), m)) if progress_callback else None,
                )
                vocals_waveform = decode_audio_16k_mono(vocals_path)
                model, token_dict, device = _load_runtime(
                    (lambda p, m: progress_callback(86, m)) if progress_callback else None
                )

                if mix_lines is not None and options.demucs_weak_lines_only and weak_indices:
                    vocal_subset = _align_lines(
                        vocals_waveform,
                        source_lines,
                        language=language,
                        profile=profile,
                        model=model,
                        token_dict=token_dict,
                        device=device,
                        progress_callback=(lambda p, m: progress_callback(87 + round(p * 0.10), m)) if progress_callback else None,
                        retry_weak_lines=options.retry_weak_lines,
                        only_indices=weak_indices,
                    )
                    vocal_by_index = {idx: line for idx, line in zip(sorted(weak_indices), vocal_subset)}
                    merged: list[dict[str, Any]] = []
                    improved = 0
                    for idx, mix_line in enumerate(mix_lines):
                        vocal_line = vocal_by_index.get(idx)
                        if vocal_line is not None and _alignment_candidate_score(vocal_line) > _alignment_candidate_score(mix_line) + 0.01:
                            vocal_line["method"] = str(vocal_line.get("method")) + "/demucs-selected"
                            merged.append(vocal_line)
                            improved += 1
                        else:
                            merged.append(mix_line)
                    vocal_lines = merged
                    label = f"hybrid/mix+demucs({improved})"
                else:
                    vocal_lines = _align_lines(
                        vocals_waveform,
                        source_lines,
                        language=language,
                        profile=profile,
                        model=model,
                        token_dict=token_dict,
                        device=device,
                        progress_callback=(lambda p, m: progress_callback(87 + round(p * 0.10), m)) if progress_callback else None,
                        retry_weak_lines=options.retry_weak_lines,
                    )
                    label = "vocals/local-ctc"

                vocal_lines, _issues = repair_lines(vocal_lines, duration=len(vocals_waveform) / 16000.0)
                vocal_quality = quality_report(vocal_lines)
                vocal_quality["runtimeSec"] = round(time.perf_counter() - started, 3)
                candidates.append({
                    "config": {"label": label, "use_demucs": True},
                    "lines": vocal_lines,
                    "quality": vocal_quality,
                })
            except Exception as exc:
                if options.use_demucs is True and not candidates:
                    raise
                if progress_callback:
                    progress_callback(96, f"Demucs retry пропущен: {exc}")
            finally:
                if temp_handle is not None:
                    temp_handle.cleanup()

        if not candidates:
            raise RuntimeError("Forced alignment не создал ни одного кандидата")
        best = candidates[0]
        for candidate in candidates[1:]:
            if compare_quality(candidate["quality"], best["quality"]) > 0:
                best = candidate

        runtime = round(time.perf_counter() - started, 3)
        result = {
            "engineVersion": ALIGNMENT_ENGINE_VERSION,
            "backend": BACKEND_NAME,
            "cacheHit": False,
            "device": accel.device,
            "gpuName": accel.gpu_name,
            "lines": best["lines"],
            "quality": best["quality"],
            "selectedCandidate": best["config"],
            "candidates": [
                {"config": c["config"], "quality": c["quality"]}
                for c in candidates
            ],
            "runtimeSec": runtime,
        }
        if options.use_cache:
            self.cache.put(key, result)
        if progress_callback:
            progress_callback(
                100,
                f"Готово: {BACKEND_NAME}, {best['quality'].get('grade', 'unknown')} / {best['quality'].get('score', 0):.3f}",
            )
        return result
