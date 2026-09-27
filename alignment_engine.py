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
import statistics
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

ALIGNMENT_ENGINE_VERSION = "7.2.0"
BACKEND_NAME = "MMS_FA/custom-CTC-viterbi-adaptive-rescue"
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
    lookahead_words: int
    boundary_confidence_threshold: float
    boundary_danger_zone: float
    boundary_rescue_pre_margin: float
    boundary_rescue_post_margin: float
    boundary_refine_all: bool


PROFILES: dict[str, AlignmentProfile] = {
    "fast": AlignmentProfile(
        name="fast",
        pre_margin=0.65,
        post_margin=0.45,
        last_line_window=9.0,
        retry_pre_margin=0.85,
        retry_post_margin=0.70,
        weak_line_threshold=0.42,
        anchor_retry_delta=1.10,
        demucs_trigger_score=0.0,
        demucs_trigger_weak_ratio=2.0,
        lookahead_words=1,
        boundary_confidence_threshold=0.34,
        boundary_danger_zone=0.14,
        boundary_rescue_pre_margin=0.70,
        boundary_rescue_post_margin=0.80,
        boundary_refine_all=False,
    ),
    "balanced": AlignmentProfile(
        name="balanced",
        pre_margin=0.95,
        post_margin=0.85,
        last_line_window=11.0,
        retry_pre_margin=1.65,
        retry_post_margin=1.15,
        weak_line_threshold=0.50,
        anchor_retry_delta=0.95,
        demucs_trigger_score=0.78,
        demucs_trigger_weak_ratio=0.22,
        lookahead_words=2,
        boundary_confidence_threshold=0.46,
        boundary_danger_zone=0.22,
        boundary_rescue_pre_margin=0.95,
        boundary_rescue_post_margin=1.15,
        boundary_refine_all=False,
    ),
    "max": AlignmentProfile(
        name="max",
        pre_margin=1.25,
        post_margin=1.20,
        last_line_window=13.0,
        retry_pre_margin=2.20,
        retry_post_margin=1.55,
        weak_line_threshold=0.56,
        anchor_retry_delta=0.80,
        demucs_trigger_score=0.84,
        demucs_trigger_weak_ratio=0.14,
        lookahead_words=3,
        boundary_confidence_threshold=0.54,
        boundary_danger_zone=0.30,
        boundary_rescue_pre_margin=1.20,
        boundary_rescue_post_margin=1.45,
        boundary_refine_all=True,
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
    words = line.get("words") or []
    last_word = next((w for w in reversed(words) if isinstance(w.get("start"), (int, float))), None)
    last_conf = float(last_word.get("confidence") or confidence) if last_word else confidence

    # Do not let a high mean hide one broken word. V7.2 scores the weakest
    # internal phrase too, not only the final word.
    confidences = [
        float(word.get("confidence") or 0.0)
        for word in words
        if isinstance(word.get("start"), (int, float))
    ]
    median_conf = float(statistics.median(confidences)) if confidences else confidence
    min_conf = min(confidences, default=confidence)
    weak_ratio = sum(value < 0.45 for value in confidences) / max(1, len(confidences))
    score = (
        0.50 * confidence
        + 0.18 * last_conf
        + 0.20 * median_conf
        + 0.12 * min_conf
        - 0.16 * weak_ratio
    )
    score -= min(0.22, delta * 0.055)

    lead_ms = line.get("boundary_lead_ms")
    if isinstance(lead_ms, (int, float)):
        if float(lead_ms) < -80.0:
            score -= 0.28
        elif float(lead_ms) < 30.0:
            score -= 0.12
    context_gap_ms = line.get("boundary_context_gap_ms")
    if isinstance(context_gap_ms, (int, float)):
        if float(context_gap_ms) <= 0.0:
            score -= 0.30
        elif float(context_gap_ms) < 55.0:
            score -= 0.10
    return score


def _first_context_words(source_line: dict[str, Any] | None, count: int) -> str:
    if not source_line or count <= 0:
        return ""
    parts = [part for part in re.findall(r"\S+", str(source_line.get("text") or "")) if part.strip()]
    return " ".join(parts[:count])


def _lookahead_diagnostics(
    token_spans: list[dict[str, Any]],
    *,
    current_token_count: int,
    lookahead_words_normalized: list[NormalizedWord],
    window_start: float,
    frame_sec: float,
) -> tuple[float | None, float | None]:
    """Return the acoustic start/confidence of the first alignable lookahead word."""
    for item in lookahead_words_normalized:
        if item.token_start is None or item.token_end is None:
            continue
        spans = token_spans[
            current_token_count + item.token_start : current_token_count + item.token_end
        ]
        if not spans:
            continue
        first = spans[0]
        start = window_start + float(first["start_frame"]) * frame_sec
        confidence = float(np.mean([float(span["confidence"]) for span in spans]))
        return round(start, 3), round(max(0.0, min(1.0, confidence)), 4)
    return None, None


def _align_line_once(
    waveform: np.ndarray,
    source_line: dict[str, Any],
    next_anchor: float | None,
    *,
    next_line: dict[str, Any] | None,
    lookahead_words: int,
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
    """Align one known line, optionally appending words from the next line.

    The appended words are *context only*.  They force CTC to explain the
    acoustic transition into the next lyric line instead of being free to push
    the last word of the current line against the right edge of the window.
    Only current-line words are returned to the caller.
    """
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
    normalized_words, current_target = normalize_words(text, language, romanizer)
    if not normalized_words:
        raise ValueError("Пустая строка текста")

    requested_context_words = max(0, int(lookahead_words)) if next_anchor is not None else 0
    context_text = _first_context_words(next_line, requested_context_words) if requested_context_words else ""
    lookahead_normalized: list[NormalizedWord] = []
    lookahead_target = ""
    if context_text:
        lookahead_normalized, lookahead_target = normalize_words(context_text, language, romanizer)

    if not current_target:
        # No alignable characters: preserve the LRC anchor and distribute only
        # the unsupported visible tokens.  No recognition is attempted.
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
            "lookahead_text": context_text or None,
            "lookahead_start": None,
            "lookahead_confidence": None,
            "lookahead_words_requested": requested_context_words,
            "lookahead_words_used": 0,
        }

    log_probs = _emission_for_window(waveform, window_start, window_end, model=model, device=device)

    # Long next-line phrases sometimes do not physically fit into the configured
    # overlap.  Do not fail the current line because of optional context: reuse
    # the same acoustic emission and back off 3 -> 2 -> 1 -> 0 context words.
    token_spans: list[dict[str, Any]] | None = None
    alignment_error: Exception | None = None
    used_context_words = requested_context_words
    for context_count in range(requested_context_words, -1, -1):
        context_text = _first_context_words(next_line, context_count) if context_count else ""
        if context_text:
            lookahead_normalized, lookahead_target = normalize_words(context_text, language, romanizer)
        else:
            lookahead_normalized, lookahead_target = [], ""
        full_target = current_target + lookahead_target
        try:
            target_ids = [token_dict[ch] for ch in full_target]
        except KeyError as exc:
            raise RuntimeError(f"MMS_FA alphabet unexpectedly lacks {exc.args[0]!r}") from exc
        try:
            token_spans = ctc_viterbi_align(log_probs, target_ids, blank_id=token_dict["-"])
            used_context_words = context_count
            break
        except RuntimeError as exc:
            alignment_error = exc
            continue
    if token_spans is None:
        raise alignment_error or RuntimeError("CTC path не найден")

    frame_sec = (window_end - window_start) / max(1, log_probs.shape[0])

    # We intentionally discard the appended next-line context from the visible
    # result after it has constrained the CTC path.
    current_spans = token_spans[: len(current_target)]
    lookahead_start, lookahead_confidence = _lookahead_diagnostics(
        token_spans,
        current_token_count=len(current_target),
        lookahead_words_normalized=lookahead_normalized,
        window_start=window_start,
        frame_sec=frame_sec,
    )

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
        spans = current_spans[item.token_start:item.token_end]
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

    # The old v7 cap was next_anchor + 350 ms.  That could cut off a sustained
    # final word.  v7.2 uses the acoustically aligned next-line context as the
    # natural right boundary.  Only when context is unavailable do we use the
    # (larger) configured overlap as a conservative hard cap.
    if lookahead_start is not None:
        hard_cap = float(lookahead_start)
    elif next_anchor is not None and next_anchor > anchor:
        hard_cap = float(next_anchor) + float(post_margin)
    else:
        hard_cap = None
    if hard_cap is not None:
        line_end = min(line_end, max(line_start + 0.05, hard_cap))
        for word in words:
            if float(word["end"]) > hard_cap:
                word["end"] = round(max(float(word["start"]) + 0.01, hard_cap), 3)
                word["origin"] = "repaired_boundary_context_cap"
                word["confidence"] = min(float(word.get("confidence") or 0.0), 0.35)

    line_conf = float(np.mean(aligned_confidences)) if aligned_confidences else 0.10
    anchor_delta_ms = (line_start - anchor) * 1000.0
    last_word = next((w for w in reversed(words) if w.get("start") is not None), None)
    boundary_lead_ms = None
    if last_word is not None and next_anchor is not None:
        boundary_lead_ms = (float(next_anchor) - float(last_word["start"])) * 1000.0
    context_gap_ms = None
    if last_word is not None and lookahead_start is not None:
        context_gap_ms = (float(lookahead_start) - float(last_word["start"])) * 1000.0

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
        "lookahead_text": context_text or None,
        "lookahead_start": lookahead_start,
        "lookahead_confidence": lookahead_confidence,
        "lookahead_words_requested": requested_context_words,
        "lookahead_words_used": used_context_words,
        "boundary_lead_ms": round(boundary_lead_ms, 1) if boundary_lead_ms is not None else None,
        "boundary_context_gap_ms": round(context_gap_ms, 1) if context_gap_ms is not None else None,
    }


def _last_word_candidate_score(
    word: dict[str, Any],
    *,
    next_anchor: float | None,
    lookahead_start: float | None,
) -> float:
    """Score a last-word candidate without pretending the LRC anchor is exact."""
    confidence = float(word.get("confidence") or 0.0)
    start = float(word.get("start") or 0.0)
    end = float(word.get("end") or start)
    score = confidence
    origin = str(word.get("origin") or "")
    if "low_confidence" in origin or origin.startswith("interpolated"):
        score -= 0.12
    if next_anchor is not None:
        relative = start - float(next_anchor)
        if relative >= 0.08:
            score -= 0.52
        elif relative >= -0.03:
            score -= 0.18
    if lookahead_start is not None:
        gap = float(lookahead_start) - start
        if gap <= 0.0:
            score -= 0.65
        elif gap < 0.055:
            score -= 0.16
        elif gap >= 0.10:
            score += min(0.06, gap * 0.03)
        if end > float(lookahead_start) + 0.05:
            score -= 0.18
    if end - start < 0.025:
        score -= 0.05
    return score


def _boundary_is_suspicious(
    line: dict[str, Any],
    *,
    next_anchor: float | None,
    profile: AlignmentProfile,
) -> bool:
    if next_anchor is None:
        return False
    words = line.get("words") or []
    last_word = next((w for w in reversed(words) if isinstance(w.get("start"), (int, float))), None)
    if last_word is None:
        return True
    start = float(last_word["start"])
    confidence = float(last_word.get("confidence") or 0.0)
    if confidence < profile.boundary_confidence_threshold:
        return True
    if start >= float(next_anchor) - profile.boundary_danger_zone:
        return True
    lookahead_start = line.get("lookahead_start")
    if isinstance(lookahead_start, (int, float)) and float(lookahead_start) - start < 0.065:
        return True
    if float(line.get("window_end") or 0.0) - float(last_word.get("end") or start) < 0.055:
        return True
    return False


def _refine_last_word_boundary(
    waveform: np.ndarray,
    source_line: dict[str, Any],
    next_line: dict[str, Any] | None,
    next_anchor: float | None,
    current: dict[str, Any],
    *,
    language: str,
    profile: AlignmentProfile,
    model: Any,
    token_dict: dict[str, int],
    device: str,
    romanizer: Any,
) -> dict[str, Any]:
    """Second, narrow CTC pass around the last word and the next-line onset.

    It uses the already-aligned penultimate/antepenultimate word only as a crop
    hint.  The actual last-word timestamp must still be supported acoustically.
    """
    if next_anchor is None or next_line is None:
        return current
    normalized_words, _ = normalize_words(str(source_line.get("text") or ""), language, romanizer)
    alignable_indices = [i for i, item in enumerate(normalized_words) if item.token_start is not None]
    if not alignable_indices:
        return current
    last_index = alignable_indices[-1]
    tail_start_index = max(0, last_index - 2)
    tail_items = normalized_words[tail_start_index:last_index + 1]
    tail_text = " ".join(item.display for item in tail_items).strip()
    if not tail_text:
        return current

    current_words = current.get("words") or []
    seed_start = None
    if tail_start_index < len(current_words):
        value = current_words[tail_start_index].get("start")
        if isinstance(value, (int, float)):
            seed_start = float(value)
    if seed_start is None:
        anchor_raw = source_line.get("anchor_start")
        if not isinstance(anchor_raw, (int, float)):
            anchor_raw = source_line.get("start")
        seed_start = float(anchor_raw) if isinstance(anchor_raw, (int, float)) else max(0.0, next_anchor - 2.0)

    pseudo_line = {
        "text": tail_text,
        "start": seed_start,
        "anchor_start": seed_start,
        "anchor_origin": "boundary_rescue_seed",
    }
    try:
        rescue = _align_line_once(
            waveform,
            pseudo_line,
            next_anchor,
            next_line=next_line,
            lookahead_words=max(1, profile.lookahead_words),
            language=language,
            model=model,
            token_dict=token_dict,
            device=device,
            pre_margin=profile.boundary_rescue_pre_margin,
            post_margin=profile.boundary_rescue_post_margin,
            last_line_window=profile.last_line_window,
            method_suffix="boundary-rescue",
            romanizer=romanizer,
        )
    except Exception:
        return current

    rescue_words = rescue.get("words") or []
    if not rescue_words or not current_words:
        return current
    rescue_last = rescue_words[-1]
    primary_last = current_words[last_index] if last_index < len(current_words) else current_words[-1]
    if not isinstance(rescue_last.get("start"), (int, float)) or not isinstance(primary_last.get("start"), (int, float)):
        return current

    primary_score = _last_word_candidate_score(
        primary_last,
        next_anchor=next_anchor,
        lookahead_start=current.get("lookahead_start") if isinstance(current.get("lookahead_start"), (int, float)) else None,
    )
    rescue_score = _last_word_candidate_score(
        rescue_last,
        next_anchor=next_anchor,
        lookahead_start=rescue.get("lookahead_start") if isinstance(rescue.get("lookahead_start"), (int, float)) else None,
    )
    primary_start = float(primary_last["start"])
    rescue_start = float(rescue_last["start"])
    primary_conf = float(primary_last.get("confidence") or 0.0)
    rescue_conf = float(rescue_last.get("confidence") or 0.0)
    suspicious = _boundary_is_suspicious(current, next_anchor=next_anchor, profile=profile)

    accept = rescue_score > primary_score + 0.015
    # If the primary timestamp is stuck against the next anchor, accept a clear
    # earlier acoustic solution even when its raw CTC confidence is almost tied.
    if (
        not accept
        and suspicious
        and rescue_start <= primary_start - 0.060
        and rescue_conf >= primary_conf - 0.035
        and rescue_start < float(next_anchor) - 0.035
    ):
        accept = True
    if not accept:
        current["boundary_rescue_checked"] = True
        current["boundary_rescue_used"] = False
        current["boundary_rescue_score_delta"] = round(rescue_score - primary_score, 4)
        return current

    replacement = dict(rescue_last)
    replacement["origin"] = "aligned_boundary_rescue"
    replacement["boundary_previous_start"] = round(primary_start, 3)
    current_words[last_index] = replacement
    current["words"] = current_words
    current["lookahead_start"] = rescue.get("lookahead_start") or current.get("lookahead_start")
    current["lookahead_confidence"] = rescue.get("lookahead_confidence") or current.get("lookahead_confidence")
    current["boundary_rescue_checked"] = True
    current["boundary_rescue_used"] = True
    current["boundary_rescue_score_delta"] = round(rescue_score - primary_score, 4)
    current["boundary_rescue_delta_ms"] = round((rescue_start - primary_start) * 1000.0, 1)
    current["method"] = f"{current.get('method') or BACKEND_NAME}+boundary-rescue"
    current["end"] = round(max(float(w.get("end") or w.get("start") or current.get("start") or 0.0) for w in current_words), 3)
    aligned_confs = [
        float(w.get("confidence") or 0.0)
        for w in current_words
        if not str(w.get("origin") or "").startswith("interpolated")
    ]
    if aligned_confs:
        current["confidence"] = round(float(np.mean(aligned_confs)), 4)
    current["boundary_lead_ms"] = round((float(next_anchor) - rescue_start) * 1000.0, 1)
    if isinstance(current.get("lookahead_start"), (int, float)):
        current["boundary_context_gap_ms"] = round((float(current["lookahead_start"]) - rescue_start) * 1000.0, 1)
    return current


def _word_units(word: dict[str, Any]) -> float:
    """Approximate pronunciation mass for tempo fallback.

    We intentionally keep this language-agnostic: MMS/uroman already stores a
    romanized form per word.  Vowels receive a small extra weight because sung
    vowels commonly carry most of a syllable's duration.
    """
    normalized = str(word.get("normalized") or "").lower()
    if not normalized:
        normalized = re.sub(r"[^a-z0-9]+", "", _deobfuscate_word(str(word.get("word") or "")).lower())
    if not normalized:
        return 1.0
    vowels = sum(ch in "aeiouy" for ch in normalized)
    return max(1.0, len(normalized) * 0.72 + vowels * 0.28)


def _line_tempo_sec_per_unit(line: dict[str, Any]) -> float | None:
    words = [w for w in (line.get("words") or []) if isinstance(w.get("start"), (int, float))]
    if len(words) < 2 or float(line.get("confidence") or 0.0) < 0.45:
        return None
    start = float(words[0]["start"])
    last = words[-1]
    right = line.get("lookahead_start")
    if not isinstance(right, (int, float)) or float(right) <= float(last["start"]):
        right = last.get("end")
    if not isinstance(right, (int, float)):
        return None
    span = float(right) - start
    units = sum(_word_units(word) for word in words)
    if span <= 0.08 or units <= 0:
        return None
    value = span / units
    # Broad enough for rap and sustained vocals, narrow enough to reject pauses.
    if not 0.018 <= value <= 0.42:
        return None
    return value


def _tempo_prior_for_index(lines: list[dict[str, Any]], index: int, radius: int = 3) -> float | None:
    local: list[float] = []
    for distance in range(1, radius + 1):
        for candidate_index in (index - distance, index + distance):
            if not 0 <= candidate_index < len(lines):
                continue
            candidate = lines[candidate_index]
            words = candidate.get("words") or []
            if not words:
                continue
            bad = sum(
                1 for word in words
                if float(word.get("confidence") or 0.0) < 0.42
                or str(word.get("origin") or "").startswith(("interpolated", "tempo_"))
            )
            if bad / max(1, len(words)) > 0.25:
                continue
            tempo = _line_tempo_sec_per_unit(candidate)
            if tempo is not None:
                local.append(tempo)
    if local:
        return float(statistics.median(local))

    global_values = [value for line in lines if (value := _line_tempo_sec_per_unit(line)) is not None]
    return float(statistics.median(global_values)) if global_values else None


def _weighted_median(values: list[tuple[float, float]]) -> float:
    if not values:
        raise ValueError("weighted median requires values")
    ordered = sorted((float(value), max(0.001, float(weight))) for value, weight in values)
    total = sum(weight for _value, weight in ordered)
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= total / 2.0:
            return value
    return ordered[-1][0]


def _energy_onset_near(waveform: np.ndarray, predicted: float, radius: float = 0.095) -> float:
    """Snap a fallback timestamp to a nearby positive energy transition.

    This is deliberately only a *fallback* for words CTC could not place
    consistently.  It never overrides a stable acoustic alignment.
    """
    sr = 16000
    center = int(max(0.0, predicted) * sr)
    left = max(0, center - int(radius * sr))
    right = min(len(waveform), center + int(radius * sr))
    segment = np.asarray(waveform[left:right], dtype=np.float32)
    if segment.size < 640:
        return predicted
    frame = 320
    hop = 160
    energies: list[float] = []
    positions: list[int] = []
    for pos in range(0, max(1, segment.size - frame + 1), hop):
        chunk = segment[pos:pos + frame]
        if chunk.size < frame:
            break
        energies.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-9)))
        positions.append(pos)
    if len(energies) < 3:
        return predicted
    log_energy = np.log(np.asarray(energies, dtype=np.float32) + 1e-6)
    novelty = np.maximum(0.0, np.diff(log_energy, prepend=log_energy[0]))
    best = int(np.argmax(novelty))
    median = float(np.median(novelty))
    strength = float(novelty[best])
    if strength < median + 0.10:
        return predicted
    snapped = (left + positions[best]) / sr
    # Avoid a dramatic jump caused by a drum transient at the search edge.
    return snapped if abs(snapped - predicted) <= radius * 0.92 else predicted


def _word_tempo_outliers(words: list[dict[str, Any]], tempo_prior: float | None, right_boundary: float | None) -> set[int]:
    if tempo_prior is None or not words:
        return set()
    result: set[int] = set()
    for index, word in enumerate(words):
        if not isinstance(word.get("start"), (int, float)):
            result.add(index)
            continue
        start = float(word["start"])
        if index + 1 < len(words) and isinstance(words[index + 1].get("start"), (int, float)):
            right = float(words[index + 1]["start"])
        elif isinstance(right_boundary, (int, float)):
            right = float(right_boundary)
        elif isinstance(word.get("end"), (int, float)):
            right = float(word["end"])
        else:
            continue
        observed = right - start
        expected = max(0.035, _word_units(word) * tempo_prior)
        ratio = observed / expected
        conf = float(word.get("confidence") or 0.0)
        # High-confidence acoustic evidence is allowed to violate the tempo:
        # a singer may sustain one vowel for a very long time. Tempo becomes a
        # rescue signal only when the acoustic evidence is weak *and* the ratio
        # is extreme.
        if conf < 0.45 and (ratio < 0.22 or ratio > 3.40):
            result.add(index)
        elif conf < 0.62 and (ratio < 0.12 or ratio > 5.00):
            result.add(index)
    return result


def _adaptive_line_reasons(
    line: dict[str, Any],
    *,
    profile: AlignmentProfile,
    tempo_prior: float | None,
    next_anchor: float | None,
) -> list[str]:
    words = line.get("words") or []
    if not words:
        return ["no_words"]
    confidences = [float(word.get("confidence") or 0.0) for word in words]
    threshold = 0.48 if profile.name == "max" else 0.40
    reasons: list[str] = []
    low = [value < threshold for value in confidences]
    if min(confidences, default=0.0) < (0.32 if profile.name == "max" else 0.26):
        reasons.append("very_low_word_confidence")
    if sum(low) / max(1, len(low)) >= (0.18 if profile.name == "max" else 0.28):
        reasons.append("low_confidence_phrase")
    if any(low[i] and low[i + 1] for i in range(len(low) - 1)):
        reasons.append("consecutive_weak_words")
    if any(str(word.get("origin") or "").startswith("interpolated") for word in words):
        reasons.append("interpolated_word")
    right_boundary = line.get("lookahead_start")
    if not isinstance(right_boundary, (int, float)):
        right_boundary = next_anchor
    if _word_tempo_outliers(words, tempo_prior, right_boundary):
        reasons.append("tempo_outlier")
    if _boundary_is_suspicious(line, next_anchor=next_anchor, profile=profile):
        reasons.append("boundary")
    return reasons


def _candidate_consensus(
    candidates: list[dict[str, Any]],
    *,
    waveform: np.ndarray,
    tempo_prior: float | None,
    next_boundary: float | None,
) -> dict[str, Any]:
    """Merge several genuinely different CTC crops, then tempo-repair only uncertainty."""
    if not candidates:
        raise ValueError("No adaptive candidates")
    base = max(candidates, key=_alignment_candidate_score)
    merged = {**base, "words": [dict(word) for word in (base.get("words") or [])]}
    words = merged["words"]
    if not words:
        return merged

    unstable: set[int] = set()
    consensus_used = 0
    for index, word in enumerate(words):
        observations: list[tuple[float, float, float, str]] = []
        for candidate in candidates:
            candidate_words = candidate.get("words") or []
            if index >= len(candidate_words):
                continue
            item = candidate_words[index]
            if not isinstance(item.get("start"), (int, float)):
                continue
            observations.append((
                float(item["start"]),
                float(item.get("confidence") or 0.0),
                float(item.get("end") or item["start"]),
                str(item.get("origin") or ""),
            ))
        if not observations:
            unstable.add(index)
            continue
        starts = [value for value, _conf, _end, _origin in observations]
        spread = max(starts) - min(starts)
        best_obs = max(observations, key=lambda item: item[1])
        median_start = _weighted_median([(value, 0.15 + conf * conf) for value, conf, _end, _origin in observations])
        median_end = _weighted_median([(end, 0.15 + conf * conf) for _value, conf, end, _origin in observations])
        mean_conf = float(statistics.fmean(conf for _value, conf, _end, _origin in observations))
        word["consensus_spread_ms"] = round(spread * 1000.0, 1)
        if len(observations) >= 2 and spread <= 0.155:
            word["start"] = round(median_start, 3)
            word["end"] = round(max(median_start + 0.01, median_end), 3)
            word["confidence"] = round(min(0.97, mean_conf + 0.025), 4)
            word["origin"] = "adaptive_consensus"
            consensus_used += 1
        else:
            word["start"] = round(best_obs[0], 3)
            word["end"] = round(max(best_obs[0] + 0.01, best_obs[2]), 3)
            word["confidence"] = round(best_obs[1], 4)
            word["origin"] = best_obs[3] or "adaptive_best_pass"
            if spread > 0.240 or best_obs[1] < 0.36:
                unstable.add(index)

    unstable |= _word_tempo_outliers(words, tempo_prior, next_boundary)
    unstable |= {
        index for index, word in enumerate(words)
        if float(word.get("confidence") or 0.0) < 0.31
        or str(word.get("origin") or "").startswith("interpolated")
    }

    reconstructed = 0
    if unstable and tempo_prior is not None:
        reconstructed = _tempo_reconstruct_words(
            words,
            unstable=unstable,
            waveform=waveform,
            tempo_prior=tempo_prior,
            line_start=float(merged.get("start") or words[0].get("start") or 0.0),
            right_boundary=next_boundary,
        )

    # Rebuild safe end-times after per-word consensus/reconstruction.
    previous = float(merged.get("start") or 0.0) - 0.001
    for index, word in enumerate(words):
        start = max(previous + 0.001, float(word.get("start") or previous + 0.02))
        word["start"] = round(start, 3)
        previous = start
    for index, word in enumerate(words):
        if index + 1 < len(words):
            cap = float(words[index + 1]["start"])
        elif isinstance(next_boundary, (int, float)) and float(next_boundary) > float(word["start"]):
            cap = float(next_boundary)
        else:
            cap = max(float(word["start"]) + 0.04, float(word.get("end") or 0.0))
        word["end"] = round(max(float(word["start"]) + 0.01, min(cap, float(word.get("end") or cap))), 3)

    merged["start"] = round(float(words[0]["start"]), 3)
    merged["end"] = round(max(float(word.get("end") or word["start"]) for word in words), 3)
    merged["confidence"] = round(float(statistics.fmean(float(word.get("confidence") or 0.0) for word in words)), 4)
    merged["adaptive_consensus_words"] = consensus_used
    merged["tempo_reconstructed_words"] = reconstructed
    return merged


def _tempo_reconstruct_words(
    words: list[dict[str, Any]],
    *,
    unstable: set[int],
    waveform: np.ndarray,
    tempo_prior: float,
    line_start: float,
    right_boundary: float | None,
) -> int:
    if not words or not unstable:
        return 0
    reconstructed = 0
    sorted_indices = sorted(index for index in unstable if 0 <= index < len(words))
    blocks: list[tuple[int, int]] = []
    block_start = block_end = sorted_indices[0]
    for index in sorted_indices[1:]:
        if index == block_end + 1:
            block_end = index
        else:
            blocks.append((block_start, block_end))
            block_start = block_end = index
    blocks.append((block_start, block_end))

    for first, last in blocks:
        left_word = words[first - 1] if first > 0 else None
        right_word = words[last + 1] if last + 1 < len(words) else None
        if left_word and isinstance(left_word.get("end"), (int, float)):
            left = float(left_word["end"])
        elif left_word and isinstance(left_word.get("start"), (int, float)):
            left = float(left_word["start"]) + max(0.03, _word_units(left_word) * tempo_prior * 0.55)
        else:
            left = max(0.0, float(line_start))

        if right_word and isinstance(right_word.get("start"), (int, float)):
            right = float(right_word["start"])
        elif isinstance(right_boundary, (int, float)):
            right = float(right_boundary)
        else:
            right = left + sum(_word_units(words[i]) for i in range(first, last + 1)) * tempo_prior

        block_units = [_word_units(words[i]) for i in range(first, last + 1)]
        total_units = sum(block_units)
        expected_span = max(0.04 * len(block_units), total_units * tempo_prior)
        available = max(0.02 * len(block_units), right - left)

        # Between two acoustic anchors the interval is authoritative. At a free
        # edge use the learned tempo and leave any large silence untouched.
        if left_word is not None and right_word is not None:
            scale = available / max(total_units, 1e-6)
        else:
            scale = tempo_prior
            if available < expected_span:
                scale = available / max(total_units, 1e-6)

        cumulative = 0.0
        previous_start = left - 0.001
        for offset, index in enumerate(range(first, last + 1)):
            predicted = left + cumulative * scale
            predicted = _energy_onset_near(waveform, predicted)
            max_start = right - 0.012 * (last - index + 1)
            predicted = max(previous_start + 0.012, min(predicted, max_start))
            previous = dict(words[index])
            words[index]["start"] = round(predicted, 3)
            words[index]["confidence"] = round(min(0.44, max(0.24, float(previous.get("confidence") or 0.0) * 0.72 + 0.16)), 4)
            words[index]["origin"] = "tempo_energy_rescue"
            words[index]["rescue_previous_start"] = previous.get("start")
            reconstructed += 1
            cumulative += block_units[offset]
            previous_start = predicted

        for index in range(first, last + 1):
            start = float(words[index]["start"])
            if index + 1 < len(words):
                end = float(words[index + 1]["start"])
            else:
                end = right
            words[index]["end"] = round(max(start + 0.01, end), 3)
    return reconstructed


def _adaptive_rescue_line(
    waveform: np.ndarray,
    source_line: dict[str, Any],
    next_line: dict[str, Any] | None,
    next_anchor: float | None,
    current: dict[str, Any],
    *,
    reasons: list[str],
    tempo_prior: float | None,
    language: str,
    profile: AlignmentProfile,
    model: Any,
    token_dict: dict[str, int],
    device: str,
    romanizer: Any,
) -> dict[str, Any]:
    candidates = [current]
    # Different crops/context deliberately change the acoustic evidence seen by
    # MMS. Re-running the exact same deterministic emission would add no value.
    variants: list[tuple[str, dict[str, Any], float, float, int]] = []
    reanchor = float(current.get("start") or source_line.get("anchor_start") or source_line.get("start") or 0.0)
    tight_source = dict(source_line)
    tight_source["anchor_start"] = reanchor
    tight_source["start"] = reanchor
    variants.append((
        "adaptive-tight-reanchor",
        tight_source,
        max(0.58, profile.pre_margin * 0.62),
        max(1.00, profile.post_margin),
        max(1, profile.lookahead_words),
    ))
    variants.append((
        "adaptive-wide-context",
        source_line,
        profile.retry_pre_margin + (0.55 if profile.name == "max" else 0.30),
        profile.retry_post_margin + (0.75 if profile.name == "max" else 0.40),
        profile.lookahead_words + (1 if profile.name == "max" else 0),
    ))
    if profile.name == "max":
        variants.append((
            "adaptive-no-lookahead",
            source_line,
            profile.retry_pre_margin + 0.25,
            profile.retry_post_margin + 0.95,
            0,
        ))

    for label, variant_source, pre_margin, post_margin, lookahead in variants:
        try:
            candidate = _align_line_once(
                waveform,
                variant_source,
                next_anchor,
                next_line=next_line,
                lookahead_words=lookahead,
                language=language,
                model=model,
                token_dict=token_dict,
                device=device,
                pre_margin=pre_margin,
                post_margin=post_margin,
                last_line_window=profile.last_line_window + 4.0,
                method_suffix=label,
                romanizer=romanizer,
            )
            candidates.append(candidate)
        except Exception:
            continue

    if len(candidates) == 1:
        current["adaptive_rescue_checked"] = True
        current["adaptive_rescue_used"] = False
        current["adaptive_rescue_reasons"] = list(reasons)
        current["adaptive_candidate_count"] = 1
        return current

    boundary = current.get("lookahead_start")
    if not isinstance(boundary, (int, float)):
        boundary = next_anchor
    merged = _candidate_consensus(
        candidates,
        waveform=waveform,
        tempo_prior=tempo_prior,
        next_boundary=float(boundary) if isinstance(boundary, (int, float)) else None,
    )
    old_score = _alignment_candidate_score(current)
    new_score = _alignment_candidate_score(merged)
    changed = any(
        str(word.get("origin") or "").startswith(("adaptive_", "tempo_"))
        for word in (merged.get("words") or [])
    )
    # Consensus may lower raw confidence slightly while fixing a structurally
    # impossible phrase, so reasons + reconstructed words can override a tiny
    # score loss. Large degradations are rejected.
    accept = new_score >= old_score - 0.025 and changed
    if not accept:
        current["adaptive_rescue_checked"] = True
        current["adaptive_rescue_used"] = False
        current["adaptive_rescue_reasons"] = list(reasons)
        current["adaptive_candidate_count"] = len(candidates)
        current["adaptive_score_delta"] = round(new_score - old_score, 4)
        return current

    merged["adaptive_rescue_checked"] = True
    merged["adaptive_rescue_used"] = True
    merged["adaptive_rescue_reasons"] = list(reasons)
    merged["adaptive_candidate_count"] = len(candidates)
    merged["adaptive_score_delta"] = round(new_score - old_score, 4)
    merged["tempo_prior_ms_per_unit"] = round((tempo_prior or 0.0) * 1000.0, 2) if tempo_prior else None
    merged["method"] = f"{merged.get('method') or BACKEND_NAME}+adaptive-rescue"
    return merged


def _reconcile_display_switches(lines: list[dict[str, Any]], profile: AlignmentProfile) -> list[dict[str, Any]]:
    """Derive line-switch timestamps from both sides of every boundary.

    Word timestamps remain acoustic. `display_start` is allowed to be later than
    the first word when a single-line karaoke UI cannot display two genuinely
    overlapping phrases at once. This fixes the visible 'jump to next line while
    the previous line is still singing' without corrupting word alignment.
    """
    if not lines:
        return lines
    lines[0]["display_start"] = round(float(lines[0].get("start") or 0.0), 3)
    previous_display = float(lines[0]["display_start"])
    for index in range(1, len(lines)):
        previous = lines[index - 1]
        current = lines[index]
        current_words = current.get("words") or []
        previous_words = previous.get("words") or []
        if not current_words or not previous_words:
            current["display_start"] = round(float(current.get("start") or 0.0), 3)
            continue
        first = current_words[0]
        last = previous_words[-1]
        if not isinstance(first.get("start"), (int, float)) or not isinstance(last.get("start"), (int, float)):
            current["display_start"] = round(float(current.get("start") or 0.0), 3)
            continue
        acoustic_start = float(first["start"])
        previous_end = float(last.get("end") or last["start"])
        lookahead = previous.get("lookahead_start")
        lookahead_conf = float(previous.get("lookahead_confidence") or 0.0)
        first_conf = float(first.get("confidence") or 0.0)
        display = acoustic_start
        disagreement_ms = None
        if isinstance(lookahead, (int, float)):
            lookahead = float(lookahead)
            disagreement_ms = (acoustic_start - lookahead) * 1000.0
            # Two independent observations of the next-line onset. If they
            # agree, gently fuse them. If they disagree, trust the cleaner one.
            if abs(acoustic_start - lookahead) <= 0.18:
                display = _weighted_median([
                    (acoustic_start, 0.20 + first_conf * first_conf),
                    (lookahead, 0.20 + lookahead_conf * lookahead_conf),
                ])
            elif lookahead_conf >= first_conf + 0.06:
                display = lookahead
            elif acoustic_start < previous_end - 0.025 and lookahead >= previous_end - 0.035:
                display = lookahead

        overlap = previous_end - display
        delayed_for_previous = False
        if overlap > 0.025:
            # In a one-active-line UI the previous lyric should finish before
            # switching, unless both sides very confidently prove a real overlap.
            real_overlap = first_conf >= 0.84 and lookahead_conf >= 0.84 and overlap <= 0.16
            if not real_overlap:
                if profile.name == "max":
                    display = previous_end + 0.012
                else:
                    display = min(previous_end + 0.012, acoustic_start + 0.36)
                delayed_for_previous = display > acoustic_start + 0.01

        display = max(previous_display + 0.001, display)
        current["display_start"] = round(max(0.0, display), 3)
        previous_display = float(current["display_start"])
        current["line_switch_shift_ms"] = round((display - acoustic_start) * 1000.0, 1)
        current["cross_line_disagreement_ms"] = round(disagreement_ms, 1) if disagreement_ms is not None else None
        current["cross_line_reconciled"] = abs(display - acoustic_start) >= 0.020
        current["line_switch_delayed_for_previous"] = delayed_for_previous
    return lines

def _next_timed_line(source_lines: list[dict[str, Any]], index: int) -> tuple[dict[str, Any] | None, float | None]:
    for future in source_lines[index + 1:]:
        value = future.get("anchor_start")
        if not isinstance(value, (int, float)):
            value = future.get("start")
        if isinstance(value, (int, float)):
            return future, float(value)
    return None, None


def _next_anchor(source_lines: list[dict[str, Any]], index: int) -> float | None:
    return _next_timed_line(source_lines, index)[1]


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
        next_line, next_anchor = _next_timed_line(source_lines, index)
        if progress_callback:
            pct = 4 + round((order / total) * 90)
            progress_callback(pct, f"Boundary-aware alignment: строка {index + 1}/{len(source_lines)}")

        primary = _align_line_once(
            waveform, source, next_anchor,
            next_line=next_line,
            lookahead_words=profile.lookahead_words,
            language=language,
            model=model,
            token_dict=token_dict,
            device=device,
            pre_margin=profile.pre_margin,
            post_margin=profile.post_margin,
            last_line_window=profile.last_line_window,
            method_suffix="local+lookahead",
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
                    next_line=next_line,
                    lookahead_words=profile.lookahead_words,
                    language=language,
                    model=model,
                    token_dict=token_dict,
                    device=device,
                    pre_margin=profile.retry_pre_margin,
                    post_margin=profile.retry_post_margin,
                    last_line_window=profile.last_line_window + 3.0,
                    method_suffix="expanded-retry+lookahead",
                    romanizer=romanizer,
                )
                if _alignment_candidate_score(retry) > _alignment_candidate_score(primary) + 0.005:
                    best = retry
            except Exception:
                # A failed retry must never destroy a valid first pass.
                pass

        # Max-quality mode always verifies the cross-line boundary in a short
        # window. Balanced/fast do it only when the last word is suspicious.
        if (
            next_anchor is not None
            and next_line is not None
            and (
                profile.boundary_refine_all
                or _boundary_is_suspicious(best, next_anchor=next_anchor, profile=profile)
            )
        ):
            best = _refine_last_word_boundary(
                waveform,
                source,
                next_line,
                next_anchor,
                best,
                language=language,
                profile=profile,
                model=model,
                token_dict=token_dict,
                device=device,
                romanizer=romanizer,
            )
        result_map[index] = best

    # Second stage: only suspicious phrases pay for multi-pass alignment.
    # This is intentionally separate from the cheap first pass so an already
    # clean song does not get 3-4x slower.
    if retry_weak_lines and profile.name != "fast" and indices:
        preliminary = [result_map[i] for i in indices]
        adaptive_total = max(1, len(indices))
        for local_pos, index in enumerate(indices):
            current = result_map[index]
            next_line, next_anchor = _next_timed_line(source_lines, index)
            tempo_prior = _tempo_prior_for_index(preliminary, local_pos)
            reasons = _adaptive_line_reasons(
                current,
                profile=profile,
                tempo_prior=tempo_prior,
                next_anchor=next_anchor,
            )
            # Balanced mode rescues only clearly bad phrases. Max mode also
            # handles tempo/boundary inconsistencies that may still have a
            # deceptively high average CTC confidence.
            severe = any(reason in {
                "very_low_word_confidence",
                "consecutive_weak_words",
                "interpolated_word",
                "tempo_outlier",
            } for reason in reasons)
            if not reasons or reasons == ["boundary"] or (profile.name == "balanced" and not severe):
                continue
            if progress_callback:
                pct = 92 + round((local_pos / adaptive_total) * 6)
                progress_callback(
                    pct,
                    f"Adaptive rescue: строка {index + 1} ({', '.join(reasons[:3])})",
                )
            rescued = _adaptive_rescue_line(
                waveform,
                source_lines[index],
                next_line,
                next_anchor,
                current,
                reasons=reasons,
                tempo_prior=tempo_prior,
                language=language,
                profile=profile,
                model=model,
                token_dict=token_dict,
                device=device,
                romanizer=romanizer,
            )
            result_map[index] = rescued
            preliminary[local_pos] = rescued

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
    temp_dir = tempfile.TemporaryDirectory(prefix="lrc_extandator_v7_2_demucs_")
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
            "Forced Alignment v7.2 требует синхронизированный LRC с таймкодом каждой строки. "
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
            progress_callback(1, "Forced Alignment v7.2: CTC → boundary consensus → adaptive rescue → tempo fallback")
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
            mix_lines = _reconcile_display_switches(mix_lines, profile)
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
                or _boundary_is_suspicious(
                    line,
                    next_anchor=_next_anchor(source_lines, i),
                    profile=profile,
                )
                or int(line.get("tempo_reconstructed_words") or 0) > 0
                or bool(line.get("adaptive_rescue_used"))
            }
            weak_ratio = len(weak_indices) / max(1, len(mix_lines))
            hard_weak_indices = {
                i for i, line in enumerate(mix_lines)
                if int(line.get("tempo_reconstructed_words") or 0) > 0
                or (
                    bool(line.get("adaptive_rescue_used"))
                    and min(
                        [float(word.get("confidence") or 0.0) for word in (line.get("words") or [])] or [1.0]
                    ) < 0.42
                )
            }
            if options.use_demucs == "auto" and profile.name != "fast":
                should_demucs = (
                    float(mix_quality.get("score") or 0.0) < profile.demucs_trigger_score
                    or weak_ratio >= profile.demucs_trigger_weak_ratio
                    or (profile.name == "max" and bool(hard_weak_indices))
                )

        if should_demucs:
            if progress_callback:
                progress_callback(80, "Слабые фразы: Demucs даёт независимый вокальный кандидат, без ASR")
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

                vocal_lines = _reconcile_display_switches(vocal_lines, profile)
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
