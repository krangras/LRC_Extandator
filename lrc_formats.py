"""LRC / Enhanced LRC parsing, validation and serialization.

V7 keeps one lossless internal representation everywhere::

    {
        "start": float,
        "end": float,
        "text": str,
        "words": [
            {
                "word": str,
                "start": float,
                "end": float,
                # Optional provenance produced by the alignment engine:
                "confidence": float,
                "origin": "aligned" | "anchor_shifted" | "interpolated" | ...,
            }
        ],
        # Optional line-level provenance is preserved as well.
        "confidence": float,
        "method": str,
    }

The old implementation repaired timings by rebuilding dictionaries and therefore
silently discarded confidence/provenance metadata.  V7 repairs copies in-place
and preserves unknown keys so the editor, benchmarker and Better Lyrics bridge
can reason about alignment quality instead of receiving only pretty timestamps.
"""
from __future__ import annotations

import copy
import math
import re
from typing import Any, Iterable

TIME_BODY = r"(?P<minutes>\d{1,4}):(?P<seconds>[0-5]?\d)(?:[\.:](?P<fraction>\d{1,3}))?"
LEADING_TIME_RE = re.compile(rf"^\s*\[(?P<time>{TIME_BODY})\]")
INLINE_TIME_RE = re.compile(rf"(?:<|\[)(?P<time>{TIME_BODY})(?:>|\])")
META_RE = re.compile(r"^\s*\[([A-Za-z][\w-]*):(.*)\]\s*$")
SUPPORTED_META = ("ar", "ti", "al", "au", "lr", "length", "by", "re", "ve")


def _number(value: Any, fallback: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if math.isfinite(value) else fallback


def parse_time(value: str) -> float:
    match = re.fullmatch(r"\s*(\d{1,4}):([0-5]?\d)(?:[\.:](\d{1,3}))?\s*", value)
    if not match:
        raise ValueError(f"Некорректный таймкод: {value!r}")
    fraction = match.group(3) or ""
    fraction_seconds = int(fraction) / (10 ** len(fraction)) if fraction else 0.0
    return int(match.group(1)) * 60 + int(match.group(2)) + fraction_seconds


def format_time(seconds: float, precision: int = 3) -> str:
    precision = max(2, min(int(precision), 3))
    value = max(0.0, _number(seconds, 0.0))
    scale = 10**precision
    total_units = int(round(value * scale))
    minutes, remainder = divmod(total_units, 60 * scale)
    whole_seconds, fraction = divmod(remainder, scale)
    return f"{minutes:02d}:{whole_seconds:02d}.{fraction:0{precision}d}"


def _parse_inline(content: str, line_start: float) -> tuple[str, list[dict[str, Any]], float | None]:
    matches = list(INLINE_TIME_RE.finditer(content))
    if not matches:
        return content.strip(), [], None

    prefix = content[: matches[0].start()]
    words: list[dict[str, Any]] = []
    terminal: float | None = None
    visible_parts = [prefix]

    for index, match in enumerate(matches):
        start = parse_time(match.group("time"))
        end_pos = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        segment = content[match.end() : end_pos]
        visible_parts.append(segment)
        cleaned = segment.strip()
        if cleaned:
            words.append(
                {
                    "word": cleaned,
                    "start": start,
                    "end": None,
                    "confidence": 1.0,
                    "origin": "imported_elrc",
                }
            )
        else:
            terminal = start
            if words and words[-1].get("end") is None and start > words[-1]["start"]:
                words[-1]["end"] = start

    prefix_clean = prefix.strip()
    if prefix_clean:
        words.insert(
            0,
            {
                "word": prefix_clean,
                "start": line_start,
                "end": None,
                "confidence": 1.0,
                "origin": "imported_elrc",
            },
        )

    text = " ".join(part.strip() for part in visible_parts if part.strip())
    for index, word in enumerate(words):
        if word.get("end") is not None:
            continue
        if index + 1 < len(words):
            word["end"] = words[index + 1]["start"]
        elif terminal is not None and terminal > word["start"]:
            word["end"] = terminal
    return text, words, terminal


def parse_lyrics(text: str) -> dict[str, Any]:
    """Parse plain text, standard LRC, A2/foobar ELRC and metadata.

    Standard LRC lines intentionally keep ``words=[]``.  The line timestamp is
    an anchor, not fake word-level information.
    """
    metadata: dict[str, str] = {}
    raw_lines: list[dict[str, Any]] = []
    plain_lines: list[str] = []

    for source_line in (text or "").lstrip("\ufeff").splitlines():
        stripped = source_line.strip()
        if not stripped:
            continue
        meta = META_RE.match(stripped)
        if meta and not re.fullmatch(r"\d{1,4}", meta.group(1)):
            metadata[meta.group(1).lower()] = meta.group(2).strip()
            continue

        rest = source_line
        timestamps: list[float] = []
        while True:
            stamp = LEADING_TIME_RE.match(rest)
            if not stamp:
                break
            timestamps.append(parse_time(stamp.group("time")))
            rest = rest[stamp.end() :]

        if not timestamps:
            plain_lines.append(stripped)
            continue

        for line_start in timestamps:
            line_text, words, terminal = _parse_inline(rest, line_start)
            acoustic_start = min(
                [line_start] + [float(word["start"]) for word in words if isinstance(word.get("start"), (int, float))]
            ) if words else line_start
            raw_lines.append(
                {
                    "start": acoustic_start,
                    "display_start": line_start if words else None,
                    "end": terminal,
                    "text": line_text,
                    "words": words,
                    "anchor_start": line_start,
                    "anchor_origin": "input_lrc",
                    "confidence": 1.0 if words else 0.0,
                    "method": "imported_elrc" if words else "input_lrc_anchor",
                }
            )

    offset_ms = _number(metadata.get("offset"), 0.0)
    if offset_ms:
        shift = offset_ms / 1000.0
        for line in raw_lines:
            line["start"] += shift
            if isinstance(line.get("display_start"), (int, float)):
                line["display_start"] += shift
            line["anchor_start"] += shift
            if line.get("end") is not None:
                line["end"] += shift
            for word in line["words"]:
                word["start"] += shift
                if word.get("end") is not None:
                    word["end"] += shift
        metadata.pop("offset", None)

    raw_lines.sort(key=lambda item: item["start"])
    lines, issues = repair_lines(raw_lines, synthesize_word_times=False)
    return {"metadata": metadata, "lines": lines, "plain_lines": plain_lines, "issues": issues}


def repair_lines(
    lines: Iterable[dict[str, Any]],
    duration: float | None = None,
    *,
    synthesize_word_times: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return a safe monotonic copy while preserving quality metadata.

    ``synthesize_word_times=False`` is used while importing ordinary LRC.  It
    deliberately avoids inventing ELRC data.  The alignment engine is the only
    place that should synthesize missing word timings.
    """
    repaired = copy.deepcopy(list(lines or []))
    issues: list[str] = []
    duration_value = _number(duration, 0.0) if duration is not None else None

    previous_line_start = -0.001
    for index, line in enumerate(repaired):
        original_start = line.get("start")
        start = max(0.0, _number(original_start, previous_line_start + 0.001))
        if start < previous_line_start:
            issues.append(f"Строка {index + 1}: начало перемещено вперёд")
            start = previous_line_start + 0.001
        line["start"] = round(start, 3)
        previous_line_start = start

    previous_display = -0.001
    for index, line in enumerate(repaired):
        if isinstance(line.get("display_start"), (int, float)):
            display_start = max(float(line["display_start"]), previous_display + 0.001)
            line["display_start"] = round(display_start, 3)
            previous_display = display_start
        else:
            previous_display = max(previous_display, float(line["start"]))

        next_start = repaired[index + 1]["start"] if index + 1 < len(repaired) else None
        fallback_end = next_start
        if fallback_end is None:
            fallback_end = duration_value if duration_value and duration_value > line["start"] else line["start"] + 4.0
        explicit_end = _number(line.get("end"), fallback_end)
        line_end = max(line["start"] + 0.001, explicit_end)
        if next_start is not None:
            line_end = min(line_end, max(line["start"] + 0.001, next_start))

        raw_words = line.get("words") or []
        clean_words: list[dict[str, Any]] = []
        for word_index, original_word in enumerate(raw_words):
            word = copy.deepcopy(original_word)
            label = str(word.get("word") or "").strip()
            if not label:
                issues.append(f"Строка {index + 1}: удалён пустой фрагмент")
                continue
            word["word"] = label
            clean_words.append(word)

        if clean_words:
            has_any_real_time = any(isinstance(w.get("start"), (int, float)) for w in clean_words)
            if synthesize_word_times or has_any_real_time:
                available = max(0.04 * len(clean_words), line_end - line["start"])
                step = available / len(clean_words)
                previous_start = line["start"] - 0.001
                for word_index, word in enumerate(clean_words):
                    fallback_start = line["start"] + step * word_index
                    had_start = isinstance(word.get("start"), (int, float))
                    word_start = max(line["start"], _number(word.get("start"), fallback_start))
                    if word_start < previous_start:
                        issues.append(f"Строка {index + 1}, слово {word_index + 1}: исправлен порядок")
                        word_start = previous_start + 0.001
                        word["origin"] = word.get("origin") or "repaired"
                        word["confidence"] = min(_number(word.get("confidence"), 0.25), 0.25)
                    if not had_start:
                        word["origin"] = word.get("origin") or "interpolated"
                        word["confidence"] = min(_number(word.get("confidence"), 0.12), 0.12)
                    word["start"] = round(word_start, 3)
                    previous_start = word_start

                for word_index, word in enumerate(clean_words):
                    next_word_start = (
                        clean_words[word_index + 1]["start"]
                        if word_index + 1 < len(clean_words)
                        else max(line_end, word["start"] + 0.04)
                    )
                    had_end = isinstance(word.get("end"), (int, float))
                    proposed_end = _number(word.get("end"), next_word_start)
                    max_end = max(word["start"] + 0.001, next_word_start)
                    word_end = min(max_end, max(word["start"] + 0.001, proposed_end))
                    if not had_end:
                        word["origin"] = word.get("origin") or "interpolated"
                        word["confidence"] = min(_number(word.get("confidence"), 0.12), 0.12)
                    word["end"] = round(word_end, 3)

                line["start"] = min(line["start"], clean_words[0]["start"])
                line_end = max(line_end, clean_words[-1]["end"])
            line["text"] = " ".join(word["word"] for word in clean_words)
        else:
            line["text"] = str(line.get("text") or "").strip()

        line["words"] = clean_words
        line["end"] = round(line_end, 3)

    return repaired, issues


def _metadata_lines(metadata: dict[str, Any] | None) -> list[str]:
    meta = metadata or {}
    result = []
    keys = list(SUPPORTED_META) + sorted(key for key in meta if key not in SUPPORTED_META and key != "offset")
    seen = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        value = str(meta.get(key) or "").replace("\r", " ").replace("\n", " ").strip()
        if value:
            result.append(f"[{key}:{value}]")
    return result


def serialize_lrc(lines: Iterable[dict[str, Any]], metadata: dict[str, Any] | None = None, precision: int = 3) -> str:
    fixed, _ = repair_lines(lines)
    output = _metadata_lines(metadata)
    if output:
        output.append("")
    output.extend(
        f"[{format_time(line.get('display_start') if isinstance(line.get('display_start'), (int, float)) else line['start'], precision)}]{line.get('text', '')}"
        for line in fixed
    )
    return "\n".join(output)


def serialize_elrc(
    lines: Iterable[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    precision: int = 3,
    terminal_timestamps: bool = True,
) -> str:
    fixed, _ = repair_lines(lines)
    output = _metadata_lines(metadata)
    if output:
        output.append("")
    for line in fixed:
        switch_start = line.get("display_start") if isinstance(line.get("display_start"), (int, float)) else line["start"]
        line_tag = f"[{format_time(switch_start, precision)}]"
        words = line.get("words") or []
        if not words:
            output.append(f"{line_tag}{line.get('text', '')}")
            continue
        if terminal_timestamps:
            body = " ".join(
                f"<{format_time(word['start'], precision)}>{word['word']} <{format_time(word['end'], precision)}>"
                for word in words
            )
        else:
            body = " ".join(f"<{format_time(word['start'], precision)}>{word['word']}" for word in words)
        output.append(f"{line_tag}{body}")
    return "\n".join(output)


def build_outputs(lines: Iterable[dict[str, Any]], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    fixed, issues = repair_lines(lines)
    return {
        "lines": fixed,
        "lrc": serialize_lrc(fixed, metadata, precision=3),
        "elrc": serialize_elrc(fixed, metadata, precision=3, terminal_timestamps=True),
        "elrc_compatible": serialize_elrc(fixed, metadata, precision=2, terminal_timestamps=False),
        "issues": issues,
    }
