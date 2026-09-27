"""Quality scoring for automatically generated word-level lyrics.

The score is intentionally conservative.  A syntactically valid ELRC is not
necessarily a good ELRC, so V7 keeps alignment provenance/confidence and uses
it to decide whether a result is trustworthy enough to auto-promote in clients
such as Better Lyrics.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

QUALITY_SCHEMA_VERSION = 2


def _num(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _pct(value: float) -> float:
    return round(_clamp01(value) * 100.0, 2)


def quality_report(lines: Iterable[dict[str, Any]]) -> dict[str, Any]:
    lines = list(lines or [])
    total_words = 0
    aligned_words = 0
    interpolated_words = 0
    repaired_words = 0
    word_confidences: list[float] = []
    line_confidences: list[float] = []
    anchor_deltas: list[float] = []
    non_monotonic = 0
    invalid_durations = 0
    boundary_warnings = 0
    boundary_rescues = 0
    last_word_confidences: list[float] = []
    line_diagnostics: list[dict[str, Any]] = []

    previous_line_start = -1.0
    for index, line in enumerate(lines):
        start = _num(line.get("start"), 0.0)
        if start < previous_line_start:
            non_monotonic += 1
        previous_line_start = max(previous_line_start, start)

        line_conf = _clamp01(_num(line.get("confidence"), 0.0))
        if line_conf > 0:
            line_confidences.append(line_conf)
        if isinstance(line.get("anchor_delta_ms"), (int, float)):
            anchor_deltas.append(abs(float(line["anchor_delta_ms"])) / 1000.0)

        words = line.get("words") or []
        local_interpolated = 0
        local_aligned = 0
        local_conf: list[float] = []
        prev_word_start = start - 1e-6
        for word in words:
            total_words += 1
            origin = str(word.get("origin") or "unknown")
            conf = _clamp01(_num(word.get("confidence"), 0.0))
            local_conf.append(conf)
            word_confidences.append(conf)

            if origin.startswith("interpolated"):
                interpolated_words += 1
                local_interpolated += 1
            elif origin.startswith("repaired"):
                repaired_words += 1
            elif origin not in {"unknown", "synthetic"}:
                aligned_words += 1
                local_aligned += 1

            ws = _num(word.get("start"), start)
            we = _num(word.get("end"), ws)
            if ws < prev_word_start:
                non_monotonic += 1
            if we <= ws:
                invalid_durations += 1
            prev_word_start = max(prev_word_start, ws)

        last_word_conf = local_conf[-1] if local_conf else 0.0
        if local_conf:
            last_word_confidences.append(last_word_conf)
        lead_ms = line.get("boundary_lead_ms")
        context_gap_ms = line.get("boundary_context_gap_ms")
        local_boundary_warning = False
        if isinstance(lead_ms, (int, float)) and float(lead_ms) < 25.0:
            local_boundary_warning = True
        if isinstance(context_gap_ms, (int, float)) and float(context_gap_ms) < 45.0:
            local_boundary_warning = True
        if local_boundary_warning:
            boundary_warnings += 1
        if bool(line.get("boundary_rescue_used")):
            boundary_rescues += 1

        line_diagnostics.append(
            {
                "line": index + 1,
                "text": str(line.get("text") or ""),
                "confidence": round(line_conf, 4),
                "words": len(words),
                "alignedWords": local_aligned,
                "interpolatedWords": local_interpolated,
                "meanWordConfidence": round(statistics.fmean(local_conf), 4) if local_conf else 0.0,
                "lastWordConfidence": round(last_word_conf, 4),
                "method": line.get("method") or "unknown",
                "anchorDeltaMs": round(_num(line.get("anchor_delta_ms"), 0.0), 1)
                if line.get("anchor_delta_ms") is not None
                else None,
                "boundaryLeadMs": round(float(lead_ms), 1) if isinstance(lead_ms, (int, float)) else None,
                "boundaryContextGapMs": round(float(context_gap_ms), 1) if isinstance(context_gap_ms, (int, float)) else None,
                "boundaryRescueUsed": bool(line.get("boundary_rescue_used")),
                "lookaheadWordsUsed": int(line.get("lookahead_words_used") or 0),
                "boundaryWarning": local_boundary_warning,
            }
        )

    aligned_ratio = aligned_words / total_words if total_words else 0.0
    interpolated_ratio = interpolated_words / total_words if total_words else 0.0
    repaired_ratio = repaired_words / total_words if total_words else 0.0
    mean_word_conf = statistics.fmean(word_confidences) if word_confidences else 0.0
    mean_line_conf = statistics.fmean(line_confidences) if line_confidences else 0.0
    mean_last_word_conf = statistics.fmean(last_word_confidences) if last_word_confidences else 0.0

    if anchor_deltas:
        anchor_mae = statistics.fmean(anchor_deltas)
        # 0s => 1.0, 0.5s => ~0.78, 2s => ~0.37, 5s => ~0.08
        anchor_consistency = math.exp(-anchor_mae / 2.0)
    else:
        anchor_mae = None
        anchor_consistency = 0.75  # neutral when there is no trusted LRC anchor

    structural_penalty = min(0.35, non_monotonic * 0.035 + invalid_durations * 0.02)
    interpolation_penalty = min(0.35, interpolated_ratio * 0.55 + repaired_ratio * 0.25)
    boundary_ratio = boundary_warnings / max(1, len(lines))
    boundary_penalty = min(0.18, boundary_ratio * 0.24)

    score = (
        0.34 * aligned_ratio
        + 0.24 * mean_word_conf
        + 0.16 * mean_line_conf
        + 0.12 * mean_last_word_conf
        + 0.14 * anchor_consistency
        - structural_penalty
        - interpolation_penalty
        - boundary_penalty
    )
    score = _clamp01(score)

    if score >= 0.90 and interpolated_ratio <= 0.08:
        grade = "excellent"
    elif score >= 0.80 and interpolated_ratio <= 0.18:
        grade = "good"
    elif score >= 0.68:
        grade = "usable"
    else:
        grade = "reject"

    publishable = grade in {"excellent", "good"} and non_monotonic == 0 and invalid_durations == 0

    return {
        "schemaVersion": QUALITY_SCHEMA_VERSION,
        "score": round(score, 4),
        "grade": grade,
        "publishable": publishable,
        "totalLines": len(lines),
        "totalWords": total_words,
        "alignedWords": aligned_words,
        "interpolatedWords": interpolated_words,
        "repairedWords": repaired_words,
        "alignedWordRatio": round(aligned_ratio, 4),
        "interpolatedWordRatio": round(interpolated_ratio, 4),
        "meanWordConfidence": round(mean_word_conf, 4),
        "meanLineConfidence": round(mean_line_conf, 4),
        "meanLastWordConfidence": round(mean_last_word_conf, 4),
        "boundaryWarnings": boundary_warnings,
        "boundaryRescues": boundary_rescues,
        "boundaryWarningRatio": round(boundary_ratio, 4),
        "anchorMaeMs": round(anchor_mae * 1000.0, 1) if anchor_mae is not None else None,
        "nonMonotonicViolations": non_monotonic,
        "invalidWordDurations": invalid_durations,
        "lineDiagnostics": line_diagnostics,
        "percent": {
            "aligned": _pct(aligned_ratio),
            "interpolated": _pct(interpolated_ratio),
        },
    }


def compare_quality(a: dict[str, Any], b: dict[str, Any]) -> int:
    """Return 1 when a is preferable, -1 when b is preferable, else 0.

    The score is primary.  Ties prefer fewer interpolated words and then faster
    candidates (if runtimeSec is supplied by the caller).
    """
    sa, sb = _num(a.get("score")), _num(b.get("score"))
    if abs(sa - sb) > 1e-6:
        return 1 if sa > sb else -1
    ia, ib = _num(a.get("interpolatedWordRatio"), 1.0), _num(b.get("interpolatedWordRatio"), 1.0)
    if abs(ia - ib) > 1e-6:
        return 1 if ia < ib else -1
    ra, rb = _num(a.get("runtimeSec"), 1e9), _num(b.get("runtimeSec"), 1e9)
    if abs(ra - rb) > 1e-6:
        return 1 if ra < rb else -1
    return 0
