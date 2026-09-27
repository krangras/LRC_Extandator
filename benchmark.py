"""Benchmark LRC Extandator against hand-corrected ELRC ground truth.

Usage:
    python benchmark.py score reference.elrc candidate.elrc
    python benchmark.py run benchmarks/manifest.json --out benchmark-results.json

A benchmark dataset measures the *whole pipeline*, not just a model.  It lets us
compare mix/Demucs passes, quality profiles, future acoustic models, parser
changes and post-processing with the same songs and the same reference timing.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

from alignment_engine import AlignmentEngine, AlignmentOptions
from lrc_formats import parse_lyrics


def _norm(value: str) -> str:
    return "".join(ch.casefold() for ch in str(value or "") if ch.isalnum() or ch == "'")


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def _ms(value: float | None) -> float | None:
    return round(value * 1000.0, 2) if value is not None else None


def score_lines(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    line_errors: list[float] = []
    word_start_errors: list[float] = []
    word_end_errors: list[float] = []
    matched_words = 0
    reference_words = 0
    line_text_mismatches = 0

    max_lines = max(len(reference), len(candidate))
    for index in range(max_lines):
        if index >= len(reference) or index >= len(candidate):
            line_text_mismatches += 1
            if index < len(reference):
                reference_words += len(reference[index].get("words") or [])
            continue
        ref = reference[index]
        cand = candidate[index]
        if _norm(ref.get("text", "")) != _norm(cand.get("text", "")):
            # Line order/text mismatch is much more severe than a timing error.
            line_text_mismatches += 1
        line_errors.append(abs(float(cand.get("start", 0.0)) - float(ref.get("start", 0.0))))

        ref_words = ref.get("words") or []
        cand_words = cand.get("words") or []
        reference_words += len(ref_words)
        cursor = 0
        for rw in ref_words:
            token = _norm(rw.get("word", ""))
            if not token:
                continue
            match_idx = None
            for j in range(cursor, min(len(cand_words), cursor + 5)):
                if _norm(cand_words[j].get("word", "")) == token:
                    match_idx = j
                    break
            if match_idx is None:
                continue
            cw = cand_words[match_idx]
            cursor = match_idx + 1
            matched_words += 1
            word_start_errors.append(abs(float(cw.get("start", 0.0)) - float(rw.get("start", 0.0))))
            word_end_errors.append(abs(float(cw.get("end", 0.0)) - float(rw.get("end", 0.0))))

    def rate_under(values: list[float], threshold: float) -> float:
        return round(sum(1 for v in values if v <= threshold + 1e-9) / len(values), 4) if values else 0.0

    coverage = matched_words / reference_words if reference_words else 0.0
    onset_mae = statistics.fmean(word_start_errors) if word_start_errors else None
    end_mae = statistics.fmean(word_end_errors) if word_end_errors else None
    line_mae = statistics.fmean(line_errors) if line_errors else None

    return {
        "referenceLines": len(reference),
        "candidateLines": len(candidate),
        "lineTextMismatches": line_text_mismatches,
        "referenceWords": reference_words,
        "matchedWords": matched_words,
        "wordCoverage": round(coverage, 4),
        "lineOnsetMaeMs": _ms(line_mae),
        "wordOnsetMaeMs": _ms(onset_mae),
        "wordOnsetMedianMs": _ms(_percentile(word_start_errors, 0.50)),
        "wordOnsetP95Ms": _ms(_percentile(word_start_errors, 0.95)),
        "wordEndMaeMs": _ms(end_mae),
        "wordEndP95Ms": _ms(_percentile(word_end_errors, 0.95)),
        "within50ms": rate_under(word_start_errors, 0.050),
        "within100ms": rate_under(word_start_errors, 0.100),
        "within200ms": rate_under(word_start_errors, 0.200),
    }


def score_files(reference_path: str | Path, candidate_path: str | Path) -> dict[str, Any]:
    reference = parse_lyrics(Path(reference_path).read_text(encoding="utf-8"))["lines"]
    candidate = parse_lyrics(Path(candidate_path).read_text(encoding="utf-8"))["lines"]
    return score_lines(reference, candidate)


DEFAULT_MATRIX = [
    {"name": "mix-fast", "use_demucs": False, "quality_mode": "fast"},
    {"name": "mix-balanced", "use_demucs": False, "quality_mode": "balanced"},
    {"name": "auto-max", "use_demucs": "auto", "quality_mode": "max"},
    {"name": "vocals-balanced", "use_demucs": True, "quality_mode": "balanced"},
]



def run_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    cases = manifest.get("cases") or []
    matrix = manifest.get("matrix") or DEFAULT_MATRIX
    engine = AlignmentEngine()
    results: list[dict[str, Any]] = []

    for case in cases:
        name = case["name"]
        audio = (base / case["audio"]).resolve()
        lyrics = (base / case["lyrics"]).resolve()
        reference = (base / case["reference"]).resolve()
        language = case.get("language", "mul")
        parsed_source = parse_lyrics(lyrics.read_text(encoding="utf-8"))
        source_lines = parsed_source["lines"]
        if not source_lines:
            raise ValueError(f"Benchmark lyrics must be timed LRC: {lyrics}")
        ref_lines = parse_lyrics(reference.read_text(encoding="utf-8"))["lines"]

        for config in matrix:
            options = AlignmentOptions(
                use_demucs=config.get("use_demucs", "auto"),
                quality_mode=config.get("quality_mode", "balanced"),
                use_cache=False,
            )
            started = time.monotonic()
            generated = engine.align(str(audio), source_lines, language=language, options=options)
            runtime = time.monotonic() - started
            metrics = score_lines(ref_lines, generated["lines"])
            results.append(
                {
                    "case": name,
                    "config": config.get("name") or json.dumps(config, sort_keys=True),
                    "runtimeSec": round(runtime, 3),
                    "predictedQuality": generated.get("quality", {}),
                    "metrics": metrics,
                }
            )

    by_config: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_config.setdefault(row["config"], []).append(row)

    aggregate = []
    for config, rows in by_config.items():
        onset = [r["metrics"]["wordOnsetMaeMs"] for r in rows if r["metrics"]["wordOnsetMaeMs"] is not None]
        p95 = [r["metrics"]["wordOnsetP95Ms"] for r in rows if r["metrics"]["wordOnsetP95Ms"] is not None]
        coverage = [r["metrics"]["wordCoverage"] for r in rows]
        aggregate.append(
            {
                "config": config,
                "cases": len(rows),
                "meanWordOnsetMaeMs": round(statistics.fmean(onset), 2) if onset else None,
                "meanWordOnsetP95Ms": round(statistics.fmean(p95), 2) if p95 else None,
                "meanWordCoverage": round(statistics.fmean(coverage), 4) if coverage else 0.0,
                "meanRuntimeSec": round(statistics.fmean(r["runtimeSec"] for r in rows), 3),
            }
        )
    aggregate.sort(key=lambda row: (row["meanWordOnsetMaeMs"] is None, row["meanWordOnsetMaeMs"] or 1e9))
    return {"manifest": str(manifest_path), "results": results, "aggregate": aggregate}


def main() -> int:
    parser = argparse.ArgumentParser(description="LRC Extandator Forced Alignment V7 benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    p_score = sub.add_parser("score", help="Compare candidate ELRC with hand-corrected ground truth")
    p_score.add_argument("reference")
    p_score.add_argument("candidate")
    p_run = sub.add_parser("run", help="Run a whole benchmark manifest/config matrix")
    p_run.add_argument("manifest")
    p_run.add_argument("--out", default="benchmark-results.json")
    args = parser.parse_args()

    if args.command == "score":
        print(json.dumps(score_files(args.reference, args.candidate), ensure_ascii=False, indent=2))
        return 0

    result = run_manifest(Path(args.manifest).resolve())
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"Full report: {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
