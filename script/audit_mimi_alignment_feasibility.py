"""Audit whether low-rate Mimi frames can support hard phoneme MAS.

This is a preflight tool: it does not require Mimi or PyTorch.  It estimates
the 12.5 Hz codec length from each manifest row's measured audio duration and
reports hard-MAS feasibility under several text-token policies.  Floor, round,
and ceil estimates are all reported so the conclusion does not depend on the
codec's exact end-padding convention.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Iterable


NON_DURATION_IPA_MARKS = {
    "\u02c8",  # primary stress
    "\u02cc",  # secondary stress
    "\u02d0",  # length mark; modifies the preceding phone
    "\u02d1",  # half-length mark
}
PUNCTUATION_RUN = re.compile(r"[,;:.!?]+")
ENGLISH_COMPOSITE_PHONES = {
    ("a", "\u026a"),  # aI
    ("a", "\u028a"),  # aU
    ("e", "\u026a"),  # eI
    ("o", "\u028a"),  # oU
    ("\u0254", "\u026a"),  # OI
    ("t", "\u0283"),  # tS
    ("d", "\u0292"),  # dZ
    ("\u026a", "\u025a"),  # centering/rhotic diphthong
    ("\u028a", "\u025a"),
}


def is_duration_phone(symbol: str) -> bool:
    """Return whether an espeak symbol should own duration by itself."""

    if not symbol or symbol.isspace() or symbol in NON_DURATION_IPA_MARKS:
        return False
    # Combining diacritics modify the preceding phone and should not receive
    # a separate positive duration under an unchanged VITS2 SDP.
    if all(unicodedata.category(character).startswith("M") for character in symbol):
        return False
    return True


def count_english_phone_clusters(symbols: list[str]) -> int:
    """Estimate phoneme units by merging common English multi-symbol phones.

    The current frontend stores IPA codepoints, so affricates and diphthongs
    occupy multiple entries.  This policy is deliberately reported as a
    heuristic alternative, not silently substituted for the existing token
    definition.
    """

    count = 0
    word_phones: list[str] = []

    def consume_word() -> int:
        clusters = 0
        index = 0
        while index < len(word_phones):
            if (
                index + 1 < len(word_phones)
                and (word_phones[index], word_phones[index + 1])
                in ENGLISH_COMPOSITE_PHONES
            ):
                index += 2
            else:
                index += 1
            clusters += 1
        return clusters

    for symbol in symbols:
        if symbol.isspace():
            count += consume_word()
            word_phones.clear()
        elif is_duration_phone(symbol):
            word_phones.append(symbol)
    count += consume_word()
    return count


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return float("nan")
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "minimum": ordered[0],
        "p01": percentile(ordered, 0.01),
        "p05": percentile(ordered, 0.05),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "maximum": ordered[-1],
    }


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def frame_count(duration: float, frame_rate: float, mode: str) -> int:
    value = duration * frame_rate
    if mode == "floor":
        return max(1, math.floor(value))
    if mode == "round":
        return max(1, round(value))
    if mode == "ceil":
        return max(1, math.ceil(value))
    raise ValueError(f"Unknown frame-count mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frame-rate", type=float, default=12.5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worst", type=int, default=25)
    args = parser.parse_args()

    rows = load_rows(args.manifest)
    policies = {
        # Current model sequence: BOS + interspersed blank + EOS.
        "full_phoneme_ids": lambda row: len(row["phoneme_ids"]),
        # Raw espeak symbols still include spaces, stress and length marks.
        "raw_espeak_symbols": lambda row: len(row["phonemes"]),
        # Phones that can own positive duration under the unchanged SDP.
        "duration_phones": lambda row: sum(
            is_duration_phone(symbol) for symbol in row["phonemes"]
        ),
        "english_phone_clusters": lambda row: count_english_phone_clusters(
            row["phonemes"]
        ),
        # Same phones plus one explicit pause per punctuation run.
        "duration_phones_plus_punctuation": lambda row: sum(
            is_duration_phone(symbol) for symbol in row["phonemes"]
        )
        + len(PUNCTUATION_RUN.findall(row.get("text", ""))),
        "english_phone_clusters_plus_punctuation": lambda row: count_english_phone_clusters(
            row["phonemes"]
        )
        + len(PUNCTUATION_RUN.findall(row.get("text", ""))),
    }

    report: dict[str, object] = {
        "manifest": str(args.manifest.resolve()),
        "num_utterances": len(rows),
        "codec_frame_rate": args.frame_rate,
        "note": (
            "Frame counts are duration-based estimates. Ceil is the optimistic "
            "bound; a policy failing under ceil cannot be rescued by floor/round."
        ),
        "policies": {},
    }

    for policy_name, token_counter in policies.items():
        token_counts = [max(1, token_counter(row)) for row in rows]
        policy_result: dict[str, object] = {
            "token_count": summarize(token_counts),
            "frame_estimates": {},
        }
        for mode in ("floor", "round", "ceil"):
            codec_frames = [
                frame_count(float(row["duration_seconds"]), args.frame_rate, mode)
                for row in rows
            ]
            slacks = [frames - tokens for frames, tokens in zip(codec_frames, token_counts)]
            ratios = [frames / tokens for frames, tokens in zip(codec_frames, token_counts)]
            infeasible_indices = [index for index, slack in enumerate(slacks) if slack < 0]
            worst_indices = sorted(range(len(rows)), key=lambda index: (slacks[index], ratios[index]))[
                : args.worst
            ]
            policy_result["frame_estimates"][mode] = {
                "codec_frames": summarize(codec_frames),
                "slack": summarize(slacks),
                "frames_per_token": summarize(ratios),
                "num_infeasible": len(infeasible_indices),
                "percent_infeasible": 100.0 * len(infeasible_indices) / len(rows),
                "num_exactly_saturated": sum(slack == 0 for slack in slacks),
                "percent_below_1_25_frames_per_token": 100.0
                * sum(ratio < 1.25 for ratio in ratios)
                / len(rows),
                "worst_examples": [
                    {
                        "utterance_id": rows[index].get("utterance_id"),
                        "duration_seconds": rows[index].get("duration_seconds"),
                        "codec_frames": codec_frames[index],
                        "text_tokens": token_counts[index],
                        "slack": slacks[index],
                        "frames_per_token": ratios[index],
                        "text": rows[index].get("text"),
                    }
                    for index in worst_indices
                ],
            }
        report["policies"][policy_name] = policy_result

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
