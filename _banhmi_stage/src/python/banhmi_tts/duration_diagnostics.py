"""Inspect token durations sampled by a trained duration predictor."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from banhmi_tts.inference import (
    EncodedText,
    LoadedInferenceModel,
    TextFrontend,
    load_inference_model,
    load_text_frontend,
)
from banhmi_tts.model import decode_durations
from banhmi_tts.preprocessing.text import BOS, EOS, PAD, UNK


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def _token_name(symbol: str) -> str:
    return "<blank>" if symbol == PAD else symbol


def _token_kind(symbol: str) -> str:
    if symbol == PAD:
        return "blank"
    if symbol in {BOS, EOS, UNK}:
        return "special"
    return "phoneme"


def build_token_rows(
    phoneme_ids: Sequence[int],
    id_to_symbol: Mapping[int, str],
    predicted_log_durations: Sequence[float],
    durations: Sequence[int],
    *,
    milliseconds_per_frame: float,
    maximum_duration_frames: int,
) -> list[dict[str, Any]]:
    """Build a human-readable token timeline for one duration sample."""
    if not (
        len(phoneme_ids) == len(predicted_log_durations) == len(durations)
    ):
        raise ValueError("duration diagnostic sequences must have equal lengths")

    rows: list[dict[str, Any]] = []
    cumulative_frames = 0
    for index, (phoneme_id, log_duration, frame_count) in enumerate(
        zip(phoneme_ids, predicted_log_durations, durations)
    ):
        symbol = id_to_symbol.get(int(phoneme_id), f"<id:{phoneme_id}>")
        start_frame = cumulative_frames
        cumulative_frames += int(frame_count)
        rows.append(
            {
                "token_index": index,
                "symbol": _token_name(symbol),
                "vocabulary_symbol": symbol,
                "phoneme_id": int(phoneme_id),
                "kind": _token_kind(symbol),
                "predicted_log_duration": float(log_duration),
                "frames": int(frame_count),
                "milliseconds": float(frame_count) * milliseconds_per_frame,
                "start_milliseconds": start_frame * milliseconds_per_frame,
                "end_milliseconds": cumulative_frames * milliseconds_per_frame,
                "at_maximum_duration": int(frame_count)
                >= maximum_duration_frames,
            }
        )
    return rows


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize total and per-token variation for one noise scale."""
    if not samples:
        raise ValueError("at least one duration sample is required")

    totals = [float(sample["total_frames"]) for sample in samples]
    seconds = [float(sample["duration_seconds"]) for sample in samples]
    blank_ratios = [float(sample["blank_frame_ratio"]) for sample in samples]
    word_boundary_ratios = [
        float(sample["word_boundary_frame_ratio"]) for sample in samples
    ]
    token_count = len(samples[0]["tokens"])
    if any(len(sample["tokens"]) != token_count for sample in samples):
        raise ValueError("all duration samples must have the same token count")

    per_token: list[dict[str, Any]] = []
    for index in range(token_count):
        reference = samples[0]["tokens"][index]
        frames = [float(sample["tokens"][index]["frames"]) for sample in samples]
        per_token.append(
            {
                "token_index": index,
                "symbol": reference["symbol"],
                "kind": reference["kind"],
                "mean_frames": statistics.fmean(frames),
                "std_frames": statistics.pstdev(frames) if len(frames) > 1 else 0.0,
                "min_frames": int(min(frames)),
                "max_frames": int(max(frames)),
            }
        )

    def distribution(values: Sequence[float]) -> dict[str, float]:
        return {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    return {
        "sample_count": len(samples),
        "total_frames": distribution(totals),
        "duration_seconds": distribution(seconds),
        "blank_frame_ratio": distribution(blank_ratios),
        "word_boundary_frame_ratio": distribution(word_boundary_ratios),
        "tokens_by_mean_duration": sorted(
            per_token, key=lambda row: row["mean_frames"], reverse=True
        )[:15],
        "tokens_by_duration_variation": sorted(
            per_token, key=lambda row: row["std_frames"], reverse=True
        )[:15],
        "per_token": per_token,
    }


@torch.inference_mode()
def diagnose_durations(
    loaded: LoadedInferenceModel,
    frontend: TextFrontend,
    text: str,
    *,
    noise_scales: Sequence[float],
    seed_start: int,
    num_seeds: int,
    length_scale: float,
) -> dict[str, Any]:
    if not noise_scales or any(scale < 0.0 for scale in noise_scales):
        raise ValueError("duration noise scales must be non-negative")
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive")
    if length_scale <= 0.0:
        raise ValueError("length_scale must be positive")
    if len(frontend.vocabulary) != loaded.model_config.text_encoder.num_symbols:
        raise ValueError("dataset vocabulary does not match checkpoint")

    device = next(loaded.model.parameters()).device
    encoded: EncodedText = frontend.encode(text)
    phoneme_ids, phoneme_lengths = encoded.tensors(device)
    text_output = loaded.model.text_encoder(phoneme_ids, phoneme_lengths)
    valid_mask = text_output.mask.squeeze(1).bool()
    id_to_symbol = {value: key for key, value in frontend.vocabulary.items()}
    milliseconds_per_frame = 1000.0 * frontend.hop_length / frontend.sample_rate
    predictor_config = loaded.model_config.predictor
    stochastic = predictor_config.duration_type == "stochastic"

    scale_reports: list[dict[str, Any]] = []
    for noise_scale in noise_scales:
        # At zero noise, SDP output is deterministic; duplicate seeds add no data.
        seeds = [seed_start] if stochastic and noise_scale == 0.0 else [
            seed_start + offset for offset in range(num_seeds)
        ]
        if not stochastic:
            seeds = [seed_start]

        samples: list[dict[str, Any]] = []
        for seed in seeds:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            if stochastic:
                predicted_log_duration = loaded.model.duration_predictor(
                    text_output.hidden,
                    text_output.mask,
                    reverse=True,
                    noise_scale=float(noise_scale),
                    generator=generator,
                )
            else:
                predicted_log_duration = loaded.model.duration_predictor(
                    text_output.hidden, text_output.mask
                )
            durations = decode_durations(
                predicted_log_duration,
                valid_mask,
                length_scale=length_scale,
                maximum_log_duration=predictor_config.maximum_log_duration,
                maximum_duration_frames=predictor_config.maximum_duration_frames,
            )
            token_rows = build_token_rows(
                encoded.phoneme_ids,
                id_to_symbol,
                predicted_log_duration[0].detach().cpu().tolist(),
                durations[0].detach().cpu().tolist(),
                milliseconds_per_frame=milliseconds_per_frame,
                maximum_duration_frames=predictor_config.maximum_duration_frames,
            )
            total_frames = int(durations[0].sum().item())
            blank_frames = sum(
                row["frames"] for row in token_rows if row["kind"] == "blank"
            )
            word_boundary_frames = sum(
                row["frames"]
                for row in token_rows
                if row["vocabulary_symbol"] == " "
            )
            samples.append(
                {
                    "seed": seed,
                    "total_frames": total_frames,
                    "duration_seconds": total_frames
                    * frontend.hop_length
                    / frontend.sample_rate,
                    "blank_frames": blank_frames,
                    "blank_frame_ratio": blank_frames / max(total_frames, 1),
                    "word_boundary_frames": word_boundary_frames,
                    "word_boundary_frame_ratio": word_boundary_frames
                    / max(total_frames, 1),
                    "tokens_at_maximum_duration": sum(
                        bool(row["at_maximum_duration"]) for row in token_rows
                    ),
                    "tokens": token_rows,
                }
            )
        scale_reports.append(
            {
                "duration_noise_scale": float(noise_scale),
                "summary": summarize_samples(samples),
                "samples": samples,
            }
        )

    return {
        "checkpoint": str(loaded.checkpoint_path),
        "checkpoint_step": loaded.global_step,
        "duration_predictor_type": predictor_config.duration_type,
        "text": text,
        "phonemes": "".join(encoded.phonemes),
        "phoneme_ids": list(encoded.phoneme_ids),
        "unknown_phonemes": list(encoded.unknown_phonemes),
        "sample_rate": frontend.sample_rate,
        "hop_length": frontend.hop_length,
        "milliseconds_per_frame": milliseconds_per_frame,
        "length_scale": length_scale,
        "seed_start": seed_start,
        "requested_num_seeds": num_seeds,
        "noise_scales": scale_reports,
    }


def write_reports(report: Mapping[str, Any], output_directory: Path) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    step = int(report["checkpoint_step"])
    json_path = output_directory / f"duration-diagnostics-step-{step}.json"
    csv_path = output_directory / f"duration-diagnostics-step-{step}.csv"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)

    fieldnames = [
        "duration_noise_scale",
        "seed",
        "token_index",
        "symbol",
        "vocabulary_symbol",
        "phoneme_id",
        "kind",
        "predicted_log_duration",
        "frames",
        "milliseconds",
        "start_milliseconds",
        "end_milliseconds",
        "at_maximum_duration",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for scale_report in report["noise_scales"]:
            for sample in scale_report["samples"]:
                for token in sample["tokens"]:
                    writer.writerow(
                        {
                            "duration_noise_scale": scale_report[
                                "duration_noise_scale"
                            ],
                            "seed": sample["seed"],
                            **token,
                        }
                    )
    return json_path, csv_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose predicted token durations")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--noise-scales", type=float, nargs="+", default=(0.0, 0.2, 0.4, 0.8)
    )
    parser.add_argument("--seed-start", type=int, default=1234)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--length-scale", type=float, default=1.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    device = _resolve_device(args.device)
    frontend = load_text_frontend(args.dataset_dir)
    loaded = load_inference_model(
        args.checkpoint, device, remove_weight_norm=False
    )
    report = diagnose_durations(
        loaded,
        frontend,
        args.text,
        noise_scales=args.noise_scales,
        seed_start=args.seed_start,
        num_seeds=args.num_seeds,
        length_scale=args.length_scale,
    )
    json_path, csv_path = write_reports(report, args.output_dir)
    console_summary = {
        "checkpoint_step": report["checkpoint_step"],
        "json_report": str(json_path.resolve()),
        "csv_report": str(csv_path.resolve()),
        "noise_scales": [
            {
                "duration_noise_scale": scale["duration_noise_scale"],
                **{
                    key: scale["summary"][key]
                    for key in (
                        "sample_count",
                        "total_frames",
                        "duration_seconds",
                        "blank_frame_ratio",
                        "word_boundary_frame_ratio",
                        "tokens_by_mean_duration",
                        "tokens_by_duration_variation",
                    )
                },
            }
            for scale in report["noise_scales"]
        ],
    }
    print(json.dumps(console_summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "build_token_rows",
    "diagnose_durations",
    "summarize_samples",
    "write_reports",
]
