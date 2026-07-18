"""Compare SDP inference durations with deterministic MAS validation targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from banhmi_tts.data import BanhmiCollator, BanhmiDataset
from banhmi_tts.duration_diagnostics import _token_kind, _token_name
from banhmi_tts.inference import load_inference_model, load_text_frontend
from banhmi_tts.model import decode_durations


def summarize_comparisons(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    """Summarize token-level target/prediction rows."""
    if not rows:
        raise ValueError("at least one duration comparison row is required")
    targets = [float(row["target_frames"]) for row in rows]
    predictions = [float(row["predicted_frames"]) for row in rows]
    errors = [prediction - target for target, prediction in zip(targets, predictions)]
    absolute_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    return {
        "token_count": len(rows),
        "target_mean_frames": statistics.fmean(targets),
        "predicted_mean_frames": statistics.fmean(predictions),
        "mean_error_frames": statistics.fmean(errors),
        "mae_frames": statistics.fmean(absolute_errors),
        "rmse_frames": math.sqrt(statistics.fmean(squared_errors)),
        "overprediction_fraction": sum(error > 0 for error in errors) / len(errors),
        "underprediction_fraction": sum(error < 0 for error in errors) / len(errors),
        "exact_fraction": sum(error == 0 for error in errors) / len(errors),
    }


def summarize_utterances(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    """Summarize utterance-level total duration comparisons."""
    if not rows:
        raise ValueError("at least one utterance comparison row is required")
    ratios = [float(row["predicted_to_target_ratio"]) for row in rows]
    errors = [float(row["duration_error_seconds"]) for row in rows]
    absolute_errors = [abs(error) for error in errors]
    return {
        "utterance_count": len(rows),
        "predicted_to_target_ratio_mean": statistics.fmean(ratios),
        "predicted_to_target_ratio_std": (
            statistics.pstdev(ratios) if len(ratios) > 1 else 0.0
        ),
        "predicted_to_target_ratio_min": min(ratios),
        "predicted_to_target_ratio_max": max(ratios),
        "duration_error_seconds_mean": statistics.fmean(errors),
        "duration_error_seconds_mae": statistics.fmean(absolute_errors),
    }


def grouped_summaries(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        group: summarize_comparisons(group_rows)
        for group, group_rows in sorted(groups.items())
    }


def _select_indices(length: int, count: int, seed: int) -> list[int]:
    if count <= 0:
        raise ValueError("num utterances must be positive")
    if count >= length:
        return list(range(length))
    generator = random.Random(seed)
    return sorted(generator.sample(range(length), count))


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _symbol_report_name(vocabulary_symbol: str) -> str:
    if vocabulary_symbol == " ":
        return "<space>"
    return _token_name(vocabulary_symbol)


@torch.inference_mode()
def compare_validation_durations(
    checkpoint: str | Path,
    dataset_directory: str | Path,
    *,
    noise_scales: Sequence[float],
    num_utterances: int,
    selection_seed: int,
    duration_seed: int,
    device: torch.device,
    progress_every: int = 10,
) -> dict[str, Any]:
    if not noise_scales or any(scale < 0.0 for scale in noise_scales):
        raise ValueError("duration noise scales must be non-negative")
    dataset_directory = Path(dataset_directory)
    manifest_path = dataset_directory / "splits" / "validation.jsonl"
    dataset = BanhmiDataset(manifest_path)
    indices = _select_indices(len(dataset), num_utterances, selection_seed)
    frontend = load_text_frontend(dataset_directory)
    loaded = load_inference_model(checkpoint, device, remove_weight_norm=False)
    model = loaded.model
    model.eval()
    predictor_config = loaded.model_config.predictor
    if predictor_config.duration_type != "stochastic":
        raise ValueError("validation duration diagnostic requires stochastic duration")

    id_to_symbol = {value: key for key, value in frontend.vocabulary.items()}
    collator = BanhmiCollator(sort_by_spectrogram_length=False)
    milliseconds_per_frame = 1000.0 * frontend.hop_length / frontend.sample_rate
    token_rows_by_scale: dict[float, list[dict[str, Any]]] = {
        float(scale): [] for scale in noise_scales
    }
    utterance_rows_by_scale: dict[float, list[dict[str, Any]]] = {
        float(scale): [] for scale in noise_scales
    }
    selected_utterance_ids: list[str] = []

    for selection_position, dataset_index in enumerate(indices):
        batch = _move_batch(collator([dataset[dataset_index]]), device)
        utterance_id = str(batch["utterance_ids"][0])
        selected_utterance_ids.append(utterance_id)
        text = model.text_encoder(batch["phoneme_ids"], batch["phoneme_lengths"])
        posterior = model.posterior_encoder(
            batch["spectrogram"],
            batch["spectrogram_lengths"],
            sample=False,
        )
        prior_space = model.flow(posterior.latent, posterior.mask)
        mas = model.mas(
            prior_space.latent,
            text.prior_mean,
            text.prior_log_scale,
            batch["spectrogram_lengths"],
            batch["phoneme_lengths"],
            global_step=loaded.global_step,
            add_noise=False,
        )
        text_length = int(batch["phoneme_lengths"][0].item())
        phoneme_ids = batch["phoneme_ids"][0, :text_length].tolist()
        target_durations = mas.durations[0, :text_length].cpu().tolist()
        target_total = int(sum(target_durations))
        expected_total = int(batch["spectrogram_lengths"][0].item())
        if target_total != expected_total:
            raise RuntimeError(
                f"{utterance_id}: MAS duration sum {target_total} != {expected_total}"
            )

        for scale_index, noise_scale in enumerate(noise_scales):
            scale = float(noise_scale)
            generator = torch.Generator(device=device)
            generator.manual_seed(
                duration_seed + selection_position * len(noise_scales) + scale_index
            )
            predicted_log_duration = model.duration_predictor(
                text.hidden,
                text.mask,
                reverse=True,
                noise_scale=scale,
                generator=generator,
            )
            predicted_durations_tensor = decode_durations(
                predicted_log_duration,
                text.mask.squeeze(1).bool(),
                maximum_log_duration=predictor_config.maximum_log_duration,
                maximum_duration_frames=predictor_config.maximum_duration_frames,
            )
            predicted_durations = predicted_durations_tensor[
                0, :text_length
            ].cpu().tolist()
            predicted_total = int(sum(predicted_durations))
            utterance_rows_by_scale[scale].append(
                {
                    "utterance_id": utterance_id,
                    "dataset_index": dataset_index,
                    "duration_noise_scale": scale,
                    "target_frames": target_total,
                    "predicted_frames": predicted_total,
                    "predicted_to_target_ratio": predicted_total
                    / max(target_total, 1),
                    "duration_error_seconds": (predicted_total - target_total)
                    * frontend.hop_length
                    / frontend.sample_rate,
                }
            )
            for token_index, (phoneme_id, target_frames, predicted_frames) in enumerate(
                zip(phoneme_ids, target_durations, predicted_durations)
            ):
                vocabulary_symbol = id_to_symbol.get(
                    int(phoneme_id), f"<id:{phoneme_id}>"
                )
                token_rows_by_scale[scale].append(
                    {
                        "utterance_id": utterance_id,
                        "dataset_index": dataset_index,
                        "duration_noise_scale": scale,
                        "token_index": token_index,
                        "symbol": _symbol_report_name(vocabulary_symbol),
                        "vocabulary_symbol": vocabulary_symbol,
                        "phoneme_id": int(phoneme_id),
                        "kind": _token_kind(vocabulary_symbol),
                        "target_frames": int(target_frames),
                        "predicted_frames": int(predicted_frames),
                        "error_frames": int(predicted_frames - target_frames),
                        "absolute_error_frames": abs(
                            int(predicted_frames - target_frames)
                        ),
                        "target_milliseconds": target_frames
                        * milliseconds_per_frame,
                        "predicted_milliseconds": predicted_frames
                        * milliseconds_per_frame,
                    }
                )
        if progress_every > 0 and (
            (selection_position + 1) % progress_every == 0
            or selection_position + 1 == len(indices)
        ):
            print(
                json.dumps(
                    {
                        "processed": selection_position + 1,
                        "total": len(indices),
                        "utterance_id": utterance_id,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )

    scale_reports: list[dict[str, Any]] = []
    for noise_scale in noise_scales:
        scale = float(noise_scale)
        token_rows = token_rows_by_scale[scale]
        symbol_summaries = grouped_summaries(token_rows, "symbol")
        sufficiently_frequent = [
            (symbol, summary)
            for symbol, summary in symbol_summaries.items()
            if int(summary["token_count"]) >= 10
        ]
        scale_reports.append(
            {
                "duration_noise_scale": scale,
                "overall_tokens": summarize_comparisons(token_rows),
                "overall_utterances": summarize_utterances(
                    utterance_rows_by_scale[scale]
                ),
                "by_kind": grouped_summaries(token_rows, "kind"),
                "selected_symbols": {
                    symbol: symbol_summaries[symbol]
                    for symbol in ("<space>", "<blank>", "s", "k")
                    if symbol in symbol_summaries
                },
                "symbols_by_absolute_mean_bias": [
                    {"symbol": symbol, **summary}
                    for symbol, summary in sorted(
                        sufficiently_frequent,
                        key=lambda item: abs(
                            float(item[1]["mean_error_frames"])
                        ),
                        reverse=True,
                    )[:20]
                ],
                "symbols": symbol_summaries,
                "utterances": utterance_rows_by_scale[scale],
                "tokens": token_rows,
            }
        )

    return {
        "checkpoint": str(Path(checkpoint)),
        "checkpoint_step": loaded.global_step,
        "manifest": str(manifest_path),
        "available_validation_utterances": len(dataset),
        "analyzed_utterances": len(indices),
        "selection_seed": selection_seed,
        "duration_seed": duration_seed,
        "posterior_sampling": False,
        "mas_noise": False,
        "sample_rate": frontend.sample_rate,
        "hop_length": frontend.hop_length,
        "milliseconds_per_frame": milliseconds_per_frame,
        "selected_utterance_ids": selected_utterance_ids,
        "noise_scales": scale_reports,
    }


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_validation_reports(
    report: Mapping[str, Any], output_directory: Path
) -> tuple[Path, Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    step = int(report["checkpoint_step"])
    json_path = output_directory / f"duration-validation-step-{step}.json"
    token_csv_path = output_directory / f"duration-validation-tokens-step-{step}.csv"
    utterance_csv_path = (
        output_directory / f"duration-validation-utterances-step-{step}.csv"
    )
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)

    token_rows = [
        row for scale in report["noise_scales"] for row in scale["tokens"]
    ]
    utterance_rows = [
        row for scale in report["noise_scales"] for row in scale["utterances"]
    ]
    _write_csv(token_csv_path, token_rows, list(token_rows[0].keys()))
    _write_csv(
        utterance_csv_path, utterance_rows, list(utterance_rows[0].keys())
    )
    return json_path, token_csv_path, utterance_csv_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare SDP duration predictions with validation MAS targets"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument(
        "--noise-scales", type=float, nargs="+", default=(0.0, 0.2, 0.4, 0.8)
    )
    parser.add_argument("--num-utterances", type=int, default=100)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--duration-seed", type=int, default=1234)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def main() -> None:
    args = _parser().parse_args()
    report = compare_validation_durations(
        args.checkpoint,
        args.dataset_dir,
        noise_scales=args.noise_scales,
        num_utterances=args.num_utterances,
        selection_seed=args.selection_seed,
        duration_seed=args.duration_seed,
        device=_resolve_device(args.device),
        progress_every=args.progress_every,
    )
    json_path, token_csv_path, utterance_csv_path = write_validation_reports(
        report, args.output_dir
    )
    print(
        json.dumps(
            {
                "checkpoint_step": report["checkpoint_step"],
                "analyzed_utterances": report["analyzed_utterances"],
                "json_report": str(json_path.resolve()),
                "token_csv": str(token_csv_path.resolve()),
                "utterance_csv": str(utterance_csv_path.resolve()),
                "noise_scales": [
                    {
                        "duration_noise_scale": scale["duration_noise_scale"],
                        "overall_tokens": scale["overall_tokens"],
                        "overall_utterances": scale["overall_utterances"],
                        "selected_symbols": scale["selected_symbols"],
                        "symbols_by_absolute_mean_bias": scale[
                            "symbols_by_absolute_mean_bias"
                        ][:10],
                    }
                    for scale in report["noise_scales"]
                ],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "compare_validation_durations",
    "grouped_summaries",
    "summarize_comparisons",
    "summarize_utterances",
    "write_validation_reports",
]
