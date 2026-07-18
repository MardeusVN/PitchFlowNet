"""End-to-end, resumable offline preprocessing pipeline."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

from .audio import (
    AudioRejected,
    artifact_paths,
    atomic_torch_save,
    audio_quality,
    cache_key,
    extract_pitch,
    load_audio,
    sha256_file,
    spectrogram,
)
from .config import PreprocessConfig
from .manifest import (
    FailureRecord,
    ProcessedRecord,
    SourceRecord,
    assign_splits,
    atomic_write_json,
    atomic_write_jsonl,
    load_ljspeech,
)
from .text import (
    build_vocabulary,
    encode_phonemes,
    frontend_metadata,
    phonemize_batch,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerTask:
    source: SourceRecord
    phonemes: List[str]
    phoneme_ids: List[int]
    frontend_fingerprint: str
    output_dir: Path
    config: PreprocessConfig
    force: bool


WorkerResult = Union[ProcessedRecord, FailureRecord]


def validate_runtime_dependencies() -> None:
    missing = []
    for module_name in ("torch", "soundfile", "soxr", "pyworld"):
        try:
            __import__(module_name)
        except ImportError as exc:
            missing.append(f"{module_name} ({exc})")
    if missing:
        raise RuntimeError(
            "missing preprocessing dependencies; install src/python/requirements.txt: "
            + ", ".join(missing)
        )


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _cached_record_is_valid(record: ProcessedRecord, output_dir: Path) -> bool:
    return all(
        (output_dir / relative_path).is_file()
        for relative_path in (
            record.audio_path,
            record.spectrogram_path,
            record.pitch_path,
        )
    )


def process_one(task: WorkerTask) -> WorkerResult:
    source = task.source
    try:
        if not source.audio_path.is_file():
            raise AudioRejected("missing_audio", str(source.audio_path))
        if source.audio_path.stat().st_size == 0:
            raise AudioRejected("empty_audio_file", str(source.audio_path))

        source_sha = sha256_file(source.audio_path)
        key = cache_key(
            source_sha,
            source.utterance_id,
            source.text,
            f"{task.config.fingerprint}:{task.frontend_fingerprint}",
        )
        audio_path, spec_path, pitch_path, record_path = artifact_paths(
            task.output_dir, key
        )

        if not task.force and record_path.is_file():
            try:
                cached = ProcessedRecord.from_dict(
                    json.loads(record_path.read_text(encoding="utf-8"))
                )
                if cached.cache_key == key and _cached_record_is_valid(
                    cached, task.output_dir
                ):
                    return cached
            except Exception:
                # A malformed/partial sidecar is a cache miss, not a fatal error.
                pass

        audio = load_audio(source.audio_path, task.config.audio.sample_rate)
        quality = audio_quality(audio, task.config.audio.sample_rate, task.config)
        waveform, magnitude = spectrogram(audio, task.config)
        pitch = extract_pitch(audio, magnitude.shape[-1], task.config)

        num_frames = int(magnitude.shape[-1])
        if not (
            len(pitch["f0_hz"]) == len(pitch["log_f0"]) == len(pitch["voiced"]) == num_frames
        ):
            raise AudioRejected(
                "feature_length_mismatch",
                f"spectrogram={num_frames}, pitch={len(pitch['f0_hz'])}",
            )

        atomic_torch_save(waveform, audio_path)
        atomic_torch_save(magnitude, spec_path)
        atomic_torch_save(
            {
                "f0_hz": pitch["f0_hz"],
                "log_f0": pitch["log_f0"],
                "voiced": pitch["voiced"],
            },
            pitch_path,
        )

        pitch_summary = {
            "voiced_frames": float(pitch["voiced_frames"]),
            "voiced_ratio": float(pitch["voiced_frames"] / max(1, num_frames)),
            "log_f0_sum": float(pitch["log_f0_sum"]),
            "log_f0_squared_sum": float(pitch["log_f0_squared_sum"]),
        }
        record = ProcessedRecord(
            schema_version=task.config.schema_version,
            utterance_id=source.utterance_id,
            text=source.text,
            phonemes=task.phonemes,
            phoneme_ids=task.phoneme_ids,
            audio_path=_relative(audio_path, task.output_dir),
            spectrogram_path=_relative(spec_path, task.output_dir),
            pitch_path=_relative(pitch_path, task.output_dir),
            sample_rate=task.config.audio.sample_rate,
            num_samples=int(waveform.shape[-1]),
            num_spectrogram_frames=num_frames,
            duration_seconds=float(quality["duration_seconds"]),
            source_sha256=source_sha,
            cache_key=key,
            quality=quality,
            pitch=pitch_summary,
        )
        atomic_write_json(record_path, record.to_dict())
        return record
    except AudioRejected as exc:
        return FailureRecord(
            utterance_id=source.utterance_id,
            audio_path=str(source.audio_path),
            reason=exc.reason,
            detail=exc.detail,
        )
    except Exception as exc:
        return FailureRecord(
            utterance_id=source.utterance_id,
            audio_path=str(source.audio_path),
            reason=type(exc).__name__,
            detail=str(exc),
        )


def _run_tasks(tasks: Sequence[WorkerTask], max_workers: int):
    if max_workers == 1:
        for task in tasks:
            yield process_one(task)
        return

    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_initialize_worker
    ) as executor:
        yield from executor.map(process_one, tasks, chunksize=1)


def _initialize_worker() -> None:
    # Multiple process workers each creating a full CPU thread pool causes
    # severe oversubscription during STFT. One Torch thread per worker gives
    # predictable throughput; pyworld is already process-isolated.
    import torch

    torch.set_num_threads(1)


def _summary(records: Sequence[ProcessedRecord]) -> dict:
    durations = [record.duration_seconds for record in records]
    phoneme_lengths = [len(record.phoneme_ids) for record in records]
    total_voiced = sum(record.pitch["voiced_frames"] for record in records)
    total_log_f0 = sum(record.pitch["log_f0_sum"] for record in records)
    total_log_f0_squared = sum(
        record.pitch["log_f0_squared_sum"] for record in records
    )
    if total_voiced > 0:
        mean_log_f0 = total_log_f0 / total_voiced
        variance = max(
            total_log_f0_squared / total_voiced - mean_log_f0 * mean_log_f0,
            0.0,
        )
        std_log_f0 = math.sqrt(variance)
    else:
        mean_log_f0 = 0.0
        std_log_f0 = 1.0
    return {
        "duration_seconds": {
            "total": float(sum(durations)),
            "minimum": float(min(durations)),
            "maximum": float(max(durations)),
            "mean": float(sum(durations) / len(durations)),
        },
        "pitch": {
            "voiced_frames": int(total_voiced),
            "log_f0_mean": float(mean_log_f0),
            "log_f0_std": float(std_log_f0),
        },
        "phoneme_ids": {
            "minimum": int(min(phoneme_lengths)),
            "maximum": int(max(phoneme_lengths)),
            "mean": float(sum(phoneme_lengths) / len(phoneme_lengths)),
            "above_400": int(sum(length > 400 for length in phoneme_lengths)),
        },
    }


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    config: PreprocessConfig,
    max_workers: Optional[int] = None,
    force: bool = False,
    limit: Optional[int] = None,
) -> dict:
    config.validate()
    validate_runtime_dependencies()
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = load_ljspeech(input_dir)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        sources = sources[:limit]

    if max_workers is None or max_workers <= 0:
        max_workers = max(1, min(8, (os.cpu_count() or 2) // 2))

    _LOGGER.info("Phonemizing %d English utterances with eSpeak-ng", len(sources))
    phoneme_sequences, backend = phonemize_batch(
        [source.text for source in sources], config.text
    )
    vocabulary = build_vocabulary(phoneme_sequences)
    frontend = frontend_metadata(config.text, vocabulary, backend)
    tasks = [
        WorkerTask(
            source=source,
            phonemes=phonemes,
            phoneme_ids=encode_phonemes(phonemes, vocabulary),
            frontend_fingerprint=frontend["fingerprint"],
            output_dir=output_dir,
            config=config,
            force=force,
        )
        for source, phonemes in zip(sources, phoneme_sequences)
    ]

    _LOGGER.info(
        "Processing %d utterances with %d worker(s); VAD/trim disabled",
        len(tasks),
        max_workers,
    )
    started = time.perf_counter()
    records: List[ProcessedRecord] = []
    failures: List[FailureRecord] = []
    for index, result in enumerate(_run_tasks(tasks, max_workers), start=1):
        if isinstance(result, ProcessedRecord):
            records.append(result)
        else:
            failures.append(result)
        if index % 100 == 0 or index == len(tasks):
            _LOGGER.info(
                "Progress %d/%d (success=%d, failed=%d)",
                index,
                len(tasks),
                len(records),
                len(failures),
            )

    if not records:
        raise RuntimeError("preprocessing produced no valid records")
    records = assign_splits(
        records,
        config.split.num_validation,
        config.split.num_test,
        config.split.seed,
    )

    atomic_write_jsonl(output_dir / "manifest.jsonl", (r.to_dict() for r in records))
    for split in ("train", "validation", "test"):
        atomic_write_jsonl(
            output_dir / "splits" / f"{split}.jsonl",
            (record.to_dict() for record in records if record.split == split),
        )
    atomic_write_jsonl(
        output_dir / "failures.jsonl",
        (failure.__dict__ for failure in failures),
    )

    summary = _summary(records)
    split_counts = Counter(record.split for record in records)
    failure_counts = Counter(failure.reason for failure in failures)
    elapsed = time.perf_counter() - started
    dataset_config = {
        "schema_version": config.schema_version,
        "preprocessing": config.to_dict(),
        "preprocessing_fingerprint": config.fingerprint,
        "frontend": frontend,
        "feature_statistics": summary,
        "splits": dict(split_counts),
    }
    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_source_records": len(sources),
        "num_processed": len(records),
        "num_failed": len(failures),
        "failure_reasons": dict(failure_counts),
        "elapsed_seconds": elapsed,
        "vad_enabled": False,
        **summary,
    }
    atomic_write_json(output_dir / "config.json", dataset_config)
    atomic_write_json(output_dir / "report.json", report)
    return report
