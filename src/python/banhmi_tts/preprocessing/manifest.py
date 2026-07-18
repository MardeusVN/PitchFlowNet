"""Input and output manifest contracts."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class SourceRecord:
    utterance_id: str
    text: str
    audio_path: Path


@dataclass(frozen=True)
class ProcessedRecord:
    schema_version: int
    utterance_id: str
    text: str
    phonemes: List[str]
    phoneme_ids: List[int]
    audio_path: str
    spectrogram_path: str
    pitch_path: str
    sample_rate: int
    num_samples: int
    num_spectrogram_frames: int
    duration_seconds: float
    source_sha256: str
    cache_key: str
    quality: Dict[str, float]
    pitch: Dict[str, float]
    split: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(value: Dict[str, Any]) -> "ProcessedRecord":
        return ProcessedRecord(**value)


@dataclass(frozen=True)
class FailureRecord:
    utterance_id: str
    audio_path: str
    reason: str
    detail: str


def load_ljspeech(input_dir: Path) -> List[SourceRecord]:
    metadata_path = input_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing metadata file: {metadata_path}")

    wav_dir = input_dir / "wavs"
    if not wav_dir.is_dir():
        wav_dir = input_dir / "wav"
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"missing wav/wavs directory below {input_dir}")

    records: List[SourceRecord] = []
    seen_ids = set()
    with metadata_path.open("r", encoding="utf-8", newline="") as metadata_file:
        reader = csv.reader(metadata_file, delimiter="|", quoting=csv.QUOTE_NONE)
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(
                    f"metadata line {line_number} has {len(row)} columns; expected 2"
                )
            utterance_id, text = row[0].strip(), row[1].strip()
            if not utterance_id:
                raise ValueError(f"metadata line {line_number} has an empty id")
            if utterance_id in seen_ids:
                raise ValueError(f"duplicate utterance id: {utterance_id}")
            if not text:
                raise ValueError(f"metadata line {line_number} has empty text")
            seen_ids.add(utterance_id)
            records.append(SourceRecord(utterance_id, text, wav_dir / f"{utterance_id}.wav"))

    if not records:
        raise ValueError(f"no records found in {metadata_path}")
    return records


def assign_splits(
    records: Sequence[ProcessedRecord], num_validation: int, num_test: int, seed: int
) -> List[ProcessedRecord]:
    if num_validation + num_test >= len(records):
        raise ValueError(
            "validation + test sizes must leave at least one training utterance"
        )

    import random

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    test_indices = set(indices[:num_test])
    validation_indices = set(indices[num_test : num_test + num_validation])

    assigned: List[ProcessedRecord] = []
    for index, record in enumerate(records):
        split = "test" if index in test_indices else "validation" if index in validation_indices else "train"
        value = record.to_dict()
        value["split"] = split
        assigned.append(ProcessedRecord.from_dict(value))
    return assigned


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output_file:
            json.dump(value, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_write_jsonl(path: Path, values: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output_file:
            for value in values:
                json.dump(value, output_file, ensure_ascii=False, separators=(",", ":"))
                output_file.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
