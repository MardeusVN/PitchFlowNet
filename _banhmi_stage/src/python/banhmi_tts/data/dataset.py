"""Manifest-backed dataset with strict preprocessing contract validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch.utils.data import Dataset


class DataContractError(ValueError):
    """Raised when a manifest record or cached tensor violates schema v2."""


_REQUIRED_RECORD_FIELDS = {
    "schema_version",
    "utterance_id",
    "phoneme_ids",
    "audio_path",
    "spectrogram_path",
    "pitch_path",
    "sample_rate",
    "num_samples",
    "num_spectrogram_frames",
    "split",
}
_REQUIRED_PITCH_FIELDS = {
    "f0_hz",
    "log_f0",
    "log_f0_continuous",
    "voiced",
}


def _infer_dataset_root(manifest_path: Path) -> Path:
    for candidate in (manifest_path.parent, *manifest_path.parents):
        if (candidate / "config.json").is_file() and (candidate / "artifacts").is_dir():
            return candidate
    raise DataContractError(
        f"cannot infer preprocessed root from manifest: {manifest_path}"
    )


def _artifact_path(root: Path, relative_path: str, field: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DataContractError(f"{field} escapes preprocessed root: {relative_path}") from exc
    if not path.is_file():
        raise DataContractError(f"missing {field}: {path}")
    return path


def _load_tensor(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True)


class BanhmiDataset(Dataset):
    """Load schema-v2 Banhmi-TTS artifacts from a JSONL manifest.

    Records are kept in memory, while large tensors remain lazy and are loaded
    only by ``__getitem__``. This keeps dataset construction cheap for workers.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: Optional[str | Path] = None,
        expected_schema_version: int = 2,
        validate_paths: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.dataset_root = (
            Path(dataset_root).resolve()
            if dataset_root is not None
            else _infer_dataset_root(self.manifest_path)
        )
        self.expected_schema_version = expected_schema_version
        self.records = self._read_manifest(validate_paths=validate_paths)
        self.spectrogram_lengths = [
            int(record["num_spectrogram_frames"]) for record in self.records
        ]

    def _read_manifest(self, validate_paths: bool) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        seen_ids = set()
        with self.manifest_path.open(encoding="utf-8") as manifest_file:
            for line_number, line in enumerate(manifest_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataContractError(
                        f"invalid JSON at {self.manifest_path}:{line_number}"
                    ) from exc
                missing = _REQUIRED_RECORD_FIELDS - record.keys()
                if missing:
                    raise DataContractError(
                        f"record {line_number} missing fields: {sorted(missing)}"
                    )
                if int(record["schema_version"]) != self.expected_schema_version:
                    raise DataContractError(
                        f"record {line_number} uses schema {record['schema_version']}; "
                        f"expected {self.expected_schema_version}"
                    )
                utterance_id = str(record["utterance_id"])
                if utterance_id in seen_ids:
                    raise DataContractError(f"duplicate utterance id: {utterance_id}")
                seen_ids.add(utterance_id)
                phoneme_ids = record["phoneme_ids"]
                if not isinstance(phoneme_ids, list) or not phoneme_ids:
                    raise DataContractError(f"{utterance_id}: empty phoneme_ids")
                if any(not isinstance(value, int) or value < 0 for value in phoneme_ids):
                    raise DataContractError(f"{utterance_id}: invalid phoneme id")
                if int(record["num_samples"]) <= 0 or int(record["num_spectrogram_frames"]) <= 0:
                    raise DataContractError(f"{utterance_id}: invalid cached lengths")
                if validate_paths:
                    for field in ("audio_path", "spectrogram_path", "pitch_path"):
                        _artifact_path(self.dataset_root, str(record[field]), field)
                records.append(record)
        if not records:
            raise DataContractError(f"empty manifest: {self.manifest_path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        utterance_id = str(record["utterance_id"])
        audio = _load_tensor(
            _artifact_path(self.dataset_root, str(record["audio_path"]), "audio_path")
        )
        spectrogram = _load_tensor(
            _artifact_path(
                self.dataset_root,
                str(record["spectrogram_path"]),
                "spectrogram_path",
            )
        )
        pitch = _load_tensor(
            _artifact_path(self.dataset_root, str(record["pitch_path"]), "pitch_path")
        )
        return self._validate_and_pack(record, utterance_id, audio, spectrogram, pitch)

    @staticmethod
    def _validate_and_pack(
        record: Mapping[str, Any],
        utterance_id: str,
        audio: Any,
        spectrogram: Any,
        pitch: Any,
    ) -> Dict[str, Any]:
        if not isinstance(audio, torch.Tensor) or audio.ndim != 2 or audio.shape[0] != 1:
            raise DataContractError(f"{utterance_id}: audio must have shape [1, samples]")
        if not isinstance(spectrogram, torch.Tensor) or spectrogram.ndim != 2:
            raise DataContractError(f"{utterance_id}: spectrogram must have shape [bins, frames]")
        if not isinstance(pitch, dict):
            raise DataContractError(f"{utterance_id}: pitch artifact must be a dictionary")
        missing_pitch = _REQUIRED_PITCH_FIELDS - pitch.keys()
        if missing_pitch:
            raise DataContractError(
                f"{utterance_id}: pitch artifact missing {sorted(missing_pitch)}"
            )

        expected_samples = int(record["num_samples"])
        expected_frames = int(record["num_spectrogram_frames"])
        if audio.shape[-1] != expected_samples:
            raise DataContractError(
                f"{utterance_id}: audio frames={audio.shape[-1]}, expected={expected_samples}"
            )
        if spectrogram.shape[-1] != expected_frames:
            raise DataContractError(
                f"{utterance_id}: spectrogram frames={spectrogram.shape[-1]}, "
                f"expected={expected_frames}"
            )

        packed_pitch: Dict[str, torch.Tensor] = {}
        for field in sorted(_REQUIRED_PITCH_FIELDS):
            value = pitch[field]
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                raise DataContractError(f"{utterance_id}: {field} must have shape [frames]")
            if len(value) != expected_frames:
                raise DataContractError(
                    f"{utterance_id}: {field} frames={len(value)}, expected={expected_frames}"
                )
            packed_pitch[field] = value

        if not torch.isfinite(audio).all() or not torch.isfinite(spectrogram).all():
            raise DataContractError(f"{utterance_id}: audio/spectrogram contains NaN/Inf")
        for field in ("f0_hz", "log_f0", "log_f0_continuous"):
            if not torch.isfinite(packed_pitch[field]).all():
                raise DataContractError(f"{utterance_id}: {field} contains NaN/Inf")

        voiced = packed_pitch["voiced"].bool()
        if not torch.equal(packed_pitch["f0_hz"][~voiced], torch.zeros_like(packed_pitch["f0_hz"][~voiced])):
            raise DataContractError(f"{utterance_id}: unvoiced f0_hz must be zero")
        if not torch.equal(packed_pitch["log_f0"][~voiced], torch.zeros_like(packed_pitch["log_f0"][~voiced])):
            raise DataContractError(f"{utterance_id}: unvoiced log_f0 must be zero")

        return {
            "utterance_id": utterance_id,
            "phoneme_ids": torch.tensor(record["phoneme_ids"], dtype=torch.long),
            "audio": audio.float(),
            "spectrogram": spectrogram.float(),
            "f0_hz": packed_pitch["f0_hz"].float(),
            "log_f0": packed_pitch["log_f0"].float(),
            "log_f0_continuous": packed_pitch["log_f0_continuous"].float(),
            "voiced": voiced,
            "sample_rate": int(record["sample_rate"]),
        }
