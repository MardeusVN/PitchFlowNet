"""Validated, serializable preprocessing configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22_050
    channels: int = 1

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels != 1:
            raise ValueError("Banhmi-TTS currently supports mono audio only")


@dataclass(frozen=True)
class SpectrogramConfig:
    n_fft: int = 1_024
    hop_length: int = 256
    win_length: int = 1_024
    center: bool = False

    def validate(self) -> None:
        if min(self.n_fft, self.hop_length, self.win_length) <= 0:
            raise ValueError("spectrogram dimensions must be positive")
        if self.win_length > self.n_fft:
            raise ValueError("win_length cannot exceed n_fft")
        if self.n_fft < self.hop_length:
            raise ValueError("n_fft must be greater than or equal to hop_length")


@dataclass(frozen=True)
class PitchConfig:
    extractor: str = "world-dio-stonemask"
    f0_floor: float = 50.0
    f0_ceil: float = 600.0

    def validate(self) -> None:
        if self.extractor != "world-dio-stonemask":
            raise ValueError(f"unsupported pitch extractor: {self.extractor}")
        if self.f0_floor <= 0 or self.f0_ceil <= self.f0_floor:
            raise ValueError("invalid F0 range")


@dataclass(frozen=True)
class QualityConfig:
    min_duration_seconds: float = 0.5
    max_duration_seconds: float = 30.0
    min_rms_dbfs: float = -55.0
    clipping_threshold: float = 0.999
    max_clipping_ratio: float = 0.01
    silence_threshold_dbfs: float = -50.0
    silence_frame_ms: float = 20.0

    def validate(self) -> None:
        if self.min_duration_seconds <= 0:
            raise ValueError("min_duration_seconds must be positive")
        if self.max_duration_seconds <= self.min_duration_seconds:
            raise ValueError("max_duration_seconds must exceed minimum duration")
        if not 0 < self.clipping_threshold <= 1:
            raise ValueError("clipping_threshold must be in (0, 1]")
        if not 0 <= self.max_clipping_ratio <= 1:
            raise ValueError("max_clipping_ratio must be in [0, 1]")
        if self.silence_frame_ms <= 0:
            raise ValueError("silence_frame_ms must be positive")


@dataclass(frozen=True)
class TextConfig:
    language: str = "en-us"
    phoneme_type: str = "espeak"

    def validate(self) -> None:
        if self.language not in {"en", "en-us", "en-gb"}:
            raise ValueError("Banhmi-TTS preprocessing is currently English-only")
        if self.phoneme_type != "espeak":
            raise ValueError("only eSpeak phonemization is currently supported")


@dataclass(frozen=True)
class SplitConfig:
    num_validation: int = 100
    num_test: int = 500
    seed: int = 1_234

    def validate(self) -> None:
        if self.num_validation < 0 or self.num_test < 0:
            raise ValueError("split sizes cannot be negative")


@dataclass(frozen=True)
class PreprocessConfig:
    schema_version: int = SCHEMA_VERSION
    audio: AudioConfig = field(default_factory=AudioConfig)
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    text: TextConfig = field(default_factory=TextConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version: {self.schema_version}")
        self.audio.validate()
        self.spectrogram.validate()
        self.pitch.validate()
        self.quality.validate()
        self.text.validate()
        self.split.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @staticmethod
    def from_dict(value: Dict[str, Any]) -> "PreprocessConfig":
        config = PreprocessConfig(
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
            audio=AudioConfig(**value.get("audio", {})),
            spectrogram=SpectrogramConfig(**value.get("spectrogram", {})),
            pitch=PitchConfig(**value.get("pitch", {})),
            quality=QualityConfig(**value.get("quality", {})),
            text=TextConfig(**value.get("text", {})),
            split=SplitConfig(**value.get("split", {})),
        )
        config.validate()
        return config
