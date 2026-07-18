"""Synchronized latent, prosody, and waveform segment slicing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class SegmentConfig:
    segment_frames: int = 32
    hop_length: int = 256

    def validate(self) -> None:
        if self.segment_frames <= 0 or self.hop_length <= 0:
            raise ValueError("segment_frames and hop_length must be positive")

    @property
    def segment_samples(self) -> int:
        return self.segment_frames * self.hop_length


@dataclass
class TrainingSegments:
    latent: torch.Tensor
    normalized_log_f0: torch.Tensor
    voicing: torch.Tensor
    audio: torch.Tensor
    frame_starts: torch.Tensor
    sample_starts: torch.Tensor


def _slice_3d(
    values: torch.Tensor, starts: torch.Tensor, segment_length: int
) -> torch.Tensor:
    segments = []
    for batch_index, start in enumerate(starts.tolist()):
        segments.append(values[batch_index, :, start : start + segment_length])
    return torch.stack(segments)


def slice_training_segments(
    latent: torch.Tensor,
    normalized_log_f0: torch.Tensor,
    voicing: torch.Tensor,
    audio: torch.Tensor,
    acoustic_lengths: torch.Tensor,
    config: SegmentConfig = SegmentConfig(),
    *,
    generator: Optional[torch.Generator] = None,
    frame_starts: Optional[torch.Tensor] = None,
) -> TrainingSegments:
    """Slice synchronized frame/audio segments for windowed generator training."""
    config.validate()
    if latent.ndim != 3:
        raise ValueError("latent must have shape [batch, channels, frames]")
    batch, _, maximum_frames = latent.shape
    if normalized_log_f0.shape != (batch, maximum_frames):
        raise ValueError("normalized_log_f0 must match latent frames")
    if voicing.shape != (batch, maximum_frames):
        raise ValueError("voicing must match latent frames")
    if audio.ndim != 3 or audio.shape[:2] != (batch, 1):
        raise ValueError("audio must have shape [batch, 1, samples]")
    if acoustic_lengths.shape != (batch,):
        raise ValueError("acoustic_lengths must have shape [batch]")
    acoustic_lengths = acoustic_lengths.to(device=latent.device, dtype=torch.long)
    if torch.any(acoustic_lengths < config.segment_frames):
        raise ValueError("every utterance must be at least segment_frames long")
    if torch.any(acoustic_lengths > maximum_frames):
        raise ValueError("acoustic length exceeds latent frames")

    maximum_starts = acoustic_lengths - config.segment_frames
    if frame_starts is None:
        random_values = torch.rand(
            batch, device=latent.device, generator=generator
        )
        frame_starts = torch.floor(
            random_values * (maximum_starts + 1).to(random_values.dtype)
        ).long()
    else:
        frame_starts = frame_starts.to(device=latent.device, dtype=torch.long)
        if frame_starts.shape != (batch,):
            raise ValueError("frame_starts must have shape [batch]")
        if torch.any(frame_starts < 0) or torch.any(frame_starts > maximum_starts):
            raise ValueError("frame start lies outside a valid utterance")

    sample_starts = frame_starts * config.hop_length
    required_audio_ends = sample_starts + config.segment_samples
    if torch.any(required_audio_ends > audio.shape[-1]):
        raise ValueError("waveform is too short for the selected synchronized segment")
    return TrainingSegments(
        latent=_slice_3d(latent, frame_starts, config.segment_frames),
        normalized_log_f0=_slice_3d(
            normalized_log_f0.unsqueeze(1), frame_starts, config.segment_frames
        ).squeeze(1),
        voicing=_slice_3d(
            voicing.to(latent.dtype).unsqueeze(1), frame_starts, config.segment_frames
        ).squeeze(1),
        audio=_slice_3d(audio, sample_starts, config.segment_samples),
        frame_starts=frame_starts,
        sample_starts=sample_starts,
    )
