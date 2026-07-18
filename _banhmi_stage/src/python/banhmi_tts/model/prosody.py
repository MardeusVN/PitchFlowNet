"""MAS-guided frame-to-phoneme pitch and voicing targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ProsodyTargetConfig:
    voiced_label_threshold: float = 0.5
    validation_tolerance: float = 1e-4

    def validate(self) -> None:
        if not 0.0 <= self.voiced_label_threshold <= 1.0:
            raise ValueError("voiced_label_threshold must be in [0, 1]")
        if self.validation_tolerance < 0.0:
            raise ValueError("validation_tolerance cannot be negative")


@dataclass
class PhonemeProsodyTargets:
    durations: torch.Tensor
    continuous_log_f0: torch.Tensor
    voiced_log_f0: torch.Tensor
    voiced_ratio: torch.Tensor
    voiced: torch.Tensor
    voiced_frame_counts: torch.Tensor
    mask: torch.Tensor


def normalize_log_f0(
    values: torch.Tensor,
    mask: torch.Tensor,
    mean: float,
    standard_deviation: float,
) -> torch.Tensor:
    """Normalize log-F0 while preserving exact zero in padded positions."""
    if standard_deviation <= 0.0:
        raise ValueError("log-F0 standard deviation must be positive")
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    return ((values - mean) / standard_deviation) * mask.to(values.dtype)


class PhonemeProsodyTargetBuilder(nn.Module):
    """Aggregate frame targets with the hard monotonic MAS alignment.

    Continuous log-F0 is averaged over every aligned frame. Original voiced
    log-F0 is averaged only over voiced frames. Voicing remains available both
    as a soft frame ratio and a thresholded label.
    """

    def __init__(self, config: ProsodyTargetConfig = ProsodyTargetConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config

    def forward(
        self,
        alignment: torch.Tensor,
        log_f0: torch.Tensor,
        log_f0_continuous: torch.Tensor,
        voiced: torch.Tensor,
        acoustic_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> PhonemeProsodyTargets:
        if alignment.ndim != 4 or alignment.shape[1] != 1:
            raise ValueError("alignment must have shape [batch, 1, acoustic, text]")
        weights = alignment.squeeze(1)
        batch, acoustic_time, text_time = weights.shape
        expected_frames = (batch, acoustic_time)
        if any(value.shape != expected_frames for value in (log_f0, log_f0_continuous, voiced)):
            raise ValueError("frame-level F0 and voiced tensors must match alignment")
        if acoustic_lengths.shape != (batch,) or text_lengths.shape != (batch,):
            raise ValueError("length tensors must have shape [batch]")
        if not all(value.is_floating_point() for value in (weights, log_f0, log_f0_continuous)):
            raise ValueError("alignment and F0 tensors must use floating-point dtypes")
        if not all(torch.isfinite(value).all() for value in (weights, log_f0, log_f0_continuous)):
            raise ValueError("prosody target inputs contain NaN/Inf")

        acoustic_lengths = acoustic_lengths.to(device=weights.device, dtype=torch.long)
        text_lengths = text_lengths.to(device=weights.device, dtype=torch.long)
        acoustic_positions = torch.arange(acoustic_time, device=weights.device)
        text_positions = torch.arange(text_time, device=weights.device)
        acoustic_mask = acoustic_positions.unsqueeze(0) < acoustic_lengths.unsqueeze(1)
        text_mask = text_positions.unsqueeze(0) < text_lengths.unsqueeze(1)
        expected_frame_assignments = acoustic_mask.to(weights.dtype)
        actual_frame_assignments = weights.sum(dim=2)
        if not torch.allclose(
            actual_frame_assignments,
            expected_frame_assignments,
            atol=self.config.validation_tolerance,
            rtol=0.0,
        ):
            raise ValueError("alignment must assign every valid frame exactly once")
        if torch.any(weights < -self.config.validation_tolerance):
            raise ValueError("alignment weights cannot be negative")

        weights = weights * acoustic_mask.unsqueeze(2).to(weights.dtype)
        durations_float = weights.sum(dim=1)
        if torch.any(durations_float[text_mask] < 1.0 - self.config.validation_tolerance):
            raise ValueError("every valid text token must receive at least one frame")

        voiced_float = voiced.to(dtype=weights.dtype) * acoustic_mask.to(weights.dtype)
        continuous_sum = torch.bmm(
            log_f0_continuous.unsqueeze(1), weights
        ).squeeze(1)
        voiced_frame_counts = torch.bmm(
            voiced_float.unsqueeze(1), weights
        ).squeeze(1)
        voiced_log_sum = torch.bmm(
            (log_f0 * voiced_float).unsqueeze(1), weights
        ).squeeze(1)

        safe_durations = durations_float.clamp_min(1.0)
        safe_voiced_counts = voiced_frame_counts.clamp_min(1.0)
        continuous_target = continuous_sum / safe_durations
        voiced_target = voiced_log_sum / safe_voiced_counts
        voiced_target = torch.where(
            voiced_frame_counts > 0.0,
            voiced_target,
            torch.zeros_like(voiced_target),
        )
        voiced_ratio = voiced_frame_counts / safe_durations
        mask_float = text_mask.to(weights.dtype)
        continuous_target = continuous_target * mask_float
        voiced_target = voiced_target * mask_float
        voiced_ratio = voiced_ratio * mask_float
        voiced_label = (
            voiced_ratio >= self.config.voiced_label_threshold
        ) & text_mask
        durations = durations_float.round().long() * text_mask.long()
        return PhonemeProsodyTargets(
            durations=durations,
            continuous_log_f0=continuous_target,
            voiced_log_f0=voiced_target,
            voiced_ratio=voiced_ratio,
            voiced=voiced_label,
            voiced_frame_counts=voiced_frame_counts * mask_float,
            mask=text_mask,
        )
