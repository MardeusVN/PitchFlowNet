"""Alignment-based token-to-frame expansion and VITS KL objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .predictors import ProsodyPredictorOutput


@dataclass
class FrameExpansionOutput:
    prior_mean: torch.Tensor
    prior_log_scale: torch.Tensor
    normalized_continuous_log_f0: torch.Tensor
    voicing_logits: torch.Tensor
    voicing_probability: torch.Tensor
    mask: torch.Tensor


@dataclass
class DurationPath:
    """Hard monotonic token-to-frame path decoded from token durations."""

    alignment: torch.Tensor
    frame_mask: torch.Tensor
    frame_lengths: torch.Tensor


def durations_to_path(
    durations: torch.Tensor,
    text_mask: torch.Tensor,
) -> DurationPath:
    """Build the VITS inference alignment from integer token durations.

    ``durations`` is ``[B, T_text]`` and ``text_mask`` is
    ``[B, 1, T_text]``. Every generated frame is assigned to exactly one
    valid token; padded tokens always receive zero frames.
    """
    if durations.ndim != 2:
        raise ValueError("durations must have shape [batch, text_time]")
    if text_mask.shape != (durations.shape[0], 1, durations.shape[1]):
        raise ValueError("text_mask must have shape [batch, 1, text_time]")
    if durations.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("durations must use an integer dtype")
    valid_tokens = text_mask.squeeze(1).bool()
    if torch.any(durations < 0):
        raise ValueError("durations cannot be negative")
    if torch.any(durations[~valid_tokens] != 0):
        raise ValueError("padded tokens must have zero duration")
    if torch.any(durations[valid_tokens] < 1):
        raise ValueError("valid tokens must have at least one frame")

    frame_lengths = durations.sum(dim=1)
    if torch.any(frame_lengths < 1):
        raise ValueError("every sequence must contain at least one frame")
    maximum_frames = int(frame_lengths.max().item())
    frame_positions = torch.arange(
        maximum_frames, device=durations.device
    ).view(1, 1, maximum_frames)
    ends = durations.cumsum(dim=1).unsqueeze(-1)
    starts = (ends.squeeze(-1) - durations).unsqueeze(-1)
    assigned = (frame_positions >= starts) & (frame_positions < ends)
    assigned = assigned & valid_tokens.unsqueeze(-1)
    alignment = assigned.transpose(1, 2).unsqueeze(1).to(text_mask.dtype)
    frame_mask = (
        torch.arange(maximum_frames, device=durations.device).unsqueeze(0)
        < frame_lengths.unsqueeze(1)
    ).unsqueeze(1).to(text_mask.dtype)
    return DurationPath(
        alignment=alignment,
        frame_mask=frame_mask,
        frame_lengths=frame_lengths,
    )


def expand_token_features(
    token_features: torch.Tensor,
    alignment: torch.Tensor,
    frame_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Expand ``[B, C, T_text]`` features with ``[B, 1, T_frame, T_text]``."""
    if token_features.ndim != 3:
        raise ValueError("token_features must have shape [batch, channels, text_time]")
    if alignment.ndim != 4 or alignment.shape[1] != 1:
        raise ValueError("alignment must have shape [batch, 1, frame_time, text_time]")
    if token_features.shape[0] != alignment.shape[0]:
        raise ValueError("token features and alignment batch sizes must match")
    if token_features.shape[2] != alignment.shape[3]:
        raise ValueError("token features and alignment text lengths must match")
    if not token_features.is_floating_point() or not alignment.is_floating_point():
        raise ValueError("token features and alignment must be floating point")
    expanded = torch.bmm(token_features, alignment.squeeze(1).transpose(1, 2))
    if frame_mask is not None:
        if frame_mask.shape != (expanded.shape[0], 1, expanded.shape[2]):
            raise ValueError("frame_mask must have shape [batch, 1, frame_time]")
        expanded = expanded * frame_mask.to(expanded.dtype)
    return expanded


def expand_token_scalars(
    token_values: torch.Tensor,
    alignment: torch.Tensor,
    frame_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if token_values.ndim != 2:
        raise ValueError("token_values must have shape [batch, text_time]")
    return expand_token_features(
        token_values.unsqueeze(1), alignment, frame_mask
    ).squeeze(1)


def denormalize_log_f0(
    normalized_values: torch.Tensor,
    mask: torch.Tensor,
    mean: float,
    standard_deviation: float,
) -> torch.Tensor:
    if normalized_values.shape != mask.shape:
        raise ValueError("normalized log-F0 and mask must have identical shapes")
    if standard_deviation <= 0.0:
        raise ValueError("log-F0 standard deviation must be positive")
    return (
        normalized_values * standard_deviation + mean
    ) * mask.to(normalized_values.dtype)


class AlignmentFrameExpander(nn.Module):
    """Expand prior and predicted token prosody to acoustic-frame resolution."""

    def forward(
        self,
        alignment: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_log_scale: torch.Tensor,
        predicted_prosody: ProsodyPredictorOutput,
        frame_mask: torch.Tensor,
    ) -> FrameExpansionOutput:
        if prior_mean.shape != prior_log_scale.shape:
            raise ValueError("prior mean and log-scale shapes must match")
        if predicted_prosody.normalized_continuous_log_f0.shape != (
            prior_mean.shape[0],
            prior_mean.shape[2],
        ):
            raise ValueError("predicted pitch shape must match text prior")
        if predicted_prosody.voicing_logits.shape != (
            prior_mean.shape[0],
            prior_mean.shape[2],
        ):
            raise ValueError("predicted voicing shape must match text prior")
        expanded_pitch = expand_token_scalars(
            predicted_prosody.normalized_continuous_log_f0,
            alignment,
            frame_mask,
        )
        expanded_voicing_logits = expand_token_scalars(
            predicted_prosody.voicing_logits,
            alignment,
            frame_mask,
        )
        scalar_mask = frame_mask.squeeze(1).to(expanded_pitch.dtype)
        return FrameExpansionOutput(
            prior_mean=expand_token_features(prior_mean, alignment, frame_mask),
            prior_log_scale=expand_token_features(
                prior_log_scale, alignment, frame_mask
            ),
            normalized_continuous_log_f0=expanded_pitch,
            voicing_logits=expanded_voicing_logits,
            voicing_probability=(
                torch.sigmoid(expanded_voicing_logits) * scalar_mask
            ),
            mask=frame_mask,
        )


def kl_divergence_loss(
    prior_space_latent: torch.Tensor,
    posterior_mean: torch.Tensor,
    posterior_log_scale: torch.Tensor,
    expanded_prior_mean: torch.Tensor,
    expanded_prior_log_scale: torch.Tensor,
    frame_mask: torch.Tensor,
    flow_log_determinant: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Monte-Carlo VITS KL objective with optional affine-flow correction."""
    shapes = {
        tuple(value.shape)
        for value in (
            prior_space_latent,
            posterior_mean,
            posterior_log_scale,
            expanded_prior_mean,
            expanded_prior_log_scale,
        )
    }
    if len(shapes) != 1:
        raise ValueError("all KL latent/statistic tensors must have identical shapes")
    if prior_space_latent.ndim != 3:
        raise ValueError("KL tensors must have shape [batch, channels, frames]")
    if frame_mask.shape != (
        prior_space_latent.shape[0],
        1,
        prior_space_latent.shape[2],
    ):
        raise ValueError("frame_mask must have shape [batch, 1, frames]")
    values = (
        prior_space_latent,
        posterior_mean,
        posterior_log_scale,
        expanded_prior_mean,
        expanded_prior_log_scale,
    )
    if not all(torch.isfinite(value).all() for value in values):
        raise ValueError("KL inputs contain NaN/Inf")

    # EdgeTTS evaluates KL outside autocast. Explicit FP32 casts provide the
    # same numerical behavior when this helper is called from an autocast region.
    prior_space_latent = prior_space_latent.float()
    posterior_log_scale = posterior_log_scale.float()
    expanded_prior_mean = expanded_prior_mean.float()
    expanded_prior_log_scale = expanded_prior_log_scale.float()
    mask = frame_mask.float()
    inverse_prior_variance = torch.exp(-2.0 * expanded_prior_log_scale)
    elementwise = (
        expanded_prior_log_scale
        - posterior_log_scale
        - 0.5
        + 0.5
        * (prior_space_latent - expanded_prior_mean).square()
        * inverse_prior_variance
    )
    numerator = (elementwise * mask).sum()
    if flow_log_determinant is not None:
        if flow_log_determinant.shape != (prior_space_latent.shape[0],):
            raise ValueError("flow_log_determinant must have shape [batch]")
        numerator = numerator - flow_log_determinant.float().sum()
    return numerator / mask.sum().clamp_min(1.0)
