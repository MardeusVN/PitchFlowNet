"""Duration and joint continuous-pitch/voicing predictors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PredictorConfig:
    input_channels: int = 192
    hidden_channels: int = 256
    kernel_size: int = 3
    num_layers: int = 2
    dropout: float = 0.5
    duration_type: str = "stochastic"
    duration_flows: int = 4
    maximum_log_duration: float = 6.0
    maximum_duration_frames: int = 500

    def validate(self) -> None:
        if min(self.input_channels, self.hidden_channels, self.num_layers) <= 0:
            raise ValueError("predictor channels and layer count must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.duration_type not in {"stochastic", "deterministic"}:
            raise ValueError("duration_type must be stochastic or deterministic")
        if self.duration_flows <= 0:
            raise ValueError("duration_flows must be positive")
        if self.maximum_log_duration <= 0.0 or self.maximum_duration_frames <= 0:
            raise ValueError("duration safety limits must be positive")


@dataclass
class ProsodyPredictorOutput:
    normalized_continuous_log_f0: torch.Tensor
    voicing_logits: torch.Tensor

    @property
    def voicing_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.voicing_logits)


@dataclass
class PredictorLosses:
    total: torch.Tensor
    duration: torch.Tensor
    pitch: torch.Tensor
    voicing: torch.Tensor


@dataclass(frozen=True)
class PredictorLossConfig:
    duration_weight: float = 1.0
    pitch_weight: float = 1.0
    voicing_weight: float = 1.0

    def validate(self) -> None:
        if min(self.duration_weight, self.pitch_weight, self.voicing_weight) < 0.0:
            raise ValueError("predictor loss weights cannot be negative")


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.normalization(inputs.transpose(1, 2)).transpose(1, 2)


class TokenPredictorBackbone(nn.Module):
    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        config.validate()
        self.layers = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        input_channels = config.input_channels
        for _ in range(config.num_layers):
            self.layers.append(
                nn.Conv1d(
                    input_channels,
                    config.hidden_channels,
                    config.kernel_size,
                    padding=config.kernel_size // 2,
                )
            )
            self.normalizations.append(ChannelLayerNorm(config.hidden_channels))
            input_channels = config.hidden_channels
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = inputs * mask
        for convolution, normalization in zip(self.layers, self.normalizations):
            values = convolution(values * mask)
            values = F.relu(values)
            values = self.dropout(normalization(values))
        return values * mask


def _validate_predictor_inputs(
    hidden: torch.Tensor, mask: torch.Tensor, expected_channels: int
) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[1] != expected_channels:
        raise ValueError(
            f"hidden must have shape [batch, {expected_channels}, text_time]"
        )
    if mask.shape != (hidden.shape[0], 1, hidden.shape[2]):
        raise ValueError("mask must have shape [batch, 1, text_time]")
    if not hidden.is_floating_point() or not torch.isfinite(hidden).all():
        raise ValueError("hidden must be a finite floating-point tensor")
    return mask.to(device=hidden.device, dtype=hidden.dtype)


class DurationPredictor(nn.Module):
    """VITS deterministic log-duration predictor with detached text features."""

    def __init__(self, config: PredictorConfig = PredictorConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.backbone = TokenPredictorBackbone(config)
        self.output_projection = nn.Conv1d(config.hidden_channels, 1, kernel_size=1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = _validate_predictor_inputs(hidden, mask, self.config.input_channels)
        # Matching VITS: duration loss must not reshape the text encoder.
        features = self.backbone(hidden.detach(), mask)
        return self.output_projection(features).squeeze(1) * mask.squeeze(1)


class JointProsodyPredictor(nn.Module):
    """Predict continuous log-F0 and voicing from a shared token backbone."""

    def __init__(self, config: PredictorConfig = PredictorConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.backbone = TokenPredictorBackbone(config)
        self.pitch_projection = nn.Conv1d(config.hidden_channels, 1, kernel_size=1)
        self.voicing_projection = nn.Conv1d(config.hidden_channels, 1, kernel_size=1)

    def forward(
        self, hidden: torch.Tensor, mask: torch.Tensor
    ) -> ProsodyPredictorOutput:
        mask = _validate_predictor_inputs(hidden, mask, self.config.input_channels)
        features = self.backbone(hidden, mask)
        text_mask = mask.squeeze(1)
        return ProsodyPredictorOutput(
            normalized_continuous_log_f0=(
                self.pitch_projection(features).squeeze(1) * text_mask
            ),
            voicing_logits=self.voicing_projection(features).squeeze(1) * text_mask,
        )


def duration_targets(durations: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Convert positive MAS durations to the VITS log-duration domain."""
    if durations.shape != mask.shape:
        raise ValueError("durations and mask must have identical shapes")
    valid = mask.bool()
    if torch.any(durations[valid] < 1):
        raise ValueError("valid MAS durations must be at least one frame")
    return torch.log(durations.to(torch.float32).clamp_min(1.0)) * valid


def decode_durations(
    predicted_log_duration: torch.Tensor,
    mask: torch.Tensor,
    length_scale: float = 1.0,
    maximum_log_duration: float = 6.0,
    maximum_duration_frames: int = 500,
) -> torch.Tensor:
    if predicted_log_duration.shape != mask.shape:
        raise ValueError("predicted durations and mask must have identical shapes")
    if length_scale <= 0.0:
        raise ValueError("length_scale must be positive")
    if maximum_log_duration <= 0.0 or maximum_duration_frames <= 0:
        raise ValueError("duration safety limits must be positive")
    valid = mask.bool()
    safe_log_duration = torch.nan_to_num(
        predicted_log_duration,
        nan=0.0,
        posinf=maximum_log_duration,
        neginf=-maximum_log_duration,
    ).clamp(-maximum_log_duration, maximum_log_duration)
    durations = torch.ceil(torch.exp(safe_log_duration) * length_scale).long()
    durations = torch.clamp(durations, min=1, max=maximum_duration_frames)
    return durations * valid.long()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def compute_predictor_losses(
    predicted_log_duration: torch.Tensor,
    predicted_prosody: ProsodyPredictorOutput,
    target_durations: torch.Tensor,
    target_normalized_continuous_log_f0: torch.Tensor,
    target_voiced_ratio: torch.Tensor,
    mask: torch.Tensor,
    config: PredictorLossConfig = PredictorLossConfig(),
    duration_loss_override: torch.Tensor | None = None,
) -> PredictorLosses:
    """Masked MSE duration/pitch losses and soft-target voicing BCE."""
    config.validate()
    expected_shape = predicted_log_duration.shape
    values = (
        predicted_prosody.normalized_continuous_log_f0,
        predicted_prosody.voicing_logits,
        target_durations,
        target_normalized_continuous_log_f0,
        target_voiced_ratio,
        mask,
    )
    if any(value.shape != expected_shape for value in values):
        raise ValueError("all predictor outputs, targets, and masks must match")
    if torch.any((target_voiced_ratio < 0.0) | (target_voiced_ratio > 1.0)):
        raise ValueError("voiced-ratio targets must be in [0, 1]")

    target_log_duration = duration_targets(target_durations, mask)
    duration_loss = (
        _masked_mean((predicted_log_duration - target_log_duration).square(), mask)
        if duration_loss_override is None
        else duration_loss_override
    )
    if duration_loss.ndim != 0 or not torch.isfinite(duration_loss):
        raise ValueError("duration loss must be a finite scalar")
    pitch_loss = _masked_mean(
        (
            predicted_prosody.normalized_continuous_log_f0
            - target_normalized_continuous_log_f0
        ).square(),
        mask,
    )
    voicing_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(
            predicted_prosody.voicing_logits,
            target_voiced_ratio.to(predicted_prosody.voicing_logits.dtype),
            reduction="none",
        ),
        mask,
    )
    total = (
        config.duration_weight * duration_loss
        + config.pitch_weight * pitch_loss
        + config.voicing_weight * voicing_loss
    )
    return PredictorLosses(
        total=total,
        duration=duration_loss,
        pitch=pitch_loss,
        voicing=voicing_loss,
    )
