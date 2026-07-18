"""VITS-style WaveNet posterior encoder for linear spectrograms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm

from .text_encoder import sequence_mask


@dataclass(frozen=True)
class PosteriorEncoderConfig:
    input_channels: int = 513
    latent_channels: int = 192
    hidden_channels: int = 192
    kernel_size: int = 5
    dilation_rate: int = 1
    num_layers: int = 16
    dropout: float = 0.0
    use_weight_norm: bool = True

    def validate(self) -> None:
        if min(self.input_channels, self.latent_channels, self.hidden_channels) <= 0:
            raise ValueError("posterior encoder channel counts must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if self.dilation_rate <= 0 or self.num_layers <= 0:
            raise ValueError("dilation_rate and num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class PosteriorEncoderOutput:
    latent: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_log_scale: torch.Tensor
    mask: torch.Tensor


def _maybe_weight_norm(module: nn.Module, enabled: bool) -> nn.Module:
    return weight_norm(module) if enabled else module


class WaveNetResidualStack(nn.Module):
    """Non-causal gated WaveNet stack used by the VITS posterior encoder."""

    def __init__(self, config: PosteriorEncoderConfig) -> None:
        super().__init__()
        self.hidden_channels = config.hidden_channels
        self.num_layers = config.num_layers
        self.input_layers = nn.ModuleList()
        self.residual_skip_layers = nn.ModuleList()
        self.dropout = nn.Dropout(config.dropout)

        for layer_index in range(config.num_layers):
            dilation = config.dilation_rate**layer_index
            padding = dilation * (config.kernel_size - 1) // 2
            self.input_layers.append(
                _maybe_weight_norm(
                    nn.Conv1d(
                        config.hidden_channels,
                        2 * config.hidden_channels,
                        config.kernel_size,
                        dilation=dilation,
                        padding=padding,
                    ),
                    config.use_weight_norm,
                )
            )
            output_channels = (
                2 * config.hidden_channels
                if layer_index < config.num_layers - 1
                else config.hidden_channels
            )
            self.residual_skip_layers.append(
                _maybe_weight_norm(
                    nn.Conv1d(config.hidden_channels, output_channels, kernel_size=1),
                    config.use_weight_norm,
                )
            )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = inputs * mask
        skip = torch.zeros_like(values)
        for layer_index, (input_layer, residual_skip_layer) in enumerate(
            zip(self.input_layers, self.residual_skip_layers)
        ):
            gates = input_layer(values * mask)
            tanh_values, sigmoid_values = gates.chunk(2, dim=1)
            activated = torch.tanh(tanh_values) * torch.sigmoid(sigmoid_values)
            activated = self.dropout(activated)
            residual_skip = residual_skip_layer(activated) * mask
            if layer_index < self.num_layers - 1:
                residual, layer_skip = residual_skip.chunk(2, dim=1)
                values = (values + residual) * mask
                skip = skip + layer_skip
            else:
                skip = skip + residual_skip
        return skip * mask

    def remove_weight_norm(self) -> None:
        for module in (*self.input_layers, *self.residual_skip_layers):
            if parametrize.is_parametrized(module, "weight"):
                parametrize.remove_parametrizations(
                    module, "weight", leave_parametrized=True
                )


class PosteriorEncoder(nn.Module):
    """Parameterize and sample the acoustic posterior ``q(z | spectrogram)``."""

    def __init__(self, config: PosteriorEncoderConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.input_projection = nn.Conv1d(
            config.input_channels, config.hidden_channels, kernel_size=1
        )
        self.wavenet = WaveNetResidualStack(config)
        self.statistics_projection = nn.Conv1d(
            config.hidden_channels, 2 * config.latent_channels, kernel_size=1
        )

    def forward(
        self,
        spectrogram: torch.Tensor,
        spectrogram_lengths: torch.Tensor,
        *,
        sample: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> PosteriorEncoderOutput:
        if spectrogram.ndim != 3:
            raise ValueError("spectrogram must have shape [batch, bins, frames]")
        if spectrogram.shape[1] != self.config.input_channels:
            raise ValueError(
                f"spectrogram has {spectrogram.shape[1]} bins; "
                f"expected {self.config.input_channels}"
            )
        if spectrogram_lengths.ndim != 1 or len(spectrogram_lengths) != spectrogram.shape[0]:
            raise ValueError("spectrogram_lengths must have shape [batch]")
        if not spectrogram.is_floating_point():
            raise ValueError("spectrogram must use a floating-point dtype")
        if not torch.isfinite(spectrogram).all():
            raise ValueError("spectrogram contains NaN/Inf")

        spectrogram_lengths = spectrogram_lengths.to(device=spectrogram.device)
        valid = sequence_mask(spectrogram_lengths, spectrogram.shape[-1])
        mask = valid.unsqueeze(1).to(dtype=spectrogram.dtype)
        hidden = self.input_projection(spectrogram * mask) * mask
        hidden = self.wavenet(hidden, mask)
        statistics = self.statistics_projection(hidden) * mask
        posterior_mean, posterior_log_scale = statistics.chunk(2, dim=1)
        posterior_mean_fp32 = posterior_mean.float()
        posterior_log_scale_fp32 = posterior_log_scale.float()
        diagnostic_tensors = {
            "posterior_mean": posterior_mean_fp32,
            "posterior_log_scale": posterior_log_scale_fp32,
        }
        if sample:
            # Keep reparameterization in FP32 under BF16 autocast. This keeps
            # the same Gaussian objective without clamping log-scale values.
            posterior_scale = torch.exp(posterior_log_scale_fp32)
            noise = torch.randn(
                posterior_mean_fp32.shape,
                dtype=torch.float32,
                device=posterior_mean_fp32.device,
                generator=generator,
            )
            latent = posterior_mean_fp32 + noise * posterior_scale
            diagnostic_tensors["posterior_scale"] = posterior_scale
        else:
            latent = posterior_mean_fp32
        diagnostic_tensors["posterior_latent"] = latent
        finite_flags = torch.stack(
            [torch.isfinite(value).all() for value in diagnostic_tensors.values()]
        )
        if not bool(finite_flags.all()):
            failed = [
                name
                for name, finite
                in zip(diagnostic_tensors, finite_flags.tolist())
                if not finite
            ]
            finite_log_scale = posterior_log_scale_fp32[
                torch.isfinite(posterior_log_scale_fp32)
            ]
            log_scale_range = (
                (float(finite_log_scale.min()), float(finite_log_scale.max()))
                if finite_log_scale.numel()
                else None
            )
            raise FloatingPointError(
                "posterior produced NaN/Inf; "
                f"failed={failed}, finite_log_scale_range={log_scale_range}"
            )
        latent = latent * mask.float()
        return PosteriorEncoderOutput(
            latent=latent,
            posterior_mean=posterior_mean,
            posterior_log_scale=posterior_log_scale,
            mask=mask,
        )

    def remove_weight_norm(self) -> None:
        """Materialize normalized weights for inference/export."""
        self.wavenet.remove_weight_norm()
