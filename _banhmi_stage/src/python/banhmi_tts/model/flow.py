"""Invertible VITS2-style Transformer coupling flow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .posterior_encoder import PosteriorEncoderConfig, WaveNetResidualStack
from .text_encoder import TextEncoderBlock, TextEncoderConfig


@dataclass(frozen=True)
class FlowConfig:
    channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 384
    num_heads: int = 2
    transformer_layers: int = 1
    relative_position_window: int = 4
    kernel_size: int = 5
    dilation_rate: int = 1
    wavenet_layers: int = 4
    num_flows: int = 4
    dropout: float = 0.0
    mean_only: bool = True
    use_weight_norm: bool = True

    def validate(self) -> None:
        if self.channels <= 0 or self.channels % 2 != 0:
            raise ValueError("flow channels must be a positive even integer")
        if min(self.hidden_channels, self.filter_channels, self.num_heads) <= 0:
            raise ValueError("flow hidden/filter channels and heads must be positive")
        if self.hidden_channels % self.num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")
        if min(self.transformer_layers, self.wavenet_layers, self.num_flows) <= 0:
            raise ValueError("flow layer counts must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if self.dilation_rate <= 0 or self.relative_position_window < 0:
            raise ValueError("invalid dilation rate or relative-position window")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class FlowOutput:
    latent: torch.Tensor
    log_determinant: torch.Tensor


class TransformerWaveNetConditioner(nn.Module):
    """Global Transformer context followed by local WaveNet refinement."""

    def __init__(self, input_channels: int, output_channels: int, config: FlowConfig):
        super().__init__()
        self.input_projection = nn.Conv1d(
            input_channels, config.hidden_channels, kernel_size=1
        )
        transformer_config = TextEncoderConfig(
            num_symbols=1,
            latent_channels=config.hidden_channels,
            hidden_channels=config.hidden_channels,
            filter_channels=config.filter_channels,
            num_heads=config.num_heads,
            num_layers=config.transformer_layers,
            kernel_size=3,
            dropout=config.dropout,
            relative_position_window=config.relative_position_window,
            padding_id=0,
        )
        self.transformer_blocks = nn.ModuleList(
            TextEncoderBlock(transformer_config)
            for _ in range(config.transformer_layers)
        )
        wavenet_config = PosteriorEncoderConfig(
            input_channels=config.hidden_channels,
            latent_channels=config.hidden_channels,
            hidden_channels=config.hidden_channels,
            kernel_size=config.kernel_size,
            dilation_rate=config.dilation_rate,
            num_layers=config.wavenet_layers,
            dropout=config.dropout,
            use_weight_norm=config.use_weight_norm,
        )
        self.wavenet = WaveNetResidualStack(wavenet_config)
        self.output_projection = nn.Conv1d(
            config.hidden_channels, output_channels, kernel_size=1
        )
        # Each coupling starts as identity, matching the stable VITS setup.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(inputs * mask) * mask
        valid = mask.squeeze(1).bool()
        hidden_time_major = hidden.transpose(1, 2)
        for block in self.transformer_blocks:
            hidden_time_major = block(hidden_time_major, valid)
        hidden = hidden_time_major.transpose(1, 2) * mask
        hidden = self.wavenet(hidden, mask)
        return self.output_projection(hidden) * mask

    def remove_weight_norm(self) -> None:
        self.wavenet.remove_weight_norm()


class TransformerCouplingLayer(nn.Module):
    """Affine coupling whose parameters are predicted from the unchanged half."""

    def __init__(self, config: FlowConfig) -> None:
        super().__init__()
        self.half_channels = config.channels // 2
        self.mean_only = config.mean_only
        output_channels = self.half_channels if config.mean_only else 2 * self.half_channels
        self.conditioner = TransformerWaveNetConditioner(
            self.half_channels, output_channels, config
        )

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor, reverse: bool = False
    ) -> FlowOutput:
        first, second = inputs.chunk(2, dim=1)
        statistics = self.conditioner(first, mask)
        if self.mean_only:
            shift = statistics
            log_scale = torch.zeros_like(shift)
        else:
            shift, raw_log_scale = statistics.chunk(2, dim=1)
            # Bounded affine scales avoid overflow while preserving invertibility.
            log_scale = torch.tanh(raw_log_scale) * mask

        if reverse:
            second = (second - shift) * torch.exp(-log_scale)
            log_determinant = -(log_scale * mask).sum(dim=(1, 2))
        else:
            second = shift + second * torch.exp(log_scale)
            log_determinant = (log_scale * mask).sum(dim=(1, 2))
        latent = torch.cat((first, second), dim=1) * mask
        return FlowOutput(latent=latent, log_determinant=log_determinant)

    def remove_weight_norm(self) -> None:
        self.conditioner.remove_weight_norm()


class ChannelFlip(nn.Module):
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> FlowOutput:
        return FlowOutput(
            latent=torch.flip(inputs, dims=(1,)) * mask,
            log_determinant=torch.zeros(
                inputs.shape[0], dtype=inputs.dtype, device=inputs.device
            ),
        )


class TransformerCouplingFlow(nn.Module):
    """Stack Transformer coupling layers and channel flips."""

    def __init__(self, config: FlowConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.couplings = nn.ModuleList(
            TransformerCouplingLayer(config) for _ in range(config.num_flows)
        )
        self.flips = nn.ModuleList(ChannelFlip() for _ in range(config.num_flows))

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor, reverse: bool = False
    ) -> FlowOutput:
        if inputs.ndim != 3:
            raise ValueError("flow inputs must have shape [batch, channels, time]")
        if inputs.shape[1] != self.config.channels:
            raise ValueError(
                f"flow inputs have {inputs.shape[1]} channels; expected {self.config.channels}"
            )
        if mask.shape != (inputs.shape[0], 1, inputs.shape[2]):
            raise ValueError("flow mask must have shape [batch, 1, time]")
        if not inputs.is_floating_point() or not torch.isfinite(inputs).all():
            raise ValueError("flow inputs must be finite floating-point tensors")

        mask = mask.to(device=inputs.device, dtype=inputs.dtype)
        latent = inputs * mask
        total_log_determinant = torch.zeros(
            inputs.shape[0], dtype=inputs.dtype, device=inputs.device
        )
        if reverse:
            for coupling, flip in zip(reversed(self.couplings), reversed(self.flips)):
                flipped = flip(latent, mask)
                transformed = coupling(flipped.latent, mask, reverse=True)
                latent = transformed.latent
                total_log_determinant = (
                    total_log_determinant
                    + flipped.log_determinant
                    + transformed.log_determinant
                )
        else:
            for coupling, flip in zip(self.couplings, self.flips):
                transformed = coupling(latent, mask, reverse=False)
                flipped = flip(transformed.latent, mask)
                latent = flipped.latent
                total_log_determinant = (
                    total_log_determinant
                    + transformed.log_determinant
                    + flipped.log_determinant
                )
        return FlowOutput(latent=latent * mask, log_determinant=total_log_determinant)

    def remove_weight_norm(self) -> None:
        for coupling in self.couplings:
            coupling.remove_weight_norm()
