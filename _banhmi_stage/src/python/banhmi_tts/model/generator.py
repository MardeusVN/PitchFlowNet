"""F0/voicing-conditioned BigVGAN-style waveform generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm


@dataclass(frozen=True)
class GeneratorConfig:
    latent_channels: int = 192
    # EdgeTTS Config H: HiFi-GAN ResBlock2 decoder with SnakeBeta.
    initial_channels: int = 256
    upsample_rates: Tuple[int, ...] = (8, 8, 4)
    upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 8)
    amp_kernel_sizes: Tuple[int, ...] = (3, 5, 7)
    amp_dilations: Tuple[Tuple[int, ...], ...] = (
        (1, 2),
        (2, 6),
        (3, 12),
    )
    use_weight_norm: bool = True
    activation: str = "snake_beta"
    use_prosody_conditioning: bool = True

    def validate(self) -> None:
        if self.latent_channels <= 0 or self.initial_channels <= 0:
            raise ValueError("generator channels must be positive")
        if self.activation not in {"filtered_snake_beta", "snake_beta"}:
            raise ValueError("generator activation must be filtered_snake_beta or snake_beta")
        if not self.upsample_rates or len(self.upsample_rates) != len(
            self.upsample_kernel_sizes
        ):
            raise ValueError("upsample rates and kernels must be non-empty and match")
        if len(self.amp_kernel_sizes) != len(self.amp_dilations):
            raise ValueError("AMP kernels and dilation groups must match")
        if any(rate <= 0 for rate in self.upsample_rates):
            raise ValueError("upsample rates must be positive")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in self.amp_kernel_sizes):
            raise ValueError("AMP kernels must be positive odd integers")
        if any(not dilations or any(value <= 0 for value in dilations) for dilations in self.amp_dilations):
            raise ValueError("AMP dilation groups must contain positive values")
        channels = self.initial_channels
        for _ in self.upsample_rates:
            if channels % 2 != 0:
                raise ValueError("initial_channels must remain divisible across upsampling")
            channels //= 2

    @property
    def total_upsample_factor(self) -> int:
        factor = 1
        for rate in self.upsample_rates:
            factor *= rate
        return factor


def _maybe_weight_norm(module: nn.Module, enabled: bool) -> nn.Module:
    return weight_norm(module) if enabled else module


def _edge_init_weights(module: nn.Module) -> None:
    """Match the HiFi-GAN/VITS generator convolution initialization."""
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        if parametrize.is_parametrized(module, "weight"):
            # Parametrized weight norm computes ``weight`` from magnitude (g)
            # and direction (v). Initializing the computed tensor is temporary;
            # initialize v and recompute g so the effective weight remains
            # N(0, 0.01) on every subsequent forward.
            parameters = module.parametrizations.weight
            direction = parameters.original1
            magnitude = parameters.original0
            nn.init.normal_(direction, mean=0.0, std=0.01)
            reduction_dimensions = tuple(range(1, direction.ndim))
            with torch.no_grad():
                magnitude.copy_(
                    torch.linalg.vector_norm(
                        direction, dim=reduction_dimensions, keepdim=True
                    )
                )
        else:
            nn.init.normal_(module.weight, mean=0.0, std=0.01)


class SnakeBeta(nn.Module):
    """EdgeTTS SnakeBeta with independent frequency and magnitude parameters."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Match piper_train.vits.modules.Snake1d exactly.
        alpha = self.alpha.abs().clamp_min(1e-9)
        beta = self.beta.abs().clamp_min(1e-9)
        return inputs + torch.sin(alpha * inputs).square() / beta


class FilteredSnakeBeta(nn.Module):
    """Portable filtered periodic nonlinearity following BigVGAN's AMP idea."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.activation = SnakeBeta(channels)
        kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        self.register_buffer("lowpass_kernel", kernel.view(1, 1, -1))
        self.channels = channels

    def _lowpass(self, inputs: torch.Tensor) -> torch.Tensor:
        kernel = self.lowpass_kernel.to(dtype=inputs.dtype).expand(
            self.channels, 1, -1
        )
        return F.conv1d(inputs, kernel, padding=2, groups=self.channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_length = inputs.shape[-1]
        values = F.interpolate(
            inputs, scale_factor=2.0, mode="linear", align_corners=False
        )
        values = self._lowpass(values)
        values = self.activation(values)
        values = self._lowpass(values)
        return values[:, :, ::2][:, :, :original_length]


class EdgeResBlock2(nn.Module):
    """EdgeTTS/HiFi-GAN ResBlock2 with SnakeBeta before each convolution."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: Sequence[int],
        use_weight_norm: bool,
        activation_type: type[nn.Module] = FilteredSnakeBeta,
    ) -> None:
        super().__init__()
        self.activations = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        for dilation in dilations:
            self.activations.append(activation_type(channels))
            self.convolutions.append(
                _maybe_weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=dilation,
                        padding=dilation * (kernel_size - 1) // 2,
                    ),
                    use_weight_norm,
                )
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs
        for activation, convolution in zip(self.activations, self.convolutions):
            values = values + convolution(activation(values))
        return values

    def remove_weight_norm(self) -> None:
        for module in self.convolutions:
            if parametrize.is_parametrized(module, "weight"):
                parametrize.remove_parametrizations(
                    module, "weight", leave_parametrized=True
                )


class BigVGANGenerator(nn.Module):
    """Decode frame-rate VITS latents to waveform samples."""

    def __init__(self, config: GeneratorConfig = GeneratorConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.input_projection = nn.Conv1d(
            config.latent_channels, config.initial_channels, kernel_size=7, padding=3
        )
        self.prosody_conditioning = nn.Conv1d(
            2, config.initial_channels, kernel_size=1
        )
        nn.init.zeros_(self.prosody_conditioning.weight)
        nn.init.zeros_(self.prosody_conditioning.bias)

        activation_type = (
            FilteredSnakeBeta
            if config.activation == "filtered_snake_beta"
            else SnakeBeta
        )

        self.pre_upsample_activations = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.amp_stages = nn.ModuleList()
        channels = config.initial_channels
        for rate, kernel_size in zip(
            config.upsample_rates, config.upsample_kernel_sizes
        ):
            output_channels = channels // 2
            self.pre_upsample_activations.append(activation_type(channels))
            self.upsamplers.append(
                _maybe_weight_norm(
                    nn.ConvTranspose1d(
                        channels,
                        output_channels,
                        kernel_size,
                        stride=rate,
                        padding=(kernel_size - rate) // 2,
                    ),
                    config.use_weight_norm,
                )
            )
            self.amp_stages.append(
                nn.ModuleList(
                    EdgeResBlock2(
                        output_channels,
                        amp_kernel,
                        dilations,
                        config.use_weight_norm,
                        activation_type,
                    )
                    for amp_kernel, dilations in zip(
                        config.amp_kernel_sizes, config.amp_dilations
                    )
                )
            )
            channels = output_channels
        self.output_activation = activation_type(channels)
        self.output_projection = nn.Conv1d(
            channels, 1, kernel_size=7, padding=3, bias=False
        )
        # EdgeTTS applies N(0, 0.01) to transposed convolutions and every
        # residual-block convolution. Keep pre/post projections on PyTorch's
        # defaults, matching its Generator implementation.
        self.upsamplers.apply(_edge_init_weights)
        self.amp_stages.apply(_edge_init_weights)

    def forward(
        self,
        latent: torch.Tensor,
        normalized_log_f0: torch.Tensor,
        voicing: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[1] != self.config.latent_channels:
            raise ValueError(
                f"latent must have shape [batch, {self.config.latent_channels}, frames]"
            )
        expected = (latent.shape[0], latent.shape[2])
        if normalized_log_f0.shape != expected or voicing.shape != expected:
            raise ValueError("pitch and voicing must match latent frame dimensions")
        if not all(
            torch.isfinite(value).all()
            for value in (latent, normalized_log_f0, voicing)
        ):
            raise ValueError("generator inputs contain NaN/Inf")
        if torch.any((voicing < 0.0) | (voicing > 1.0)):
            raise ValueError("voicing conditioning must be in [0, 1]")

        values = self.input_projection(latent)
        if self.config.use_prosody_conditioning:
            conditioning = torch.stack((normalized_log_f0, voicing), dim=1)
            values = values + self.prosody_conditioning(conditioning)
        for activation, upsampler, amp_blocks in zip(
            self.pre_upsample_activations, self.upsamplers, self.amp_stages
        ):
            values = upsampler(activation(values))
            values = sum(block(values) for block in amp_blocks) / len(amp_blocks)
        return torch.tanh(self.output_projection(self.output_activation(values)))

    def remove_weight_norm(self) -> None:
        for module in self.upsamplers:
            if parametrize.is_parametrized(module, "weight"):
                parametrize.remove_parametrizations(
                    module, "weight", leave_parametrized=True
                )
        for stage in self.amp_stages:
            for block in stage:
                block.remove_weight_norm()

    @torch.no_grad()
    def snake_parameter_statistics(self) -> dict[str, float]:
        activations = [module for module in self.modules() if isinstance(module, SnakeBeta)]
        if not activations:
            return {}
        alpha = torch.cat(
            [module.alpha.float().abs().clamp_min(1e-9).flatten() for module in activations]
        )
        beta = torch.cat(
            [module.beta.float().abs().clamp_min(1e-9).flatten() for module in activations]
        )
        return {
            "snake_alpha_min": float(alpha.min()),
            "snake_alpha_max": float(alpha.max()),
            "snake_beta_min": float(beta.min()),
            "snake_beta_max": float(beta.max()),
        }
