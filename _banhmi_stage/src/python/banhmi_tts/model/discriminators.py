"""BigVGAN discriminator ensemble: multi-period and multi-resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm


@dataclass(frozen=True)
class DiscriminatorConfig:
    periods: Tuple[int, ...] = (2, 3, 5, 7, 11)
    resolutions: Tuple[Tuple[int, int, int], ...] = (
        (1024, 120, 600),
        (2048, 240, 1200),
        (512, 50, 240),
    )
    use_weight_norm: bool = True
    include_scale_discriminator: bool = True
    use_magnitude_spectrogram: bool = True

    def validate(self) -> None:
        if not self.periods or any(period <= 1 for period in self.periods):
            raise ValueError("discriminator periods must be greater than one")
        if len(set(self.periods)) != len(self.periods):
            raise ValueError("discriminator periods must be unique")
        if not self.resolutions:
            raise ValueError("at least one STFT resolution is required")
        for n_fft, hop_length, win_length in self.resolutions:
            if min(n_fft, hop_length, win_length) <= 0 or win_length > n_fft:
                raise ValueError("invalid discriminator STFT resolution")


@dataclass
class DiscriminatorOutput:
    score: torch.Tensor
    feature_maps: List[torch.Tensor]


def _normalized(module: nn.Module, enabled: bool) -> nn.Module:
    return weight_norm(module) if enabled else module


def _remove_weight_norm(modules) -> None:
    for module in modules:
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=True
            )


class PeriodDiscriminator(nn.Module):
    """HiFi-GAN/BigVGAN periodic waveform discriminator."""

    def __init__(self, period: int, use_weight_norm: bool = True) -> None:
        super().__init__()
        self.period = period
        channels = (1, 32, 128, 512, 1024, 1024)
        self.convolutions = nn.ModuleList(
            _normalized(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=(5, 1),
                    stride=(3, 1) if layer_index < 4 else (1, 1),
                    padding=(2, 0),
                ),
                use_weight_norm,
            )
            for layer_index, (input_channels, output_channels) in enumerate(
                zip(channels[:-1], channels[1:])
            )
        )
        self.output_projection = _normalized(
            nn.Conv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0)),
            use_weight_norm,
        )

    def forward(self, waveform: torch.Tensor) -> DiscriminatorOutput:
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must have shape [batch, 1, samples]")
        remainder = waveform.shape[-1] % self.period
        if remainder:
            waveform = F.pad(
                waveform,
                (0, self.period - remainder),
                mode="reflect",
            )
        batch, channels, samples = waveform.shape
        values = waveform.view(batch, channels, samples // self.period, self.period)
        feature_maps = []
        for convolution in self.convolutions:
            values = F.leaky_relu(convolution(values), negative_slope=0.1)
            feature_maps.append(values)
        values = self.output_projection(values)
        feature_maps.append(values)
        return DiscriminatorOutput(
            score=values.flatten(1), feature_maps=feature_maps
        )

    def remove_weight_norm(self) -> None:
        _remove_weight_norm((*self.convolutions, self.output_projection))


class ScaleDiscriminator(nn.Module):
    """HiFi-GAN waveform-scale discriminator used by EdgeTTS MPD."""

    def __init__(self, use_weight_norm: bool = True) -> None:
        super().__init__()
        channels = (1, 16, 64, 256, 1024, 1024, 1024)
        kernels = (15, 41, 41, 41, 41, 5)
        strides = (1, 4, 4, 4, 4, 1)
        groups = (1, 4, 16, 64, 256, 1)
        self.convolutions = nn.ModuleList(
            _normalized(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size,
                    stride=stride,
                    groups=group,
                    padding=kernel_size // 2,
                ),
                use_weight_norm,
            )
            for input_channels, output_channels, kernel_size, stride, group in zip(
                channels[:-1], channels[1:], kernels, strides, groups
            )
        )
        self.output_projection = _normalized(
            nn.Conv1d(1024, 1, kernel_size=3, padding=1), use_weight_norm
        )

    def forward(self, waveform: torch.Tensor) -> DiscriminatorOutput:
        values = waveform
        feature_maps = []
        for convolution in self.convolutions:
            values = F.leaky_relu(convolution(values), negative_slope=0.1)
            feature_maps.append(values)
        values = self.output_projection(values)
        feature_maps.append(values)
        return DiscriminatorOutput(values.flatten(1), feature_maps)

    def remove_weight_norm(self) -> None:
        _remove_weight_norm((*self.convolutions, self.output_projection))


class ResolutionDiscriminator(nn.Module):
    """UnivNet/BigVGAN discriminator over complex STFT representations."""

    def __init__(
        self,
        resolution: Tuple[int, int, int],
        use_weight_norm: bool = True,
        use_magnitude: bool = False,
    ) -> None:
        super().__init__()
        self.n_fft, self.hop_length, self.win_length = resolution
        self.use_magnitude = use_magnitude
        channel_pairs = ((1 if use_magnitude else 2, 32), (32, 32), (32, 32), (32, 32), (32, 32))
        strides = ((1, 1), (1, 2), (1, 2), (1, 2), (1, 1))
        self.convolutions = nn.ModuleList(
            _normalized(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=(3, 9),
                    stride=stride,
                    padding=(1, 4),
                ),
                use_weight_norm,
            )
            for (input_channels, output_channels), stride in zip(
                channel_pairs, strides
            )
        )
        self.output_projection = _normalized(
            nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1)),
            use_weight_norm,
        )

    def _spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        # Match EdgeTTS MRD: cuFFT does not support fp16/bf16, so only the
        # spectral transform is forced to fp32 outside autocast.
        with torch.amp.autocast(device_type=waveform.device.type, enabled=False):
            window = torch.hann_window(
                self.win_length,
                dtype=torch.float32,
                device=waveform.device,
            )
            values = waveform.float().squeeze(1)
            pad = (self.n_fft - self.hop_length) // 2
            values = F.pad(values, (pad, pad), mode="reflect")
            spectrum = torch.stft(
                values,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=False,
                return_complex=True,
            )
        if self.use_magnitude:
            return spectrum.abs().unsqueeze(1)
        return torch.stack((spectrum.real, spectrum.imag), dim=1)

    def forward(self, waveform: torch.Tensor) -> DiscriminatorOutput:
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must have shape [batch, 1, samples]")
        values = self._spectrogram(waveform)
        feature_maps = []
        for convolution in self.convolutions:
            values = F.leaky_relu(convolution(values), negative_slope=0.1)
            feature_maps.append(values)
        values = self.output_projection(values)
        feature_maps.append(values)
        return DiscriminatorOutput(
            score=values.flatten(1), feature_maps=feature_maps
        )

    def remove_weight_norm(self) -> None:
        _remove_weight_norm((*self.convolutions, self.output_projection))


class BigVGANDiscriminator(nn.Module):
    def __init__(
        self, config: DiscriminatorConfig = DiscriminatorConfig()
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.period_discriminators = nn.ModuleList(
            PeriodDiscriminator(period, config.use_weight_norm)
            for period in config.periods
        )
        self.scale_discriminators = nn.ModuleList(
            [ScaleDiscriminator(config.use_weight_norm)]
            if config.include_scale_discriminator
            else []
        )
        self.resolution_discriminators = nn.ModuleList(
            ResolutionDiscriminator(
                resolution,
                config.use_weight_norm,
                config.use_magnitude_spectrogram,
            )
            for resolution in config.resolutions
        )

    def forward(self, waveform: torch.Tensor) -> List[DiscriminatorOutput]:
        if not waveform.is_floating_point() or not torch.isfinite(waveform).all():
            raise ValueError("waveform must be a finite floating-point tensor")
        return [
            discriminator(waveform)
            for discriminator in (
                *self.scale_discriminators,
                *self.period_discriminators,
                *self.resolution_discriminators,
            )
        ]

    def remove_weight_norm(self) -> None:
        for discriminator in (
            *self.period_discriminators,
            *self.scale_discriminators,
            *self.resolution_discriminators,
        ):
            discriminator.remove_weight_norm()
