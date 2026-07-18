"""Adversarial, feature-matching, and multi-resolution spectral losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .discriminators import DiscriminatorOutput


@dataclass
class AdversarialLosses:
    discriminator: torch.Tensor
    generator: torch.Tensor
    feature_matching: torch.Tensor


def discriminator_least_squares_loss(
    real_outputs: Sequence[DiscriminatorOutput],
    fake_outputs: Sequence[DiscriminatorOutput],
) -> torch.Tensor:
    if len(real_outputs) != len(fake_outputs) or not real_outputs:
        raise ValueError("real and fake discriminator output lists must match")
    losses = []
    for real, fake in zip(real_outputs, fake_outputs):
        real_score = real.score.float()
        fake_score = fake.score.float()
        losses.append(
            (real_score - 1.0).square().mean() + fake_score.square().mean()
        )
    return torch.stack(losses).sum()


def generator_least_squares_loss(
    fake_outputs: Sequence[DiscriminatorOutput],
) -> torch.Tensor:
    if not fake_outputs:
        raise ValueError("fake discriminator outputs cannot be empty")
    return torch.stack(
        [(output.score.float() - 1.0).square().mean() for output in fake_outputs]
    ).sum()


def feature_matching_loss(
    real_outputs: Sequence[DiscriminatorOutput],
    fake_outputs: Sequence[DiscriminatorOutput],
) -> torch.Tensor:
    if len(real_outputs) != len(fake_outputs) or not real_outputs:
        raise ValueError("real and fake discriminator output lists must match")
    losses = []
    for real, fake in zip(real_outputs, fake_outputs):
        if len(real.feature_maps) != len(fake.feature_maps):
            raise ValueError("real and fake feature-map lists must match")
        for real_feature, fake_feature in zip(
            real.feature_maps, fake.feature_maps
        ):
            losses.append(
                F.l1_loss(fake_feature.float(), real_feature.detach().float())
            )
    return 2.0 * torch.stack(losses).sum()


@dataclass(frozen=True)
class MultiResolutionSpectralLossConfig:
    resolutions: Tuple[Tuple[int, int, int], ...] = (
        (1024, 256, 1024),
        (2048, 512, 2048),
        (512, 128, 512),
    )
    spectral_convergence_weight: float = 1.0
    log_magnitude_weight: float = 1.0
    epsilon: float = 1e-7

    def validate(self) -> None:
        if not self.resolutions:
            raise ValueError("spectral loss requires at least one resolution")
        if min(
            self.spectral_convergence_weight,
            self.log_magnitude_weight,
            self.epsilon,
        ) <= 0.0:
            raise ValueError("spectral loss weights and epsilon must be positive")
        for n_fft, hop_length, win_length in self.resolutions:
            if min(n_fft, hop_length, win_length) <= 0 or win_length > n_fft:
                raise ValueError("invalid spectral-loss STFT resolution")


class MultiResolutionSpectralLoss(nn.Module):
    def __init__(
        self,
        config: MultiResolutionSpectralLossConfig = MultiResolutionSpectralLossConfig(),
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config

    @staticmethod
    def _magnitude(
        waveform: torch.Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=waveform.device.type, enabled=False):
            window = torch.hann_window(
                win_length, dtype=torch.float32, device=waveform.device
            )
            spectrum = torch.stft(
                waveform.float().squeeze(1),
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                return_complex=True,
            )
        return spectrum.abs()

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if generated.shape != target.shape or generated.ndim != 3:
            raise ValueError("generated and target waveforms must share [B, 1, T]")
        losses = []
        for n_fft, hop_length, win_length in self.config.resolutions:
            generated_magnitude = self._magnitude(
                generated, n_fft, hop_length, win_length
            )
            target_magnitude = self._magnitude(
                target, n_fft, hop_length, win_length
            )
            difference = generated_magnitude - target_magnitude
            spectral_convergence = torch.linalg.vector_norm(
                difference.flatten(1), dim=1
            ) / torch.linalg.vector_norm(
                target_magnitude.flatten(1), dim=1
            ).clamp_min(self.config.epsilon)
            log_magnitude = F.l1_loss(
                torch.log(generated_magnitude + self.config.epsilon),
                torch.log(target_magnitude + self.config.epsilon),
            )
            losses.append(
                self.config.spectral_convergence_weight
                * spectral_convergence.mean()
                + self.config.log_magnitude_weight * log_magnitude
            )
        return torch.stack(losses).mean()


@dataclass(frozen=True)
class MelReconstructionLossConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    weight: float = 45.0

    def validate(self) -> None:
        if min(self.sample_rate, self.n_fft, self.hop_length, self.win_length, self.n_mels) <= 0:
            raise ValueError("mel dimensions must be positive")
        if self.win_length > self.n_fft or not 0.0 <= self.fmin < self.fmax <= self.sample_rate / 2:
            raise ValueError("invalid mel frequency range")
        if self.weight < 0.0:
            raise ValueError("mel loss weight cannot be negative")


def _mel_filter_bank(config: MelReconstructionLossConfig) -> torch.Tensor:
    def hz_to_mel(value: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + value / 700.0)

    def mel_to_hz(value: torch.Tensor) -> torch.Tensor:
        return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)

    lower = hz_to_mel(torch.tensor(config.fmin))
    upper = hz_to_mel(torch.tensor(config.fmax))
    mel_points = torch.linspace(lower, upper, config.n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    frequencies = torch.linspace(0.0, config.sample_rate / 2, config.n_fft // 2 + 1)
    left = hz_points[:-2].unsqueeze(1)
    center = hz_points[1:-1].unsqueeze(1)
    right = hz_points[2:].unsqueeze(1)
    rising = (frequencies - left) / (center - left).clamp_min(1e-7)
    falling = (right - frequencies) / (right - center).clamp_min(1e-7)
    return torch.minimum(rising, falling).clamp_min(0.0)


class MelReconstructionLoss(nn.Module):
    """EdgeTTS/VITS-style log-mel L1 reconstruction objective."""

    def __init__(self, config: MelReconstructionLossConfig = MelReconstructionLossConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.register_buffer("mel_basis", _mel_filter_bank(config))

    def _mel(self, waveform: torch.Tensor) -> torch.Tensor:
        config = self.config
        with torch.amp.autocast(device_type=waveform.device.type, enabled=False):
            values = F.pad(
                waveform.float(),
                ((config.n_fft - config.hop_length) // 2,) * 2,
                mode="reflect",
            ).squeeze(1)
            spectrum = torch.stft(
                values,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                win_length=config.win_length,
                window=torch.hann_window(config.win_length, device=waveform.device),
                center=False,
                return_complex=True,
            )
            magnitude = torch.sqrt(spectrum.real.square() + spectrum.imag.square() + 1e-6)
            mel = torch.matmul(self.mel_basis.float(), magnitude)
            return torch.log(mel.clamp_min(1e-5))

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._mel(generated), self._mel(target)) * self.config.weight
