"""VITS2 monotonic alignment search with early Gaussian exploration noise."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

try:
    from .monotonic_align import maximum_path as _compiled_maximum_path
except ImportError:
    _compiled_maximum_path = None


@dataclass(frozen=True)
class MASConfig:
    noise_enabled: bool = True
    noise_initial_scale: float = 0.01
    noise_decay: float = 2e-6

    def validate(self) -> None:
        if self.noise_initial_scale < 0.0 or self.noise_decay < 0.0:
            raise ValueError("MAS noise scale and decay cannot be negative")

    def noise_scale(self, global_step: int, training: bool) -> float:
        if global_step < 0:
            raise ValueError("global_step cannot be negative")
        if not self.noise_enabled or not training:
            return 0.0
        return max(self.noise_initial_scale - global_step * self.noise_decay, 0.0)


@dataclass
class MASOutput:
    alignment: torch.Tensor
    durations: torch.Tensor
    noise_scale: float


def gaussian_log_likelihood(
    latent: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_scale: torch.Tensor,
) -> torch.Tensor:
    """Compute ``log N(z_frame; prior_token)`` as ``[B, T_frame, T_text]``.

    The expanded ``[B, C, T_frame, T_text]`` tensor is avoided to keep memory
    practical for full-resolution VITS alignments.
    """
    if latent.ndim != 3 or prior_mean.ndim != 3 or prior_log_scale.ndim != 3:
        raise ValueError("latent and prior statistics must have shape [B, C, T]")
    if prior_mean.shape != prior_log_scale.shape:
        raise ValueError("prior mean and log-scale shapes must match")
    if latent.shape[:2] != prior_mean.shape[:2]:
        raise ValueError("latent and prior channel/batch dimensions must match")
    if not all(value.is_floating_point() for value in (latent, prior_mean, prior_log_scale)):
        raise ValueError("alignment inputs must use floating-point dtypes")

    inverse_variance = torch.exp(-2.0 * prior_log_scale)
    constant_and_scale = (
        -0.5 * math.log(2.0 * math.pi) - prior_log_scale
    ).sum(dim=1, keepdim=True)
    latent_quadratic = torch.matmul(
        (-0.5 * latent.square()).transpose(1, 2), inverse_variance
    )
    cross = torch.matmul(
        latent.transpose(1, 2), prior_mean * inverse_variance
    )
    mean_quadratic = (
        -0.5 * prior_mean.square() * inverse_variance
    ).sum(dim=1, keepdim=True)
    return latent_quadratic + cross + mean_quadratic + constant_and_scale


def alignment_mask(
    acoustic_lengths: torch.Tensor,
    text_lengths: torch.Tensor,
    maximum_acoustic_length: int,
    maximum_text_length: int,
) -> torch.Tensor:
    acoustic_positions = torch.arange(
        maximum_acoustic_length, device=acoustic_lengths.device
    )
    text_positions = torch.arange(maximum_text_length, device=text_lengths.device)
    acoustic_valid = acoustic_positions.unsqueeze(0) < acoustic_lengths.unsqueeze(1)
    text_valid = text_positions.unsqueeze(0) < text_lengths.unsqueeze(1)
    return acoustic_valid.unsqueeze(2) & text_valid.unsqueeze(1)


def maximum_path(
    scores: torch.Tensor,
    acoustic_lengths: torch.Tensor,
    text_lengths: torch.Tensor,
    *,
    use_compiled: Optional[bool] = None,
) -> torch.Tensor:
    """Find the highest-scoring monotonic path with stay/advance transitions."""
    if scores.ndim != 3:
        raise ValueError("scores must have shape [batch, acoustic_time, text_time]")
    batch, maximum_acoustic, maximum_text = scores.shape
    if acoustic_lengths.shape != (batch,) or text_lengths.shape != (batch,):
        raise ValueError("length tensors must have shape [batch]")
    acoustic_lengths = acoustic_lengths.to(device=scores.device, dtype=torch.long)
    text_lengths = text_lengths.to(device=scores.device, dtype=torch.long)
    if torch.any(acoustic_lengths <= 0) or torch.any(text_lengths <= 0):
        raise ValueError("alignment lengths must be positive")
    if torch.any(acoustic_lengths > maximum_acoustic) or torch.any(text_lengths > maximum_text):
        raise ValueError("alignment length exceeds score tensor")
    if torch.any(acoustic_lengths < text_lengths):
        raise ValueError("MAS requires acoustic length >= text length")

    if use_compiled is None:
        use_compiled = _compiled_maximum_path is not None
    if use_compiled:
        if _compiled_maximum_path is None:
            raise RuntimeError(
                "compiled MAS is unavailable; run scripts/build_monotonic_align.sh"
            )
        return _compiled_maximum_path(scores, acoustic_lengths, text_lengths)

    # MAS is a cumulative dynamic program. Run it in float32 even under AMP;
    # long sequences can overflow float16 although each individual score is
    # finite.
    working_scores = scores.float()
    # A finite dtype minimum is not an unreachable state: adding another
    # negative score can overflow under AMP, allowing an invalid predecessor
    # to win. True -inf remains unreachable through every DP transition.
    negative_infinity = float("-inf")
    dynamic = torch.full(
        (batch, maximum_text),
        negative_infinity,
        dtype=working_scores.dtype,
        device=scores.device,
    )
    dynamic[:, 0] = working_scores[:, 0, 0]
    backpointers = torch.zeros(
        batch,
        maximum_acoustic,
        maximum_text,
        dtype=torch.bool,
        device=scores.device,
    )
    token_positions = torch.arange(maximum_text, device=scores.device).unsqueeze(0)

    for acoustic_index in range(1, maximum_acoustic):
        stay = dynamic
        advance = torch.cat(
            (
                torch.full(
                    (batch, 1),
                    negative_infinity,
                    dtype=working_scores.dtype,
                    device=scores.device,
                ),
                dynamic[:, :-1],
            ),
            dim=1,
        )
        choose_advance = advance > stay
        candidate = torch.maximum(stay, advance) + working_scores[:, acoustic_index, :]

        frame_active = acoustic_index < acoustic_lengths
        maximum_reachable = token_positions <= acoustic_index
        minimum_reachable = (
            token_positions
            >= (text_lengths - acoustic_lengths + acoustic_index).unsqueeze(1)
        )
        token_valid = token_positions < text_lengths.unsqueeze(1)
        state_valid = maximum_reachable & minimum_reachable & token_valid
        updated = torch.where(
            state_valid,
            candidate,
            torch.full_like(candidate, negative_infinity),
        )
        dynamic = torch.where(frame_active.unsqueeze(1), updated, dynamic)
        backpointers[:, acoustic_index, :] = (
            choose_advance & state_valid & frame_active.unsqueeze(1)
        )

    path = torch.zeros_like(scores)
    batch_indices = torch.arange(batch, device=scores.device)
    current_token = text_lengths - 1
    for acoustic_index in range(maximum_acoustic - 1, -1, -1):
        frame_active = acoustic_index < acoustic_lengths
        active_batches = batch_indices[frame_active]
        active_tokens = current_token[frame_active]
        path[active_batches, acoustic_index, active_tokens] = 1.0
        advances = backpointers[batch_indices, acoustic_index, current_token]
        current_token = current_token - (advances & frame_active).long()
    if torch.any(current_token != 0):
        raise RuntimeError("MAS backtracking did not terminate at the first text token")
    return path


def compiled_maximum_path_available() -> bool:
    return _compiled_maximum_path is not None


class MonotonicAlignmentSearch(nn.Module):
    """Compute deterministic/noisy VITS2 alignments outside autograd."""

    def __init__(self, config: MASConfig = MASConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config

    def forward(
        self,
        latent: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_log_scale: torch.Tensor,
        acoustic_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
        *,
        global_step: int = 0,
        add_noise: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
    ) -> MASOutput:
        if not all(
            torch.isfinite(value).all()
            for value in (latent, prior_mean, prior_log_scale)
        ):
            raise ValueError("MAS inputs contain NaN/Inf")
        acoustic_lengths = acoustic_lengths.to(device=latent.device, dtype=torch.long)
        text_lengths = text_lengths.to(device=latent.device, dtype=torch.long)
        if latent.shape[-1] < int(acoustic_lengths.max()):
            raise ValueError("acoustic length exceeds latent frames")
        if prior_mean.shape[-1] < int(text_lengths.max()):
            raise ValueError("text length exceeds prior frames")

        use_noise = self.training if add_noise is None else bool(add_noise)
        noise_scale = self.config.noise_scale(global_step, training=use_noise)
        with torch.no_grad():
            # Match the float32 MAS input used by EdgeTTS/VITS. Computing the
            # Gaussian quadratic in autocast float16 can overflow to -inf
            # before the dynamic program even starts.
            with torch.amp.autocast(device_type=latent.device.type, enabled=False):
                scores = gaussian_log_likelihood(
                    latent.detach().float(),
                    prior_mean.detach().float(),
                    prior_log_scale.detach().float(),
                )
            if not torch.isfinite(scores).all():
                raise RuntimeError("MAS likelihood scores contain NaN/Inf")
            valid = alignment_mask(
                acoustic_lengths,
                text_lengths,
                scores.shape[1],
                scores.shape[2],
            )
            if noise_scale > 0.0:
                standard_deviations = []
                for batch_index in range(scores.shape[0]):
                    standard_deviations.append(
                        scores[batch_index][valid[batch_index]].std(unbiased=False)
                    )
                score_std = torch.stack(standard_deviations).view(-1, 1, 1)
                noise = torch.randn(
                    scores.shape,
                    dtype=scores.dtype,
                    device=scores.device,
                    generator=generator,
                )
                scores = scores + noise * score_std * noise_scale * valid
            path = maximum_path(scores, acoustic_lengths, text_lengths)
            path = path * valid.to(path.dtype)
            durations = path.sum(dim=1).long()
        return MASOutput(
            alignment=path.unsqueeze(1),
            durations=durations,
            noise_scale=noise_scale,
        )
