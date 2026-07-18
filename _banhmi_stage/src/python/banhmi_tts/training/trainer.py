"""Two-optimizer adversarial training step."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from typing import Any, Dict

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from banhmi_tts.model import (
    BigVGANDiscriminator,
    DurationDiscriminator,
    MultiResolutionSpectralLoss,
    MelReconstructionLoss,
    discriminator_least_squares_loss,
    feature_matching_loss,
    generator_least_squares_loss,
)

from .config import TrainingConfig
from .model import BanhmiTTSModel, ReconstructionOutput


@dataclass
class TrainingStepMetrics:
    global_step: int
    discriminator_loss: float
    generator_loss: float
    adversarial_loss: float
    feature_matching_loss: float
    spectral_loss: float
    mel_loss: float
    kl_loss: float
    weighted_kl_loss: float
    c_kl: float
    duration_loss: float
    duration_discriminator_loss: float
    duration_adversarial_loss: float
    pitch_loss: float
    voicing_loss: float
    mas_noise_scale: float
    prosody_teacher_forcing_ratio: float
    snake_alpha_min: float
    snake_alpha_max: float
    snake_beta_min: float
    snake_beta_max: float


@dataclass
class ValidationStepResult:
    metrics: TrainingStepMetrics
    generated_audio: torch.Tensor
    target_audio: torch.Tensor


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _wrap_distributed(module: nn.Module, device: torch.device) -> nn.Module:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return module
    device_ids = [device.index] if device.type == "cuda" else None
    output_device = device.index if device.type == "cuda" else None
    return DistributedDataParallel(
        module,
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=False,
    )


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _duration_discriminator_loss(real, fake) -> torch.Tensor:
    return torch.stack(
        [(real_score.float() - 1.0).square().mean() + fake_score.float().square().mean()
         for real_score, fake_score in zip(real, fake)]
    ).sum()


def _duration_generator_loss(fake) -> torch.Tensor:
    return torch.stack(
        [(score.float() - 1.0).square().mean() for score in fake]
    ).sum()


def _clip_or_check_gradients(
    parameters,
    maximum_norm: float | None,
    device: torch.device,
) -> tuple[bool, torch.Tensor]:
    """Clip when requested and synchronize finite status across DDP ranks."""
    total_norm = torch.nn.utils.clip_grad_norm_(
        list(parameters),
        float("inf") if maximum_norm is None else maximum_norm,
        error_if_nonfinite=False,
    )
    finite = torch.isfinite(total_norm).to(device=device, dtype=torch.int32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
    return bool(finite.item()), total_norm.detach()


def _report_skipped_update(stage: str, global_step: int, total_norm: torch.Tensor) -> None:
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    if rank == 0:
        print(
            json.dumps(
                {
                    "training_warning": "nonfinite_gradient_update_skipped",
                    "stage": stage,
                    "global_step": global_step + 1,
                    "gradient_norm": float(total_norm),
                }
            ),
            flush=True,
        )


class BanhmiTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        config.validate()
        self.config = config
        self.device = device
        self.model_module = BanhmiTTSModel(config.model).to(device)
        self.discriminator_module = BigVGANDiscriminator(config.discriminator).to(device)
        self.duration_discriminator_module = (
            DurationDiscriminator(
                config.model.predictor.input_channels,
                config.model.predictor.hidden_channels,
                config.model.predictor.kernel_size,
                config.model.predictor.dropout,
            ).to(device)
            if config.model.predictor.duration_type == "stochastic"
            else None
        )
        self.model = _wrap_distributed(self.model_module, device)
        self.discriminator = _wrap_distributed(self.discriminator_module, device)
        self.duration_discriminator = (
            _wrap_distributed(self.duration_discriminator_module, device)
            if self.duration_discriminator_module is not None
            else None
        )
        self.spectral_loss = MultiResolutionSpectralLoss(config.spectral_loss).to(device)
        self.mel_loss = MelReconstructionLoss(config.mel_loss).to(device)
        optimization = config.optimization
        self.generator_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=optimization.learning_rate,
            betas=optimization.betas,
            weight_decay=optimization.weight_decay,
            eps=optimization.epsilon,
        )
        discriminator_parameters = (
            itertools.chain(
                self.discriminator.parameters(),
                self.duration_discriminator.parameters(),
            )
            if self.duration_discriminator is not None
            else self.discriminator.parameters()
        )
        self.discriminator_optimizer = torch.optim.AdamW(
            discriminator_parameters,
            lr=optimization.learning_rate,
            betas=optimization.betas,
            weight_decay=optimization.weight_decay,
            eps=optimization.epsilon,
        )
        self.generator_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.generator_optimizer, gamma=optimization.learning_rate_decay
        )
        self.discriminator_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.discriminator_optimizer, gamma=optimization.learning_rate_decay
        )
        self.mixed_precision_enabled = (
            optimization.use_mixed_precision and device.type == "cuda"
        )
        self.autocast_dtype = (
            torch.bfloat16
            if self.mixed_precision_enabled and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        scaler_enabled = (
            self.mixed_precision_enabled and self.autocast_dtype == torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            device.type, enabled=scaler_enabled
        )

    def step_learning_rate_schedulers(self) -> None:
        """Step once per completed epoch, matching EdgeTTS/Piper."""
        self.generator_scheduler.step()
        self.discriminator_scheduler.step()

    def train_step(
        self,
        batch: Dict[str, Any],
        global_step: int,
        *,
        random_generator: torch.Generator | None = None,
        collect_metrics: bool = True,
    ) -> TrainingStepMetrics | None:
        self.model.train()
        self.discriminator.train()
        if self.duration_discriminator is not None:
            self.duration_discriminator.train()
        batch = _move_batch(batch, self.device)
        optimization = self.config.optimization
        c_kl = optimization.c_kl
        teacher_forcing = optimization.prosody_teacher_forcing_ratio(global_step)
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.mixed_precision_enabled,
        ):
            forward = self.model(
                batch,
                global_step=global_step,
                prosody_teacher_forcing_ratio=teacher_forcing,
                random_generator=random_generator,
            )

        # Match EdgeTTS manual optimization: update G first against the current
        # discriminator, then update D from the cached (pre-G-step) waveform.
        _set_requires_grad(self.discriminator_module, False)
        if self.duration_discriminator_module is not None:
            _set_requires_grad(self.duration_discriminator_module, False)
        self.generator_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.mixed_precision_enabled,
        ):
            with torch.no_grad():
                real_generator = self.discriminator_module(forward.target_audio)
            fake_generator = self.discriminator_module(forward.generated_audio)
            if self.duration_discriminator_module is not None:
                _, duration_fake_for_generator = self.duration_discriminator_module(
                    forward.text_hidden,
                    forward.text_mask,
                    forward.target_log_duration,
                    forward.predicted_log_duration,
                )
            else:
                duration_fake_for_generator = None
        # Match EdgeTTS Config H: discriminator forwards may use BF16, while
        # loss arithmetic and the final sum stay in FP32.
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            adversarial = generator_least_squares_loss(fake_generator)
            feature_matching = feature_matching_loss(
                real_generator, fake_generator
            )
            spectral = (
                self.spectral_loss(forward.generated_audio, forward.target_audio)
                if optimization.spectral_weight > 0.0
                else forward.generated_audio.new_zeros(())
            )
            mel = self.mel_loss(forward.generated_audio, forward.target_audio)
            if duration_fake_for_generator is not None:
                duration_adversarial = _duration_generator_loss(
                    duration_fake_for_generator
                )
            else:
                duration_adversarial = mel.new_zeros(())
            generator_loss = (
                optimization.adversarial_weight * adversarial
                + optimization.feature_matching_weight * feature_matching
                + optimization.spectral_weight * spectral
                + mel
                + c_kl * forward.kl_loss
                + forward.predictor_losses.total
                + duration_adversarial
            )
        self.scaler.scale(generator_loss).backward()
        self.scaler.unscale_(self.generator_optimizer)
        generator_gradients_finite, generator_gradient_norm = _clip_or_check_gradients(
            self.model.parameters(),
            optimization.generator_gradient_clip,
            self.device,
        )
        if not generator_gradients_finite:
            self.generator_optimizer.zero_grad(set_to_none=True)
            _set_requires_grad(self.discriminator_module, True)
            if self.duration_discriminator_module is not None:
                _set_requires_grad(self.duration_discriminator_module, True)
            self.scaler.update()
            _report_skipped_update("generator", global_step, generator_gradient_norm)
            return None
        self.scaler.step(self.generator_optimizer)

        _set_requires_grad(self.discriminator_module, True)
        if self.duration_discriminator_module is not None:
            _set_requires_grad(self.duration_discriminator_module, True)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.mixed_precision_enabled,
        ):
            real_discriminator = self.discriminator(forward.target_audio)
            fake_discriminator = self.discriminator(forward.generated_audio.detach())
            if self.duration_discriminator is not None:
                duration_real, duration_fake = self.duration_discriminator(
                    forward.text_hidden.detach(),
                    forward.text_mask,
                    forward.target_log_duration.detach(),
                    forward.predicted_log_duration.detach(),
                )
            else:
                duration_real = duration_fake = None
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            discriminator_loss = discriminator_least_squares_loss(
                real_discriminator, fake_discriminator
            )
            if duration_real is not None and duration_fake is not None:
                duration_discriminator_loss = _duration_discriminator_loss(
                    duration_real, duration_fake
                )
                discriminator_loss = discriminator_loss + duration_discriminator_loss
            else:
                duration_discriminator_loss = discriminator_loss.new_zeros(())
        self.scaler.scale(discriminator_loss).backward()
        self.scaler.unscale_(self.discriminator_optimizer)
        discriminator_clip_parameters = list(self.discriminator.parameters())
        if self.duration_discriminator is not None:
            discriminator_clip_parameters += list(self.duration_discriminator.parameters())
        discriminator_gradients_finite, discriminator_gradient_norm = (
            _clip_or_check_gradients(
                discriminator_clip_parameters,
                optimization.discriminator_gradient_clip,
                self.device,
            )
        )
        if not discriminator_gradients_finite:
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            self.scaler.update()
            _report_skipped_update(
                "discriminator", global_step, discriminator_gradient_norm
            )
            return None
        self.scaler.step(self.discriminator_optimizer)
        self.scaler.update()

        # EdgeTTS keeps loss tensors on-device during ordinary steps. Converting
        # every scalar to Python and reading Snake parameters forces repeated
        # CUDA synchronizations, so only collect them on logging steps.
        if not collect_metrics:
            return None

        snake_statistics = self.model_module.generator.snake_parameter_statistics()
        return TrainingStepMetrics(
            global_step=global_step + 1,
            discriminator_loss=float(discriminator_loss.detach()),
            generator_loss=float(generator_loss.detach()),
            adversarial_loss=float(adversarial.detach()),
            feature_matching_loss=float(feature_matching.detach()),
            spectral_loss=float(spectral.detach()),
            mel_loss=float(mel.detach()),
            kl_loss=float(forward.kl_loss.detach()),
            weighted_kl_loss=float((c_kl * forward.kl_loss).detach()),
            c_kl=c_kl,
            duration_loss=float(forward.predictor_losses.duration.detach()),
            duration_discriminator_loss=float(duration_discriminator_loss.detach()),
            duration_adversarial_loss=float(duration_adversarial.detach()),
            pitch_loss=float(forward.predictor_losses.pitch.detach()),
            voicing_loss=float(forward.predictor_losses.voicing.detach()),
            mas_noise_scale=forward.mas_noise_scale,
            prosody_teacher_forcing_ratio=teacher_forcing,
            **snake_statistics,
        )

    @torch.no_grad()
    def reconstruct_posterior_mean(
        self,
        batch: Dict[str, Any],
        *,
        full_utterance: bool,
        sample_posterior: bool = False,
        random_generator: torch.Generator | None = None,
    ) -> ReconstructionOutput:
        self.model_module.eval()
        batch = _move_batch(batch, self.device)
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.mixed_precision_enabled,
        ):
            result = self.model_module.reconstruct_posterior_mean(
                batch,
                full_utterance=full_utterance,
                sample_posterior=sample_posterior,
                generator=random_generator,
            )
        return ReconstructionOutput(
            generated_audio=result.generated_audio.detach().cpu(),
            target_audio=result.target_audio.detach().cpu(),
        )

    @torch.no_grad()
    def reconstruction_spectral_loss(
        self, result: ReconstructionOutput
    ) -> float:
        generated = result.generated_audio.to(self.device)
        target = result.target_audio.to(self.device)
        return float(self.spectral_loss(generated, target))

    @torch.no_grad()
    def validate_step(
        self,
        batch: Dict[str, Any],
        global_step: int,
        *,
        random_generator: torch.Generator | None = None,
    ) -> ValidationStepResult:
        """Deterministic reconstruction validation when given a fixed generator."""
        self.model_module.eval()
        self.discriminator_module.eval()
        if self.duration_discriminator_module is not None:
            self.duration_discriminator_module.eval()
        batch = _move_batch(batch, self.device)
        optimization = self.config.optimization
        teacher_forcing = optimization.prosody_teacher_forcing_ratio(global_step)
        c_kl = optimization.c_kl
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.mixed_precision_enabled,
        ):
            forward = self.model_module.forward_train(
                batch,
                global_step=global_step,
                prosody_teacher_forcing_ratio=teacher_forcing,
                random_generator=random_generator,
            )
            real = self.discriminator_module(forward.target_audio)
            fake = self.discriminator_module(forward.generated_audio)
            if self.duration_discriminator_module is not None:
                duration_real, duration_fake = self.duration_discriminator_module(
                    forward.text_hidden,
                    forward.text_mask,
                    forward.target_log_duration,
                    forward.predicted_log_duration,
                )
            else:
                duration_real = duration_fake = None
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            discriminator_loss = discriminator_least_squares_loss(real, fake)
            adversarial = generator_least_squares_loss(fake)
            feature_matching = feature_matching_loss(real, fake)
            spectral = (
                self.spectral_loss(forward.generated_audio, forward.target_audio)
                if optimization.spectral_weight > 0.0
                else forward.generated_audio.new_zeros(())
            )
            mel = self.mel_loss(forward.generated_audio, forward.target_audio)
            if duration_real is not None and duration_fake is not None:
                duration_discriminator_loss = _duration_discriminator_loss(
                    duration_real, duration_fake
                )
                duration_adversarial = _duration_generator_loss(duration_fake)
                discriminator_loss = discriminator_loss + duration_discriminator_loss
            else:
                duration_discriminator_loss = mel.new_zeros(())
                duration_adversarial = mel.new_zeros(())
            generator_loss = (
                optimization.adversarial_weight * adversarial
                + optimization.feature_matching_weight * feature_matching
                + optimization.spectral_weight * spectral
                + mel
                + c_kl * forward.kl_loss
                + forward.predictor_losses.total
                + duration_adversarial
            )
        snake_statistics = self.model_module.generator.snake_parameter_statistics()
        metrics = TrainingStepMetrics(
            global_step=global_step,
            discriminator_loss=float(discriminator_loss),
            generator_loss=float(generator_loss),
            adversarial_loss=float(adversarial),
            feature_matching_loss=float(feature_matching),
            spectral_loss=float(spectral),
            mel_loss=float(mel),
            kl_loss=float(forward.kl_loss),
            weighted_kl_loss=float(c_kl * forward.kl_loss),
            c_kl=c_kl,
            duration_loss=float(forward.predictor_losses.duration),
            duration_discriminator_loss=float(duration_discriminator_loss),
            duration_adversarial_loss=float(duration_adversarial),
            pitch_loss=float(forward.predictor_losses.pitch),
            voicing_loss=float(forward.predictor_losses.voicing),
            mas_noise_scale=forward.mas_noise_scale,
            prosody_teacher_forcing_ratio=teacher_forcing,
            **snake_statistics,
        )
        return ValidationStepResult(
            metrics=metrics,
            generated_audio=forward.generated_audio.detach().cpu(),
            target_audio=forward.target_audio.detach().cpu(),
        )
