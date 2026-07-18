"""Composite generator-side Banhmi-TTS training graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from banhmi_tts.model import (
    AlignmentFrameExpander,
    BigVGANGenerator,
    DurationPredictor,
    StochasticDurationPredictor,
    JointProsodyPredictor,
    MonotonicAlignmentSearch,
    PhonemeProsodyTargetBuilder,
    PosteriorEncoder,
    PredictorLosses,
    TrainingSegments,
    TransformerCouplingFlow,
    TextEncoder,
    compute_predictor_losses,
    decode_durations,
    durations_to_path,
    kl_divergence_loss,
    normalize_log_f0,
    duration_targets,
    slice_training_segments,
)

from .config import ModelConfig


@dataclass
class GeneratorForwardOutput:
    generated_audio: torch.Tensor
    target_audio: torch.Tensor
    kl_loss: torch.Tensor
    predictor_losses: PredictorLosses
    alignment: torch.Tensor
    durations: torch.Tensor
    segments: TrainingSegments
    mas_noise_scale: float
    prosody_teacher_forcing_ratio: float
    text_hidden: torch.Tensor
    text_mask: torch.Tensor
    target_log_duration: torch.Tensor
    predicted_log_duration: torch.Tensor


@dataclass
class ReconstructionOutput:
    generated_audio: torch.Tensor
    target_audio: torch.Tensor


@dataclass
class InferenceOutput:
    """Waveform and intermediate values useful for inference diagnostics."""

    audio: torch.Tensor
    audio_lengths: torch.Tensor
    frame_lengths: torch.Tensor
    durations: torch.Tensor
    alignment: torch.Tensor
    predicted_log_duration: torch.Tensor
    normalized_log_f0: torch.Tensor
    voicing_probability: torch.Tensor
    prior_space_latent: torch.Tensor
    decoder_latent: torch.Tensor


class BanhmiTTSModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.text_encoder = TextEncoder(config.text_encoder)
        self.posterior_encoder = PosteriorEncoder(config.posterior_encoder)
        self.flow = TransformerCouplingFlow(config.flow)
        self.mas = MonotonicAlignmentSearch(config.mas)
        self.prosody_target_builder = PhonemeProsodyTargetBuilder(
            config.prosody_targets
        )
        self.duration_predictor = (
            StochasticDurationPredictor(config.predictor)
            if config.predictor.duration_type == "stochastic"
            else DurationPredictor(config.predictor)
        )
        self.prosody_predictor = JointProsodyPredictor(config.predictor)
        self.frame_expander = AlignmentFrameExpander()
        self.generator = BigVGANGenerator(config.generator)

    def forward(
        self,
        batch: Dict[str, Any],
        *,
        global_step: int,
        prosody_teacher_forcing_ratio: float,
        random_generator: Optional[torch.Generator] = None,
    ) -> GeneratorForwardOutput:
        """DDP-compatible entry point for the complete generator training graph."""
        return self.forward_train(
            batch,
            global_step=global_step,
            prosody_teacher_forcing_ratio=prosody_teacher_forcing_ratio,
            random_generator=random_generator,
        )

    def forward_train(
        self,
        batch: Dict[str, Any],
        *,
        global_step: int,
        prosody_teacher_forcing_ratio: float,
        random_generator: Optional[torch.Generator] = None,
    ) -> GeneratorForwardOutput:
        if not 0.0 <= prosody_teacher_forcing_ratio <= 1.0:
            raise ValueError("prosody_teacher_forcing_ratio must be in [0, 1]")
        text = self.text_encoder(batch["phoneme_ids"], batch["phoneme_lengths"])
        posterior = self.posterior_encoder(
            batch["spectrogram"],
            batch["spectrogram_lengths"],
            sample=True,
            generator=random_generator,
        )
        prior_space = self.flow(posterior.latent, posterior.mask)
        alignment = self.mas(
            prior_space.latent,
            text.prior_mean,
            text.prior_log_scale,
            batch["spectrogram_lengths"],
            batch["phoneme_lengths"],
            global_step=global_step,
            generator=random_generator,
        )
        targets = self.prosody_target_builder(
            alignment.alignment,
            batch["log_f0"],
            batch["log_f0_continuous"],
            batch["voiced"],
            batch["spectrogram_lengths"],
            batch["phoneme_lengths"],
        )
        normalized_token_pitch = normalize_log_f0(
            targets.continuous_log_f0,
            targets.mask,
            self.config.pitch_mean,
            self.config.pitch_standard_deviation,
        )
        if self.config.predictor.duration_type == "stochastic":
            duration_nll = self.duration_predictor(
                text.hidden,
                text.mask,
                targets.durations.to(text.hidden.dtype).unsqueeze(1),
                generator=random_generator,
            )
            duration_loss_override = duration_nll.sum() / text.mask.sum().clamp_min(1.0)
            predicted_log_duration = self.duration_predictor(
                text.hidden,
                text.mask,
                reverse=True,
                noise_scale=1.0,
                generator=random_generator,
            )
        else:
            predicted_log_duration = self.duration_predictor(text.hidden, text.mask)
            duration_loss_override = None
        predicted_prosody = self.prosody_predictor(text.hidden, text.mask)
        predictor_losses = compute_predictor_losses(
            predicted_log_duration,
            predicted_prosody,
            targets.durations,
            normalized_token_pitch,
            targets.voiced_ratio,
            targets.mask,
            self.config.predictor_losses,
            duration_loss_override=duration_loss_override,
        )
        expanded = self.frame_expander(
            alignment.alignment,
            text.prior_mean,
            text.prior_log_scale,
            predicted_prosody,
            posterior.mask,
        )
        kl_loss = kl_divergence_loss(
            prior_space.latent,
            posterior.posterior_mean,
            posterior.posterior_log_scale,
            expanded.prior_mean,
            expanded.prior_log_scale,
            posterior.mask,
            flow_log_determinant=prior_space.log_determinant,
        )
        normalized_frame_target = normalize_log_f0(
            batch["log_f0_continuous"],
            batch["spectrogram_mask"].squeeze(1),
            self.config.pitch_mean,
            self.config.pitch_standard_deviation,
        )
        predicted_frame_pitch = expanded.normalized_continuous_log_f0
        predicted_frame_voicing = expanded.voicing_probability
        ratio = prosody_teacher_forcing_ratio
        generator_pitch = (
            ratio * normalized_frame_target
            + (1.0 - ratio) * predicted_frame_pitch
        )
        generator_voicing = (
            ratio * batch["voiced"].to(predicted_frame_voicing.dtype)
            + (1.0 - ratio) * predicted_frame_voicing
        )
        segments = slice_training_segments(
            posterior.latent,
            generator_pitch,
            generator_voicing,
            batch["audio"],
            batch["spectrogram_lengths"],
            self.config.segment,
            generator=random_generator,
        )
        generated_audio = self.generator(
            segments.latent,
            segments.normalized_log_f0,
            segments.voicing,
        )
        return GeneratorForwardOutput(
            generated_audio=generated_audio,
            target_audio=segments.audio,
            kl_loss=kl_loss,
            predictor_losses=predictor_losses,
            alignment=alignment.alignment,
            durations=alignment.durations,
            segments=segments,
            mas_noise_scale=alignment.noise_scale,
            prosody_teacher_forcing_ratio=ratio,
            text_hidden=text.hidden,
            text_mask=text.mask,
            target_log_duration=duration_targets(
                targets.durations, text.mask.squeeze(1).bool()
            ).unsqueeze(1),
            predicted_log_duration=predicted_log_duration.unsqueeze(1),
        )

    def reconstruct_posterior_mean(
        self,
        batch: Dict[str, Any],
        *,
        full_utterance: bool,
        frame_starts: Optional[torch.Tensor] = None,
        sample_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> ReconstructionOutput:
        """Decode deterministic posterior means for reconstruction diagnostics."""
        posterior = self.posterior_encoder(
            batch["spectrogram"],
            batch["spectrogram_lengths"],
            sample=sample_posterior,
            generator=generator,
        )
        latent_source = (
            posterior.latent if sample_posterior else posterior.posterior_mean
        )
        normalized_pitch = normalize_log_f0(
            batch["log_f0_continuous"],
            batch["spectrogram_mask"].squeeze(1),
            self.config.pitch_mean,
            self.config.pitch_standard_deviation,
        )
        voicing = batch["voiced"].to(posterior.latent.dtype)
        if full_utterance:
            frames = int(batch["spectrogram_lengths"][0])
            latent = latent_source[:1, :, :frames]
            pitch = normalized_pitch[:1, :frames]
            voiced = voicing[:1, :frames]
            generated = self.generator(latent, pitch, voiced)
            samples = frames * self.config.segment.hop_length
            target = batch["audio"][:1, :, :samples]
        else:
            if frame_starts is None:
                frame_starts = torch.zeros(
                    posterior.latent.shape[0],
                    dtype=torch.long,
                    device=posterior.latent.device,
                )
            segments = slice_training_segments(
                latent_source,
                normalized_pitch,
                voicing,
                batch["audio"],
                batch["spectrogram_lengths"],
                self.config.segment,
                frame_starts=frame_starts,
            )
            generated = self.generator(
                segments.latent,
                segments.normalized_log_f0,
                segments.voicing,
            )
            target = segments.audio
        return ReconstructionOutput(generated_audio=generated, target_audio=target)

    @torch.inference_mode()
    def infer(
        self,
        phoneme_ids: torch.Tensor,
        phoneme_lengths: torch.Tensor,
        *,
        noise_scale: float = 0.667,
        duration_noise_scale: float = 0.8,
        length_scale: float = 1.0,
        maximum_frames: int = 10_000,
        generator: Optional[torch.Generator] = None,
    ) -> InferenceOutput:
        """Synthesize waveforms from padded phoneme-id sequences.

        The path follows VITS inference: predict durations and token prosody,
        expand the text prior to frames, sample in prior space, invert the
        normalizing flow, then decode the resulting latent with BigVGAN.
        """
        if noise_scale < 0.0 or duration_noise_scale < 0.0:
            raise ValueError("inference noise scales cannot be negative")
        if maximum_frames <= 0:
            raise ValueError("maximum_frames must be positive")

        text = self.text_encoder(phoneme_ids, phoneme_lengths)
        if self.config.predictor.duration_type == "stochastic":
            predicted_log_duration = self.duration_predictor(
                text.hidden,
                text.mask,
                reverse=True,
                noise_scale=duration_noise_scale,
                generator=generator,
            )
        else:
            predicted_log_duration = self.duration_predictor(
                text.hidden, text.mask
            )
        durations = decode_durations(
            predicted_log_duration,
            text.mask.squeeze(1).bool(),
            length_scale=length_scale,
            maximum_log_duration=self.config.predictor.maximum_log_duration,
            maximum_duration_frames=(
                self.config.predictor.maximum_duration_frames
            ),
        )
        frame_lengths = durations.sum(dim=1)
        if torch.any(frame_lengths > maximum_frames):
            longest = int(frame_lengths.max().item())
            raise ValueError(
                f"predicted sequence has {longest} frames, exceeding "
                f"maximum_frames={maximum_frames}; shorten the text or reduce "
                "length_scale"
            )
        path = durations_to_path(durations, text.mask)
        predicted_prosody = self.prosody_predictor(text.hidden, text.mask)
        expanded = self.frame_expander(
            path.alignment,
            text.prior_mean,
            text.prior_log_scale,
            predicted_prosody,
            path.frame_mask,
        )
        if noise_scale == 0.0:
            prior_space_latent = expanded.prior_mean
        else:
            prior_noise = torch.randn(
                expanded.prior_mean.shape,
                device=expanded.prior_mean.device,
                dtype=expanded.prior_mean.dtype,
                generator=generator,
            )
            prior_space_latent = expanded.prior_mean + (
                prior_noise
                * torch.exp(expanded.prior_log_scale)
                * noise_scale
            )
        prior_space_latent = prior_space_latent * path.frame_mask
        decoder_latent = self.flow(
            prior_space_latent, path.frame_mask, reverse=True
        ).latent
        audio = self.generator(
            decoder_latent,
            expanded.normalized_continuous_log_f0,
            expanded.voicing_probability,
        )
        audio_lengths = path.frame_lengths * self.config.segment.hop_length
        sample_positions = torch.arange(
            audio.shape[-1], device=audio.device
        ).unsqueeze(0)
        audio_mask = sample_positions < audio_lengths.unsqueeze(1)
        audio = audio * audio_mask.unsqueeze(1).to(audio.dtype)
        return InferenceOutput(
            audio=audio,
            audio_lengths=audio_lengths,
            frame_lengths=path.frame_lengths,
            durations=durations,
            alignment=path.alignment,
            predicted_log_duration=predicted_log_duration,
            normalized_log_f0=expanded.normalized_continuous_log_f0,
            voicing_probability=expanded.voicing_probability,
            prior_space_latent=prior_space_latent,
            decoder_latent=decoder_latent,
        )

    def remove_weight_norm(self) -> None:
        self.posterior_encoder.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.generator.remove_weight_norm()
