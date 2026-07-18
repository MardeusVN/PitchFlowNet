"""Validated model and optimization configurations."""

from __future__ import annotations

from dataclasses import dataclass, field

from banhmi_tts.model import (
    DiscriminatorConfig,
    FlowConfig,
    GeneratorConfig,
    MASConfig,
    MultiResolutionSpectralLossConfig,
    MelReconstructionLossConfig,
    PosteriorEncoderConfig,
    PredictorConfig,
    PredictorLossConfig,
    ProsodyTargetConfig,
    SegmentConfig,
    TextEncoderConfig,
)


@dataclass(frozen=True)
class ModelConfig:
    text_encoder: TextEncoderConfig
    posterior_encoder: PosteriorEncoderConfig
    flow: FlowConfig
    predictor: PredictorConfig
    generator: GeneratorConfig
    segment: SegmentConfig
    mas: MASConfig = field(default_factory=MASConfig)
    prosody_targets: ProsodyTargetConfig = field(default_factory=ProsodyTargetConfig)
    predictor_losses: PredictorLossConfig = field(default_factory=PredictorLossConfig)
    pitch_mean: float = 0.0
    pitch_standard_deviation: float = 1.0

    @staticmethod
    def default(
        num_symbols: int,
        pitch_mean: float,
        pitch_standard_deviation: float,
    ) -> "ModelConfig":
        return ModelConfig(
            text_encoder=TextEncoderConfig(num_symbols=num_symbols),
            posterior_encoder=PosteriorEncoderConfig(),
            flow=FlowConfig(),
            predictor=PredictorConfig(),
            generator=GeneratorConfig(),
            segment=SegmentConfig(),
            pitch_mean=pitch_mean,
            pitch_standard_deviation=pitch_standard_deviation,
        )

    @staticmethod
    def from_dict(value: dict) -> "ModelConfig":
        """Reconstruct the nested model config stored in a checkpoint."""
        config = ModelConfig(
            text_encoder=TextEncoderConfig(**value["text_encoder"]),
            posterior_encoder=PosteriorEncoderConfig(**value["posterior_encoder"]),
            flow=FlowConfig(**value["flow"]),
            predictor=PredictorConfig(**value["predictor"]),
            generator=GeneratorConfig(**value["generator"]),
            segment=SegmentConfig(**value["segment"]),
            mas=MASConfig(**value.get("mas", {})),
            prosody_targets=ProsodyTargetConfig(**value.get("prosody_targets", {})),
            predictor_losses=PredictorLossConfig(**value.get("predictor_losses", {})),
            pitch_mean=float(value.get("pitch_mean", 0.0)),
            pitch_standard_deviation=float(
                value.get("pitch_standard_deviation", 1.0)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        self.text_encoder.validate()
        self.posterior_encoder.validate()
        self.flow.validate()
        self.predictor.validate()
        self.generator.validate()
        self.segment.validate()
        self.mas.validate()
        self.prosody_targets.validate()
        self.predictor_losses.validate()
        latent_channels = {
            self.text_encoder.latent_channels,
            self.posterior_encoder.latent_channels,
            self.flow.channels,
            self.generator.latent_channels,
        }
        if len(latent_channels) != 1:
            raise ValueError("text/posterior/flow/generator latent channels must match")
        if self.text_encoder.hidden_channels != self.predictor.input_channels:
            raise ValueError("predictor input channels must match text hidden channels")
        if self.generator.total_upsample_factor != self.segment.hop_length:
            raise ValueError("generator upsample factor must equal segment hop_length")
        if self.pitch_standard_deviation <= 0.0:
            raise ValueError("pitch_standard_deviation must be positive")


@dataclass(frozen=True)
class OptimizationConfig:
    learning_rate: float = 2e-4
    betas: tuple[float, float] = (0.8, 0.99)
    weight_decay: float = 0.01
    epsilon: float = 1e-9
    learning_rate_decay: float = 0.999875
    # EdgeTTS Config H leaves clipping disabled. Existing checkpoints trained
    # with clipping must pass their saved values explicitly when resumed.
    generator_gradient_clip: float | None = None
    discriminator_gradient_clip: float | None = None
    use_mixed_precision: bool = True
    # Match the VITS/EdgeTTS name and default: KL is active from the first step.
    c_kl: float = 1.0
    adversarial_weight: float = 1.0
    feature_matching_weight: float = 1.0
    # EdgeTTS uses mel L1 plus GAN/FM losses, without an extra MR-STFT
    # reconstruction objective. The module remains available for diagnostics.
    spectral_weight: float = 0.0
    prosody_teacher_forcing_start: float = 1.0
    prosody_teacher_forcing_end: float = 0.0
    prosody_teacher_forcing_steps: int = 100_000

    def validate(self) -> None:
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid learning rate or weight decay")
        if self.epsilon <= 0.0 or not 0.0 < self.learning_rate_decay <= 1.0:
            raise ValueError("invalid optimizer epsilon or learning-rate decay")
        if not all(0.0 <= beta < 1.0 for beta in self.betas):
            raise ValueError("optimizer betas must be in [0, 1)")
        gradient_clips = (
            self.generator_gradient_clip,
            self.discriminator_gradient_clip,
        )
        if any(value is not None and value <= 0.0 for value in gradient_clips):
            raise ValueError("gradient clipping thresholds must be positive when set")
        if min(
            self.c_kl,
            self.adversarial_weight,
            self.feature_matching_weight,
            self.spectral_weight,
        ) < 0.0:
            raise ValueError("training loss weights cannot be negative")
        if not 0.0 <= self.prosody_teacher_forcing_start <= 1.0:
            raise ValueError("teacher forcing start must be in [0, 1]")
        if not 0.0 <= self.prosody_teacher_forcing_end <= 1.0:
            raise ValueError("teacher forcing end must be in [0, 1]")
        if self.prosody_teacher_forcing_steps < 0:
            raise ValueError("teacher forcing steps cannot be negative")

    def prosody_teacher_forcing_ratio(self, global_step: int) -> float:
        if global_step < 0:
            raise ValueError("global_step cannot be negative")
        if self.prosody_teacher_forcing_steps == 0:
            return self.prosody_teacher_forcing_end
        progress = min(global_step / self.prosody_teacher_forcing_steps, 1.0)
        return self.prosody_teacher_forcing_start + progress * (
            self.prosody_teacher_forcing_end
            - self.prosody_teacher_forcing_start
        )

@dataclass(frozen=True)
class TrainingConfig:
    model: ModelConfig
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    spectral_loss: MultiResolutionSpectralLossConfig = field(
        default_factory=MultiResolutionSpectralLossConfig
    )
    mel_loss: MelReconstructionLossConfig = field(
        default_factory=MelReconstructionLossConfig
    )
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    def validate(self) -> None:
        self.model.validate()
        self.discriminator.validate()
        self.spectral_loss.validate()
        self.mel_loss.validate()
        self.optimization.validate()
