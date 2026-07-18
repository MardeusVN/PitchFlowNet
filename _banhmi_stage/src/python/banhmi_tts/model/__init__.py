"""Neural network building blocks for Banhmi-TTS."""

from .adversarial_losses import (
    AdversarialLosses,
    MultiResolutionSpectralLoss,
    MultiResolutionSpectralLossConfig,
    MelReconstructionLoss,
    MelReconstructionLossConfig,
    discriminator_least_squares_loss,
    feature_matching_loss,
    generator_least_squares_loss,
)
from .alignment import MASConfig, MASOutput, MonotonicAlignmentSearch
from .flow import FlowConfig, FlowOutput, TransformerCouplingFlow
from .frame_expansion import (
    AlignmentFrameExpander,
    DurationPath,
    FrameExpansionOutput,
    denormalize_log_f0,
    durations_to_path,
    expand_token_features,
    expand_token_scalars,
    kl_divergence_loss,
)
from .generator import BigVGANGenerator, GeneratorConfig
from .discriminators import (
    BigVGANDiscriminator,
    DiscriminatorConfig,
    DiscriminatorOutput,
)
from .duration import DurationDiscriminator, StochasticDurationPredictor
from .posterior_encoder import (
    PosteriorEncoder,
    PosteriorEncoderConfig,
    PosteriorEncoderOutput,
)
from .predictors import (
    DurationPredictor,
    JointProsodyPredictor,
    PredictorConfig,
    PredictorLossConfig,
    PredictorLosses,
    ProsodyPredictorOutput,
    compute_predictor_losses,
    decode_durations,
    duration_targets,
)
from .prosody import (
    PhonemeProsodyTargetBuilder,
    PhonemeProsodyTargets,
    ProsodyTargetConfig,
    normalize_log_f0,
)
from .segments import SegmentConfig, TrainingSegments, slice_training_segments
from .text_encoder import TextEncoder, TextEncoderConfig, TextEncoderOutput

__all__ = [
    "FlowConfig",
    "FlowOutput",
    "AlignmentFrameExpander",
    "AdversarialLosses",
    "BigVGANGenerator",
    "BigVGANDiscriminator",
    "DiscriminatorConfig",
    "DiscriminatorOutput",
    "DurationPath",
    "FrameExpansionOutput",
    "GeneratorConfig",
    "DurationPredictor",
    "DurationDiscriminator",
    "StochasticDurationPredictor",
    "JointProsodyPredictor",
    "MASConfig",
    "MASOutput",
    "MonotonicAlignmentSearch",
    "MultiResolutionSpectralLoss",
    "MultiResolutionSpectralLossConfig",
    "MelReconstructionLoss",
    "MelReconstructionLossConfig",
    "PosteriorEncoder",
    "PosteriorEncoderConfig",
    "PosteriorEncoderOutput",
    "PredictorConfig",
    "PredictorLossConfig",
    "PredictorLosses",
    "ProsodyPredictorOutput",
    "PhonemeProsodyTargetBuilder",
    "PhonemeProsodyTargets",
    "ProsodyTargetConfig",
    "SegmentConfig",
    "TextEncoder",
    "TextEncoderConfig",
    "TextEncoderOutput",
    "TrainingSegments",
    "TransformerCouplingFlow",
    "compute_predictor_losses",
    "denormalize_log_f0",
    "decode_durations",
    "duration_targets",
    "durations_to_path",
    "discriminator_least_squares_loss",
    "expand_token_features",
    "expand_token_scalars",
    "feature_matching_loss",
    "generator_least_squares_loss",
    "kl_divergence_loss",
    "normalize_log_f0",
    "slice_training_segments",
]
