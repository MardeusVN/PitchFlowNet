import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from banhmi_tts.model import (
    DiscriminatorConfig,
    DurationPredictor,
    FlowConfig,
    GeneratorConfig,
    MultiResolutionSpectralLossConfig,
    MelReconstructionLossConfig,
    PosteriorEncoderConfig,
    PredictorConfig,
    SegmentConfig,
    TextEncoderConfig,
)
from banhmi_tts.training import (
    BanhmiTrainer,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
    load_checkpoint,
    save_checkpoint,
)
from banhmi_tts.training.trainer import _clip_or_check_gradients


def small_config() -> TrainingConfig:
    model = ModelConfig(
        text_encoder=TextEncoderConfig(
            num_symbols=20,
            latent_channels=8,
            hidden_channels=16,
            filter_channels=32,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        ),
        posterior_encoder=PosteriorEncoderConfig(
            input_channels=9,
            latent_channels=8,
            hidden_channels=16,
            kernel_size=3,
            num_layers=2,
            use_weight_norm=False,
        ),
        flow=FlowConfig(
            channels=8,
            hidden_channels=16,
            filter_channels=32,
            num_heads=2,
            transformer_layers=1,
            kernel_size=3,
            wavenet_layers=1,
            num_flows=2,
            dropout=0.0,
            use_weight_norm=False,
        ),
        predictor=PredictorConfig(
            input_channels=16,
            hidden_channels=16,
            num_layers=1,
            dropout=0.0,
        ),
        generator=GeneratorConfig(
            latent_channels=8,
            initial_channels=16,
            upsample_rates=(4, 2),
            upsample_kernel_sizes=(8, 4),
            amp_kernel_sizes=(3,),
            amp_dilations=((1,),),
            use_weight_norm=False,
        ),
        segment=SegmentConfig(segment_frames=3, hop_length=8),
        pitch_mean=5.0,
        pitch_standard_deviation=0.5,
    )
    return TrainingConfig(
        model=model,
        discriminator=DiscriminatorConfig(
            periods=(2,), resolutions=((32, 8, 32),), use_weight_norm=False
        ),
        spectral_loss=MultiResolutionSpectralLossConfig(
            resolutions=((32, 8, 32),)
        ),
        mel_loss=MelReconstructionLossConfig(
            n_fft=32, hop_length=8, win_length=32, n_mels=8
        ),
        optimization=OptimizationConfig(use_mixed_precision=False),
    )


def small_batch() -> dict:
    voiced = torch.tensor(
        [[True, True, False, True, False, True], [True, False, True, True, True, False]]
    )
    continuous = torch.tensor(
        [[5.0, 5.1, 5.2, 5.3, 5.4, 5.5], [4.8, 4.9, 5.0, 5.1, 5.2, 5.3]]
    )
    return {
        "phoneme_ids": torch.tensor([[1, 5, 2], [1, 6, 2]]),
        "phoneme_lengths": torch.tensor([3, 3]),
        "spectrogram": torch.randn(2, 9, 6),
        "spectrogram_lengths": torch.tensor([6, 6]),
        "spectrogram_mask": torch.ones(2, 1, 6, dtype=torch.bool),
        "audio": torch.randn(2, 1, 48).clamp(-1, 1),
        "log_f0": torch.where(voiced, continuous, torch.zeros_like(continuous)),
        "log_f0_continuous": continuous,
        "voiced": voiced,
    }


class TrainingTests(unittest.TestCase):
    def test_deterministic_duration_baseline_remains_available(self):
        config = small_config()
        config = replace(
            config,
            model=replace(
                config.model,
                predictor=replace(config.model.predictor, duration_type="deterministic"),
            ),
        )
        trainer = BanhmiTrainer(config, torch.device("cpu"))
        self.assertIsInstance(trainer.model.duration_predictor, DurationPredictor)
        self.assertIsNone(trainer.duration_discriminator)
        metrics = trainer.train_step(
            small_batch(),
            global_step=0,
            random_generator=torch.Generator().manual_seed(43),
        )
        self.assertTrue(torch.isfinite(torch.tensor(metrics.duration_loss)))

    def test_train_step_is_finite_and_updates_parameters(self):
        trainer = BanhmiTrainer(small_config(), torch.device("cpu"))
        update_order = []
        generator_step = trainer.generator_optimizer.step
        discriminator_step = trainer.discriminator_optimizer.step
        trainer.generator_optimizer.step = lambda *args, **kwargs: (
            update_order.append("generator"), generator_step(*args, **kwargs)
        )[1]
        trainer.discriminator_optimizer.step = lambda *args, **kwargs: (
            update_order.append("discriminator"), discriminator_step(*args, **kwargs)
        )[1]
        before = trainer.model.generator.output_projection.weight.detach().clone()
        metrics = trainer.train_step(
            small_batch(),
            global_step=0,
            random_generator=torch.Generator().manual_seed(3),
        )
        after = trainer.model.generator.output_projection.weight.detach()
        self.assertTrue(torch.isfinite(torch.tensor(metrics.generator_loss)))
        self.assertTrue(torch.isfinite(torch.tensor(metrics.discriminator_loss)))
        self.assertFalse(torch.equal(before, after))
        self.assertEqual(metrics.mas_noise_scale, 0.01)
        self.assertEqual(metrics.c_kl, 1.0)
        self.assertAlmostEqual(metrics.weighted_kl_loss, metrics.kl_loss, places=6)
        self.assertEqual(update_order, ["generator", "discriminator"])

    def test_checkpoint_round_trip(self):
        trainer = BanhmiTrainer(small_config(), torch.device("cpu"))
        trainer.train_step(
            small_batch(),
            global_step=0,
            random_generator=torch.Generator().manual_seed(4),
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "checkpoint.pt"
            save_checkpoint(path, trainer, global_step=7, extra={"epoch": 2})
            restored = BanhmiTrainer(small_config(), torch.device("cpu"))
            metadata = load_checkpoint(path, restored)
            self.assertEqual(metadata, {"global_step": 7, "extra": {"epoch": 2}})
            for first, second in zip(
                trainer.model.parameters(), restored.model.parameters()
            ):
                self.assertTrue(torch.equal(first, second))
            self.assertIsNotNone(trainer.duration_discriminator)
            for first, second in zip(
                trainer.duration_discriminator.parameters(),
                restored.duration_discriminator.parameters(),
            ):
                self.assertTrue(torch.equal(first, second))

    def test_checkpoint_restores_rng_state(self):
        trainer = BanhmiTrainer(small_config(), torch.device("cpu"))
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "checkpoint.pt"
            torch.manual_seed(77)
            save_checkpoint(path, trainer, global_step=0)
            expected = torch.rand(4)
            torch.manual_seed(999)
            load_checkpoint(path, trainer)
            self.assertTrue(torch.equal(torch.rand(4), expected))

    def test_teacher_forcing_schedule(self):
        optimization = OptimizationConfig(
            prosody_teacher_forcing_start=1.0,
            prosody_teacher_forcing_end=0.0,
            prosody_teacher_forcing_steps=100,
        )
        self.assertEqual(optimization.prosody_teacher_forcing_ratio(0), 1.0)
        self.assertEqual(optimization.prosody_teacher_forcing_ratio(50), 0.5)
        self.assertEqual(optimization.prosody_teacher_forcing_ratio(100), 0.0)

    def test_edge_kl_coefficient_parity(self):
        optimization = OptimizationConfig()
        self.assertEqual(optimization.c_kl, 1.0)
        self.assertIsNone(optimization.generator_gradient_clip)
        self.assertIsNone(optimization.discriminator_gradient_clip)
        self.assertEqual(optimization.spectral_weight, 0.0)
        optimization.validate()
        with self.assertRaises(ValueError):
            OptimizationConfig(c_kl=-1.0).validate()
        with self.assertRaises(ValueError):
            OptimizationConfig(generator_gradient_clip=0.0).validate()

    def test_nonfinite_gradient_is_detected_before_optimizer_step(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        parameter.grad = torch.tensor([1.0, float("inf")])
        finite, total_norm = _clip_or_check_gradients(
            [parameter], 1.0, torch.device("cpu")
        )
        self.assertFalse(finite)
        self.assertFalse(torch.isfinite(total_norm))

    def test_validation_is_deterministic_with_fixed_seed(self):
        trainer = BanhmiTrainer(small_config(), torch.device("cpu"))
        batch = small_batch()
        first = trainer.validate_step(
            batch, 10, random_generator=torch.Generator().manual_seed(9)
        )
        second = trainer.validate_step(
            batch, 10, random_generator=torch.Generator().manual_seed(9)
        )
        self.assertEqual(first.metrics, second.metrics)
        self.assertTrue(torch.equal(first.generated_audio, second.generated_audio))
        self.assertTrue(torch.equal(first.target_audio, second.target_audio))

    def test_posterior_mean_reconstruction_supports_segment_and_full_audio(self):
        trainer = BanhmiTrainer(small_config(), torch.device("cpu"))
        batch = small_batch()
        segment = trainer.reconstruct_posterior_mean(batch, full_utterance=False)
        full = trainer.reconstruct_posterior_mean(batch, full_utterance=True)
        sampled = trainer.reconstruct_posterior_mean(
            batch,
            full_utterance=False,
            sample_posterior=True,
            random_generator=torch.Generator().manual_seed(12),
        )
        self.assertEqual(segment.generated_audio.shape, (2, 1, 24))
        self.assertEqual(segment.target_audio.shape, (2, 1, 24))
        self.assertEqual(full.generated_audio.shape, (1, 1, 48))
        self.assertEqual(full.target_audio.shape, (1, 1, 48))
        self.assertEqual(sampled.generated_audio.shape, segment.generated_audio.shape)
        self.assertTrue(torch.isfinite(torch.tensor(
            trainer.reconstruction_spectral_loss(sampled)
        )))


if __name__ == "__main__":
    unittest.main()
