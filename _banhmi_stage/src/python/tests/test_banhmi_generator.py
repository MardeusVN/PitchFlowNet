import unittest

import torch

from banhmi_tts.model import (
    BigVGANGenerator,
    GeneratorConfig,
    SegmentConfig,
    slice_training_segments,
)
from banhmi_tts.model.generator import FilteredSnakeBeta, SnakeBeta


class SegmentAndGeneratorTests(unittest.TestCase):
    def small_generator(
        self,
        use_weight_norm: bool = False,
        activation: str = "filtered_snake_beta",
        use_prosody_conditioning: bool = True,
    ) -> BigVGANGenerator:
        return BigVGANGenerator(
            GeneratorConfig(
                latent_channels=8,
                initial_channels=32,
                upsample_rates=(4, 2),
                upsample_kernel_sizes=(8, 4),
                amp_kernel_sizes=(3, 5),
                amp_dilations=((1, 2), (1, 2)),
                use_weight_norm=use_weight_norm,
                activation=activation,
                use_prosody_conditioning=use_prosody_conditioning,
            )
        )

    def test_synchronized_segment_slicing(self):
        latent = torch.arange(2 * 3 * 10).reshape(2, 3, 10).float()
        pitch = torch.arange(20).reshape(2, 10).float()
        voicing = torch.ones(2, 10)
        audio = torch.arange(2 * 1 * 40).reshape(2, 1, 40).float()
        segments = slice_training_segments(
            latent,
            pitch,
            voicing,
            audio,
            torch.tensor([10, 8]),
            SegmentConfig(segment_frames=3, hop_length=4),
            frame_starts=torch.tensor([2, 4]),
        )
        self.assertEqual(segments.latent.shape, (2, 3, 3))
        self.assertEqual(segments.audio.shape, (2, 1, 12))
        self.assertEqual(segments.sample_starts.tolist(), [8, 16])
        self.assertTrue(torch.equal(segments.normalized_log_f0[0], pitch[0, 2:5]))
        self.assertTrue(torch.equal(segments.audio[1], audio[1, :, 16:28]))

    def test_random_segment_starts_are_reproducible(self):
        arguments = (
            torch.randn(2, 3, 10),
            torch.randn(2, 10),
            torch.ones(2, 10),
            torch.randn(2, 1, 40),
            torch.tensor([10, 8]),
            SegmentConfig(segment_frames=3, hop_length=4),
        )
        first = slice_training_segments(
            *arguments, generator=torch.Generator().manual_seed(7)
        )
        second = slice_training_segments(
            *arguments, generator=torch.Generator().manual_seed(7)
        )
        self.assertTrue(torch.equal(first.frame_starts, second.frame_starts))

    def test_generator_output_shape_range_and_backward(self):
        generator = self.small_generator()
        latent = torch.randn(2, 8, 6)
        pitch = torch.randn(2, 6)
        voicing = torch.rand(2, 6)
        waveform = generator(latent, pitch, voicing)
        self.assertEqual(waveform.shape, (2, 1, 48))
        self.assertTrue(torch.all(waveform >= -1.0))
        self.assertTrue(torch.all(waveform <= 1.0))
        waveform.square().mean().backward()
        self.assertIsNotNone(generator.input_projection.weight.grad)
        self.assertTrue(torch.isfinite(generator.input_projection.weight.grad).all())

    def test_activation_configuration_selects_edge_tts_snake(self):
        filtered = self.small_generator()
        direct = self.small_generator(activation="snake_beta")
        self.assertIsInstance(filtered.pre_upsample_activations[0], FilteredSnakeBeta)
        self.assertIsInstance(direct.pre_upsample_activations[0], SnakeBeta)
        self.assertIsInstance(direct.amp_stages[0][0].activations[0], SnakeBeta)
        self.assertNotIsInstance(direct.output_activation, FilteredSnakeBeta)

    def test_default_topology_matches_edge_tts_config_h(self):
        config = GeneratorConfig()
        self.assertEqual(config.initial_channels, 256)
        self.assertEqual(config.upsample_rates, (8, 8, 4))
        self.assertEqual(config.upsample_kernel_sizes, (16, 16, 8))
        self.assertEqual(config.amp_kernel_sizes, (3, 5, 7))
        self.assertEqual(config.amp_dilations, ((1, 2), (2, 6), (3, 12)))

    def test_edge_convolution_initialization_with_production_weight_norm(self):
        generator = self.small_generator(
            use_weight_norm=True, activation="snake_beta"
        )
        modules = list(generator.upsamplers)
        for stage in generator.amp_stages:
            for block in stage:
                modules.extend(block.convolutions)
        weights = torch.cat([module.weight.detach().flatten() for module in modules])
        self.assertLess(abs(float(weights.mean())), 0.003)
        self.assertAlmostEqual(float(weights.std()), 0.01, delta=0.003)
        # A fresh access recomputes parametrized weights; initialization must
        # remain effective rather than modifying a temporary tensor.
        recomputed = torch.cat([module.weight.detach().flatten() for module in modules])
        self.assertTrue(torch.equal(weights, recomputed))

    def test_edge_snake_beta_is_finite_at_parameter_safety_bounds(self):
        activation = SnakeBeta(2)
        with torch.no_grad():
            activation.alpha.copy_(torch.tensor([[[100.0], [-100.0]]]))
            activation.beta.copy_(torch.tensor([[[0.0], [100.0]]]))
        output = activation(torch.randn(1, 2, 32))
        self.assertTrue(torch.isfinite(output).all())
        statistics_generator = self.small_generator(activation="snake_beta")
        statistics = statistics_generator.snake_parameter_statistics()
        self.assertEqual(
            set(statistics),
            {"snake_alpha_min", "snake_alpha_max", "snake_beta_min", "snake_beta_max"},
        )

    def test_zero_initialized_prosody_is_initially_noop_then_learns(self):
        generator = self.small_generator().eval()
        latent = torch.randn(1, 8, 5)
        zeros = torch.zeros(1, 5)
        with torch.no_grad():
            first = generator(latent, zeros, zeros)
            second = generator(latent, torch.ones_like(zeros), torch.ones_like(zeros))
        self.assertTrue(torch.allclose(first, second, atol=1e-6))
        with torch.no_grad():
            generator.prosody_conditioning.weight.fill_(0.05)
            conditioned = generator(
                latent, torch.ones_like(zeros), torch.ones_like(zeros)
            )
        self.assertFalse(torch.allclose(first, conditioned))

    def test_disabled_prosody_conditioning_is_an_exact_noop(self):
        generator = self.small_generator(
            activation="snake_beta", use_prosody_conditioning=False
        ).eval()
        latent = torch.randn(1, 8, 5)
        zeros = torch.zeros(1, 5)
        ones = torch.ones(1, 5)
        with torch.no_grad():
            first = generator(latent, zeros, zeros)
            second = generator(latent, ones, ones)
        self.assertTrue(torch.equal(first, second))

    def test_weight_norm_removal_and_invalid_voicing(self):
        generator = self.small_generator(use_weight_norm=True).eval()
        latent = torch.randn(1, 8, 4)
        pitch = torch.zeros(1, 4)
        voicing = torch.ones(1, 4)
        with torch.no_grad():
            before = generator(latent, pitch, voicing)
        generator.remove_weight_norm()
        with torch.no_grad():
            output = generator(latent, pitch, voicing)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(before, output, atol=1e-6, rtol=1e-5))
        with self.assertRaises(ValueError):
            generator(torch.randn(1, 8, 4), torch.zeros(1, 4), torch.full((1, 4), 2.0))


if __name__ == "__main__":
    unittest.main()
