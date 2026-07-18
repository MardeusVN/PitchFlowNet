import unittest

import torch

from banhmi_tts.model import (
    BigVGANDiscriminator,
    DiscriminatorConfig,
    DiscriminatorOutput,
    MultiResolutionSpectralLoss,
    MultiResolutionSpectralLossConfig,
    discriminator_least_squares_loss,
    feature_matching_loss,
    generator_least_squares_loss,
)


class DiscriminatorAndLossTests(unittest.TestCase):
    def discriminator(self, use_weight_norm: bool = False) -> BigVGANDiscriminator:
        return BigVGANDiscriminator(
            DiscriminatorConfig(
                periods=(2, 3),
                resolutions=((64, 16, 64),),
                use_weight_norm=use_weight_norm,
            )
        )

    def test_discriminator_outputs_scores_and_feature_maps(self):
        discriminator = self.discriminator()
        outputs = discriminator(torch.randn(2, 1, 256))
        self.assertEqual(len(outputs), 4)
        self.assertTrue(all(output.score.ndim == 2 for output in outputs))
        self.assertEqual([len(output.feature_maps) for output in outputs], [7, 6, 6, 6])
        self.assertTrue(all(torch.isfinite(output.score).all() for output in outputs))

    def test_edge_compatible_ensemble_adds_scale_and_uses_magnitude(self):
        discriminator = BigVGANDiscriminator(
            DiscriminatorConfig(
                periods=(2, 3),
                resolutions=((64, 16, 64),),
                use_weight_norm=False,
                include_scale_discriminator=True,
                use_magnitude_spectrogram=True,
            )
        )
        outputs = discriminator(torch.randn(1, 1, 256))
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(outputs[0].feature_maps), 7)
        resolution = discriminator.resolution_discriminators[0]
        self.assertEqual(resolution.convolutions[0].in_channels, 1)

    def test_adversarial_and_feature_losses_backward(self):
        discriminator = self.discriminator()
        real = torch.randn(2, 1, 256)
        fake = torch.randn(2, 1, 256, requires_grad=True)
        real_outputs = discriminator(real)
        fake_outputs = discriminator(fake)
        generator_loss = generator_least_squares_loss(fake_outputs)
        feature_loss = feature_matching_loss(real_outputs, fake_outputs)
        (generator_loss + feature_loss).backward(retain_graph=True)
        self.assertIsNotNone(fake.grad)
        self.assertTrue(torch.isfinite(fake.grad).all())
        discriminator_loss = discriminator_least_squares_loss(
            real_outputs, discriminator(fake.detach())
        )
        self.assertTrue(torch.isfinite(discriminator_loss))

    def test_adversarial_losses_promote_bfloat16_outputs_to_float32(self):
        real_score = torch.randn(2, 8, dtype=torch.bfloat16)
        fake_score = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)
        real_feature = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        fake_feature = torch.randn(
            2, 4, 8, dtype=torch.bfloat16, requires_grad=True
        )
        real = [DiscriminatorOutput(real_score, [real_feature])]
        fake = [DiscriminatorOutput(fake_score, [fake_feature])]
        losses = (
            discriminator_least_squares_loss(real, fake),
            generator_least_squares_loss(fake),
            feature_matching_loss(real, fake),
        )
        self.assertTrue(all(loss.dtype == torch.float32 for loss in losses))
        sum(losses).backward()
        self.assertTrue(torch.isfinite(fake_score.grad).all())
        self.assertTrue(torch.isfinite(fake_feature.grad).all())

    def test_multi_resolution_spectral_loss(self):
        loss_module = MultiResolutionSpectralLoss(
            MultiResolutionSpectralLossConfig(
                resolutions=((64, 16, 64), (32, 8, 32))
            )
        )
        target = torch.randn(2, 1, 256)
        identical = loss_module(target, target)
        self.assertLess(float(identical), 1e-6)
        generated = torch.randn(2, 1, 256, requires_grad=True)
        loss = loss_module(generated, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(generated.grad)

    def test_weight_norm_removal_and_validation(self):
        discriminator = self.discriminator(use_weight_norm=True)
        discriminator.remove_weight_norm()
        outputs = discriminator(torch.randn(1, 1, 256))
        self.assertTrue(all(torch.isfinite(output.score).all() for output in outputs))
        with self.assertRaises(ValueError):
            DiscriminatorConfig(periods=(2, 2)).validate()


if __name__ == "__main__":
    unittest.main()
