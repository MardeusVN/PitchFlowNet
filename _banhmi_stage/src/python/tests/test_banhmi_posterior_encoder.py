import unittest

import torch

from banhmi_tts.model import PosteriorEncoder, PosteriorEncoderConfig
from banhmi_tts.model.text_encoder import sequence_mask


class PosteriorEncoderTests(unittest.TestCase):
    def make_encoder(self, use_weight_norm: bool = False) -> PosteriorEncoder:
        return PosteriorEncoder(
            PosteriorEncoderConfig(
                input_channels=9,
                latent_channels=8,
                hidden_channels=16,
                kernel_size=5,
                dilation_rate=1,
                num_layers=4,
                dropout=0.0,
                use_weight_norm=use_weight_norm,
            )
        )

    def test_output_shapes_and_padding_are_masked(self):
        encoder = self.make_encoder()
        spectrogram = torch.randn(2, 9, 12)
        output = encoder(spectrogram, torch.tensor([7, 12]), sample=False)
        self.assertEqual(output.latent.shape, (2, 8, 12))
        self.assertEqual(output.posterior_mean.shape, (2, 8, 12))
        self.assertEqual(output.posterior_log_scale.shape, (2, 8, 12))
        self.assertEqual(output.mask.shape, (2, 1, 12))
        self.assertTrue((output.latent[0, :, 7:] == 0).all())
        self.assertTrue((output.posterior_log_scale[0, :, 7:] == 0).all())

    def test_deterministic_mode_returns_posterior_mean(self):
        encoder = self.make_encoder().eval()
        output = encoder(torch.randn(1, 9, 10), torch.tensor([10]), sample=False)
        self.assertTrue(torch.equal(output.latent, output.posterior_mean))

    def test_sampling_is_reproducible_with_generator(self):
        encoder = self.make_encoder().eval()
        spectrogram = torch.randn(1, 9, 10)
        first_generator = torch.Generator().manual_seed(123)
        second_generator = torch.Generator().manual_seed(123)
        first = encoder(spectrogram, torch.tensor([10]), generator=first_generator).latent
        second = encoder(spectrogram, torch.tensor([10]), generator=second_generator).latent
        self.assertTrue(torch.equal(first, second))

    def test_sampling_stays_float32_under_bfloat16_autocast(self):
        encoder = self.make_encoder().eval()
        spectrogram = torch.randn(1, 9, 10)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = encoder(
                spectrogram,
                torch.tensor([10]),
                generator=torch.Generator().manual_seed(123),
            )
        self.assertEqual(output.latent.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output.latent).all())

    def test_padding_values_do_not_affect_valid_outputs(self):
        encoder = self.make_encoder().eval()
        first = torch.randn(1, 9, 12)
        second = first.clone()
        second[:, :, 7:] = torch.randn_like(second[:, :, 7:]) * 100
        lengths = torch.tensor([7])
        with torch.no_grad():
            first_output = encoder(first, lengths, sample=False).posterior_mean[:, :, :7]
            second_output = encoder(second, lengths, sample=False).posterior_mean[:, :, :7]
        self.assertTrue(torch.allclose(first_output, second_output, atol=1e-6))

    def test_backward_and_weight_norm_removal(self):
        encoder = self.make_encoder(use_weight_norm=True)
        output = encoder(torch.randn(2, 9, 10), torch.tensor([10, 8]))
        loss = (output.latent.square() * output.mask).mean()
        loss.backward()
        self.assertIsNotNone(encoder.input_projection.weight.grad)
        self.assertTrue(torch.isfinite(encoder.input_projection.weight.grad).all())
        encoder.remove_weight_norm()
        deterministic = encoder(torch.randn(1, 9, 6), torch.tensor([6]), sample=False)
        self.assertTrue(torch.isfinite(deterministic.posterior_mean).all())

    def test_invalid_configuration_and_input_bins(self):
        with self.assertRaises(ValueError):
            PosteriorEncoderConfig(kernel_size=4).validate()
        encoder = self.make_encoder()
        with self.assertRaises(ValueError):
            encoder(torch.randn(1, 8, 10), torch.tensor([10]))

    def test_edge_vits_wavenet_statistics_numerical_parity(self):
        """Compare against the explicit EdgeTTS/VITS posterior equations."""
        torch.manual_seed(23)
        encoder = self.make_encoder(use_weight_norm=False).eval()
        spectrogram = torch.randn(2, 9, 11)
        lengths = torch.tensor([7, 11])
        with torch.no_grad():
            output = encoder(spectrogram, lengths, sample=False)
            mask = sequence_mask(lengths, spectrogram.shape[-1]).unsqueeze(1)
            mask = mask.to(spectrogram.dtype)
            values = encoder.input_projection(spectrogram * mask) * mask
            skip = torch.zeros_like(values)
            for index, (input_layer, res_skip_layer) in enumerate(
                zip(
                    encoder.wavenet.input_layers,
                    encoder.wavenet.residual_skip_layers,
                )
            ):
                gates = input_layer(values)
                tanh_part, sigmoid_part = gates.chunk(2, dim=1)
                acts = torch.tanh(tanh_part) * torch.sigmoid(sigmoid_part)
                res_skip = res_skip_layer(acts)
                if index < encoder.wavenet.num_layers - 1:
                    residual, layer_skip = res_skip.chunk(2, dim=1)
                    values = (values + residual) * mask
                    skip = skip + layer_skip
                else:
                    skip = skip + res_skip
            reference_stats = encoder.statistics_projection(skip * mask) * mask
            reference_mean, reference_log_scale = reference_stats.chunk(2, dim=1)
        self.assertTrue(torch.allclose(output.posterior_mean, reference_mean, atol=1e-7))
        self.assertTrue(
            torch.allclose(output.posterior_log_scale, reference_log_scale, atol=1e-7)
        )
        self.assertTrue(torch.equal(output.latent, reference_mean))


if __name__ == "__main__":
    unittest.main()
