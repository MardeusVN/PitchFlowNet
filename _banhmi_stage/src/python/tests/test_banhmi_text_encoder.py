import unittest
import math

import torch

from banhmi_tts.model import TextEncoder, TextEncoderConfig
from banhmi_tts.model.text_encoder import sequence_mask


class TextEncoderTests(unittest.TestCase):
    def make_encoder(self, dropout: float = 0.0) -> TextEncoder:
        return TextEncoder(
            TextEncoderConfig(
                num_symbols=52,
                latent_channels=16,
                hidden_channels=32,
                filter_channels=64,
                num_heads=4,
                num_layers=2,
                kernel_size=3,
                dropout=dropout,
                relative_position_window=4,
            )
        )

    def test_output_shapes_and_padding_are_masked(self):
        encoder = self.make_encoder()
        ids = torch.tensor([[1, 10, 0, 2, 0], [1, 11, 0, 12, 2]])
        lengths = torch.tensor([4, 5])
        output = encoder(ids, lengths)
        self.assertEqual(output.hidden.shape, (2, 32, 5))
        self.assertEqual(output.prior_mean.shape, (2, 16, 5))
        self.assertEqual(output.prior_log_scale.shape, (2, 16, 5))
        self.assertEqual(output.mask.shape, (2, 1, 5))
        self.assertTrue((output.hidden[0, :, 4] == 0).all())
        self.assertTrue((output.prior_mean[0, :, 4] == 0).all())
        self.assertTrue((output.prior_log_scale[0, :, 4] == 0).all())

    def test_padding_token_changes_do_not_affect_valid_outputs(self):
        torch.manual_seed(7)
        encoder = self.make_encoder().eval()
        first = torch.tensor([[1, 10, 0, 2, 0, 0]])
        second = torch.tensor([[1, 10, 0, 2, 20, 21]])
        lengths = torch.tensor([4])
        with torch.no_grad():
            first_output = encoder(first, lengths).hidden[:, :, :4]
            second_output = encoder(second, lengths).hidden[:, :, :4]
        self.assertTrue(torch.allclose(first_output, second_output, atol=1e-6))

    def test_backward_reaches_embedding_and_prior_projection(self):
        encoder = self.make_encoder(dropout=0.1).train()
        ids = torch.tensor([[1, 10, 0, 2], [1, 12, 0, 2]])
        output = encoder(ids, torch.tensor([4, 4]))
        loss = output.hidden.square().mean() + output.prior_mean.square().mean()
        loss.backward()
        self.assertIsNotNone(encoder.embedding.weight.grad)
        self.assertIsNotNone(encoder.prior_projection.weight.grad)
        self.assertTrue(torch.isfinite(encoder.embedding.weight.grad).all())

    def test_blank_id_zero_is_trainable_while_padding_is_masked(self):
        encoder = self.make_encoder().train()
        self.assertIsNone(encoder.embedding.padding_idx)
        ids = torch.tensor([[1, 10, 0, 2, 0, 0]])
        output = encoder(ids, torch.tensor([4]))
        loss = output.hidden[:, :, 2].square().mean()
        loss.backward()
        blank_gradient = encoder.embedding.weight.grad[0]
        self.assertGreater(float(blank_gradient.abs().sum()), 0.0)
        self.assertTrue((output.hidden[:, :, 4:] == 0).all())

    def test_edge_vits_embedding_projection_contract_parity(self):
        """Match EdgeTTS TextEncoder's scale, layout, mask and stats split."""
        torch.manual_seed(17)
        encoder = self.make_encoder().eval()
        encoder.blocks = torch.nn.ModuleList()
        ids = torch.tensor([[1, 10, 0, 2, 0], [1, 11, 2, 0, 0]])
        lengths = torch.tensor([4, 3])
        with torch.no_grad():
            output = encoder(ids, lengths)
            valid = sequence_mask(lengths, ids.shape[1])
            mask = valid.unsqueeze(1).to(encoder.embedding.weight.dtype)
            reference_hidden = (
                encoder.embedding(ids) * math.sqrt(encoder.config.hidden_channels)
            ).transpose(1, 2) * mask
            reference_stats = encoder.prior_projection(reference_hidden) * mask
            reference_mean, reference_log_scale = reference_stats.chunk(2, dim=1)
        self.assertTrue(torch.equal(output.hidden, reference_hidden))
        self.assertTrue(torch.equal(output.prior_mean, reference_mean))
        self.assertTrue(torch.equal(output.prior_log_scale, reference_log_scale))

    def test_sequence_mask_and_invalid_configuration(self):
        mask = sequence_mask(torch.tensor([2, 4]), maximum_length=5)
        self.assertEqual(mask.tolist(), [[True, True, False, False, False], [True, True, True, True, False]])
        with self.assertRaises(ValueError):
            TextEncoderConfig(num_symbols=52, hidden_channels=30, num_heads=8).validate()

    def test_rejects_out_of_vocabulary_ids(self):
        encoder = self.make_encoder()
        with self.assertRaises(ValueError):
            encoder(torch.tensor([[1, 52, 2]]), torch.tensor([3]))

    def test_uses_vits_relative_key_and_value_embeddings(self):
        encoder = self.make_encoder()
        attention = encoder.blocks[0].attention
        self.assertEqual(attention.relative_key.shape, (1, 9, 8))
        self.assertEqual(attention.relative_value.shape, (1, 9, 8))
        self.assertFalse(hasattr(attention, "relative_bias"))


if __name__ == "__main__":
    unittest.main()
