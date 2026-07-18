import unittest

import torch

from banhmi_tts.model import (
    AlignmentFrameExpander,
    ProsodyPredictorOutput,
    denormalize_log_f0,
    durations_to_path,
    expand_token_features,
    kl_divergence_loss,
)


class FrameExpansionTests(unittest.TestCase):
    def alignment(self) -> torch.Tensor:
        return torch.tensor(
            [[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]]
        )

    def test_expand_token_features_repeats_by_alignment(self):
        values = torch.tensor([[[2.0, 7.0], [3.0, 9.0]]])
        expanded = expand_token_features(values, self.alignment())
        expected = torch.tensor(
            [[[2.0, 2.0, 7.0, 7.0, 7.0], [3.0, 3.0, 9.0, 9.0, 9.0]]]
        )
        self.assertTrue(torch.equal(expanded, expected))

    def test_duration_path_is_monotonic_and_masks_padded_sequences(self):
        durations = torch.tensor([[2, 1, 2], [1, 2, 0]])
        text_mask = torch.tensor(
            [[[1.0, 1.0, 1.0]], [[1.0, 1.0, 0.0]]]
        )
        path = durations_to_path(durations, text_mask)
        self.assertTrue(torch.equal(path.frame_lengths, torch.tensor([5, 3])))
        self.assertEqual(path.alignment.shape, (2, 1, 5, 3))
        first_tokens = path.alignment[0, 0].argmax(dim=1)
        self.assertTrue(
            torch.equal(first_tokens, torch.tensor([0, 0, 1, 2, 2]))
        )
        self.assertTrue(torch.equal(path.frame_mask[1, 0], torch.tensor([1., 1., 1., 0., 0.])))
        assigned = path.alignment.sum(dim=-1)
        self.assertTrue(torch.equal(assigned, path.frame_mask))

    def test_duration_path_rejects_frames_on_padding(self):
        with self.assertRaises(ValueError):
            durations_to_path(
                torch.tensor([[1, 1]]), torch.tensor([[[1.0, 0.0]]])
            )

    def test_frame_expander_shapes_and_voicing_probability(self):
        prior_mean = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        prior_logs = torch.zeros_like(prior_mean)
        prosody = ProsodyPredictorOutput(
            normalized_continuous_log_f0=torch.tensor([[0.5, -0.5]]),
            voicing_logits=torch.tensor([[2.0, -2.0]]),
        )
        mask = torch.ones(1, 1, 5)
        output = AlignmentFrameExpander()(
            self.alignment(), prior_mean, prior_logs, prosody, mask
        )
        self.assertEqual(output.prior_mean.shape, (1, 2, 5))
        self.assertEqual(output.normalized_continuous_log_f0.shape, (1, 5))
        self.assertGreater(float(output.voicing_probability[0, 0]), 0.5)
        self.assertLess(float(output.voicing_probability[0, -1]), 0.5)

    def test_kl_matches_direct_formula_and_logdet_correction(self):
        torch.manual_seed(5)
        z = torch.randn(2, 3, 4)
        posterior_mean = torch.randn(2, 3, 4)
        posterior_logs = torch.randn(2, 3, 4) * 0.1
        prior_mean = torch.randn(2, 3, 4)
        prior_logs = torch.randn(2, 3, 4) * 0.1
        mask = torch.tensor(
            [[[1.0, 1.0, 1.0, 0.0]], [[1.0, 1.0, 1.0, 1.0]]]
        )
        actual = kl_divergence_loss(
            z, posterior_mean, posterior_logs, prior_mean, prior_logs, mask
        )
        direct = (
            prior_logs
            - posterior_logs
            - 0.5
            + 0.5 * (z - prior_mean).square() * torch.exp(-2.0 * prior_logs)
        )
        expected = (direct * mask).sum() / mask.sum()
        self.assertTrue(torch.allclose(actual, expected))
        logdet = torch.tensor([1.0, 2.0])
        corrected = kl_divergence_loss(
            z,
            posterior_mean,
            posterior_logs,
            prior_mean,
            prior_logs,
            mask,
            flow_log_determinant=logdet,
        )
        self.assertTrue(torch.allclose(corrected, actual - logdet.sum() / mask.sum()))

    def test_kl_uses_float32_for_mixed_precision_inputs(self):
        z = torch.tensor([[[0.25, -0.5]]], dtype=torch.float16)
        posterior_mean = torch.zeros_like(z)
        posterior_logs = torch.tensor([[[0.1, -0.2]]], dtype=torch.float16)
        prior_mean = torch.tensor([[[0.0, 0.1]]], dtype=torch.float16)
        prior_logs = torch.tensor([[[0.2, -0.1]]], dtype=torch.float16)
        mask = torch.ones(1, 1, 2, dtype=torch.float16)
        loss = kl_divergence_loss(
            z, posterior_mean, posterior_logs, prior_mean, prior_logs, mask
        )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))

    def test_denormalization_preserves_padding(self):
        normalized = torch.tensor([[0.0, 2.0, 8.0]])
        mask = torch.tensor([[True, True, False]])
        result = denormalize_log_f0(normalized, mask, mean=4.0, standard_deviation=0.5)
        self.assertTrue(torch.equal(result, torch.tensor([[4.0, 5.0, 0.0]])))

    def test_rejects_mismatched_expansion_shapes(self):
        with self.assertRaises(ValueError):
            expand_token_features(torch.randn(1, 2, 3), self.alignment())


if __name__ == "__main__":
    unittest.main()
