import unittest

import torch

from banhmi_tts.model import (
    PhonemeProsodyTargetBuilder,
    ProsodyTargetConfig,
    normalize_log_f0,
)


class ProsodyTargetTests(unittest.TestCase):
    def test_known_alignment_aggregates_pitch_and_voicing(self):
        alignment = torch.tensor(
            [[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]]
        )
        log_continuous = torch.tensor([[1.0, 3.0, 5.0, 7.0, 9.0]])
        voiced = torch.tensor([[True, False, True, True, False]])
        log_f0 = torch.where(voiced, log_continuous, torch.zeros_like(log_continuous))
        targets = PhonemeProsodyTargetBuilder()(
            alignment,
            log_f0,
            log_continuous,
            voiced,
            acoustic_lengths=torch.tensor([5]),
            text_lengths=torch.tensor([2]),
        )
        self.assertEqual(targets.durations.tolist(), [[2, 3]])
        self.assertTrue(torch.allclose(targets.continuous_log_f0, torch.tensor([[2.0, 7.0]])))
        self.assertTrue(torch.allclose(targets.voiced_log_f0, torch.tensor([[1.0, 6.0]])))
        self.assertTrue(
            torch.allclose(targets.voiced_ratio, torch.tensor([[0.5, 2.0 / 3.0]]))
        )
        self.assertEqual(targets.voiced.tolist(), [[True, True]])

    def test_unvoiced_token_keeps_continuous_target(self):
        alignment = torch.tensor([[[[1.0], [1.0], [1.0]]]])
        continuous = torch.tensor([[4.8, 4.9, 5.0]])
        targets = PhonemeProsodyTargetBuilder()(
            alignment,
            torch.zeros_like(continuous),
            continuous,
            torch.zeros(1, 3, dtype=torch.bool),
            torch.tensor([3]),
            torch.tensor([1]),
        )
        self.assertAlmostEqual(float(targets.continuous_log_f0[0, 0]), 4.9, places=5)
        self.assertEqual(float(targets.voiced_log_f0[0, 0]), 0.0)
        self.assertFalse(bool(targets.voiced[0, 0]))

    def test_padding_is_zero_and_masked(self):
        alignment = torch.zeros(1, 1, 4, 3)
        alignment[0, 0, 0, 0] = 1
        alignment[0, 0, 1, 0] = 1
        values = torch.tensor([[4.0, 5.0, 100.0, 100.0]])
        targets = PhonemeProsodyTargetBuilder()(
            alignment,
            values,
            values,
            torch.tensor([[True, True, True, True]]),
            torch.tensor([2]),
            torch.tensor([1]),
        )
        self.assertEqual(targets.mask.tolist(), [[True, False, False]])
        self.assertTrue((targets.continuous_log_f0[0, 1:] == 0).all())
        self.assertTrue((targets.durations[0, 1:] == 0).all())

    def test_normalization_preserves_padding(self):
        values = torch.tensor([[4.0, 5.0, 0.0]])
        mask = torch.tensor([[True, True, False]])
        normalized = normalize_log_f0(values, mask, mean=4.0, standard_deviation=0.5)
        self.assertTrue(torch.equal(normalized, torch.tensor([[0.0, 2.0, 0.0]])))

    def test_rejects_incomplete_alignment_and_bad_threshold(self):
        with self.assertRaises(ValueError):
            ProsodyTargetConfig(voiced_label_threshold=1.1).validate()
        builder = PhonemeProsodyTargetBuilder()
        with self.assertRaises(ValueError):
            builder(
                torch.zeros(1, 1, 3, 2),
                torch.zeros(1, 3),
                torch.zeros(1, 3),
                torch.zeros(1, 3, dtype=torch.bool),
                torch.tensor([3]),
                torch.tensor([2]),
            )


if __name__ == "__main__":
    unittest.main()
