import itertools
import unittest

import torch

from banhmi_tts.model import MASConfig, MonotonicAlignmentSearch
from banhmi_tts.model.alignment import (
    compiled_maximum_path_available,
    gaussian_log_likelihood,
    maximum_path,
)


class AlignmentTests(unittest.TestCase):
    def test_gaussian_log_likelihood_matches_broadcast_formula(self):
        torch.manual_seed(2)
        latent = torch.randn(2, 3, 5)
        mean = torch.randn(2, 3, 4)
        log_scale = torch.randn(2, 3, 4) * 0.1
        actual = gaussian_log_likelihood(latent, mean, log_scale)
        expanded_latent = latent.unsqueeze(3)
        expanded_mean = mean.unsqueeze(2)
        expanded_log_scale = log_scale.unsqueeze(2)
        expected = (
            -0.5 * torch.log(torch.tensor(2.0 * torch.pi))
            - expanded_log_scale
            - 0.5
            * (expanded_latent - expanded_mean).square()
            * torch.exp(-2.0 * expanded_log_scale)
        ).sum(dim=1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_maximum_path_matches_brute_force(self):
        scores = torch.tensor(
            [[[2.0, -3.0, -4.0], [1.0, 3.0, -2.0], [0.0, 2.0, 1.0], [-2.0, 1.0, 4.0], [-3.0, -2.0, 5.0]]]
        )
        path = maximum_path(scores, torch.tensor([5]), torch.tensor([3]))
        actual_score = float((path * scores).sum())
        candidates = []
        for advances in itertools.combinations(range(1, 5), 2):
            token = 0
            total = float(scores[0, 0, 0])
            for frame in range(1, 5):
                if frame in advances:
                    token += 1
                total += float(scores[0, frame, token])
            candidates.append(total)
        self.assertEqual(actual_score, max(candidates))
        self.assertTrue(torch.equal(path.sum(dim=2), torch.ones(1, 5)))
        self.assertTrue(torch.equal(path.sum(dim=1).long(), torch.tensor([[1, 2, 2]])))

    @unittest.skipUnless(
        compiled_maximum_path_available(), "compiled Cython MAS is not built"
    )
    def test_compiled_maximum_path_matches_torch_reference(self):
        generator = torch.Generator().manual_seed(17)
        scores = torch.randn(4, 19, 9, generator=generator)
        acoustic_lengths = torch.tensor([19, 17, 15, 13])
        text_lengths = torch.tensor([9, 8, 7, 6])
        reference = maximum_path(
            scores,
            acoustic_lengths,
            text_lengths,
            use_compiled=False,
        )
        compiled = maximum_path(
            scores,
            acoustic_lengths,
            text_lengths,
            use_compiled=True,
        )
        self.assertTrue(torch.equal(compiled, reference))

    def test_noise_schedule_matches_vits2(self):
        config = MASConfig()
        self.assertAlmostEqual(config.noise_scale(0, True), 0.01)
        self.assertAlmostEqual(config.noise_scale(2500, True), 0.005)
        self.assertEqual(config.noise_scale(5000, True), 0.0)
        self.assertEqual(config.noise_scale(0, False), 0.0)

    def test_module_returns_positive_durations_and_valid_mask(self):
        torch.manual_seed(4)
        mas = MonotonicAlignmentSearch().eval()
        latent = torch.randn(2, 4, 8)
        mean = torch.randn(2, 4, 5)
        logs = torch.zeros_like(mean)
        output = mas(
            latent,
            mean,
            logs,
            acoustic_lengths=torch.tensor([6, 8]),
            text_lengths=torch.tensor([4, 5]),
        )
        self.assertEqual(output.alignment.shape, (2, 1, 8, 5))
        self.assertEqual(output.durations.shape, (2, 5))
        self.assertTrue((output.durations[0, :4] >= 1).all())
        self.assertEqual(int(output.durations[0].sum()), 6)
        self.assertEqual(int(output.durations[1].sum()), 8)
        self.assertFalse(output.alignment.requires_grad)
        self.assertEqual(output.noise_scale, 0.0)

    def test_noisy_search_is_reproducible_and_anneals(self):
        mas = MonotonicAlignmentSearch().train()
        latent = torch.randn(1, 3, 7)
        mean = torch.randn(1, 3, 4)
        logs = torch.zeros_like(mean)
        first = mas(
            latent,
            mean,
            logs,
            torch.tensor([7]),
            torch.tensor([4]),
            global_step=0,
            generator=torch.Generator().manual_seed(9),
        )
        second = mas(
            latent,
            mean,
            logs,
            torch.tensor([7]),
            torch.tensor([4]),
            global_step=0,
            generator=torch.Generator().manual_seed(9),
        )
        self.assertTrue(torch.equal(first.alignment, second.alignment))
        self.assertEqual(first.noise_scale, 0.01)
        annealed = mas(
            latent,
            mean,
            logs,
            torch.tensor([7]),
            torch.tensor([4]),
            global_step=5000,
        )
        self.assertEqual(annealed.noise_scale, 0.0)

    def test_rejects_impossible_alignment(self):
        with self.assertRaises(ValueError):
            maximum_path(
                torch.randn(1, 3, 4), torch.tensor([3]), torch.tensor([4])
            )

    def test_float16_negative_scores_keep_valid_start_state(self):
        scores = torch.full((2, 12, 7), -60000.0, dtype=torch.float16)
        path = maximum_path(
            scores,
            acoustic_lengths=torch.tensor([12, 9]),
            text_lengths=torch.tensor([7, 5]),
        )
        self.assertTrue(torch.equal(path[0].sum(dim=1), torch.ones(12)))
        self.assertTrue(torch.equal(path[1, :9].sum(dim=1), torch.ones(9)))
        self.assertEqual(int(path[0, :, 0].sum()), 1)
        self.assertEqual(int(path[1, :9, 0].sum()), 1)

    def test_mas_float16_extreme_inputs_produce_a_valid_path(self):
        mas = MonotonicAlignmentSearch().train()
        latent = (torch.randn(1, 4, 30) * 20).half()
        mean = (torch.randn(1, 4, 8) * 20).half()
        logs = torch.full_like(mean, -3.0)
        output = mas(
            latent,
            mean,
            logs,
            acoustic_lengths=torch.tensor([30]),
            text_lengths=torch.tensor([8]),
            generator=torch.Generator().manual_seed(1),
        )
        self.assertEqual(int(output.durations.sum()), 30)
        self.assertTrue((output.durations[0, :8] >= 1).all())


if __name__ == "__main__":
    unittest.main()
