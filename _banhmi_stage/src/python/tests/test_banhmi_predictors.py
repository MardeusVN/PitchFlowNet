import unittest

import torch

from banhmi_tts.model import (
    DurationDiscriminator,
    DurationPredictor,
    JointProsodyPredictor,
    PredictorConfig,
    StochasticDurationPredictor,
    compute_predictor_losses,
    decode_durations,
    duration_targets,
)
from banhmi_tts.model.duration import _ConvFlow


class PredictorTests(unittest.TestCase):
    def config(self) -> PredictorConfig:
        return PredictorConfig(
            input_channels=16,
            hidden_channels=24,
            kernel_size=3,
            num_layers=2,
            dropout=0.0,
        )

    def test_shapes_and_padding_are_masked(self):
        hidden = torch.randn(2, 16, 7)
        mask = torch.tensor(
            [[[1, 1, 1, 1, 0, 0, 0]], [[1, 1, 1, 1, 1, 1, 1]]],
            dtype=torch.float32,
        )
        duration = DurationPredictor(self.config())(hidden, mask)
        prosody = JointProsodyPredictor(self.config())(hidden, mask)
        self.assertEqual(duration.shape, (2, 7))
        self.assertEqual(prosody.normalized_continuous_log_f0.shape, (2, 7))
        self.assertEqual(prosody.voicing_logits.shape, (2, 7))
        self.assertTrue((duration[0, 4:] == 0).all())
        self.assertTrue((prosody.voicing_logits[0, 4:] == 0).all())

    def test_duration_predictor_detaches_text_hidden(self):
        hidden = torch.randn(1, 16, 5, requires_grad=True)
        predictor = DurationPredictor(self.config())
        predictor(hidden, torch.ones(1, 1, 5)).square().mean().backward()
        self.assertIsNone(hidden.grad)
        self.assertIsNotNone(predictor.output_projection.weight.grad)

    def test_prosody_predictor_updates_text_hidden(self):
        hidden = torch.randn(1, 16, 5, requires_grad=True)
        predictor = JointProsodyPredictor(self.config())
        output = predictor(hidden, torch.ones(1, 1, 5))
        (output.normalized_continuous_log_f0.mean() + output.voicing_logits.mean()).backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_duration_target_and_decoding(self):
        durations = torch.tensor([[1, 2, 4, 0]])
        mask = torch.tensor([[True, True, True, False]])
        targets = duration_targets(durations, mask)
        self.assertTrue(
            torch.allclose(targets, torch.tensor([[0.0, 0.6931472, 1.3862944, 0.0]]))
        )
        decoded = decode_durations(targets, mask, length_scale=1.0)
        self.assertEqual(decoded.tolist(), [[1, 2, 4, 0]])

    def test_duration_decoding_is_safe_for_non_finite_predictions(self):
        predictions = torch.tensor([[float("nan"), float("inf"), -float("inf"), 0.0]])
        mask = torch.tensor([[True, True, True, False]])
        decoded = decode_durations(
            predictions,
            mask,
            maximum_log_duration=3.0,
            maximum_duration_frames=20,
        )
        self.assertEqual(decoded.tolist(), [[1, 20, 1, 0]])

    def test_stochastic_duration_forward_reverse_and_backward(self):
        config = self.config()
        predictor = StochasticDurationPredictor(config)
        hidden = torch.randn(2, 16, 6)
        mask = torch.tensor(
            [[[1, 1, 1, 1, 0, 0]], [[1, 1, 1, 1, 1, 1]]], dtype=torch.float32
        )
        durations = torch.tensor(
            [[[1, 2, 3, 2, 0, 0]], [[2, 1, 4, 2, 3, 1]]], dtype=torch.float32
        )
        nll = predictor(
            hidden,
            mask,
            durations,
            generator=torch.Generator().manual_seed(31),
        )
        self.assertEqual(nll.shape, (2,))
        self.assertTrue(torch.isfinite(nll).all())
        nll.mean().backward()
        self.assertIsNotNone(predictor.input_projection.weight.grad)
        first = predictor(
            hidden,
            mask,
            reverse=True,
            noise_scale=0.8,
            generator=torch.Generator().manual_seed(32),
        )
        second = predictor(
            hidden,
            mask,
            reverse=True,
            noise_scale=0.8,
            generator=torch.Generator().manual_seed(32),
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue((first[0, 4:] == 0).all())

    def test_duration_discriminator_scores_real_and_sampled_durations(self):
        discriminator = DurationDiscriminator(16, 24, 3, 0.0)
        hidden = torch.randn(2, 16, 5)
        mask = torch.ones(2, 1, 5)
        real = torch.randn(2, 1, 5)
        fake = torch.randn(2, 1, 5, requires_grad=True)
        real_scores, fake_scores = discriminator(hidden, mask, real, fake)
        self.assertEqual(real_scores[0].shape, (2, 5, 1))
        self.assertEqual(fake_scores[0].shape, (2, 5, 1))
        self.assertTrue(torch.isfinite(fake_scores[0]).all())
        fake_scores[0].mean().backward()
        self.assertIsNotNone(fake.grad)

    def test_duration_spline_flow_round_trip(self):
        torch.manual_seed(41)
        flow = _ConvFlow(2, 16, 3, 3).eval()
        values = torch.randn(2, 2, 7).clamp(-4.0, 4.0)
        mask = torch.ones(2, 1, 7)
        conditioning = torch.randn(2, 16, 7)
        with torch.no_grad():
            transformed, logdet = flow(
                values, mask, conditioning=conditioning, reverse=False
            )
            restored = flow(
                transformed, mask, conditioning=conditioning, reverse=True
            )
        self.assertTrue(torch.allclose(restored, values, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.isfinite(logdet).all())

    def test_losses_are_finite_and_ignore_padding(self):
        hidden = torch.randn(2, 16, 5)
        mask_3d = torch.tensor(
            [[[1, 1, 1, 0, 0]], [[1, 1, 1, 1, 1]]], dtype=torch.float32
        )
        mask = mask_3d.squeeze(1).bool()
        duration_predictor = DurationPredictor(self.config())
        prosody_predictor = JointProsodyPredictor(self.config())
        predicted_duration = duration_predictor(hidden, mask_3d)
        predicted_prosody = prosody_predictor(hidden, mask_3d)
        durations = torch.tensor([[1, 2, 3, 0, 0], [1, 1, 2, 2, 3]])
        pitch = torch.randn(2, 5) * mask
        voiced_ratio = torch.rand(2, 5) * mask
        losses = compute_predictor_losses(
            predicted_duration,
            predicted_prosody,
            durations,
            pitch,
            voiced_ratio,
            mask,
        )
        self.assertTrue(torch.isfinite(losses.total))
        self.assertTrue(torch.allclose(
            losses.total, losses.duration + losses.pitch + losses.voicing
        ))


if __name__ == "__main__":
    unittest.main()
