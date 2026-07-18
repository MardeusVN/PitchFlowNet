import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from banhmi_tts.inference import load_inference_model
from banhmi_tts.model import (
    FlowConfig,
    GeneratorConfig,
    PosteriorEncoderConfig,
    PredictorConfig,
    SegmentConfig,
    TextEncoderConfig,
)
from banhmi_tts.training import BanhmiTTSModel, ModelConfig, TrainingConfig


def inference_config() -> ModelConfig:
    return ModelConfig(
        text_encoder=TextEncoderConfig(
            num_symbols=12,
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
            duration_type="deterministic",
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
    )


class InferenceTests(unittest.TestCase):
    def model(self) -> BanhmiTTSModel:
        model = BanhmiTTSModel(inference_config()).eval()
        torch.nn.init.zeros_(model.duration_predictor.output_projection.weight)
        torch.nn.init.zeros_(model.duration_predictor.output_projection.bias)
        return model

    def test_tensor_inference_expands_durations_and_masks_audio(self):
        model = self.model()
        ids = torch.tensor([[1, 4, 2], [1, 2, 0]])
        lengths = torch.tensor([3, 2])
        output = model.infer(
            ids,
            lengths,
            noise_scale=0.0,
            duration_noise_scale=0.0,
        )
        self.assertTrue(torch.equal(output.durations, torch.tensor([[1, 1, 1], [1, 1, 0]])))
        self.assertTrue(torch.equal(output.frame_lengths, torch.tensor([3, 2])))
        self.assertTrue(torch.equal(output.audio_lengths, torch.tensor([24, 16])))
        self.assertEqual(output.audio.shape, (2, 1, 24))
        self.assertTrue(torch.equal(output.audio[1, :, 16:], torch.zeros(1, 8)))
        self.assertTrue(torch.equal(output.alignment.sum(dim=-1), torch.tensor([[[1., 1., 1.]], [[1., 1., 0.]]])))

    def test_length_scale_controls_decoded_frame_count(self):
        model = self.model()
        output = model.infer(
            torch.tensor([[1, 4, 2]]),
            torch.tensor([3]),
            noise_scale=0.0,
            length_scale=2.0,
        )
        self.assertTrue(torch.equal(output.durations, torch.tensor([[2, 2, 2]])))
        self.assertEqual(int(output.frame_lengths[0]), 6)

    def test_maximum_frames_stops_runaway_duration(self):
        model = self.model()
        with self.assertRaises(ValueError):
            model.infer(
                torch.tensor([[1, 4, 2]]),
                torch.tensor([3]),
                noise_scale=0.0,
                maximum_frames=2,
            )

    def test_checkpoint_loader_reconstructs_model_config(self):
        model = self.model()
        training_config = TrainingConfig(model=inference_config())
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.pt"
            torch.save(
                {
                    "global_step": 17,
                    "training_config": asdict(training_config),
                    "model": model.state_dict(),
                },
                path,
            )
            loaded = load_inference_model(
                path, "cpu", remove_weight_norm=False
            )
        self.assertEqual(loaded.global_step, 17)
        self.assertEqual(loaded.model_config, inference_config())
        for expected, actual in zip(model.parameters(), loaded.model.parameters()):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
