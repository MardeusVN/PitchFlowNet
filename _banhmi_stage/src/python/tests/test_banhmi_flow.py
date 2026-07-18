import unittest

import torch

from banhmi_tts.model import FlowConfig, TransformerCouplingFlow


class TransformerFlowTests(unittest.TestCase):
    def make_flow(self, mean_only: bool = True) -> TransformerCouplingFlow:
        return TransformerCouplingFlow(
            FlowConfig(
                channels=16,
                hidden_channels=16,
                filter_channels=32,
                num_heads=2,
                transformer_layers=1,
                kernel_size=3,
                wavenet_layers=2,
                num_flows=2,
                dropout=0.0,
                mean_only=mean_only,
                use_weight_norm=False,
            )
        )

    def test_forward_reverse_reconstructs_valid_latent(self):
        torch.manual_seed(3)
        flow = self.make_flow(mean_only=False).eval()
        with torch.no_grad():
            for coupling in flow.couplings:
                coupling.conditioner.output_projection.weight.normal_(0.0, 0.02)
                coupling.conditioner.output_projection.bias.normal_(0.0, 0.02)
        inputs = torch.randn(2, 16, 11)
        mask = torch.zeros(2, 1, 11)
        mask[0, :, :7] = 1
        mask[1, :, :11] = 1
        forward = flow(inputs, mask)
        reverse = flow(forward.latent, mask, reverse=True)
        valid = mask.bool().expand_as(inputs)
        self.assertTrue(torch.allclose(reverse.latent[valid], inputs[valid], atol=1e-5))
        self.assertTrue((reverse.latent[~valid] == 0).all())
        self.assertTrue(
            torch.allclose(
                forward.log_determinant + reverse.log_determinant,
                torch.zeros(2),
                atol=1e-5,
            )
        )

    def test_mean_only_flow_has_zero_log_determinant(self):
        flow = self.make_flow(mean_only=True)
        output = flow(torch.randn(2, 16, 9), torch.ones(2, 1, 9))
        self.assertTrue(torch.equal(output.log_determinant, torch.zeros(2)))

    def test_identity_initialization_and_backward(self):
        flow = self.make_flow(mean_only=True)
        inputs = torch.randn(2, 16, 9)
        mask = torch.ones(2, 1, 9)
        output = flow(inputs, mask)
        # Two flips return the original channel order; zero post projections
        # make every coupling identity at initialization.
        self.assertTrue(torch.allclose(output.latent, inputs, atol=1e-6))
        output.latent.square().mean().backward()
        gradient = flow.couplings[0].conditioner.output_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_padding_values_do_not_affect_valid_outputs(self):
        flow = self.make_flow().eval()
        first = torch.randn(1, 16, 10)
        second = first.clone()
        second[:, :, 6:] = 1000.0
        mask = torch.zeros(1, 1, 10)
        mask[:, :, :6] = 1
        with torch.no_grad():
            first_output = flow(first, mask).latent[:, :, :6]
            second_output = flow(second, mask).latent[:, :, :6]
        self.assertTrue(torch.allclose(first_output, second_output, atol=1e-6))

    def test_invalid_config_and_shape(self):
        with self.assertRaises(ValueError):
            FlowConfig(channels=15).validate()
        flow = self.make_flow()
        with self.assertRaises(ValueError):
            flow(torch.randn(1, 15, 8), torch.ones(1, 1, 8))


if __name__ == "__main__":
    unittest.main()
