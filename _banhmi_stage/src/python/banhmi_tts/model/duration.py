"""EdgeTTS/VITS2 stochastic duration predictor and duration discriminator."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class _ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class _DDSConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.ModuleList()
        self.pointwise = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for index in range(layers):
            dilation = kernel_size**index
            padding = (kernel_size * dilation - dilation) // 2
            self.depthwise.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.pointwise.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(_ChannelLayerNorm(channels))
            self.norms_2.append(_ChannelLayerNorm(channels))

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor, conditioning: torch.Tensor | None = None
    ) -> torch.Tensor:
        if conditioning is not None:
            values = values + conditioning
        for depthwise, pointwise, norm_1, norm_2 in zip(
            self.depthwise, self.pointwise, self.norms_1, self.norms_2
        ):
            residual = F.gelu(norm_1(depthwise(values * mask)))
            residual = F.gelu(norm_2(pointwise(residual)))
            values = values + self.dropout(residual)
        return values * mask


def _searchsorted(bins: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    bins = bins.clone()
    bins[..., -1] += 1e-6
    return torch.sum(inputs.unsqueeze(-1) >= bins, dim=-1) - 1


def _rational_quadratic_spline(
    inputs: torch.Tensor,
    raw_widths: torch.Tensor,
    raw_heights: torch.Tensor,
    raw_derivatives: torch.Tensor,
    *,
    inverse: bool,
    bound: float = 5.0,
    minimum_width: float = 1e-3,
    minimum_height: float = 1e-3,
    minimum_derivative: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    inside = (inputs >= -bound) & (inputs <= bound)
    outputs = inputs.clone()
    logabsdet = torch.zeros_like(inputs)
    derivatives = F.pad(raw_derivatives, (1, 1))
    boundary_value = math.log(math.exp(1.0 - minimum_derivative) - 1.0)
    derivatives[..., 0] = boundary_value
    derivatives[..., -1] = boundary_value
    if not torch.any(inside):
        return outputs, logabsdet

    selected_inputs = inputs[inside]
    selected_widths = raw_widths[inside]
    selected_heights = raw_heights[inside]
    selected_derivatives = derivatives[inside]
    bins = selected_widths.shape[-1]

    widths = minimum_width + (1.0 - minimum_width * bins) * F.softmax(
        selected_widths, dim=-1
    )
    cumulative_widths = F.pad(torch.cumsum(widths, dim=-1), (1, 0))
    cumulative_widths = 2.0 * bound * cumulative_widths - bound
    cumulative_widths[..., 0], cumulative_widths[..., -1] = -bound, bound
    widths = cumulative_widths[..., 1:] - cumulative_widths[..., :-1]

    heights = minimum_height + (1.0 - minimum_height * bins) * F.softmax(
        selected_heights, dim=-1
    )
    cumulative_heights = F.pad(torch.cumsum(heights, dim=-1), (1, 0))
    cumulative_heights = 2.0 * bound * cumulative_heights - bound
    cumulative_heights[..., 0], cumulative_heights[..., -1] = -bound, bound
    heights = cumulative_heights[..., 1:] - cumulative_heights[..., :-1]
    derivatives = minimum_derivative + F.softplus(selected_derivatives)

    indices = _searchsorted(
        cumulative_heights if inverse else cumulative_widths, selected_inputs
    ).clamp(0, bins - 1).unsqueeze(-1)
    input_cumwidths = cumulative_widths.gather(-1, indices).squeeze(-1)
    input_widths = widths.gather(-1, indices).squeeze(-1)
    input_cumheights = cumulative_heights.gather(-1, indices).squeeze(-1)
    input_heights = heights.gather(-1, indices).squeeze(-1)
    deltas = heights / widths
    input_delta = deltas.gather(-1, indices).squeeze(-1)
    input_derivative = derivatives.gather(-1, indices).squeeze(-1)
    next_derivative = derivatives[..., 1:].gather(-1, indices).squeeze(-1)

    if inverse:
        offset = selected_inputs - input_cumheights
        a = offset * (input_derivative + next_derivative - 2 * input_delta)
        a = a + input_heights * (input_delta - input_derivative)
        b = input_heights * input_derivative
        b = b - offset * (input_derivative + next_derivative - 2 * input_delta)
        c = -input_delta * offset
        discriminant = (b.square() - 4 * a * c).clamp_min(0)
        root = (2 * c) / (-b - torch.sqrt(discriminant)).clamp_max(-1e-12)
        theta = root
        transformed = root * input_widths + input_cumwidths
    else:
        theta = (selected_inputs - input_cumwidths) / input_widths
        theta_one_minus = theta * (1 - theta)
        numerator = input_heights * (
            input_delta * theta.square() + input_derivative * theta_one_minus
        )
        denominator = input_delta + (
            input_derivative + next_derivative - 2 * input_delta
        ) * theta_one_minus
        transformed = input_cumheights + numerator / denominator

    theta_one_minus = theta * (1 - theta)
    denominator = input_delta + (
        input_derivative + next_derivative - 2 * input_delta
    ) * theta_one_minus
    derivative_numerator = input_delta.square() * (
        next_derivative * theta.square()
        + 2 * input_delta * theta_one_minus
        + input_derivative * (1 - theta).square()
    )
    selected_logdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
    outputs[inside] = transformed
    logabsdet[inside] = -selected_logdet if inverse else selected_logdet
    return outputs, logabsdet


class _LogFlow(nn.Module):
    def forward(self, values: torch.Tensor, mask: torch.Tensor):
        output = torch.log(values.clamp_min(1e-5)) * mask
        return output, torch.sum(-output, dim=(1, 2))


class _Flip(nn.Module):
    def forward(self, values: torch.Tensor, mask: torch.Tensor, *, reverse: bool, **_):
        output = torch.flip(values, dims=(1,)) * mask
        if reverse:
            return output
        return output, torch.zeros(values.shape[0], device=values.device, dtype=values.dtype)


class _ElementwiseAffine(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, values: torch.Tensor, mask: torch.Tensor, *, reverse: bool, **_):
        if reverse:
            return (values - self.shift) * torch.exp(-self.log_scale) * mask
        output = (self.shift + torch.exp(self.log_scale) * values) * mask
        logdet = torch.sum(self.log_scale * mask, dim=(1, 2))
        return output, logdet


class _ConvFlow(nn.Module):
    def __init__(self, channels: int, hidden: int, kernel_size: int, layers: int) -> None:
        super().__init__()
        if channels != 2:
            raise ValueError("duration ConvFlow expects two channels")
        self.hidden = hidden
        self.bins = 10
        self.pre = nn.Conv1d(1, hidden, 1)
        self.convs = _DDSConv(hidden, kernel_size, layers, 0.0)
        self.projection = nn.Conv1d(hidden, self.bins * 3 - 1, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        *,
        conditioning: torch.Tensor | None = None,
        reverse: bool,
    ):
        first, second = values.chunk(2, dim=1)
        hidden = self.convs(self.pre(first), mask, conditioning)
        statistics = self.projection(hidden).transpose(1, 2).unsqueeze(1)
        widths = statistics[..., : self.bins] / math.sqrt(self.hidden)
        heights = statistics[..., self.bins : 2 * self.bins] / math.sqrt(self.hidden)
        derivatives = statistics[..., 2 * self.bins :]
        transformed, logabsdet = _rational_quadratic_spline(
            second, widths, heights, derivatives, inverse=reverse
        )
        output = torch.cat((first, transformed), dim=1) * mask
        if reverse:
            return output
        return output, torch.sum(logabsdet * mask, dim=(1, 2))


class StochasticDurationPredictor(nn.Module):
    """VITS stochastic duration flow, matching EdgeTTS's forward/reverse API."""

    def __init__(self, config) -> None:
        super().__init__()
        channels = config.input_channels
        self.input_projection = nn.Conv1d(channels, channels, 1)
        self.input_convs = _DDSConv(channels, config.kernel_size, 3, config.dropout)
        self.input_output = nn.Conv1d(channels, channels, 1)
        self.log_flow = _LogFlow()
        self.flows = nn.ModuleList([_ElementwiseAffine(2)])
        for _ in range(config.duration_flows):
            self.flows.extend((_ConvFlow(2, channels, config.kernel_size, 3), _Flip()))

        self.posterior_input = nn.Conv1d(1, channels, 1)
        self.posterior_convs = _DDSConv(channels, config.kernel_size, 3, config.dropout)
        self.posterior_output = nn.Conv1d(channels, channels, 1)
        self.posterior_flows = nn.ModuleList([_ElementwiseAffine(2)])
        for _ in range(4):
            self.posterior_flows.extend(
                (_ConvFlow(2, channels, config.kernel_size, 3), _Flip())
            )

    def _condition(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = self.input_projection(hidden.detach())
        values = self.input_convs(values, mask)
        return self.input_output(values) * mask

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        durations: torch.Tensor | None = None,
        *,
        reverse: bool = False,
        noise_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        conditioning = self._condition(hidden, mask)
        if reverse:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]
            latent = torch.randn(
                hidden.shape[0], 2, hidden.shape[2],
                device=hidden.device, dtype=hidden.dtype, generator=generator,
            ) * noise_scale
            for flow in flows:
                latent = flow(
                    latent, mask, conditioning=conditioning, reverse=True
                )
            return latent[:, 0] * mask.squeeze(1)

        if durations is None or durations.shape != (hidden.shape[0], 1, hidden.shape[2]):
            raise ValueError("durations must have shape [batch, 1, text_time]")
        posterior = self.posterior_input(durations)
        posterior = self.posterior_convs(posterior, mask)
        posterior = self.posterior_output(posterior) * mask
        base = torch.randn(
            hidden.shape[0], 2, hidden.shape[2],
            device=hidden.device, dtype=hidden.dtype, generator=generator,
        ) * mask
        latent_q = base
        logdet_q = torch.zeros(hidden.shape[0], device=hidden.device, dtype=hidden.dtype)
        for flow in self.posterior_flows:
            latent_q, logdet = flow(
                latent_q,
                mask,
                conditioning=conditioning + posterior,
                reverse=False,
            )
            logdet_q = logdet_q + logdet
        latent_u, latent_1 = latent_q.chunk(2, dim=1)
        unit = torch.sigmoid(latent_u) * mask
        latent_0 = (durations - unit) * mask
        logdet_q = logdet_q + torch.sum(
            (F.logsigmoid(latent_u) + F.logsigmoid(-latent_u)) * mask,
            dim=(1, 2),
        )
        log_q = torch.sum(
            -0.5 * (math.log(2 * math.pi) + base.square()) * mask,
            dim=(1, 2),
        ) - logdet_q

        latent_0, logdet = self.log_flow(latent_0, mask)
        latent = torch.cat((latent_0, latent_1), dim=1)
        total_logdet = logdet
        for flow in self.flows:
            latent, logdet = flow(
                latent, mask, conditioning=conditioning, reverse=False
            )
            total_logdet = total_logdet + logdet
        nll = torch.sum(
            0.5 * (math.log(2 * math.pi) + latent.square()) * mask,
            dim=(1, 2),
        ) - total_logdet
        return nll + log_q


class DurationDiscriminator(nn.Module):
    """VITS2 phoneme-level discriminator conditioned on text hidden states."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(input_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            ]
        )
        self.norms = nn.ModuleList(
            [_ChannelLayerNorm(hidden_channels), _ChannelLayerNorm(hidden_channels)]
        )
        self.duration_projection = nn.Conv1d(1, hidden_channels, 1)
        self.output_convs = nn.ModuleList(
            [
                nn.Conv1d(2 * hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            ]
        )
        self.output_norms = nn.ModuleList(
            [_ChannelLayerNorm(hidden_channels), _ChannelLayerNorm(hidden_channels)]
        )
        self.output = nn.Linear(hidden_channels, 1)

    def _probability(
        self, hidden: torch.Tensor, mask: torch.Tensor, duration: torch.Tensor
    ) -> torch.Tensor:
        values = torch.cat((hidden, self.duration_projection(duration)), dim=1)
        for convolution, normalization in zip(self.output_convs, self.output_norms):
            values = self.dropout(normalization(F.relu(convolution(values * mask))))
        return torch.sigmoid(self.output((values * mask).transpose(1, 2)))

    def forward(
        self,
        text_hidden: torch.Tensor,
        mask: torch.Tensor,
        real_log_duration: torch.Tensor,
        fake_log_duration: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        values = text_hidden.detach()
        for convolution, normalization in zip(self.convs, self.norms):
            values = self.dropout(normalization(F.relu(convolution(values * mask))))
        real = self._probability(values, mask, real_log_duration)
        fake = self._probability(values, mask, fake_log_duration)
        return [real], [fake]
