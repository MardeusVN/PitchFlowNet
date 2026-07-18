"""Masked VITS-style text encoder with relative-position attention."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def sequence_mask(lengths: torch.Tensor, maximum_length: int | None = None) -> torch.Tensor:
    """Return ``[batch, time]`` boolean masks from positive sequence lengths."""
    if lengths.ndim != 1:
        raise ValueError("lengths must have shape [batch]")
    if lengths.numel() == 0 or torch.any(lengths <= 0):
        raise ValueError("all sequence lengths must be positive")
    maximum = int(lengths.max()) if maximum_length is None else int(maximum_length)
    if maximum < int(lengths.max()):
        raise ValueError("maximum_length is smaller than a sequence length")
    return torch.arange(maximum, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


@dataclass(frozen=True)
class TextEncoderConfig:
    num_symbols: int
    latent_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    num_heads: int = 2
    num_layers: int = 6
    kernel_size: int = 3
    dropout: float = 0.1
    relative_position_window: int = 4
    padding_id: int = 0

    def validate(self) -> None:
        if self.num_symbols <= 0:
            raise ValueError("num_symbols must be positive")
        if min(self.latent_channels, self.hidden_channels, self.filter_channels) <= 0:
            raise ValueError("encoder channel counts must be positive")
        if self.hidden_channels % self.num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")
        if self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("num_layers and num_heads must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.relative_position_window < 0:
            raise ValueError("relative_position_window cannot be negative")
        if not 0 <= self.padding_id < self.num_symbols:
            raise ValueError("padding_id must be inside the symbol vocabulary")


@dataclass
class TextEncoderOutput:
    hidden: torch.Tensor
    prior_mean: torch.Tensor
    prior_log_scale: torch.Tensor
    mask: torch.Tensor


class RelativePositionAttention(nn.Module):
    """VITS-style self-attention with learned relative keys and values."""

    def __init__(self, channels: int, num_heads: int, dropout: float, window: int):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("attention channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.window = window
        self.query_projection = nn.Linear(channels, channels)
        self.key_projection = nn.Linear(channels, channels)
        self.value_projection = nn.Linear(channels, channels)
        self.output_projection = nn.Linear(channels, channels)
        self.attention_dropout = nn.Dropout(dropout)
        relative_std = self.head_channels**-0.5
        # VITS shares relative embeddings across attention heads by default.
        self.relative_key = nn.Parameter(
            torch.randn(1, 2 * window + 1, self.head_channels) * relative_std
        )
        self.relative_value = nn.Parameter(
            torch.randn(1, 2 * window + 1, self.head_channels) * relative_std
        )
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)

    def _relative_embeddings(self, embeddings: torch.Tensor, length: int) -> torch.Tensor:
        pad_length = max(length - (self.window + 1), 0)
        slice_start = max((self.window + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        if pad_length:
            embeddings = F.pad(embeddings, (0, 0, pad_length, pad_length))
        return embeddings[:, slice_start:slice_end]

    @staticmethod
    def _relative_to_absolute(values: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = values.shape
        values = F.pad(values, (0, 1))
        values = values.reshape(batch, heads, length * 2 * length)
        values = F.pad(values, (0, length - 1))
        values = values.reshape(batch, heads, length + 1, 2 * length - 1)
        return values[:, :, :length, length - 1 :]

    @staticmethod
    def _absolute_to_relative(values: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = values.shape
        values = F.pad(values, (0, length - 1))
        values = values.reshape(batch, heads, length * (2 * length - 1))
        values = F.pad(values, (length, 0))
        return values.reshape(batch, heads, length, 2 * length)[:, :, :, 1:]

    def forward(self, inputs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, time, _ = inputs.shape
        query = self.query_projection(inputs).reshape(
            batch, time, self.num_heads, self.head_channels
        ).transpose(1, 2)
        key = self.key_projection(inputs).reshape(
            batch, time, self.num_heads, self.head_channels
        ).transpose(1, 2)
        value = self.value_projection(inputs).reshape(
            batch, time, self.num_heads, self.head_channels
        ).transpose(1, 2)

        scaled_query = query / math.sqrt(self.head_channels)
        scores = torch.matmul(scaled_query, key.transpose(-2, -1))
        relative_key = self._relative_embeddings(self.relative_key, time)
        relative_logits = torch.matmul(
            scaled_query, relative_key.unsqueeze(0).transpose(-2, -1)
        )
        scores = scores + self._relative_to_absolute(relative_logits)
        scores = scores.masked_fill(~valid[:, None, None, :], -1e4)
        probabilities = self.attention_dropout(torch.softmax(scores, dim=-1))

        output = torch.matmul(probabilities, value)
        relative_probabilities = self._absolute_to_relative(probabilities)
        relative_value = self._relative_embeddings(self.relative_value, time)
        output = output + torch.matmul(relative_probabilities, relative_value.unsqueeze(0))
        output = output.transpose(1, 2).contiguous().reshape(batch, time, self.channels)
        output = self.output_projection(output)
        return output * valid.unsqueeze(-1).to(output.dtype)


class ConvolutionalFeedForward(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.input_projection = nn.Conv1d(
            hidden_channels, filter_channels, kernel_size, padding=padding
        )
        self.output_projection = nn.Conv1d(
            filter_channels, hidden_channels, kernel_size, padding=padding
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = inputs.transpose(1, 2) * mask
        values = self.input_projection(values * mask)
        values = self.dropout(F.relu(values))
        values = self.output_projection(values * mask)
        return (values * mask).transpose(1, 2)


class TextEncoderBlock(nn.Module):
    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.attention = RelativePositionAttention(
            config.hidden_channels,
            config.num_heads,
            config.dropout,
            config.relative_position_window,
        )
        self.feed_forward = ConvolutionalFeedForward(
            config.hidden_channels,
            config.filter_channels,
            config.kernel_size,
            config.dropout,
        )
        self.attention_norm = nn.LayerNorm(config.hidden_channels)
        self.feed_forward_norm = nn.LayerNorm(config.hidden_channels)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mask = valid.unsqueeze(1).to(inputs.dtype)
        values = self.attention(inputs, valid)
        inputs = self.attention_norm(inputs + self.dropout(values))
        inputs = inputs * valid.unsqueeze(-1).to(inputs.dtype)
        values = self.feed_forward(inputs, mask)
        inputs = self.feed_forward_norm(inputs + self.dropout(values))
        return inputs * valid.unsqueeze(-1).to(inputs.dtype)


class TextEncoder(nn.Module):
    """Encode phoneme ids and parameterize the text-side latent prior.

    Outputs use the channel-first VITS contract ``[batch, channels, time]`` so
    they can be consumed directly by MAS and the duration/F0 predictors.
    """

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        # ID 0 is both an in-sequence blank symbol and the value used to fill
        # padded batch positions. It must remain trainable; sequence masks,
        # not nn.Embedding.padding_idx, exclude padding (as in VITS/EdgeTTS).
        self.embedding = nn.Embedding(
            config.num_symbols,
            config.hidden_channels,
        )
        nn.init.normal_(self.embedding.weight, 0.0, config.hidden_channels**-0.5)
        self.blocks = nn.ModuleList(
            TextEncoderBlock(config) for _ in range(config.num_layers)
        )
        self.prior_projection = nn.Conv1d(
            config.hidden_channels, 2 * config.latent_channels, kernel_size=1
        )

    def forward(
        self, phoneme_ids: torch.Tensor, phoneme_lengths: torch.Tensor
    ) -> TextEncoderOutput:
        if phoneme_ids.ndim != 2:
            raise ValueError("phoneme_ids must have shape [batch, time]")
        if phoneme_lengths.ndim != 1 or len(phoneme_lengths) != phoneme_ids.shape[0]:
            raise ValueError("phoneme_lengths must have shape [batch]")
        if phoneme_ids.dtype != torch.long:
            raise ValueError("phoneme_ids must use torch.long")
        if phoneme_ids.numel() == 0:
            raise ValueError("phoneme_ids cannot be empty")
        if torch.any(phoneme_ids < 0) or torch.any(phoneme_ids >= self.config.num_symbols):
            raise ValueError("phoneme_ids contains a symbol outside the vocabulary")

        phoneme_lengths = phoneme_lengths.to(device=phoneme_ids.device)
        valid = sequence_mask(phoneme_lengths, phoneme_ids.shape[1])
        mask = valid.unsqueeze(1).to(self.embedding.weight.dtype)
        hidden = self.embedding(phoneme_ids) * math.sqrt(self.config.hidden_channels)
        hidden = hidden * valid.unsqueeze(-1).to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, valid)
        hidden = hidden.transpose(1, 2) * mask
        statistics = self.prior_projection(hidden) * mask
        prior_mean, prior_log_scale = statistics.chunk(2, dim=1)
        return TextEncoderOutput(
            hidden=hidden,
            prior_mean=prior_mean,
            prior_log_scale=prior_log_scale,
            mask=mask,
        )
