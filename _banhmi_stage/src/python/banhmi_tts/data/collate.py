"""Padding and mask construction for variable-length TTS batches."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch

from .dataset import DataContractError


def _length_mask(lengths: torch.Tensor, maximum: int) -> torch.Tensor:
    positions = torch.arange(maximum, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


class BanhmiCollator:
    """Collate full utterances and expose explicit boolean masks."""

    def __init__(self, phoneme_pad_id: int = 0, sort_by_spectrogram_length: bool = True):
        self.phoneme_pad_id = phoneme_pad_id
        self.sort_by_spectrogram_length = sort_by_spectrogram_length

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            raise DataContractError("cannot collate an empty batch")
        items = list(samples)
        if self.sort_by_spectrogram_length:
            items.sort(key=lambda item: item["spectrogram"].shape[-1], reverse=True)

        sample_rates = {int(item["sample_rate"]) for item in items}
        spectrogram_bins = {int(item["spectrogram"].shape[0]) for item in items}
        if len(sample_rates) != 1:
            raise DataContractError(f"mixed sample rates in batch: {sorted(sample_rates)}")
        if len(spectrogram_bins) != 1:
            raise DataContractError(
                f"mixed spectrogram bins in batch: {sorted(spectrogram_bins)}"
            )

        batch_size = len(items)
        text_lengths = torch.tensor(
            [len(item["phoneme_ids"]) for item in items], dtype=torch.long
        )
        spectrogram_lengths = torch.tensor(
            [item["spectrogram"].shape[-1] for item in items], dtype=torch.long
        )
        audio_lengths = torch.tensor(
            [item["audio"].shape[-1] for item in items], dtype=torch.long
        )
        max_text = int(text_lengths.max())
        max_spec = int(spectrogram_lengths.max())
        max_audio = int(audio_lengths.max())
        bins = next(iter(spectrogram_bins))

        phoneme_ids = torch.full(
            (batch_size, max_text), self.phoneme_pad_id, dtype=torch.long
        )
        spectrogram = torch.zeros(batch_size, bins, max_spec, dtype=torch.float32)
        audio = torch.zeros(batch_size, 1, max_audio, dtype=torch.float32)
        f0_hz = torch.zeros(batch_size, max_spec, dtype=torch.float32)
        log_f0 = torch.zeros_like(f0_hz)
        log_f0_continuous = torch.zeros_like(f0_hz)
        voiced = torch.zeros(batch_size, max_spec, dtype=torch.bool)

        for batch_index, item in enumerate(items):
            text_length = int(text_lengths[batch_index])
            spec_length = int(spectrogram_lengths[batch_index])
            audio_length = int(audio_lengths[batch_index])
            phoneme_ids[batch_index, :text_length] = item["phoneme_ids"]
            spectrogram[batch_index, :, :spec_length] = item["spectrogram"]
            audio[batch_index, :, :audio_length] = item["audio"]
            f0_hz[batch_index, :spec_length] = item["f0_hz"]
            log_f0[batch_index, :spec_length] = item["log_f0"]
            log_f0_continuous[batch_index, :spec_length] = item["log_f0_continuous"]
            voiced[batch_index, :spec_length] = item["voiced"]

        text_mask = _length_mask(text_lengths, max_text).unsqueeze(1)
        spectrogram_mask = _length_mask(spectrogram_lengths, max_spec).unsqueeze(1)
        audio_mask = _length_mask(audio_lengths, max_audio).unsqueeze(1)
        return {
            "utterance_ids": [item["utterance_id"] for item in items],
            "phoneme_ids": phoneme_ids,
            "phoneme_lengths": text_lengths,
            "text_mask": text_mask,
            "spectrogram": spectrogram,
            "spectrogram_lengths": spectrogram_lengths,
            "spectrogram_mask": spectrogram_mask,
            "audio": audio,
            "audio_lengths": audio_lengths,
            "audio_mask": audio_mask,
            "f0_hz": f0_hz,
            "log_f0": log_f0,
            "log_f0_continuous": log_f0_continuous,
            "voiced": voiced,
            "sample_rate": next(iter(sample_rates)),
        }
