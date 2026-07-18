"""Cython-accelerated monotonic alignment search used by VITS/EdgeTTS."""

from __future__ import annotations

import numpy as np
import torch

from ._core import maximum_path_c


def maximum_path(
    scores: torch.Tensor,
    acoustic_lengths: torch.Tensor,
    text_lengths: torch.Tensor,
) -> torch.Tensor:
    """Run the EdgeTTS Cython dynamic program and return on the input device."""
    device = scores.device
    dtype = scores.dtype
    values = scores.detach().float().cpu().numpy().astype(np.float32, copy=True)
    paths = np.zeros(values.shape, dtype=np.int32)
    acoustic = acoustic_lengths.detach().to("cpu", torch.int32).numpy()
    text = text_lengths.detach().to("cpu", torch.int32).numpy()
    maximum_path_c(paths, values, acoustic, text)
    return torch.from_numpy(paths).to(device=device, dtype=dtype)
