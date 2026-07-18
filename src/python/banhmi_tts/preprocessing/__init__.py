"""Offline preprocessing for Banhmi-TTS.

The pipeline intentionally does not trim audio or run a voice activity
detector. Source waveforms are decoded, converted to mono, resampled only when
necessary, validated, and cached together with their spectral and pitch
features.
"""

from .config import PreprocessConfig

__all__ = ["PreprocessConfig"]
