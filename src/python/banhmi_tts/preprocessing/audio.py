"""Audio decoding, validation, spectrogram, and pitch extraction."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

from .config import PreprocessConfig


class AudioRejected(ValueError):
    """Raised when an audio file violates a hard quality constraint."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(
    source_sha256: str, utterance_id: str, text: str, config_fingerprint: str
) -> str:
    digest = hashlib.sha256()
    for value in (source_sha256, utterance_id, text, config_fingerprint):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_audio(path: Path, sample_rate: int):
    import numpy as np
    import soundfile
    import soxr

    try:
        decoded, decoded_rate = soundfile.read(
            path, dtype="float32", always_2d=True
        )
    except Exception as exc:
        raise AudioRejected("decode_error", str(exc)) from exc
    if decoded.size == 0:
        raise AudioRejected("empty_audio", f"invalid decoded shape: {decoded.shape}")
    audio = decoded.mean(axis=1, dtype=np.float32)
    if decoded_rate != sample_rate:
        audio = soxr.resample(audio, decoded_rate, sample_rate, quality="HQ")
        audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise AudioRejected("empty_audio", f"invalid decoded shape: {audio.shape}")
    if not np.isfinite(audio).all():
        raise AudioRejected("non_finite_audio", "audio contains NaN or infinity")
    return audio


def audio_quality(audio, sample_rate: int, config: PreprocessConfig) -> Dict[str, float]:
    import numpy as np

    quality = config.quality
    duration = float(audio.size / sample_rate)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    rms_dbfs = float(20.0 * math.log10(max(rms, 1e-12)))
    clipping_ratio = float(np.mean(np.abs(audio) >= quality.clipping_threshold))

    frame_size = max(1, int(round(sample_rate * quality.silence_frame_ms / 1000.0)))
    num_frames = int(math.ceil(audio.size / frame_size))
    padded = np.pad(audio, (0, num_frames * frame_size - audio.size))
    frames = padded.reshape(num_frames, frame_size)
    frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    frame_dbfs = 20.0 * np.log10(np.maximum(frame_rms, 1e-12))
    silent = frame_dbfs < quality.silence_threshold_dbfs

    leading_frames = 0
    for is_silent in silent:
        if not is_silent:
            break
        leading_frames += 1
    trailing_frames = 0
    for is_silent in silent[::-1]:
        if not is_silent:
            break
        trailing_frames += 1

    metrics = {
        "duration_seconds": duration,
        "peak": peak,
        "rms_dbfs": rms_dbfs,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": float(np.mean(silent)),
        "leading_silence_seconds": float(leading_frames * frame_size / sample_rate),
        "trailing_silence_seconds": float(trailing_frames * frame_size / sample_rate),
    }

    if duration < quality.min_duration_seconds:
        raise AudioRejected("duration_too_short", f"duration={duration:.3f}s")
    if duration > quality.max_duration_seconds:
        raise AudioRejected("duration_too_long", f"duration={duration:.3f}s")
    if rms_dbfs < quality.min_rms_dbfs:
        raise AudioRejected("audio_too_quiet", f"rms={rms_dbfs:.2f} dBFS")
    if clipping_ratio > quality.max_clipping_ratio:
        raise AudioRejected(
            "audio_clipped", f"clipping_ratio={clipping_ratio:.6f}"
        )
    return metrics


def spectrogram(audio, config: PreprocessConfig):
    import torch
    from torch.nn import functional as functional

    spec_config = config.spectrogram
    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    pad = (spec_config.n_fft - spec_config.hop_length) // 2
    if waveform.shape[-1] <= pad:
        raise AudioRejected(
            "audio_too_short_for_stft",
            f"samples={waveform.shape[-1]}, reflection_pad={pad}",
        )
    waveform_padded = functional.pad(
        waveform.unsqueeze(1), (pad, pad), mode="reflect"
    ).squeeze(1)
    window = torch.hann_window(spec_config.win_length, dtype=waveform.dtype)
    complex_spec = torch.stft(
        waveform_padded,
        n_fft=spec_config.n_fft,
        hop_length=spec_config.hop_length,
        win_length=spec_config.win_length,
        window=window,
        center=spec_config.center,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = torch.sqrt(complex_spec.abs().pow(2) + 1e-6).squeeze(0)
    if not torch.isfinite(magnitude).all():
        raise AudioRejected("non_finite_spectrogram", "spectrogram contains NaN/Inf")
    return waveform, magnitude


def extract_pitch(audio, num_frames: int, config: PreprocessConfig) -> Dict[str, Any]:
    """Extract WORLD pitch and align it to STFT frame centers without VAD.

    Unvoiced frames remain explicitly unvoiced. Log-F0 is interpolated from
    voiced WORLD observations only and is exposed only where the nearest WORLD
    observation is voiced.
    """
    import numpy as np
    import pyworld
    import torch

    sample_rate = config.audio.sample_rate
    hop_length = config.spectrogram.hop_length
    frame_period_ms = hop_length / sample_rate * 1000.0
    audio64 = audio.astype(np.float64, copy=False)
    f0, times = pyworld.dio(
        audio64,
        sample_rate,
        f0_floor=config.pitch.f0_floor,
        f0_ceil=config.pitch.f0_ceil,
        frame_period=frame_period_ms,
    )
    f0 = pyworld.stonemask(audio64, f0, times, sample_rate)
    if f0.size == 0 or times.size == 0:
        raise AudioRejected("pitch_extraction_failed", "WORLD returned no frames")

    target_times = (
        np.arange(num_frames, dtype=np.float64) * hop_length + hop_length / 2.0
    ) / sample_rate
    right = np.searchsorted(times, target_times, side="left")
    right = np.clip(right, 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    choose_left = np.abs(target_times - times[left]) <= np.abs(times[right] - target_times)
    nearest = np.where(choose_left, left, right)
    voiced = f0[nearest] > 0.0

    source_voiced = f0 > 0.0
    log_f0 = np.zeros(num_frames, dtype=np.float32)
    if source_voiced.any():
        interpolated = np.interp(
            target_times, times[source_voiced], np.log(f0[source_voiced])
        )
        log_f0[voiced] = interpolated[voiced].astype(np.float32)
    f0_hz = np.zeros(num_frames, dtype=np.float32)
    f0_hz[voiced] = np.exp(log_f0[voiced]).astype(np.float32)

    if not np.isfinite(log_f0).all() or not np.isfinite(f0_hz).all():
        raise AudioRejected("non_finite_pitch", "pitch contains NaN/Inf")

    voiced_log_f0 = log_f0[voiced].astype(np.float64)
    if voiced_log_f0.size == 0:
        raise AudioRejected("no_voiced_pitch", "WORLD found no voiced frames")
    return {
        "f0_hz": torch.from_numpy(f0_hz),
        "log_f0": torch.from_numpy(log_f0),
        "voiced": torch.from_numpy(voiced.astype(np.bool_)),
        "voiced_frames": int(voiced.sum()),
        "log_f0_sum": float(voiced_log_f0.sum()),
        "log_f0_squared_sum": float(np.square(voiced_log_f0).sum()),
    }


def atomic_torch_save(value: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary_name)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def artifact_paths(output_dir: Path, key: str) -> Tuple[Path, Path, Path, Path]:
    prefix = key[:2]
    return (
        output_dir / "artifacts" / "audio" / prefix / f"{key}.pt",
        output_dir / "artifacts" / "spectrogram" / prefix / f"{key}.pt",
        output_dir / "artifacts" / "pitch" / prefix / f"{key}.pt",
        output_dir / "records" / prefix / f"{key}.json",
    )
