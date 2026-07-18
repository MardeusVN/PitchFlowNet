"""Checkpoint loading and English text-to-waveform inference."""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from banhmi_tts.preprocessing.config import PreprocessConfig
from banhmi_tts.preprocessing.text import UNK, encode_phonemes, phonemize_batch
from banhmi_tts.training.config import ModelConfig
from banhmi_tts.training.model import BanhmiTTSModel, InferenceOutput


@dataclass(frozen=True)
class TextFrontend:
    preprocess_config: PreprocessConfig
    vocabulary: Mapping[str, int]

    @property
    def sample_rate(self) -> int:
        return self.preprocess_config.audio.sample_rate

    @property
    def hop_length(self) -> int:
        return self.preprocess_config.spectrogram.hop_length

    def encode(self, text: str) -> "EncodedText":
        if not text or not text.strip():
            raise ValueError("inference text cannot be empty")
        sequences, _ = phonemize_batch([text], self.preprocess_config.text)
        phonemes = sequences[0]
        ids = encode_phonemes(phonemes, self.vocabulary)
        unknown_phonemes = sorted(
            {symbol for symbol in phonemes if symbol not in self.vocabulary}
        )
        return EncodedText(
            text=text,
            phonemes=tuple(phonemes),
            phoneme_ids=tuple(ids),
            unknown_phonemes=tuple(unknown_phonemes),
        )


@dataclass(frozen=True)
class EncodedText:
    text: str
    phonemes: tuple[str, ...]
    phoneme_ids: tuple[int, ...]
    unknown_phonemes: tuple[str, ...]

    def tensors(
        self, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.tensor([self.phoneme_ids], dtype=torch.long, device=device)
        lengths = torch.tensor(
            [len(self.phoneme_ids)], dtype=torch.long, device=device
        )
        return ids, lengths


@dataclass(frozen=True)
class LoadedInferenceModel:
    model: BanhmiTTSModel
    model_config: ModelConfig
    global_step: int
    checkpoint_path: Path


def load_text_frontend(dataset_directory: str | Path) -> TextFrontend:
    directory = Path(dataset_directory)
    config_path = directory if directory.is_file() else directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"preprocessed config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = json.load(config_file)
    preprocess_config = PreprocessConfig.from_dict(payload["preprocessing"])
    frontend = payload["frontend"]
    vocabulary = {
        symbol: int(values[0])
        for symbol, values in frontend["phoneme_id_map"].items()
    }
    if UNK not in vocabulary:
        raise ValueError("frontend vocabulary is missing <unk>")
    if len(vocabulary) != int(frontend["num_symbols"]):
        raise ValueError("frontend vocabulary size does not match num_symbols")
    if sorted(vocabulary.values()) != list(range(len(vocabulary))):
        raise ValueError("frontend vocabulary ids must be contiguous from zero")
    return TextFrontend(preprocess_config, vocabulary)


def load_inference_model(
    checkpoint_path: str | Path,
    device: torch.device | str,
    *,
    remove_weight_norm: bool = True,
) -> LoadedInferenceModel:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    target_device = torch.device(device)
    state = torch.load(path, map_location=target_device, weights_only=True)
    saved_training_config = state.get("training_config")
    if (
        not isinstance(saved_training_config, dict)
        or "model" not in saved_training_config
    ):
        raise ValueError("checkpoint does not contain a serialized model config")
    model_config = ModelConfig.from_dict(saved_training_config["model"])
    model = BanhmiTTSModel(model_config).to(target_device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    if remove_weight_norm:
        model.remove_weight_norm()
    return LoadedInferenceModel(
        model=model,
        model_config=model_config,
        global_step=int(state.get("global_step", 0)),
        checkpoint_path=path,
    )


def synthesize_text(
    loaded: LoadedInferenceModel,
    frontend: TextFrontend,
    text: str,
    *,
    noise_scale: float = 0.667,
    duration_noise_scale: float = 0.8,
    length_scale: float = 1.0,
    maximum_frames: int = 10_000,
    seed: int = 1234,
) -> tuple[InferenceOutput, EncodedText]:
    if len(frontend.vocabulary) != loaded.model_config.text_encoder.num_symbols:
        raise ValueError(
            "dataset vocabulary size does not match the checkpoint text encoder"
        )
    if frontend.hop_length != loaded.model_config.segment.hop_length:
        raise ValueError("dataset hop length does not match the checkpoint model")
    encoded = frontend.encode(text)
    device = next(loaded.model.parameters()).device
    phoneme_ids, phoneme_lengths = encoded.tensors(device)
    random_generator = torch.Generator(device=device)
    random_generator.manual_seed(seed)
    output = loaded.model.infer(
        phoneme_ids,
        phoneme_lengths,
        noise_scale=noise_scale,
        duration_noise_scale=duration_noise_scale,
        length_scale=length_scale,
        maximum_frames=maximum_frames,
        generator=random_generator,
    )
    return output, encoded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthesize English speech")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda")
    )
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--duration-noise-scale", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--maximum-frames", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--keep-weight-norm",
        action="store_true",
        help="keep weight-norm parametrizations for parity debugging",
    )
    return parser


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def write_pcm16(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pcm = (
        waveform.detach()
        .float()
        .cpu()
        .clamp(-1.0, 1.0)
        .mul(32767.0)
        .round()
        .to(torch.int16)
        .tolist()
    )
    samples = array("h", pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    with wave.open(str(destination), "wb") as output_file:
        output_file.setnchannels(1)
        output_file.setsampwidth(2)
        output_file.setframerate(sample_rate)
        output_file.writeframes(samples.tobytes())


def main() -> None:
    args = _parser().parse_args()
    device = _resolve_device(args.device)
    frontend = load_text_frontend(args.dataset_dir)
    loaded = load_inference_model(
        args.checkpoint,
        device,
        remove_weight_norm=not args.keep_weight_norm,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output, encoded = synthesize_text(
        loaded,
        frontend,
        args.text,
        noise_scale=args.noise_scale,
        duration_noise_scale=args.duration_noise_scale,
        length_scale=args.length_scale,
        maximum_frames=args.maximum_frames,
        seed=args.seed,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    sample_count = int(output.audio_lengths[0].item())
    waveform = output.audio[0, 0, :sample_count]
    write_pcm16(args.output, waveform, frontend.sample_rate)
    duration_seconds = sample_count / frontend.sample_rate
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "checkpoint_step": loaded.global_step,
                "device": str(device),
                "phonemes": "".join(encoded.phonemes),
                "phoneme_ids": list(encoded.phoneme_ids),
                "unknown_phonemes": list(encoded.unknown_phonemes),
                "frames": int(output.frame_lengths[0].item()),
                "samples": sample_count,
                "audio_seconds": duration_seconds,
                "inference_seconds": elapsed,
                "real_time_factor": elapsed / max(duration_seconds, 1e-9),
            },
            # Keep CLI output portable on Windows consoles using cp1252 while
            # preserving every IPA code point through JSON escapes.
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "EncodedText",
    "LoadedInferenceModel",
    "TextFrontend",
    "load_inference_model",
    "load_text_frontend",
    "synthesize_text",
    "write_pcm16",
]
