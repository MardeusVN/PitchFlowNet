"""Run a one-utterance Mimi encode/decode preflight on CPU by default."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from transformers import MimiModel


def tensor_shape(value: torch.Tensor) -> list[int]:
    return list(value.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model", default="kyutai/mimi")
    parser.add_argument("--num-quantizers", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = MimiModel.from_pretrained(args.model).to(args.device).eval()
    config = model.config

    audio_array, source_rate = sf.read(
        args.audio, dtype="float32", always_2d=True
    )
    waveform = torch.from_numpy(audio_array.T).mean(dim=0, keepdim=True)
    if source_rate != config.sampling_rate:
        waveform = torchaudio.functional.resample(
            waveform, source_rate, config.sampling_rate
        )
    waveform = waveform.unsqueeze(0).to(args.device)
    padding_mask = torch.ones_like(waveform, dtype=torch.bool)

    with torch.inference_mode():
        encoded = model.encode(
            waveform,
            padding_mask=padding_mask,
            num_quantizers=args.num_quantizers,
            return_dict=True,
        )
        codes = encoded.audio_codes

        # Raw codebook-vector dimension and post-projection semantic feature
        # dimension are intentionally both measured; Mimi uses 256-dim VQ
        # vectors projected back to its 512-dim hidden representation.
        semantic_layer = model.quantizer.semantic_residual_vector_quantizer.layers[0]
        c0_raw = semantic_layer.decode(codes[:, 0])
        c0_projected = model.quantizer.semantic_residual_vector_quantizer.decode(
            codes[:, :1]
        )

        decode_reports = {}
        requested_depths = sorted(
            set(depth for depth in (1, 4, 8, 12, args.num_quantizers) if depth <= codes.shape[1])
        )
        for depth in requested_depths:
            decoded = model.decode(
                codes[:, :depth],
                padding_mask=padding_mask,
                return_dict=True,
            ).audio_values
            decoded = decoded[..., : waveform.shape[-1]]
            output_path = args.output_dir / f"oracle_q{depth:02d}.wav"
            sf.write(
                output_path,
                decoded[0].detach().cpu().transpose(0, 1).numpy(),
                config.sampling_rate,
            )
            reference = waveform[..., : decoded.shape[-1]]
            error = reference - decoded
            signal_power = reference.square().mean().item()
            error_power = error.square().mean().item()
            snr_db = 10.0 * math.log10(signal_power / max(error_power, 1e-12))
            decode_reports[str(depth)] = {
                "waveform_shape": tensor_shape(decoded),
                "output_path": str(output_path.resolve()),
                "unaligned_waveform_l1": error.abs().mean().item(),
                "unaligned_waveform_snr_db": snr_db,
            }

    report = {
        "model": args.model,
        "audio": str(args.audio.resolve()),
        "source_sample_rate": source_rate,
        "model_sample_rate": config.sampling_rate,
        "frame_rate": config.frame_rate,
        "configured_num_quantizers": config.num_quantizers,
        "requested_num_quantizers": args.num_quantizers,
        "num_semantic_quantizers": config.num_semantic_quantizers,
        "codebook_size": config.codebook_size,
        "codebook_dim": config.codebook_dim,
        "vq_hidden_dimension": config.vector_quantization_hidden_dimension,
        "model_hidden_size": config.hidden_size,
        "input_waveform_shape": tensor_shape(waveform),
        "audio_codes_shape": tensor_shape(codes),
        "c0_ids_shape": tensor_shape(codes[:, 0]),
        "c0_raw_embedding_shape": tensor_shape(c0_raw),
        "c0_projected_semantic_shape": tensor_shape(c0_projected),
        "duration_seconds": waveform.shape[-1] / config.sampling_rate,
        "actual_codec_frames": codes.shape[-1],
        "measured_codec_frame_rate": codes.shape[-1]
        / (waveform.shape[-1] / config.sampling_rate),
        "c0_unique_tokens": int(torch.unique(codes[:, 0]).numel()),
        "decode": decode_reports,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
