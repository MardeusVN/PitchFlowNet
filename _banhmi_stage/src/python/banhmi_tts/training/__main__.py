"""Command-line entry point for single-process and torchrun/DDP training."""

from __future__ import annotations

import argparse
from pathlib import Path

from .distributed import DistributedContext
from .runner import run_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m banhmi_tts.training")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    # Per-GPU value. Batch 12 completed the DDP peak smoke test on each 4070 Ti;
    # batch 16 repeatedly exhausted the practical VRAM/driver headroom.
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--validation-every", type=int, default=1000)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument(
        "--tensorboard-log-dir",
        type=Path,
        help="TensorBoard event directory (default: OUTPUT_DIR/tensorboard)",
    )
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--reconstruction-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--c-kl", type=float, default=1.0,
        help="KL loss coefficient (EdgeTTS/VITS default: 1.0)",
    )
    parser.add_argument(
        "--generator-gradient-clip",
        type=float,
        default=None,
        help="Generator maximum gradient norm (default: disabled, matching Config H)",
    )
    parser.add_argument(
        "--discriminator-gradient-clip",
        type=float,
        default=None,
        help="Discriminator maximum gradient norm (default: disabled, matching Config H)",
    )
    parser.add_argument(
        "--generator-activation",
        choices=("filtered_snake_beta", "snake_beta"),
        default="snake_beta",
    )
    parser.add_argument("--disable-prosody-conditioning", action="store_true")
    parser.add_argument(
        "--duration-predictor",
        choices=("stochastic", "deterministic"),
        default="stochastic",
    )
    parser.add_argument("--edge-discriminator", action="store_true")
    parser.add_argument(
        "--legacy-discriminator",
        action="store_true",
        help="Ablation: disable EdgeTTS scale discriminator and use complex MRD",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.validation_batches <= 0:
        raise ValueError("steps, batch-size, and validation-batches must be positive")
    if min(args.checkpoint_every, args.validation_every, args.sample_every) < 0:
        raise ValueError("checkpoint/validation/sample intervals cannot be negative")
    if args.c_kl < 0.0:
        raise ValueError("c-kl cannot be negative")
    gradient_clips = (
        args.generator_gradient_clip,
        args.discriminator_gradient_clip,
    )
    if any(value is not None and value <= 0.0 for value in gradient_clips):
        raise ValueError("gradient clipping thresholds must be positive when set")
    context = DistributedContext.initialize(force_cpu=args.cpu)
    try:
        run_training(args, context)
    finally:
        context.close()


if __name__ == "__main__":
    main()
