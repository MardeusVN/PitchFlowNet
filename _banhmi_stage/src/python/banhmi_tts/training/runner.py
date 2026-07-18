"""EdgeTTS-parity lifecycle around the explicit BanhmiTTS training step."""

from __future__ import annotations

import json
import wave
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from banhmi_tts.data import BanhmiCollator, BanhmiDataset, LengthBucketBatchSampler

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ModelConfig, TrainingConfig
from .distributed import DistributedContext
from .trainer import BanhmiTrainer


def _log_scalars(writer, prefix: str, values: dict, global_step: int) -> None:
    for name, value in values.items():
        if name != "global_step" and isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{name}", float(value), global_step)


def _write_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    samples = (
        audio.detach().float().flatten().clamp(-1.0, 1.0).mul(32767.0)
        .round().to(torch.int16).cpu().numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


def _training_config(args, dataset_config: dict) -> TrainingConfig:
    pitch = dataset_config["feature_statistics"]["pitch"]
    model = ModelConfig.default(
        num_symbols=int(dataset_config["frontend"]["num_symbols"]),
        pitch_mean=float(pitch["log_f0_mean"]),
        pitch_standard_deviation=float(pitch["log_f0_std"]),
    )
    model = replace(
        model,
        predictor=replace(model.predictor, duration_type=args.duration_predictor),
        generator=replace(
            model.generator,
            activation=args.generator_activation,
            use_prosody_conditioning=not args.disable_prosody_conditioning,
        ),
    )
    config = TrainingConfig(model=model)
    config = replace(
        config,
        optimization=replace(
            config.optimization,
            c_kl=args.c_kl,
            generator_gradient_clip=args.generator_gradient_clip,
            discriminator_gradient_clip=args.discriminator_gradient_clip,
        ),
    )
    if args.legacy_discriminator:
        config = replace(
            config,
            discriminator=replace(
                config.discriminator,
                include_scale_discriminator=False,
                use_magnitude_spectrogram=False,
            ),
        )
    elif args.edge_discriminator:
        config = replace(
            config,
            discriminator=replace(
                config.discriminator,
                include_scale_discriminator=True,
                use_magnitude_spectrogram=True,
            ),
        )
    return config


def _progress_extra(
    epoch: int, batch_in_epoch: int, best_validation_mel: float
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "best_validation_mel": float(best_validation_mel),
    }


def _run_reconstruction_diagnostics(
    trainer: BanhmiTrainer,
    batch: dict,
    output_dir: Path,
    global_step: int,
    sample_rate: int,
    writer,
) -> None:
    reconstruction_dir = output_dir / "posterior_mean_reconstructions"
    losses = {}
    for posterior_label, sample_posterior in (("mean", False), ("sampled", True)):
        for length_label, full_utterance in (("segment", False), ("full", True)):
            generator = torch.Generator(device=trainer.device).manual_seed(20_000)
            result = trainer.reconstruct_posterior_mean(
                batch,
                full_utterance=full_utterance,
                sample_posterior=sample_posterior,
                random_generator=generator,
            )
            label = f"{length_label}-{posterior_label}"
            losses[label] = trainer.reconstruction_spectral_loss(result)
            _write_wav(
                reconstruction_dir / f"step-{global_step:08d}-{label}-generated.wav",
                result.generated_audio[0], sample_rate,
            )
            _write_wav(
                reconstruction_dir / f"step-{global_step:08d}-{label}-target.wav",
                result.target_audio[0], sample_rate,
            )
    print(json.dumps({"reconstruction_step": global_step, **losses}))
    if writer is not None:
        _log_scalars(writer, "diagnostic", losses, global_step)


def _validate(
    trainer: BanhmiTrainer,
    validation_loader: DataLoader,
    validation_batches: int,
    global_step: int,
) -> tuple[dict[str, float], Any, dict | None]:
    rows = []
    first_result = None
    first_batch = None
    for batch_index, batch in enumerate(validation_loader):
        if batch_index >= validation_batches:
            break
        result = trainer.validate_step(
            batch,
            global_step,
            random_generator=torch.Generator(device=trainer.device).manual_seed(
                10_000 + batch_index
            ),
        )
        rows.append(asdict(result.metrics))
        if first_result is None:
            first_result = result
            first_batch = batch
    if not rows:
        raise RuntimeError("validation loader produced no batches")
    averaged = {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in rows[0]
        if key != "global_step"
    }
    return averaged, first_result, first_batch


def _log_validation_audio(
    trainer: BanhmiTrainer,
    result,
    batch: dict,
    output_dir: Path,
    global_step: int,
    sample_rate: int,
    writer,
) -> None:
    sample_dir = output_dir / "validation_reconstructions"
    _write_wav(
        sample_dir / f"step-{global_step:08d}-generated.wav",
        result.generated_audio[0], sample_rate,
    )
    _write_wav(
        sample_dir / f"step-{global_step:08d}-target.wav",
        result.target_audio[0], sample_rate,
    )
    mean_result = trainer.reconstruct_posterior_mean(
        batch, full_utterance=True, sample_posterior=False
    )
    sampled_result = trainer.reconstruct_posterior_mean(
        batch,
        full_utterance=True,
        sample_posterior=True,
        random_generator=torch.Generator(device=trainer.device).manual_seed(20_000),
    )
    _write_wav(
        sample_dir / f"step-{global_step:08d}-posterior-mean-full.wav",
        mean_result.generated_audio[0], sample_rate,
    )
    _write_wav(
        sample_dir / f"step-{global_step:08d}-posterior-sampled-full.wav",
        sampled_result.generated_audio[0], sample_rate,
    )
    if writer is not None:
        for tag, audio in (
            ("audio/validation_generated", result.generated_audio[0]),
            ("audio/validation_target", result.target_audio[0]),
            ("audio/posterior_mean_full", mean_result.generated_audio[0]),
            ("audio/posterior_sampled_full", sampled_result.generated_audio[0]),
        ):
            writer.add_audio(
                tag, audio.float().clamp(-1.0, 1.0), global_step,
                sample_rate=sample_rate,
            )


def run_training(args, context: DistributedContext) -> None:
    if context.enabled and args.reconstruction_only:
        raise ValueError("reconstruction-only diagnostics must run without torchrun")
    torch.set_float32_matmul_precision("high")
    if context.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed + context.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + context.rank)

    dataset_dir = args.dataset_dir.resolve()
    dataset_config = json.loads(
        (dataset_dir / "config.json").read_text(encoding="utf-8")
    )
    config = _training_config(args, dataset_config)
    trainer = BanhmiTrainer(config, context.device)
    train_dataset = BanhmiDataset(dataset_dir / "splits" / "train.jsonl")
    sampler = LengthBucketBatchSampler(
        train_dataset.spectrogram_lengths,
        batch_size=args.batch_size,
        seed=args.seed,
        num_replicas=context.world_size,
        rank=context.rank,
    )
    loader_generator = torch.Generator().manual_seed(args.seed + context.rank)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=BanhmiCollator(),
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=context.device.type == "cuda",
        generator=loader_generator,
    )
    validation_loader = None
    if context.is_main:
        validation_dataset = BanhmiDataset(
            dataset_dir / "splits" / "validation.jsonl"
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=BanhmiCollator(),
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=context.device.type == "cuda",
            generator=torch.Generator().manual_seed(args.seed),
        )

    output_dir = args.output_dir.resolve()
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "training_config.json").write_text(
            json.dumps(asdict(config), indent=2), encoding="utf-8"
        )
    context.barrier()
    writer = None
    if context.is_main and not args.disable_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TensorBoard is enabled but unavailable; install it or pass "
                "--disable-tensorboard"
            ) from error
        tensorboard_dir = (
            args.tensorboard_log_dir.resolve()
            if args.tensorboard_log_dir is not None
            else output_dir / "tensorboard"
        )
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        writer.add_text("config/training", json.dumps(asdict(config), indent=2))

    global_step = 0
    epoch = 0
    batch_in_epoch = 0
    best_validation_mel = float("inf")
    if args.resume is not None:
        metadata = load_checkpoint(args.resume, trainer)
        global_step = metadata["global_step"]
        extra = metadata["extra"]
        epoch = int(extra.get("epoch", 0))
        batch_in_epoch = int(extra.get("batch_in_epoch", 0))
        best_validation_mel = float(extra.get("best_validation_mel", float("inf")))
        if not 0 <= batch_in_epoch < len(train_loader):
            raise ValueError("checkpoint batch_in_epoch is outside the current epoch")

    sampler.set_epoch(epoch)
    fixed_batch = next(iter(train_loader)) if args.overfit_one_batch else None
    if args.reconstruction_only:
        batch = fixed_batch if fixed_batch is not None else next(iter(train_loader))
        _run_reconstruction_diagnostics(
            trainer, batch, output_dir, global_step,
            int(dataset_config["preprocessing"]["audio"]["sample_rate"]), writer,
        )
        if writer is not None:
            writer.close()
        return

    sample_rate = int(dataset_config["preprocessing"]["audio"]["sample_rate"])
    while global_step < args.steps:
        sampler.set_epoch(epoch)
        if fixed_batch is not None:
            batches = [(0, fixed_batch)]
        else:
            iterator = iter(train_loader)
            for _ in range(batch_in_epoch):
                try:
                    next(iterator)
                except StopIteration as error:
                    raise RuntimeError("checkpoint batch offset exceeds epoch") from error
            batches = enumerate(iterator, start=batch_in_epoch)

        processed_batch = False
        for batch_index, batch in batches:
            processed_batch = True
            next_global_step = global_step + 1
            collect_metrics = next_global_step == 1 or next_global_step % 10 == 0
            local_metrics = trainer.train_step(
                batch,
                global_step,
                collect_metrics=collect_metrics,
            )
            global_step += 1
            metrics = (
                context.average_metrics(local_metrics)
                if local_metrics is not None
                else None
            )
            if fixed_batch is None:
                next_batch = batch_index + 1
                if next_batch >= len(train_loader):
                    trainer.step_learning_rate_schedulers()
                    epoch += 1
                    batch_in_epoch = 0
                else:
                    batch_in_epoch = next_batch

            if context.is_main and metrics is not None:
                print(json.dumps(asdict(metrics)))
            if writer is not None and metrics is not None:
                _log_scalars(writer, "train", asdict(metrics), global_step)
                writer.add_scalar(
                    "learning_rate/generator",
                    trainer.generator_optimizer.param_groups[0]["lr"], global_step,
                )
                writer.add_scalar(
                    "learning_rate/discriminator",
                    trainer.discriminator_optimizer.param_groups[0]["lr"], global_step,
                )

            extra = _progress_extra(epoch, batch_in_epoch, best_validation_mel)
            if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                save_checkpoint(
                    output_dir / f"checkpoint-{global_step:08d}.pt",
                    trainer, global_step, extra=extra,
                )

            if args.validation_every > 0 and global_step % args.validation_every == 0:
                improved = False
                averaged = None
                first_result = None
                first_batch = None
                if context.is_main:
                    averaged, first_result, first_batch = _validate(
                        trainer, validation_loader, args.validation_batches, global_step
                    )
                    improved = averaged["mel_loss"] < best_validation_mel
                    if improved:
                        best_validation_mel = averaged["mel_loss"]
                    print(json.dumps({"validation_step": global_step, **averaged}))
                    if writer is not None:
                        _log_scalars(writer, "validation", averaged, global_step)
                    if (
                        first_result is not None
                        and args.sample_every > 0
                        and global_step % args.sample_every == 0
                    ):
                        _log_validation_audio(
                            trainer, first_result, first_batch, output_dir,
                            global_step, sample_rate, writer,
                        )
                best_validation_mel = context.broadcast_float(best_validation_mel)
                improved = context.broadcast_bool(improved)
                if improved:
                    save_checkpoint(
                        output_dir / "checkpoint-best-mel.pt",
                        trainer,
                        global_step,
                        extra=_progress_extra(
                            epoch, batch_in_epoch, best_validation_mel
                        ),
                    )

            if global_step >= args.steps or fixed_batch is not None:
                break
        if not processed_batch:
            raise RuntimeError("training loader produced no batches")

    save_checkpoint(
        output_dir / "checkpoint-final.pt",
        trainer,
        global_step,
        extra=_progress_extra(epoch, batch_in_epoch, best_validation_mel),
    )
    if writer is not None:
        writer.flush()
        writer.close()
