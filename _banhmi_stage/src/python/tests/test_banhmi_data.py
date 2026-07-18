import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from banhmi_tts.data import (
    BanhmiCollator,
    BanhmiDataset,
    DataContractError,
    LengthBucketBatchSampler,
)


def write_example(root: Path, index: int, frames: int, samples: int) -> dict:
    key = f"sample-{index}"
    paths = {
        "audio_path": f"artifacts/audio/{key}.pt",
        "spectrogram_path": f"artifacts/spectrogram/{key}.pt",
        "pitch_path": f"artifacts/pitch/{key}.pt",
    }
    for relative in paths.values():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    voiced = torch.arange(frames) % 2 == 0
    log_continuous = torch.linspace(4.5, 5.0, frames)
    log_f0 = torch.where(voiced, log_continuous, torch.zeros(frames))
    f0_hz = torch.where(voiced, log_continuous.exp(), torch.zeros(frames))
    torch.save(torch.randn(1, samples), root / paths["audio_path"])
    torch.save(torch.randn(513, frames), root / paths["spectrogram_path"])
    torch.save(
        {
            "f0_hz": f0_hz,
            "log_f0": log_f0,
            "log_f0_continuous": log_continuous,
            "voiced": voiced,
        },
        root / paths["pitch_path"],
    )
    return {
        "schema_version": 2,
        "utterance_id": key,
        "phoneme_ids": [1, 10 + index, 0, 2],
        **paths,
        "sample_rate": 22050,
        "num_samples": samples,
        "num_spectrogram_frames": frames,
        "split": "train",
    }


class DataLayerTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> BanhmiDataset:
        (root / "config.json").write_text('{"schema_version":2}', encoding="utf-8")
        records = [write_example(root, 0, 8, 2048), write_example(root, 1, 5, 1280)]
        manifest = root / "splits" / "train.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        return BanhmiDataset(manifest)

    def test_dataset_loads_and_validates_schema_v2(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            dataset = self.make_dataset(Path(temporary_dir))
            sample = dataset[0]
            self.assertEqual(sample["spectrogram"].shape, (513, 8))
            self.assertEqual(len(sample["log_f0_continuous"]), 8)
            self.assertEqual(dataset.spectrogram_lengths, [8, 5])

    def test_collator_shapes_masks_and_padding(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            dataset = self.make_dataset(Path(temporary_dir))
            batch = BanhmiCollator()([dataset[1], dataset[0]])
            self.assertEqual(batch["spectrogram"].shape, (2, 513, 8))
            self.assertEqual(batch["audio"].shape, (2, 1, 2048))
            self.assertEqual(batch["spectrogram_mask"].shape, (2, 1, 8))
            self.assertEqual(batch["spectrogram_mask"].sum().item(), 13)
            self.assertFalse(batch["voiced"][1, 5:].any())
            self.assertTrue((batch["log_f0_continuous"][1, 5:] == 0).all())

    def test_dataloader_uses_bucket_batch_sampler(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            dataset = self.make_dataset(Path(temporary_dir))
            sampler = LengthBucketBatchSampler(
                dataset.spectrogram_lengths, batch_size=2, boundaries=(6,), shuffle=False
            )
            loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=BanhmiCollator())
            batches = list(loader)
            self.assertEqual(len(batches), 2)
            self.assertEqual(sum(len(batch["utterance_ids"]) for batch in batches), 2)

    def test_bucket_sampler_is_deterministic_per_epoch(self):
        sampler = LengthBucketBatchSampler(
            [100, 120, 320, 350, 710, 750], batch_size=2, seed=7
        )
        first = list(iter(sampler))
        second = list(iter(sampler))
        self.assertEqual(first, second)
        sampler.set_epoch(1)
        third = list(iter(sampler))
        self.assertNotEqual(first, third)
        self.assertEqual(sorted(index for batch in third for index in batch), list(range(6)))

    def test_distributed_bucket_sampler_has_equal_rank_lengths(self):
        lengths = [100, 120, 320, 350, 710]
        rank_zero = LengthBucketBatchSampler(
            lengths, batch_size=2, seed=7, num_replicas=2, rank=0
        )
        rank_one = LengthBucketBatchSampler(
            lengths, batch_size=2, seed=7, num_replicas=2, rank=1
        )
        zero_batches = list(rank_zero)
        one_batches = list(rank_one)
        self.assertEqual(len(zero_batches), len(one_batches))
        self.assertEqual(len(rank_zero), len(zero_batches))
        covered = {index for batch in zero_batches + one_batches for index in batch}
        self.assertEqual(covered, set(range(len(lengths))))

    def test_dataset_rejects_wrong_schema(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset = self.make_dataset(root)
            manifest = dataset.manifest_path
            record = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
            record["schema_version"] = 1
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaises(DataContractError):
                BanhmiDataset(manifest)


if __name__ == "__main__":
    unittest.main()
