"""Deterministic length-bucketed batch sampling."""

from __future__ import annotations

import bisect
import math
import random
from typing import Iterator, List, Sequence

from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler[List[int]]):
    """Group similar spectrogram lengths to reduce padding and VRAM waste."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        boundaries: Sequence[int] = (300, 500, 700, 900),
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 1234,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if not lengths or any(int(length) <= 0 for length in lengths):
            raise ValueError("lengths must be a non-empty sequence of positive integers")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if any(boundary <= 0 for boundary in boundaries):
            raise ValueError("bucket boundaries must be positive")
        if list(boundaries) != sorted(set(boundaries)):
            raise ValueError("bucket boundaries must be strictly increasing")
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.lengths = [int(length) for length in lengths]
        self.batch_size = batch_size
        self.boundaries = list(boundaries)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.buckets: List[List[int]] = [list() for _ in range(len(boundaries) + 1)]
        for index, length in enumerate(self.lengths):
            self.buckets[bisect.bisect_right(self.boundaries, length)].append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _global_batches(self) -> List[List[int]]:
        generator = random.Random(self.seed + self.epoch)
        batches: List[List[int]] = []
        for source_bucket in self.buckets:
            bucket = list(source_bucket)
            if self.shuffle:
                generator.shuffle(bucket)
            for start in range(0, len(bucket), self.batch_size):
                batch = bucket[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            generator.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._global_batches()
        if self.num_replicas > 1:
            total = math.ceil(len(batches) / self.num_replicas) * self.num_replicas
            batches.extend(
                list(batches[index % len(batches)])
                for index in range(total - len(batches))
            )
            batches = batches[self.rank:total:self.num_replicas]
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            count = sum(len(bucket) // self.batch_size for bucket in self.buckets)
        else:
            count = sum(math.ceil(len(bucket) / self.batch_size) for bucket in self.buckets)
        return math.ceil(count / self.num_replicas)
