from __future__ import annotations

import math
import random
from typing import Iterator, Sequence


class BucketBatchSampler:
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        bucket_mult: int = 50,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is not valid for world_size {world_size}")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        n = len(self.lengths)
        self.chunk = batch_size * bucket_mult if shuffle else max(n, 1)
        n_batches = sum(math.ceil(min(self.chunk, n - s) / batch_size) for s in range(0, n, self.chunk))
        self._per_rank = math.ceil(n_batches / world_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._per_rank

    def _global_batches(self) -> list[list[int]]:
        n = len(self.lengths)
        order = list(range(n))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(order)
        batches: list[list[int]] = []
        for s in range(0, n, self.chunk):
            chunk = sorted(order[s : s + self.chunk], key=self.lengths.__getitem__)
            batches.extend(chunk[i : i + self.batch_size] for i in range(0, len(chunk), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._global_batches()
        total = self._per_rank * self.world_size
        if len(batches) < total:
            batches += (batches * self.world_size)[: total - len(batches)]
        yield from batches[self.rank :: self.world_size]