"""Bounded CPU prefetch with sampler ownership at consumption, not submission."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor

from owac.training.n01_v12_sampling import validation_indices  # noqa: F401


def training_indices(length, seed, consumed=0):
    import random

    epoch, cursor = divmod(consumed, length)
    while True:
        order = list(range(length))
        random.Random(seed * 1000003 + epoch).shuffle(order)
        yield from order[cursor:]
        epoch, cursor = epoch + 1, 0


class PrefetchBatches:
    def __init__(self, dataset, seed, consumed, batch_size=1, prefetch=0):
        self.dataset, self.batch_size = dataset, batch_size
        self.indices = training_indices(len(dataset), seed, consumed)
        self.executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
        self.pending = deque()
        for _ in range(prefetch):
            self.pending.append(self._submit())

    def _load(self, indices):
        from owac.raw.rawvla_sequence import collate_sequence_samples

        samples = [self.dataset[i] for i in indices]
        return indices, samples, collate_sequence_samples(samples)

    def _submit(self):
        return self.executor.submit(
            self._load, [next(self.indices) for _ in range(self.batch_size)]
        )

    def __next__(self):
        if self.executor is None:
            return self._load([next(self.indices) for _ in range(self.batch_size)])
        item = self.pending.popleft().result()
        self.pending.append(self._submit())
        return item

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
