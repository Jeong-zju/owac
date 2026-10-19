"""One CPU worker, bounded lookahead; training consumes exactly the original anchor order."""

from concurrent.futures import ThreadPoolExecutor

from owac.frameworks.n02_full.sequence_v12 import training_anchor_order


class WindowPrefetch:
    def __init__(self, trainer, cursor: int, *, depth: int = 0):
        if not 0 <= depth <= 4:
            raise ValueError("prefetch depth must be in [0,4]")
        self.trainer, self.cursor, self.depth = trainer, cursor, depth
        self.executor = None
        self.pending = {}

    def _submit(self, cursor):
        plan = self.trainer.plan
        total = self.trainer.sources["train"].anchor_count
        blocks = (total + plan.batch_size - 1) // plan.batch_size
        if hasattr(self.trainer, "training_bounds"):
            start, stop = self.trainer.training_bounds(cursor)
        else:
            start = training_anchor_order(blocks, plan.seed, cursor)[0] * plan.batch_size
            stop = min(total, start + plan.batch_size)
        future = self.executor.submit(self.trainer.prepare_microbatch, start, stop)
        self.pending[cursor] = (start, stop, future)

    def __enter__(self):
        if self.depth:
            # The source owns a mutable last-episode cache: one worker is essential.
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="n02-window")
            for cursor in range(self.cursor, self.cursor + self.depth):
                self._submit(cursor)
        return self

    def get(self, cursor, start, stop):
        if not self.depth:
            return None
        expected_start, expected_stop, future = self.pending.pop(cursor)
        if (start, stop) != (expected_start, expected_stop):
            raise RuntimeError("prefetch cursor differs from the frozen training order")
        value = future.result()
        self._submit(cursor + self.depth)
        return value

    def __exit__(self, *exc):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
