"""Bounded, fair, rate-based scheduler for one emulated engine.

Requests have at most one pending compute step. A scheduling round completes
all currently ready steps together, costing the sum of their token/rate costs.
This represents aggregate engine capacity without pretending to model kernels.
I/O-blocked requests submit no work and do not make progress in the background.
"""

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

from kvbench.core.config import SchedulerConfig


class OverloadedError(RuntimeError):
    pass


@dataclass
class Work:
    kind: Literal["prefill", "decode"]
    tokens: int
    done: asyncio.Future[None]


class RateScheduler:
    def __init__(self, config: SchedulerConfig, full_speed: bool = False):
        self.config = config
        self.full_speed = full_speed
        self._slots = asyncio.Semaphore(config.max_active_requests)
        self._ready: deque[Work] = deque()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self.closed = False
        self.active = 0
        self.pending = 0
        self.rounds = 0
        self.compute_seconds = 0.0
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.max_batch = 0

    @asynccontextmanager
    async def admit(self) -> AsyncIterator[float]:
        if self.closed:
            raise RuntimeError("Scheduler is closed")
        if self.active + self.pending >= self.config.max_active_requests + self.config.max_pending_requests:
            raise OverloadedError("Request queue is full")
        self.pending += 1
        acquired = False
        start = time.monotonic()
        try:
            await self._slots.acquire()
            acquired = True
            self.pending -= 1
            self.active += 1
            if self.closed:
                raise RuntimeError("Scheduler is closed")
            yield time.monotonic() - start
        finally:
            if acquired:
                self.active -= 1
                self._slots.release()
            else:
                self.pending -= 1

    async def run(self, kind: Literal["prefill", "decode"], tokens: int) -> float:
        if self.closed:
            raise RuntimeError("Scheduler is closed")
        if tokens <= 0:
            return 0.0
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._ready.append(Work(kind, tokens, done))
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="kvbench-rate-scheduler")
        self._wake.set()
        start = time.monotonic()
        try:
            await done
        finally:
            if not done.done():
                done.cancel()
        return time.monotonic() - start

    async def _loop(self) -> None:
        batch: list[Work] = []
        try:
            while True:
                if not self._ready:
                    self._wake.clear()
                    await self._wake.wait()
                # Let other ready request tasks enqueue their next step.
                await asyncio.sleep(0)
                batch = [w for w in self._ready if not w.done.cancelled()]
                self._ready.clear()
                if not batch:
                    continue
                prefill = sum(w.tokens for w in batch if w.kind == "prefill")
                decode = sum(w.tokens for w in batch if w.kind == "decode")
                seconds = prefill / self.config.prefill_tokens_per_second + decode / self.config.decode_tokens_per_second
                self.rounds += 1
                self.compute_seconds += seconds
                self.prefill_tokens += prefill
                self.decode_tokens += decode
                self.max_batch = max(self.max_batch, len(batch))
                await asyncio.sleep(0 if self.full_speed else seconds)
                for work in batch:
                    if not work.done.done():
                        work.done.set_result(None)
                batch = []
        finally:
            for work in [*batch, *self._ready]:
                if not work.done.done():
                    work.done.cancel()
            self._ready.clear()

    def info(self) -> dict[str, float | int]:
        return {"active": self.active, "pending": self.pending, "rounds": self.rounds,
                "compute_seconds": self.compute_seconds, "prefill_tokens": self.prefill_tokens,
                "decode_tokens": self.decode_tokens, "max_batch": self.max_batch}

    async def close(self) -> None:
        self.closed = True
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
