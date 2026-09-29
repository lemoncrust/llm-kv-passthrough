"""Cancellation of blocking manager calls must not leave late pins."""

import asyncio
import threading

import pytest

from kvbench.kv.lmcache_stack import LMCacheStack


@pytest.mark.asyncio
async def test_cancel_waits_for_lookup_thread_then_unpins():
    entered = threading.Event()
    release = threading.Event()

    class Engine:
        pinned = False

        def lookup(self, tokens, **kwargs):  # noqa: ARG002 - test adapter
            entered.set()
            release.wait(2)
            self.pinned = True
            return 256

        def lookup_unpin(self, lookup_id):  # noqa: ARG002 - test adapter
            self.pinned = False

    engine = Engine()
    stack = LMCacheStack("llama-3.1-8b")
    stack._engine = engine
    stack._engines = [engine]
    task = asyncio.create_task(stack.lookup_pinned(list(range(256)), "cancelled"))
    while not entered.is_set():
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()  # Cleanup cannot outrun the still-running thread.
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not engine.pinned
