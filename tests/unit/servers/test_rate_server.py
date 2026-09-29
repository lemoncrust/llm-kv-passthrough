"""V1 acceptance: workload sensitivity, causal ordering, bounds, cleanup."""

import asyncio
import time

import pytest

from kvbench.core.config import KVBenchConfig, SchedulerConfig, TimingConfig
from kvbench.servers.openai_compat import ChatCompletionRequest, ErrorResponse
from kvbench.servers.rate_scheduler import RateScheduler
from kvbench.servers.rate_server import RateServer
from tests.fakes import FakeKVStack


def config(**scheduler):
    return KVBenchConfig(timing=TimingConfig(full_speed=True), scheduler=SchedulerConfig(**scheduler))


def request(text="prefix " * 32, output=20, messages=None):
    return ChatCompletionRequest(model="llama-3.1-8b", max_tokens=output,
                                 messages=messages or [{"role": "user", "content": text}])


class RecordingKV(FakeKVStack):
    def __init__(self):
        super().__init__(8)
        self.pins = set()
        self.operations = []

    async def lookup_pinned(self, tokens, lookup_id):
        self.pins.add(lookup_id)
        return await self.lookup(tokens)

    async def unpin(self, lookup_id):
        self.pins.discard(lookup_id)

    async def store(self, tokens, skip_leading=0):
        self.operations.append(("store", len(tokens), skip_leading))
        await super().store(tokens, skip_leading)

    async def retrieve(self, tokens, skip_leading=0):  # noqa: ARG002 - test adapter
        self.operations.append(("retrieve", len(tokens), skip_leading))
        return await super().retrieve(tokens, skip_leading)


@pytest.mark.asyncio
@pytest.mark.parametrize("input_chars,output", [(32, 1), (32, 40), (320, 40), (3200, 80)])
async def test_exact_cold_chunk_growth(input_chars, output):
    kv = RecordingKV()
    s = RateServer(config(prefill_chunk_tokens=8), kv)
    try:
        result = await s.chat_completions(request("a" * input_chars, output))
        assert not isinstance(result, ErrorResponse)
        expected = (result.usage.prompt_tokens + output - 1) // 8 * 8
        assert kv.stats.stored_tokens == expected
        assert all(end % 8 == 0 and skip % 8 == 0 for _, end, skip in kv.operations)
        assert sum(end - skip for _, end, skip in kv.operations) == expected
        assert not kv.pins
    finally:
        await s.stop()


@pytest.mark.asyncio
async def test_output_length_and_turn_count_change_storage_work():
    async def session(turns, output):
        kv = RecordingKV()
        server = RateServer(config(prefill_chunk_tokens=8), kv)
        messages = [{"role": "user", "content": "a" * 128}]
        previous_end = 0
        expected_new = 0
        expected_reads = 0
        try:
            for turn in range(turns):
                result = await server.chat_completions(request(output=output, messages=messages))
                end = (result.usage.prompt_tokens + output - 1) // 8 * 8
                expected_new += end - previous_end
                expected_reads += previous_end
                assert kv.stats.hit_tokens == expected_reads
                previous_end = end
                messages.extend([result.choices[0].message.model_dump(),
                                 {"role": "user", "content": f"next turn {turn}"}])
            assert kv.stats.stored_tokens == expected_new
            assert kv.stats.retrieved_tokens == expected_reads
            return kv.stats.stored_tokens, kv.stats.retrieved_tokens
        finally:
            await server.stop()

    short = await session(2, 16)
    longer = await session(2, 64)
    more_turns = await session(5, 16)
    assert longer[0] > short[0] and longer[1] > short[1]
    assert more_turns[0] > short[0] and more_turns[1] > short[1]


@pytest.mark.asyncio
async def test_stores_happen_during_prefill_not_after_it():
    kv = RecordingKV()
    s = RateServer(config(prefill_chunk_tokens=8), kv)
    steps = []
    original = kv.store

    async def store(tokens, skip_leading=0):
        steps.append(s.scheduler.prefill_tokens)
        await original(tokens, skip_leading)

    kv.store = store
    try:
        response = await s.chat_completions(request(output=1))
        assert len(steps) > 1
        assert steps[0] == 8
        assert steps[0] < response.usage.prompt_tokens
    finally:
        await s.stop()


class SlowRead(RecordingKV):
    def __init__(self, short=False):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.short = short

    async def lookup(self, tokens):
        return len(tokens) // 8 * 8

    async def retrieve(self, tokens, skip_leading=0):  # noqa: ARG002 - test adapter
        self.started.set()
        await self.release.wait()
        return 0 if self.short else len(tokens)


@pytest.mark.asyncio
async def test_slow_storage_blocks_compute_and_does_not_burst_output():
    kv = SlowRead()
    cfg = KVBenchConfig(scheduler=SchedulerConfig(decode_tokens_per_second=50))
    s = RateServer(cfg, kv)

    async def consume():
        stamps = []
        async for chunk in s.chat_completions_stream(request(output=4)):
            if '"content":"' in chunk and '"content":""' not in chunk:
                stamps.append(time.monotonic())
        return stamps

    task = asyncio.create_task(consume())
    try:
        await kv.started.wait()
        await asyncio.sleep(0.08)
        assert s.scheduler.rounds == 0
        kv.release.set()
        times = await task
        assert len(times) == 4
        assert times[-1] - times[0] >= 0.055
    finally:
        await s.stop()


@pytest.mark.asyncio
async def test_short_retrieval_is_an_error_not_a_hit():
    kv = SlowRead(short=True)
    kv.release.set()
    s = RateServer(config(), kv)
    try:
        response = await s.chat_completions(request())
        assert isinstance(response, ErrorResponse)
        assert "incomplete" in response.error["message"]
        assert s.scheduler.rounds == 0
        assert s.stats.requests_failed == 1
        assert not kv.pins
    finally:
        await s.stop()


@pytest.mark.asyncio
async def test_cancel_during_lookup_releases_pins_and_admission():
    kv = RecordingKV()
    started = asyncio.Event()

    async def lookup(tokens, lookup_id):  # noqa: ARG001 - test adapter
        kv.pins.add(lookup_id)
        started.set()
        await asyncio.Event().wait()

    kv.lookup_pinned = lookup
    s = RateServer(config(), kv)
    task = asyncio.create_task(s.chat_completions(request()))
    await started.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert not kv.pins
    assert s.scheduler.active == s.scheduler.pending == 0
    assert s.stats.requests_cancelled == 1
    await s.stop()


@pytest.mark.asyncio
async def test_queue_bound_and_shutdown_while_reading():
    kv = SlowRead()
    s = RateServer(config(max_active_requests=1, max_pending_requests=0), kv)
    task = asyncio.create_task(s.chat_completions(request()))
    await kv.started.wait()
    other = await s.chat_completions(request())
    assert other.error["code"] == "503"
    await s.stop()
    assert task.done()
    assert not kv.pins
    assert s.scheduler.active == s.scheduler.pending == 0


@pytest.mark.asyncio
async def test_rates_are_aggregate_not_multiplied_by_concurrency():
    s = RateScheduler(SchedulerConfig(prefill_tokens_per_second=200, decode_tokens_per_second=100))
    try:
        start = time.monotonic()
        await asyncio.gather(s.run("prefill", 20), s.run("decode", 10))
        assert time.monotonic() - start >= 0.19
        assert s.compute_seconds == pytest.approx(0.2)
        assert s.max_batch == 2
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_request_timeout_runs_cleanup():
    kv = SlowRead()
    s = RateServer(config(request_timeout_s=0.02), kv)
    response = await s.chat_completions(request())
    assert response.error["code"] == "504"
    assert not kv.pins
    assert s.scheduler.active == 0
    await s.stop()


@pytest.mark.asyncio
async def test_stream_disconnect_closes_the_live_request():
    kv = RecordingKV()
    s = RateServer(config(), kv)
    stream = s.chat_completions_stream(request(output=100))
    for _ in range(5):
        await anext(stream)
    await stream.aclose()
    assert not s._requests
    assert not kv.pins
    assert s.scheduler.active == 0
    assert s.stats.requests_cancelled == 1
    await s.stop()


@pytest.mark.asyncio
async def test_concurrent_distinct_sessions_preserve_byte_work():
    kv = RecordingKV()
    s = RateServer(config(max_active_requests=3), kv)
    try:
        responses = await asyncio.gather(*(s.chat_completions(request(chr(65 + n) * 128, 33)) for n in range(3)))
        expected = sum((r.usage.prompt_tokens + 32) // 8 * 8 for r in responses)
        assert kv.stats.stored_tokens == expected
        assert s.scheduler.max_batch == 3
        assert s.scheduler.active == 0
        assert s.stats.requests_success == 3
    finally:
        await s.stop()


@pytest.mark.asyncio
async def test_classic_prefix_leases_do_not_overlap_under_concurrency():
    class SingleLease(RecordingKV):
        async def lookup_pinned(self, tokens, lookup_id):
            assert not self.pins, "Concurrent prefix leases can exhaust pinned staging memory"
            hit = await super().lookup_pinned(tokens, lookup_id)
            await asyncio.sleep(0.01)
            return hit

    kv = SingleLease()
    s = RateServer(config(max_active_requests=3), kv)
    try:
        results = await asyncio.gather(*(s.chat_completions(request(output=4)) for _ in range(3)))
        assert all(not isinstance(result, ErrorResponse) for result in results)
        assert not kv.pins
    finally:
        await s.stop()


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_abort_unpin_cleanup():
    entered = asyncio.Event()
    release = asyncio.Event()

    class GatedUnpin(RecordingKV):
        async def unpin(self, lookup_id):
            entered.set()
            await release.wait()
            await super().unpin(lookup_id)

    kv = GatedUnpin()
    s = RateServer(config(), kv)
    task = asyncio.create_task(s.chat_completions(request()))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not kv.pins
    assert s.scheduler.active == 0
    await s.stop()
