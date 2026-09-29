"""Real LMCache disk acceptance tests, using <=32 MiB aggregate CPU pools.

The oracle is independently calculated chunk geometry and actual file I/O,
not the emulator's reported submitted-token counters alone.
"""

import asyncio
import uuid

import pytest

pytest.importorskip("lmcache")

from lmcache.v1.cache_engine import LMCacheEngineBuilder  # noqa: E402

from kvbench.core.config import KVBenchConfig  # noqa: E402
from kvbench.kv.factory import create_kv_stack  # noqa: E402
from kvbench.servers.openai_compat import ChatCompletionRequest, ErrorResponse  # noqa: E402
from kvbench.servers.rate_server import RateServer  # noqa: E402
from kvbench.trace.analyze import load_events  # noqa: E402

pytestmark = pytest.mark.integration


def make_config(tmp_path, tp=1, pp=1, dtype="bf16", chunk=16, extra=""):
    catalog = tmp_path / "profiles.yaml"
    catalog.write_text(f"""version: 1
models:
  storage-test:
    name: Small storage test
    layers: 4
    hidden: 128
    kv_heads: 2
    num_heads: 2
    intermediate: 256
    kv_dtype: {dtype}
""")
    disk = tmp_path / "disk"
    disk.mkdir(exist_ok=True)
    lmcache = tmp_path / "lmcache.yaml"
    lmcache.write_text(f"""chunk_size: {chunk}
local_cpu: true
max_local_cpu_size: 0.0078125
local_disk: "file://{disk}/"
max_local_disk_size: 0.03125
{extra}
""")
    return KVBenchConfig.model_validate({
        "instance_id": f"v1-test-{uuid.uuid4().hex[:8]}",
        "profiles_file": catalog,
        "server": {"model_profile": "storage-test"},
        "gpu": {"tp_size": tp}, "timing": {"pp_size": pp, "full_speed": True},
        "scheduler": {"prefill_chunk_tokens": chunk},
        "kv": {"lmcache_config_file": lmcache, "trace_file": tmp_path / "trace.jsonl",
               "max_cpu_memory_gb": 0.0625, "max_chunk_mb": 1},
    })


def req(content, output=1, messages=None):
    return ChatCompletionRequest(model="storage-test", max_completion_tokens=output,
                                 messages=messages or [{"role": "user", "content": content}])


async def settle(stack):
    """Wait for LMCache's disk writer queue, not an arbitrary fixed sleep."""
    async with asyncio.timeout(10):
        while True:
            pending = any(backend.disk_worker.put_tasks
                          for engine in stack._engines
                          for backend in engine.storage_manager.storage_backends.values()
                          if hasattr(backend, "disk_worker"))
            if not pending:
                break
            await asyncio.sleep(0.01)
    stack.trace_recorder.flush()


def io_totals(config):
    events = load_events(config.kv.trace_file)
    return {op: sum(e["size"] for e in events if e["event"] == "io" and e["op"] == op)
            for op in ("read", "write")}


@pytest.mark.asyncio
async def test_workload_changes_reach_physical_storage(tmp_path):
    cfg = make_config(tmp_path)
    stack = create_kv_stack(cfg)
    await stack.start()
    server = RateServer(cfg, stack)
    # 2 planes * 4 layers * 2 KV heads * 64 dims * 2-byte KV dtype
    bytes_per_token = 2048
    try:
        previous = 0
        sizes = []
        for content, output in [("A" * 128, 1), ("B" * 128, 65), ("C" * 512, 1)]:
            response = await server.chat_completions(req(content, output))
            assert not isinstance(response, ErrorResponse)
            await settle(stack)
            current = io_totals(cfg)["write"]
            expected = (response.usage.prompt_tokens + output - 1) // 16 * 16 * bytes_per_token
            assert current - previous == expected
            sizes.append(current - previous)
            previous = current
        assert sizes[1] > sizes[0]  # More output tokens create more KV writes.
        assert sizes[2] > sizes[0]  # More input tokens create more KV writes.

        # Reuse an actual response in the next turn: generated KV is reusable.
        first = await server.chat_completions(req("D" * 128, 65))
        await settle(stack)
        messages = [{"role": "user", "content": "D" * 128},
                    first.choices[0].message.model_dump(), {"role": "user", "content": "next"}]
        before = stack.stats.hit_tokens
        follow = await server.chat_completions(req("", 33, messages))
        assert not isinstance(follow, ErrorResponse)
        assert stack.stats.hit_tokens - before > first.usage.prompt_tokens
        await settle(stack)

        # Force the next hit through disk by evicting just the CPU tier.
        for engine in stack._engines:
            await asyncio.to_thread(engine.clear, locations=["LocalCPUBackend"])
        before = io_totals(cfg)
        result = await server.chat_completions(req("A" * 128, 1))
        assert not isinstance(result, ErrorResponse)
        await settle(stack)
        after = io_totals(cfg)
        expected_read = result.usage.prompt_tokens // 16 * 16 * bytes_per_token
        assert after["read"] - before["read"] == expected_read
        assert after["write"] == before["write"]
        assert not any(engine.lookup_pins for engine in stack._engines)
    finally:
        await server.stop()
        await stack.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("tp,pp,dtype,chunk", [(1, 1, "bf16", 16), (2, 2, "bf16", 16),
                                             (4, 1, "bf16", 16), (2, 1, "fp8", 32)])
async def test_rank_files_have_correct_sizes_and_shutdown(tmp_path, tp, pp, dtype, chunk):
    cfg = make_config(tmp_path, tp, pp, dtype, chunk)
    stack = create_kv_stack(cfg)
    await stack.start()
    owned = list(stack._instance_ids)
    try:
        await stack.store(list(range(chunk * 2)))
        await settle(stack)
        events = [e for e in load_events(cfg.kv.trace_file) if e["event"] == "io" and e["op"] == "write"]
        assert len(events) == 2 * tp * pp
        bytes_per_element = 1 if dtype == "fp8" else 2
        per_rank_chunk = 2 * (4 // pp) * max(1, 2 // tp) * 64 * bytes_per_element * chunk
        assert all(e["size"] == per_rank_chunk for e in events)
        files = list((tmp_path / "disk").glob("*.pt"))
        assert len(files) == 2 * tp * pp
        assert all(f.stat().st_size == per_rank_chunk for f in files)
        assert len({f.read_bytes() for f in files}) == len(files)  # no cross-rank/chunk dedup
        assert {e["rank"] for e in events} == set(range(tp * pp))
        assert stack.capacity_info()["local_cpu_capacity_bytes"] == 8 * 1024**2 * tp * pp
        assert stack.stats.submitted_store_bytes == per_rank_chunk * 2 * tp * pp
        assert await stack.lookup_pinned(list(range(chunk * 2)), "test") == chunk * 2
        assert await stack.retrieve(list(range(chunk * 2))) == chunk * 2
        assert stack.stats.retrieved_bytes == per_rank_chunk * 2 * tp * pp
        await stack.unpin("test")
        assert not any(e.lookup_pins for e in stack._engines)
    finally:
        await stack.close()
    assert all(LMCacheEngineBuilder.get(identifier) is None for identifier in owned)
    assert not stack._engines and not stack._connectors
    await stack.close()  # Idempotent.


@pytest.mark.asyncio
async def test_partial_startup_cleans_created_ranks(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, tp=2)
    original = LMCacheEngineBuilder.get_or_create

    def fail_second(instance_id, *args, **kwargs):
        if instance_id.endswith("-r1"):
            raise RuntimeError("injected rank startup failure")
        return original(instance_id, *args, **kwargs)

    monkeypatch.setattr(LMCacheEngineBuilder, "get_or_create", fail_second)
    stack = create_kv_stack(cfg)
    with pytest.raises(RuntimeError, match="injected"):
        await stack.start()
    assert LMCacheEngineBuilder.get(cfg.instance_id + "-r0") is None
    assert not stack._engines and stack.trace_recorder is None


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", ["enable_async_loading: true", "use_layerwise: true"])
async def test_unsupported_lmcache_modes_fail_before_allocation(tmp_path, extra):
    cfg = make_config(tmp_path, extra=extra)
    stack = create_kv_stack(cfg)
    with pytest.raises(ValueError, match="unsupported modes"):
        await stack.start()
    assert not stack._instance_ids


@pytest.mark.asyncio
async def test_host_memory_guard_fails_before_allocation(tmp_path):
    cfg = make_config(tmp_path)
    stack = create_kv_stack(cfg)
    stack.max_cpu_memory_gb = 0.001
    with pytest.raises(ValueError, match="CPU pools request"):
        await stack.start()
    assert not stack._instance_ids


@pytest.mark.asyncio
async def test_prefix_larger_than_cpu_tier_can_be_retrieved(tmp_path):
    cfg = make_config(tmp_path)
    cfg.kv.lmcache_config_file.write_text(cfg.kv.lmcache_config_file.read_text().replace(
        "max_local_cpu_size: 0.0078125", "max_local_cpu_size: 0.000244140625"))
    stack = create_kv_stack(cfg)
    await stack.start()
    tokens = list(range(16 * 16))  # 512 KiB, twice the CPU pool capacity.
    try:
        for end in range(16, len(tokens) + 1, 16):
            await stack.store(tokens[:end], skip_leading=end - 16)
            await settle(stack)
        assert await stack.lookup_pinned(tokens, "large") == len(tokens)
        assert await stack.retrieve(tokens) == len(tokens)
        await stack.unpin("large")
    finally:
        await stack.close()


def test_payload_identity_is_stable_but_not_shared_between_models():
    from types import SimpleNamespace

    import torch

    from kvbench.kv.mock_gpu import MockGPUConnector

    def payload(namespace):
        connector = MockGPUConnector(1, 64, torch.bfloat16, namespace=namespace)
        tensor = torch.empty((2, 1, 16, 64), dtype=torch.bfloat16)
        connector.from_gpu(SimpleNamespace(tensor=tensor), 0, 16, payload_tokens=torch.arange(16))
        return tensor.view(torch.uint8).numpy().tobytes()

    assert payload("model-a") == payload("model-a")
    assert payload("model-a") != payload("model-b")
