"""Unit tests for kvbench.servers.combined module."""

from __future__ import annotations

import pytest

from kvbench.core.config import KVBenchConfig
from kvbench.servers.combined import CombinedServer, CombinedStats
from kvbench.servers.openai_compat import ChatCompletionRequest, ChatMessage, MessageRole
from tests.fakes import FakeKVStack


class TestCombinedStats:
    """Tests for CombinedStats dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        stats = CombinedStats()
        assert stats.requests_total == 0
        assert stats.prompt_tokens == 0
        assert stats.completion_tokens == 0

    def test_total_tokens(self) -> None:
        """Test total tokens calculation."""
        stats = CombinedStats(prompt_tokens=100, completion_tokens=50)
        assert stats.total_tokens == 150

    def test_avg_latency(self) -> None:
        """Test average latency calculation."""
        stats = CombinedStats(requests_total=10, total_latency_ms=1000.0)
        assert stats.avg_latency_ms == 100.0


class TestCombinedServer:
    """Tests for CombinedServer."""

    @pytest.fixture
    def config(self) -> KVBenchConfig:
        """Create a test configuration."""
        return KVBenchConfig(instance_id="test-combined")

    @pytest.fixture
    def kv(self) -> FakeKVStack:
        """Create a fake KV stack."""
        return FakeKVStack(chunk_size=16)

    @pytest.fixture
    async def server(self, config: KVBenchConfig, kv: FakeKVStack) -> CombinedServer:
        """Create a test server instance."""
        server = CombinedServer(config=config, kv=kv)
        yield server
        await server.stop()

    def test_init(self, config: KVBenchConfig, kv: FakeKVStack) -> None:
        """Test server initialization."""
        server = CombinedServer(config=config, kv=kv)
        assert server.instance_id == "test-combined"
        assert server.model_name == "llama-3.1-8b"
        assert server.chunk_size == kv.chunk_size

    @pytest.mark.asyncio
    async def test_chat_completions(self, server: CombinedServer) -> None:
        """Test non-streaming chat completion."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello!")],
            max_tokens=10,
        )

        response = await server.chat_completions(request)

        assert response.model == "llama-3.1-8b"
        assert response.usage.completion_tokens == 10
        assert server.stats.requests_total == 1
        assert server.stats.requests_success == 1
        assert server.stats.completion_tokens == 10

    @pytest.mark.asyncio
    async def test_chat_completions_stream(self, server: CombinedServer) -> None:
        """Test streaming chat completion."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello!")],
            max_tokens=5,
            stream=True,
        )

        chunks = []
        async for chunk in server.chat_completions_stream(request):
            chunks.append(chunk)

        # Should have multiple chunks
        assert len(chunks) >= 5
        assert "data:" in chunks[0]
        assert "[DONE]" in chunks[-1]
        assert server.stats.requests_success == 1

    @pytest.mark.asyncio
    async def test_cache_behavior(self, server: CombinedServer, kv: FakeKVStack) -> None:
        """Test cache hit/miss behavior through the KV stack."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[
                ChatMessage(role=MessageRole.USER, content="Same prompt for caching " * 10)
            ],
            max_tokens=5,
        )

        # First request
        await server.chat_completions(request)
        first_hits = server.stats.cache_hits
        first_misses = server.stats.cache_misses
        assert kv.stats.stores == 1

        # Second request with same prompt
        await server.chat_completions(request)
        second_hits = server.stats.cache_hits

        # Should have more hits on second request, and the cached prefix
        # read through the stack
        assert second_hits > first_hits
        assert server.stats.cache_misses >= first_misses
        assert kv.stats.retrieves == 1

    @pytest.mark.asyncio
    async def test_list_models(self, server: CombinedServer) -> None:
        """Test listing models."""
        models = await server.list_models()
        assert len(models.data) == 1
        assert models.data[0].id == "llama-3.1-8b"

    @pytest.mark.asyncio
    async def test_health_check(self, server: CombinedServer) -> None:
        """Test health check."""
        health = await server.health_check()
        assert health.status == "healthy"
        assert health.server_type == "combined"
        assert health.instance_id == "test-combined"

    @pytest.mark.asyncio
    async def test_get_metrics(self, server: CombinedServer) -> None:
        """Test getting metrics."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Test")],
            max_tokens=5,
        )
        await server.chat_completions(request)

        metrics = await server.get_metrics()
        assert metrics.requests_total == 1
        assert metrics.requests_success == 1

    @pytest.mark.asyncio
    async def test_start_stop(self, config: KVBenchConfig, kv: FakeKVStack) -> None:
        """Server start/stop does not close the app-owned KV stack."""
        server = CombinedServer(config=config, kv=kv)

        await server.start()
        assert server._running is True

        await server.stop()
        assert server._running is False
        assert kv.closed is False

    def test_repr(self, server: CombinedServer) -> None:
        """Test string representation."""
        repr_str = repr(server)
        assert "CombinedServer" in repr_str
        assert "test-combined" in repr_str


class TestCacheKeyCorrectness:
    """Regression tests for cache-key correctness at the server level.

    The original implementation derived cache keys from prompt *length*
    only, so any two same-length prompts shared cache entries and hit-rate
    metrics were meaningless.
    """

    @pytest.fixture
    async def server(self) -> CombinedServer:
        """Create a test server instance."""
        server = CombinedServer(
            config=KVBenchConfig(instance_id="test-cache"),
            kv=FakeKVStack(chunk_size=64),
        )
        yield server
        await server.stop()

    @staticmethod
    def _request(content: str) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content=content)],
            max_tokens=1,
        )

    @pytest.mark.asyncio
    async def test_same_length_different_content_does_not_share_cache(
        self, server: CombinedServer
    ) -> None:
        """Two same-length, different-content prompts must not hit each
        other's cache entries."""
        await server.chat_completions(self._request("A" * 2000))
        hits_after_first = server.stats.cache_hits
        misses_after_first = server.stats.cache_misses
        assert hits_after_first == 0
        assert misses_after_first > 0

        await server.chat_completions(self._request("B" * 2000))
        # No new hits: different content must miss
        assert server.stats.cache_hits == hits_after_first
        assert server.stats.cache_misses > misses_after_first

    @pytest.mark.asyncio
    async def test_identical_prompt_hits_cache(self, server: CombinedServer) -> None:
        """A repeated identical prompt must hit the cache."""
        await server.chat_completions(self._request("C" * 2000))
        misses_after_first = server.stats.cache_misses

        await server.chat_completions(self._request("C" * 2000))
        assert server.stats.cache_hits > 0
        assert server.stats.cache_misses <= misses_after_first + 1

    @pytest.mark.asyncio
    async def test_shared_prefix_hits_only_prefix_chunks(self, server: CombinedServer) -> None:
        """A prompt sharing a long prefix with a previous one must hit the
        prefix chunks and miss its own distinct suffix."""
        prefix = "You are a helpful assistant. " * 200  # ~1450 tokens
        await server.chat_completions(self._request(prefix + "Question one?"))
        hits_first = server.stats.cache_hits

        await server.chat_completions(self._request(prefix + "A different question two?"))
        # Prefix chunks hit, suffix chunk misses
        assert server.stats.cache_hits > hits_first
        assert server.stats.cache_misses > 0


class TestCombinedServerTiming:
    """Timing configuration must flow into served latency."""

    def _server(self, timing: dict | None = None, gpu: dict | None = None) -> CombinedServer:
        data: dict = {"instance_id": "test-timing"}
        if timing:
            data["timing"] = timing
        if gpu:
            data["gpu"] = gpu
        config = KVBenchConfig.model_validate(data)
        return CombinedServer(config=config, kv=FakeKVStack(chunk_size=16))

    def test_simple_timing_mode_used(self) -> None:
        """simple_mode gives exact ms/token latencies."""
        server = self._server(
            timing={
                "simple_mode": True,
                "prefill_ms_per_token": 0.5,
                "decode_ms_per_token": 2.0,
            }
        )
        assert server._calculate_prefill_latency_ms(100) == pytest.approx(50.0)
        assert server._calculate_decode_latency_ms(1000) == pytest.approx(2.0)

    def test_tp_communication_increases_decode_latency(self) -> None:
        """With TP > 1, enabling AllReduce timing makes decode slower."""
        with_comm = self._server(
            gpu={"tp_size": 4},
            timing={"include_tp_communication": True},
        )
        without_comm = self._server(
            gpu={"tp_size": 4},
            timing={"include_tp_communication": False},
        )
        assert with_comm._calculate_decode_latency_ms(
            1000
        ) > without_comm._calculate_decode_latency_ms(1000)

    def test_pp_communication_increases_prefill_latency(self) -> None:
        """With PP > 1, enabling send/recv timing makes prefill slower."""
        with_comm = self._server(
            timing={"pp_size": 4, "include_pp_communication": True}
        )
        without_comm = self._server(
            timing={"pp_size": 4, "include_pp_communication": False}
        )
        assert with_comm._calculate_prefill_latency_ms(
            1000
        ) > without_comm._calculate_prefill_latency_ms(1000)


class TestTracing:
    """Stage spans emitted while serving a request."""

    LONG_PROMPT = "alpha bravo charlie delta echo foxtrot golf hotel " * 20  # ~3 chunks of 16

    @pytest.fixture
    def tracer(self, tmp_path):
        from kvbench.trace import spans

        t = spans.install(tmp_path / "spans.jsonl")
        yield t
        spans.uninstall()

    @pytest.fixture
    def server(self, request) -> CombinedServer:
        from kvbench.core.config import KVStackConfig

        decode_steps = getattr(request, "param", True)
        config = KVBenchConfig(
            instance_id="test-trace", kv=KVStackConfig(trace_decode_steps=decode_steps)
        )
        return CombinedServer(config=config, kv=FakeKVStack(chunk_size=16))

    @staticmethod
    def spans_of(tracer) -> list[dict]:
        import json

        tracer.flush()
        return [json.loads(line) for line in tracer.path.read_text().splitlines()]

    @staticmethod
    def names(records: list[dict]) -> list[str]:
        return [r["name"] for r in records]

    def make_request(self, prompt: str, max_tokens: int = 3, stream: bool = False):
        return ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content=prompt)],
            max_tokens=max_tokens,
            stream=stream,
        )

    async def test_cold_request_stages(self, server: CombinedServer, tracer) -> None:
        from kvbench.trace import spans

        with spans.root("request", trace_id="t1"):
            response = await server.chat_completions(
                self.make_request(self.LONG_PROMPT, max_tokens=3), completion_id="chatcmpl-abc"
            )

        recs = self.spans_of(tracer)
        by = {r["name"]: r for r in recs}
        assert all(r["trace_id"] == "t1" for r in recs)
        # closing order == processing order for sequential stages
        stage_order = [n for n in self.names(recs) if n not in ("decode.step", "request")]
        assert stage_order == [
            "tokenize",
            "kv.lookup",
            "prefill.compute",
            "kv.store",
            "decode",
            "response.end",
        ]
        assert self.names(recs).count("decode.step") == 3
        root = by["request"]["span_id"]
        for name in ("tokenize", "kv.lookup", "prefill.compute", "kv.store", "decode"):
            assert by[name]["parent_id"] == root, name
        steps = [r for r in recs if r["name"] == "decode.step"]
        assert all(s["parent_id"] == by["decode"]["span_id"] for s in steps)
        assert [s["attrs"]["i"] for s in steps] == [0, 1, 2]

        tok = by["tokenize"]["attrs"]
        assert tok["chunk_size"] == 16 and tok["tokens"] > 32
        assert tok["full_chunks"] == tok["tokens"] // 16
        assert by["kv.lookup"]["attrs"] == {"tokens": tok["tokens"], "hit_tokens": 0, "hit_chunks": 0}
        pre = by["prefill.compute"]["attrs"]
        assert pre["miss_tokens"] == tok["tokens"] and pre["context_tokens"] == 0
        assert pre["emulated_ms"] > 0 and "compute_bound" in pre and "breakdown" in pre
        assert by["prefill.compute"]["dur_ms"] >= pre["emulated_ms"] * 0.9
        store = by["kv.store"]["attrs"]
        assert store == {
            "tokens": tok["full_chunks"] * 16,
            "skip_leading": 0,
            "chunks": tok["full_chunks"],
        }
        dec = by["decode"]["attrs"]
        assert dec["generated"] == 3 and dec["max_tokens"] == 3
        assert dec["context_length"] == tok["tokens"]
        assert dec["emulated_ms_total"] > 0 and "token_ts_ms" not in dec
        assert by["response.end"]["attrs"] == {"finish_reason": "stop", "generated": 3}
        assert "kv.retrieve" not in by
        assert response.id == "chatcmpl-abc"

    async def test_repeat_request_retrieves_and_skips_store(
        self, server: CombinedServer, tracer
    ) -> None:
        from kvbench.trace import spans

        await server.chat_completions(self.make_request(self.LONG_PROMPT, max_tokens=1))
        with spans.root("request", trace_id="t2"):
            await server.chat_completions(self.make_request(self.LONG_PROMPT, max_tokens=1))

        recs = [r for r in self.spans_of(tracer) if r["trace_id"] == "t2"]
        by = {r["name"]: r for r in recs}
        full = by["tokenize"]["attrs"]["full_chunks"]
        assert by["kv.lookup"]["attrs"]["hit_chunks"] == full
        assert by["kv.retrieve"]["attrs"] == {
            "tokens": full * 16,
            "chunks": full,
            "retrieved_tokens": full * 16,
        }
        # the trailing partial chunk is still computed, but nothing new is stored
        assert by["prefill.compute"]["attrs"]["context_tokens"] == full * 16
        assert "kv.store" not in by

    async def test_streaming_marks_first_chunk_before_decode(
        self, server: CombinedServer, tracer
    ) -> None:
        from kvbench.trace import spans

        with spans.root("request", trace_id="t3"):
            chunks = [
                c
                async for c in server.chat_completions_stream(
                    self.make_request("Hello!", max_tokens=2, stream=True),
                    completion_id="chatcmpl-s",
                )
            ]

        recs = self.spans_of(tracer)
        by = {r["name"]: r for r in recs}
        assert by["response.first_chunk"]["dur_ms"] == 0.0
        first_step = min(r["start_ms"] for r in recs if r["name"] == "decode.step")
        assert by["response.first_chunk"]["start_ms"] <= first_step
        assert by["response.end"]["attrs"]["generated"] == 2
        assert chunks[-1] == "data: [DONE]\n\n"
        assert all("chatcmpl-s" in c for c in chunks[:-1])

    @pytest.mark.parametrize("server", [False], indirect=True)
    async def test_decode_steps_can_collapse_to_timestamps(
        self, server: CombinedServer, tracer
    ) -> None:
        from kvbench.trace import spans

        with spans.root("request", trace_id="t4"):
            await server.chat_completions(self.make_request("Hello!", max_tokens=4))

        recs = self.spans_of(tracer)
        assert "decode.step" not in self.names(recs)
        dec = next(r for r in recs if r["name"] == "decode")["attrs"]
        assert dec["generated"] == 4
        assert len(dec["token_ts_ms"]) == 4
        assert dec["token_ts_ms"] == sorted(dec["token_ts_ms"])

    async def test_no_tracer_means_no_spans_and_no_errors(self, server: CombinedServer) -> None:
        from kvbench.trace import spans

        assert spans.get_tracer() is None
        response = await server.chat_completions(self.make_request("Hello!", max_tokens=2))
        assert response.usage.completion_tokens == 2
