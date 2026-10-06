"""Unit tests for kvbench.servers.app module.

These tests exercise the FastAPI HTTP layer and — critically — the
configuration wiring: the original implementation constructed the app from
default configuration regardless of CLI flags, env vars, or YAML files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import kvbench.servers.app as app_module
from kvbench.core.config import (
    GPUEmulationConfig,
    KVBenchConfig,
    ServerConfig,
)
from kvbench.servers.app import create_app, load_config
from kvbench.servers.warmup import WarmupController
from tests.fakes import FakeKVStack


@pytest.fixture(autouse=True)
def fast_eviction_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(WarmupController, "EVICTION_CHECK_ATTEMPTS", 1)
    monkeypatch.setattr(WarmupController, "EVICTION_CHECK_DELAY_S", 0.0)


@pytest.fixture(autouse=True)
def reset_lazy_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the lazy app singleton and scrub KVBENCH_* env vars."""
    monkeypatch.setattr(app_module, "_app", None)
    import os

    for key in list(os.environ):
        if key.startswith("KVBENCH_"):
            monkeypatch.delenv(key)


@pytest.fixture(autouse=True)
def fake_kv_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Substitute the real LMCache stack with an in-memory fake.

    The HTTP-layer tests exercise the app, not LMCache; the real stack is
    covered by tests/integration/test_lmcache_stack.py.
    """
    monkeypatch.setattr(app_module, "create_kv_stack", lambda _config: FakeKVStack())


def make_config(**server_kwargs: object) -> KVBenchConfig:
    """Build a small test configuration."""
    return KVBenchConfig(
        instance_id="test-app",
        server=ServerConfig(**server_kwargs),  # type: ignore[arg-type]
    )


class TestConfigWiring:
    """The app must reflect the configuration it is given."""

    def test_create_app_uses_provided_config(self) -> None:
        config = KVBenchConfig(
            instance_id="wired-1",
            server=ServerConfig(model_profile="llama-3.1-70b", server_type="combined"),
            gpu=GPUEmulationConfig(gpu_profile="A100_SXM"),
        )
        with TestClient(create_app(config)) as client:
            health = client.get("/health").json()
            assert health["instance_id"] == "wired-1"
            assert health["model"] == "llama-3.1-70b"
            assert health["server_type"] == "combined"

    def test_create_app_server_type_is_respected(self) -> None:
        config = make_config(server_type="prefill")
        with TestClient(create_app(config)) as client:
            health = client.get("/health").json()
            assert health["server_type"] == "prefill"

    def test_get_app_reads_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`uvicorn kvbench.servers.app:app` deployments configure via env."""
        monkeypatch.setenv("KVBENCH_INSTANCE_ID", "env-test-42")
        monkeypatch.setenv("KVBENCH_SERVER__MODEL_PROFILE", "qwen-2.5-7b")
        monkeypatch.setenv("KVBENCH_SERVER__SERVER_TYPE", "prefill")

        with TestClient(app_module.get_app()) as client:
            health = client.get("/health").json()
            assert health["instance_id"] == "env-test-42"
            assert health["model"] == "qwen-2.5-7b"
            assert health["server_type"] == "prefill"

    def test_load_config_prefers_config_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The CLI hands its resolved config over via KVBENCH_CONFIG_FILE."""
        config_path = tmp_path / "config.yaml"
        KVBenchConfig(
            instance_id="from-file",
            server=ServerConfig(model_profile="llama-3.1-70b"),
        ).to_yaml(config_path)
        monkeypatch.setenv("KVBENCH_CONFIG_FILE", str(config_path))
        # A stray env var must not override the resolved file
        monkeypatch.setenv("KVBENCH_INSTANCE_ID", "should-be-ignored")

        config = load_config()
        assert config.instance_id == "from-file"
        assert config.server.model_profile == "llama-3.1-70b"

    def test_load_config_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KVBENCH_GPU__GPU_PROFILE", "L4")
        config = load_config()
        assert config.gpu.gpu_profile == "L4"

    def test_invalid_profile_env_var_fails_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A typo'd profile must fail at startup, not silently use defaults."""
        monkeypatch.setenv("KVBENCH_GPU__GPU_PROFILE", "H100_SMX")  # typo
        with pytest.raises(Exception, match="Unknown GPU profile"):
            load_config()

    def test_endpoint_lists_parse_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "KVBENCH_DISTRIBUTED__PREFILL_ENDPOINTS",
            "http://prefill-1:8000,http://prefill-2:8000",
        )
        monkeypatch.setenv(
            "KVBENCH_DISTRIBUTED__DECODE_ENDPOINTS",
            '["http://decode-1:8000", "http://decode-2:8000"]',
        )
        config = load_config()
        assert config.distributed.prefill_endpoints == [
            "http://prefill-1:8000",
            "http://prefill-2:8000",
        ]
        assert config.distributed.decode_endpoints == [
            "http://decode-1:8000",
            "http://decode-2:8000",
        ]


class TestHttpEndpoints:
    """Exercise the HTTP layer end-to-end through TestClient."""

    @pytest.fixture
    def client(self) -> TestClient:
        with TestClient(create_app(make_config())) as client:
            yield client

    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_models(self, client: TestClient) -> None:
        response = client.get("/v1/models")
        assert response.status_code == 200
        assert response.json()["data"][0]["id"] == "llama-3.1-8b"

    def test_chat_completion_roundtrip(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.1-8b",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 2,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"]
        assert body["usage"]["completion_tokens"] == 2

    def test_chat_completion_streaming(self, client: TestClient) -> None:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "llama-3.1-8b",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 2,
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            payload = "".join(response.iter_text())
        assert "chat.completion.chunk" in payload
        assert payload.rstrip().endswith("data: [DONE]")

    def test_metrics_counts_requests(self, client: TestClient) -> None:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.1-8b",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
            },
        )
        metrics = client.get("/stats").json()
        assert metrics["requests_total"] == 1
        assert metrics["requests_success"] == 1


class TestObservabilityEndpoints:
    """Prometheus metrics, JSON stats, and KV state endpoints."""

    def test_metrics_is_prometheus_format(self) -> None:
        with TestClient(create_app(make_config())) as client:
            response = client.get("/metrics")
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/plain")
            body = response.text
            assert "kvbench_kv_ops_total" in body
            assert "kvbench_request_duration_seconds" in body

    def test_metrics_reflect_kv_activity(self) -> None:
        with TestClient(create_app(make_config())) as client:
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-3.1-8b",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hello " * 300}],
                },
            )
            body = client.get("/metrics").text
            assert 'kvbench_kv_ops_total{op="lookup"} 1.0' in body
            assert 'kvbench_requests_total{status="success"} 1.0' in body

    def test_stats_is_json(self) -> None:
        with TestClient(create_app(make_config())) as client:
            data = client.get("/stats").json()
            assert "requests_total" in data
            assert "cache_hits" in data

    def test_kv_state_reports_capacity_and_usage(self) -> None:
        with TestClient(create_app(make_config())) as client:
            state = client.get("/kvbench/state").json()
            assert state["capacity"]["total_capacity_bytes"] == 65536
            assert "FakeBackend" in state["usage"]
            assert state["kv_stats"]["lookups"] == 0


class TestWarmupEndpoints:
    """Warmup lifecycle over HTTP."""

    def test_warmup_runs_to_completion(self) -> None:
        import time as _time

        with TestClient(create_app(make_config())) as client:
            response = client.post("/kvbench/warmup", json={"seq_tokens": 256, "concurrency": 2})
            assert response.status_code == 202
            assert response.json()["state"] == "running"

            for _ in range(100):
                status = client.get("/kvbench/warmup").json()
                if status["state"] != "running":
                    break
                _time.sleep(0.05)
            assert status["state"] == "done"
            assert status["stored_bytes"] >= 65536

    def test_warmup_status_idle_initially(self) -> None:
        with TestClient(create_app(make_config())) as client:
            assert client.get("/kvbench/warmup").json()["state"] == "idle"

    def test_warmup_unavailable_on_proxy(self) -> None:
        config = KVBenchConfig(
            instance_id="proxy-test",
            server=ServerConfig(server_type="proxy"),
        )
        with TestClient(create_app(config)) as client:
            assert client.post("/kvbench/warmup", json={}).status_code == 404
            assert client.get("/kvbench/state").status_code == 404


class TestTracing:
    """Request root spans and client identity at the HTTP layer."""

    @pytest.fixture
    def tracer(self, tmp_path: Path):
        from kvbench.trace import spans

        t = spans.install(tmp_path / "spans.jsonl")
        yield t
        spans.uninstall()

    @pytest.fixture
    def client(self) -> TestClient:
        with TestClient(create_app(make_config())) as client:
            yield client

    @staticmethod
    def spans_of(tracer) -> list[dict]:
        import json

        tracer.flush()
        return [json.loads(line) for line in tracer.path.read_text().splitlines()]

    @staticmethod
    def body(max_tokens: int = 2, stream: bool = False) -> dict:
        return {
            "model": "llama-3.1-8b",
            "messages": [{"role": "user", "content": "Hello there!"}],
            "max_tokens": max_tokens,
            "stream": stream,
        }

    def test_client_ids_become_trace_identity(self, client: TestClient, tracer) -> None:
        headers = {"X-Request-ID": "req-1", "X-Correlation-ID": "conv-A"}
        r1 = client.post("/v1/chat/completions", json=self.body(), headers=headers)
        assert r1.status_code == 200
        headers["X-Request-ID"] = "req-2"
        r2 = client.post("/v1/chat/completions", json=self.body(), headers=headers)
        assert r2.status_code == 200

        recs = self.spans_of(tracer)
        roots = [r for r in recs if r["name"] == "request"]
        assert [r["trace_id"] for r in roots] == ["req-1", "req-2"]
        first, second = (r["attrs"] for r in roots)
        assert first["conversation_id"] == "conv-A" and first["turn_index"] == 0
        assert first["previous_request_id"] is None
        assert first["client_request_id"] is True
        assert first["completion_id"] == r1.json()["id"]
        assert first["model"] == "llama-3.1-8b" and first["stream"] is False
        assert first["status"] == "ok"
        assert second["turn_index"] == 1 and second["previous_request_id"] == "req-1"
        # every stage of a request shares its trace id and hangs off its root
        for root in roots:
            stages = [r for r in recs if r["trace_id"] == root["trace_id"] and r is not root]
            assert stages, root["trace_id"]
            assert {"tokenize", "kv.lookup", "decode", "response.end"} <= {s["name"] for s in stages}
            assert all(
                s["parent_id"] == root["span_id"] or s["name"] == "decode.step" for s in stages
            )

    def test_without_headers_trace_id_is_completion_id(self, client: TestClient, tracer) -> None:
        r = client.post("/v1/chat/completions", json=self.body())
        root = next(x for x in self.spans_of(tracer) if x["name"] == "request")
        assert root["trace_id"] == r.json()["id"]
        assert root["attrs"]["client_request_id"] is False
        assert "conversation_id" not in root["attrs"]

    def test_streaming_request_root_records_first_chunk(self, client: TestClient, tracer) -> None:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json=self.body(max_tokens=3, stream=True),
            headers={"X-Request-ID": "stream-1"},
        ) as response:
            assert response.status_code == 200
            "".join(response.iter_text())

        recs = self.spans_of(tracer)
        root = next(x for x in recs if x["name"] == "request")
        assert root["trace_id"] == "stream-1"
        assert root["attrs"]["stream"] is True and root["attrs"]["status"] == "ok"
        assert root["attrs"]["first_chunk_ms"] >= 0
        assert root["attrs"]["chunks"] == 3 + 3  # role chunk + 3 tokens + end chunk + [DONE]
        names = [x["name"] for x in recs if x["trace_id"] == "stream-1"]
        assert "response.first_chunk" in names and names.count("decode.step") == 3
        # the root closes last, after the stream is fully consumed
        assert names[-1] == "request"
