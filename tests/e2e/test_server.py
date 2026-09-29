"""ASGI contract tests. Real disk workloads live in tests/integration/."""

import json

import pytest
from fastapi.testclient import TestClient

import kvbench.servers.app as app_module
from kvbench.core.config import KVBenchConfig, MetricsConfig, SchedulerConfig, TimingConfig
from kvbench.servers.app import create_app
from tests.fakes import FakeKVStack

pytestmark = pytest.mark.e2e


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "create_kv_stack", lambda _: FakeKVStack(16))
    cfg = KVBenchConfig(timing=TimingConfig(full_speed=True), scheduler=SchedulerConfig(max_request_bytes=8192))
    with TestClient(create_app(cfg)) as client:
        yield client


def request(**updates):
    return {"model": "llama-3.1-8b", "messages": [{"role": "user", "content": "hello " * 100}],
            "max_completion_tokens": 40, **updates}


def test_non_streaming_usage_and_length(client):
    response = client.post("/v1/chat/completions", json=request())
    assert response.status_code == 200
    data = response.json()
    assert data["usage"]["prompt_tokens"] > 100
    assert data["usage"]["completion_tokens"] == 40
    assert data["choices"][0]["finish_reason"] == "length"
    assert client.get("/stats").json()["requests_success"] == 1


def test_streaming_usage_matches_nonstreaming(client):
    normal = client.post("/v1/chat/completions", json=request()).json()
    stream = client.post("/v1/chat/completions", json=request(stream=True, stream_options={"include_usage": True}))
    assert stream.text.endswith("data: [DONE]\n\n")
    events = [json.loads(line[6:]) for line in stream.text.splitlines()
              if line.startswith("data: {")]
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert events[-1]["usage"][key] == normal["usage"][key]
    assert normal["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert events[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
    text = "".join(c["delta"].get("content") or "" for e in events for c in e["choices"])
    assert text == normal["choices"][0]["message"]["content"]


def test_shared_prefix_has_real_hits_in_server_contract(client):
    first = client.post("/v1/chat/completions", json=request()).json()
    followup = request(messages=[*request()["messages"], first["choices"][0]["message"],
                                 {"role": "user", "content": "next turn"}])
    assert client.post("/v1/chat/completions", json=followup).status_code == 200
    state = client.get("/kvbench/state").json()
    # Includes computed completion KV, not just the first prompt.
    assert state["kv_stats"]["hit_tokens"] > first["usage"]["prompt_tokens"]


@pytest.mark.parametrize("updates,code", [({"model": "wrong"}, 400), ({"max_tokens": 0}, 422),
                                         ({"max_tokens": -1}, 422), ({"n": 2}, 400)])
def test_invalid_requests(client, updates, code):
    assert client.post("/v1/chat/completions", json=request(**updates)).status_code == code


def test_body_is_bounded_before_tokenization(client):
    assert client.post("/v1/chat/completions", json=request(messages=[{"role": "user", "content": "x" * 9000}])).status_code == 413


def test_observability(client):
    assert client.get("/health").json()["status"] == "healthy"
    assert client.get("/v1/models").json()["data"][0]["id"] == "llama-3.1-8b"
    assert "kvbench_kv_ops_total" in client.get("/metrics").text
    manifest = client.get("/kvbench/manifest").json()
    assert manifest["fidelity"] == "rate-v1"
    assert manifest["ranks"][0]["bytes_per_token"] == 131072


def test_disabled_metrics(monkeypatch):
    monkeypatch.setattr(app_module, "create_kv_stack", lambda _: FakeKVStack())
    with TestClient(create_app(KVBenchConfig(metrics=MetricsConfig(enabled=False)))) as client:
        assert client.get("/metrics").status_code == 404
