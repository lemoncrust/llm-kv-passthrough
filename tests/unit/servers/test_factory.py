"""Production entry points must never select the old simulated proxy."""

import pytest

from kvbench.core.config import KVBenchConfig, ServerConfig, TimingConfig
from kvbench.servers.factory import create_server
from kvbench.servers.rate_server import RateServer
from tests.fakes import FakeKVStack


def test_rate_server_is_the_production_default():
    kv = FakeKVStack()
    assert isinstance(create_server(KVBenchConfig(), kv), RateServer)


@pytest.mark.parametrize("mode", ["prefill", "decode", "proxy"])
def test_deferred_modes_fail_before_running(mode):
    with pytest.raises(ValueError, match="combined serving only"):
        create_server(KVBenchConfig(server=ServerConfig(server_type=mode)), FakeKVStack())


def test_missing_stack_is_an_error():
    with pytest.raises(ValueError, match="KV stack is required"):
        create_server(KVBenchConfig())


def test_legacy_timing_does_not_silently_override_rates():
    with pytest.raises(ValueError, match="legacy timing"):
        create_server(KVBenchConfig(timing=TimingConfig(batching=True)), FakeKVStack())


def test_worker_processes_cannot_multiply_engine_capacity():
    with pytest.raises(ValueError, match="workers=1"):
        create_server(KVBenchConfig(server=ServerConfig(workers=2)), FakeKVStack())
