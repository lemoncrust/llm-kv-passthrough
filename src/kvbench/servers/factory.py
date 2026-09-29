"""Production serving boundary: fidelity level 1, combined engine only."""

from kvbench.core.config import KVBenchConfig
from kvbench.kv.base import KVStack
from kvbench.servers.rate_server import RateServer


def create_server(config: KVBenchConfig, kv: KVStack | None = None) -> RateServer:
    config.validate_serving()
    if kv is None:
        raise ValueError("a KV stack is required for combined serving")
    return RateServer(config, kv)
