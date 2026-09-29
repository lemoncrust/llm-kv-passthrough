# Validation

```bash
bash scripts/test-safe.sh tests/
bash scripts/test-safe.sh tests/integration -m integration
bash scripts/test-safe.sh tests/e2e -m e2e
```

The runner caps virtual memory at 4 GiB, limits allocator arenas and numerical
threads, and avoids repository cache writes. Real integration fixtures use
small temporary CPU/disk tiers. The live HTTP subprocess has its own cap.

- `unit/core/test_v1_profiles.py`: file catalogs, rank geometry, precision,
  fractional tokenization and multi-turn prefix identity.
- `unit/servers/test_rate_server.py`: storage growth, rate sharing, causal I/O,
  queue limits, cancellation and deadlines.
- `unit/kv/test_lifecycle.py`: cancellation cannot abandon a blocking lookup.
- `e2e/test_server.py`: bounded HTTP/ASGI protocol, SSE usage and validation.
- `integration/test_v1_storage.py`: real file bytes, disk reads after CPU
  eviction, workload deltas, rank files, shutdown and startup failure.
- `integration/test_live_server.py`: actual CLI/uvicorn/HTTP/SSE/LMCache/disk.

Older analytical/server unit tests remain regression coverage for experimental
utilities. They are not evidence of V1 framework or hardware fidelity.
