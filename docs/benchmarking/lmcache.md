# Validating LMCache storage load

Run memory-bounded real-manager tests:

```bash
bash scripts/test-safe.sh tests/integration -m integration
```

`test_v1_storage.py` independently calculates expected chunk/file bytes and
checks real disk writes, disk reads after CPU eviction, generated-prefix reuse,
rank geometry, KV dtype, large-prefix retrieval and cleanup. Small fixtures keep
host memory bounded. `test_live_server.py` additionally launches a separately
memory-limited CLI server and drives real HTTP/SSE traffic.

Store counters represent submissions; backend deduplication and admission under
pressure can reduce physical writes. Trace `io` events and files are the oracle
for actual disk activity. Remote connectors require separate infrastructure
validation and are not covered by this matrix.

`kvbench trace2fio` is an experimental workload-shape approximation. It does not
replay request dependencies, exact temporal bursts, or delete/recreate churn.
