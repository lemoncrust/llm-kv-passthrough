# Scripts

- `bash scripts/test-safe.sh tests/`: memory-bounded tests using `.venv/bin/python`
  (override `PYTHON`). A 4 GiB virtual-memory cap and small allocator/thread
  settings prevent test allocations from exhausting a shared development host.
- `bash scripts/lmcache_test.sh`: the real-manager integration subset.
- `bash scripts/genai_perf_test.sh`: run your installed GenAI-Perf against an
  existing server. It propagates failures and does not install/start services.
  Options use `KVBENCH_ENDPOINT`, `KVBENCH_MODEL`, `KVBENCH_INPUT_TOKENS`,
  `KVBENCH_OUTPUT_TOKENS`, `KVBENCH_CONCURRENCY`, `KVBENCH_REQUEST_COUNT`.
- `coverage_report.sh`: historical coverage helper; functional storage
  invariants and sensitivity tests are the V1 acceptance criteria.
