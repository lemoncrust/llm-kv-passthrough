# KV-Bench V1

Generate representative storage load by driving **real LMCache** with an
external query workload. V1 prioritizes how workload changes affect I/O over
matching a particular inference system's latency.

- Explicit aggregate prefill/decode rates and bounded shared scheduling.
- File-backed model/hardware catalogs and TP/PP KV geometry.
- Prompt and generated-token KV, reusable across conversation turns.
- Real LMCache CPU/disk operations, traces and observed serving metrics.

Start with [quickstart](getting-started/quickstart.md), read the
[configuration](getting-started/configuration.md), and see the
[architect's plan](V1_PLAN.md) for assumptions and future fidelity levels.

No native serving-framework scheduler or disaggregated proxy runs in V1.
