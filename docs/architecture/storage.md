# Storage ownership and measurements

All KV payload I/O goes through **real LMCache**. KV-Bench has no independent
storage backend implementations. LMCache configuration controls chunk size,
CPU/local-disk capacities and optional remote backends.

Each logical TP/PP rank owns a CPU pool and backend instance. Configured tier
sizes are per rank; `/kvbench/state` reports their aggregate and per-rank chunk
geometry. A shared local-disk directory alone does not provide cross-process
discovery: LMCache's local-disk index is process-local. Use an appropriate
LMCache remote backend for cross-process sharing, and validate that deployment.

Use a fresh directory for cold-run comparisons. A restarted local-disk engine
does not account for old files in its new in-memory index; its per-run capacity
limit is not a cleanup policy for artifacts from earlier runs.

KV `stored_tokens` counts bytes/tokens **submitted** to LMCache, not confirmed
durable writes. LMCache can deduplicate or decline cache admission under
pressure, and disk writes happen on its workers. Trace `io/write` and `io/read`
events measure completed file operations. These are application-level file
operations, not block-device tracing; remote physical I/O is not instrumented.

Warmup reports submitted bytes and an earliest-prefix eviction probe. That is
evidence of eviction, not a proof that every tier is in steady state. Warmup
and measured serving must not overlap. Keep separate traces/run windows when
deriving workload characteristics.
