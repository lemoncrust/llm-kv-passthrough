# V1 configuration

The runnable reference is `examples/rate-v1.yaml`. Schema implementation:
`src/kvbench/core/config.py`.

## Resolution

- CLI flags override a selected `--config` YAML file.
- Without a file, `KVBENCH_*` environment variables are used; `__` separates
  nested fields, e.g. `KVBENCH_SCHEDULER__DECODE_TOKENS_PER_SECOND=500`.
- File and environment bases are alternatives, not merged layers.
- Paths inside a KV-Bench file are relative to that file. LMCache's own path
  semantics apply inside its configuration.
- Unknown fields, nonfinite values, invalid profile names and invalid KV head
  partitioning fail. One process runs one combined engine (`workers: 1`).

## Rate scheduler

| Field under `scheduler` | Default | Meaning |
|---|---:|---|
| `prefill_tokens_per_second` | 10000 | Aggregate engine prefill rate |
| `decode_tokens_per_second` | 1000 | Aggregate engine decode rate |
| `prefill_chunk_tokens` | 256 | Maximum prefill work per scheduler round/request |
| `max_active_requests` | 16 | Active request slots, including I/O-blocked requests |
| `max_pending_requests` | 64 | Additional queued requests |
| `max_context_tokens` | 131072 | Prompt + output bound, also limited by the model |
| `max_output_tokens` | 4096 | Maximum requested completion length |
| `max_request_bytes` | 4194304 | HTTP chat-body limit before JSON parsing |
| `request_timeout_s` | 600 | Request deadline after request handling begins |

Cancellation/deadline cleanup drains any already-running blocking LMCache call
before releasing its pins. This can outlast the request deadline; the underlying
manager/backend controls its own I/O timeouts. Prefix-load leases are serialized
to avoid circular waits for a fully pinned CPU staging pool.

`timing.full_speed` skips compute sleeps without changing scheduling/KV semantics.
`timing.chars_per_token` defaults to 4.0 and means **UTF-8 bytes per synthetic
token**; fractional spans work. `timing.pp_size` and `gpu.tp_size` specify the
logical KV rank layout. Rates already describe the whole rank group and are
not multiplied by either rank count.

GPU peak specifications are descriptive/offline in V1. Set rates explicitly
for each intended hardware/model operating point.

## Profiles

`profiles_file` extends/overrides the bundled YAML catalog for this run.
See [profile schema](../architecture/gpu-emulation.md). Resolved definitions
appear in `/kvbench/manifest` with rates, ranks and package versions.

## KV adapter

| Field under `kv` | Default | Meaning |
|---|---|---|
| `stack` | `lmcache` | Only implemented production manager |
| `lmcache_config_file` | unset | LMCache YAML; otherwise `LMCACHE_*` environment |
| `trace_file` | unset | JSONL trace; use a unique path for each process/run |
| `max_cpu_memory_gb` | 2 | Aggregate CPU-pool allocation guard, GiB |
| `max_chunk_mb` | 256 | Largest permitted rank-chunk payload, MiB |
| `random_fill` | true | High-entropy deterministic payloads; false means zeros |
| `seed` | 24301 | Payload seed, mixed with rank and token prefix |

LMCache sizes are **per rank**. The guard checks their sum before creating any
engine. With no explicit CPU sizing, the adapter uses 0.25 GiB per rank. It keeps
process-wide cyclic GC enabled. It does not allocate model weights or simulated
HBM, and it does not change LMCache chunk size or explicit tier sizes.

## Metrics and migration

`metrics.enabled` and `metrics.include_histograms` control the exporter. Metrics
are on the same HTTP port as serving. The historical `prometheus_port` setting
does not start a separate listener; nondefault values are rejected for V1.

Legacy per-request/roofline/batching/calibration timing flags are offline
experiments, not selectable V1 serving modes. Replace them with explicit rates.
Legacy `storage`, `connector`, and `random_pool_mb` settings are rejected. The
LMCache file owns storage. Standalone prefill/decode/proxy serving is deferred;
run independent combined replicas instead.
