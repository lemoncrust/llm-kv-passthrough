# vLLM KV Tracing Connector

A drop-in vLLM KV connector that records every storage operation LMCache
performs during real inference — disk reads, GPU copies, cache lookups — as
time-stamped spans in the same JSONL format kvbench uses. The resulting
traces can be compared directly against a kvbench emulation run to measure
how faithfully the emulator reproduces real vLLM behaviour.

## Requirements

- vLLM with LMCache KV connector support
- kvbench installed (`pip install "kvbench[lmcache]"`)
- An LMCache configuration file (the same one used by the kvbench server)

## Usage

Pass the connector through `--kv-transfer-config` when starting vLLM:

```bash
export LMCACHE_CONFIG_FILE=/path/to/lmcache.yaml

KV_CFG='{
  "kv_connector": "TracingLMCacheConnector",
  "kv_connector_module_path": "kvbench.vllm_trace.connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {"trace_file": "/path/to/vllm-kv-trace.jsonl"}
}'

vllm serve <model> \
    --kv-transfer-config "$KV_CFG" \
    --no-enable-prefix-caching
```

Each vLLM process (scheduler + worker) writes its own file so they do not
overwrite each other: `/path/to/vllm-kv-trace.<pid>.jsonl`.

Combine the per-process files when analysing:

```bash
cat /path/to/vllm-kv-trace.*.jsonl > combined.jsonl
```

## Spans recorded

All spans use real wall-clock durations, not emulated values.

| Span | Side | What it measures |
|---|---|---|
| `kv.lookup` | scheduler | Prefix-hit check; `hit_tokens` attr |
| `kv.load_init` | worker | LMCache disk-to-CPU fetch (start\_load\_kv) |
| `gpu.copy_in` | worker | CPU-pinned-buffer to GPU-HBM per chunk; synchronous |
| `gpu.forward_wait` | worker | CPU blocked on CUDA stream until forward pass drains |
| `kv.store` | worker | GPU-to-CPU copies + disk-write submission |
| `gpu.copy_out` | worker | One span per chunk inside kv.store |
| `io.read` | LMCache thread | Physical file read (O\_DIRECT when configured) |
| `io.write` | LMCache thread | Physical file write; async, may overlap next request |
| `lmcache.put` | worker | Chunk hand-off to storage backend |
| `lmcache.get` | worker | Chunk retrieval from storage backend |
| `lmcache.block_map` | worker | Tier assignment per chunk |

### Why gpu.forward\_wait is a separate span

vLLM queues the forward-pass kernels on a CUDA stream and returns before they
complete. LMCache's save path begins with a synchronous device-to-device copy
that blocks until the stream drains. Without separating this wait, `kv.store`
absorbs the entire GPU compute time and becomes incomparable with the kvbench
emulator (which has no real GPU). The connector drains the stream explicitly
before opening `kv.store` so each span measures only what its name says.

## Span format

Each line in the JSONL file is one closed span:

```json
{
  "trace_id": "chatcmpl-<uuid>",
  "span_id": "<12-hex>",
  "parent_id": "<12-hex> or null",
  "name": "gpu.copy_in",
  "ts": 1234567890.123456,
  "start_ms": 142.3,
  "dur_ms": 2.41,
  "thread": "MainThread",
  "attrs": {"tokens": 256}
}
```

`trace_id` matches the `chatcmpl-` ID returned to the client, so spans from
the KV trace can be joined with OTLP request spans from vLLM's own tracing.

## Files

| File | Purpose |
|---|---|
| `connector.py` | `TracingLMCacheConnector` — the vLLM connector class |
| `install.py` | Tracer setup and GPU connector hook installation |
| `otlp.py` | OTLP receiver for vLLM request-level spans |
