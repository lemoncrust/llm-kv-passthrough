# vLLM KV Tracing Connector

A drop-in vLLM KV connector that records every storage operation LMCache
performs during real inference. It captures disk reads, GPU copies, and cache
lookups as time-stamped spans in the same JSONL format kvbench uses. The
resulting traces can be compared directly against a kvbench emulation run to
measure how faithfully the emulator reproduces real vLLM behaviour.

---

## Step 1: Install kvbench in the vLLM environment

The connector is part of the kvbench package. Install it inside the vLLM
virtual environment so vLLM can import it at startup.

```bash
pip install "kvbench[lmcache]"
```

---

## Step 2: Start vLLM with the tracing connector

Pass the connector through `--kv-transfer-config` and set
`LMCACHE_CONFIG_FILE` to the same LMCache config you use with kvbench.

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

Each vLLM process (scheduler and workers) writes its own file so they do not
overwrite each other. For example, with PID 1234 the file becomes
`/path/to/vllm-kv-trace.1234.jsonl`.

---

## Step 3: Send requests

Send requests to vLLM as normal. The connector records spans for every request
in the background without changing any responses.

---

## Step 4: Collect the trace files

After the run, combine the per-process files:

```bash
cat /path/to/vllm-kv-trace.*.jsonl > combined.jsonl
```

Or pass the directory to compare_report.py and it will find and merge them
automatically.

---

## What gets recorded

All spans use real wall-clock durations.

| Span | Side | What it measures |
|---|---|---|
| `kv.lookup` | scheduler | Prefix-hit check; `hit_tokens` attr shows how many tokens matched |
| `kv.load_init` | worker | LMCache disk-to-CPU fetch (wraps start\_load\_kv) |
| `gpu.copy_in` | worker | CPU-pinned-buffer to GPU-HBM copy per chunk; synchronous |
| `gpu.forward_wait` | worker | CPU blocked on CUDA stream until the forward pass completes |
| `kv.store` | worker | GPU-to-CPU copies plus disk-write submission |
| `gpu.copy_out` | worker | One span per chunk inside kv.store |
| `io.read` | LMCache thread | Physical file read |
| `io.write` | LMCache thread | Physical file write; runs asynchronously after kv.store returns |
| `lmcache.put` | worker | Chunk hand-off to storage backend |
| `lmcache.get` | worker | Chunk retrieval from storage backend |
| `lmcache.block_map` | worker | Tier assignment per chunk |

### Why gpu.forward\_wait is a separate span

vLLM queues the forward-pass kernels on a CUDA stream and returns before they
finish. LMCache's save path starts with a device copy that blocks until the
stream drains. Without separating this wait, kv.store would absorb the entire
GPU compute time and become incomparable with the kvbench emulator, which has
no real GPU. The connector drains the stream explicitly before opening
kv.store so each span measures only what its name says.

---

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

---

## Files

| File | Purpose |
|---|---|
| `connector.py` | TracingLMCacheConnector, the vLLM connector class |
| `install.py` | Tracer setup and GPU connector hook installation |
| `otlp.py` | OTLP receiver for vLLM request-level spans |
