# Source map

- `core/config.py`: strict run configuration and scoped profile resolution.
- `core/profiles/catalog.yaml`: GPU/model data; `models.py` owns KV sizing.
- `core/synthetic.py`: deterministic byte-span tokenizer and conversation format.
- `servers/rate_scheduler.py`: bounded shared aggregate-rate scheduler.
- `servers/rate_server.py`: V1 request/KV lifecycle.
- `servers/app.py`: bounded HTTP API and metrics/warmup/manifest endpoints.
- `kv/`: real LMCache classic adapter and bounded payload provider.
- `trace/`, `metrics/`: file-level I/O evidence and serving observations.
- `cli/`: serving, profile inspection, warmup and experimental FIO export.

Older `combined`, `prefill`, `decode`, `proxy`, `gpu_engine`, `hotset`, and timing
modules are experimental reference implementations; the production factory
selects only `RateServer`.
