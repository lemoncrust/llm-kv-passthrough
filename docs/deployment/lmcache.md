# LMCache deployment

Install `.[lmcache]` and pass LMCache's own YAML file through
`kv.lmcache_config_file` or `--lmcache-config`. Without a file its `LMCACHE_*`
environment variables are used. KV-Bench supplies a safe 0.25 GiB CPU-pool
default only when no CPU size was specified, and keeps cyclic GC enabled.

See `examples/lmcache-local-disk.yaml` and the
[adapter contract](../architecture/connectors.md). Tier sizes are per rank;
chunk and aggregate CPU allocation guards are checked before engine creation.

Only classic synchronous operation is supported. Configuration pass-through
does not imply support for arbitrary LMCache execution modes. A real shared
remote backend, not merely a shared local-disk path, is needed for independent
processes to discover one another's cache entries.
