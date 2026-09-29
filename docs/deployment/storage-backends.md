# Storage under test

KV-Bench does not implement storage backends. Select storage through LMCache
0.5.4's application configuration.

- **Local NVMe or mounted filesystem:** set `local_disk: file:///mount/cache/`
  and `max_local_disk_size`. Make the CPU tier smaller than the working set
  when measuring storage rather than CPU-only hits.
- **Cross-process sharing:** use a suitable LMCache remote connector, such as
  Redis, rather than expecting `local_disk` indexes to discover another
  process's files. Give replicas compatible model definitions and rank layouts.
- **Other remote backends:** install/configure their dependencies according to
  the pinned LMCache release and validate them independently. They are outside
  this repository's CPU/local-disk integration matrix.

Do not format devices as part of a benchmark launch. Mount an existing test
filesystem and give its path to LMCache. Confirm artifact bytes and traced I/O
before interpreting a benchmark result. `max_local_cpu_size` and
`max_local_disk_size` are GiB per logical rank.
