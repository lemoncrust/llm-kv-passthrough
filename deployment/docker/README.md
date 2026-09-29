# Docker examples

See [the guide](../../docs/deployment/docker.md).

`docker-compose.yml` is one combined engine. `docker-compose.distributed.yml`
is two independent combined replicas, **not** prefill/decode disaggregation.
Both use bounded example memory sizes. Bind the cache directory to your storage
under test and validate real I/O before interpreting benchmark results.
