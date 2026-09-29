# Docker

```bash
docker build -t kvbench:latest .
docker compose -f deployment/docker/docker-compose.yml up -d
```

The image includes LMCache/Torch and no model weights. The single-node example
has a 2 GiB container memory limit, a 0.25 GiB LMCache CPU tier and a bounded
disk tier. Bind `/var/lib/lmcache` to the storage under test instead of the
example named volume.

`docker-compose.distributed.yml` now demonstrates **two independent combined
replicas**, not a simulated prefill/decode proxy. It exposes ports 8000 and 8001
and configures a shared Redis remote tier. This is an illustrative deployment;
remote-tier behavior must be validated against the installed LMCache connector.

No separate metrics port exists; scrape each replica's `/metrics`. Use unique
trace paths/volumes per replica. Per-rank CPU sizes multiply by TP × PP; the
adapter's aggregate guard rejects oversized configurations before allocation.
