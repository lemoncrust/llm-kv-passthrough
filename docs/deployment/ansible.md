# Ansible starting point

The examples deploy independent **combined V1 engines**, one per host, with a
Python 3.11+ virtual environment, LMCache, rendered configs and a systemd unit.
They are not a tested fleet-deployment guarantee or a disaggregated topology.

```bash
ansible-playbook -i deployment/ansible/inventory/example.yml deployment/ansible/playbooks/deploy.yml
```

Mount storage separately; the playbook creates the configured cache directory
but does not configure NFS or format devices. A shared local-disk mount does not
create shared LMCache discovery. The default systemd memory cap is 2 GiB;
reconcile it with your aggregate per-rank CPU pools and runtime overhead.

Variables: `kvbench_prefill_tps`, `kvbench_decode_tps`, `kvbench_model`,
`kvbench_gpu`, `kvbench_tp_size`, `kvbench_pp_size`, `kvbench_memory_limit`,
`lmcache_max_local_cpu_size_gb`, `lmcache_local_disk_path`,
`lmcache_max_local_disk_size_gb`, and optional `lmcache_remote_url`.

`configure.yml` renders updates and restarts. `benchmark.yml` is explicitly a
connectivity/storage smoke test; use an external workload generator for timing.
