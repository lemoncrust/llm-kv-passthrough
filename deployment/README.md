# Deployment

V1 runs one combined engine per process. GPU/model profiles and explicit rates
describe the intended system; LMCache's own file/environment controls storage.

- `docker/docker-compose.yml`: one memory-limited engine, CPU + disk tiers.
- `docker/docker-compose.distributed.yml`: two independent combined replicas,
  with illustrative shared Redis configuration. No prefill/decode proxy.
- `ansible/`: fleet starting point, rendered rate/LMCache configs and systemd.

See [Docker](../docs/deployment/docker.md) and
[Ansible](../docs/deployment/ansible.md). These deployment artifacts are not
substitutes for validating the storage under test. CPU/local-disk integration
tests and a real CLI/HTTP smoke test are provided in `tests/integration/`.
