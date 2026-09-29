# Ansible examples

See [the guide](../../docs/deployment/ansible.md). The inventory contains combined
replicas; the templates render aggregate rates and LMCache storage settings.
Python 3.11+ and a mounted test filesystem are prerequisites.

`deploy.yml` installs from this repository (override `kvbench_pip_spec` to pin a
revision/wheel). `configure.yml` updates settings. `benchmark.yml` is a storage
smoke, not a performance harness. Review `teardown.yml` before use.
