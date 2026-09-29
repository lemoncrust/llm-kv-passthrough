# Installation

Python 3.11+; Linux is the tested platform. Install from this repository:

```bash
git clone https://github.com/wvaske/llm-kv-passthrough.git
cd llm-kv-passthrough
python -m pip install -e '.[lmcache]'
kvbench version
kvbench serve --config examples/rate-v1.yaml
```

LMCache 0.5.4 and Torch are required to serve; no inference GPU or weights are
required. `.[dev]` adds test/lint dependencies, `.[docs]` adds the MkDocs tools.
There are no independent `redis`/`s3` extras in KV-Bench; LMCache owns backends.

For tests use `bash scripts/test-safe.sh tests/` with an installed `.venv`, or
set `PYTHON` to your interpreter. This imposes a process memory limit.
