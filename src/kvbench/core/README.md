# Core

Run configuration is `KVBenchConfig` in `config.py`. Model/GPU catalogs live in
`profiles/catalog.yaml`; per-run catalogs can extend/override them without
mutating the globals used by offline tools. `models.py` separates KV dtype from
weight dtype and computes validated TP/PP byte layouts.

`synthetic.py` provides the production V1 tokenizer/formatter. `tokens.py` and
`latency.py` retain older offline utilities; their key formatter is not the
production LMCache key implementation. LMCache owns production cache identity.

See [configuration](../../../docs/getting-started/configuration.md) and
[profile semantics](../../../docs/architecture/gpu-emulation.md).
