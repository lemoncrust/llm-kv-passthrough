# Quickstart

```bash
kvbench serve --config examples/rate-v1.yaml
```

In another terminal:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama-3.2-1b","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":32,"stream":true,"stream_options":{"include_usage":true}}'
curl http://localhost:8000/kvbench/manifest
curl http://localhost:8000/stats
curl http://localhost:8000/metrics
```

Short prompts may not complete an LMCache chunk. Use longer prompts/output or
multi-turn histories to exercise storage, rather than expecting a tiny greeting
to create disk artifacts. Put returned assistant content into subsequent
messages to reuse generated KV.

See [benchmarking](../benchmarking/genai-perf.md) and
[configuration](configuration.md) for rates, workload controls and limits.
