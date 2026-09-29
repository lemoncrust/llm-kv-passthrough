# External workload generators

The V1 API supports ordinary OpenAI chat requests with fixed output lengths,
SSE, `max_completion_tokens`/`max_tokens`, streamed usage, and
`usage.prompt_tokens_details.cached_tokens`. A real CLI/HTTP integration test
exercises this protocol through LMCache and disk. **NVIDIA AIPerf 0.13.0** was
also exercised against the server; it remains an external client dependency.

## Verified AIPerf invocation

Start `kvbench serve --config examples/aiperf-validation.yaml` for the small
storage smoke. Use a fresh disk directory and restart between cold comparisons.
This invocation uses the client's built-in tokenizer for
input synthesis and the emulator's usage fields for metrics:

```bash
aiperf profile --model llama-3.2-1b --tokenizer builtin \
  --url http://localhost:8000 --endpoint-type chat --streaming \
  --use-server-token-count --concurrency 2 --max-workers 1 --record-processors 1 \
  --request-count 6 --num-dataset-entries 6 --random-seed 42 \
  --synthetic-input-tokens-mean 128 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 16 --output-tokens-stddev 0 \
  --conversation-turn-mean 1 --conversation-turn-stddev 0 \
  --no-gpu-telemetry --no-server-metrics --ui none --artifact-dir ./aiperf-results
```

The validation used fresh cache state for each scenario, a 16-token prefill
slice and LMCache
chunk, an 8 MiB CPU tier and 512 MiB disk tier. The model was Llama 3.2 1B
(32 KiB KV/token), rates were 10,000 prefill / 200 decode tokens/s. Every
scenario completed six requests with zero errors:

| Workload | File writes | Written | File reads | Read | Cache-hit tokens |
|---|---:|---:|---:|---:|---:|
| ISL target 128, output 16, one turn | 56 | 28 MiB | 0 | 0 | 0 |
| ISL target 128, output 64, one turn | 74 | 37 MiB | 0 | 0 | 0 |
| ISL target 128, output 16, three turns | 58 | 29 MiB | 47 | 23.5 MiB | 928 |

The extra 48 output tokens × 6 requests × 32 KiB/token explain the **exact
9 MiB write increase**. Multi-turn history introduced real prefix reads and
reuse. This is a workload-sensitivity result, not a real-GPU performance claim.
The experiment's server/client process tree ran inside a 3 GiB, swap-disabled
systemd memory scope in addition to process limits.
A second run with the cache-usage fields enabled reproduced the same I/O counts;
AIPerf reported 928 cached prompt tokens and zero output-length mismatches for
the multi-turn scenario.

Start the server with `examples/rate-v1.yaml`. Point your installed NVIDIA
AIPerf or GenAI-Perf at `http://localhost:8000`, endpoint type `chat`, and the
served model ID `llama-3.2-1b`. Configure the client's tokenizer separately if
it needs a Hugging Face tokenizer; the served model ID is a profile identifier.

For a GenAI-Perf installation supporting these options:

```bash
genai-perf profile -m llama-3.2-1b \
  --service-kind openai --endpoint-type chat --url http://localhost:8000 \
  --streaming --synthetic-input-tokens-mean 2048 --output-tokens-mean 256 \
  --concurrency 4 --request-count 100
```

`scripts/genai_perf_test.sh` invokes this against an already running endpoint;
it neither installs tools nor suppresses failures. Consult the installed
client's help for version-specific workload/replay options.

## Workload experiments

Hold rates, rank layout, chunk size, tier sizes and initial cache state constant.
Vary one dimension at a time:

| Change | Expected direction |
|---|---|
| Longer unique prompts | More prefill work and full KV chunks |
| Longer output | More decode work and generated KV chunks |
| More conversation turns | More prefix reads and incremental KV writes |
| More shared prefix | More reuse and less new prefill/storage admission |
| More independent sessions | Larger working set and eventual eviction pressure |
| More concurrency | More shared scheduling/queueing; aggregate compute rates do not multiply |
| More TP ranks | More, smaller rank artifacts; replication can increase aggregate bytes |
| Smaller KV dtype | Proportionally smaller payloads |

Capture `/kvbench/manifest`, `/stats`, `/kvbench/state`, and JSONL I/O traces.
For closed-loop traffic, rate changes also change arrivals/session overlap. Use
open-loop arrivals when you need to separate those effects.

The emulator's usage counts synthetic tokens. Client re-tokenization of output
will differ. Treat storage byte/chunk counts and synthetic usage as the V1
accounting boundary, not semantic output quality or tokenizer equivalence.

Warmup and serving do not overlap. Record a measurement start timestamp or use
separate runs; trace-to-FIO does not automatically remove warmup events.
