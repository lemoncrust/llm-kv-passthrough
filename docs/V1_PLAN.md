# Version 1: workload-sensitive KV storage exerciser

## Architect's intent (2026-09-29)

The product exercises a real KV manager, initially LMCache, from an external
query workload such as NVIDIA AIPerf. Its primary output is the resulting
storage workload. Changes in session length, prompt/output tokens per turn,
number of turns, prefix reuse, and concurrency must produce credible changes
in KV reads/writes, sizes, timing, and eviction pressure. Matching a particular
real inference run is secondary.

Three fidelity levels:

1. **V1 / rate-based:** configured prefill tokens/s, decode tokens/s, and model
   rank layout; a bounded naive scheduler; real LMCache; correct KV bytes and
   LMCache-defined chunks; sensible operation ordering.
2. **Later / resource-aware:** hardware-dependent coupled prefill/decode costs,
   realistic batching, scheduling, memory pressure, framework behavior.
3. **Future / trace-driven:** Chakra-style dependency/event replay, with
   emulated compute steps and actual KV-manager operations.

No architect answers are required to finish V1. The decisions below are the
implementation assumptions; open questions are recorded for later review.

## V1 decisions

- One combined serving engine per process. Multiple independent processes are
  replicas, not HTTP workers sharing one simulated GPU. Reject unsupported
  proxy/prefill/decode deployment modes instead of returning simulated success.
- Rate configuration is authoritative, per emulated engine/rank group, not per
  request or per rank. A FIFO-admitted, round-robin step scheduler shares one
  compute lane; prefill and decode compete for that lane. Prefill work is
  chunked, decode advances one token per scheduled sequence. Rates describe
  aggregate token budgets, not hardware-calibrated claims.
- Bound admission, pending requests, input/context/output size, output queues,
  host-side KV operations, and per-operation payload memory. Slow clients and
  storage backpressure their requests. Compute cannot run before retrieval.
- Look up/pin across all ranks, retrieve the available prefix, then release
  pins in a cancellation-safe scope. A short retrieve fails explicitly rather
  than counting unavailable KV as a hit. Infrastructure errors are not misses.
- Store completed prompt chunks after each prefill slice. Generated decode
  tokens extend the sequence and newly complete chunks are stored too. Do not
  store trailing partial chunks. Keep the final sampled token uncomputed, as
  in autoregressive execution; it is computed on the next turn if needed.
- Synthetic text tokenization is deterministic and prefix-stable at complete
  token boundaries. The formatter includes the assistant-generation prefix;
  generated text fed back into a later conversation reproduces cached tokens.
  Token counts describe emulator tokens, not a particular model tokenizer.
- Hardware/model catalogs live in YAML. Custom catalogs extend/override
  bundled definitions per configuration instance, without mutating globals.
  KV dtype is separate from weights. Validate TP head partition/replication and
  PP layer partition. V1 models byte geometry, not native attention kernels.
- LMCache owns chunking, hashing, tier capacities, and physical I/O. Emulation
  buffers are bounded and never prompt-sized. Rank engines are serialized at
  their API boundary; LMCache retains its own backend I/O concurrency.
- Support the classic synchronous LMCache engine API. Reject unsupported
  async/layerwise/PD/device-only options. Pin the supported LMCache version.
- Report actual request/queue/compute/KV timing and submitted-vs-observed
  storage counters separately. Publish the resolved run configuration and
  profile definitions. Trace physical file operations with rank identity.
- Keep analytical timing utilities as experimental offline tools; do not use
  their old calibration claims as V1 acceptance criteria. Full-speed mode
  skips only scheduler sleeps, preserving the same request/KV lifecycle.

## Implementation sequence

1. File-backed catalogs, strict configuration, bounded defaults, example run.
2. Shared rate scheduler and unified serving lifecycle with decode KV growth.
3. Bounded LMCache payload generation, rank layout, cancellation and shutdown.
4. API/CLI/metrics integration; truthful supported deployment and documentation.
5. Differential workload tests: input length, output length, turns, prefix
   reuse, concurrency, chunk size, rank count, and KV dtype. Real LMCache disk
   tests validate artifacts/bytes/read paths, not just fake-stack counters.

## Acceptance criteria

- Cold/warm/shared-prefix requests exercise real lookup/store/retrieve paths.
- Larger input/output histories create predictably more full KV chunks.
- Follow-up turns reuse both prompt and computed generated KV.
- Chunk bytes equal configured rank geometry; TP/PP produce the expected
  number and size of rank artifacts, including replicated KV heads.
- A shared scheduler bounds aggregate simulated compute throughput; storage
  delays cannot result in precomputed token bursts.
- Cancellation, partial retrieval, startup failure, queue exhaustion, and
  shutdown leave no pins, rank engines, or unowned background operations.
- One config file can select/create profiles and rates without changing code.
- Existing numerical utilities and supported APIs have regression coverage;
  unsupported legacy behavior fails explicitly with migration guidance.
- All dev-host tests use process memory limits and small LMCache CPU tiers.
  Larger experiments may use GB10/Frame. No unbounded local memory tests.

## Deferred architect questions, by domain

### Product and fidelity
1. Which storage metrics and change ratios should eventually carry numerical
   acceptance tolerances? V1 validates exact byte/chunk invariants and trends.
2. Which reference workloads should become release fixtures (AgentX, synthetic
   chat, RAG, captured AIPerf sessions)?
3. Should projected hardware estimates remain distributed with measured ones?

### Rates and scheduling
4. Should future profiles express decode rate per active sequence as well as
   aggregate throughput? V1 uses aggregate engine rates explicitly.
5. Which fairness/priority/preemption policies matter for fidelity level 2?
6. What are the first required hardware/model/framework validation pairs?

### KV managers and data
7. After classic LMCache, which integration comes next: async/layerwise
   LMCache, native framework KV, Dynamo KVBM, or Mooncake?
8. Should payloads eventually reproduce numerical/compression distributions,
   or is deterministic high-entropy data sufficient?
9. Which native KV layouts and unusual hybrid/recurrent states must be modeled
   beyond byte geometry?

### Distributed topology
10. Is disaggregated prefill/decode the next topology after combined replicas?
11. Must ranks become processes/hosts to reproduce real transport contention?
12. Is heterogeneous prefill/decode rank reshaping required?

### Workloads and APIs
13. When should real tokenizer/chat-template support replace or supplement
    synthetic tokenization? Should token-ID trace input be first?
14. How should hidden reasoning, speculative tokens, branching, tools, and
    multimodal inputs contribute to the storage workload?
15. Which additional AIPerf/API options must be implemented beyond ordinary
    chat, streamed usage, fixed output lengths, and multi-turn histories?

### Observability and trace replay
16. Is FIO export a supported product output or only an approximation aid?
17. Which Chakra schema and execution boundary should level 3 target?
18. What trace retention, cross-host clock correlation, and artifact format
    should be standardized for external benchmark orchestration?

### Release and compatibility
19. Which deployment forms are required beyond a single process/container?
20. What compatibility guarantees should apply to old experimental timing
    flags, Python utilities, profile IDs, and trace formats?

## Validation results

Implemented the rate-based production path, declarative catalogs, rank byte
geometry, bounded LMCache payload generation, prompt/decode chunk lifecycle,
HTTP limits/usage, real disk workload tests, and truthful deployment examples.
The old analytical modules remain experimental and cannot select a production
serving mode. Deferred topology modes fail at startup.

Validation uses `bash scripts/test-safe.sh tests/` with a 4 GiB virtual-memory
cap, `MALLOC_ARENA_MAX=2`, and one numerical-library thread. The live-server
child also has a 4 GiB cap. A measured full-suite run used approximately
954 MiB peak RSS as reported by `/usr/bin/time -v`, and no swap. This is a
reported per-process maximum, not a summed parent/child memory budget.
No unbounded tests ran on dev; GB10/Frame were
not required for the small real-manager fixtures.

The real storage matrix checks exact files/bytes for TP/PP, head replication,
FP8 KV, independent rank entropy, cold input/output growth, generated-prefix
reuse, disk reads after CPU eviction, and retrieval of prefixes larger than
the CPU tier. Failure tests cover cancellation during a blocking lookup,
partial retrieval, startup failure, queue limits, timeouts and disconnects.

Final full acceptance run: **556 passed, 1 skipped** (the missing-LMCache hint
test skips because LMCache is installed), including deployment rendering.
The built wheel was
installed into an isolated target and loaded all **14 model / 17 GPU profiles**.

Ruff, strict mypy, strict MkDocs, sdist/wheel building and wheel catalog loading
are additional release checks. NVIDIA AIPerf **0.13.0** was additionally run
against fresh real servers in three scenarios (six requests each, no errors).
Increasing output from 16 to 64 tokens increased physical writes from 28 to
37 MiB, exactly the additional token/KV volume. Three-turn histories produced
928 cache-hit tokens and 23.5 MiB of real file reads. The whole experiment
process tree had a 3 GiB systemd memory cap and swap disabled. Commands/results
are recorded in `benchmarking/genai-perf.md`; detailed local artifacts are in
`/tmp/opencode/kvbench-v1-aiperf-validation/`.
The repeat run after adding cached-token usage is in
`/tmp/opencode/kvbench-v1-aiperf-cache-usage/`; AIPerf independently reported
928 cached prompt tokens, six completed requests, no errors, and no output
length mismatches for the multi-turn scenario. Runnable small-tier configs are
in `examples/aiperf-validation.yaml` and `examples/lmcache-aiperf-validation.yaml`.
Final confirmation after serializing prefix-load leases and isolating payload
identity by model/rank reproduced the same three results; those artifacts are
in `/tmp/opencode/kvbench-v1-aiperf-final/`. Repeated-cancellation and CLI
migration/rate-wiring regressions are included in the final passing suite.

Docker image/fleet execution remains untested. CI runs contracts/docs/build on
changes and offers the real-manager matrix via manual dispatch.

## Original review question register and disposition

This preserves the original architecture review questions in domain order.
“Assumed” means a reversible V1 implementation decision, not an architect
answer. The deferred questions above summarize the remaining discussion.

### A. Purpose and acceptance
- **A1 — Primary decision supported?** Answered: effects of query workload
  changes on KV-manager/storage load.
- **A2 — Which outputs must match?** Answered: credible workload-to-I/O changes
  matter more than single-run real/emulated performance parity.
- **A3 — Numerical error bounds?** Deferred; V1 uses exact chunk/byte invariants
  and directional workload tests, not hardware-parity percentages.
- **A4 — CPU-only versus real device operations?** Assumed CPU-only V1; real
  hardware is available for testing but no inference/device copy is required.
- **A5 — Projected hardware?** Deferred to resource-aware fidelity; rates are
  explicit in V1 and hardware labels are descriptive.

### B. Serving framework
- **B1 — Real scheduler or approximation?** Answered for V1: a reasonable naive
  scheduler is sufficient; real framework integration is not first priority.
- **B2 — First framework/version?** Deferred to fidelity level 2.
- **B3 — Which framework components remain real?** Assumed only the KV manager
  and underlying storage are real in V1.
- **B4 — Upstream forks/patches acceptable?** Deferred.
- **B5 — Common versus framework-specific behavior?** Deferred; V1 has an
  explicitly documented generic aggregate-rate scheduler.

### C. Timing/calibration
- **C1 — Measured, analytical or trace costs?** Answered: explicit rates first,
  resource-aware scheduling second, Chakra-like replay third.
- **C2 — Raw GB10 captures/fitting tools?** Still deferred; no V1 dependency on
  old fitted calibration claims.
- **C3 — Calibration matching/extrapolation rules?** Deferred to level 2.
- **C4 — Borrowed/adjusted anchors?** Retained as labeled offline experiments,
  not used to pace V1.
- **C5 — Host/I/O slower than model?** Assumed real backpressure; observed
  serving rate may be lower than the configured compute budget.
- **C6 — Full speed stress or virtual simulation?** Assumed stress mode with
  compute sleeps disabled and the same KV lifecycle, not virtual-time replay.

### D. Hardware/model semantics
- **D1 — Authoritative model source?** Assumed curated file catalogs plus user
  overrides; pinned native model-metadata import is deferred.
- **D2 — Required architectures?** V1 supports declarative GQA/MQA, cache-bearing
  hybrid layer counts and replicated MLA latent byte geometry.
- **D3 — Independent precisions?** Implemented separate KV/weight dtype.
- **D4 — MoE residency/expert traffic?** Deferred to level 2; V1 rates describe
  the chosen model/rank group's operating point.
- **D5 — Impossible hardware configurations?** V1 validates KV partitioning and
  host allocation, not physical model/HBM feasibility (explicit rates govern).

### E. Configuration
- **E1 — Distribution of profiles?** Assumed bundled YAML plus scoped user
  catalogs; no edits to Python required.
- **E2 — Precedence?** Assumed CLI over a YAML base, otherwise env over defaults;
  paths in a YAML file are relative to that file.
- **E3 — Reject unknown/unsupported configuration?** Implemented fail-fast V1
  boundary with migration documentation.
- **E4 — Reproducible manifest?** Implemented resolved definitions, topology,
  configuration and dependency versions at `/kvbench/manifest`.

### F. KV manager and payloads
- **F1 — First manager?** Assumed real LMCache classic API, pinned 0.5.4.
- **F2 — Which LMCache modes?** Classic synchronous whole chunks; async,
  layerwise and native PD/device modes deferred and rejected.
- **F3 — Payload fidelity?** Assumed correct rank bytes and deterministic high
  entropy, not numerical model state/compressibility distributions.
- **F4 — Real memory copies?** Fill LMCache's allocated object directly; omit
  the final GPU copy. Actual lower-tier I/O remains real.
- **F5 — Partial retrieval/failure policy?** Explicit request error, never a
  falsely successful cache hit. Recompute policy is deferred.
- **F6 — Store decode KV?** Implemented as tokens are computed and complete a
  chunk; the final sampled output token remains uncomputed.

### G. Scheduling and memory
- **G1 — Memory-driven admission/preemption?** Deferred to level 2; V1 bounds
  request counts, context/body/output lengths and host allocations.
- **G2 — What is the hot set?** Not used by V1; real LMCache owns CPU retention.
- **G3 — Scheduling features?** Assumed FIFO admission and round-robin ready
  steps on a shared rate budget, without speculative/priority scheduling.
- **G4 — Required scale?** No numerical target supplied; explicit bounds expose
  host constraints rather than pretending throughput is unlimited.
- **G5 — Slow clients?** Assumed backpressure; no unbounded token queues.

### H. Distribution
- **H1 — Initial topology?** Assumed combined engines and independent replicas;
  logical TP/PP KV layout supported. Disaggregation is deferred.
- **H2 — Rank processes or logical ranks?** Assumed logical engines in one
  process with serialized manager API calls and real backend worker I/O.
- **H3 — Prefill/decode handoff?** Deferred; the old fake proxy is not selectable.
- **H4 — Heterogeneous ranks/layout conversion?** Deferred.
- **H5 — Routing/discovery/failover policy?** Deferred; external clients can
  choose independent combined endpoints.

### I. Workloads and API
- **I1 — Clients?** Answered: tools such as NVIDIA AIPerf drive query workloads;
  V1 implements ordinary OpenAI-shaped chat/streamed usage.
- **I2 — Real tokenizer?** Assumed deterministic synthetic tokenization with
  explicit byte ratio and prefix-stable multi-turn generation.
- **I3 — Output budget/reasoning?** Assumed requested fixed completion length;
  reasoning/speculation and captured token-ID replay are deferred.
- **I4 — API feature scope?** One text completion; unsupported stop/multiple
  choices/tool execution/structured output are rejected, sampling is synthetic.

### J. Observability and replay
- **J1 — Authoritative measurement?** Actual traced storage I/O and independent
  byte/chunk oracles; server/client timing remains observed, not claimed parity.
- **J2 — Correlation?** Implemented rank identity in storage events; full causal
  request/step replay identifiers remain future work.
- **J3 — FIO fidelity?** Deferred; existing export is explicitly experimental.
- **J4 — Warmup/measurement?** Enforce non-overlap, report submitted warmup
  bytes and eviction probe; separate run windows remain client-controlled.

### K. Deployment and release
- **K1 — Environments?** Linux/Python 3.11+, bounded process/container examples;
  actual fleet and additional platforms need independent validation.
- **K2 — Hardware acceptance matrix?** Deferred; V1 storage fixtures run without
  inference hardware and directly test workload sensitivity.
- **K3 — Compatibility?** Assumed migration may reject ineffective old modes;
  ordinary chat, inspection, warmup and tracing retain supported entry points.
- **K4 — Obsolete code/docs?** Historical plans are labeled; old server/timing
  experiments remain reference code outside the production factory/export.
- **K5 — Smallest complete release?** Rate-based combined engine, real LMCache,
  file-configured rank geometry and passing workload/storage invariants.
