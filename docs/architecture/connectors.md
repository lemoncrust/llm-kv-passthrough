# KV manager adapter

The production adapter is `LMCacheStack` in `src/kvbench/kv/`. It drives the real
LMCache 0.5.4 classic engine API. `KVStack` is the boundary for future managers.

Servers operate on token sequences using lookup/pin, retrieve, store and unpin.
The adapter validates rank geometry, checks CPU-pool/chunk allocation guards,
and serializes blocking calls. Cancellation waits for an already-running call
before cleanup. Every created rank engine is destroyed on close or partial
startup failure.

LMCache itself owns prefix hashing, chunk identity, backend allocation,
tiering, eviction and physical storage I/O. KV-Bench fills manager-allocated
payloads with deterministic high-entropy bytes; no inference tensors or weights
are computed. Final device transfer is omitted at fidelity level 1.

Only whole-chunk synchronous operations are supported. Async loading,
layerwise/blending, PD, controller, hidden-state and device-specific paths are
rejected rather than silently mapped onto the wrong protocol. Remote-backend
compatibility depends on LMCache and its dependencies; CPU/disk paths are the
integration-tested matrix in this repository.

KVBM remains unimplemented and is rejected. Earlier notes about its CUDA-only
worker are historical research, not a current upstream compatibility guarantee.
