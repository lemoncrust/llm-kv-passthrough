# V1 architecture

```text
External workload (AIPerf / GenAI-Perf / HTTP client)
    → bounded HTTP admission and synthetic conversation tokenization
    → RateServer request lifecycle
    → shared RateScheduler compute rounds
    → LMCacheStack logical TP/PP ranks
    → real LMCache chunking, hashing, CPU/disk/remote storage
```

The scheduler is intentionally naive. Its aggregate round cost is
`prefill_tokens / prefill_rate + decode_tokens / decode_rate`. Requests blocked
on KV I/O do not submit further compute. Ready requests share rounds; admission
and pending queues are bounded. This models a plausible shared compute budget,
not the batching algorithm of a particular framework.

Lookups/pins occur on every rank. Retrieval must complete before prefill/decode.
Prefill slices submit newly complete chunks; decode computes the previously
sampled token and submits newly complete chunks. The last output token has not
yet acquired KV. Full chunks only are stored. Synthetic generated text included
in a later conversation produces the same cached prefix.

`MockGPUConnector` fills LMCache's allocated chunk directly. There are no
prompt-sized stand-in tensors. Final device copies are omitted. Rank API calls
are serialized; LMCache backend workers retain their own I/O concurrency.
The whole lookup/retrieve/unpin lease is serialized too, so concurrent requests
cannot pin one another's entire shared CPU staging pool while waiting to load.

Only combined serving is supported. There is no production proxy, service
registry, native serving-framework scheduler, or GPU block allocator. Legacy
server modules remain experimental reference code and are not exported or
selected by the application. See the [plan](../V1_PLAN.md) for levels 2 and 3.
