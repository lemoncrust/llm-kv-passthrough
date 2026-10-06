"""
Request tracing and KV workload derivation.

Every processing step of a request (tokenize, KV lookup/retrieve/store,
emulated compute, decode steps) and the storage operations LMCache
performs for it are recorded as spans — named intervals with parent ids —
written as JSONL (see `kvbench.trace.spans`). Requests are identified by
the client's X-Request-ID / X-Correlation-ID when present
(`kvbench.trace.identity`). The resulting trace is also the ground truth
for deriving representative storage workloads — see `kvbench trace2fio`.
"""

from kvbench.trace.lmcache_hooks import install_lmcache_trace
from kvbench.trace.spans import Span, Tracer

__all__ = ["Span", "Tracer", "install_lmcache_trace"]
