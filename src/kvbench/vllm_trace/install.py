"""
Helpers for installing kvbench span tracing inside a vLLM process.

Kept separate from the connector so they can be tested without importing vLLM.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from kvbench.trace import spans
from kvbench.trace.lmcache_hooks import install_lmcache_trace
from kvbench.trace.spans import Tracer

logger = logging.getLogger(__name__)


def pid_trace_path(trace_file: str, pid: int | None = None) -> Path:
    """Return trace_file with the PID inserted before the suffix.

    Example: /tmp/trace.jsonl -> /tmp/trace.1234.jsonl

    vLLM runs the scheduler and workers in separate processes, each of which
    truncates its file on start, so every process needs its own path.
    """
    path = Path(trace_file)
    return path.with_name(f"{path.stem}.{os.getpid() if pid is None else pid}{path.suffix}")


def install_tracer(trace_file: str | None) -> Tracer | None:
    """Install the process-wide tracer writing to trace_file, or return None."""
    if not trace_file:
        return None
    return spans.install(pid_trace_path(trace_file))


def install_engine_hooks(adapter: Any, tracer: Tracer) -> list[str]:
    """Wrap LMCache storage backends and GPU connector with span hooks.

    Args:
        adapter: vLLM's LMCacheConnectorV1Impl (the worker-side adapter).
        tracer: The active span tracer.

    Returns:
        Names of the wrapped backends; empty when the engine is not ready yet.
    """
    engine = getattr(adapter, "lmcache_engine", None)
    if engine is None:
        return []
    try:
        wrapped = install_lmcache_trace(engine, tracer)
    except RuntimeError as e:
        logger.debug(f"KV trace hooks not installed yet: {e}")
        return []

    # Wrap to_gpu / from_gpu on the real GPU connector so each chunk's
    # CPU-pinned-to-GPU-HBM copy is recorded as a gpu.copy_in span, and each
    # GPU-to-CPU copy as gpu.copy_out. to_gpu is synchronous (it calls
    # load_stream.synchronize() before returning), so wall-clock timing is accurate.
    gpu_connector = getattr(engine, "gpu_connector", None)
    if gpu_connector is not None and not getattr(gpu_connector, "_kvtrace_hooked", False):
        orig_to_gpu   = gpu_connector.to_gpu
        orig_from_gpu = gpu_connector.from_gpu

        def traced_to_gpu(memory_obj, start: int, end: int, **kwargs):
            with spans.span("gpu.copy_in", tokens=end - start):
                return orig_to_gpu(memory_obj, start, end, **kwargs)

        def traced_from_gpu(memory_obj, start: int, end: int, **kwargs):
            with spans.span("gpu.copy_out", tokens=end - start):
                return orig_from_gpu(memory_obj, start, end, **kwargs)

        gpu_connector.to_gpu   = traced_to_gpu
        gpu_connector.from_gpu = traced_from_gpu
        gpu_connector._kvtrace_hooked = True
        logger.info("KV trace: gpu_connector.to_gpu/from_gpu wrapped")
        wrapped.extend(["gpu.copy_in", "gpu.copy_out"])

    return wrapped


def store_plan(requests: Any, chunk_size: int) -> dict[str, int] | None:
    """Summarise what one wait_for_save step will store.

    Mirrors LMCache's skip rules: requests that cannot save, or whose prompt
    is fully covered by the cache hit, store nothing, and only full chunks are
    stored. Returns None when the step stores nothing.
    """
    tokens = skip_leading = chunks = n = 0
    for req in requests:
        spec = req.save_spec
        if spec is None or not spec.can_save:
            continue
        total = len(req.token_ids)
        if spec.skip_leading_tokens == total:
            continue
        skip = spec.skip_leading_tokens // chunk_size * chunk_size
        end  = total // chunk_size * chunk_size
        if end <= skip:
            continue
        tokens       += end
        skip_leading += skip
        chunks       += (end - skip) // chunk_size
        n            += 1
    if not n:
        return None
    return {"tokens": tokens, "skip_leading": skip_leading, "chunks": chunks, "requests": n}
