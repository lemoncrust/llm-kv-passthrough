"""
vLLM KV connector that records kvbench-format spans around the real LMCache.

Load it with::

    vllm serve <model> --kv-transfer-config '{
      "kv_connector": "TracingLMCacheConnector",
      "kv_connector_module_path": "kvbench.vllm_trace.connector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {"trace_file": "/tmp/vllm-kv-trace.jsonl"}}'

Each vLLM process (scheduler, worker) writes ``<stem>.<pid><suffix>``.

Spans emitted, with real (not emulated) durations:

- storage ``io.*`` / ``lmcache.*``: the same hooks as ``kvbench serve``
  (worker, installed once LMCache has created its storage backends)
- ``kv.lookup``: time in the prefix-hit lookup (scheduler)
- ``gpu.copy_in`` / ``gpu.copy_out``: time vLLM waits on KV load / save (worker)
- ``kv.store``: a ``wait_for_save`` step that stores new chunks, with
  ``tokens``/``skip_leading``/``chunks`` attrs; ``gpu.copy_out`` nests inside
  it. Steps with nothing to store emit only ``gpu.copy_out``.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

# vLLM's input_processor rewrites request_id as "{external_req_id}-{uuid:.8}"
# before the scheduler sees it. Strip the suffix so spans share the same
# chatcmpl- trace_id as the OTLP request-trace spans.
_SUFFIX_RE = re.compile(r'-[0-9a-f]{8}$')

def _trace_id(request_id: str) -> str:
    return _SUFFIX_RE.sub('', request_id)

from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
)

from kvbench.trace import spans
from kvbench.vllm_trace.install import (
    install_engine_hooks,
    install_tracer,
    store_plan,
)

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.forward_context import ForwardContext
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class TracingLMCacheConnector(LMCacheConnectorV1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self._tracer = install_tracer(
            vllm_config.kv_transfer_config.get_from_extra_config("trace_file", None)
        )
        self._hooks_installed = False
        # Retrieval-side accumulators — initialised here so wait_for_layer_load
        # is safe even if called before start_load_kv (shouldn't happen but
        # defensive initialisation costs nothing).
        self._loading_req_ids: list[str] = []
        self._gpu_copy_in_t0: float | None = None
        self._gpu_copy_in_dur: float = 0.0
        self._kv_retrieve_t0: float = 0.0

    def _ensure_hooks(self) -> None:
        # post_init, which creates the storage backends, runs lazily inside
        # LMCache's adapter, so retry until the backends exist.
        if self._tracer is None or self._hooks_installed:
            return
        self._hooks_installed = bool(
            install_engine_hooks(self._lmcache_engine, self._tracer)
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        super().register_kv_caches(kv_caches)
        self._ensure_hooks()

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        """Starts the layerwise KV retrieval pipeline.

        Does NOT emit gpu.copy_in here — only disk→CPU happens in this call
        (layerwise_retrievers are set up and first two layers fetched from disk/CPU).
        The actual CPU→GPU copy happens per layer in wait_for_layer_load().
        We accumulate those timings there and emit one gpu.copy_in per request.
        """
        self._ensure_hooks()
        # Collect request IDs for cache-hit requests so wait_for_layer_load
        # can emit correctly-attributed spans.
        try:
            meta = self._get_connector_metadata()
            self._loading_req_ids = [
                _trace_id(r.req_id) for r in meta.requests
                if getattr(r.load_spec, "can_load", False)
            ]
        except Exception:
            self._loading_req_ids = []
        self._gpu_copy_in_t0: float | None = None
        self._gpu_copy_in_dur: float = 0.0
        self._kv_retrieve_t0: float = time.perf_counter()
        # Wrap super() with a context span so that child spans created on this
        # thread during start_load_kv (io.read, lmcache.batched_get etc.)
        # inherit the chatcmpl- trace_id. The span itself is named kv.load_init
        # and is intentionally short — it covers only disk→CPU initiation.
        # The full kv.retrieve span (disk→CPU + CPU→GPU) is emitted later in
        # wait_for_layer_load once all layers are loaded.
        tid = self._loading_req_ids[0] if len(self._loading_req_ids) == 1 else None
        if tid:
            with spans.span("kv.load_init", parent=None, trace_id=tid):
                super().start_load_kv(forward_context, **kwargs)
        else:
            super().start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Called once per attention layer during the forward pass.

        Each call advances the layerwise retriever by one layer, which copies
        one layer's KV chunks from CPU pinned memory into GPU HBM.
        This is the pure CPU→GPU copy — no disk I/O involved here.

        We accumulate wall-clock time across all 16 layer calls and emit a
        single gpu.copy_in span per request when the last layer is done
        (detected by the layerwise_retrievers list becoming empty after the
        call, or by a fixed layer counter on the engine).
        Only fires when there are real cache hits (layerwise_retrievers non-empty).
        """
        # Check if there is real work to do — skip if no cache hits.
        # layerwise_retrievers lives on self._lmcache_engine (the adapter),
        # not on self._lmcache_engine._lmcache_engine (the engine itself).
        adapter = getattr(self, "_lmcache_engine", None)
        has_work = bool(getattr(adapter, "layerwise_retrievers", None))
        if not has_work:
            return super().wait_for_layer_load(layer_name)

        t0 = time.perf_counter()
        super().wait_for_layer_load(layer_name)
        dur = time.perf_counter() - t0
        if self._gpu_copy_in_t0 is None:
            self._gpu_copy_in_t0 = t0
        self._gpu_copy_in_dur += dur

        # After the call, check if all layers are now loaded
        still_loading = bool(getattr(adapter, "layerwise_retrievers", None))
        if not still_loading and self._gpu_copy_in_t0 is not None:
            _tracer = spans.get_tracer()
            if _tracer and self._loading_req_ids:
                tid = self._loading_req_ids[0] if len(self._loading_req_ids) == 1 else None
                t_done = time.perf_counter()
                # gpu.copy_in — pure CPU→GPU layer copies (wall-clock accumulated)
                _tracer.record(
                    "gpu.copy_in",
                    start_mono=self._gpu_copy_in_t0,
                    end_mono=self._gpu_copy_in_t0 + self._gpu_copy_in_dur,
                    trace_id=tid,
                    layers=16,
                )
                # kv.retrieve — full retrieval pipeline: io.read + gpu.copy_in
                # (disk→CPU happened in start_load_kv, CPU→GPU completed here)
                # Comparable to kvbench kv.retrieve minus gpu.alloc which does
                # not exist in real vLLM (GPU KV cache is pre-allocated at startup)
                _tracer.record(
                    "kv.retrieve",
                    start_mono=self._kv_retrieve_t0,
                    end_mono=t_done,
                    trace_id=tid,
                )
            self._gpu_copy_in_t0 = None
            self._gpu_copy_in_dur = 0.0

    def wait_for_save(self) -> None:
        self._ensure_hooks()
        meta = self._get_connector_metadata()
        chunk_size = getattr(self._lmcache_engine, "_lmcache_chunk_size", 256)
        plan = store_plan(meta.requests, chunk_size)

        # Only use IDs from requests that are actually saving — filters out
        # requests in the batch that have nothing to store this step.
        saving_ids = [
            _trace_id(r.req_id) for r in meta.requests
            if getattr(r.save_spec, "can_save", False)
        ]
        # Fall back to all request IDs when save_spec info is unavailable
        if not saving_ids:
            saving_ids = [_trace_id(r.req_id) for r in meta.requests]
        tid = saving_ids[0] if len(saving_ids) == 1 else None

        # kv.store wraps the full wait_for_save() pipeline:
        #   GPU→CPU DMA (from_gpu, measured by gpu_connector hook) +
        #   LMCache bookkeeping + buffer-wait + disk-write submission.
        # gpu.copy_out is now emitted by the gpu_connector.from_gpu() hook
        # so it is not repeated here — just emit kv.store for the total.
        if plan is None:
            with spans.root("kv.store_noop", trace_id=tid):
                super().wait_for_save()
            return
        with spans.root("kv.store", trace_id=tid, **plan):
            super().wait_for_save()

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        with spans.root("kv.lookup", trace_id=_trace_id(request.request_id)) as sp:
            result = super().get_num_new_matched_tokens(request, num_computed_tokens)
            sp.set(num_tokens=request.num_tokens, hit_tokens=result[0])
        return result
