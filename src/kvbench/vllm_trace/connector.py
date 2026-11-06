"""
vLLM KV connector that records kvbench-format spans around the real LMCache.

Load it with:

    vllm serve <model> --kv-transfer-config '{
      "kv_connector": "TracingLMCacheConnector",
      "kv_connector_module_path": "kvbench.vllm_trace.connector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {"trace_file": "/path/to/vllm-kv-trace.jsonl"}}'

Each vLLM process (scheduler, worker) writes <stem>.<pid><suffix> so
concurrent processes do not clobber each other.

Spans emitted (all with real, not emulated, durations):

  kv.lookup        prefix-hit check on the scheduler side
  kv.load_init     disk-to-CPU fetch initiated by start_load_kv
  gpu.copy_in      CPU-pinned-buffer to GPU-HBM copy (from gpu_connector.to_gpu)
  gpu.forward_wait CPU blocked on CUDA stream until the forward pass drains
  kv.store         LMCache save: GPU-to-CPU copies + disk-write submission
  gpu.copy_out     one span per chunk inside kv.store (from gpu_connector.from_gpu)
  io.read          physical file read on an LMCache worker thread
  io.write         physical file write on an LMCache worker thread (async)
  lmcache.*        LMCache internal operations (put, get, contains, block_map)
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

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
        self._loading_req_ids: list[str] = []
        self._gpu_copy_in_t0: float | None = None
        self._gpu_copy_in_dur: float = 0.0
        self._kv_retrieve_t0: float = 0.0

    def _ensure_hooks(self) -> None:
        if self._tracer is None or self._hooks_installed:
            return
        self._hooks_installed = bool(
            install_engine_hooks(self._lmcache_engine, self._tracer)
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        super().register_kv_caches(kv_caches)
        self._ensure_hooks()

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        self._ensure_hooks()
        try:
            meta = self._get_connector_metadata()
            self._loading_req_ids = [
                _trace_id(r.req_id) for r in meta.requests
                if getattr(r.load_spec, "can_load", False)
            ]
        except Exception:
            self._loading_req_ids = []
        self._gpu_copy_in_t0 = None
        self._gpu_copy_in_dur = 0.0
        self._kv_retrieve_t0 = time.perf_counter()
        # Open kv.load_init so child spans (io.read, lmcache.batched_get)
        # inherit the request's trace_id from this thread's context.
        tid = self._loading_req_ids[0] if len(self._loading_req_ids) == 1 else None
        if tid:
            with spans.span("kv.load_init", parent=None, trace_id=tid):
                super().start_load_kv(forward_context, **kwargs)
        else:
            super().start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # Accumulate CPU-to-GPU copy time across all attention layers and emit
        # a single gpu.copy_in span once the last layer finishes loading.
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

        still_loading = bool(getattr(adapter, "layerwise_retrievers", None))
        if not still_loading and self._gpu_copy_in_t0 is not None:
            _tracer = spans.get_tracer()
            if _tracer and self._loading_req_ids:
                tid = self._loading_req_ids[0] if len(self._loading_req_ids) == 1 else None
                t_done = time.perf_counter()
                _tracer.record(
                    "gpu.copy_in",
                    start_mono=self._gpu_copy_in_t0,
                    end_mono=self._gpu_copy_in_t0 + self._gpu_copy_in_dur,
                    trace_id=tid,
                    layers=16,
                )
                _tracer.record(
                    "kv.retrieve",
                    start_mono=self._kv_retrieve_t0,
                    end_mono=t_done,
                    trace_id=tid,
                )
            self._gpu_copy_in_t0 = None
            self._gpu_copy_in_dur = 0.0

    def wait_for_save(self) -> None:
        # Drain the compute stream before opening kv.store so that gpu.forward_wait
        # captures the GPU compute time and kv.store reflects only LMCache work.
        # Without this split, the GPU wait inflates kv.store and makes it
        # incomparable with kvbench's kv.store (which has no real GPU).
        self._ensure_hooks()
        meta = self._get_connector_metadata()
        chunk_size = getattr(self._lmcache_engine, "_lmcache_chunk_size", 256)
        plan = store_plan(meta.requests, chunk_size)

        saving_ids = [
            _trace_id(r.req_id) for r in meta.requests
            if getattr(r.save_spec, "can_save", False)
        ]
        if not saving_ids:
            saving_ids = [_trace_id(r.req_id) for r in meta.requests]
        tid = saving_ids[0] if len(saving_ids) == 1 else None

        if plan is None:
            with spans.root("kv.store_noop", trace_id=tid):
                super().wait_for_save()
            return
        import torch

        with spans.root("gpu.forward_wait", trace_id=tid):
            torch.cuda.current_stream().synchronize()
        with spans.root("kv.store", trace_id=tid, **plan):
            super().wait_for_save()

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        with spans.root("kv.lookup", trace_id=_trace_id(request.request_id)) as sp:
            result = super().get_num_new_matched_tokens(request, num_computed_tokens)
            sp.set(num_tokens=request.num_tokens, hit_tokens=result[0])
        return result
