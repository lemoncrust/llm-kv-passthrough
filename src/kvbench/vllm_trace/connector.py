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
- ``gpu.forward_wait``: time the CPU is blocked waiting for the GPU to finish
  the forward pass before this step's KV can be saved (worker; emitted only
  for steps that store). It is the GPU's compute time showing up inside
  ``wait_for_save`` and is NOT LMCache work; see
  ``TracingLMCacheConnector.wait_for_save`` for the full explanation.
- ``kv.store``: the LMCache save of new chunks (GPU->host copies plus the
  hand-off to the disk backend), with ``tokens``/``skip_leading``/``chunks``
  attrs; ``gpu.copy_out`` nests inside it. It starts only after
  ``gpu.forward_wait`` ends, so it excludes the forward-pass wait.
  Steps with nothing to store emit ``kv.store_noop`` and no ``gpu.forward_wait``.
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
        """Save this step's new KV chunks, timing the GPU wait and the save separately.

        Spans for a step that stores (all share the request's ``chatcmpl-`` trace id)::

            gpu.forward_wait   CPU blocked until the GPU drains its queue   (~55 ms)
            kv.store           LMCache's save, starting after the wait      (~2-3 ms)
              gpu.copy_out     one per 256-token chunk, GPU -> pinned host buffer

        Steps with nothing to store (e.g. decode steps) emit only ``kv.store_noop``
        and get no extra synchronization.

        Why ``gpu.forward_wait`` exists
        -------------------------------
        The CPU does not run the model. ``execute_model`` only *queues* the forward-pass
        kernels on the worker's CUDA stream (a FIFO queue the GPU drains in order) and
        returns while the GPU is still computing. vLLM then calls this method from
        ``ActiveKVConnector.post_forward`` (``vllm/v1/worker/gpu/kv_connector.py``),
        invoked by ``sample_tokens``, i.e. with the forward pass, and the sampling
        kernels queued after it, typically still running.

        The first GPU-touching line of LMCache's save path
        (``LMCacheConnectorV1Impl.wait_for_save`` in ``vllm_v1_adapter.py``) is
        ``slot_mapping = slot_mapping.to(self.device)``. That copy is queued behind the
        forward pass on the same stream and, being a default ``non_blocking=False``
        copy, makes the CPU wait until it completes. So the CPU sits there until the
        whole queue ahead of it has drained, about 55 ms for a 2048-token prompt on
        Llama-3.2-1B / GB10. Only then do LMCache's copy-out kernels run.

        Without this span, that wait is charged to ``kv.store``: ~59 ms, of which only
        ~2-3 ms is storage work. That is misleading in two ways:

        - it looks like LMCache is slow, when the time is the model computing;
        - ``kvbench serve`` has no real GPU (``prefill.compute`` is a sleep that ends
          before ``kv.store`` opens), so its ``kv.store`` contains no such wait and the
          two spans would not measure the same thing.

        We therefore block on the stream ourselves, in its own span, and open
        ``kv.store`` afterwards so it covers only the save: seven GPU -> host copies of
        8 MiB plus the hand-off to the disk backend. The disk writes (``io.write``)
        are asynchronous and belong to neither span.

        How to read ``gpu.forward_wait``
        --------------------------------
        - It is the GPU time *remaining* when the CPU arrives, not the full forward
          time: some of it already ran while the CPU was loading KV and launching
          kernels. It also includes the sampling kernels queued before this call.
        - It is not I/O and not LMCache overhead.
        - Compare it with kvbench's ``prefill.compute``, never with its ``kv.store``.
          Real ``kv.store`` pairs with kvbench ``kv.store``.

        Why ``current_stream()`` and what it changes
        --------------------------------------------
        The forward pass runs on the thread's current stream, so that is the one to
        drain. ``torch.cuda.synchronize()`` would also wait for unrelated streams, and
        LMCache's own ``store_stream`` must not be waited on: its copies belong inside
        ``kv.store``.

        The explicit call moves the wait rather than adding one: LMCache's ``.to()``
        would block on the same stream a moment later, and ``from_gpu`` never makes
        ``store_stream`` wait for the compute stream, so that CPU-side block is what
        orders the copy-out kernels after the forward pass. Measured: ``kv.store``
        without this call 59.1 ms; with it, ``gpu.forward_wait`` 54.0 ms plus
        ``kv.store`` 2.7 ms (10 requests each; same totals within noise). It is still
        an added call in the real code path. To observe passively, delete the
        ``gpu.forward_wait`` block and read the wait as the gap from the start of
        ``kv.store`` to its first ``gpu.copy_out``.

        Evidence that the blocking call is that ``.to()`` (PyTorch sync-debug stacks,
        10 of 10 requests) is kept in ``traces/proof_gpu_sync/``.
        """
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

        # kv.store covers LMCache's save: GPU→CPU copies (gpu.copy_out, emitted by the
        # gpu_connector.from_gpu() hook) + bookkeeping + disk-write submission.
        if plan is None:
            with spans.root("kv.store_noop", trace_id=tid):
                super().wait_for_save()
            return
        import torch

        # See the docstring: drain the compute stream here so kv.store excludes the wait.
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
