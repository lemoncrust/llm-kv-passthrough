"""
LMCache storage hooks: record what the KV stack does to storage as spans.

Wraps, per storage backend instance of a started lmcache.v1 engine, the
same seam LMCache's own AuditBackend uses:

- batched_submit_put_task  -> ``lmcache.put`` per chunk (submission time)
- get_blocking / batched_get_blocking / batched_get_non_blocking
                           -> ``lmcache.get`` / ``lmcache.batched_get``
- contains / batched_contains -> ``lmcache.contains``
- remove                   -> ``lmcache.evict`` (the eviction path)
- LocalDiskBackend.write_file / read_file -> ``io.write`` / ``io.read``,
  the operations a disk actually sees (path, bytes, duration)

and the engine's StorageManager.get_block_mapping -> ``lmcache.block_map``
per tier: the point where LMCache assigns each chunk of a retrieve to the
tier that serves it (covers the CPU tier too, which is otherwise left
unwrapped: RAM, not storage I/O).

Spans nest under whatever request step is current on the calling thread
(contains under kv.lookup, block_map/get under rank.retrieve, put and
evict under rank.store). The file I/O itself runs on LMCache's own worker
threads, which carry no context: at put submission (and at get time) each
chunk's file path is bound to the put/get span, and the write_file /
read_file hooks look it up, so ``io.write`` lands under the ``lmcache.put``
that scheduled it and ``io.read`` under the get that requested it.
Evictions record the trace that stored the victim (``owner_trace_id``).

Coupled to lmcache.v1 internals (version pinned in pyproject);
install_lmcache_trace() fails loudly if the expected surface is missing.
"""

from __future__ import annotations

import logging
from typing import Any

from kvbench.trace import spans
from kvbench.trace.spans import Tracer

logger = logging.getLogger(__name__)


def install_lmcache_trace(engine: Any, tracer: Tracer) -> list[str]:
    """Install span hooks on a started LMCache engine.

    Args:
        engine: A started lmcache.v1 cache engine.
        tracer: The active span tracer (owns bindings and counters).

    Returns:
        Names of the storage backends that were wrapped.

    Raises:
        RuntimeError: If the engine does not expose the expected
            storage_manager/storage_backends surface.
    """
    storage_manager = getattr(engine, "storage_manager", None)
    backends = getattr(storage_manager, "storage_backends", None)
    if not backends:
        raise RuntimeError(
            "Cannot install KV trace hooks: LMCache engine has no "
            "storage_manager.storage_backends (lmcache internals changed?)"
        )

    wrapped: list[str] = []
    for name, backend in backends.items():
        tier = type(backend).__name__
        if tier == "LocalCPUBackend":
            continue
        _wrap_logical(backend, tier, tracer)
        if hasattr(backend, "write_file") and hasattr(backend, "read_file"):
            _wrap_disk_io(backend, tier, tracer)
            tracer.mark(
                "lmcache.backend_info",
                parent=None,
                tier=tier,
                backend_name=name,
                use_odirect=bool(getattr(backend, "use_odirect", False)),
                os_disk_bs=getattr(backend, "os_disk_bs", None),
            )
        wrapped.append(name)
        logger.info(f"KV trace hooks installed on backend {name} ({tier})")

    _wrap_block_mapping(storage_manager, tracer)
    return wrapped


def _chunk_hash(key: Any) -> int | str | None:
    value = getattr(key, "chunk_hash", None)
    if value is None or isinstance(value, (int, str)):
        return value
    return str(value)


def _obj_size(memory_obj: Any) -> int | None:
    try:
        return int(len(memory_obj.byte_array))
    except Exception:
        return None


def _target_path(backend: Any, key: Any) -> str | None:
    """The file a disk backend will write ``key`` to, if it can tell us."""
    key_to_path = getattr(backend, "_key_to_path", None)
    if key_to_path is None:
        return None
    try:
        return str(key_to_path(key))
    except Exception:
        return None


def _bind_reads(backend: Any, tracer: Tracer, keys: Any, parent: Any) -> None:
    """Attach the upcoming file reads for ``keys`` to ``parent``.

    LocalDiskBackend performs reads on its own threads; binding by target
    path lets the read_file hook find the get that requested them.
    """
    for key in keys:
        path = _target_path(backend, key)
        if path is not None:
            tracer.bind(("read", path), parent)


def _wrap_logical(backend: Any, tier: str, tracer: Tracer) -> None:
    """Wrap key-level operations on a backend instance."""
    orig_put = backend.batched_submit_put_task
    orig_get = backend.get_blocking
    orig_contains = backend.contains
    orig_remove = backend.remove

    def traced_batched_submit_put_task(keys, memory_objs, *args, **kwargs):  # type: ignore[no-untyped-def]
        for key, obj in zip(keys, memory_objs, strict=False):
            size = _obj_size(obj)
            chunk_hash = _chunk_hash(key)
            put = tracer.mark("lmcache.put", tier=tier, chunk_hash=chunk_hash, bytes=size)
            tracer.count("logical", "put", tier, size)
            # The write happens later on an LMCache worker thread; let it
            # find this put by the path it will write, and let a future
            # eviction find who stored the chunk.
            path = _target_path(backend, key)
            if path is not None:
                tracer.bind(("write", path), put)
            tracer.bind(("owner", tier, chunk_hash), put)
        return orig_put(keys, memory_objs, *args, **kwargs)

    def traced_get_blocking(key, *args, **kwargs):  # type: ignore[no-untyped-def]
        with spans.span("lmcache.get", tier=tier, chunk_hash=_chunk_hash(key)) as sp:
            _bind_reads(backend, tracer, [key], sp)
            result = orig_get(key, *args, **kwargs)
            size = _obj_size(result) if result is not None else 0
            sp.set(hit=result is not None, bytes=size)
            tracer.count("logical", "get", tier, size)
        return result

    def traced_contains(key, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = orig_contains(key, *args, **kwargs)
        tracer.mark("lmcache.contains", tier=tier, chunk_hash=_chunk_hash(key), hit=bool(result))
        tracer.count("logical", "contains", tier)
        return result

    def traced_remove(key, *args, **kwargs):  # type: ignore[no-untyped-def]
        chunk_hash = _chunk_hash(key)
        owner = tracer.unbind(("owner", tier, chunk_hash))
        with spans.span("lmcache.evict", tier=tier, chunk_hash=chunk_hash) as sp:
            result = orig_remove(key, *args, **kwargs)
            sp.set(removed=bool(result))
            if owner is not None:
                sp.set(owner_trace_id=owner.trace_id)
            tracer.count("logical", "remove", tier)
        return result

    backend.batched_submit_put_task = traced_batched_submit_put_task
    backend.get_blocking = traced_get_blocking
    backend.contains = traced_contains
    backend.remove = traced_remove

    orig_batched_contains = getattr(backend, "batched_contains", None)
    if orig_batched_contains is not None:

        def traced_batched_contains(keys, *args, **kwargs):  # type: ignore[no-untyped-def]
            result = orig_batched_contains(keys, *args, **kwargs)
            tracer.mark(
                "lmcache.contains",
                tier=tier,
                n=len(keys) if hasattr(keys, "__len__") else None,
                hit_chunks=int(result) if isinstance(result, int) else bool(result),
            )
            tracer.count("logical", "batched_contains", tier)
            return result

        backend.batched_contains = traced_batched_contains

    # Batched blocking read path (used by the retrieve block-mapping path)
    orig_batched_get_blocking = getattr(backend, "batched_get_blocking", None)
    if orig_batched_get_blocking is not None:

        def traced_batched_get_blocking(keys, *args, **kwargs):  # type: ignore[no-untyped-def]
            n = len(keys) if hasattr(keys, "__len__") else None
            with spans.span("lmcache.batched_get", tier=tier, n=n) as sp:
                _bind_reads(backend, tracer, keys, sp)
                result = orig_batched_get_blocking(keys, *args, **kwargs)
                objs = [o for o in (result or []) if o is not None]
                size = sum(filter(None, (_obj_size(o) for o in objs)))
                sp.set(n_hit=len(objs), bytes=size)
                tracer.count("logical", "get", tier, size)
            return result

        backend.batched_get_blocking = traced_batched_get_blocking

    # Async batched read path (used by the storage manager's prefetcher)
    orig_batched_get = getattr(backend, "batched_get_non_blocking", None)
    if orig_batched_get is not None:

        async def traced_batched_get_non_blocking(*args, **kwargs):  # type: ignore[no-untyped-def]
            with spans.span("lmcache.batched_get", tier=tier, blocking=False) as sp:
                keys = kwargs.get("keys", args[1] if len(args) > 1 else [])
                _bind_reads(backend, tracer, keys, sp)
                result = await orig_batched_get(*args, **kwargs)
                objs = [o for o in (result or []) if o is not None]
                size = sum(filter(None, (_obj_size(o) for o in objs)))
                sp.set(n_hit=len(objs), bytes=size)
                tracer.count("logical", "get", tier, size)
            return result

        backend.batched_get_non_blocking = traced_batched_get_non_blocking


def _wrap_disk_io(backend: Any, tier: str, tracer: Tracer) -> None:
    """Wrap physical file I/O on a LocalDiskBackend-shaped instance."""
    orig_write_file = backend.write_file
    orig_read_file = backend.read_file

    def traced_write_file(buffer, path, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Runs on an LMCache worker thread: no request context here, so
        # attach to the put that scheduled this path (falls back to the
        # current span, or a root of its own).
        parent = tracer.unbind(("write", str(path)))
        with spans.span(
            "io.write",
            parent=parent if parent is not None else spans.current(),
            tier=tier,
            path=str(path),
            bytes=len(buffer),
        ):
            result = orig_write_file(buffer, path, *args, **kwargs)
            tracer.count("io", "write", tier, len(buffer))
        return result

    def traced_read_file(key, buffer, path, *args, **kwargs):  # type: ignore[no-untyped-def]
        parent = tracer.unbind(("read", str(path)))
        with spans.span(
            "io.read",
            parent=parent if parent is not None else spans.current(),
            tier=tier,
            path=str(path),
            bytes=len(buffer),
        ):
            result = orig_read_file(key, buffer, path, *args, **kwargs)
            tracer.count("io", "read", tier, len(buffer))
        return result

    backend.write_file = traced_write_file
    backend.read_file = traced_read_file


def _wrap_block_mapping(storage_manager: Any, tracer: Tracer) -> None:
    """Wrap StorageManager.get_block_mapping to record per-tier chunk serving.

    get_block_mapping is called once per retrieve and assigns a contiguous
    prefix of the requested chunks to each tier in lookup order; the marks
    it produces are the authoritative "which chunk was served from which
    tier" record for a request.
    """
    orig_get_block_mapping = storage_manager.get_block_mapping

    def traced_get_block_mapping(chunk_infos, *args, **kwargs):  # type: ignore[no-untyped-def]
        mapping = orig_get_block_mapping(chunk_infos, *args, **kwargs)
        for tier_name, blocks in mapping.items():
            if not blocks:
                continue
            tracer.mark(
                "lmcache.block_map",
                tier=tier_name,
                n=len(blocks),
                token_start=blocks[0][1],
                token_end=blocks[-1][2],
                n_requested=len(chunk_infos),
                chunk_hashes=[_chunk_hash(key) for key, _, _ in blocks],
            )
        return mapping

    storage_manager.get_block_mapping = traced_get_block_mapping
