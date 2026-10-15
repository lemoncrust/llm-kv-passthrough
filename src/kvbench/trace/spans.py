"""
Span-based request tracing.

Every distinct processing step of a request (tokenize, KV lookup, retrieve,
prefill compute, store, each decode step, ...) is recorded as a *span*: a
named interval with a parent, so the steps of one request form a tree and
concurrent requests stay separable. Spans are written as JSON Lines, one
object per span when it closes:

    {"trace_id": "<request id>", "span_id": "...", "parent_id": "..."|null,
     "name": "kv.retrieve", "ts": <epoch seconds of start>,
     "start_ms": <ms since tracer start, monotonic>, "dur_ms": <float>,
     "thread": "<thread name>", "tid": <kernel thread id>, "attrs": {...}}

``thread`` is the Python thread name, which pools reuse across instances
(two LMCache engines each own a ``disk-read_0``); ``tid`` is the OS thread
id (``threading.get_native_id()``) and tells such threads apart.

The current span travels in a ContextVar, so nested calls — including work
moved off the event loop with asyncio.to_thread, which copies the caller's
context — attach as children without any plumbing. Code running on threads
LMCache owns (async disk writes) has no such context and is attached to a
parent explicitly through Tracer.bind()/bound().

With no tracer installed every entry point is a cheap no-op yielding
NULL_SPAN, so instrumented code needs no `if tracing:` guards.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """One traced interval. Mutate attributes through set(); end via the tracer."""

    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    start_mono: float
    attrs: dict[str, Any] = field(default_factory=dict)
    end_mono: float | None = None
    thread: str = ""
    tid: int = 0

    def set(self, **attrs: Any) -> Span:
        self.attrs.update(attrs)
        return self

    @property
    def ended(self) -> bool:
        return self.end_mono is not None


class _NullSpan(Span):
    def set(self, **attrs: Any) -> Span:  # noqa: ARG002 - intentionally inert
        return self


NULL_SPAN: Span = _NullSpan(trace_id="", span_id="", parent_id=None, name="", start_mono=0.0)


class _UseCurrent:
    """Sentinel: parent the new span to whatever span is current."""


_USE_CURRENT = _UseCurrent()
_current: ContextVar[Span | None] = ContextVar("kvbench_current_span", default=None)
_active: Tracer | None = None


def _new_id(nbytes: int = 6) -> str:
    return secrets.token_hex(nbytes)


class Tracer:
    """Thread-safe JSONL span writer.

    Also keeps the per-(event, op, backend) op/byte counters that the
    Prometheus exporter publishes, and a key -> span binding table used to
    attach work done on foreign threads to the span that caused it.
    """

    FLUSH_EVERY_RECORDS = 256
    FLUSH_EVERY_SECONDS = 1.0

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate: start_ms is process-relative, so spans from a previous
        # run would corrupt timeline analysis if appended to.
        self._fh = open(self.path, "w", buffering=1024 * 1024)  # noqa: SIM115 - closed by close()
        self._lock = threading.Lock()
        # Anchor wall clock to the monotonic clock once so ts and start_ms
        # stay mutually consistent for the life of the trace.
        self._mono0 = time.perf_counter()
        self._wall0 = time.time()
        self.counters: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
        self._bindings: dict[Any, Span] = {}
        self._closed = False
        self._unflushed = 0
        self._last_flush = time.monotonic()
        # Flush on a timer too, so a trace can be tailed during a run even
        # when no new spans arrive to trigger the write-time flush policy.
        self._stop = threading.Event()
        self._flusher = threading.Thread(
            target=self._flush_loop, name="kvbench-trace-flush", daemon=True
        )
        self._flusher.start()

    def _flush_loop(self) -> None:
        while not self._stop.wait(self.FLUSH_EVERY_SECONDS):
            self.flush()

    # -- spans -----------------------------------------------------------

    def start(
        self,
        name: str,
        *,
        parent: Span | None | _UseCurrent = _USE_CURRENT,
        trace_id: str | None = None,
        **attrs: Any,
    ) -> Span:
        """Create (but do not make current) a span under ``parent``.

        ``parent`` defaults to the current span; pass None for a root span.
        ``trace_id`` defaults to the parent's, or a fresh id for roots.
        """
        if isinstance(parent, _UseCurrent):
            parent = _current.get()
        if trace_id is None:
            trace_id = parent.trace_id if parent is not None else _new_id(8)
        return Span(
            trace_id=trace_id,
            span_id=_new_id(),
            parent_id=parent.span_id if parent is not None else None,
            name=name,
            start_mono=time.perf_counter(),
            attrs=dict(attrs),
            thread=threading.current_thread().name,
            tid=threading.get_native_id(),
        )

    def end(self, span: Span, **attrs: Any) -> None:
        """Close ``span`` (idempotent) and write it."""
        if span is NULL_SPAN or span.ended:
            return
        span.attrs.update(attrs)
        span.end_mono = time.perf_counter()
        self._write(span)

    def mark(
        self,
        name: str,
        *,
        parent: Span | None | _UseCurrent = _USE_CURRENT,
        **attrs: Any,
    ) -> Span:
        """Record an instantaneous (zero-duration) span under ``parent``."""
        span = self.start(name, parent=parent, **attrs)
        span.end_mono = span.start_mono
        self._write(span)
        return span

    def now_ms(self) -> float:
        """Milliseconds since the tracer started (same clock as start_ms)."""
        return (time.perf_counter() - self._mono0) * 1000

    # -- foreign-thread attribution ------------------------------------------

    def bind(self, key: Any, span: Span) -> None:
        """Remember that work identified by ``key`` belongs under ``span``."""
        with self._lock:
            self._bindings[key] = span

    def bound(self, key: Any) -> Span | None:
        with self._lock:
            return self._bindings.get(key)

    def unbind(self, key: Any) -> Span | None:
        with self._lock:
            return self._bindings.pop(key, None)

    # -- counters (Prometheus) --------------------------------------------------

    def count(self, event: str, op: str, backend: str, size: int | None = None) -> None:
        with self._lock:
            counter = self.counters[(event, op, backend)]
            counter[0] += 1
            counter[1] += size or 0

    # -- output ---------------------------------------------------------------

    def _write(self, span: Span) -> None:
        if self._closed:
            return
        start = span.start_mono - self._mono0
        end = (span.end_mono if span.end_mono is not None else span.start_mono) - self._mono0
        rec = {
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_id": span.parent_id,
            "name": span.name,
            "ts": round(self._wall0 + start, 6),
            "start_ms": round(start * 1000, 3),
            "dur_ms": round((end - start) * 1000, 3),
            "thread": span.thread,
            "tid": span.tid,
            "attrs": span.attrs,
        }
        line = json.dumps(rec, separators=(",", ":"), default=str) + "\n"
        with self._lock:
            if self._closed:
                return
            self._fh.write(line)
            self._unflushed += 1
            now = time.monotonic()
            if (
                self._unflushed >= self.FLUSH_EVERY_RECORDS
                or now - self._last_flush >= self.FLUSH_EVERY_SECONDS
            ):
                self._fh.flush()
                self._unflushed = 0
                self._last_flush = now

    def flush(self) -> None:
        with self._lock:
            if not self._closed:
                self._fh.flush()
                self._unflushed = 0
                self._last_flush = time.monotonic()

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._fh.flush()
            self._fh.close()
        if threading.current_thread() is not self._flusher:
            self._flusher.join(timeout=2.0)


# -- module-level API (what instrumented code uses) -------------------------------


def install(path: Path | str) -> Tracer:
    """Make a tracer writing to ``path`` the process-wide active tracer."""
    global _active
    if _active is not None:
        _active.close()
    _active = Tracer(path)
    logger.info(f"Span tracing enabled -> {path}")
    return _active


def uninstall() -> None:
    global _active
    if _active is not None:
        _active.close()
        _active = None


def get_tracer() -> Tracer | None:
    return _active


def current() -> Span | None:
    """The span the calling code is executing under, if any."""
    return _current.get()


@contextmanager
def span(
    name: str,
    *,
    parent: Span | None | _UseCurrent = _USE_CURRENT,
    trace_id: str | None = None,
    **attrs: Any,
) -> Iterator[Span]:
    """Run the block as a span and make it current for nested calls.

    Exceptions are recorded (``error``, ``error_msg``) and re-raised;
    cancellation (client disconnect closing a streaming generator, task
    cancellation) is recorded as ``cancelled: true``.
    """
    tracer = _active
    if tracer is None:
        yield NULL_SPAN
        return
    sp = tracer.start(name, parent=parent, trace_id=trace_id, **attrs)
    previous = _current.get()
    # set(previous) rather than reset(token): a streaming handler's root span
    # may be entered and exited in different async-generator steps.
    _current.set(sp)
    try:
        yield sp
    except (GeneratorExit, asyncio.CancelledError):
        sp.set(cancelled=True)
        raise
    except BaseException as e:
        sp.set(error=type(e).__name__, error_msg=str(e)[:200])
        raise
    finally:
        tracer.end(sp)
        _current.set(previous)


def root(name: str, *, trace_id: str | None = None, **attrs: Any) -> Any:
    """Start a new trace: a span with no parent (e.g. one request)."""
    return span(name, parent=None, trace_id=trace_id, **attrs)


def mark(name: str, **attrs: Any) -> Span:
    """Record an instantaneous span under the current span."""
    tracer = _active
    if tracer is None:
        return NULL_SPAN
    return tracer.mark(name, **attrs)
