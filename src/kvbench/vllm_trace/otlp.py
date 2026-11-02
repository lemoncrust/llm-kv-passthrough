"""
OTLP/HTTP receiver that turns vLLM request traces into kvbench spans.

vLLM exports one ``llm_request`` span per finished request (``--otlp-traces-endpoint``)
carrying its real queue / prefill / decode / TTFT / e2e times. This receiver
rewrites each one as kvbench-format spans, so a vLLM run can be compared with
the emulated ``request`` timings of ``kvbench serve``::

    request                       arrival -> end of generation
      queue                       waiting in the scheduler
      prefill                     scheduled -> first token
        response.first_chunk      mark at the first token
      decode                      first token -> last token

Point vLLM at it with ``OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf`` and
``--otlp-traces-endpoint http://localhost:4318/v1/traces``.

Span ``ts`` is wall-clock epoch seconds taken from vLLM's own timestamps; use
it, not ``start_ms``, to line these up with the connector's per-process files.
"""

from __future__ import annotations

import gzip
import threading
import time
from pathlib import Path
from typing import Any

import orjson
from fastapi import FastAPI, Request, Response

_LATENCY = "gen_ai.latency."
_REQUEST_ID = "gen_ai.request.id"
_REQUIRED = ("time_to_first_token", "e2e", "time_in_queue", "time_in_model_prefill", "time_in_model_decode")
_THREAD = "vllm-otlp"


def request_spans(
    attrs: dict[str, Any], start_ns: int, otlp_trace_id: str, t0_ns: int
) -> list[dict[str, Any]]:
    """Convert one ``llm_request`` span's attributes to kvbench span records.

    Args:
        attrs: The span's attributes, already decoded to plain Python values.
        start_ns: The span start (request arrival), unix nanoseconds.
        otlp_trace_id: Hex OTLP trace id, used when no request id attribute exists.
        t0_ns: Receiver start (unix ns), the origin of ``start_ms``.

    Returns:
        Span records, or an empty list when the latency attributes are missing.
    """
    lat = {k: attrs.get(_LATENCY + k) for k in _REQUIRED}
    if any(v is None for v in lat.values()):
        return []
    trace_id = str(attrs.get(_REQUEST_ID) or otlp_trace_id)
    arrival_ns = start_ns
    first_ns = arrival_ns + int(lat["time_to_first_token"] * 1e9)
    prefill_ns = int(lat["time_in_model_prefill"] * 1e9)
    queue_ns = int(lat["time_in_queue"] * 1e9)
    prefill_start = first_ns - prefill_ns

    def rec(name: str, span_id: str, parent: str | None, start: int, dur_ns: int, **a: Any) -> dict[str, Any]:
        return {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_id": parent,
            "name": name,
            "ts": round(start / 1e9, 6),
            "start_ms": round((start - t0_ns) / 1e6, 3),
            "dur_ms": round(dur_ns / 1e6, 3),
            "thread": _THREAD,
            "tid": 0,
            "attrs": a,
        }

    root = f"{trace_id}-request"[-12:]
    request_attrs = {
        k: attrs[k]
        for k in (
            "gen_ai.usage.prompt_tokens",
            "gen_ai.usage.completion_tokens",
            "gen_ai.request.max_tokens",
        )
        if k in attrs
    }
    return [
        rec("request", root, None, arrival_ns, int(lat["e2e"] * 1e9), source="vllm", **request_attrs),
        rec("queue", f"{root}-q", root, prefill_start - queue_ns, queue_ns),
        rec("prefill", f"{root}-p", root, prefill_start, prefill_ns),
        rec("response.first_chunk", f"{root}-f", root, first_ns, 0),
        rec("decode", f"{root}-d", root, first_ns, int(lat["time_in_model_decode"] * 1e9)),
    ]


def _attr_value(value: Any) -> Any:
    kind = value.WhichOneof("value")
    return getattr(value, kind) if kind else None


def parse_otlp(body: bytes) -> list[tuple[str, dict[str, Any], int, str]]:
    """Decode an ExportTraceServiceRequest into ``llm_request`` span tuples.

    Returns:
        ``(name, attributes, start_unix_ns, hex_trace_id)`` per ``llm_request`` span.
    """
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    msg = ExportTraceServiceRequest()
    msg.ParseFromString(body)
    out = []
    for resource in msg.resource_spans:
        for scope in resource.scope_spans:
            for span in scope.spans:
                if span.name != "llm_request":
                    continue
                attrs = {kv.key: _attr_value(kv.value) for kv in span.attributes}
                out.append((span.name, attrs, span.start_time_unix_nano, span.trace_id.hex()))
    return out


class SpanFileWriter:
    """Thread-safe JSONL span sink; truncates on start like ``Tracer``."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "wb")  # noqa: SIM115 - closed by close()
        self._lock = threading.Lock()
        self.t0_ns = time.time_ns()

    def write(self, records: list[dict[str, Any]]) -> None:
        with self._lock:
            for r in records:
                self._fh.write(orjson.dumps(r) + b"\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def create_app(writer: SpanFileWriter) -> FastAPI:
    app = FastAPI(title="kvbench OTLP receiver")
    app.state.requests_seen = 0

    @app.post("/v1/traces")
    async def traces(request: Request) -> Response:
        body = await request.body()
        if request.headers.get("content-encoding") == "gzip":
            body = gzip.decompress(body)
        for _name, attrs, start_ns, trace_id in parse_otlp(body):
            records = request_spans(attrs, start_ns, trace_id, writer.t0_ns)
            if records:
                writer.write(records)
                app.state.requests_seen += 1
        # An empty ExportTraceServiceResponse is a valid success body.
        return Response(content=b"", media_type="application/x-protobuf")

    return app
