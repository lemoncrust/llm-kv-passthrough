"""Tests for span-based request tracing."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from kvbench.trace import spans


@pytest.fixture
def tracer(tmp_path):
    t = spans.install(tmp_path / "spans.jsonl")
    yield t
    spans.uninstall()


def read(tracer):
    tracer.flush()
    return [json.loads(line) for line in tracer.path.read_text().splitlines()]


def by_name(records):
    return {r["name"]: r for r in records}


class TestSpanTree:
    def test_nested_spans_form_a_tree(self, tracer):
        with spans.root("request", trace_id="req-1", model="m"):
            with spans.span("tokenize") as tok:
                tok.set(tokens=10)
            with spans.span("kv.lookup", tokens=10), spans.span("lmcache.contains", tier="disk"):
                pass

        recs = read(tracer)
        by = by_name(recs)
        assert set(by) == {"request", "tokenize", "kv.lookup", "lmcache.contains"}
        assert all(r["trace_id"] == "req-1" for r in recs)
        assert by["request"]["parent_id"] is None
        assert by["request"]["attrs"] == {"model": "m"}
        assert by["tokenize"]["parent_id"] == by["request"]["span_id"]
        assert by["tokenize"]["attrs"] == {"tokens": 10}
        assert by["kv.lookup"]["parent_id"] == by["request"]["span_id"]
        assert by["lmcache.contains"]["parent_id"] == by["kv.lookup"]["span_id"]
        # Spans are written as they close: children before parents
        assert [r["name"] for r in recs] == ["tokenize", "lmcache.contains", "kv.lookup", "request"]
        assert by["request"]["dur_ms"] >= by["kv.lookup"]["dur_ms"] >= 0
        for key in ("ts", "start_ms", "dur_ms", "thread", "span_id"):
            assert key in by["request"]

    @pytest.mark.usefixtures("tracer")
    def test_root_without_trace_id_gets_a_fresh_one(self):
        with spans.root("request") as a:
            pass
        with spans.root("request") as b:
            pass
        assert a.trace_id and b.trace_id and a.trace_id != b.trace_id

    def test_sequential_roots_do_not_nest(self, tracer):
        with spans.root("request", trace_id="a"):
            pass
        with spans.root("request", trace_id="b"):
            pass
        assert all(r["parent_id"] is None for r in read(tracer))
        assert spans.current() is None

    def test_mark_is_instantaneous_child(self, tracer):
        with spans.root("request", trace_id="t"):
            m = spans.mark("response.first_chunk", n=1)
        by = by_name(read(tracer))
        assert by["response.first_chunk"]["dur_ms"] == 0.0
        assert by["response.first_chunk"]["parent_id"] == by["request"]["span_id"]
        assert by["response.first_chunk"]["attrs"] == {"n": 1}
        assert m.span_id == by["response.first_chunk"]["span_id"]

    def test_explicit_parent_overrides_current(self, tracer):
        with spans.root("request", trace_id="t") as req:
            with spans.span("kv.store") as store, spans.span("io.write", parent=store):
                pass
            # a span parented explicitly to a non-current span
            with spans.span("other"), spans.span("late", parent=req):
                pass
        by = by_name(read(tracer))
        assert by["io.write"]["parent_id"] == by["kv.store"]["span_id"]
        assert by["late"]["parent_id"] == by["request"]["span_id"]


class TestContextPropagation:
    async def test_to_thread_work_attaches_to_caller_span(self, tracer):
        def worker():
            with spans.span("rank.store", rank=0):
                pass

        with spans.root("request", trace_id="t"):
            await asyncio.to_thread(worker)

        by = by_name(read(tracer))
        assert by["rank.store"]["parent_id"] == by["request"]["span_id"]
        assert by["rank.store"]["trace_id"] == "t"
        assert by["rank.store"]["thread"] != by["request"]["thread"]

    async def test_spans_record_os_thread_id(self, tracer):
        # Pool threads reuse names across executor instances (two LMCache
        # engines each have a "disk-read_0"); the kernel thread id tells
        # them apart, so every span carries it next to the name.
        def worker():
            with spans.span("io.read"):
                pass

        with spans.root("request", trace_id="t"):
            await asyncio.to_thread(worker)

        by = by_name(read(tracer))
        assert by["request"]["tid"] == threading.get_native_id()
        assert isinstance(by["io.read"]["tid"], int)
        assert by["io.read"]["tid"] not in (0, by["request"]["tid"])

    async def test_concurrent_requests_keep_separate_trees(self, tracer):
        async def handle(i: int) -> None:
            with spans.root("request", trace_id=f"r{i}"):
                await asyncio.sleep(0.005 * (3 - i))
                with spans.span("tokenize"):
                    await asyncio.sleep(0.002)
                await asyncio.to_thread(lambda: spans.mark("io"))

        await asyncio.gather(*(handle(i) for i in range(3)))

        recs = read(tracer)
        roots = {r["trace_id"]: r["span_id"] for r in recs if r["name"] == "request"}
        assert len(roots) == 3
        for r in recs:
            if r["name"] != "request":
                assert r["parent_id"] == roots[r["trace_id"]], r

    def test_foreign_thread_binding(self, tracer):
        with spans.root("request", trace_id="t"), spans.span("kv.store") as store:
            tracer.bind("chunk-key", store)

        def worker():
            parent = tracer.bound("chunk-key")
            assert parent is not None
            with spans.span("io.write", parent=parent, bytes=16):
                pass
            tracer.unbind("chunk-key")

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        by = by_name(read(tracer))
        assert by["io.write"]["parent_id"] == by["kv.store"]["span_id"]
        assert by["io.write"]["trace_id"] == "t"
        assert tracer.bound("chunk-key") is None


class TestFailureModes:
    def test_exception_is_recorded_and_reraised(self, tracer):
        with pytest.raises(ValueError), spans.root("request", trace_id="t"), spans.span("kv.store"):
            raise ValueError("boom")
        by = by_name(read(tracer))
        assert by["kv.store"]["attrs"]["error"] == "ValueError"
        assert "boom" in by["kv.store"]["attrs"]["error_msg"]
        assert by["request"]["attrs"]["error"] == "ValueError"
        assert by["request"]["dur_ms"] >= 0

    async def test_closed_streaming_generator_marks_cancelled(self, tracer):
        async def stream():
            with spans.root("request", trace_id="t"):
                yield "first"
                yield "second"

        gen = stream()
        assert await gen.__anext__() == "first"
        await gen.aclose()  # client disconnect

        by = by_name(read(tracer))
        assert by["request"]["attrs"] == {"cancelled": True}

    async def test_task_cancellation_marks_cancelled(self, tracer):
        async def handler():
            with spans.root("request", trace_id="t"), spans.span("decode"):
                await asyncio.sleep(10)

        task = asyncio.create_task(handler())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        by = by_name(read(tracer))
        assert by["decode"]["attrs"]["cancelled"] is True
        assert by["request"]["attrs"]["cancelled"] is True


class TestWithoutTracer:
    def test_everything_is_a_noop(self, tmp_path):
        assert spans.get_tracer() is None
        with spans.root("request", trace_id="t") as req:
            with spans.span("x") as sp:
                sp.set(a=1)
            m = spans.mark("m")
        assert req is spans.NULL_SPAN and sp is spans.NULL_SPAN and m is spans.NULL_SPAN
        assert spans.NULL_SPAN.attrs == {}
        assert spans.current() is None
        assert not list(tmp_path.iterdir())


class TestTracer:
    def test_counters_accumulate(self, tracer):
        tracer.count("io", "write", "LocalDiskBackend", 100)
        tracer.count("io", "write", "LocalDiskBackend", 50)
        tracer.count("logical", "put", "LocalDiskBackend")
        assert tracer.counters[("io", "write", "LocalDiskBackend")] == [2, 150]
        assert tracer.counters[("logical", "put", "LocalDiskBackend")] == [1, 0]

    def test_end_is_idempotent(self, tracer):
        sp = tracer.start("x", parent=None)
        tracer.end(sp, a=1)
        tracer.end(sp, b=2)
        recs = read(tracer)
        assert len(recs) == 1 and recs[0]["attrs"] == {"a": 1}

    def test_thread_safety(self, tracer):
        def writer():
            for _ in range(200):
                with spans.root("r"):
                    pass

        threads = [threading.Thread(target=writer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        recs = read(tracer)
        assert len(recs) == 1600

    def test_install_truncates_existing_file(self, tmp_path):
        path = tmp_path / "spans.jsonl"
        spans.install(path)
        with spans.root("first"):
            pass
        spans.install(path)  # re-install truncates
        with spans.root("second"):
            pass
        spans.uninstall()
        recs = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["name"] for r in recs] == ["second"]

    def test_non_json_attr_values_are_stringified(self, tracer):
        with spans.root("r", path=tracer.path):
            pass
        recs = read(tracer)
        assert recs[0]["attrs"]["path"] == str(tracer.path)

    def test_periodic_flush_makes_trace_readable_live(self, tmp_path, monkeypatch):
        monkeypatch.setattr(spans.Tracer, "FLUSH_EVERY_SECONDS", 0.05)
        tracer = spans.install(tmp_path / "live.jsonl")
        try:
            with spans.root("first"):
                pass
            with spans.root("second"):  # buffered: no write-time flush is due yet
                pass
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                lines = tracer.path.read_text().splitlines()
                if len(lines) == 2:
                    break
                time.sleep(0.02)
            assert [json.loads(line)["name"] for line in lines] == ["first", "second"]
        finally:
            spans.uninstall()
        assert not tracer._flusher.is_alive()
