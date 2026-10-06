"""Tests for the LMCache storage hooks (span-emitting)."""

from __future__ import annotations

import json
import threading

import pytest

from kvbench.trace import spans
from kvbench.trace.lmcache_hooks import install_lmcache_trace


class FakeMemoryObj:
    def __init__(self, size: int) -> None:
        self.byte_array = bytearray(size)


class FakeKey:
    """Duck-typed stand-in for LMCache's CacheEngineKey."""

    def __init__(self, chunk_hash: int) -> None:
        self.chunk_hash = chunk_hash

    def to_string(self) -> str:
        return f"model@1@0@{self.chunk_hash:x}"

    def __str__(self) -> str:
        return f"key-{self.chunk_hash}"


class FakeDiskBackend:
    """Duck-typed LocalDiskBackend: puts write files on a worker thread."""

    use_odirect = True
    os_disk_bs = 4096

    def __init__(self) -> None:
        self.store: dict[str, FakeMemoryObj] = {}
        self.writer_threads: list[threading.Thread] = []

    def _key_to_path(self, key) -> str:
        return f"/fake/{key.to_string()}.pt"

    def batched_submit_put_task(self, keys, memory_objs, transfer_spec=None):  # noqa: ARG002
        for key, obj in zip(keys, memory_objs, strict=True):
            self.store[str(key)] = obj

            def write(obj=obj, key=key):
                self.write_file(obj.byte_array, self._key_to_path(key))

            t = threading.Thread(target=write, name="lmcache-worker")
            self.writer_threads.append(t)
            t.start()

    def join_writers(self) -> None:
        for t in self.writer_threads:
            t.join()

    def get_blocking(self, key):
        obj = self.store.get(str(key))
        if obj is not None:
            # like LocalDiskBackend, perform the read off the calling thread
            t = threading.Thread(
                target=self.read_file,
                args=(key, obj.byte_array, self._key_to_path(key)),
                name="lmcache-reader",
            )
            t.start()
            t.join()
        return obj

    def batched_get_blocking(self, keys):
        return [self.get_blocking(key) for key in keys]

    def contains(self, key, pin=False):  # noqa: ARG002
        return str(key) in self.store

    def batched_contains(self, keys):
        n = 0
        for key in keys:
            if str(key) not in self.store:
                break
            n += 1
        return n

    def remove(self, key, force=True):  # noqa: ARG002
        return self.store.pop(str(key), None) is not None

    def write_file(self, buffer, path):  # noqa: ARG002
        pass

    def read_file(self, key, buffer, path):  # noqa: ARG002
        pass


class FakeCPUBackend:
    pass


FakeCPUBackend.__name__ = "LocalCPUBackend"


class FakeStorageManager:
    def __init__(self, backends, block_mapping=None):
        self.storage_backends = backends
        self._block_mapping = block_mapping or {}

    def get_block_mapping(self, chunk_infos):  # noqa: ARG002
        return self._block_mapping


class FakeEngine:
    def __init__(self, backends, block_mapping=None):
        self.storage_manager = FakeStorageManager(backends, block_mapping)


@pytest.fixture
def tracer(tmp_path):
    t = spans.install(tmp_path / "spans.jsonl")
    yield t
    spans.uninstall()


def read(tracer):
    tracer.flush()
    return [json.loads(line) for line in tracer.path.read_text().splitlines()]


class TestInstall:
    def test_wraps_disk_backend_and_reports_it(self, tracer):
        disk = FakeDiskBackend()
        wrapped = install_lmcache_trace(FakeEngine({"LocalDiskBackend": disk}), tracer)
        assert wrapped == ["LocalDiskBackend"]
        info = next(r for r in read(tracer) if r["name"] == "lmcache.backend_info")
        assert info["parent_id"] is None
        assert info["attrs"] == {
            "tier": "FakeDiskBackend",
            "backend_name": "LocalDiskBackend",
            "use_odirect": True,
            "os_disk_bs": 4096,
        }

    def test_skips_cpu_backend(self, tracer):
        wrapped = install_lmcache_trace(FakeEngine({"LocalCPUBackend": FakeCPUBackend()}), tracer)
        assert wrapped == []

    def test_missing_surface_raises(self, tracer):
        with pytest.raises(RuntimeError, match="storage_backends"):
            install_lmcache_trace(object(), tracer)


class TestStorePath:
    def test_put_and_worker_thread_write_nest_under_the_store(self, tracer):
        disk = FakeDiskBackend()
        install_lmcache_trace(FakeEngine({"LocalDiskBackend": disk}), tracer)

        with spans.root("request", trace_id="t"), spans.span("rank.store") as store:
            disk.batched_submit_put_task([FakeKey(1), FakeKey(2)], [FakeMemoryObj(1024)] * 2)
        disk.join_writers()

        recs = read(tracer)
        puts = [r for r in recs if r["name"] == "lmcache.put"]
        writes = [r for r in recs if r["name"] == "io.write"]
        assert len(puts) == 2 and len(writes) == 2
        assert all(p["parent_id"] == store.span_id for p in puts)
        assert [p["attrs"]["chunk_hash"] for p in puts] == [1, 2]
        assert all(p["attrs"] == {"tier": "FakeDiskBackend", "chunk_hash": p["attrs"]["chunk_hash"], "bytes": 1024} for p in puts)
        # the write ran on a foreign thread yet attaches to its put, inside trace "t"
        put_ids = {p["attrs"]["chunk_hash"]: p["span_id"] for p in puts}
        for w in writes:
            assert w["trace_id"] == "t"
            assert w["thread"] == "lmcache-worker"
            assert w["attrs"]["bytes"] == 1024
            hash_in_path = int(w["attrs"]["path"].split("@")[-1].split(".")[0], 16)
            assert w["parent_id"] == put_ids[hash_in_path]
        assert tracer.bound(("path", "/fake/model@1@0@1.pt")) is None  # consumed
        assert tracer.counters[("logical", "put", "FakeDiskBackend")] == [2, 2048]
        assert tracer.counters[("io", "write", "FakeDiskBackend")] == [2, 2048]

    def test_eviction_records_victim_owner(self, tracer):
        disk = FakeDiskBackend()
        install_lmcache_trace(FakeEngine({"LocalDiskBackend": disk}), tracer)
        with spans.root("request", trace_id="owner"):
            disk.batched_submit_put_task([FakeKey(7)], [FakeMemoryObj(16)])
        disk.join_writers()
        with spans.root("request", trace_id="evictor"), spans.span("rank.store") as store:
            assert disk.remove(FakeKey(7)) is True

        evict = next(r for r in read(tracer) if r["name"] == "lmcache.evict")
        assert evict["trace_id"] == "evictor" and evict["parent_id"] == store.span_id
        assert evict["attrs"] == {
            "tier": "FakeDiskBackend",
            "chunk_hash": 7,
            "removed": True,
            "owner_trace_id": "owner",
        }
        assert tracer.counters[("logical", "remove", "FakeDiskBackend")] == [1, 0]


class TestReadPath:
    def test_lookup_contains_marks(self, tracer):
        disk = FakeDiskBackend()
        install_lmcache_trace(FakeEngine({"LocalDiskBackend": disk}), tracer)
        disk.batched_submit_put_task([FakeKey(1)], [FakeMemoryObj(8)])
        disk.join_writers()
        with spans.root("request", trace_id="t"), spans.span("kv.lookup") as lookup:
            assert disk.contains(FakeKey(1)) is True
            assert disk.contains(FakeKey(2)) is False
            assert disk.batched_contains([FakeKey(1), FakeKey(2)]) == 1

        marks = [r for r in read(tracer) if r["name"] == "lmcache.contains"]
        assert len(marks) == 3 and all(m["parent_id"] == lookup.span_id for m in marks)
        assert [m["attrs"].get("hit") for m in marks[:2]] == [True, False]
        assert marks[2]["attrs"] == {"tier": "FakeDiskBackend", "n": 2, "hit_chunks": 1}

    def test_batched_get_nests_reads_and_block_map(self, tracer):
        mapping = {
            "LocalCPUBackend": [(FakeKey(1), 0, 256), (FakeKey(2), 256, 512)],
            "LocalDiskBackend": [(FakeKey(3), 512, 768)],
        }
        disk = FakeDiskBackend()
        engine = FakeEngine({"LocalDiskBackend": disk}, block_mapping=mapping)
        install_lmcache_trace(engine, tracer)
        disk.batched_submit_put_task([FakeKey(3)], [FakeMemoryObj(32)])
        disk.join_writers()

        with spans.root("request", trace_id="t"), spans.span("rank.retrieve") as retrieve:
            engine.storage_manager.get_block_mapping([(k, s, e) for v in mapping.values() for k, s, e in v])
            result = disk.batched_get_blocking([FakeKey(3), FakeKey(9)])
        assert result[0] is not None and result[1] is None

        recs = read(tracer)
        by_name = {}
        for r in recs:
            by_name.setdefault(r["name"], []).append(r)
        block_maps = {m["attrs"]["tier"]: m for m in by_name["lmcache.block_map"]}
        assert set(block_maps) == {"LocalCPUBackend", "LocalDiskBackend"}
        cpu = block_maps["LocalCPUBackend"]["attrs"]
        assert (cpu["n"], cpu["token_start"], cpu["token_end"], cpu["n_requested"]) == (2, 0, 512, 3)
        assert cpu["chunk_hashes"] == [1, 2]
        assert all(m["parent_id"] == retrieve.span_id for m in by_name["lmcache.block_map"])

        (bget,) = by_name["lmcache.batched_get"]
        assert bget["parent_id"] == retrieve.span_id
        assert bget["attrs"] == {"tier": "FakeDiskBackend", "n": 2, "n_hit": 1, "bytes": 32}
        # the fake's batched_get calls get_blocking per key (one hit, one miss),
        # and the hit's disk read nests under its get
        gets = by_name["lmcache.get"]
        assert len(gets) == 2 and all(g["parent_id"] == bget["span_id"] for g in gets)
        assert sorted(g["attrs"]["hit"] for g in gets) == [False, True]
        hit = next(g for g in gets if g["attrs"]["hit"])
        (read_span,) = by_name["io.read"]
        assert read_span["parent_id"] == hit["span_id"]
        assert read_span["thread"] == "lmcache-reader" and read_span["trace_id"] == "t"
        assert tracer.bound(("read", disk._key_to_path(FakeKey(3)))) is None  # consumed
        assert read_span["attrs"]["bytes"] == 32 and read_span["attrs"]["path"].endswith("@3.pt")
        assert tracer.counters[("io", "read", "FakeDiskBackend")] == [1, 32]
