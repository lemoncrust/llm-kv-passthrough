"""Tests for span-trace assembly."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from kvbench.cli.main import app
from kvbench.trace.tree import build, load


def span(name, span_id, parent_id, trace_id, start_ms, dur_ms=1.0, thread="main", **attrs):
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_id": parent_id,
        "name": name,
        "ts": 1000.0 + start_ms / 1000,
        "start_ms": start_ms,
        "dur_ms": dur_ms,
        "thread": thread,
        "attrs": attrs,
    }


def request_records(trace_id, start, *, conversation=None, turn=None, previous=None, stored=True):
    """A plausible request tree, written children-first like the tracer does."""
    root_attrs = {"model": "m", "stream": True, "completion_id": f"cmpl-{trace_id}"}
    if conversation is not None:
        root_attrs.update(conversation_id=conversation, turn_index=turn, previous_request_id=previous)
    r = f"{trace_id}-root"
    recs = [
        span("tokenize", f"{trace_id}-tok", r, trace_id, start + 0.5, 0.4, tokens=1024),
        span("lmcache.contains", f"{trace_id}-c", f"{trace_id}-lk", trace_id, start + 1.1, 0.0, tier="LocalDiskBackend"),
        span("kv.lookup", f"{trace_id}-lk", r, trace_id, start + 1.0, 0.5, hit_tokens=512),
        span("prefill.compute", f"{trace_id}-pf", r, trace_id, start + 2.0, 30.0, emulated_ms=29.5),
    ]
    if stored:
        recs += [
            # worker-thread write closes after the store span it belongs to
            span("lmcache.put", f"{trace_id}-put", f"{trace_id}-rs", trace_id, start + 33.0, 0.0, tier="LocalDiskBackend", bytes=16),
            span("rank.store", f"{trace_id}-rs", f"{trace_id}-st", trace_id, start + 32.5, 2.0, rank=0, thread="pool"),
            span("kv.store", f"{trace_id}-st", r, trace_id, start + 32.0, 3.0, chunks=4),
        ]
    recs += [
        span("decode.step", f"{trace_id}-d0", f"{trace_id}-dec", trace_id, start + 36.0, 9.0, i=0),
        span("decode.step", f"{trace_id}-d1", f"{trace_id}-dec", trace_id, start + 45.5, 9.0, i=1),
        span("decode", f"{trace_id}-dec", r, trace_id, start + 36.0, 19.0, generated=2),
        span("request", r, None, trace_id, start, 56.0, first_chunk_ms=35.0, **root_attrs),
    ]
    if stored:
        recs.append(
            span("io.write", f"{trace_id}-io", f"{trace_id}-put", trace_id, start + 40.0, 4.0, thread="disk", tier="LocalDiskBackend", bytes=16, path="/c/x")
        )
    return recs


def sample_records():
    recs = [span("lmcache.backend_info", "info", None, "", 0.0, 0.0, tier="LocalDiskBackend")]
    recs += request_records("a1", 100.0, conversation="convA", turn=0)
    recs += request_records("b1", 110.0, conversation="convB", turn=0)
    recs += request_records("a2", 400.0, conversation="convA", turn=1, previous="a1", stored=False)
    recs += request_records("solo", 300.0)
    # an unattributed write (binding missed): must survive as an orphan
    recs.append(span("io.write", "stray", "missing-parent", "zz", 500.0, 2.0, thread="disk", tier="LocalDiskBackend"))
    return recs


class TestBuild:
    def test_groups_turns_into_conversations_in_order(self):
        trace = build(sample_records())
        assert trace.span_count == len(sample_records())
        ids = [c.conversation_id for c in trace.conversations]
        assert ids == ["convA", "convB", None]  # ordered by first turn start
        conv_a = trace.conversation("convA")
        assert [t.trace_id for t in conv_a.turns] == ["a1", "a2"]
        assert [t.turn_index for t in conv_a.turns] == [0, 1]
        assert conv_a.turns[1].previous_request_id == "a1"
        assert conv_a.start_ms == 100.0 and conv_a.end_ms == 456.0
        solo = trace.conversations[2]
        assert solo.turns[0].trace_id == "solo" and solo.turns[0].turn_index is None

    def test_children_nest_and_sort_regardless_of_file_order(self):
        turn = build(sample_records()).turn("a1")
        assert [c.name for c in turn.root.children] == [
            "tokenize", "kv.lookup", "prefill.compute", "kv.store", "decode",
        ]
        assert turn.stage("kv.lookup").child("lmcache.contains") is not None
        store = turn.stage("kv.store")
        assert [n.name for n in store.walk()] == ["kv.store", "rank.store", "lmcache.put", "io.write"]
        assert [s.attrs["i"] for s in turn.stage("decode").children] == [0, 1]

    def test_turn_helpers(self):
        trace = build(sample_records())
        a1, a2 = trace.conversation("convA").turns
        assert a1.ttft_ms == 35.0
        assert a1.stage_durations() == {
            "tokenize": 0.4, "kv.lookup": 0.5, "prefill.compute": 30.0, "kv.store": 3.0, "decode": 19.0,
        }
        assert a1.io_ms("write") == 4.0 and a1.io_ms("read") == 0.0
        assert a2.stage("kv.store") is None and a2.io_ms("write") == 0.0
        assert a1.served_from() == {}  # no block_map marks in this synthetic trace

    def test_lookup_by_unique_prefix(self):
        trace = build(sample_records())
        assert trace.conversation("convB").conversation_id == "convB"
        assert trace.turn("a2").trace_id == "a2"            # exact match wins
        assert trace.turn("so").trace_id == "solo"           # unique prefix
        assert trace.turn("zzz") is None
        with pytest.raises(ValueError, match="ambiguous"):
            trace.turn("a")                                  # a1 and a2

    def test_unattributed_spans_become_orphans(self):
        trace = build(sample_records())
        assert [n.name for n in trace.orphans] == ["lmcache.backend_info", "io.write"]
        assert trace.orphans[1].span_id == "stray"

    def test_non_span_records_are_ignored(self):
        legacy = {"event": "io", "op": "write", "backend": "LocalDiskBackend", "size": 1}
        trace = build([legacy, *request_records("x", 0.0)])
        assert trace.span_count == len(request_records("x", 0.0))
        assert trace.turn("x") is not None

    def test_to_dict_is_json_and_nested(self):
        payload = build(sample_records()).to_dict()
        text = json.dumps(payload)
        data = json.loads(text)
        assert data["spans"] == len(sample_records())
        conv_a = next(c for c in data["conversations"] if c["conversation_id"] == "convA")
        assert conv_a["turns"] == 2
        req = conv_a["requests"][0]
        assert req["name"] == "request" and req["attrs"]["turn_index"] == 0
        assert [c["name"] for c in req["children"]][:2] == ["tokenize", "kv.lookup"]
        assert data["orphans"][1]["name"] == "io.write"

    def test_os_thread_id_round_trips_and_defaults_to_zero(self):
        # Pool thread names repeat across LMCache engines; the OS thread id
        # is what tells them apart. Old traces without it read back as 0.
        recs = request_records("a1", 100.0, conversation="convA", turn=0)
        next(r for r in recs if r["name"] == "request")["tid"] = 4242
        data = build(recs).to_dict()
        req = data["conversations"][0]["requests"][0]
        assert req["tid"] == 4242
        assert {n["tid"] for n in req["children"]} == {0}


class TestLoadAndCli:
    def write(self, tmp_path):
        p = tmp_path / "spans.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in sample_records()) + "\ngarbage\n")
        return p

    def test_load_skips_garbage(self, tmp_path):
        trace = load(self.write(tmp_path))
        assert len(trace.turns) == 4

    def test_cli_tree_whole_trace(self, tmp_path):
        result = CliRunner().invoke(app, ["trace", "tree", str(self.write(tmp_path)), "--indent", "0"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert [c["conversation_id"] for c in data["conversations"]] == ["convA", "convB", None]

    def test_cli_tree_single_request_to_file(self, tmp_path):
        out = tmp_path / "a2.json"
        result = CliRunner().invoke(
            app, ["trace", "tree", str(self.write(tmp_path)), "--request", "a2", "-o", str(out)]
        )
        assert result.exit_code == 0, result.output
        data = json.loads(out.read_text())
        assert data["trace_id"] == "a2" and data["attrs"]["previous_request_id"] == "a1"
        # the confirmation describes what was written, not the whole trace
        assert "request a2" in result.output and "(8 spans)" in result.output
        assert "conversations" not in result.output

    def test_cli_tree_conversation_to_file_reports_its_size(self, tmp_path):
        out = tmp_path / "convA.json"
        result = CliRunner().invoke(
            app, ["trace", "tree", str(self.write(tmp_path)), "-c", "convA", "-o", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert "conversation convA (2 requests, 20 spans)" in result.output
        assert json.loads(out.read_text())["turns"] == 2

    def test_cli_tree_rejects_request_with_conversation(self, tmp_path):
        result = CliRunner().invoke(
            app, ["trace", "tree", str(self.write(tmp_path)), "-r", "a1", "-c", "convA"]
        )
        assert result.exit_code == 2 and "mutually exclusive" in result.output

    def test_cli_tree_conversation_and_missing(self, tmp_path):
        p = self.write(tmp_path)
        ok = CliRunner().invoke(app, ["trace", "tree", str(p), "-c", "convB", "--indent", "0"])
        assert ok.exit_code == 0 and json.loads(ok.output)["turns"] == 1
        missing = CliRunner().invoke(app, ["trace", "tree", str(p), "-r", "nope"])
        assert missing.exit_code == 1

    def test_cli_tree_accepts_unique_prefix_and_rejects_ambiguous(self, tmp_path):
        p = self.write(tmp_path)
        out = tmp_path / "b.json"
        ok = CliRunner().invoke(app, ["trace", "tree", str(p), "-c", "convB"[:5], "-o", str(out)])
        assert ok.exit_code == 0, ok.output
        assert "conversation convB" in ok.output  # resolved full id is reported
        assert json.loads(out.read_text())["conversation_id"] == "convB"
        ambiguous = CliRunner().invoke(app, ["trace", "tree", str(p), "-r", "a"])
        assert ambiguous.exit_code == 1 and "ambiguous" in ambiguous.output
