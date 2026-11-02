#!/home/admin/aditya/ai-perf/bin/python
"""
compare_report.py — PDF comparison: real vLLM vs kvbench emulated.

One page per test case. Each page has one table where rows = metrics and
columns = mean_real | mean_emul | Δ | p50_real | p50_emul | Δ | p90_real | p90_emul | Δ

Usage:
    python compare_report.py \\
        --vllm-dir  ~/aditya/traces/vllm_run_YYYYMMDD \\
        --kvbench-dir ~/aditya/traces/kvbench_run_YYYYMMDD \\
        --output    ~/aditya/traces/comparison.pdf \\
        [--no-stddev]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ── Palette ───────────────────────────────────────────────────────────────────
C_NAVY  = colors.HexColor("#1a2e4a")
C_BLUE  = colors.HexColor("#2471a3")
C_ALT   = colors.HexColor("#eef3f8")
C_WHITE = colors.white
C_MUTED = colors.HexColor("#6c757d")
C_GREEN = colors.HexColor("#1a7a3c")
C_RED   = colors.HexColor("#a93226")
C_GREY  = colors.HexColor("#aaaaaa")
C_DELTA = colors.HexColor("#fdf6e3")

PAGE_W, PAGE_H = landscape(A4)
MARGIN  = 1.6 * cm
AVAIL_W = PAGE_W - 2 * MARGIN

MODEL    = "meta-llama/Llama-3.2-1B-Instruct"
HARDWARE = "GB10 DGX Spark"

# Span name aliases
ALIASES = {"prefill.compute": "prefill", "decode.step": "decode_step"}

# gpu.copy_in no-op marker (our connector sets this when no requests are loading)
NO_LOAD_MARKER = "no-load"

# ── Styles ────────────────────────────────────────────────────────────────────
def ps(name, **kw): return ParagraphStyle(name, **kw)

S = {
    "title": ps("T",  fontName="Helvetica-Bold",     fontSize=17, textColor=C_NAVY, spaceAfter=4),
    "h1":    ps("H1", fontName="Helvetica-Bold",     fontSize=12, textColor=C_NAVY, spaceBefore=10, spaceAfter=4),
    "note":  ps("N",  fontName="Helvetica-Oblique",  fontSize=7.5, textColor=C_MUTED, spaceAfter=3),
    "meta_k":ps("MK", fontName="Helvetica-Bold",     fontSize=10, textColor=C_NAVY),
    "meta_v":ps("MV", fontName="Helvetica",           fontSize=10, textColor=colors.HexColor("#1a1a1a")),
}

# ── Data loading ──────────────────────────────────────────────────────────────
def load_and_merge(paths: list[Path]) -> list[dict]:
    """Load all JSONL files, apply name aliases, sort by ts."""
    spans = []
    for p in paths:
        try:
            for line in p.read_text().splitlines():
                try:
                    rec = json.loads(line)
                    name = rec.get("name", "")
                    rec["name"] = ALIASES.get(name, name)
                    if rec.get("dur_ms") is not None:
                        spans.append(rec)
                except json.JSONDecodeError:
                    pass
        except FileNotFoundError:
            pass
    return sorted(spans, key=lambda s: s.get("ts", 0))


import re as _re
_UUID_RE = _re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
)

def is_real_request(trace_id: str | None) -> bool:
    """True if the span belongs to a real request, False for no-op/internal spans.

    Accepts:
      - chatcmpl-UUID...  vLLM request IDs (our token_precise_bench.py path)
      - xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx  kvbench UUID IDs (aiperf path,
        where aiperf sends its own UUID as the completion ID)
    Rejects:
      - empty / None
      - 'no-load'  (our explicit no-op marker for gpu.copy_in)
      - 16-hex strings like '4e57878852a3e560'  (vLLM random IDs from no-op
        decode-step gpu.copy_out calls — these are NOT UUID format)
    """
    if not trace_id or trace_id == "no-load":
        return False
    return trace_id.startswith("chatcmpl-") or bool(_UUID_RE.match(trace_id))


# ── Stats ─────────────────────────────────────────────────────────────────────
def pct(sv: list[float], q: float) -> float:
    n = len(sv)
    return sv[max(0, min(n - 1, int(q * (n - 1))))]


def make_stats(vals: list[float], show_std: bool) -> dict | None:
    if not vals:
        return None
    sv = sorted(vals)
    d = {"mean": statistics.mean(sv), "p50": pct(sv, 0.5), "p90": pct(sv, 0.9), "n": len(sv)}
    if show_std:
        d["std"] = statistics.pstdev(sv) if len(sv) > 1 else 0.0
    return d


CHUNK_SIZE = 256   # from lmcache.yaml


def per_chunk_request(spans: list[dict], op: str, show_std: bool,
                      only_real: bool = True,
                      io_op: str | None = None) -> dict | None:
    """Per-chunk cost: total op duration / number of chunks, per request.

    Chunk count comes from attrs['chunks'] if present, else attrs['tokens']/CHUNK_SIZE,
    else counts io_op spans for the same trace_id (used for kv.load_init).
    """
    # Build io_op counts per trace_id if needed
    io_counts: dict[str, int] = {}
    if io_op:
        for s in spans:
            if s["name"] == io_op:
                tid = s.get("trace_id", "_")
                io_counts[tid] = io_counts.get(tid, 0) + 1

    by_req: dict[str, tuple[float, int]] = {}   # trace_id → (dur_ms, chunks)
    for s in spans:
        if s["name"] != op:
            continue
        tid = s.get("trace_id")
        if only_real and not is_real_request(tid):
            continue
        dur = s["dur_ms"]
        attrs = s.get("attrs", {})
        n_chunks = (attrs.get("chunks") or
                    (attrs.get("tokens", 0) // CHUNK_SIZE) or
                    io_counts.get(tid, 1) or 1)
        if tid not in by_req:
            by_req[tid or "_"] = (0.0, n_chunks)
        prev_dur, prev_n = by_req[tid or "_"]
        by_req[tid or "_"] = (prev_dur + dur, max(prev_n, n_chunks))

    vals = [dur / max(n, 1) for dur, n in by_req.values() if n > 0]
    return make_stats(vals, show_std)


def per_request(spans: list[dict], op: str, show_std: bool,
                only_real: bool = True) -> dict | None:
    """Sum op durations per trace_id, stats across requests."""
    by_req: dict[str, float] = defaultdict(float)
    for s in spans:
        if s["name"] != op:
            continue
        tid = s.get("trace_id")
        if only_real and not is_real_request(tid):
            continue
        by_req[tid or "_"] += s["dur_ms"]
    return make_stats(list(by_req.values()), show_std)


def per_span(spans: list[dict], op: str, show_std: bool,
             only_real: bool = True, min_ms: float = 0.0) -> dict | None:
    """Stats across individual span durations."""
    vals = []
    for s in spans:
        if s["name"] != op:
            continue
        if only_real and not is_real_request(s.get("trace_id")):
            continue
        if s["dur_ms"] >= min_ms:
            vals.append(s["dur_ms"])
    return make_stats(vals, show_std)


def assign_turns(spans: list[dict], n_turns: int) -> dict[str, int]:
    """Map each chatcmpl- trace_id to a turn index (0-based).

    Concurrency=1 means conversations run sequentially. Sorting all request
    trace_ids by their earliest span timestamp gives the request order.
    rank % n_turns gives the turn index within a conversation.
    """
    # Collect earliest ts per trace_id
    earliest: dict[str, float] = {}
    for s in spans:
        tid = s.get("trace_id", "")
        if not is_real_request(tid):
            continue
        t = s.get("ts", 0)
        if tid not in earliest or t < earliest[tid]:
            earliest[tid] = t
    ordered = sorted(earliest, key=lambda t: earliest[t])
    return {tid: rank % n_turns for rank, tid in enumerate(ordered)}


def per_span_by_turn(spans: list[dict], op: str, show_std: bool,
                      turn_map: dict[str, int], n_turns: int) -> dict[int, dict | None]:
    """Stats across individual span durations grouped by turn (for per-chunk ops)."""
    by_turn: dict[int, list[float]] = {i: [] for i in range(n_turns)}
    for s in spans:
        if s["name"] != op:
            continue
        tid = s.get("trace_id", "")
        if not is_real_request(tid) or tid not in turn_map:
            continue
        by_turn[turn_map[tid]].append(s["dur_ms"])
    return {i: make_stats(by_turn[i], show_std) for i in range(n_turns)}


def per_chunk_by_turn(spans: list[dict], op: str, show_std: bool,
                       turn_map: dict[str, int], n_turns: int,
                       io_op: str | None = None) -> dict[int, dict | None]:
    """Per-chunk stats grouped by turn: total op / chunk_count, per turn."""
    io_counts: dict[str, int] = {}
    if io_op:
        for s in spans:
            if s["name"] == io_op and is_real_request(s.get("trace_id", "")):
                tid = s.get("trace_id", "")
                io_counts[tid] = io_counts.get(tid, 0) + 1

    by_turn: dict[int, list[float]] = {i: [] for i in range(n_turns)}
    totals: dict[str, tuple[float, int]] = {}
    for s in spans:
        if s["name"] != op:
            continue
        tid = s.get("trace_id", "")
        if not is_real_request(tid) or tid not in turn_map:
            continue
        attrs = s.get("attrs", {})
        n_chunks = (attrs.get("chunks") or
                    (attrs.get("tokens", 0) // CHUNK_SIZE) or
                    io_counts.get(tid, 1) or 1)
        prev_dur, prev_n = totals.get(tid, (0.0, n_chunks))
        totals[tid] = (prev_dur + s["dur_ms"], max(prev_n, n_chunks))

    for tid, (dur, n_chunks) in totals.items():
        if tid in turn_map:
            by_turn[turn_map[tid]].append(dur / max(n_chunks, 1))

    return {i: make_stats(by_turn[i], show_std) for i in range(n_turns)}


def per_request_by_turn(spans: list[dict], op: str, show_std: bool,
                         turn_map: dict[str, int], n_turns: int
                         ) -> dict[int, dict | None]:
    """Per-request stats broken down by turn index."""
    by_turn: dict[int, dict[str, float]] = {i: defaultdict(float) for i in range(n_turns)}
    for s in spans:
        if s["name"] != op:
            continue
        tid = s.get("trace_id", "")
        if not is_real_request(tid) or tid not in turn_map:
            continue
        by_turn[turn_map[tid]][tid] += s["dur_ms"]
    return {i: make_stats(list(by_turn[i].values()), show_std) for i in range(n_turns)}


def decode_per_step_by_turn(spans: list[dict], show_std: bool,
                              turn_map: dict[str, int], n_turns: int
                              ) -> dict[int, dict | None]:
    """Decode per step broken down by turn index."""
    # kvbench path: decode_step spans
    step_by_turn: dict[int, list[float]] = {i: [] for i in range(n_turns)}
    for s in spans:
        if s["name"] != "decode_step":
            continue
        tid = s.get("trace_id", "")
        if not is_real_request(tid) or tid not in turn_map:
            continue
        step_by_turn[turn_map[tid]].append(s["dur_ms"])
    if any(step_by_turn[i] for i in range(n_turns)):
        return {i: make_stats(step_by_turn[i], show_std) for i in range(n_turns)}

    # vLLM OTLP path: total decode / completion_tokens
    completion: dict[str, int] = {}
    for s in spans:
        if s["name"] == "request" and is_real_request(s.get("trace_id", "")):
            ct = s.get("attrs", {}).get("gen_ai.usage.completion_tokens")
            if ct and ct > 1:
                completion[s["trace_id"]] = ct
    by_req: dict[str, float] = defaultdict(float)
    for s in spans:
        if s["name"] == "decode" and is_real_request(s.get("trace_id", "")):
            by_req[s["trace_id"]] += s["dur_ms"]
    per_tok_by_turn: dict[int, list[float]] = {i: [] for i in range(n_turns)}
    for tid, total_ms in by_req.items():
        if tid not in turn_map:
            continue
        ct = completion.get(tid)
        if ct and ct > 1:
            per_tok_by_turn[turn_map[tid]].append(total_ms / ct)
    return {i: make_stats(per_tok_by_turn[i], show_std) for i in range(n_turns)}


def decode_per_step(spans: list[dict], show_std: bool) -> dict | None:
    """
    Per-step decode time.
    - kvbench: mean of individual decode_step spans (one per token per request).
    - vLLM OTLP: total 'decode' span / actual completion_tokens from request attrs.
      Uses gen_ai.usage.completion_tokens so the divisor reflects what was
      actually generated, not what was requested.
    """
    # Try decode_step spans first (kvbench path)
    steps = [s["dur_ms"] for s in spans if s["name"] == "decode_step"
             and is_real_request(s.get("trace_id"))]
    if steps:
        return make_stats(steps, show_std)

    # vLLM OTLP path: pair 'decode' total with completion_tokens from 'request'
    completion: dict[str, int] = {}
    for s in spans:
        if s["name"] == "request" and is_real_request(s.get("trace_id")):
            ct = s.get("attrs", {}).get("gen_ai.usage.completion_tokens")
            if ct and ct > 1:
                completion[s["trace_id"]] = ct

    per_tok = []
    by_req: dict[str, float] = defaultdict(float)
    for s in spans:
        if s["name"] == "decode" and is_real_request(s.get("trace_id")):
            by_req[s["trace_id"]] += s["dur_ms"]

    for tid, total_ms in by_req.items():
        ct = completion.get(tid)
        if ct and ct > 1:
            per_tok.append(total_ms / ct)

    return make_stats(per_tok, show_std) if per_tok else None


# ── Formatting ────────────────────────────────────────────────────────────────
def fmt(v: float | None) -> str:
    if v is None:
        return "—"
    if v >= 1000:
        return f"{v/1000:.2f}s"
    if v >= 1:
        return f"{v:.2f}ms"
    return f"{v*1000:.0f}µs"


def fmt_mean_std(d: dict | None, show_std: bool) -> str:
    """Format mean with ±σ embedded: '14.3ms ±2.1ms'"""
    if d is None:
        return "—"
    m = fmt(d["mean"])
    if not show_std or d.get("std") is None:
        return m
    return f"{m} ±{fmt(d['std'])}"


def delta_cell(real: float | None, emul: float | None) -> tuple[str, object]:
    if real is None or emul is None or real == 0:
        return "—", C_MUTED
    d = (emul - real) / real * 100.0
    col = C_GREEN if d > 5 else (C_RED if d < -5 else colors.HexColor("#1a1a1a"))
    return f"{'+'if d>=0 else''}{d:.0f}%", col


def build_turn_table(metric_label: str,
                     vllm_by_turn: dict[int, dict | None],
                     kb_by_turn:   dict[int, dict | None],
                     n_turns: int, show_std: bool) -> Table:
    """Table with rows = Turn 1..N, same column layout as build_table."""
    rows = [{"label": f"Turn {i+1}",
             "real":  vllm_by_turn.get(i),
             "emul":  kb_by_turn.get(i)}
            for i in range(n_turns)]
    return build_table(rows, show_std)


# ── Metrics for a case ────────────────────────────────────────────────────────
def gather_metrics(vllm_spans: list[dict], kb_spans: list[dict],
                   is_multiturn: bool, show_std: bool,
                   per_chunk: bool = False) -> list[dict]:
    # kb/kb_span helpers — is_real_request() now accepts UUID format IDs
    def kb(op, **kw):      return per_request(kb_spans, op, show_std, **kw)
    def kb_span(op, **kw): return per_span(kb_spans,    op, show_std, **kw)
    """
    Returns list of metric rows, each:
      {"label": str, "real": stats|None, "emul": stats|None}
    Only includes a row when at least one side has data.
    Asymmetric ops: vLLM and kvbench may read different span names.
    """
    rows = []

    def add(label, rv, ev, required_both=False):
        if required_both and (rv is None or ev is None):
            return
        if rv is None and ev is None:
            return
        rows.append({"label": label, "real": rv, "emul": ev})

    # Prefill — OTLP clean GPU compute (vLLM), roofline (kvbench)
    add("Prefill",
        per_request(vllm_spans, "prefill", show_std),
        kb("prefill"))

    # Decode per step
    add("Decode  (per output token)",
        decode_per_step(vllm_spans, show_std),
        decode_per_step(kb_spans,   show_std))

    # kv.lookup — scheduler-side cache check
    add("kv.lookup",
        per_request(vllm_spans, "kv.lookup", show_std),
        kb("kv.lookup"))

    # kv.store — full wait_for_save pipeline
    add("kv.store",
        per_request(vllm_spans, "kv.store",  show_std),
        kb("rank.store"))

    # gpu.copy_out — directly below kv.store (true GPU→CPU DMA per chunk)
    add("gpu.copy_out",
        per_span(vllm_spans, "gpu.copy_out", show_std, only_real=True),
        kb_span("gpu.copy_out"))

    # gpu.alloc — only kvbench (vLLM pre-allocates at startup)
    add("gpu.alloc  (emul only)",
        None,
        kb("gpu.alloc"))

    # io.write — total per request (sum of all chunks written for that request)
    add("io.write",
        per_request(vllm_spans, "io.write", show_std, only_real=True),
        kb("io.write"))

    # kv.retrieve — present on cache hits
    add("kv.retrieve",
        per_request(vllm_spans, "kv.load_init", show_std, only_real=True),
        kb("rank.retrieve"),
        required_both=True)

    # gpu.copy_in — directly below kv.retrieve (CPU pinned → GPU HBM)
    add("gpu.copy_in",
        per_request(vllm_spans, "gpu.copy_in", show_std, only_real=True),
        kb("gpu.copy_in"),
        required_both=True)

    # io.read — total per request (sum of all chunks read for that request)
    add("io.read",
        per_request(vllm_spans, "io.read", show_std, only_real=True),
        kb("io.read"),
        required_both=True)

    # Per-chunk breakdown — only when --per-chunk flag is passed
    if per_chunk:
        add("kv.store / chunk",
            per_chunk_request(vllm_spans, "kv.store",   show_std),
            per_chunk_request(kb_spans, "rank.store", show_std))
        add("gpu.copy_out / chunk",
            per_chunk_request(vllm_spans, "gpu.copy_out", show_std,
                              only_real=True, io_op="io.write"),
            kb_span("gpu.copy_out", show_std, only_real=True))
        add("kv.retrieve / chunk",
            per_chunk_request(vllm_spans, "kv.load_init", show_std,
                              only_real=True, io_op="io.read"),
            per_chunk_request(kb_spans, "rank.retrieve", show_std,
                              only_real=True),
            required_both=True)
        add("io.read / chunk",
            per_span(vllm_spans, "io.read", show_std, only_real=False),
            kb_span("io.read", show_std, only_real=False),
            required_both=True)
        add("gpu.copy_in / chunk",
            per_chunk_request(vllm_spans, "gpu.copy_in", show_std,
                              only_real=True, io_op="io.read"),
            kb("gpu.copy_in", show_std, only_real=True),
            required_both=True)

    return rows


# ── Table builder ─────────────────────────────────────────────────────────────
def build_table(rows: list[dict], show_std: bool) -> Table:
    """
    Columns: Metric | mean±σ(real) | mean±σ(emul) | Δ | p50(real) | p50(emul) | Δ | p90(real) | p90(emul) | Δ
    σ is embedded in the mean cell as '14.3ms ±2.1ms' — no separate σ columns.
    """
    grp = ["Metric", "mean" + (" ± σ" if show_std else ""), "", "", "p50", "", "", "p90", "", ""]
    hdr = ["", "real (vLLM)", "emul (kvbench)", "Δ",
               "real", "emul", "Δ",
               "real", "emul", "Δ"]

    data_rows  = []
    delta_data = []

    for ri, rd in enumerate(rows, start=2):
        rv, ev = rd.get("real"), rd.get("emul")
        dm, dmc = delta_cell(rv["mean"] if rv else None, ev["mean"] if ev else None)
        d5, d5c = delta_cell(rv["p50"]  if rv else None, ev["p50"]  if ev else None)
        d9, d9c = delta_cell(rv["p90"]  if rv else None, ev["p90"]  if ev else None)

        delta_data += [(ri, 3, dmc), (ri, 6, d5c), (ri, 9, d9c)]

        row = [rd["label"],
               fmt_mean_std(rv, show_std), fmt_mean_std(ev, show_std), dm,
               fmt(rv["p50"] if rv else None), fmt(ev["p50"] if ev else None), d5,
               fmt(rv["p90"] if rv else None), fmt(ev["p90"] if ev else None), d9]
        data_rows.append(row)

    # mean±σ columns are wider since they contain more text
    cw = AVAIL_W / 11.0
    metric_w  = cw * 2.8
    mean_w    = cw * 1.5   # wider for "14.3ms ±2.1ms"
    delta_w   = cw * 0.8
    val_w     = cw

    widths = [metric_w, mean_w, mean_w, delta_w, val_w, val_w, delta_w, val_w, val_w, delta_w]

    tbl = Table([grp, hdr] + data_rows, colWidths=widths, repeatRows=2)

    cmds = [
        ("FONTNAME",      (0,0), (-1,1),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,-1), 8),
        ("BACKGROUND",     (0,0), (-1,0),  C_NAVY),
        ("TEXTCOLOR",      (0,0), (-1,0),  C_WHITE),
        ("BACKGROUND",     (0,1), (-1,1),  colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",      (0,1), (-1,1),  C_WHITE),
        ("LINEBELOW",      (0,1), (-1,1),  0.8, C_BLUE),
        ("ROWBACKGROUND",  (0,2), (-1,-1), [C_WHITE, C_ALT]),
        ("FONTNAME",       (0,2), (0,-1),  "Helvetica-Bold"),
        ("GRID",           (0,0), (-1,-1), 0.25, C_GREY),
        ("ALIGN",          (0,0), (-1,-1), "CENTER"),
        ("ALIGN",          (0,0), (0,-1),  "LEFT"),
        ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",     (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
        ("LEFTPADDING",    (0,0), (-1,-1), 5),
        ("RIGHTPADDING",   (0,0), (-1,-1), 5),
        # group header band colours
        ("BACKGROUND",     (1,0), (3,0),   C_BLUE),
        ("BACKGROUND",     (4,0), (6,0),   C_NAVY),
        ("BACKGROUND",     (7,0), (9,0),   C_BLUE),
    ]
    # no extra σ columns — std is embedded in mean cells

    for r, c, col in delta_data:
        cmds += [
            ("BACKGROUND", (c,r),(c,r), C_DELTA),
            ("TEXTCOLOR",  (c,r),(c,r), col),
            ("FONTNAME",   (c,r),(c,r), "Helvetica-Bold"),
        ]
    tbl.setStyle(TableStyle(cmds))
    return tbl


# ── Page decoration ───────────────────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    w, h = landscape(A4)
    canvas.setFillColor(C_NAVY); canvas.rect(0, h-24, w, 24, fill=1, stroke=0)
    canvas.setFillColor(C_WHITE); canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(MARGIN, h-15, "Performance Comparison Report  ·  vLLM (real) vs kvbench (emulated)")
    canvas.setFont("Helvetica", 8.5)
    canvas.drawRightString(w-MARGIN, h-15, f"{HARDWARE}  ·  {MODEL}")
    canvas.setFillColor(C_NAVY); canvas.rect(0,0,w,16,fill=1,stroke=0)
    canvas.setFillColor(C_WHITE); canvas.setFont("Helvetica", 7)
    canvas.drawString(MARGIN, 5,
        "Δ = (kvbench − vLLM) / vLLM × 100%  ·  "
        "Green Δ = kvbench over-predicts  ·  Red Δ = kvbench under-predicts  ·  "
        "gpu spans filtered to real requests (chatcmpl- trace_id) only")
    canvas.drawRightString(w-MARGIN, 5, f"Page {doc.page}")
    canvas.restoreState()


# ── Cases ─────────────────────────────────────────────────────────────────────
CASES = [
    {"tag": "seq_100",    "label": "Sequential — 100 requests  (concurrency=1)",
     "multiturn": False, "n_turns": 5,
     "note": "100 requests · 2048 total input (256 shared prefix + 1792 unique) · 64 output · "
             "concurrency=1 · prefix caching OFF · shared prefix triggers io.read on requests 2-100"},
    {"tag": "seq_1000",   "label": "Sequential — 1000 requests  (concurrency=1)",
     "multiturn": False, "n_turns": 5,
     "note": "1000 requests · 2048 total input (256 shared prefix + 1792 unique) · 64 output · concurrency=1"},
    {"tag": "conc_5",     "label": "High Concurrency — 100 requests  (concurrency=5)",
     "multiturn": False, "n_turns": 5,
     "note": "100 requests · 2048 total input (256 shared prefix + 1792 unique) · 64 output · concurrency=5"},
    {"tag": "conc_10",    "label": "High Concurrency — 100 requests  (concurrency=10)",
     "multiturn": False, "n_turns": 5,
     "note": "100 requests · 2048 total input (256 shared prefix + 1792 unique) · 64 output · concurrency=10"},
    {"tag": "mt_10c_5t",  "label": "Multi-turn — 10 conversations × 5 turns",
     "multiturn": True,   "n_turns": 5,
     "note": "10 conversations · 5 turns · 512 input tokens/turn · 64 output tokens/turn · concurrency=1"},
    {"tag": "mt_50c_5t",  "label": "Multi-turn — 50 conversations × 5 turns",
     "multiturn": True,   "n_turns": 5,
     "note": "50 conversations · 5 turns · 512 input tokens/turn · 64 output tokens/turn · concurrency=1"},
]


# ── Build ─────────────────────────────────────────────────────────────────────
def build(vllm_dir: Path, kb_dir: Path, out: Path, show_std: bool,
          vllm_mt_dir: Path | None = None, kb_mt_dir: Path | None = None,
          vllm_conc_dir: Path | None = None, kb_conc_dir: Path | None = None,
          per_chunk: bool = False) -> None:
    """vllm_mt_dir / kb_mt_dir override for multi-turn; *_conc_dir for concurrency cases."""
    doc = SimpleDocTemplate(str(out), pagesize=landscape(A4),
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN+22, bottomMargin=MARGIN+10)
    story = []

    # Cover
    story += [
        Spacer(1, 1.5*cm),
        Paragraph("Comparison between KV-Bench (Emulated) and vLLM Real Inference Server", S["title"]),
        HRFlowable(width="100%", thickness=2, color=C_BLUE, spaceAfter=14),
        Spacer(1, 0.3*cm),
    ]
    meta = [
        [Paragraph("Hardware",     S["meta_k"]), Paragraph(HARDWARE, S["meta_v"])],
        [Paragraph("Model",        S["meta_k"]), Paragraph(MODEL,    S["meta_v"])],
        [Paragraph("Date",         S["meta_k"]), Paragraph(datetime.now().strftime("%B %d, %Y"), S["meta_v"])],
        [Paragraph("vLLM mode",    S["meta_k"]), Paragraph("Single-process  (VLLM_ENABLE_V1_MULTIPROCESSING=0)  ·  prefix caching OFF", S["meta_v"])],
        [Paragraph("vLLM tracing", S["meta_k"]), Paragraph("OTLP request spans + TracingLMCacheConnector KV spans", S["meta_v"])],
        [Paragraph("kvbench",      S["meta_k"]), Paragraph("Calibrated roofline  ·  GB10 profile  ·  same lmcache config", S["meta_v"])],
    ]
    mt = Table(meta, colWidths=[3.8*cm, AVAIL_W-3.8*cm])
    mt.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LINEBELOW",(0,0),(-1,-2),0.3,C_GREY),
    ]))
    story += [mt, Spacer(1,0.5*cm)]
    story += [
        Paragraph(
            "<b>Δ colour:</b>  "
            "<font color='#1a7a3c'>Green</font> = kvbench over-predicts (conservative).  "
            "<font color='#a93226'>Red</font> = kvbench under-predicts (optimistic).  "
            "No colour = within ±5%.  "
            "gpu.copy_in / gpu.copy_out filtered to real request spans only — "
            "no-op per-decode-step calls excluded.",
            S["note"]),
        PageBreak(),
    ]

    for i, case in enumerate(CASES, 1):
        tag = case["tag"]
        # Use override dirs for multi-turn cases if provided
        is_conc = case["tag"].startswith("conc_")
        _vd = (vllm_conc_dir if (is_conc and vllm_conc_dir) else
               vllm_mt_dir   if (case["multiturn"] and vllm_mt_dir) else vllm_dir)
        _kd = (kb_conc_dir   if (is_conc and kb_conc_dir) else
               kb_mt_dir     if (case["multiturn"] and kb_mt_dir) else kb_dir)
        vd  = _vd / tag
        kd  = _kd / tag
        if not vd.exists() or not kd.exists():
            continue

        # Merge vLLM files (request-trace + kv-trace) on shared chatcmpl- trace_ids
        vllm_files = sorted(vd.glob("*.jsonl"))
        kb_files   = sorted(kd.glob("kvbench-trace.jsonl"))

        vllm_spans = load_and_merge(vllm_files)
        kb_spans   = load_and_merge(kb_files)

        story.append(Paragraph(f"{i}.  {case['label']}", S["h1"]))
        story.append(Paragraph(case["note"], S["note"]))

        if not case["multiturn"]:
            rows = gather_metrics(vllm_spans, kb_spans, False, show_std, per_chunk)
            story.append(build_table(rows, show_std))

        else:
            n_turns = 5
            # Turn assignment — same for both systems: sort by ts, rank % n_turns
            vllm_turns = assign_turns(vllm_spans, n_turns)
            kb_turns   = assign_turns(kb_spans,   n_turns)

            # Per-turn prefill table
            story.append(Paragraph("<b>Prefill</b>", S["note"]))
            story.append(build_turn_table(
                "Prefill",
                per_request_by_turn(vllm_spans, "prefill", show_std, vllm_turns, n_turns),
                per_request_by_turn(kb_spans,   "prefill", show_std, kb_turns,   n_turns),
                n_turns, show_std))
            story.append(Spacer(1, 8))

            # Per-turn decode-per-step table
            story.append(Paragraph("<b>Decode</b>  (per output token)", S["note"]))
            story.append(build_turn_table(
                "Decode/step",
                decode_per_step_by_turn(vllm_spans, show_std, vllm_turns, n_turns),
                decode_per_step_by_turn(kb_spans,   show_std, kb_turns,   n_turns),
                n_turns, show_std))
            story.append(Spacer(1, 8))

            # Per-turn storage metrics — only ops with chatcmpl- trace_ids.
            # io.read/io.write have random trace_ids (async worker threads)
            # and cannot be grouped per-turn; they appear in the flat table.
            for op_label, op_name_v, op_name_k, required_both in [
                ("kv.lookup",          "kv.lookup",    "kv.lookup",    False),
                ("kv.store / chunk",   "kv.store",     "rank.store",   False),
                ("gpu.copy_out / chunk","gpu.copy_out", "gpu.copy_out", False),
                ("io.read  / chunk",   "io.read",      "io.read",      True),
                ("gpu.copy_in / chunk","gpu.copy_in",  "gpu.copy_in",  True),
            ]:
                # For per-chunk ops: divide per-request total by chunk count
                is_per_chunk = "/ chunk" in op_label
                if is_per_chunk:
                    vllm_by_turn = per_chunk_by_turn(
                        vllm_spans, op_name_v, show_std, vllm_turns, n_turns,
                        io_op="io.read" if "copy_in" in op_label or "retrieve" in op_label else "io.write")
                    kb_by_turn = per_span_by_turn(
                        kb_spans, op_name_k, show_std, kb_turns, n_turns)
                else:
                    vllm_by_turn = per_request_by_turn(vllm_spans, op_name_v, show_std, vllm_turns, n_turns)
                    kb_by_turn   = per_request_by_turn(kb_spans,   op_name_k, show_std, kb_turns,   n_turns)
                # required_both: skip if neither side has any data
                has_vllm = any(vllm_by_turn[i] for i in range(n_turns))
                has_kb   = any(kb_by_turn[i]   for i in range(n_turns))
                if required_both and not (has_vllm and has_kb):
                    continue
                if not (has_vllm or has_kb):
                    continue
                story.append(Paragraph(f"<b>{op_label}</b>", S["note"]))
                story.append(build_turn_table(op_label, vllm_by_turn, kb_by_turn, n_turns, show_std))
                story.append(Spacer(1, 6))

            story.append(Paragraph(
                "io.read shown only when cache hits occurred in both systems.  "
                "decode time uses actual completion_tokens from each server's response.",
                S["note"]))

        story.append(PageBreak())

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Report saved → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-dir",     required=True,
                    help="vLLM traces dir (used for sequential cases)")
    ap.add_argument("--kvbench-dir",  required=True,
                    help="kvbench traces dir (used for sequential cases)")
    ap.add_argument("--vllm-mt-dir",    default=None,
                    help="vLLM traces dir for multi-turn cases")
    ap.add_argument("--kvbench-mt-dir", default=None,
                    help="kvbench traces dir for multi-turn cases")
    ap.add_argument("--vllm-conc-dir",    default=None,
                    help="vLLM traces dir for concurrency cases")
    ap.add_argument("--kvbench-conc-dir", default=None,
                    help="kvbench traces dir for concurrency cases")
    ap.add_argument("--output", "-o", default=None)
    ap.add_argument("--no-stddev",  action="store_true", help="Omit ±σ from mean cells")
    ap.add_argument("--per-chunk",  action="store_true",
                    help="Add per-chunk breakdown rows (kv.store/chunk, io.read/chunk etc.)")
    ap.add_argument("--name", "-n", default=None,
                    help="Report name used in filename, e.g. odirect_1 → comparison_odirect_1.pdf")
    args = ap.parse_args()

    vllm_dir    = Path(args.vllm_dir).expanduser()
    kb_dir      = Path(args.kvbench_dir).expanduser()
    vllm_mt_dir   = Path(args.vllm_mt_dir).expanduser()      if args.vllm_mt_dir      else None
    kb_mt_dir     = Path(args.kvbench_mt_dir).expanduser()   if args.kvbench_mt_dir   else None
    vllm_conc_dir = Path(args.vllm_conc_dir).expanduser()    if args.vllm_conc_dir    else None
    kb_conc_dir   = Path(args.kvbench_conc_dir).expanduser() if args.kvbench_conc_dir else None
    suffix = args.name or datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path(args.output).expanduser() if args.output else \
          Path("~/aditya/traces").expanduser() / f"comparison_{suffix}.pdf"

    build(vllm_dir, kb_dir, out, show_std=not args.no_stddev,
          vllm_mt_dir=vllm_mt_dir, kb_mt_dir=kb_mt_dir,
          vllm_conc_dir=vllm_conc_dir, kb_conc_dir=kb_conc_dir,
          per_chunk=args.per_chunk)


if __name__ == "__main__":
    main()
