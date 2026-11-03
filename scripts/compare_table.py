#!/usr/bin/env python3
"""
compare_table.py — print the real-vLLM vs kvbench comparison as plain tables.

Uses the same loading, filtering and metric rows as compare_report.py, so the
numbers match the PDF, but writes to the terminal instead of building a report.
Needs the same Python environment as compare_report.py (it imports reportlab).

Usage:
    python scripts/compare_table.py --vllm-dir DIR --kvbench-dir DIR
    python scripts/compare_table.py --vllm-dir DIR --kvbench-dir DIR --cases seq_10,seq_100

DIR holds one sub-directory per case (seq_10, seq_100, ...). The two DIRs may
be the same directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import compare_report as cr

HEADER = ["Metric",
          "real mean", "emul mean", "Δ",
          "real p50", "emul p50", "Δ",
          "real p90", "emul p90", "Δ",
          "n real/emul"]


def delta(real: float | None, emul: float | None) -> str:
    if real is None or emul is None or real == 0:
        return "-"
    return f"{(emul - real) / real * 100.0:+.0f}%"


def cell(stats: dict | None, key: str) -> str:
    return "-" if stats is None else cr.fmt(stats[key])


def row_cells(row: dict) -> list[str]:
    rv, ev = row["real"], row["emul"]
    out = [row["label"]]
    for key in ("mean", "p50", "p90"):
        out += [cell(rv, key), cell(ev, key),
                delta(rv[key] if rv else None, ev[key] if ev else None)]
    out.append(f"{rv['n'] if rv else '-'}/{ev['n'] if ev else '-'}")
    return out


def render(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for n, r in enumerate(rows):
        cells = [r[0].ljust(widths[0])] + [c.rjust(w) for c, w in zip(r[1:], widths[1:])]
        lines.append("  ".join(cells))
        if n == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vllm-dir", required=True, help="traces dir with the real vLLM runs")
    ap.add_argument("--kvbench-dir", required=True, help="traces dir with the kvbench runs")
    ap.add_argument("--cases", default=None,
                    help="comma-separated case tags (default: every case found)")
    args = ap.parse_args()

    vllm_dir = Path(args.vllm_dir).expanduser()
    kb_dir = Path(args.kvbench_dir).expanduser()
    wanted = set(args.cases.split(",")) if args.cases else None

    shown = 0
    for case in cr.CASES:
        tag = case["tag"]
        if wanted is not None and tag not in wanted:
            continue
        vd, kd = vllm_dir / tag, kb_dir / tag
        if not vd.exists() or not kd.exists():
            if wanted is not None:
                print(f"== {tag}: directory missing ({vd} / {kd})\n")
            continue

        vllm_spans = cr.load_and_merge(cr.vllm_trace_files(vd))
        kb_spans = cr.load_and_merge(sorted(kd.glob("kvbench-trace.jsonl")))
        rows = cr.gather_metrics(vllm_spans, kb_spans, case["multiturn"], False)

        print(f"== {tag}: {case['label']}")
        if not rows:
            print("   (no data)\n")
            continue
        print(render([HEADER] + [row_cells(r) for r in rows]))
        print()
        shown += 1

    if shown:
        print("Δ = (kvbench - vLLM) / vLLM. gpu spans are limited to real requests. "
              "Multi-turn cases are shown flat (all turns together).")
    elif wanted is None:
        print("No case directories found under the given paths.")


if __name__ == "__main__":
    main()
