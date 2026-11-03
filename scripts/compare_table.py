#!/usr/bin/env python3
"""
compare_table.py — print the real-vLLM vs kvbench comparison as plain tables.

Uses the same loading, filtering and metric rows as compare_report.py, so the
numbers match the PDF, but writes to the terminal instead of building a report.
Needs the same Python environment as compare_report.py (it imports reportlab).

Usage:
    python scripts/compare_table.py RUN_DIR
    python scripts/compare_table.py RUN_DIR --cases seq_10,seq_100

RUN_DIR is a whole run directory. Every sub-directory with traces (seq_100,
conc_5, mt_10c_5t, ...) is shown, so only the cases that were run appear. In
each one, vLLM spans come from the vllm-*.jsonl files and kvbench spans from
kvbench-trace.jsonl. If the kvbench traces live in a different run directory,
pass it with --kvbench-dir.
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


def case_tags(run_dir: Path, kb_dir: Path) -> list[str]:
    """Case sub-directories that hold traces, known cases first in report order."""
    found = set()
    for base in {run_dir, kb_dir}:
        found |= {p.name for p in base.iterdir() if p.is_dir() and any(p.glob("*.jsonl"))}
    known = [c["tag"] for c in cr.CASES if c["tag"] in found]
    return known + sorted(found - set(known))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", help="run directory containing one sub-directory per case")
    ap.add_argument("--kvbench-dir", default=None,
                    help="run directory holding the kvbench traces (default: RUN_DIR)")
    ap.add_argument("--cases", default=None,
                    help="comma-separated case tags to show (default: every case found)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    kb_dir = Path(args.kvbench_dir).expanduser() if args.kvbench_dir else run_dir
    for d in {run_dir, kb_dir}:
        if not d.is_dir():
            ap.error(f"not a directory: {d}")
    wanted = set(args.cases.split(",")) if args.cases else None
    labels = {c["tag"]: c["label"] for c in cr.CASES}

    shown = 0
    for tag in case_tags(run_dir, kb_dir):
        if wanted is not None and tag not in wanted:
            continue
        vd, kd = run_dir / tag, kb_dir / tag
        vllm_spans = cr.load_and_merge(cr.vllm_trace_files(vd) if vd.is_dir() else [])
        kb_spans = cr.load_and_merge(sorted(kd.glob("kvbench-trace.jsonl")) if kd.is_dir() else [])
        rows = cr.gather_metrics(vllm_spans, kb_spans, tag.startswith("mt_"), False)

        print(f"== {tag}: {labels.get(tag, tag)}")
        if not rows:
            print("   (no data)\n")
            continue
        print(render([HEADER] + [row_cells(r) for r in rows]))
        print()
        shown += 1

    if wanted is not None:
        for tag in sorted(wanted - set(case_tags(run_dir, kb_dir))):
            print(f"== {tag}: no traces under {run_dir}")
    if shown:
        print("Δ = (kvbench - vLLM) / vLLM. gpu spans are limited to real requests. "
              "Multi-turn cases are shown flat (all turns together).")
    elif wanted is None:
        print(f"No case directories with traces found under {run_dir}.")


if __name__ == "__main__":
    main()
