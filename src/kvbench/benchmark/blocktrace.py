"""Bounded block-issue samples: distinguish file layout from actual LBA order."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any


def validate_trace_log(text: str) -> None:
    if "failed" in text.lower():
        raise RuntimeError("blktrace reported an error despite its exit status; inspect blktrace.log")
    dropped = re.findall(r"dropped\s+(\d+)", text, flags=re.IGNORECASE)
    if not dropped or any(int(count) for count in dropped):
        raise RuntimeError("Missing or nonzero blktrace dropped-event count; LBA sample is incomplete")


def summarize_issues(path: Path) -> dict[str, Any]:
    """Parse blkparse's '%T.%t %a %d %S %n' on ONE namespace.

    Sector units in the Linux block trace ABI are always 512 bytes, irrespective
    of the namespace's formatted logical block size. This measures issue order,
    not physical NAND placement or per-file stream adjacency.
    """
    count = total = sequential = backwards = run_bytes = max_run = 0
    end_sector: int | None = None
    sizes: Counter[int] = Counter()
    with path.open() as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 5 or fields[1] != "D" or "W" not in fields[2]:
                continue
            sector, sectors = int(fields[3]), int(fields[4])
            if sectors <= 0:  # flush/barrier commands carry no payload
                continue
            size = sectors * 512
            count += 1
            total += size
            sizes[size] += 1
            if end_sector is not None and sector == end_sector:
                sequential += 1
                run_bytes += size
            else:
                if end_sector is not None and sector < end_sector:
                    backwards += 1
                run_bytes = size
            max_run = max(max_run, run_bytes)
            end_sector = sector + sectors
    return {"scope": "sampled block write issue order; not NAND placement or per-file streams",
            "write_ops": count, "write_bytes": total, "write_size_counts": dict(sizes),
            "adjacent_write_transitions": sequential, "backward_write_transitions": backwards,
            "sequential_transition_fraction": sequential / (count - 1) if count > 1 else None,
            "max_contiguous_run_bytes": max_run}
