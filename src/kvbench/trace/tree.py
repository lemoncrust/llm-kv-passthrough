"""
Assemble a span trace into conversations -> requests (turns) -> stage trees.

The server writes spans as flat JSON Lines (see `kvbench.trace.spans`),
children before parents and with LMCache worker-thread I/O arriving after
the request that caused it. This module rebuilds the trees for analysis:

    Trace
    └── Conversation            synthesized from requests sharing a
        └── Turn (request)      conversation_id, ordered by turn_index
            └── Node tree       tokenize, kv.lookup, kv.retrieve [rank.retrieve
                                [lmcache.block_map, lmcache.batched_get
                                [io.read ...]]], prefill.compute, kv.store
                                [rank.store [lmcache.put [io.write] ...]],
                                decode [decode.step ...], response.end

Requests without a conversation id each form a single-turn conversation
(conversation_id None). Roots that are not requests (backend_info marks,
I/O that could not be attributed) are kept as ``orphans`` so nothing in
the file is silently dropped.

Use `load(path)` for Python analysis; `kvbench trace tree` dumps the same
structure as nested JSON.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar


@dataclass
class Node:
    """One span with its children (sorted by start time)."""

    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    ts: float
    start_ms: float
    dur_ms: float
    thread: str
    attrs: dict[str, Any]
    children: list[Node] = field(default_factory=list)

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.dur_ms

    def child(self, name: str) -> Node | None:
        """First direct child named ``name``."""
        return next((c for c in self.children if c.name == name), None)

    def walk(self) -> Iterator[Node]:
        """This node and all descendants, depth-first in start order."""
        yield self
        for c in self.children:
            yield from c.walk()

    def find(self, name: str) -> list[Node]:
        """All descendants (any depth) named ``name``, in start order."""
        return [n for n in self.walk() if n is not self and n.name == name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "trace_id": self.trace_id,
            "ts": self.ts,
            "start_ms": self.start_ms,
            "dur_ms": self.dur_ms,
            "thread": self.thread,
            "attrs": self.attrs,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class Turn:
    """One request: the ``request`` root span and everything under it."""

    root: Node

    @property
    def trace_id(self) -> str:
        return self.root.trace_id

    @property
    def conversation_id(self) -> str | None:
        value = self.root.attrs.get("conversation_id")
        return str(value) if value is not None else None

    @property
    def turn_index(self) -> int | None:
        value = self.root.attrs.get("turn_index")
        return int(value) if value is not None else None

    @property
    def previous_request_id(self) -> str | None:
        value = self.root.attrs.get("previous_request_id")
        return str(value) if value is not None else None

    @property
    def ttft_ms(self) -> float | None:
        """Server-side time to first chunk (streaming requests only)."""
        value = self.root.attrs.get("first_chunk_ms")
        return float(value) if value is not None else None

    def stage(self, name: str) -> Node | None:
        """A top-level processing stage (direct child of the request)."""
        return self.root.child(name)

    def stage_durations(self) -> dict[str, float]:
        """Duration of each top-level stage, in processing order."""
        return {c.name: c.dur_ms for c in self.root.children if c.dur_ms > 0}

    def served_from(self) -> dict[str, int]:
        """Chunks served per tier on this turn's retrieve (from block_map marks)."""
        out: dict[str, int] = defaultdict(int)
        for mark in self.root.find("lmcache.block_map"):
            out[str(mark.attrs.get("tier"))] += int(mark.attrs.get("n", 0))
        return dict(out)

    def io_ms(self, op: str) -> float:
        """Summed duration of ``io.read`` / ``io.write`` spans under this request."""
        return sum(n.dur_ms for n in self.root.find(f"io.{op}"))

    def to_dict(self) -> dict[str, Any]:
        return self.root.to_dict()


@dataclass
class Conversation:
    """Turns sharing a conversation id, in turn order."""

    conversation_id: str | None
    turns: list[Turn]

    @property
    def start_ms(self) -> float:
        return min(t.root.start_ms for t in self.turns)

    @property
    def end_ms(self) -> float:
        return max(t.root.end_ms for t in self.turns)

    @property
    def dur_ms(self) -> float:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turns": len(self.turns),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "dur_ms": round(self.dur_ms, 3),
            "requests": [t.to_dict() for t in self.turns],
        }


@dataclass
class Trace:
    """A whole trace file, assembled."""

    conversations: list[Conversation]
    orphans: list[Node]
    span_count: int

    @property
    def turns(self) -> list[Turn]:
        """All requests across conversations, in start order."""
        return sorted((t for c in self.conversations for t in c.turns), key=lambda t: t.root.start_ms)

    def conversation(self, conversation_id: str) -> Conversation | None:
        """The conversation with this id, or with this unique id prefix."""
        return _match(
            conversation_id,
            [c for c in self.conversations if c.conversation_id is not None],
            lambda c: c.conversation_id or "",
        )

    def turn(self, trace_id: str) -> Turn | None:
        """The request with this trace id, or with this unique id prefix."""
        return _match(trace_id, self.turns, lambda t: t.trace_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spans": self.span_count,
            "conversations": [c.to_dict() for c in self.conversations],
            "orphans": [n.to_dict() for n in self.orphans],
        }


_T = TypeVar("_T")


def _match(wanted: str, items: list[_T], key: Callable[[_T], str]) -> _T | None:
    """Exact id match, else the single item whose id starts with ``wanted``.

    Ids are UUID-length; an unambiguous prefix is accepted so people need
    not paste whole ids. An ambiguous prefix raises rather than guessing.
    """
    exact = [item for item in items if key(item) == wanted]
    if exact:
        return exact[0]
    prefixed = [item for item in items if key(item).startswith(wanted)]
    if len(prefixed) > 1:
        raise ValueError(
            f"id prefix {wanted!r} is ambiguous: {sorted(key(i) for i in prefixed)[:5]}"
        )
    return prefixed[0] if prefixed else None


def _node(record: dict[str, Any]) -> Node:
    return Node(
        name=str(record.get("name", "")),
        trace_id=str(record.get("trace_id", "")),
        span_id=str(record.get("span_id", "")),
        parent_id=record.get("parent_id"),
        ts=float(record.get("ts", 0.0)),
        start_ms=float(record.get("start_ms", 0.0)),
        dur_ms=float(record.get("dur_ms", 0.0)),
        thread=str(record.get("thread", "")),
        attrs=dict(record.get("attrs") or {}),
    )


def build(records: Iterable[dict[str, Any]]) -> Trace:
    """Assemble span records (any order) into a Trace."""
    nodes: dict[str, Node] = {}
    all_nodes: list[Node] = []
    for record in records:
        if "span_id" not in record or "name" not in record:
            continue  # not a span record (e.g. legacy flat events)
        node = _node(record)
        nodes[node.span_id] = node
        all_nodes.append(node)

    roots: list[Node] = []
    for node in all_nodes:
        parent = nodes.get(node.parent_id) if node.parent_id else None
        if parent is None:
            roots.append(node)
        else:
            parent.children.append(node)
    for node in all_nodes:
        node.children.sort(key=lambda n: (n.start_ms, n.ts))

    turns = [Turn(n) for n in roots if n.name == "request"]
    orphans = [n for n in roots if n.name != "request"]

    grouped: dict[str | None, list[Turn]] = defaultdict(list)
    singles: list[Turn] = []
    for turn in turns:
        if turn.conversation_id is None:
            singles.append(turn)
        else:
            grouped[turn.conversation_id].append(turn)

    conversations: list[Conversation] = []
    for conversation_id, members in grouped.items():
        members.sort(key=lambda t: (t.turn_index if t.turn_index is not None else -1, t.root.start_ms))
        conversations.append(Conversation(conversation_id, members))
    conversations.extend(Conversation(None, [t]) for t in singles)
    conversations.sort(key=lambda c: c.start_ms)
    orphans.sort(key=lambda n: n.start_ms)
    return Trace(conversations=conversations, orphans=orphans, span_count=len(all_nodes))


def load(path: Path | str) -> Trace:
    """Load a span trace JSONL file and assemble it (skips unparseable lines)."""
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return build(records)
