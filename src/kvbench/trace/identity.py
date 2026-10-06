"""
Request and conversation identity for tracing.

Trace ids come from the client when it sends them: AIPerf sets
``X-Request-ID`` per request and ``X-Correlation-ID`` per session
(conversation) and exports both in ``profile_export.jsonl``
(``metadata.x_request_id`` / ``x_correlation_id``), so server traces join
to the load generator's per-request records exactly, with no time-window
guessing. Without headers the request is identified by its generated
completion id and carries no conversation attributes — continuation is
never inferred from prompt content.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

REQUEST_ID_HEADER = "x-request-id"
CONVERSATION_ID_HEADER = "x-correlation-id"


@dataclass(frozen=True)
class RequestIdentity:
    """Who a request is, for the root span of its trace.

    Attributes:
        request_id: The trace id (client-supplied header or completion id).
        completion_id: The OpenAI completion id returned to the client.
        conversation_id: Client-supplied session id, if any.
        turn_index: 0-based position within the conversation, if known.
        previous_request_id: The conversation's previous request, if any.
        client_request_id: Whether request_id came from the client.
    """

    request_id: str
    completion_id: str
    conversation_id: str | None = None
    turn_index: int | None = None
    previous_request_id: str | None = None
    client_request_id: bool = False

    def span_attrs(self) -> dict[str, Any]:
        """Root-span attributes (conversation fields only when present)."""
        attrs: dict[str, Any] = {
            "completion_id": self.completion_id,
            "client_request_id": self.client_request_id,
        }
        if self.conversation_id is not None:
            attrs["conversation_id"] = self.conversation_id
            attrs["turn_index"] = self.turn_index
            attrs["previous_request_id"] = self.previous_request_id
        return attrs


@dataclass
class _Conversation:
    turns: int
    last_request_id: str
    last_seen: float


class ConversationTracker:
    """Assigns turn numbers to requests sharing a conversation id.

    Keeps (turn count, last request id) per conversation; conversations
    idle longer than ``idle_seconds`` are forgotten, and the oldest are
    dropped past ``max_conversations`` so memory stays bounded under long
    benchmarks.
    """

    def __init__(self, idle_seconds: float = 900.0, max_conversations: int = 100_000) -> None:
        self.idle_seconds = idle_seconds
        self.max_conversations = max_conversations
        self._conversations: dict[str, _Conversation] = {}
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def next_turn(self, conversation_id: str, request_id: str) -> tuple[int, str | None]:
        """Register ``request_id`` as the conversation's next turn.

        Returns:
            (turn_index, previous_request_id) for this request.
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_sweep > min(60.0, self.idle_seconds):
                self._sweep(now)
            entry = self._conversations.pop(conversation_id, None)
            if entry is None:
                turn, previous = 0, None
            else:
                turn, previous = entry.turns, entry.last_request_id
            # Re-insert so dict order is least-recently-seen first
            self._conversations[conversation_id] = _Conversation(turn + 1, request_id, now)
            while len(self._conversations) > self.max_conversations:
                self._conversations.pop(next(iter(self._conversations)))
        return turn, previous

    def _sweep(self, now: float) -> None:
        cutoff = now - self.idle_seconds
        stale = [cid for cid, c in self._conversations.items() if c.last_seen < cutoff]
        for cid in stale:
            del self._conversations[cid]
        self._last_sweep = now

    def __len__(self) -> int:
        return len(self._conversations)


def identify(
    headers: Mapping[str, str] | None,
    completion_id: str,
    tracker: ConversationTracker | None = None,
) -> RequestIdentity:
    """Derive a request's identity from its HTTP headers.

    Args:
        headers: Request headers (any case); None when unavailable.
        completion_id: The generated completion id for this request.
        tracker: Assigns turn numbers when a conversation header is present.
    """
    lowered = {k.lower(): v for k, v in headers.items()} if headers else {}
    client_id = lowered.get(REQUEST_ID_HEADER) or None
    conversation_id = lowered.get(CONVERSATION_ID_HEADER) or None
    request_id = client_id or completion_id
    turn_index: int | None = None
    previous: str | None = None
    if conversation_id is not None and tracker is not None:
        turn_index, previous = tracker.next_turn(conversation_id, request_id)
    return RequestIdentity(
        request_id=request_id,
        completion_id=completion_id,
        conversation_id=conversation_id,
        turn_index=turn_index,
        previous_request_id=previous,
        client_request_id=client_id is not None,
    )
