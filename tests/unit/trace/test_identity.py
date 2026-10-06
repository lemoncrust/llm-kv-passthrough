"""Tests for request/conversation identity."""

from __future__ import annotations

from kvbench.trace.identity import ConversationTracker, identify


class TestIdentify:
    def test_client_headers_win(self) -> None:
        tracker = ConversationTracker()
        ident = identify(
            {"X-Request-ID": "req-1", "X-Correlation-ID": "conv-A"}, "chatcmpl-x", tracker
        )
        assert ident.request_id == "req-1"
        assert ident.client_request_id is True
        assert ident.completion_id == "chatcmpl-x"
        assert ident.conversation_id == "conv-A"
        assert ident.turn_index == 0
        assert ident.previous_request_id is None
        assert ident.span_attrs() == {
            "completion_id": "chatcmpl-x",
            "client_request_id": True,
            "conversation_id": "conv-A",
            "turn_index": 0,
            "previous_request_id": None,
        }

    def test_without_headers_falls_back_to_completion_id(self) -> None:
        ident = identify(None, "chatcmpl-y", ConversationTracker())
        assert ident.request_id == "chatcmpl-y"
        assert ident.client_request_id is False
        assert ident.conversation_id is None
        assert ident.span_attrs() == {"completion_id": "chatcmpl-y", "client_request_id": False}

    def test_empty_header_values_are_ignored(self) -> None:
        ident = identify({"x-request-id": "", "x-correlation-id": ""}, "c", ConversationTracker())
        assert ident.request_id == "c" and ident.conversation_id is None

    def test_turns_chain_within_a_conversation(self) -> None:
        tracker = ConversationTracker()
        first = identify({"x-correlation-id": "conv"}, "c1", tracker)
        second = identify({"x-correlation-id": "conv"}, "c2", tracker)
        other = identify({"x-correlation-id": "other"}, "c3", tracker)
        assert (first.turn_index, first.previous_request_id) == (0, None)
        assert (second.turn_index, second.previous_request_id) == (1, "c1")
        assert (other.turn_index, other.previous_request_id) == (0, None)

    def test_conversation_without_tracker_has_no_turn(self) -> None:
        ident = identify({"x-correlation-id": "conv"}, "c", None)
        assert ident.conversation_id == "conv" and ident.turn_index is None


class TestConversationTracker:
    def test_bounded_by_max_conversations(self) -> None:
        tracker = ConversationTracker(max_conversations=2)
        for i in range(3):
            tracker.next_turn(f"conv{i}", f"r{i}")
        assert len(tracker) == 2
        # the oldest conversation was dropped, so it restarts at turn 0
        assert tracker.next_turn("conv0", "r9") == (0, None)

    def test_idle_conversations_are_swept(self, monkeypatch) -> None:
        clock = [1000.0]
        monkeypatch.setattr("kvbench.trace.identity.time.monotonic", lambda: clock[0])
        tracker = ConversationTracker(idle_seconds=10.0)
        tracker.next_turn("old", "r1")
        clock[0] += 100.0
        assert tracker.next_turn("new", "r2") == (0, None)
        assert len(tracker) == 1  # "old" expired
        assert tracker.next_turn("old", "r3") == (0, None)
