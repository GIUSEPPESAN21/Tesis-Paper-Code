"""Unit tests for :mod:`envs.events`: FEL ordering, tie-breaking, and cancellation.

These are pure data-structure tests --- no environment, no randomness --- so every
assertion is exact.
"""

from __future__ import annotations

from envs.events import Event, EventKind, FutureEventList


class TestFELOrdering:
    def test_fel_orders_by_time(self) -> None:
        fel = FutureEventList()
        for t in (5.0, 1.0, 3.0, 2.0, 4.0):
            fel.push(t, EventKind.OP_COMPLETE, {"t": t})
        popped = [fel.pop_min().time for _ in range(5)]
        assert popped == sorted(popped)
        assert popped == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_peek_min_time_matches_next_pop(self) -> None:
        fel = FutureEventList()
        fel.push(10.0, EventKind.ARRIVAL)
        fel.push(2.0, EventKind.BREAKDOWN)
        assert fel.peek_min_time() == 2.0
        event = fel.pop_min()
        assert event.time == 2.0
        assert fel.peek_min_time() == 10.0

    def test_peek_min_time_none_when_empty(self) -> None:
        fel = FutureEventList()
        assert fel.peek_min_time() is None
        assert fel.is_empty

    def test_pop_min_raises_on_empty(self) -> None:
        fel = FutureEventList()
        import pytest

        with pytest.raises(IndexError):
            fel.pop_min()

    def test_len_tracks_pending_events(self) -> None:
        fel = FutureEventList()
        assert len(fel) == 0
        fel.push(1.0, EventKind.ARRIVAL)
        fel.push(2.0, EventKind.ARRIVAL)
        assert len(fel) == 2
        fel.pop_min()
        assert len(fel) == 1


class TestFELTieBreakOrder:
    def test_fel_tie_break_order(self) -> None:
        """Four events at identical time pop in the fixed §2.1 priority order:
        REPAIR_COMPLETE -> OP_COMPLETE -> BREAKDOWN -> ARRIVAL, regardless of push order.
        """
        fel = FutureEventList()
        fel.push(5.0, EventKind.ARRIVAL, {"tag": "arrival"})
        fel.push(5.0, EventKind.BREAKDOWN, {"tag": "breakdown"})
        fel.push(5.0, EventKind.OP_COMPLETE, {"tag": "op_complete"})
        fel.push(5.0, EventKind.REPAIR_COMPLETE, {"tag": "repair_complete"})

        order = [fel.pop_min().kind for _ in range(4)]
        assert order == [
            EventKind.REPAIR_COMPLETE,
            EventKind.OP_COMPLETE,
            EventKind.BREAKDOWN,
            EventKind.ARRIVAL,
        ]

    def test_tie_break_is_independent_of_push_order(self) -> None:
        fel = FutureEventList()
        # Push in exactly the desired output order this time; result must be unchanged.
        for kind in (
            EventKind.REPAIR_COMPLETE,
            EventKind.OP_COMPLETE,
            EventKind.BREAKDOWN,
            EventKind.ARRIVAL,
        ):
            fel.push(0.0, kind)
        order = [fel.pop_min().kind for _ in range(4)]
        assert order == [
            EventKind.REPAIR_COMPLETE,
            EventKind.OP_COMPLETE,
            EventKind.BREAKDOWN,
            EventKind.ARRIVAL,
        ]

    def test_same_time_same_kind_breaks_tie_by_insertion_order(self) -> None:
        fel = FutureEventList()
        fel.push(1.0, EventKind.ARRIVAL, {"job": "first"})
        fel.push(1.0, EventKind.ARRIVAL, {"job": "second"})
        first = fel.pop_min()
        second = fel.pop_min()
        assert first.payload["job"] == "first"
        assert second.payload["job"] == "second"


class TestFELCancel:
    def test_fel_cancel_removes_matching_event(self) -> None:
        fel = FutureEventList()
        fel.push(1.0, EventKind.OP_COMPLETE, {"op_id": 0})
        fel.push(2.0, EventKind.OP_COMPLETE, {"op_id": 1})
        fel.push(3.0, EventKind.ARRIVAL)

        removed = fel.cancel(lambda e: e.kind == EventKind.OP_COMPLETE and e.payload.get("op_id") == 0)

        assert len(removed) == 1
        assert removed[0].payload["op_id"] == 0
        assert len(fel) == 2
        remaining_kinds = sorted(e.kind for e in fel)
        assert remaining_kinds == sorted([EventKind.OP_COMPLETE, EventKind.ARRIVAL])

    def test_cancel_preserves_heap_validity_for_subsequent_pops(self) -> None:
        fel = FutureEventList()
        for i, t in enumerate([5.0, 1.0, 4.0, 2.0, 3.0]):
            fel.push(t, EventKind.OP_COMPLETE, {"op_id": i})
        fel.cancel(lambda e: e.payload.get("op_id") == 2)  # removes t=4.0

        popped_times = [fel.pop_min().time for _ in range(4)]
        assert popped_times == sorted(popped_times)
        assert popped_times == [1.0, 2.0, 3.0, 5.0]

    def test_cancel_no_match_removes_nothing(self) -> None:
        fel = FutureEventList()
        fel.push(1.0, EventKind.ARRIVAL)
        removed = fel.cancel(lambda e: e.kind == EventKind.BREAKDOWN)
        assert removed == []
        assert len(fel) == 1

    def test_cancel_multiple_matches(self) -> None:
        fel = FutureEventList()
        for t in (1.0, 2.0, 3.0, 4.0):
            fel.push(t, EventKind.BREAKDOWN, {"machine": 0})
        fel.push(5.0, EventKind.ARRIVAL)
        removed = fel.cancel(lambda e: e.kind == EventKind.BREAKDOWN)
        assert len(removed) == 4
        assert len(fel) == 1
        assert fel.pop_min().kind == EventKind.ARRIVAL


class TestEventDataclass:
    def test_event_orders_by_time_then_priority_then_seq(self) -> None:
        a = Event(time=1.0, priority=2, seq=0, kind=EventKind.OP_COMPLETE)
        b = Event(time=1.0, priority=1, seq=1, kind=EventKind.REPAIR_COMPLETE)
        c = Event(time=0.5, priority=5, seq=2, kind=EventKind.ARRIVAL)
        assert sorted([a, b, c]) == [c, b, a]

    def test_event_kind_and_payload_excluded_from_comparison(self) -> None:
        a = Event(time=1.0, priority=1, seq=0, kind=EventKind.ARRIVAL, payload={"x": 1})
        b = Event(time=1.0, priority=1, seq=0, kind=EventKind.BREAKDOWN, payload={"x": 2})
        assert a == b  # only (time, priority, seq) participate in equality/ordering
