"""Discrete-event simulation core for the dynamic JSSP extension.

A real future-event list (FEL), not the static environment's instantaneous
list-scheduling transition, is what lets :class:`~envs.dynamic_jssp_env.DynamicJSSPEnv`
interleave operation completions with asynchronous machine breakdowns, repairs, and job
arrivals on one shared simulated clock.

Event ordering
--------------
Events are ordered first by ``time``, then --- for events sharing the same instant, which
does happen (e.g. a repair completing at the exact instant a new job arrives) --- by a
fixed ``priority`` so processing order is deterministic:

``REPAIR_COMPLETE`` (1) -> ``OP_COMPLETE`` (2) -> ``BREAKDOWN`` (3) -> ``ARRIVAL`` (4)

Resource-freeing events resolve before resource-consuming ones, and a machine mid-repair
is never mistaken for "newly broken" in the same instant. A monotonic ``seq`` counter is
the final tiebreaker, both to keep the order total (heapq needs every pair comparable) and
to keep unrelated ``payload`` dicts out of comparisons entirely.

``EventKind`` deliberately has no ``DECISION_EPOCH`` member: whether the agent should be
asked for an action is *derived*, not queued --- see
:meth:`~envs.dynamic_jssp_env.DynamicJSSPEnv._has_decision_epoch`.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Iterable


class EventKind(IntEnum):
    """Kind of a queued event. Integer value doubles as the tie-break priority."""

    REPAIR_COMPLETE = 1
    OP_COMPLETE = 2
    BREAKDOWN = 3
    ARRIVAL = 4


class OpState(IntEnum):
    """Per-operation lifecycle state (extends the static env's boolean ``scheduled``).

    ``INACTIVE`` covers pre-allocated slots for jobs that have not arrived yet ---
    distinct from ``UNSCHEDULED`` so a not-yet-arrived job's operations never appear as a
    legal frontier action.
    """

    INACTIVE = 0
    UNSCHEDULED = 1
    IN_PROCESS = 2
    PREEMPTED = 3
    DONE = 4


class MachineState(IntEnum):
    """Per-machine state."""

    FREE = 0
    BUSY = 1
    BROKEN = 2


@dataclass(order=True)
class Event:
    """One entry in the future event list.

    Attributes:
        time: Simulated time the event fires.
        priority: Tie-break order for events sharing ``time`` --- see module docstring.
            Ordinarily just ``int(kind)``.
        seq: Monotonically increasing insertion sequence; the final tiebreaker so
            ``heapq`` never needs to compare two ``payload`` dicts directly.
        kind: The event type. Excluded from comparison.
        payload: Event-specific data (e.g. ``{"op_id": ..., "machine": ...}``). Excluded
            from comparison.
    """

    time: float
    priority: int = field(compare=True)
    seq: int = field(compare=True)
    kind: EventKind = field(compare=False)
    payload: dict = field(compare=False, default_factory=dict)


class FutureEventList:
    """Thin :mod:`heapq` wrapper: a priority queue of :class:`Event`.

    ``cancel`` is implemented as an O(n) filter + re-heapify rather than a true
    decrease-key structure --- FEL sizes in this codebase are small (at most a few
    hundred events per episode), so the simpler implementation is the right trade-off.
    """

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._counter: int = 0

    def push(self, time: float, kind: EventKind, payload: dict | None = None) -> Event:
        """Create and enqueue an event, returning it (e.g. for later ``cancel`` matching)."""
        event = Event(
            time=float(time),
            priority=int(kind),
            seq=self._counter,
            kind=kind,
            payload=payload if payload is not None else {},
        )
        self._counter += 1
        heapq.heappush(self._heap, event)
        return event

    def pop_min(self) -> Event:
        """Remove and return the earliest (time, priority, seq)-ordered event.

        Raises:
            IndexError: If the FEL is empty.
        """
        return heapq.heappop(self._heap)

    def peek_min_time(self) -> float | None:
        """Time of the next event to fire, or ``None`` if the FEL is empty."""
        return self._heap[0].time if self._heap else None

    @property
    def is_empty(self) -> bool:
        """Whether the FEL currently holds no events."""
        return not self._heap

    def cancel(self, predicate: Callable[[Event], bool]) -> list[Event]:
        """Remove every queued event matching ``predicate``.

        Used, e.g., to invalidate an already-scheduled ``OP_COMPLETE`` when a
        breakdown preempts the operation it belongs to.

        Args:
            predicate: Returns ``True`` for events that should be removed.

        Returns:
            The removed events, in no particular order.
        """
        removed = [e for e in self._heap if predicate(e)]
        if removed:
            self._heap[:] = [e for e in self._heap if not predicate(e)]
            heapq.heapify(self._heap)
        return removed

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterable[Event]:
        return iter(self._heap)
