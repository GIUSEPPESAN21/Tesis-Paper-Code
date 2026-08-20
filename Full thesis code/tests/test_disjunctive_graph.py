"""Unit tests for :mod:`envs.disjunctive_graph`.

Uses one hand-computable toy instance (2 jobs x 2 ops, 2 machines) throughout so every
expected value in this file can be verified with pen and paper:

    job 0: (machine 0, dur 3) -> (machine 1, dur 2)
    job 1: (machine 1, dur 4) -> (machine 0, dur 1)

Global op ids: op0 = job0/pos0, op1 = job0/pos1, op2 = job1/pos0, op3 = job1/pos1.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.disjunctive_graph import (
    build_machine_adjacency,
    build_precedence_adjacency,
    estimated_makespan,
    is_schedule_feasible,
    static_lower_bound,
)
from envs.instance import JSSPInstance


@pytest.fixture
def toy() -> JSSPInstance:
    return JSSPInstance(
        name="toy",
        machines=np.array([[0, 1], [1, 0]]),
        durations=np.array([[3, 2], [4, 1]]),
        num_machines=2,
    )


class TestPrecedenceAdjacency:
    def test_only_within_job_successor_edges_are_set(self, toy: JSSPInstance) -> None:
        adj = build_precedence_adjacency(toy)
        expected = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(adj, expected)


class TestMachineAdjacency:
    def test_full_graph_when_all_unscheduled(self, toy: JSSPInstance) -> None:
        adj = build_machine_adjacency(toy, unscheduled_mask=None)
        expected = np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(adj, expected)

    def test_symmetric_and_zero_diagonal(self, toy: JSSPInstance) -> None:
        adj = build_machine_adjacency(toy, unscheduled_mask=None)
        np.testing.assert_array_equal(adj, adj.T)
        assert np.all(np.diag(adj) == 0)

    def test_edges_drop_as_operations_are_scheduled(self, toy: JSSPInstance) -> None:
        # Once op0 (machine 0) is scheduled, its disjunction with op3 is resolved.
        unscheduled = np.array([False, True, True, True])
        adj = build_machine_adjacency(toy, unscheduled_mask=unscheduled)
        assert adj[0].sum() == 0
        assert adj[3, 0] == 0
        # The machine-1 disjunction between op1 and op2 is still open.
        assert adj[1, 2] == 1
        assert adj[2, 1] == 1


class TestStaticLowerBound:
    def test_toy_instance_lower_bound(self, toy: JSSPInstance) -> None:
        # job bound = max(5, 5) = 5; machine bound = max(4, 6) = 6 (see module docstring).
        assert static_lower_bound(toy) == 6

    def test_lower_bound_is_never_negative(self, toy: JSSPInstance) -> None:
        assert static_lower_bound(toy) >= 0


class TestEstimatedMakespan:
    def test_empty_schedule_uses_optimistic_chain_lengths(self, toy: JSSPInstance) -> None:
        completion = np.zeros(4)
        scheduled = np.zeros(4, dtype=bool)
        # Optimistic (machine-capacity-free) completion of each job chain: 5 and 5.
        assert estimated_makespan(toy, completion, scheduled) == 5.0

    def test_matches_true_makespan_at_full_schedule(self, toy: JSSPInstance) -> None:
        # The optimal toy schedule derived by hand: op0@[0,3), op2@[0,4),
        # op1@[4,6), op3@[4,5) -> makespan 6, matching the lower bound.
        completion = np.array([3.0, 6.0, 4.0, 5.0])
        scheduled = np.ones(4, dtype=bool)
        assert estimated_makespan(toy, completion, scheduled) == 6.0

    def test_nondecreasing_as_more_operations_are_scheduled(self, toy: JSSPInstance) -> None:
        completion = np.array([3.0, 0.0, 0.0, 0.0])
        h_after_op0 = estimated_makespan(
            toy, completion, np.array([True, False, False, False])
        )
        h_initial = estimated_makespan(toy, np.zeros(4), np.zeros(4, dtype=bool))
        assert h_after_op0 >= h_initial


class TestScheduleFeasibility:
    def test_hand_built_optimal_schedule_is_feasible(self, toy: JSSPInstance) -> None:
        starts = np.array([0.0, 4.0, 0.0, 4.0])
        feasible, violations = is_schedule_feasible(toy, starts)
        assert feasible
        assert violations == []

    def test_precedence_violation_is_detected(self, toy: JSSPInstance) -> None:
        # op1 (job0/pos1) starts at 2, before op0 (job0/pos0, dur 3) ends at 3.
        starts = np.array([0.0, 2.0, 0.0, 4.0])
        feasible, violations = is_schedule_feasible(toy, starts)
        assert not feasible
        assert any("precedence" in v for v in violations)

    def test_machine_overlap_is_detected(self, toy: JSSPInstance) -> None:
        # op0 and op3 both need machine 0; scheduling them to overlap is infeasible.
        starts = np.array([0.0, 4.0, 0.0, 0.0])
        feasible, violations = is_schedule_feasible(toy, starts)
        assert not feasible
        assert any("overlap" in v for v in violations)

    def test_negative_start_is_detected(self, toy: JSSPInstance) -> None:
        starts = np.array([-1.0, 4.0, 0.0, 4.0])
        feasible, violations = is_schedule_feasible(toy, starts)
        assert not feasible
        assert any("negative" in v for v in violations)

    def test_wrong_length_is_detected(self, toy: JSSPInstance) -> None:
        feasible, violations = is_schedule_feasible(toy, np.array([0.0, 1.0]))
        assert not feasible
        assert len(violations) == 1
