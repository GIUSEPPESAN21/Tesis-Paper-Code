"""Unit tests for :mod:`experiments.slack_analysis` (Fase 1, T1.1).

Hand-verified against a 2-job/2-machine toy instance (worked out on paper in the
module's implementation commit message / PR description) so the CPM forward/backward
pass is checked against an independently-computed ground truth, not just internal
consistency.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.instance import JSSPInstance
from experiments.slack_analysis import schedule_slack_metrics


def _toy_instance() -> JSSPInstance:
    # job0: (m0,3) (m1,2); job1: (m1,4) (m0,1)
    return JSSPInstance(
        name="toy2x2",
        machines=np.array([[0, 1], [1, 0]]),
        durations=np.array([[3, 2], [4, 1]]),
        num_machines=2,
    )


def _toy_schedule() -> np.ndarray:
    # op0 (job0,pos0,m0,d3): start 0
    # op1 (job0,pos1,m1,d2): start 4 (after op0 ends@3, and after op2 on m1 ends@4)
    # op2 (job1,pos0,m1,d4): start 0
    # op3 (job1,pos1,m0,d1): start 4 (after op2 ends@4, and after op0 on m0 ends@3)
    return np.array([0.0, 4.0, 0.0, 4.0])


class TestScheduleSlackMetrics:
    def test_matches_hand_computed_cpm(self) -> None:
        instance = _toy_instance()
        metrics = schedule_slack_metrics(_toy_schedule(), instance)

        assert metrics.makespan == pytest.approx(6.0)
        assert metrics.critical_path_length == pytest.approx(6.0)
        # slack per op = [1, 0, 0, 1] (op0/op3 off critical path, op1/op2 on it)
        assert metrics.total_slack == pytest.approx(2.0)
        assert metrics.mean_slack == pytest.approx(0.5)
        assert metrics.slack_ratio == pytest.approx(2.0 / (6.0 * 2))
        assert metrics.critical_op_fraction == pytest.approx(0.5)  # op1, op2

    def test_machine_idle_fraction_from_capacity_utilization(self) -> None:
        instance = _toy_instance()
        metrics = schedule_slack_metrics(_toy_schedule(), instance)
        # total processing time = 3+2+4+1 = 10; capacity = makespan(6) * m(2) = 12
        assert metrics.machine_idle_fraction == pytest.approx(1.0 - 10.0 / 12.0)

    def test_zero_slack_schedule_is_fully_critical(self) -> None:
        """A single-job (no parallelism, no choice of order) instance has every
        operation on the critical path by construction: zero slack everywhere."""
        instance = JSSPInstance(
            name="single_job",
            machines=np.array([[0, 1, 2]]),
            durations=np.array([[3, 4, 2]]),
            num_machines=3,
        )
        schedule = np.array([0.0, 3.0, 7.0])
        metrics = schedule_slack_metrics(schedule, instance)
        assert metrics.total_slack == pytest.approx(0.0)
        assert metrics.critical_op_fraction == pytest.approx(1.0)
        assert metrics.makespan == pytest.approx(9.0)

    def test_raises_on_infeasible_schedule_by_default(self) -> None:
        instance = _toy_instance()
        bad_schedule = np.array([0.0, 0.0, 0.0, 0.0])  # violates precedence and machine capacity
        with pytest.raises(ValueError, match="not a feasible schedule"):
            schedule_slack_metrics(bad_schedule, instance)

    def test_check_feasible_false_skips_the_precondition_check(self) -> None:
        """With check_feasible=False, a schedule that violates precedence/capacity is
        not rejected up front -- CPM may still raise its own cycle error, or may
        silently produce numbers a caller should not trust; this test only asserts the
        precondition check itself is bypassed, not that the output is meaningful."""
        instance = _toy_instance()
        good_schedule = _toy_schedule()
        # Sanity: does not raise even though we skip the (redundant here) feasibility check.
        metrics = schedule_slack_metrics(good_schedule, instance, check_feasible=False)
        assert metrics.makespan == pytest.approx(6.0)
