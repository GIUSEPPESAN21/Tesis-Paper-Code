"""Unit tests for :mod:`baselines.cpsat`.

``solve_cpsat`` (the JSSP path) previously had no dedicated test file; its correctness
was only exercised indirectly via ``tests/test_env.py::TestOptimalSequence``. This file
adds the explicit STEP4 requirement --- CP-SAT recovers the known optimum on
``ft06``/``ft10``/``ft20`` within its time limit --- plus tests for the new
``solve_cpsat_fjsp`` (``OptionalIntervalVar`` + exactly-one-machine) path.
"""

from __future__ import annotations

import math

import pytest

from baselines.cpsat import compute_mip_gap, solve_cpsat, solve_cpsat_fjsp
from envs.disjunctive_graph import is_schedule_feasible
from envs.fjsp_instance import FJSPInstance
from envs.instance import JSSPInstance
from envs.loader import known_optimum, load_by_name


def _jssp_as_fjsp(instance: JSSPInstance) -> FJSPInstance:
    """Wrap a fixed-routing JSSPInstance as a degenerate (single-alternative) FJSPInstance.

    Test-only bridge: real FJSP instance loading is out of scope this sprint (see
    ``Baselines_TechSpec.md`` §0). This lets solve_cpsat_fjsp be cross-checked against
    the already-validated solve_cpsat on real benchmark data, not just a hand-built toy.
    """
    eligible_machines = [
        [[int(instance.machines[job, pos])] for pos in range(instance.ops_per_job)]
        for job in range(instance.num_jobs)
    ]
    durations = [
        [[int(instance.durations[job, pos])] for pos in range(instance.ops_per_job)]
        for job in range(instance.num_jobs)
    ]
    return FJSPInstance(
        name=instance.name,
        num_jobs=instance.num_jobs,
        ops_per_job=instance.ops_per_job,
        num_machines=instance.num_machines,
        eligible_machines=eligible_machines,
        durations=durations,
    )


class TestSolveCpsatRecoversKnownOptimum:
    @pytest.mark.parametrize("instance_name,time_limit_s", [("ft06", 30.0), ("ft10", 60.0), ("ft20", 60.0)])
    def test_recovers_known_optimum(self, instance_name: str, time_limit_s: float) -> None:
        instance = load_by_name(instance_name)
        result = solve_cpsat(instance, time_limit_s=time_limit_s)
        assert result.is_optimal, f"{instance_name}: status={result.status}, bound={result.best_bound}"
        assert result.status == "OPTIMAL"
        assert result.makespan == known_optimum(instance_name)
        feasible, violations = is_schedule_feasible(instance, result.start_times)
        assert feasible, violations


class TestSolveCpsatFjsp:
    def test_ragged_eligibility_toy_reaches_hand_verified_optimum(self) -> None:
        """2 jobs x 2 ops, ragged eligible-machine sets (routing flexibility genuinely
        matters): optimum makespan 7, hand-verified (see BASELINES_RESULTS_WEEK9.md).
        """
        instance = FJSPInstance(
            name="fjsp_toy",
            num_jobs=2,
            ops_per_job=2,
            num_machines=2,
            eligible_machines=[
                [[0, 1], [1]],
                [[0], [0, 1]],
            ],
            durations=[
                [[3, 5], [2]],
                [[4], [1, 2]],
            ],
        )
        result = solve_cpsat_fjsp(instance, time_limit_s=10.0)
        assert result.is_optimal
        assert result.status == "OPTIMAL"
        assert result.makespan == 7

    def test_single_alternative_per_op_matches_jssp_path(self) -> None:
        """A degenerate FJSP instance (exactly one eligible machine per operation, i.e.
        classic fixed routing) must reproduce solve_cpsat's optimum exactly --- the FJSP
        model collapses to the JSSP model when every AddExactlyOne group has size 1.
        """
        instance = load_by_name("ft06")
        fjsp_instance = _jssp_as_fjsp(instance)

        jssp_result = solve_cpsat(instance, time_limit_s=30.0)
        fjsp_result = solve_cpsat_fjsp(fjsp_instance, time_limit_s=30.0)

        assert fjsp_result.is_optimal
        assert fjsp_result.makespan == jssp_result.makespan == 55

    def test_no_eligible_machine_raises(self) -> None:
        """``FJSPInstance.__post_init__`` (Week 15) now rejects this at construction
        time --- before this refactor, ``solve_cpsat_fjsp`` was the only thing that
        checked, so an otherwise-invalid instance could be constructed and passed
        around unchecked until someone happened to solve it."""
        with pytest.raises(ValueError, match="no eligible machine"):
            FJSPInstance(
                name="broken",
                num_jobs=1,
                ops_per_job=1,
                num_machines=1,
                eligible_machines=[[[]]],
                durations=[[[]]],
            )


class TestComputeMipGap:
    """Recomendacion 1/10 del tutor (T7.1): mip_gap must be exactly 0.0 when
    is_optimal, NaN when there is no incumbent, and the relative gap otherwise --
    a pure function, no OR-Tools solve needed to exercise it."""

    def test_zero_when_optimal(self) -> None:
        assert compute_mip_gap(makespan=55, best_bound=55, is_optimal=True) == 0.0

    def test_nan_when_no_solution(self) -> None:
        assert math.isnan(compute_mip_gap(makespan=None, best_bound=40, is_optimal=False))

    def test_positive_gap_on_feasible_not_optimal(self) -> None:
        gap = compute_mip_gap(makespan=120, best_bound=100, is_optimal=False)
        assert gap == pytest.approx(0.20)

    def test_denominator_floored_at_one_for_degenerate_zero_bound(self) -> None:
        gap = compute_mip_gap(makespan=3, best_bound=0, is_optimal=False)
        assert gap == pytest.approx(3.0)
