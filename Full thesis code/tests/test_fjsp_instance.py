"""Unit tests for :mod:`envs.fjsp_instance` (Week 15 refactor: FJSP support promoted
from a solver-local fixture in ``baselines/cpsat.py`` to a validated, canonical
instance type next to :class:`~envs.instance.JSSPInstance`).
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.fjsp_instance import FJSPInstance, generate_fjsp_instance
from envs.instance import JSSPInstance
from envs.loader import load_by_name


def _toy_fjsp(**overrides) -> FJSPInstance:
    kwargs = dict(
        name="toy",
        num_jobs=2,
        ops_per_job=2,
        num_machines=2,
        eligible_machines=[[[0, 1], [1]], [[0], [0, 1]]],
        durations=[[[3, 5], [2]], [[4], [1, 2]]],
    )
    kwargs.update(overrides)
    return FJSPInstance(**kwargs)


class TestConstructionAndIndexing:
    def test_valid_instance_constructs(self) -> None:
        instance = _toy_fjsp()
        assert instance.num_operations == 4

    def test_op_id_round_trips_through_job_and_position(self) -> None:
        instance = _toy_fjsp()
        for job in range(instance.num_jobs):
            for pos in range(instance.ops_per_job):
                op = instance.op_id(job, pos)
                assert instance.job_of(op) == job
                assert instance.position_of(op) == pos

    def test_str_reports_average_eligible_machines(self) -> None:
        instance = _toy_fjsp()
        # (2 + 1 + 1 + 2) / 4 operations == 1.5 average eligible machines/op.
        assert "1.50" in str(instance)


class TestValidation:
    def test_wrong_number_of_job_rows_raises(self) -> None:
        with pytest.raises(ValueError, match="job rows"):
            _toy_fjsp(eligible_machines=[[[0, 1], [1]]])

    def test_wrong_ops_per_job_raises(self) -> None:
        with pytest.raises(ValueError, match="operations"):
            _toy_fjsp(eligible_machines=[[[0, 1], [1]], [[0]]])

    def test_eligible_machines_durations_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="durations"):
            _toy_fjsp(durations=[[[3, 5], [2]], [[4], [1]]])

    def test_empty_eligible_set_raises(self) -> None:
        with pytest.raises(ValueError, match="no eligible machine"):
            _toy_fjsp(eligible_machines=[[[0, 1], []], [[0], [0, 1]]], durations=[[[3, 5], []], [[4], [1, 2]]])

    def test_duplicate_machine_in_eligible_set_raises(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            _toy_fjsp(eligible_machines=[[[0, 0], [1]], [[0], [0, 1]]])

    def test_out_of_range_machine_raises(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            _toy_fjsp(eligible_machines=[[[0, 5], [1]], [[0], [0, 1]]])

    def test_negative_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            _toy_fjsp(durations=[[[3, -5], [2]], [[4], [1, 2]]])

    def test_non_positive_dimensions_raise(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _toy_fjsp(num_machines=0)


class TestGenerateFjspInstance:
    def test_deterministic_given_seed(self) -> None:
        base = load_by_name("ft06")
        a = generate_fjsp_instance(base, seed=7)
        b = generate_fjsp_instance(base, seed=7)
        assert a.eligible_machines == b.eligible_machines
        assert a.durations == b.durations

    def test_original_routing_kept_as_first_alternative(self) -> None:
        """The base instance's own (machine, duration) must survive as alternative 0
        for every operation -- this is what makes a degenerate GA/CP-SAT run (forced to
        alternative 0 everywhere) reproduce the base JSSP instance's optimum exactly."""
        base = load_by_name("ft06")
        fjsp = generate_fjsp_instance(base, seed=0)
        for job in range(base.num_jobs):
            for pos in range(base.ops_per_job):
                assert fjsp.eligible_machines[job][pos][0] == int(base.machines[job, pos])
                assert fjsp.durations[job][pos][0] == int(base.durations[job, pos])

    def test_shape_matches_base_instance(self) -> None:
        base = load_by_name("la01")
        fjsp = generate_fjsp_instance(base, seed=1)
        assert fjsp.num_jobs == base.num_jobs
        assert fjsp.ops_per_job == base.ops_per_job
        assert fjsp.num_machines == base.num_machines
        assert fjsp.num_operations == base.num_operations

    def test_extra_alternatives_never_repeat_a_machine(self) -> None:
        base = load_by_name("ft06")
        fjsp = generate_fjsp_instance(base, max_extra_alternatives=3, seed=2)
        for job in range(fjsp.num_jobs):
            for pos in range(fjsp.ops_per_job):
                machines = fjsp.eligible_machines[job][pos]
                assert len(set(machines)) == len(machines)
                assert len(machines) <= 1 + 3

    def test_single_machine_instance_rejected(self) -> None:
        single_machine = JSSPInstance(
            name="single",
            machines=np.array([[0], [0]]),
            durations=np.array([[3], [4]]),
            num_machines=1,
        )
        with pytest.raises(ValueError, match="at least 2 machines"):
            generate_fjsp_instance(single_machine, seed=0)

    def test_zero_extra_alternatives_degenerates_to_fixed_routing(self) -> None:
        base = load_by_name("ft06")
        fjsp = generate_fjsp_instance(base, max_extra_alternatives=0, seed=0)
        for job in range(fjsp.num_jobs):
            for pos in range(fjsp.ops_per_job):
                assert fjsp.eligible_machines[job][pos] == [int(base.machines[job, pos])]
