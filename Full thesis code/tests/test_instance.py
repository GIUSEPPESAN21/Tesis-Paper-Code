"""Unit tests for :mod:`envs.instance`."""

from __future__ import annotations

import numpy as np
import pytest

from envs.instance import JSSPInstance, Operation


def _toy_instance() -> JSSPInstance:
    """2 jobs x 2 ops, 2 machines, hand-computable derived quantities.

    job 0: (machine 0, 3) -> (machine 1, 2)
    job 1: (machine 1, 4) -> (machine 0, 1)
    """
    return JSSPInstance(
        name="toy",
        machines=np.array([[0, 1], [1, 0]]),
        durations=np.array([[3, 2], [4, 1]]),
        num_machines=2,
    )


class TestShapeAndIndexing:
    def test_shape_properties(self) -> None:
        inst = _toy_instance()
        assert inst.num_jobs == 2
        assert inst.ops_per_job == 2
        assert inst.num_operations == 4

    def test_op_id_roundtrip(self) -> None:
        inst = _toy_instance()
        for job in range(inst.num_jobs):
            for pos in range(inst.ops_per_job):
                op_id = inst.op_id(job, pos)
                assert inst.job_of(op_id) == job
                assert inst.position_of(op_id) == pos

    def test_operations_iterator(self) -> None:
        inst = _toy_instance()
        ops = list(inst.operations())
        assert len(ops) == inst.num_operations
        assert ops[0] == Operation(op_id=0, job=0, position=0, machine=0, duration=3)
        assert ops[3] == Operation(op_id=3, job=1, position=1, machine=0, duration=1)

    def test_flat_views_match_op_id_ordering(self) -> None:
        inst = _toy_instance()
        assert inst.flat_machines.tolist() == [0, 1, 1, 0]
        assert inst.flat_durations.tolist() == [3, 2, 4, 1]


class TestDerivedQuantities:
    def test_job_and_machine_totals(self) -> None:
        inst = _toy_instance()
        assert inst.job_totals.tolist() == [5, 5]
        assert inst.machine_totals.tolist() == [4, 6]
        assert inst.total_processing_time == 10
        assert inst.max_duration == 4

    def test_remaining_work(self) -> None:
        inst = _toy_instance()
        np.testing.assert_array_equal(inst.remaining_work(), [[5, 2], [5, 1]])

    def test_head_times(self) -> None:
        inst = _toy_instance()
        np.testing.assert_array_equal(inst.head_times(), [[0, 3], [0, 4]])

    def test_tail_times(self) -> None:
        inst = _toy_instance()
        np.testing.assert_array_equal(inst.tail_times(), [[2, 0], [1, 0]])


class TestValidation:
    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            JSSPInstance(
                name="bad",
                machines=np.array([[0, 1], [1, 0]]),
                durations=np.array([[3, 2, 1], [4, 1, 1]]),
                num_machines=2,
            )

    def test_empty_instance_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            JSSPInstance(
                name="bad",
                machines=np.zeros((0, 0), dtype=np.int64),
                durations=np.zeros((0, 0), dtype=np.int64),
                num_machines=0,
            )

    def test_machine_index_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            JSSPInstance(
                name="bad",
                machines=np.array([[0, 2], [1, 0]]),
                durations=np.array([[3, 2], [4, 1]]),
                num_machines=2,
            )

    def test_negative_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            JSSPInstance(
                name="bad",
                machines=np.array([[0, 1], [1, 0]]),
                durations=np.array([[3, -2], [4, 1]]),
                num_machines=2,
            )

    def test_repeated_machine_in_job_raises(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            JSSPInstance(
                name="bad",
                machines=np.array([[0, 0], [1, 0]]),
                durations=np.array([[3, 2], [4, 1]]),
                num_machines=2,
            )

    def test_frozen_instance_is_immutable(self) -> None:
        inst = _toy_instance()
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError is a subtype
            inst.name = "renamed"  # type: ignore[misc]
