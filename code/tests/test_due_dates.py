"""Unit tests for :mod:`baselines.due_dates` (Total Work Content due-date synthesis)."""

from __future__ import annotations

import numpy as np
import pytest

from baselines.due_dates import DueDateConfig, ensure_due_dates, synthesize_due_dates
from envs.instance import JSSPInstance
from envs.jssp_env import JSSPEnv


class TestSynthesizeDueDates:
    def test_twk_formula(self) -> None:
        job_totals = np.array([10.0, 20.0, 5.0])
        release_times = np.array([0.0, 0.0, 0.0])
        due_dates = synthesize_due_dates(job_totals, release_times, tightness=2.0)
        np.testing.assert_allclose(due_dates, [20.0, 40.0, 10.0])

    def test_due_dates_use_actual_release_time_for_arrivals(self) -> None:
        job_totals = np.array([10.0, 20.0])
        release_times = np.array([0.0, 7.0])  # job 1 "arrives" at t=7
        due_dates = synthesize_due_dates(job_totals, release_times, tightness=1.5)
        # job 1's due date is arrival_time + tightness * job_total, not tightness *
        # job_total alone.
        assert due_dates[1] == pytest.approx(7.0 + 1.5 * 20.0)
        assert due_dates[1] != pytest.approx(1.5 * 20.0)
        assert due_dates[0] == pytest.approx(1.5 * 10.0)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            synthesize_due_dates(np.array([1.0, 2.0]), np.array([0.0]))


class TestEnsureDueDates:
    def test_lazily_computes_and_caches(self) -> None:
        env = JSSPEnv("ft06")
        env.reset()
        assert getattr(env, "due_dates", None) is None
        due_dates = ensure_due_dates(env)
        np.testing.assert_allclose(due_dates, 1.5 * env.instance.job_totals)
        assert env.due_dates is due_dates

    def test_idempotent_does_not_recompute(self) -> None:
        env = JSSPEnv("ft06")
        env.reset()
        first = ensure_due_dates(env)
        env.due_dates[0] = -999.0  # mutate the cached array directly
        second = ensure_due_dates(env)
        assert second is first
        assert second[0] == -999.0  # not recomputed/overwritten

    def test_custom_tightness_used_when_not_yet_cached(self) -> None:
        env = JSSPEnv("ft06")
        env.reset()
        due_dates = ensure_due_dates(env, DueDateConfig(tightness=3.0))
        np.testing.assert_allclose(due_dates, 3.0 * env.instance.job_totals)
