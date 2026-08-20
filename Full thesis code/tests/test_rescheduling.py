"""Tests for envs/rescheduling.py (Fase 6, T6.1).

Per the tutor's explicit requirement: (a) sin disrupcion reproduce el makespan fuera
de linea, (b) nunca viola precedencias, (c) nunca reasigna una operacion ya en curso.
CP-SAT-backed rollouts are marked ``slow`` (OR-Tools dependency, real solve time);
the priority-search solver's tests stay fast and OR-Tools-free.
"""

from __future__ import annotations

import numpy as np
import pytest

from baselines.cpsat import solve_cpsat
from envs.disruptions import DisruptionConfig
from envs.dynamic_jssp_env import DynamicJSSPEnv
from envs.events import OpState
from envs.loader import load_by_name
from envs.rescheduling import (
    EventTrigger,
    event_triggered_reopt,
    periodic_reopt,
    snapshot_residual_subproblem,
    solve_residual_priority_search,
)


class TestSnapshotResidualSubproblem:
    def test_fresh_reset_has_no_fixed_intervals(self) -> None:
        env = DynamicJSSPEnv("ft06", DisruptionConfig())
        env.reset(seed=0)
        sub = snapshot_residual_subproblem(env)
        assert sub.fixed_intervals == {}
        assert len(sub.unscheduled_ops) == env.active_ops

    def test_in_process_op_becomes_fixed_never_unscheduled(self) -> None:
        env = DynamicJSSPEnv("ft06", DisruptionConfig())
        env.reset(seed=0)
        legal = np.flatnonzero(env.action_masks())
        first_action = int(legal[0])
        env.step(first_action)
        sub = snapshot_residual_subproblem(env)
        # Whatever just got dispatched is either DONE (zero-duration edge case) or
        # IN_PROCESS -- either way it must be FIXED, never in unscheduled_ops again.
        assert first_action not in sub.unscheduled_ops
        assert first_action in sub.fixed_intervals or env.op_state[first_action] == OpState.DONE


class TestSolveResidualPrioritySearch:
    def test_returns_start_for_every_unscheduled_op(self) -> None:
        env = DynamicJSSPEnv("ft06", DisruptionConfig())
        env.reset(seed=0)
        sub = snapshot_residual_subproblem(env)
        starts = solve_residual_priority_search(env, sub, budget_s=0.2, rng=np.random.default_rng(0))
        assert set(starts.keys()) == set(sub.unscheduled_ops)

    def test_never_returns_a_fixed_op(self) -> None:
        env = DynamicJSSPEnv("ft06", DisruptionConfig())
        env.reset(seed=0)
        legal = np.flatnonzero(env.action_masks())
        env.step(int(legal[0]))
        sub = snapshot_residual_subproblem(env)
        starts = solve_residual_priority_search(env, sub, budget_s=0.2, rng=np.random.default_rng(0))
        assert set(starts.keys()).isdisjoint(sub.fixed_intervals.keys())

    def test_respects_job_ready_time_lower_bound(self) -> None:
        env = DynamicJSSPEnv("ft06", DisruptionConfig())
        env.reset(seed=0)
        legal = np.flatnonzero(env.action_masks())
        env.step(int(legal[0]))
        sub = snapshot_residual_subproblem(env)
        starts = solve_residual_priority_search(env, sub, budget_s=0.2, rng=np.random.default_rng(0))
        for op, start in starts.items():
            job = int(env._job_index[op])
            ready = float(sub.job_ready_time[job]) if job < len(sub.job_ready_time) else 0.0
            assert start >= ready - 1e-6

    def test_empty_when_no_unscheduled_ops(self) -> None:
        env = DynamicJSSPEnv("ft06", DisruptionConfig())
        env.reset(seed=0)
        sub = snapshot_residual_subproblem(env)
        empty_sub = sub.__class__(
            unscheduled_ops=(), fixed_intervals=sub.fixed_intervals,
            machine_free_time=sub.machine_free_time, job_ready_time=sub.job_ready_time,
            current_time=sub.current_time,
        )
        assert solve_residual_priority_search(env, empty_sub, budget_s=0.1) == {}


class TestPeriodicReoptNoDisruption:
    def test_never_reassigns_in_process_ops_ga_backend(self) -> None:
        """Fast (priority-search backend, no OR-Tools): rolls a full episode and
        checks every fixed op's realized start/end never changed across the episode
        -- the direct test of "no reasigna operaciones ya en curso"."""
        instance = load_by_name("ft06")
        result = periodic_reopt(
            instance, DisruptionConfig(), seed=0, period=5.0, solver="ga", budget_s=0.1
        )
        assert result.feasible, result.violations
        assert result.n_reoptimizations >= 1

    def test_feasible_under_breakdowns_ga_backend(self) -> None:
        instance = load_by_name("ft06")
        config = DisruptionConfig(target_availability=0.85, seed=1)
        result = periodic_reopt(instance, config, seed=1, period=8.0, solver="ga", budget_s=0.1)
        assert result.feasible, result.violations


class TestEventTriggeredReoptGaBackend:
    def test_feasible_under_arrivals(self) -> None:
        instance = load_by_name("ft06")
        config = DisruptionConfig(target_utilization=0.85, max_dynamic_jobs=5, seed=2)
        trigger = EventTrigger(on_breakdown=True, on_arrival=True, deviation_threshold=None)
        result = event_triggered_reopt(instance, config, seed=2, trigger=trigger, solver="ga", budget_s=0.1)
        assert result.feasible, result.violations
        assert result.n_reoptimizations >= 1

    def test_deviation_trigger_fires_more_often_than_no_trigger(self) -> None:
        """A tighter deviation threshold should never result in FEWER reoptimizations
        than a disabled deviation trigger (all else equal), even if breakdown/arrival
        triggers alone already force some reopts -- a monotonicity sanity check."""
        instance = load_by_name("ft06")
        config = DisruptionConfig(target_availability=0.85, seed=3)
        loose = EventTrigger(on_breakdown=False, on_arrival=False, deviation_threshold=None)
        tight = EventTrigger(on_breakdown=False, on_arrival=False, deviation_threshold=0.01)
        result_loose = event_triggered_reopt(instance, config, seed=3, trigger=loose, solver="ga", budget_s=0.1)
        result_tight = event_triggered_reopt(instance, config, seed=3, trigger=tight, solver="ga", budget_s=0.1)
        assert result_tight.n_reoptimizations >= result_loose.n_reoptimizations


@pytest.mark.slow
class TestPeriodicReoptCpsatBackendReproducesOfflineMakespan:
    def test_no_disruption_matches_offline_cpsat_makespan(self) -> None:
        """T6.1's explicit requirement: sin disrupcion, la politica de reoptimizacion
        debe reproducir el makespan fuera de linea (el mismo problema, resuelto una
        vez sin nunca reoptimizar despues de la primera solucion, ya que sin
        disrupcion no hay ningun evento que dispare una reoptimizacion distinta)."""
        instance = load_by_name("ft06")
        offline = solve_cpsat(instance, time_limit_s=30.0)

        result = periodic_reopt(
            instance, DisruptionConfig(), seed=0, period=1_000_000.0, solver="cpsat", budget_s=30.0
        )
        assert result.feasible, result.violations
        assert result.makespan == pytest.approx(offline.makespan, abs=1e-6)
