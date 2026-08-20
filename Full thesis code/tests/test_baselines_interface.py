"""Unit tests for :mod:`baselines.interface` (the common ``Baseline`` protocol).

``GABaseline`` (Week 12) is covered here alongside ``PDRBaseline``/``CPSATBaseline``;
GA-specific operator/loop tests live in ``tests/test_ga.py``. ``CPSATFJSPBaseline`` and
``GABaseline``'s FJSP path (Week 15) are covered in :class:`TestFjspBaselines`.
"""

from __future__ import annotations

import pytest

from baselines.ga import GAConfig
from baselines.interface import (
    BaselineResult,
    CPSATBaseline,
    CPSATFJSPBaseline,
    GABaseline,
    PDRBaseline,
)
from envs.disjunctive_graph import is_schedule_feasible
from envs.fjsp_instance import generate_fjsp_instance
from envs.loader import load_by_name


class TestAllBaselinesSatisfyProtocol:
    @pytest.mark.parametrize(
        "baseline", [PDRBaseline("mwkr"), CPSATBaseline(), GABaseline(GAConfig(population_size=40))]
    )
    def test_solve_returns_feasible_result(self, baseline) -> None:
        instance = load_by_name("ft06")
        result = baseline.solve(instance, seed=0, time_limit_s=10.0)
        assert isinstance(result, BaselineResult)
        assert result.instance == "ft06"
        assert result.method == baseline.name
        feasible, violations = is_schedule_feasible(instance, result.start_times)
        assert feasible, violations
        assert result.feasible
        assert result.wall_clock_time >= 0.0


class TestFjspBaselines:
    """FJSPInstance path: CPSATFJSPBaseline and GABaseline's FJSP branch."""

    @pytest.mark.parametrize(
        "baseline", [CPSATFJSPBaseline(), GABaseline(GAConfig(population_size=30))]
    )
    def test_solve_returns_feasible_result_on_fjsp_instance(self, baseline) -> None:
        fjsp = generate_fjsp_instance(load_by_name("ft06"), seed=0)
        result = baseline.solve(fjsp, seed=0, time_limit_s=10.0)
        assert isinstance(result, BaselineResult)
        assert result.instance == fjsp.name
        assert result.method == baseline.name
        assert result.feasible

    def test_cpsat_fjsp_never_beats_jssp_optimum_of_degenerate_instance(self) -> None:
        """A degenerate FJSP (every op has exactly its original single alternative) must
        reproduce the plain JSSP optimum -- cross-checked against solve_cpsat directly in
        ``tests/test_cpsat.py``; here we only check the Baseline-adapter plumbing agrees."""
        base = load_by_name("ft06")
        fjsp = generate_fjsp_instance(base, max_extra_alternatives=0, seed=0)
        result = CPSATFJSPBaseline().solve(fjsp, time_limit_s=30.0)
        assert result.is_optimal
        assert result.makespan == 55

    def test_pdr_baseline_rejects_fjsp_instance(self) -> None:
        fjsp = generate_fjsp_instance(load_by_name("ft06"), seed=0)
        with pytest.raises(TypeError, match="FJSPInstance"):
            PDRBaseline("spt").solve(fjsp, seed=0, time_limit_s=None)

    def test_cpsat_fjsp_requires_time_limit(self) -> None:
        fjsp = generate_fjsp_instance(load_by_name("ft06"), seed=0)
        with pytest.raises(ValueError, match="time_limit_s"):
            CPSATFJSPBaseline().solve(fjsp, time_limit_s=None)


class TestTimeLimitRequirement:
    def test_cpsat_requires_time_limit(self) -> None:
        instance = load_by_name("ft06")
        with pytest.raises(ValueError, match="time_limit_s"):
            CPSATBaseline().solve(instance, time_limit_s=None)

    def test_ga_requires_time_limit(self) -> None:
        instance = load_by_name("ft06")
        with pytest.raises(ValueError, match="time_limit_s"):
            GABaseline(GAConfig(population_size=20)).solve(instance, time_limit_s=None)

    def test_pdr_ignores_missing_time_limit(self) -> None:
        instance = load_by_name("ft06")
        result = PDRBaseline("spt").solve(instance, seed=0, time_limit_s=None)
        assert result.method == "spt"


class TestCPSATStatusOptimalConsistency:
    def test_is_optimal_implies_status_optimal(self) -> None:
        instance = load_by_name("ft06")
        result = CPSATBaseline().solve(instance, time_limit_s=30.0)
        if result.is_optimal:
            assert result.status == "OPTIMAL"

    def test_ft06_recovers_known_optimum(self) -> None:
        instance = load_by_name("ft06")
        result = CPSATBaseline().solve(instance, time_limit_s=30.0)
        assert result.is_optimal
        assert result.status == "OPTIMAL"
        assert result.makespan == 55


class TestUnknownRuleName:
    def test_pdr_baseline_rejects_unknown_rule(self) -> None:
        with pytest.raises(KeyError):
            PDRBaseline("not_a_real_rule")


class TestTrainedRLBaselineDynamicWiring:
    """E4-E6 RL-arm wiring: the trained checkpoint evaluated live against
    ``DynamicJSSPEnv`` via ``TrainedRLBaseline.solve_dynamic``.

    This mirrors ``tests/test_dynamic_env.py::TestRegressionAgainstStatic``'s own
    replay-equivalence pattern (derive a dispatch sequence from the dynamic env's own,
    stricter, completion-gated mask; replay the identical sequence through the static
    env; makespan must match exactly) applied specifically to the *wrapped* env
    ``solve_dynamic`` actually uses -- proving ``_DynamicNodeFeatureAdapter`` is a pure
    observation-shape transform with no side effect on env dynamics.

    Note on what this does and doesn't prove (see EXPERIMENT_LOG_WEEK24.md's Finding #4
    follow-up for the full discussion): this proves the *environment mechanics* --
    including the new ``utilization_accum``/``adj_eligibility`` fields the E4-E6 wiring
    fix added -- are correct. It deliberately does not assert that the trained policy's
    own *adaptively chosen* actions are identical between static and dynamic rollouts:
    that comparison fails by construction, for a structural, pre-existing, and correct
    reason (static list-scheduling advances an operation's frontier the instant its
    predecessor is *dispatched*; the dynamic env, correctly, only once its predecessor's
    completion event actually *fires* -- necessary for breakdowns/preemption to be
    modeled correctly at all). A frozen policy can therefore end up choosing a
    different, still-legal action at a step where the two envs' legal-action sets
    happen to differ, yielding a somewhat different final makespan even with every
    disruption axis off. This is a quantified, documented limitation of evaluating a
    statically-trained policy on the dynamic environment without retraining -- not a
    defect in this wiring -- see the experiment log for the measured magnitude.
    """

    CKPT = "experiments/results/curriculum_full/checkpoints/curriculum_stage19_la20_final.zip"

    @pytest.mark.parametrize("instance_name", ["ft06", "la01"])
    def test_wrapped_dynamic_env_matches_static_under_replayed_dispatch_order(
        self, instance_name: str
    ) -> None:
        import numpy as np

        from baselines.dispatching_rules import RULES
        from baselines.interface import TrainedRLBaseline
        from envs.disruptions import DisruptionConfig
        from envs.jssp_env import JSSPEnv

        instance = load_by_name(instance_name)
        baseline = TrainedRLBaseline(policy_path=self.CKPT)
        wrapped_env = baseline._build_dynamic_env(instance, DisruptionConfig())
        wrapped_env.reset(seed=7)
        raw_env = wrapped_env.unwrapped

        static_env = JSSPEnv(instance_name)
        static_env.reset(seed=7)

        rng = np.random.default_rng(13)
        rule = RULES["mwkr"]
        terminated = False
        info_d: dict = {}
        info_s: dict = {}
        while not terminated:
            action = rule(raw_env, rng)
            obs_d, _, terminated, _, info_d = wrapped_env.step(action)
            assert obs_d["node_features"].shape[1] == 8, "wrapper must drop is_preempted"
            _, _, _, _, info_s = static_env.step(action)

        assert info_d["makespan"] == info_s["makespan"]
        assert info_d["schedule_feasible"]

    def test_solve_dynamic_runs_end_to_end_with_disruptions_off(self) -> None:
        """Basic sanity: solve_dynamic doesn't crash and returns a feasible result,
        matching Baseline.solve's contract (BaselineResult, feasible schedule)."""
        from baselines.interface import BaselineResult, TrainedRLBaseline
        from envs.disruptions import DisruptionConfig

        instance = load_by_name("ft06")
        baseline = TrainedRLBaseline(policy_path=self.CKPT)
        result = baseline.solve_dynamic(instance, DisruptionConfig(), seed=0)
        assert isinstance(result, BaselineResult)
        assert result.method == "rl"
        assert result.feasible
        assert result.makespan > 0
