"""Unit tests for the Week 9 dispatching-rule additions: LWKR, LOR, EDD, CR, and the
``"mor"`` registry alias for ``mopnr_rule`` --- completing the 9-rule PDR suite required
by Part B6 (see ``Baselines_TechSpec.md``).

Five concerns, one class each:

1. **Formula correctness** (:class:`TestNewRuleFormulaCorrectness`) --- LWKR/LOR/EDD/CR
   pick the operation their closed-form definition predicts, verified against
   hand-computed values on a small instance, across multiple decision epochs.
2. **The MOR alias** (:class:`TestMorAlias`) --- ``RULES["mor"]`` and
   ``RULES["mopnr"]`` are the exact same callable.
3. **Regression** (:class:`TestExistingRulesByteIdentical`) --- the six pre-existing
   rules (``random``, ``spt``, ``lpt``, ``fifo``, ``mwkr``, ``mopnr``) are unchanged,
   checked against a frozen snapshot (``tests/fixtures/pdr_regression_snapshot.json``).
4. **Constraint safety** (:class:`TestAllNinePDRsFeasible`) --- every one of the nine
   PDRs produces a valid, constraint-respecting schedule (the explicit STEP4
   requirement), mirroring ``tests/test_env.py::TestConstraintSafety``.
5. **Arrival safety** (:class:`TestArrivalSafety`) --- LWKR/LOR/MOR are genuinely safe
   against ``DynamicJSSPEnv`` with real arrivals enabled; EDD/CR are documented (and
   verified) to raise a clear error instead of a confusing crash in that scenario,
   since due-date synthesis for arrivals is out of scope this sprint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from baselines.dispatching_rules import (
    RULES,
    STATIC_ONLY_RULES,
    cr_rule,
    edd_rule,
    lor_rule,
    lwkr_rule,
    run_rule,
)
from baselines.due_dates import ensure_due_dates
from envs.disjunctive_graph import is_schedule_feasible
from envs.disruptions import DisruptionConfig
from envs.dynamic_jssp_env import DynamicJSSPEnv
from envs.instance import JSSPInstance
from envs.jssp_env import JSSPEnv

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "pdr_regression_snapshot.json"

#: The nine canonical Part B6 rule names (excludes the "mopnr"/"mor" duplicate entry).
NINE_PDR_NAMES = ["spt", "lpt", "mwkr", "lwkr", "mor", "lor", "fifo", "edd", "cr"]


@pytest.fixture
def richer_toy() -> JSSPInstance:
    """2 jobs x 3 ops, 3 machines, hand-picked so remaining-work/op-count/due-date
    rankings are never coincidentally tied across the three decision epochs used below.

    job0: op0(m0,dur2), op1(m1,dur3), op2(m2,dur1)  -- job_total=6
    job1: op3(m1,dur4), op4(m2,dur2), op5(m0,dur5)  -- job_total=11
    """
    return JSSPInstance(
        name="richer_toy",
        machines=np.array([[0, 1, 2], [1, 2, 0]]),
        durations=np.array([[2, 3, 1], [4, 2, 5]]),
        num_machines=3,
    )


class TestNewRuleFormulaCorrectness:
    def test_lwkr_and_lor_pick_predicted_op_across_three_epochs(self, richer_toy) -> None:
        env = JSSPEnv(richer_toy)
        env.reset()
        rng = np.random.default_rng(0)

        # Epoch 1: legal=[op0 (job0/pos0, remaining=6), op3 (job1/pos0, remaining=11)].
        # LWKR picks the smaller remaining work. LOR is not meaningfully testable at this
        # very first epoch: every job's frontier is at position 0, so LOR's key
        # (ops_per_job - position) is identical (3) for every legal op in any rectangular
        # JSSP instance -- a structural tie, not a bug.
        assert lwkr_rule(env, rng) == 0

        env.step(0)  # advance manually, NOT via the rule under test

        # Epoch 2: legal=[op1 (job0/pos1, remaining=4), op3 (job1/pos0, remaining=11)].
        assert lwkr_rule(env, rng) == 1
        # op1 has 2 ops remaining (3 - 1), op3 has 3 (3 - 0): first non-degenerate LOR
        # decision.
        assert lor_rule(env, rng) == 1

        env.step(1)

        # Epoch 3: legal=[op2 (job0/pos2, remaining=1), op3 (job1/pos0, remaining=11)].
        assert lwkr_rule(env, rng) == 2
        assert lor_rule(env, rng) == 2

    def test_edd_and_cr_use_synthesized_due_dates(self, richer_toy) -> None:
        env = JSSPEnv(richer_toy)
        env.reset()
        rng = np.random.default_rng(0)

        due_dates = ensure_due_dates(env)  # default tightness=1.5
        # job_totals = [6, 11] -> due dates = [9.0, 16.5]
        np.testing.assert_allclose(due_dates, [9.0, 16.5])

        # Epoch 1: EDD picks job0 (due 9 < 16.5) -> op0. CR is not meaningfully testable
        # here: at t_now=0 with remaining_work == job_totals, CR == tightness for every
        # job's very first operation -- a structural tie, verified at epoch 2 instead.
        assert edd_rule(env, rng) == 0

        env.step(0)

        # Epoch 2: legal=[op1 (job0/pos1), op3 (job1/pos0)].
        assert edd_rule(env, rng) == 1  # job0's due date (9) still nearer than job1's (16.5)
        # CR(op1) = (9 - t_now=2) / remaining=4 = 1.75
        # CR(op3) = (16.5 - t_now=0) / remaining=11 = 1.5  -> op3 wins
        assert cr_rule(env, rng) == 3

        env.step(1)

        # Epoch 3: legal=[op2 (job0/pos2), op3 (job1/pos0)].
        # CR(op2) = (9 - t_now=5) / remaining=1 = 4.0
        # CR(op3) = (16.5 - t_now=5) / remaining=11 ~= 1.045 -> op3 wins again
        assert cr_rule(env, rng) == 3


class TestMorAlias:
    def test_mor_alias_matches_mopnr(self) -> None:
        assert RULES["mor"] is RULES["mopnr"]
        env_a = JSSPEnv("ft06")
        env_b = JSSPEnv("ft06")
        result_a = run_rule(env_a, "mor", seed=3)
        result_b = run_rule(env_b, "mopnr", seed=3)
        assert result_a["dispatch_order"] == result_b["dispatch_order"]
        assert result_a["makespan"] == result_b["makespan"]


class TestExistingRulesByteIdentical:
    """The single most important test in this sprint: none of the six pre-existing
    rules' behavior changed while adding the other four."""

    @pytest.fixture
    def snapshot(self) -> list[dict]:
        with FIXTURE_PATH.open(encoding="utf-8") as handle:
            return json.load(handle)

    def test_existing_rules_byte_identical_to_snapshot(self, snapshot: list[dict]) -> None:
        assert snapshot, "regression fixture must not be empty"
        for record in snapshot:
            env = JSSPEnv(record["instance"])
            result = run_rule(env, record["rule"], seed=record["seed"])
            assert result["dispatch_order"] == record["dispatch_order"], (
                record["rule"],
                record["instance"],
                record["seed"],
            )
            assert result["makespan"] == pytest.approx(record["makespan"])


class TestAllNinePDRsFeasible:
    """Explicit STEP4 requirement: every one of the nine PDRs produces a valid,
    constraint-respecting schedule (mirroring ``tests/test_env.py::TestConstraintSafety``,
    which covers this incidentally via ``RULES.keys()`` but doesn't name the nine
    canonical names explicitly)."""

    def test_registry_has_all_nine_canonical_names(self) -> None:
        for name in NINE_PDR_NAMES:
            assert name in RULES, name

    @pytest.mark.parametrize("rule_name", NINE_PDR_NAMES)
    @pytest.mark.parametrize("instance_name", ["ft06", "ft10", "la01"])
    def test_rule_produces_feasible_schedule(self, rule_name: str, instance_name: str) -> None:
        env = JSSPEnv(instance_name)
        result = run_rule(env, rule_name, seed=0)
        assert result["feasible"], (rule_name, instance_name, result["violations"])
        feasible, violations = is_schedule_feasible(env.instance, env.start_times)
        assert feasible, violations
        assert sorted(result["dispatch_order"]) == list(range(env.instance.num_operations))


class TestArrivalSafety:
    @pytest.mark.parametrize("rule_name", ["lwkr", "lor", "mor", "fifo", "spt", "lpt", "mwkr"])
    def test_arrival_safe_rules_terminate_feasibly_with_real_arrivals(self, rule_name: str) -> None:
        """spt/lpt/mwkr added T8.1 (Recomendacion del tutor): env.duration_per_op()/
        env.remaining_work_per_op() made them arrival-safe, same guarantee as the
        pre-existing four."""
        config = DisruptionConfig(target_utilization=0.85, max_dynamic_jobs=10, seed=1)
        env = DynamicJSSPEnv("ft06", config)
        result = run_rule(env, rule_name, seed=1)
        assert result["feasible"], (rule_name, result["violations"])
        assert env.num_jobs_arrived > env.base_num_jobs, "test is vacuous if no arrival occurred"

    @pytest.mark.parametrize("rule_name", ["edd", "cr"])
    def test_due_date_rules_raise_clear_error_with_real_arrivals(self, rule_name: str) -> None:
        config = DisruptionConfig(target_utilization=0.85, max_dynamic_jobs=10, seed=1)
        env = DynamicJSSPEnv("ft06", config)
        with pytest.raises(NotImplementedError, match="dynamically arrived"):
            run_rule(env, rule_name, seed=1)


def test_static_only_rules_is_empty_this_sprint() -> None:
    """All four new rules passed the arrival-safety bar for RULES membership (their
    known limitation -- EDD/CR under genuine arrivals -- raises a clear error rather
    than silently misbehaving, see :class:`TestArrivalSafety`); STATIC_ONLY_RULES is
    kept empty but exported for forward-compatibility per ``Baselines_TechSpec.md`` §8.
    """
    assert STATIC_ONLY_RULES == {}
