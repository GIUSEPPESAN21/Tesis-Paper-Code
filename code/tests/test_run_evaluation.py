"""Integration tests for ``run_evaluation.py`` (Week 15): run the unified harness
end-to-end on a small instance set and check the output CSV's schema and validity,
rather than re-testing any individual method's correctness (that's what
``tests/test_baselines_interface.py``/``tests/test_dynamic_env.py``/etc. already do).

Every ``time_limit_s`` here is deliberately tiny (ft06 is 36 operations; CP-SAT proves
optimality in well under a second and the GA converges usefully in a fraction of a
second at a small population) so this file stays fast enough to run as part of the
default suite alongside the unit tests.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import run_evaluation

EXPECTED_COLUMNS = [
    "instance",
    "family",
    "num_jobs",
    "num_machines",
    "method",
    "disruption",
    "seed",
    "makespan",
    "known_optimum",
    "rpd_pct",
    "wall_clock_s",
    "feasible",
    "status",
    "is_optimal",
    "violations_count",
    "timestamp",
    "mip_gap",
    "execution_policy",
    "n_reoptimizations",
    "reoptimization_wall_clock_s",
]


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class TestEndToEndNoDisruption:
    def test_runs_and_writes_expected_row_count(self, tmp_path: Path) -> None:
        out_path = tmp_path / "results.csv"
        exit_code = run_evaluation.main(
            [
                "--instances", "ft06", "la01",
                "--methods", "spt", "mwkr", "ga", "cpsat", "rl",
                "--seeds", "0", "1",
                "--time-limit", "1.0",
                "--population-size", "20",
                "--out", str(out_path),
                "--quiet",
            ]
        )
        assert exit_code == 0
        assert out_path.exists()

        rows = _read_csv(out_path)
        # 2 instances x 5 methods x 2 seeds.
        assert len(rows) == 2 * 5 * 2

    def test_header_matches_documented_schema(self, tmp_path: Path) -> None:
        out_path = tmp_path / "results.csv"
        run_evaluation.main(
            ["--instances", "ft06", "--methods", "spt", "--seeds", "0", "--out", str(out_path), "--quiet"]
        )
        rows = _read_csv(out_path)
        assert list(rows[0].keys()) == EXPECTED_COLUMNS

    def test_every_row_is_well_formed(self, tmp_path: Path) -> None:
        out_path = tmp_path / "results.csv"
        run_evaluation.main(
            [
                "--instances", "ft06",
                "--methods", "spt", "mwkr", "ga", "cpsat",
                "--seeds", "0",
                "--time-limit", "1.0",
                "--population-size", "20",
                "--out", str(out_path),
                "--quiet",
            ]
        )
        rows = _read_csv(out_path)
        assert rows  # non-empty
        for row in rows:
            assert row["instance"] == "ft06"
            assert row["family"] == "fisher-thompson"
            assert int(row["num_jobs"]) == 6
            assert int(row["num_machines"]) == 6
            assert row["disruption"] == "none"
            assert int(row["seed"]) == 0
            assert float(row["makespan"]) > 0
            assert int(row["known_optimum"]) == 55
            assert float(row["rpd_pct"]) >= 0.0
            assert float(row["wall_clock_s"]) >= 0.0
            assert row["feasible"] == "True"
            assert row["status"] in ("OK", "OPTIMAL", "FEASIBLE")
            assert row["is_optimal"] in ("True", "False")
            assert int(row["violations_count"]) == 0

    def test_rl_placeholder_row_is_explicit_not_a_crash(self, tmp_path: Path) -> None:
        out_path = tmp_path / "results.csv"
        run_evaluation.main(["--instances", "ft06", "--methods", "rl", "--seeds", "0", "--out", str(out_path), "--quiet"])
        rows = _read_csv(out_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["method"] == "rl"
        assert row["status"] == "NOT_IMPLEMENTED"
        assert row["feasible"] == "False"
        assert row["makespan"] == "nan"
        assert int(row["violations_count"]) == 1


class TestEndToEndUnderDisruption:
    @pytest.mark.parametrize("disruption,method", [
        ("breakdowns", "mwkr"),
        ("stochastic_times", "mwkr"),
        ("arrivals", "lwkr"),
        ("combined", "lwkr"),
    ])
    def test_pdr_runs_live_against_dynamic_env(self, tmp_path: Path, disruption: str, method: str) -> None:
        out_path = tmp_path / "results.csv"
        exit_code = run_evaluation.main(
            [
                "--instances", "ft06",
                "--methods", method,
                "--seeds", "0",
                "--disruption", disruption,
                "--out", str(out_path),
                "--quiet",
            ]
        )
        assert exit_code == 0
        rows = _read_csv(out_path)
        assert len(rows) == 1
        assert rows[0]["disruption"] == disruption
        assert rows[0]["feasible"] == "True"
        assert rows[0]["status"] == "OK"

    @pytest.mark.parametrize("disruption", ["breakdowns", "arrivals"])
    def test_ga_runs_via_predictive_reactive_replay(self, tmp_path: Path, disruption: str) -> None:
        out_path = tmp_path / "results.csv"
        exit_code = run_evaluation.main(
            [
                "--instances", "ft06",
                "--methods", "ga",
                "--seeds", "0",
                "--time-limit", "1.0",
                "--population-size", "20",
                "--disruption", disruption,
                "--out", str(out_path),
                "--quiet",
            ]
        )
        assert exit_code == 0
        rows = _read_csv(out_path)
        assert len(rows) == 1
        assert rows[0]["feasible"] == "True"
        # A disrupted execution of an offline plan is essentially never faster than the
        # nominal solve it was based on.
        assert float(rows[0]["makespan"]) >= 55.0

    def test_arrival_unsafe_pdr_rejected_before_running(self) -> None:
        # mwkr became arrival-safe in T8.1 (env.duration_per_op()/env.remaining_work_per_op());
        # cr remains unsafe (due-date synthesis for arrived jobs is still unimplemented).
        with pytest.raises(SystemExit):
            run_evaluation.main(["--instances", "ft06", "--methods", "cr", "--disruption", "arrivals"])

    def test_previously_unsafe_pdr_now_runs_under_arrivals(self, tmp_path: Path) -> None:
        """T8.1 regression guard: spt/lpt/mwkr must NOT be rejected under arrivals
        anymore, and must produce a feasible row."""
        out_path = tmp_path / "results.csv"
        exit_code = run_evaluation.main(
            [
                "--instances", "ft06", "--methods", "mwkr", "--disruption", "arrivals",
                "--utilization", "0.85", "--max-dynamic-jobs", "10", "--seeds", "1",
                "--out", str(out_path),
            ]
        )
        assert exit_code == 0
        rows = _read_csv(out_path)
        assert rows[0]["status"] == "OK"
        assert rows[0]["feasible"] == "True"

    def test_arrival_unsafe_pdr_rejected_for_combined_too(self) -> None:
        with pytest.raises(SystemExit):
            run_evaluation.main(["--instances", "ft06", "--methods", "edd", "--disruption", "combined"])


class TestEndToEndFjsp:
    def test_fjsp_ga_and_cpsat_produce_blank_known_optimum(self, tmp_path: Path) -> None:
        out_path = tmp_path / "results.csv"
        exit_code = run_evaluation.main(
            [
                "--instances", "ft06",
                "--methods", "ga", "cpsat",
                "--seeds", "0",
                "--time-limit", "1.0",
                "--population-size", "20",
                "--fjsp",
                "--out", str(out_path),
                "--quiet",
            ]
        )
        assert exit_code == 0
        rows = _read_csv(out_path)
        assert len(rows) == 2
        for row in rows:
            assert row["instance"] == "ft06_fjsp"
            assert row["family"] == "fjsp-synthetic"
            assert row["known_optimum"] == ""
            assert row["rpd_pct"] == "nan"
            assert row["feasible"] == "True"

    def test_fjsp_with_disruption_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            run_evaluation.main(["--instances", "ft06", "--methods", "ga", "--fjsp", "--disruption", "breakdowns"])

    def test_fjsp_with_pdr_method_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            run_evaluation.main(["--instances", "ft06", "--methods", "spt", "--fjsp"])


class TestCombinedSeverity:
    """T7.2: ``--disruption combined`` gets three parametrized severities instead of
    being pinned to whatever --availability/--proc-sigma/--utilization default to."""

    def test_no_severity_keeps_backward_compatible_explicit_values(self) -> None:
        config = run_evaluation.build_disruption_config(
            "combined", availability=0.8, proc_sigma=0.25, utilization=0.6, max_dynamic_jobs=20
        )
        assert config.target_availability == 0.8
        assert config.processing_time_sigma == 0.25
        assert config.target_utilization == 0.6

    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_severity_overrides_explicit_values(self, level: str) -> None:
        availability, proc_sigma, utilization = run_evaluation.COMBINED_SEVERITY_LEVELS[level]
        config = run_evaluation.build_disruption_config(
            "combined",
            availability=0.8,
            proc_sigma=0.25,
            utilization=0.6,
            max_dynamic_jobs=20,
            combined_severity=level,
        )
        assert config.target_availability == availability
        assert config.processing_time_sigma == proc_sigma
        assert config.target_utilization == utilization

    def test_severity_ignored_for_non_combined_profile(self) -> None:
        config = run_evaluation.build_disruption_config(
            "breakdowns",
            availability=0.8,
            proc_sigma=0.25,
            utilization=0.6,
            max_dynamic_jobs=20,
            combined_severity="high",
        )
        assert config.target_availability == 0.8

    def test_unknown_severity_raises(self) -> None:
        with pytest.raises(ValueError, match="combined-severity"):
            run_evaluation.build_disruption_config(
                "combined",
                availability=0.8,
                proc_sigma=0.25,
                utilization=0.6,
                max_dynamic_jobs=20,
                combined_severity="extreme",
            )

    def test_severities_are_strictly_ordered_low_to_high(self) -> None:
        """Sanity check on the D4-grid mapping itself: low must be the mildest level on
        every axis, high the harshest -- guards against a future edit silently
        transposing a tuple."""
        low = run_evaluation.COMBINED_SEVERITY_LEVELS["low"]
        medium = run_evaluation.COMBINED_SEVERITY_LEVELS["medium"]
        high = run_evaluation.COMBINED_SEVERITY_LEVELS["high"]
        assert low[0] > medium[0] > high[0]  # availability: higher is milder
        assert low[1] < medium[1] < high[1]  # sigma: lower is milder
        assert low[2] < medium[2] < high[2]  # utilization: lower is milder


class TestMipGapColumn:
    """T7.1: mip_gap is propagated from BaselineResult.extra into the CSV row, and is
    NaN (not a KeyError) for methods that never set it."""

    def test_present_from_cpsat_extra(self) -> None:
        from baselines.interface import BaselineResult
        from envs.loader import load_by_name

        instance = load_by_name("ft06")
        result = BaselineResult(
            instance="ft06",
            method="cpsat",
            start_times=None,
            makespan=60.0,
            wall_clock_time=1.0,
            feasible=True,
            violations=[],
            status="FEASIBLE",
            is_optimal=False,
            extra={"best_bound": 55, "mip_gap": 0.0909},
        )
        row = run_evaluation.build_row(instance, "cpsat", "none", 0, result, "2026-01-01T00:00:00+00:00")
        assert row["mip_gap"] == pytest.approx(0.0909)

    def test_nan_when_method_never_sets_it(self) -> None:
        import math

        from baselines.interface import BaselineResult
        from envs.loader import load_by_name

        instance = load_by_name("ft06")
        result = BaselineResult(
            instance="ft06",
            method="fifo",
            start_times=None,
            makespan=67.0,
            wall_clock_time=0.01,
            feasible=True,
            violations=[],
            status="OK",
            is_optimal=False,
            extra={"dispatch_order": []},
        )
        row = run_evaluation.build_row(instance, "fifo", "none", 0, result, "2026-01-01T00:00:00+00:00")
        assert math.isnan(row["mip_gap"])


class TestExecutionPolicyT62:
    """T6.2: --execution-policy periodic/event_triggered routes ga/cpsat through
    envs.rescheduling (T6.1) instead of the default right_shift replay, and records
    the extra reoptimization cost."""

    def test_right_shift_is_default_and_zero_reoptimizations(self, tmp_path: Path) -> None:
        out_path = tmp_path / "results.csv"
        exit_code = run_evaluation.main(
            [
                "--instances", "ft06", "--methods", "ga", "--disruption", "breakdowns",
                "--seeds", "0", "--time-limit", "3.0", "--out", str(out_path), "--quiet",
            ]
        )
        assert exit_code == 0
        rows = _read_csv(out_path)
        assert rows[0]["execution_policy"] == "right_shift"
        assert rows[0]["n_reoptimizations"] == "0"

    @pytest.mark.slow
    def test_periodic_policy_reoptimizes_and_records_cost(self, tmp_path: Path) -> None:
        out_path = tmp_path / "results.csv"
        exit_code = run_evaluation.main(
            [
                "--instances", "ft06", "--methods", "ga", "--disruption", "breakdowns",
                "--seeds", "0", "--execution-policy", "periodic",
                "--reopt-period", "8.0", "--reopt-time-limit", "0.2",
                "--out", str(out_path), "--quiet",
            ]
        )
        assert exit_code == 0
        rows = _read_csv(out_path)
        assert rows[0]["execution_policy"] == "periodic"
        assert int(rows[0]["n_reoptimizations"]) >= 1
        assert float(rows[0]["reoptimization_wall_clock_s"]) > 0.0
        assert rows[0]["feasible"] == "True"

    def test_execution_policy_ignored_for_pdr(self, tmp_path: Path) -> None:
        out_path = tmp_path / "results.csv"
        exit_code = run_evaluation.main(
            [
                "--instances", "ft06", "--methods", "fifo", "--disruption", "breakdowns",
                "--seeds", "0", "--execution-policy", "periodic",
                "--out", str(out_path), "--quiet",
            ]
        )
        assert exit_code == 0
        rows = _read_csv(out_path)
        assert rows[0]["status"] == "OK"  # PDR path unaffected


class TestWriteCsvValidation:
    def test_empty_rows_raise_instead_of_writing_a_headerless_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no rows"):
            run_evaluation.write_csv([], tmp_path / "empty.csv")

    def test_unknown_method_name_raises(self) -> None:
        with pytest.raises(SystemExit):
            run_evaluation.main(["--instances", "ft06", "--methods", "not_a_real_method"])


class TestCsvHardening:
    """``E7_E8_FinalPlan.md`` §2.1 precondition #4 / ``EXPERIMENT_LOG_WEEK24.md``
    Finding #3: a single failing cell must not discard the rest of a sweep's already-
    computed results, and every successful row must land on disk incrementally, not
    only in a write-once-at-the-end call the caller never reaches on an exception.
    """

    def test_one_failing_cell_does_not_discard_the_rest_of_the_sweep(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_evaluate_one = run_evaluation.evaluate_one
        call_count = {"n": 0}

        def flaky_evaluate_one(baseline, instance, disruption_name, disruption_config, seed, time_limit_s, **kwargs):
            call_count["n"] += 1
            if baseline.name == "mwkr" and seed == 1:
                raise ValueError("simulated checkpoint-mismatch-style failure")
            return real_evaluate_one(baseline, instance, disruption_name, disruption_config, seed, time_limit_s, **kwargs)

        monkeypatch.setattr(run_evaluation, "evaluate_one", flaky_evaluate_one)

        out_path = tmp_path / "results.csv"
        exit_code = run_evaluation.main(
            [
                "--instances", "ft06",
                "--methods", "spt", "mwkr",
                "--seeds", "0", "1",
                "--out", str(out_path),
                "--quiet",
            ]
        )
        assert exit_code == 0
        # 2 methods x 2 seeds = 4 cells attempted; 1 fails (mwkr, seed=1) -> 3 rows.
        assert call_count["n"] == 4
        rows = _read_csv(out_path)
        assert len(rows) == 3
        assert not any(r["method"] == "mwkr" and r["seed"] == "1" for r in rows)

    def test_failures_are_returned_and_logged_not_silently_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_evaluate_one = run_evaluation.evaluate_one

        def flaky_evaluate_one(baseline, instance, disruption_name, disruption_config, seed, time_limit_s, **kwargs):
            if seed == 1:
                raise RuntimeError("boom")
            return real_evaluate_one(baseline, instance, disruption_name, disruption_config, seed, time_limit_s, **kwargs)

        monkeypatch.setattr(run_evaluation, "evaluate_one", flaky_evaluate_one)
        instances = run_evaluation.load_instances(["ft06"], fjsp=False, fjsp_seed=0)
        methods = run_evaluation.resolve_methods(
            ["spt"], fjsp=False, num_workers=8, population_size=None, rl_policy=None, rl_deterministic=True
        )
        disruption_config = run_evaluation.build_disruption_config(
            "none", availability=0.8, proc_sigma=0.25, utilization=0.6, max_dynamic_jobs=20
        )
        rows, failures = run_evaluation.run_evaluation(
            instances, methods, "none", disruption_config, [0, 1], 1.0, verbose=False
        )
        assert len(rows) == 1
        assert len(failures) == 1
        assert failures[0]["seed"] == 1
        assert failures[0]["error_type"] == "RuntimeError"
        assert failures[0]["error"] == "boom"

    def test_rows_are_flushed_incrementally_not_only_at_the_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash on the *last* cell must not erase rows written by earlier cells --
        the write-once-at-the-end pattern this hardens away from would lose everything
        in that scenario since ``write_csv`` would never be reached."""
        real_evaluate_one = run_evaluation.evaluate_one
        seen_on_disk_before_crash: list[dict] = []
        out_path = tmp_path / "results.csv"

        def flaky_evaluate_one(baseline, instance, disruption_name, disruption_config, seed, time_limit_s, **kwargs):
            if seed == 1:
                # By the time the last cell runs, earlier successful rows must already
                # be readable on disk -- not buffered only in the in-memory `rows` list.
                seen_on_disk_before_crash.extend(_read_csv(out_path))
                raise RuntimeError("simulated crash on the final cell")
            return real_evaluate_one(baseline, instance, disruption_name, disruption_config, seed, time_limit_s, **kwargs)

        monkeypatch.setattr(run_evaluation, "evaluate_one", flaky_evaluate_one)
        exit_code = run_evaluation.main(
            ["--instances", "ft06", "--methods", "spt", "--seeds", "0", "1", "--out", str(out_path), "--quiet"]
        )
        assert exit_code == 0
        assert len(seen_on_disk_before_crash) == 1
        assert seen_on_disk_before_crash[0]["seed"] == "0"

    def test_all_cells_failing_raises_with_failure_count_not_a_generic_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def always_fails(baseline, instance, disruption_name, disruption_config, seed, time_limit_s, **kwargs):
            raise ValueError("every cell fails")

        monkeypatch.setattr(run_evaluation, "evaluate_one", always_fails)
        out_path = tmp_path / "results.csv"
        with pytest.raises(SystemExit, match="all 1 cell"):
            run_evaluation.main(["--instances", "ft06", "--methods", "spt", "--seeds", "0", "--out", str(out_path), "--quiet"])
