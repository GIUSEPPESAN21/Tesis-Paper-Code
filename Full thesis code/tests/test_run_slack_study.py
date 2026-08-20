"""Tests for experiments/run_slack_study.py (Fase 1, T1.2).

PDR-only cases run fast (no OR-Tools/GA needed) so they stay in the default suite;
cpsat/ga coverage is intentionally left to the existing baseline test files
(tests/test_cpsat.py, tests/test_ga.py) rather than duplicated here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.run_slack_study import CSV_FIELDNAMES, _rpd, run_slack_study, write_csv


class TestRpd:
    def test_zero_when_matches_optimum(self) -> None:
        assert _rpd(55.0, 55) == pytest.approx(0.0)

    def test_positive_when_above_optimum(self) -> None:
        assert _rpd(67.0, 55) == pytest.approx(21.818, abs=1e-3)

    def test_nan_when_no_optimum(self) -> None:
        import math

        assert math.isnan(_rpd(67.0, None))


class TestWriteCsv:
    def test_raises_on_empty_rows(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no rows"):
            write_csv([], tmp_path / "out.csv")

    def test_writes_expected_columns(self, tmp_path: Path) -> None:
        row = {k: 0 for k in CSV_FIELDNAMES}
        out_path = tmp_path / "out.csv"
        write_csv([row], out_path)
        content = out_path.read_text(encoding="utf-8")
        assert content.splitlines()[0] == ",".join(CSV_FIELDNAMES)


class TestRunSlackStudyPdrOnly:
    def test_produces_one_row_per_instance_method(self) -> None:
        rows = run_slack_study(["ft06"], ["fifo", "spt"], seed=0, time_limit_s=5.0, num_workers=1)
        assert len(rows) == 2
        methods = {r["method"] for r in rows}
        assert methods == {"fifo", "spt"}
        for row in rows:
            assert row["instance"] == "ft06"
            assert row["feasible"] is True
            assert 0.0 <= row["critical_op_fraction"] <= 1.0
            assert row["total_slack"] >= 0.0

    def test_pdr_group_expands_to_nine_rules(self) -> None:
        rows = run_slack_study(["ft06"], ["pdr"], seed=0, time_limit_s=5.0, num_workers=1)
        assert len(rows) == 9
