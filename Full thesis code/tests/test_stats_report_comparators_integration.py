"""Integration test: T4.1/T4.2's comparator tables get wired into
``generate_chapter6_artifacts`` without disturbing the pre-existing E1/E3 artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.stats.report import ALL_PDR_NAMES, generate_chapter6_artifacts


def _combined_frame() -> pd.DataFrame:
    e1_rows = []
    for instance, family in (("ft06", "fisher-thompson"), ("la01", "lawrence")):
        for i, method in enumerate(ALL_PDR_NAMES):
            e1_rows.append(
                {
                    "instance": instance, "family": family, "method": method,
                    "disruption": "none", "seed": 0, "rpd_pct": 10.0 + i, "makespan": 100.0 + i,
                    "num_operations": 36, "wall_clock_s": 0.01, "ms_per_decision": 0.01 / 36 * 1000,
                    "experiment_id": "E1",
                }
            )
        e1_rows.append(
            {
                "instance": instance, "family": family, "method": "rl", "disruption": "none",
                "seed": 0, "rpd_pct": 15.0, "makespan": 105.0, "num_operations": 36,
                "wall_clock_s": 0.02, "ms_per_decision": 0.02 / 36 * 1000, "experiment_id": "E1",
            }
        )
    return pd.DataFrame(e1_rows)


class TestComparatorTablesWiredIntoChapter6:
    def test_comparadores_pdr_table_written_when_e1_present(self, tmp_path: Path) -> None:
        combined = _combined_frame()
        written = generate_chapter6_artifacts(combined, tmp_path, h2_verdicts=None)
        assert "comparadores_pdr_e1" in written
        assert written["comparadores_pdr_e1"].exists()
        assert "oraculo" in written["comparadores_pdr_e1"].read_text(encoding="utf-8").lower()

    def test_does_not_crash_without_e3_data(self, tmp_path: Path) -> None:
        combined = _combined_frame()  # no E3 rows at all
        written = generate_chapter6_artifacts(combined, tmp_path, h2_verdicts=None)
        assert "e3_h2_dual_reading" not in written

    def test_does_not_crash_on_empty_frame(self, tmp_path: Path) -> None:
        empty = pd.DataFrame(columns=["instance", "family", "method", "disruption", "seed", "rpd_pct", "experiment_id"])
        written = generate_chapter6_artifacts(empty, tmp_path, h2_verdicts=None)
        assert "comparadores_pdr_e1" not in written
