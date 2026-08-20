"""Tests for experiments/stats/realized_quality.py (Fase 8, T8.2/T8.4)."""

from __future__ import annotations

import pandas as pd
import pytest

from experiments.stats.realized_quality import (
    compare_rl_vs_three_comparators,
    realized_quality_table,
    to_markdown,
)

CANDIDATES = ("fifo", "spt", "mor")


def _regime_df() -> pd.DataFrame:
    rows = []
    for method, base in [("fifo", 70.0), ("spt", 80.0), ("mor", 65.0), ("rl", 90.0)]:
        for seed in range(5):
            rows.append({"instance": "ft06", "method": method, "seed": seed, "makespan": base + seed})
    return pd.DataFrame(rows)


def _e1_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instance": ["ft06"] * 4,
            "family": ["fisher-thompson"] * 4,
            "method": ["fifo", "spt", "mor", "rl"],
            "rpd_pct": [27.27, 45.45, 18.18, 63.64],
            "known_optimum": [55, 55, 55, 55],
        }
    )


class TestRealizedQualityTable:
    def test_one_row_per_instance_method(self) -> None:
        table = realized_quality_table(_regime_df(), certified_optimum={"ft06": 55.0}, regime_label="test_regime")
        assert len(table) == 4  # fifo, spt, mor, rl
        assert set(table["method"]) == {"fifo", "spt", "mor", "rl"}
        assert (table["regime"] == "test_regime").all()

    def test_skips_instance_without_certified_optimum(self) -> None:
        table = realized_quality_table(_regime_df(), certified_optimum={}, regime_label="r")
        assert table.empty


class TestCompareRlVsThreeComparators:
    def test_produces_three_separate_comparator_rows(self) -> None:
        realized = realized_quality_table(_regime_df(), certified_optimum={"ft06": 55.0}, regime_label="r")
        comparison = compare_rl_vs_three_comparators(realized, _e1_df(), pdr_candidates=CANDIDATES, regime_label="r")
        assert set(comparison["comparator"]) == {"pdr_global", "pdr_por_familia", "oraculo_best_per_instance"}
        assert len(comparison) == 3

    def test_empty_when_no_rl_rows(self) -> None:
        realized = realized_quality_table(_regime_df()[_regime_df()["method"] != "rl"], certified_optimum={"ft06": 55.0}, regime_label="r")
        comparison = compare_rl_vs_three_comparators(realized, _e1_df(), pdr_candidates=CANDIDATES, regime_label="r")
        assert comparison.empty


class TestToMarkdown:
    def test_renders_both_sections(self) -> None:
        realized = realized_quality_table(_regime_df(), certified_optimum={"ft06": 55.0}, regime_label="r")
        comparison = compare_rl_vs_three_comparators(realized, _e1_df(), pdr_candidates=CANDIDATES, regime_label="r")
        md = to_markdown(realized, comparison)
        assert "RPD realizada" in md
        assert "oraculo_best_per_instance" in md
