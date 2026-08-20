"""Unit tests for :mod:`experiments.stats.comparators` (Fase 4, T4.1/T4.2)."""

from __future__ import annotations

import pandas as pd
import pytest

from experiments.stats.comparators import (
    comparator_gap_summary,
    h2_absolute_competitive_table,
    h2_dual_reading_markdown,
    oracle_best_per_instance,
    pdr_global,
    pdr_por_familia,
    tabla_comparadores_pdr_markdown,
)

CANDIDATES = ("fifo", "spt", "mor")


def _e1_frame() -> pd.DataFrame:
    # ft06/la01 -> fisher-thompson/lawrence; fifo wins both instances on mean, but mor
    # wins ft06 alone -- exercises pdr_global != oracle's per-instance pick on ft20.
    return pd.DataFrame(
        {
            "instance": ["ft06", "ft06", "ft06", "la01", "la01", "la01", "ft20", "ft20", "ft20"],
            "family": ["fisher-thompson"] * 3 + ["lawrence"] * 3 + ["fisher-thompson"] * 3,
            "method": ["fifo", "spt", "mor"] * 3,
            "rpd_pct": [20.0, 30.0, 10.0, 15.0, 25.0, 22.0, 18.0, 40.0, 19.0],
        }
    )


class TestPdrGlobal:
    def test_picks_lowest_mean_rpd_method(self):
        result = pdr_global(_e1_frame(), candidates=CANDIDATES)
        # means: fifo=(20+15+18)/3=17.667, spt=(30+25+40)/3=31.667, mor=(10+22+19)/3=17.0
        assert result.method == "mor"
        assert result.mean_rpd_pct == pytest.approx(17.0)
        assert result.n_instances == 3

    def test_raises_when_no_candidate_rows(self):
        with pytest.raises(ValueError):
            pdr_global(_e1_frame(), candidates=("nonexistent",))


class TestPdrPorFamilia:
    def test_best_rule_differs_by_family_when_it_does(self):
        results = {r.family: r for r in pdr_por_familia(_e1_frame(), candidates=CANDIDATES)}
        # fisher-thompson (ft06, ft20): fifo mean=19.0, spt mean=35.0, mor mean=14.5 -> mor
        assert results["fisher-thompson"].method == "mor"
        # lawrence (la01 only): fifo=15.0, spt=25.0, mor=22.0 -> fifo
        assert results["lawrence"].method == "fifo"


class TestOracleBestPerInstance:
    def test_labels_are_oracle_prefixed(self):
        df = oracle_best_per_instance(_e1_frame(), candidates=CANDIDATES)
        assert set(df.columns) == {"instance", "oracle_winning_pdr", "oracle_best_per_instance_rpd_pct"}

    def test_minimum_per_instance(self):
        df = oracle_best_per_instance(_e1_frame(), candidates=CANDIDATES).set_index("instance")
        assert df.loc["ft06", "oracle_winning_pdr"] == "mor"
        assert df.loc["ft06", "oracle_best_per_instance_rpd_pct"] == pytest.approx(10.0)
        assert df.loc["la01", "oracle_winning_pdr"] == "fifo"
        assert df.loc["ft20", "oracle_winning_pdr"] == "fifo"

    def test_never_beats_a_lower_bound_it_cannot_exceed(self):
        """The oracle's mean RPD can never be higher than pdr_global's -- it is a
        minimum-per-instance selection over the same candidate set."""
        oracle_df = oracle_best_per_instance(_e1_frame(), candidates=CANDIDATES)
        glob = pdr_global(_e1_frame(), candidates=CANDIDATES)
        assert oracle_df["oracle_best_per_instance_rpd_pct"].mean() <= glob.mean_rpd_pct


class TestComparatorGapSummary:
    def test_gap_is_nonnegative_by_construction(self):
        summary = comparator_gap_summary(_e1_frame(), candidates=CANDIDATES)
        assert summary["gap_pp"] >= 0
        assert summary["pdr_global_method"] == "mor"
        assert summary["oracle_n_distinct_pdrs"] == 2  # mor (ft06, ft20), fifo (la01)
        assert summary["oracle_winning_pdrs"] == ["fifo", "mor"]


class TestTablaComparadoresPdrMarkdown:
    def test_mentions_oracle_explicitly(self):
        md = tabla_comparadores_pdr_markdown(_e1_frame(), candidates=CANDIDATES)
        assert "oraculo" in md.lower()
        assert "pdr_global" in md
        assert "pdr_por_familia" in md


class TestH2AbsoluteCompetitiveTable:
    def _e3_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "instance": ["la21", "la21", "la26", "la26"],
                "method": ["rl", "fifo", "rl", "fifo"],
                "rpd_pct": [30.0, 24.0, 35.0, 27.0],
            }
        )

    def test_gap_is_rl_minus_pdr_global(self):
        tiers = {"tier1": ("la21",), "tier2": ("la26",)}
        df = h2_absolute_competitive_table(self._e3_frame(), tiers, pdr_global_method="fifo")
        row1 = df[df["tier"] == "tier1"].iloc[0]
        assert row1["gap_pp"] == pytest.approx(6.0)
        row2 = df[df["tier"] == "tier2"].iloc[0]
        assert row2["gap_pp"] == pytest.approx(8.0)

    def test_skips_tier_with_no_matching_rows(self):
        tiers = {"tier1": ("la21",), "empty_tier": ("la99",)}
        df = h2_absolute_competitive_table(self._e3_frame(), tiers, pdr_global_method="fifo")
        assert list(df["tier"]) == ["tier1"]


class TestH2DualReadingMarkdown:
    def test_includes_both_readings(self):
        abs_df = pd.DataFrame(
            [{"tier": "tier1", "rl_mean_rpd_pct": 30.0, "pdr_global_mean_rpd_pct": 24.0, "gap_pp": 6.0}]
        )
        md = h2_dual_reading_markdown(abs_df, {"tier1": "accept, rank tied"}, pdr_global_method="fifo")
        assert "accept, rank tied" in md
        assert "6.0000" in md or "+6.0000" in md
        assert "NO significa" in md
