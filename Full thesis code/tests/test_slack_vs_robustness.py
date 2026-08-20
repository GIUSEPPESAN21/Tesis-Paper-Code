"""Unit tests for :mod:`experiments.slack_vs_robustness` (Fase 1, T1.3)."""

from __future__ import annotations

import pandas as pd
import pytest

from experiments.slack_vs_robustness import (
    SLACK_METRIC_COLUMNS,
    build_slack_vs_robustness_table,
    correlate_slack_vs_target,
    to_markdown,
)


def _slack_df() -> pd.DataFrame:
    # 5 (instance, method) points, slack_ratio increasing monotonically.
    return pd.DataFrame(
        {
            "instance": ["ft06"] * 5,
            "method": ["a", "b", "c", "d", "e"],
            "total_slack": [1.0, 2.0, 3.0, 4.0, 5.0],
            "mean_slack": [0.1, 0.2, 0.3, 0.4, 0.5],
            "slack_ratio": [0.05, 0.10, 0.15, 0.20, 0.25],
            "critical_path_length": [50.0] * 5,
            "critical_op_fraction": [0.9, 0.7, 0.5, 0.3, 0.1],
            "machine_idle_fraction": [0.1, 0.1, 0.1, 0.1, 0.1],
        }
    )


class TestCorrelateSlackVsTarget:
    def test_reports_negative_correlation_when_present(self) -> None:
        """RI decreasing as slack increases -- the hypothesis's predicted direction --
        must be reported with its actual (negative) sign, not flipped or relabeled."""
        target = pd.DataFrame(
            {"instance": ["ft06"] * 5, "method": ["a", "b", "c", "d", "e"], "ri": [0.5, 0.4, 0.3, 0.2, 0.1]}
        )
        result = correlate_slack_vs_target(_slack_df(), target, target_col="ri", regime_label="test_regime")
        slack_ratio_row = result[result["metric"] == "slack_ratio"].iloc[0]
        assert slack_ratio_row["spearman_rho"] == pytest.approx(-1.0)
        assert slack_ratio_row["regime"] == "test_regime"

    def test_reports_positive_correlation_without_filtering_it_out(self) -> None:
        """The opposite (and hypothesis-contradicting) direction: RI INCREASING as
        slack increases. Must still be reported, with a positive sign, not dropped."""
        target = pd.DataFrame(
            {"instance": ["ft06"] * 5, "method": ["a", "b", "c", "d", "e"], "ri": [0.1, 0.2, 0.3, 0.4, 0.5]}
        )
        result = correlate_slack_vs_target(_slack_df(), target, target_col="ri", regime_label="contradicts_hypothesis")
        slack_ratio_row = result[result["metric"] == "slack_ratio"].iloc[0]
        assert slack_ratio_row["spearman_rho"] == pytest.approx(1.0)
        assert len(result) == len(SLACK_METRIC_COLUMNS)  # every metric still present, none dropped

    def test_raises_on_too_few_paired_observations(self) -> None:
        target = pd.DataFrame({"instance": ["ft06"], "method": ["a"], "ri": [0.5]})
        with pytest.raises(ValueError, match="at least 3"):
            correlate_slack_vs_target(_slack_df(), target, target_col="ri", regime_label="too_small")

    def test_all_six_metrics_present(self) -> None:
        target = pd.DataFrame(
            {"instance": ["ft06"] * 5, "method": ["a", "b", "c", "d", "e"], "ri": [0.5, 0.4, 0.3, 0.2, 0.1]}
        )
        result = correlate_slack_vs_target(_slack_df(), target, target_col="ri", regime_label="r")
        assert set(result["metric"]) == set(SLACK_METRIC_COLUMNS)


class TestBuildSlackVsRobustnessTable:
    def test_concatenates_across_regimes_and_targets(self) -> None:
        ri_target = pd.DataFrame(
            {"instance": ["ft06"] * 5, "method": ["a", "b", "c", "d", "e"], "ri": [0.5, 0.4, 0.3, 0.2, 0.1]}
        )
        rpd_target = pd.DataFrame(
            {"instance": ["ft06"] * 5, "method": ["a", "b", "c", "d", "e"], "rpd_realized_pct": [40, 30, 20, 10, 5]}
        )
        table = build_slack_vs_robustness_table(
            _slack_df(), ri_by_regime={"regime1": ri_target}, rpd_by_regime={"regime1": rpd_target}
        )
        assert set(table["target"]) == {"ri", "rpd_realized_pct"}
        assert len(table) == 2 * len(SLACK_METRIC_COLUMNS)

    def test_empty_when_no_regimes_given(self) -> None:
        table = build_slack_vs_robustness_table(_slack_df(), ri_by_regime={}, rpd_by_regime={})
        assert table.empty


class TestToMarkdown:
    def test_handles_empty_table(self) -> None:
        assert "PENDIENTE-CORRIDA" in to_markdown(pd.DataFrame())

    def test_renders_signed_rho(self) -> None:
        df = pd.DataFrame(
            [{"regime": "r", "metric": "slack_ratio", "target": "ri", "spearman_rho": -0.95, "spearman_p": 1e-5, "pearson_r": -0.9, "pearson_p": 1e-4, "n": 10}]
        )
        md = to_markdown(df)
        assert "-0.9500" in md
