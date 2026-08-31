"""Unit tests for :mod:`experiments.stats.pairwise`, plus the golden regression tests
``StatsAnalysis_TechSpec.md`` Section 10 pins against ``HPO_ABLATION_REPORT_WEEK21.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.stats.pairwise import (
    H4_MARGIN_PP,
    align_paired,
    h1_fjsp_test,
    h1_jssp_test,
    h3_overall_verdict,
    h3_regime_test,
    h4_overall_verdict,
    h4_test,
    h5_check,
    wilcoxon_signed_rank,
)


class TestWilcoxonSignedRank:
    def test_raises_on_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            wilcoxon_signed_rank(np.array([1.0, 2.0]), np.array([1.0]))

    def test_raises_on_empty(self) -> None:
        with pytest.raises(ValueError):
            wilcoxon_signed_rank(np.array([]), np.array([]))

    def test_all_zero_difference_is_degenerate_not_raised(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        result = wilcoxon_signed_rank(x, y)
        assert result.all_zero is True
        assert result.statistic == 0.0
        assert result.p_value == 1.0
        assert result.z == 0.0
        assert result.effect_size_r == 0.0

    def test_median_matches_week21_reward_density(self) -> None:
        """StatsAnalysis_TechSpec.md Sec 10: dense vs sparse RPDs (Axis 2, 5 seeds).
        Expect statistic (W) == 0, one-sided p == 0.0312, two-sided p == 0.0625.
        """
        dense = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        sparse = np.array([7.27, 7.27, 9.09, 7.27, 7.27])
        two_sided = wilcoxon_signed_rank(dense, sparse, alternative="two-sided")
        assert two_sided.statistic == pytest.approx(0.0)
        assert two_sided.p_value == pytest.approx(0.0625, abs=1e-4)

        one_sided = wilcoxon_signed_rank(dense, sparse, alternative="less")
        assert one_sided.p_value_one_sided == pytest.approx(0.0312, abs=1e-4)

    def test_z_sign_matches_median_diff_sign(self) -> None:
        x = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        y = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = wilcoxon_signed_rank(x, y)
        assert result.median_diff > 0
        assert result.z > 0
        assert result.effect_size_r > 0

        result_rev = wilcoxon_signed_rank(y, x)
        assert result_rev.median_diff < 0
        assert result_rev.z < 0


class TestAlignPaired:
    def test_basic_alignment(self) -> None:
        df = pd.DataFrame(
            {
                "instance": ["ft06", "la01", "ft06", "la01"],
                "method": ["pdr", "pdr", "rl", "rl"],
                "value": [10.0, 20.0, 5.0, 25.0],
            }
        )
        is_pdr = df["method"] == "pdr"
        x, y, keys = align_paired(df, key_cols=["instance"], x_col="value", y_col="value", x_filter=is_pdr, y_filter=~is_pdr)
        assert keys == [("ft06",), ("la01",)]
        np.testing.assert_array_equal(x, [10.0, 20.0])
        np.testing.assert_array_equal(y, [5.0, 25.0])

    def test_raises_on_duplicate_key(self) -> None:
        df = pd.DataFrame({"instance": ["ft06", "ft06"], "value": [1.0, 2.0]})
        with pytest.raises(ValueError):
            align_paired(df, key_cols=["instance"], x_col="value", y_col="value")


class TestH1JsspTest:
    def test_accepts_when_rl_beats_pdrs_significantly(self) -> None:
        # n=5 with all-same-sign diffs only reaches exact two-sided p=2/32=0.0625 (>0.05),
        # so this needs n=6 (exact p=2/64=0.03125<0.05) to actually cross the frozen
        # alpha=0.05 threshold -- not a fixture that merely "looks" significant.
        rpd_best_pdr = np.array([30.0, 25.0, 40.0, 35.0, 20.0, 28.0])
        rpd_rl = np.array([5.0, 4.0, 6.0, 5.0, 3.0, 4.0])
        verdict = h1_jssp_test(rpd_best_pdr, rpd_rl)
        assert verdict.verdict == "accept"

    def test_rejects_when_not_significant(self) -> None:
        rpd_best_pdr = np.array([5.0, 5.0])
        rpd_rl = np.array([5.0, 5.0])
        verdict = h1_jssp_test(rpd_best_pdr, rpd_rl)
        assert verdict.verdict == "reject"

    def test_rejects_when_significant_but_wrong_direction(self) -> None:
        rpd_best_pdr = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
        rpd_rl = np.array([10.0, 20.0, 10.0, 20.0, 10.0])
        verdict = h1_jssp_test(rpd_best_pdr, rpd_rl)
        assert verdict.verdict == "reject"


class TestH1FjspTest:
    def test_arm_label_recorded(self) -> None:
        verdict = h1_fjsp_test(np.array([100.0, 200.0]), np.array([90.0, 180.0]), arm="synthetic")
        assert "synthetic" in verdict.hypothesis
        assert verdict.raw["arm"] == "synthetic"


class TestH3RegimeTest:
    def test_label_included_in_hypothesis_name(self) -> None:
        verdict = h3_regime_test(np.array([0.1, 0.2, 0.15]), np.array([-0.1, -0.05, 0.0]), "E4_breakdowns_A0.90")
        assert "E4_breakdowns_A0.90" in verdict.hypothesis


class TestH3OverallVerdict:
    def _verdict(self, name: str, accept: bool):
        from experiments.stats.pairwise import HypothesisVerdict

        return HypothesisVerdict(name, "accept" if accept else "reject", None, "", raw={})

    def test_all_nine_accept_is_overall_accept(self) -> None:
        regimes = [self._verdict(f"r{i}", True) for i in range(9)]
        overall = h3_overall_verdict(regimes)
        assert overall.verdict == "accept"

    def test_some_accept_is_partial(self) -> None:
        regimes = [self._verdict(f"r{i}", i < 5) for i in range(9)]
        overall = h3_overall_verdict(regimes)
        assert overall.verdict == "partial"

    def test_none_accept_is_reject(self) -> None:
        regimes = [self._verdict(f"r{i}", False) for i in range(9)]
        overall = h3_overall_verdict(regimes)
        assert overall.verdict == "reject"


class TestH4:
    def test_a_and_b_both_hold_is_accept(self) -> None:
        # n=5 with all-same-sign diffs only reaches exact two-sided p=2/32=0.0625 (>0.05);
        # n=6 (exact p=2/64=0.03125) is needed to actually cross the alpha=0.05 threshold.
        ri_kappa = np.array([-0.2] * 6)
        ri_kappa0 = np.array([0.1] * 6)
        rpd_nominal_kappa = np.array([10.0])
        rpd_nominal_kappa0 = np.array([9.0])
        regimes = [
            h4_test(ri_kappa, ri_kappa0, rpd_nominal_kappa, rpd_nominal_kappa0, 0.5, regime_label=f"r{i}")
            for i in range(9)
        ]
        overall = h4_overall_verdict(regimes, 0.5)
        assert overall.raw["a_overall"] is True
        assert overall.raw["b_overall"] is True
        assert overall.verdict == "accept"

    def test_a_holds_b_fails_is_unacceptable_nominal_cost(self) -> None:
        ri_kappa = np.array([-0.2] * 6)
        ri_kappa0 = np.array([0.1] * 6)
        rpd_nominal_kappa = np.array([20.0])
        rpd_nominal_kappa0 = np.array([9.0])  # delta = 11pp > 2pp margin
        regimes = [
            h4_test(ri_kappa, ri_kappa0, rpd_nominal_kappa, rpd_nominal_kappa0, 0.5, regime_label=f"r{i}")
            for i in range(9)
        ]
        overall = h4_overall_verdict(regimes, 0.5)
        assert overall.raw["a_overall"] is True
        assert overall.raw["b_overall"] is False
        assert overall.verdict == "reject"
        assert "unacceptable nominal cost" in overall.notes

    def test_neither_holds_is_reject(self) -> None:
        ri_kappa = np.array([0.1] * 5)  # not < ri_kappa0, component (a) fails
        ri_kappa0 = np.array([0.1] * 5)
        rpd_nominal_kappa = np.array([20.0])
        rpd_nominal_kappa0 = np.array([9.0])
        regimes = [
            h4_test(ri_kappa, ri_kappa0, rpd_nominal_kappa, rpd_nominal_kappa0, 0.5, regime_label=f"r{i}")
            for i in range(9)
        ]
        overall = h4_overall_verdict(regimes, 0.5)
        assert overall.raw["a_overall"] is False
        assert overall.verdict == "reject"

    def test_margin_constant_is_two_percentage_points(self) -> None:
        assert H4_MARGIN_PP == 2.0


class TestH5Check:
    def test_accept_when_rl_under_one_percent_everywhere(self) -> None:
        rl = {"ft06": np.array([0.5, 0.6, 0.55]), "la05": np.array([1.0, 1.1])}
        cpsat = {"ft06": np.array([100.0, 110.0]), "la05": np.array([200.0, 210.0])}
        verdict = h5_check(rl, cpsat)
        assert verdict.verdict == "accept"
        assert all(r["passes"] for r in verdict.raw["per_instance"])

    def test_reject_when_any_instance_violates_threshold(self) -> None:
        rl = {"ft06": np.array([0.5]), "la05": np.array([50.0])}  # la05 fails 1% threshold
        cpsat = {"ft06": np.array([100.0]), "la05": np.array([200.0])}
        verdict = h5_check(rl, cpsat)
        assert verdict.verdict == "reject"

    def test_missing_instance_on_one_side_is_excluded_not_fatal(self, capsys) -> None:
        rl = {"ft06": np.array([0.5]), "extra": np.array([1.0])}
        cpsat = {"ft06": np.array([100.0])}
        verdict = h5_check(rl, cpsat)
        assert len(verdict.raw["per_instance"]) == 1
        assert "missing from one side" in capsys.readouterr().out


class TestFriedmanGoldenWeek21:
    def test_friedman_matches_week21_ft06_ablation(self) -> None:
        """Reference/homogeneous/gat_hybrid RPDs from HPO_ABLATION_REPORT_WEEK21.md's
        ft06 ablation table (5 seeds each). Expect chi_square == 2.0000, p == 0.3679
        (to 4 decimals), df == 2.
        """
        from experiments.stats.omnibus import friedman_test

        pivoted = pd.DataFrame(
            {
                "reference": [0.0, 0.0, 0.0, 0.0, 0.0],
                "homogeneous": [0.0, 0.0, 3.64, 0.0, 0.0],
                "gat_hybrid": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        result = friedman_test(pivoted)
        assert result.chi_square == pytest.approx(2.0000, abs=1e-4)
        assert round(result.p_value, 4) == pytest.approx(0.3679, abs=1e-4)
        assert result.df == 2

    def test_wilcoxon_handles_all_zero_difference_week21_reference_vs_gat_hybrid(self) -> None:
        """Reference vs gat_hybrid: both arms identical in every seed (all-zero diffs).
        Expect all_zero=True, statistic=0, p_value=1.0, effect_size_r=0.0.
        """
        reference = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        gat_hybrid = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        result = wilcoxon_signed_rank(reference, gat_hybrid)
        assert result.all_zero is True
        assert result.statistic == 0.0
        assert result.p_value == 1.0
        assert result.effect_size_r == 0.0
