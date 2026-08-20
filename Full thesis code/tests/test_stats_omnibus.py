"""Unit tests for :mod:`experiments.stats.omnibus`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.stats.omnibus import _DEMSAR_Q_ALPHA_005, critical_difference, friedman_test, nemenyi_posthoc


class TestFriedmanTest:
    def test_raises_on_missing_cell(self) -> None:
        pivoted = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]}, index=["i1", "i2"])
        with pytest.raises(ValueError, match="missing cell"):
            friedman_test(pivoted)

    def test_raises_on_fewer_than_two_methods(self) -> None:
        pivoted = pd.DataFrame({"a": [1.0, 2.0]}, index=["i1", "i2"])
        with pytest.raises(ValueError):
            friedman_test(pivoted)

    def test_average_ranks_lower_is_better(self) -> None:
        pivoted = pd.DataFrame(
            {"best": [1.0, 1.0, 1.0], "worst": [10.0, 10.0, 10.0]}, index=["i1", "i2", "i3"]
        )
        result = friedman_test(pivoted)
        assert result.average_ranks["best"] < result.average_ranks["worst"]
        assert result.k_methods == 2
        assert result.n_blocks == 3


class TestCriticalDifference:
    @pytest.mark.parametrize("k", sorted(_DEMSAR_Q_ALPHA_005))
    def test_matches_demsar_reference_table(self, k: int) -> None:
        """Section 7.2's own cross-check: qsturng-computed q_alpha must match Demsar
        (2006) Table 5(b) to 2 decimal places for k<=10, alpha=0.05."""
        cd = critical_difference(k, n=10, alpha=0.05)
        expected_q = _DEMSAR_Q_ALPHA_005[k]
        expected_cd = expected_q * np.sqrt(k * (k + 1) / (6.0 * 10))
        assert cd == pytest.approx(expected_cd, abs=0.02)

    def test_raises_on_k_below_two(self) -> None:
        with pytest.raises(ValueError):
            critical_difference(1, n=10)

    def test_raises_on_n_below_one(self) -> None:
        with pytest.raises(ValueError):
            critical_difference(3, n=0)

    def test_cd_shrinks_with_more_blocks(self) -> None:
        cd_small_n = critical_difference(5, n=5)
        cd_large_n = critical_difference(5, n=50)
        assert cd_large_n < cd_small_n


class TestNemenyiPosthoc:
    def test_raises_when_friedman_not_significant(self) -> None:
        from experiments.stats.omnibus import FriedmanResult

        non_sig = FriedmanResult(chi_square=0.1, df=2, p_value=0.9, k_methods=3, n_blocks=5, average_ranks={"a": 2.0, "b": 2.0, "c": 2.0})
        with pytest.raises(ValueError, match="not rejected"):
            nemenyi_posthoc(non_sig, cd=0.5)

    def test_flags_pairs_beyond_cd(self) -> None:
        from experiments.stats.omnibus import FriedmanResult

        sig = FriedmanResult(
            chi_square=10.0, df=2, p_value=0.01, k_methods=3, n_blocks=10,
            average_ranks={"a": 1.0, "b": 2.0, "c": 3.0},
        )
        flags = nemenyi_posthoc(sig, cd=1.5)
        assert flags.loc["a", "c"] is np.True_ or flags.loc["a", "c"] == True  # noqa: E712
        assert flags.loc["a", "b"] is np.False_ or flags.loc["a", "b"] == False  # noqa: E712
