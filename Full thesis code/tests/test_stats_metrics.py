"""Unit tests for :mod:`experiments.stats.metrics`."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.stats.metrics import compute_decision_latency, compute_ri, compute_rpd, summary_stats


class TestComputeRpd:
    def test_exact_match_is_zero(self) -> None:
        assert compute_rpd(55, 55) == 0.0

    def test_above_optimum_is_positive(self) -> None:
        assert compute_rpd(110, 55) == pytest.approx(100.0)

    def test_raises_on_nonpositive_optimum(self) -> None:
        with pytest.raises(ValueError):
            compute_rpd(100, 0)
        with pytest.raises(ValueError):
            compute_rpd(100, -5)


class TestComputeRi:
    def test_basic_ri_and_dispersion(self) -> None:
        nominal = 100.0
        perturbed = np.array([100.0] * 27 + [110.0, 120.0, 130.0])  # 30 replicates
        result = compute_ri(perturbed, nominal)
        assert result.n_replicates == 30
        expected_mean = (27 * 100.0 + 110.0 + 120.0 + 130.0) / 30
        assert result.ri == pytest.approx((expected_mean - nominal) / nominal)
        assert result.cvar_90 == pytest.approx(np.mean([110.0, 120.0, 130.0]))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_ri(np.array([]), 100.0)

    def test_nonpositive_nominal_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_ri(np.array([100.0]), 0.0)

    def test_partial_replicate_count_does_not_raise(self, capsys) -> None:
        result = compute_ri(np.array([100.0, 110.0]), 100.0)
        assert result.n_replicates == 2
        captured = capsys.readouterr()
        assert "expected 30" in captured.out


class TestComputeDecisionLatency:
    def test_basic(self) -> None:
        assert compute_decision_latency(1.0, 1000) == pytest.approx(1.0)

    def test_raises_on_nonpositive_num_operations(self) -> None:
        with pytest.raises(ValueError):
            compute_decision_latency(1.0, 0)


class TestSummaryStats:
    def test_basic(self) -> None:
        result = summary_stats(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert result["mean"] == pytest.approx(3.0)
        assert result["median"] == pytest.approx(3.0)
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["n"] == 5

    def test_single_value_std_is_zero(self) -> None:
        result = summary_stats(np.array([7.0]))
        assert result["std"] == 0.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            summary_stats(np.array([]))
