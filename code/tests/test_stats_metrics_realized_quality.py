"""Tests for Fase 8 additions to experiments/stats/metrics.py (T8.2, T8.3)."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.stats.metrics import (
    RealizedQualityResult,
    compute_realized_quality,
    compute_ri,
    scan_for_negative_ri,
)


class TestComputeRealizedQuality:
    def test_zero_when_all_replicates_match_optimum(self) -> None:
        result = compute_realized_quality(np.array([55.0] * 5), certified_optimum=55.0)
        assert result.mean_rpd_realized_pct == pytest.approx(0.0)
        assert result.certified_optimum == pytest.approx(55.0)

    def test_positive_rpd_above_optimum(self) -> None:
        result = compute_realized_quality(np.array([60.5]), certified_optimum=55.0)
        assert result.mean_rpd_realized_pct == pytest.approx(10.0)

    def test_cvar90_is_mean_of_worst_decile(self) -> None:
        # 10 replicates -> worst 1 (ceil(0.1*10)=1) is the max.
        values = np.array([55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0])
        result = compute_realized_quality(values, certified_optimum=50.0)
        expected_worst_rpd = 100.0 * (100.0 - 50.0) / 50.0
        assert result.cvar_90_rpd_realized_pct == pytest.approx(expected_worst_rpd)

    def test_raises_on_nonpositive_optimum(self) -> None:
        with pytest.raises(ValueError):
            compute_realized_quality(np.array([55.0]), certified_optimum=0.0)

    def test_raises_on_empty_input(self) -> None:
        with pytest.raises(ValueError):
            compute_realized_quality(np.array([]), certified_optimum=55.0)

    def test_comparable_across_methods_same_denominator(self) -> None:
        """The point of T8.2: two methods with different NOMINAL makespans still share
        the same denominator (the certified optimum), unlike RI."""
        method_a = compute_realized_quality(np.array([80.0]), certified_optimum=55.0)
        method_b = compute_realized_quality(np.array([80.0]), certified_optimum=55.0)
        assert method_a.mean_rpd_realized_pct == method_b.mean_rpd_realized_pct


class TestRiNegativeWarning:
    def test_ri_negative_flag_set_when_perturbed_below_nominal(self) -> None:
        with pytest.warns(RuntimeWarning, match="NOT interpretable as robustness"):
            result = compute_ri(np.array([80.0] * 30), nominal_makespan=171.0)
        assert result.ri_negative is True
        assert result.ri < 0.0

    def test_no_warning_when_ri_nonnegative(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = compute_ri(np.array([200.0] * 30), nominal_makespan=171.0)
        assert result.ri_negative is False


class TestScanForNegativeRi:
    def test_returns_only_negative_records(self) -> None:
        records = [
            {"instance": "ft06", "method": "cr", "regime": "E4_breakdowns_A0.90", "ri": -0.45},
            {"instance": "ft06", "method": "fifo", "regime": "E4_breakdowns_A0.90", "ri": 0.05},
            {"instance": "la01", "method": "lwkr", "regime": "E6_arrivals_u0.70", "ri": 0.39},
        ]
        negative = scan_for_negative_ri(records)
        assert len(negative) == 1
        assert negative[0]["method"] == "cr"

    def test_empty_when_none_negative(self) -> None:
        records = [{"ri": 0.1}, {"ri": 0.2}]
        assert scan_for_negative_ri(records) == []
