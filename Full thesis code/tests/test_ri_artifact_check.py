"""Tests for experiments/audit/ri_artifact_check.py (Fase 0a, AUDIT_PREFLIGHT.md).

Synthetic data only -- these do not read any real CSV under experiments/results/, so
they are safe to run without depending on the (large, environment-specific) week24
result files being present.
"""

from __future__ import annotations

import pandas as pd
import pytest

from experiments.audit.ri_artifact_check import (
    mean_ri_by_method,
    nominal_rpd_by_method,
    spearman_rpd_vs_ri,
)


def _e1_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instance": ["ft06", "ft06", "ft06"],
            "method": ["fifo", "cr", "spt"],
            "makespan": [67.0, 171.0, 70.0],
            "rpd_pct": [21.82, 210.9, 27.27],
        }
    )


class TestNominalRpdByMethod:
    def test_returns_mapping_for_instance(self):
        result = nominal_rpd_by_method(_e1_frame(), "ft06")
        assert result == {"fifo": 21.82, "cr": 210.9, "spt": 27.27}

    def test_raises_on_missing_instance(self):
        with pytest.raises(ValueError):
            nominal_rpd_by_method(_e1_frame(), "la99")


class TestMeanRiByMethod:
    def test_negative_ri_when_perturbed_below_nominal(self):
        perturbed = pd.DataFrame(
            {
                "instance": ["ft06"] * 4,
                "method": ["cr"] * 4,
                "makespan": [80.0, 90.0, 100.0, 110.0],
            }
        )
        nominal = {"cr": 171.0}
        result = mean_ri_by_method(perturbed, nominal, "ft06")
        expected = ((80.0 + 90.0 + 100.0 + 110.0) / 4.0 - 171.0) / 171.0
        assert result["cr"] == pytest.approx(expected)
        assert result["cr"] < 0

    def test_positive_ri_when_perturbed_above_nominal(self):
        perturbed = pd.DataFrame(
            {"instance": ["ft06", "ft06"], "method": ["fifo", "fifo"], "makespan": [75.0, 85.0]}
        )
        result = mean_ri_by_method(perturbed, {"fifo": 67.0}, "ft06")
        assert result["fifo"] > 0

    def test_raises_when_method_has_no_nominal(self):
        perturbed = pd.DataFrame({"instance": ["ft06"], "method": ["mor"], "makespan": [90.0]})
        with pytest.raises(ValueError):
            mean_ri_by_method(perturbed, {"fifo": 67.0}, "ft06")


class TestSpearmanRpdVsRi:
    def test_perfect_negative_correlation_detected(self):
        """Synthetic case engineered so RI is a perfectly decreasing function of RPD --
        the artifact-check hypothesis in its cleanest form."""
        e1 = pd.DataFrame(
            {
                "instance": ["ft06"] * 4,
                "method": ["a", "b", "c", "d"],
                "makespan": [100.0, 150.0, 200.0, 250.0],
                "rpd_pct": [10.0, 50.0, 90.0, 130.0],
            }
        )
        # RI decreases monotonically as nominal RPD increases (the artifact pattern).
        perturbed = pd.DataFrame(
            {
                "instance": ["ft06"] * 4,
                "method": ["a", "b", "c", "d"],
                "makespan": [130.0, 165.0, 180.0, 175.0],
            }
        )
        result = spearman_rpd_vs_ri(e1, perturbed, instances=("ft06",), regime_label="synthetic")
        assert result.n_pdrs == 4
        assert result.rho == pytest.approx(-1.0)
        assert result.regime == "synthetic"

    def test_no_correlation_not_flagged_within_expected_range(self):
        e1 = pd.DataFrame(
            {
                "instance": ["ft06"] * 4,
                "method": ["a", "b", "c", "d"],
                "makespan": [100.0, 100.0, 100.0, 100.0],
                "rpd_pct": [10.0, 50.0, 90.0, 130.0],
            }
        )
        perturbed = pd.DataFrame(
            {
                "instance": ["ft06"] * 4,
                "method": ["a", "b", "c", "d"],
                "makespan": [100.0, 100.0, 100.0, 100.0],
            }
        )
        result = spearman_rpd_vs_ri(e1, perturbed, instances=("ft06",), regime_label="flat")
        assert not result.within_expected_range
