"""Tests for experiments/stats/seeds.py (Fase 2, T2.3)."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.stats.seeds import aggregate_over_seeds, format_mean_ci


class TestAggregateOverSeeds:
    def test_single_seed_degenerates_gracefully(self) -> None:
        agg = aggregate_over_seeds([42.0])
        assert agg.mean == pytest.approx(42.0)
        assert agg.std == 0.0
        assert agg.ci95_low == agg.ci95_high == pytest.approx(42.0)
        assert agg.n_seeds == 1

    def test_raises_on_empty(self) -> None:
        with pytest.raises(ValueError):
            aggregate_over_seeds([])

    def test_ci_widens_with_variance(self) -> None:
        tight = aggregate_over_seeds([100.0, 101.0, 99.0, 100.5, 99.5])
        wide = aggregate_over_seeds([50.0, 150.0, 60.0, 140.0, 100.0])
        tight_width = tight.ci95_high - tight.ci95_low
        wide_width = wide.ci95_high - wide.ci95_low
        assert wide_width > tight_width

    def test_ci_contains_mean(self) -> None:
        agg = aggregate_over_seeds([10.0, 12.0, 11.0, 13.0, 9.0])
        assert agg.ci95_low <= agg.mean <= agg.ci95_high

    def test_range_matches_min_max(self) -> None:
        agg = aggregate_over_seeds([5.0, 1.0, 9.0, 3.0])
        assert agg.range_low == pytest.approx(1.0)
        assert agg.range_high == pytest.approx(9.0)

    def test_known_t_interval_two_seeds(self) -> None:
        # n=2: t_0.975,df=1 = 12.706 (a well-known critical value, easy to sanity check).
        agg = aggregate_over_seeds([10.0, 12.0])
        std = np.std([10.0, 12.0], ddof=1)
        sem = std / np.sqrt(2)
        expected_margin = 12.706 * sem
        assert agg.ci95_high - agg.mean == pytest.approx(expected_margin, rel=1e-3)


class TestFormatMeanCi:
    def test_single_seed_string(self) -> None:
        agg = aggregate_over_seeds([5.0])
        assert "n=1" in format_mean_ci(agg)

    def test_multi_seed_string_contains_ci(self) -> None:
        agg = aggregate_over_seeds([1.0, 2.0, 3.0])
        text = format_mean_ci(agg)
        assert "IC95" in text
        assert "n=3" in text
