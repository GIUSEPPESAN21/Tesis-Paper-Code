"""Unit tests for :mod:`experiments.cost_accounting` (Fase 5, T5.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.cost_accounting import (
    SolverCostSummary,
    TrainingCostResult,
    amortization_curve,
    cpsat_cost_summary,
    ga_cost_summary,
    inference_cost_per_instance,
    load_training_costs,
    pdr_cost_summary,
    tabla_costo_computacional_markdown,
)


class TestLoadTrainingCosts:
    def test_aggregates_mean_and_std_over_seeds(self, tmp_path: Path) -> None:
        paths = []
        for seed, wall in enumerate([2100.0, 2213.0, 2350.0]):
            p = tmp_path / f"summary_seed{seed}.json"
            p.write_text(json.dumps({"train_wall_clock_s": wall}), encoding="utf-8")
            paths.append(p)
        result = load_training_costs(paths, run_label="ft06/D1")
        assert result.n_seeds == 3
        assert result.mean_wall_clock_s == pytest.approx((2100.0 + 2213.0 + 2350.0) / 3)
        assert result.std_wall_clock_s > 0

    def test_single_seed_has_zero_std(self, tmp_path: Path) -> None:
        p = tmp_path / "summary_seed0.json"
        p.write_text(json.dumps({"train_wall_clock_s": 1000.0}), encoding="utf-8")
        result = load_training_costs([p], run_label="solo")
        assert result.n_seeds == 1
        assert result.std_wall_clock_s == 0.0

    def test_raises_on_empty_list(self) -> None:
        with pytest.raises(ValueError):
            load_training_costs([], run_label="x")

    def test_raises_on_missing_field(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"instance": "ft06"}), encoding="utf-8")
        with pytest.raises(ValueError, match="train_wall_clock_s"):
            load_training_costs([p], run_label="x")


class TestInferenceCostPerInstance:
    def test_computes_total_seconds(self) -> None:
        df = pd.DataFrame({"instance": ["ft06"], "num_operations": [36], "rl_ms_per_decision_multithread": [2.0]})
        out = inference_cost_per_instance(df)
        assert out.loc[0, "total_inference_s"] == pytest.approx(36 * 2.0 / 1000.0)

    def test_can_select_singlethread_column(self) -> None:
        df = pd.DataFrame(
            {"instance": ["ft06"], "num_operations": [36], "rl_ms_per_decision_singlethread": [3.4], "rl_ms_per_decision_multithread": [1.0]}
        )
        out = inference_cost_per_instance(df, ms_per_decision_col="rl_ms_per_decision_singlethread")
        assert out.loc[0, "total_inference_s"] == pytest.approx(36 * 3.4 / 1000.0)


class TestAmortizationCurve:
    def test_crossover_exists_when_gnn_eventually_cheaper(self) -> None:
        # training=1000s, inference negligible (0.01s/instance) -> GNN-DRL cheaper once
        # N*offline > 1000, i.e. N > 1000/offline_per_instance_s.
        inference = pd.Series([0.01] * 200)
        curve = amortization_curve(1000.0, inference, offline_per_instance_s=10.0)
        assert curve["gnn_drl_cheaper"].any()
        first_cheaper_n = curve.loc[curve["gnn_drl_cheaper"], "n_instances"].min()
        assert first_cheaper_n == pytest.approx(101, abs=1)

    def test_never_cheaper_within_range_if_training_dominates(self) -> None:
        inference = pd.Series([0.01] * 5)
        curve = amortization_curve(10_000.0, inference, offline_per_instance_s=10.0)
        assert not curve["gnn_drl_cheaper"].any()

    def test_monotonic_n_instances_column(self) -> None:
        inference = pd.Series([1.0, 2.0, 3.0])
        curve = amortization_curve(50.0, inference, offline_per_instance_s=5.0)
        assert list(curve["n_instances"]) == [1, 2, 3]


class TestSolverCostSummaries:
    def _e1(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "instance": ["ft06", "la01", "ft06", "la01", "ft06", "la01"],
                "method": ["cpsat", "cpsat", "ga", "ga", "spt", "fifo"],
                "wall_clock_s": [4.67, 60.02, 60.0, 60.0, 0.02, 0.03],
                "status": ["OPTIMAL", "FEASIBLE", "OK", "OK", "OK", "OK"],
                "is_optimal": [True, False, False, False, False, False],
                "certified_optimal": [True, False, False, False, False, False],
            }
        )

    def test_cpsat_summary_splits_certified_optimal(self) -> None:
        summary = cpsat_cost_summary(self._e1())
        assert summary.n_instances == 2
        assert summary.n_certified_optimal == 1
        assert summary.max_wall_clock_s == pytest.approx(60.02)

    def test_ga_summary(self) -> None:
        summary = ga_cost_summary(self._e1())
        assert summary.mean_wall_clock_s == pytest.approx(60.0)
        assert summary.n_certified_optimal is None

    def test_pdr_summary_pools_methods(self) -> None:
        summary = pdr_cost_summary(self._e1(), methods=("spt", "fifo"))
        assert summary.n_instances == 2
        assert summary.mean_wall_clock_s == pytest.approx((0.02 + 0.03) / 2)

    def test_raises_when_method_absent(self) -> None:
        with pytest.raises(ValueError):
            cpsat_cost_summary(self._e1()[self._e1()["method"] != "cpsat"])


class TestTablaCostoComputacionalMarkdown:
    def test_pendiente_corrida_when_all_inputs_missing(self) -> None:
        md = tabla_costo_computacional_markdown(
            training=None, inference_df=None, amortized_per_instance_s=None, cpsat=None, ga=None, pdr=None
        )
        assert md.count("PENDIENTE-CORRIDA") == 6

    def test_real_values_render_when_present(self) -> None:
        training = TrainingCostResult("ft06/D1", 2213.0, 143.0, 5, (2100, 2213, 2326, 2100, 2326))
        cpsat = SolverCostSummary("cpsat", 4.67, 60.02, 16, n_certified_optimal=7)
        md = tabla_costo_computacional_markdown(
            training=training, inference_df=None, amortized_per_instance_s=None, cpsat=cpsat, ga=None, pdr=None
        )
        assert "2213.0" in md
        assert "143.0" in md
        assert "7/16" in md
