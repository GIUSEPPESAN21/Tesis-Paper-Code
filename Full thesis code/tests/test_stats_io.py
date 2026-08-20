"""Unit tests for :mod:`experiments.stats.io`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.stats.io import (
    DataContractError,
    add_derived_columns,
    load_results,
    validate_rl_rollout_convention,
    validate_schema,
)

REQUIRED = [
    "instance", "family", "num_jobs", "num_machines", "method", "disruption", "seed",
    "makespan", "known_optimum", "rpd_pct", "wall_clock_s", "feasible", "status",
    "is_optimal", "violations_count", "timestamp",
]


def _base_row(**overrides) -> dict:
    row = {
        "instance": "ft06", "family": "fisher-thompson", "num_jobs": 6, "num_machines": 6,
        "method": "spt", "disruption": "none", "seed": 0, "makespan": 70.0,
        "known_optimum": 55, "rpd_pct": 27.27, "wall_clock_s": 0.01, "feasible": True,
        "status": "OK", "is_optimal": False, "violations_count": 0, "timestamp": "2026-07-31T00:00:00+00:00",
    }
    row.update(overrides)
    return row


class TestValidateSchema:
    def test_passes_on_full_schema(self) -> None:
        df = pd.DataFrame([_base_row()])
        validate_schema(df, source="test")  # must not raise

    def test_raises_on_missing_column(self) -> None:
        df = pd.DataFrame([_base_row()]).drop(columns=["rpd_pct"])
        with pytest.raises(DataContractError, match="rpd_pct"):
            validate_schema(df, source="test")


class TestValidateRlRolloutConvention:
    def test_passes_on_valid_convention(self) -> None:
        rows = [_base_row(method="rl", seed=s) for s in (0, 1, 2, 3, 4)]
        df = pd.DataFrame(rows)
        validate_rl_rollout_convention(df, source="test")  # must not raise

    def test_passes_when_only_deterministic_present(self) -> None:
        df = pd.DataFrame([_base_row(method="rl", seed=0)])
        validate_rl_rollout_convention(df, source="test")

    def test_raises_on_missing_deterministic_seed(self) -> None:
        df = pd.DataFrame([_base_row(method="rl", seed=1)])
        with pytest.raises(DataContractError, match="expected exactly 1"):
            validate_rl_rollout_convention(df, source="test")

    def test_raises_on_duplicate_deterministic_seed(self) -> None:
        df = pd.DataFrame([_base_row(method="rl", seed=0), _base_row(method="rl", seed=0)])
        with pytest.raises(DataContractError, match="duplicate rl seeds"):
            validate_rl_rollout_convention(df, source="test")

    def test_passes_disruption_regime_block_with_30_replicate_seeds(self) -> None:
        rows = [_base_row(method="rl", disruption="breakdowns", seed=s) for s in range(30)]
        df = pd.DataFrame(rows)
        validate_rl_rollout_convention(df, source="test")  # must not raise

    def test_raises_on_duplicate_seed_in_disruption_regime_block(self) -> None:
        df = pd.DataFrame(
            [
                _base_row(method="rl", disruption="breakdowns", seed=0),
                _base_row(method="rl", disruption="breakdowns", seed=0),
            ]
        )
        with pytest.raises(DataContractError, match="duplicate rl seeds"):
            validate_rl_rollout_convention(df, source="test")

    def test_raises_on_seed_outside_convention(self) -> None:
        df = pd.DataFrame([_base_row(method="rl", seed=0), _base_row(method="rl", seed=7)])
        with pytest.raises(DataContractError, match="outside"):
            validate_rl_rollout_convention(df, source="test")

    def test_raises_on_duplicate_stochastic_seed(self) -> None:
        df = pd.DataFrame(
            [_base_row(method="rl", seed=0), _base_row(method="rl", seed=1), _base_row(method="rl", seed=1)]
        )
        with pytest.raises(DataContractError, match="duplicate"):
            validate_rl_rollout_convention(df, source="test")

    def test_ignores_non_rl_rows(self) -> None:
        df = pd.DataFrame([_base_row(method="spt", seed=5)])
        validate_rl_rollout_convention(df, source="test")  # must not raise


class TestAddDerivedColumns:
    def test_jssp_num_operations_is_jobs_times_machines(self) -> None:
        df = pd.DataFrame([_base_row(num_jobs=6, num_machines=6)])
        out = add_derived_columns(df)
        assert out.loc[0, "num_operations"] == 36

    def test_brandimarte_uses_hardcoded_table(self) -> None:
        df = pd.DataFrame([_base_row(instance="mk01", family="brandimarte", num_jobs=10, num_machines=6)])
        out = add_derived_columns(df)
        assert out.loc[0, "num_operations"] == 60  # not 10*6=60 coincidentally same; check mk03 too

        df2 = pd.DataFrame([_base_row(instance="mk03", family="brandimarte", num_jobs=15, num_machines=8)])
        out2 = add_derived_columns(df2)
        assert out2.loc[0, "num_operations"] == 150  # NOT 15*8=120

    def test_synthetic_fjsp_num_operations_is_nan(self) -> None:
        df = pd.DataFrame([_base_row(instance="ft06_fjsp", family="fjsp-synthetic", num_jobs=6, num_machines=6)])
        out = add_derived_columns(df)
        assert np.isnan(out.loc[0, "num_operations"])

    def test_ms_per_decision(self) -> None:
        df = pd.DataFrame([_base_row(num_jobs=6, num_machines=6, wall_clock_s=0.36)])
        out = add_derived_columns(df)
        assert out.loc[0, "ms_per_decision"] == pytest.approx(10.0)

    def test_rpd_or_makespan_uses_rpd_when_present(self) -> None:
        df = pd.DataFrame([_base_row(rpd_pct=12.5)])
        out = add_derived_columns(df)
        assert out.loc[0, "rpd_or_makespan"] == pytest.approx(12.5)
        assert out.loc[0, "uses_raw_makespan"] == False  # noqa: E712

    def test_rpd_or_makespan_falls_back_to_makespan(self) -> None:
        df = pd.DataFrame([_base_row(rpd_pct=float("nan"), makespan=321.0)])
        out = add_derived_columns(df)
        assert out.loc[0, "rpd_or_makespan"] == pytest.approx(321.0)
        assert out.loc[0, "uses_raw_makespan"] == True  # noqa: E712

    def test_does_not_mutate_input(self) -> None:
        df = pd.DataFrame([_base_row()])
        add_derived_columns(df)
        assert "num_operations" not in df.columns

    def test_certified_optimal_true_only_for_optimal_status_and_zero_gap(self) -> None:
        df = pd.DataFrame(
            [
                _base_row(method="cpsat", status="OPTIMAL", is_optimal=True, mip_gap=0.0),
                _base_row(method="cpsat", status="FEASIBLE", is_optimal=False, mip_gap=0.12),
                _base_row(method="spt", status="OK", is_optimal=False),
            ]
        )
        out = add_derived_columns(df)
        assert list(out["certified_optimal"]) == [True, False, False]

    def test_missing_mip_gap_column_backfilled_as_nan_not_a_keyerror(self) -> None:
        """Pre-existing E1-E8 CSVs, written before mip_gap was added to
        run_evaluation.py's CSV_FIELDNAMES, must keep loading unchanged (Regla 5)."""
        df = pd.DataFrame([_base_row()]).drop(columns=["mip_gap"], errors="ignore")
        assert "mip_gap" not in df.columns
        out = add_derived_columns(df)
        assert np.isnan(out.loc[0, "mip_gap"])
        assert out.loc[0, "certified_optimal"] == False  # noqa: E712


class TestLoadResults:
    def test_loads_and_tags_experiment_id(self, tmp_path: Path) -> None:
        path = tmp_path / "e1.csv"
        pd.DataFrame([_base_row()]).to_csv(path, index=False)
        combined, provenance = load_results({"E1": path})
        assert list(combined["experiment_id"]) == ["E1"]
        assert provenance[0].experiment_id == "E1"
        assert provenance[0].n_rows_loaded == 1

    def test_concatenates_multi_file_experiment(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.csv"
        p2 = tmp_path / "b.csv"
        pd.DataFrame([_base_row(seed=0)]).to_csv(p1, index=False)
        pd.DataFrame([_base_row(seed=1)]).to_csv(p2, index=False)
        combined, _ = load_results({"E1": [p1, p2]})
        assert len(combined) == 2

    def test_drops_infeasible_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "e1.csv"
        pd.DataFrame([_base_row(feasible=True), _base_row(feasible=False)]).to_csv(path, index=False)
        combined, provenance = load_results({"E1": path})
        assert len(combined) == 1
        assert provenance[0].n_rows_infeasible_dropped == 1

    def test_missing_path_raises_by_default(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_results({"E1": tmp_path / "does_not_exist.csv"})

    def test_missing_path_skipped_when_requested(self, tmp_path: Path) -> None:
        combined, provenance = load_results({"E1": tmp_path / "does_not_exist.csv"}, skip_missing=True)
        assert combined.empty
        assert provenance == []

    def test_raises_on_schema_violation(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        pd.DataFrame([{"instance": "ft06"}]).to_csv(path, index=False)
        with pytest.raises(DataContractError):
            load_results({"E1": path})

    def test_empty_experiment_dict_returns_empty_frame(self) -> None:
        combined, provenance = load_results({})
        assert combined.empty
        assert provenance == []
