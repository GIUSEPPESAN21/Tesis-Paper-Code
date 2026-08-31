"""Tests for experiments/stats/rescheduling_policies.py (Fase 6, T6.3)."""

from __future__ import annotations

import pandas as pd
import pytest

from experiments.stats.rescheduling_policies import (
    compare_execution_policies,
    diagnose_fragility_source,
    tabla_politicas_reprogramacion_markdown,
)


def _df() -> pd.DataFrame:
    rows = []
    for policy, rpd, n_reopt, cost in [
        ("right_shift", 40.0, 0, 0.0), ("right_shift", 44.0, 0, 0.0),
        ("periodic", 30.0, 5, 2.0), ("periodic", 32.0, 5, 2.1),
        ("event_triggered", 38.0, 2, 0.8), ("event_triggered", 39.0, 3, 0.9),
    ]:
        rows.append({"method": "ga", "execution_policy": policy, "rpd_pct": rpd, "n_reoptimizations": n_reopt, "reoptimization_wall_clock_s": cost})
    return pd.DataFrame(rows)


class TestCompareExecutionPolicies:
    def test_one_row_per_policy(self) -> None:
        comparison = compare_execution_policies(_df(), method="ga")
        assert set(comparison["execution_policy"]) == {"right_shift", "periodic", "event_triggered"}

    def test_empty_when_method_absent(self) -> None:
        comparison = compare_execution_policies(_df(), method="cpsat")
        assert comparison.empty


class TestDiagnoseFragilitySource:
    def test_diagnoses_compromiso_con_secuencia_when_recovery_is_large(self) -> None:
        comparison = compare_execution_policies(_df(), method="ga")
        diag = diagnose_fragility_source(comparison, recovery_threshold_pp=2.0)
        # right_shift mean = 42.0, best reopt (periodic) mean = 31.0 -> recovered 11.0pp
        assert diag["diagnosis"] == "compromiso_con_secuencia"
        assert diag["best_reopt_policy"] == "periodic"
        assert diag["recovered_pp"] == pytest.approx(11.0)

    def test_diagnoses_no_explicado_when_recovery_is_small(self) -> None:
        comparison = compare_execution_policies(_df(), method="ga")
        diag = diagnose_fragility_source(comparison, recovery_threshold_pp=50.0)
        assert diag["diagnosis"] == "no_explicado_por_secuencia"

    def test_sin_datos_when_no_right_shift_row(self) -> None:
        comparison = pd.DataFrame([{"execution_policy": "periodic", "mean_rpd_pct": 10.0, "mean_n_reoptimizations": 1, "mean_reoptimization_wall_clock_s": 0.1, "n_rows": 1}])
        diag = diagnose_fragility_source(comparison)
        assert diag["diagnosis"] == "sin_datos"

    def test_sin_datos_when_empty(self) -> None:
        diag = diagnose_fragility_source(pd.DataFrame())
        assert diag["diagnosis"] == "sin_datos"


class TestTablaPoliticasReprogramacionMarkdown:
    def test_pendiente_corrida_when_no_data(self) -> None:
        md = tabla_politicas_reprogramacion_markdown({"ga": pd.DataFrame(), "cpsat": pd.DataFrame()}, {"ga": {"diagnosis": "sin_datos"}, "cpsat": {"diagnosis": "sin_datos"}})
        assert "PENDIENTE-CORRIDA" in md

    def test_real_data_renders_diagnosis(self) -> None:
        comparison = compare_execution_policies(_df(), method="ga")
        diag = diagnose_fragility_source(comparison)
        md = tabla_politicas_reprogramacion_markdown({"ga": comparison}, {"ga": diag})
        assert "compromiso_con_secuencia" in md
