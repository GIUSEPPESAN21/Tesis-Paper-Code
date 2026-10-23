"""Fase 0(a) del plan de mejoras del tutor: ¿el Indice de Robustez (RI) premia
mecanicamente los planes nominales malos?

Hipotesis a contrastar (AUDIT_PREFLIGHT.md Fase 0a): dado
``RI = (E[C_max^pert] - C_max^nom]) / C_max^nom``, una PDR con un plan nominal muy
malo (RPD nominal alto) tiene, por construccion de la formula, mas "colchon" para que
una reordenacion inducida por la disrupcion mejore accidentalmente el makespan --
mientras que una PDR ya casi optima (RPD nominal bajo) solo puede empeorar. Si esto es
cierto, RPD nominal y RI medio deberian estar fuertemente correlacionados en sentido
negativo (rho muy negativo) sobre las 9 PDR, en cada regimen de E4/E5/E6.

Este script NO se ejecuta como parte de este commit (Regla Cero del
documento de recomendaciones del tutor). Queda listo para correr con:

    python -m experiments.audit.ri_artifact_check \
        --e1-csv experiments/results/week24/e1_seed0_baselines.csv \
        --e4e5e6-dir experiments/results/week24/e4e5e6 \
        --out results/tablas/Tabla_auditoria_ri_artefacto.csv

Valor esperado que este script debe reproducir (AUDIT_PREFLIGHT.md Fase 0a, tal como
lo especifico el tutor): rho de Spearman entre -0.93 y -0.98, con p < 3e-4, en los seis
regimenes de averias y tiempos estocasticos (E4: A=0.95/0.90/0.80; E5:
sigma=0.10/0.25/0.50). Bajo llegadas (E6) solo hay 4 PDR evaluadas (ver
AUDIT_PREFLIGHT.md Fase 0c), por lo que un rho de Spearman sobre n=4 puntos se reporta
solo de forma descriptiva, nunca como confirmacion del mismo umbral -3.

No se ejecuta aqui: el numero real queda como [PENDIENTE-CORRIDA:
experiments/audit/ri_artifact_check.py] en AUDIT_PREFLIGHT.md hasta la primera corrida.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

#: The nine Part B6 PDRs, canonical registry names (baselines/dispatching_rules.py).
NINE_PDR_NAMES = ["spt", "lpt", "mwkr", "lwkr", "mor", "lor", "fifo", "edd", "cr"]

#: The six regimes this check's expected rho range (-0.98, -0.93) applies to --
#: breakdowns and stochastic times only. Arrivals restricts the PDR pool to four rules
#: (AUDIT_PREFLIGHT.md Fase 0c) and is reported separately, descriptively, never pooled
#: into the same correlation as the nine-PDR regimes.
SIX_REGIME_FILES = [
    "e4_breakdowns_avail0.95_with_rl.csv",
    "e4_breakdowns_avail0.90_with_rl.csv",
    "e4_breakdowns_avail0.80_with_rl.csv",
    "e5_stochastic_sigma0.1_with_rl.csv",
    "e5_stochastic_sigma0.25_with_rl.csv",
    "e5_stochastic_sigma0.5_with_rl.csv",
]

ARRIVALS_REGIME_FILES = [
    "e6_arrivals_util0.7_with_rl.csv",
    "e6_arrivals_util0.85_with_rl.csv",
    "e6_arrivals_util0.95_with_rl.csv",
]

EXPECTED_RHO_RANGE = (-0.98, -0.93)
EXPECTED_P_MAX = 3e-4


@dataclass(frozen=True)
class ArtifactCheckResult:
    """One regime's Spearman rho between nominal RPD and mean RI, over the PDR pool
    available in that regime."""

    regime: str
    n_pdrs: int
    rho: float
    p_value: float
    within_expected_range: bool
    pdr_names: tuple[str, ...]


def nominal_rpd_by_method(e1_df: pd.DataFrame, instance: str) -> dict[str, float]:
    """RPD nominal (``disruption == "none"``, seed 0) per PDR for one instance.

    Args:
        e1_df: E1's long-format CSV, already loaded (one row per instance/method/seed).
        instance: Instance name (e.g. ``"ft06"``).

    Returns:
        ``{method_name: rpd_pct}`` restricted to rows for ``instance``.

    Raises:
        ValueError: If no row for ``instance`` exists in ``e1_df``.
    """
    subset = e1_df[e1_df["instance"] == instance]
    if subset.empty:
        raise ValueError(f"no E1 rows for instance {instance!r}")
    return dict(zip(subset["method"], subset["rpd_pct"]))


def mean_ri_by_method(
    perturbed_df: pd.DataFrame, nominal_makespan_by_method: dict[str, float], instance: str
) -> dict[str, float]:
    """Mean RI per PDR for one instance under one regime.

    RI per replicate = ``(makespan_pert - makespan_nom) / makespan_nom``
    (Step 4 Thesis D6 / StatsAnalysis_TechSpec.md Sec. 5.2). Averaged over all seeds
    present for that (instance, method) in ``perturbed_df``.

    Raises:
        ValueError: If a method present in ``perturbed_df`` has no nominal makespan.
    """
    subset = perturbed_df[perturbed_df["instance"] == instance]
    out: dict[str, float] = {}
    for method, group in subset.groupby("method"):
        if method not in nominal_makespan_by_method:
            raise ValueError(f"no nominal makespan for method {method!r}, instance {instance!r}")
        nominal = nominal_makespan_by_method[method]
        ri_values = (group["makespan"].to_numpy(dtype=float) - nominal) / nominal
        out[method] = float(np.mean(ri_values))
    return out


def spearman_rpd_vs_ri(
    e1_df: pd.DataFrame, perturbed_df: pd.DataFrame, *, instances: tuple[str, ...], regime_label: str
) -> ArtifactCheckResult:
    """Spearman correlation between nominal RPD and mean RI, over every PDR present in
    ``perturbed_df``, pooled across ``instances`` (one (method, instance) pair per
    Spearman observation -- matching how AUDIT_PREFLIGHT.md Fase 0a poses the check
    across the ft06/la01 pair, not one instance in isolation).
    """
    rpd_points: list[float] = []
    ri_points: list[float] = []
    pdr_names: set[str] = set()
    for instance in instances:
        nominal_rpd = nominal_rpd_by_method(e1_df, instance)
        nominal_makespan = e1_df[e1_df["instance"] == instance].set_index("method")["makespan"].to_dict()
        ri_by_method = mean_ri_by_method(perturbed_df, nominal_makespan, instance)
        for method, ri in ri_by_method.items():
            if method not in nominal_rpd:
                continue
            rpd_points.append(nominal_rpd[method])
            ri_points.append(ri)
            pdr_names.add(method)

    rho, p_value = spearmanr(rpd_points, ri_points)
    within = EXPECTED_RHO_RANGE[0] <= rho <= EXPECTED_RHO_RANGE[1] and p_value < EXPECTED_P_MAX
    return ArtifactCheckResult(
        regime=regime_label,
        n_pdrs=len(pdr_names),
        rho=float(rho),
        p_value=float(p_value),
        within_expected_range=within,
        pdr_names=tuple(sorted(pdr_names)),
    )


def run_all_regimes(
    e1_csv: Path, e4e5e6_dir: Path, *, instances: tuple[str, ...] = ("ft06", "la01")
) -> list[ArtifactCheckResult]:
    """Run the check over the six breakdown/stochastic regimes plus the three
    arrivals regimes (reported separately -- see module docstring)."""
    e1_df = pd.read_csv(e1_csv)
    results: list[ArtifactCheckResult] = []
    for filename in SIX_REGIME_FILES + ARRIVALS_REGIME_FILES:
        path = e4e5e6_dir / filename
        if not path.exists():
            continue
        perturbed_df = pd.read_csv(path)
        regime_label = filename.replace("_with_rl.csv", "")
        results.append(
            spearman_rpd_vs_ri(e1_df, perturbed_df, instances=instances, regime_label=regime_label)
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-csv", type=Path, required=True)
    parser.add_argument("--e4e5e6-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="optional CSV output path")
    parser.add_argument("--instances", nargs="*", default=["ft06", "la01"])
    args = parser.parse_args(argv)

    results = run_all_regimes(args.e1_csv, args.e4e5e6_dir, instances=tuple(args.instances))
    for r in results:
        flag = "OK" if r.within_expected_range else "FUERA DE RANGO ESPERADO"
        print(f"{r.regime:<28} n_pdr={r.n_pdrs} rho={r.rho:+.4f} p={r.p_value:.2e}  {flag}")

    if args.out is not None:
        df = pd.DataFrame([r.__dict__ for r in results])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
