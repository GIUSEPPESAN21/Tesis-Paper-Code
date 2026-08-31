"""Fase 5 (Recomendacion 8 del tutor): costo computacional homogeneo, T5.1.

Today the thesis quotes training cost, inference cost, and solver cost as separate,
differently-scoped numbers scattered across chapters, with no single table that puts
them on a comparable footing. This module builds ``results/tablas/Tabla_costo_computacional``
from data this pipeline already produces (``summary_*.json`` from ``train_dynamic.py``/
``train_hetero.py``, ``experiments/results/week24/e8_cpu_timing.csv``, and E1's own
``wall_clock_s`` column for cpsat/ga/PDR rows) -- no new experiment, only a new
aggregation layer over already-collected timing data.

Rows this table separates (never fused into one "RL cost" number, per T5.1):

1. GNN-DRL, training cost (one-time): ``train_wall_clock_s`` from every
   ``summary_*.json`` for a given (instance, kappa, stage), mean +/- std over seeds.
2. GNN-DRL, inference cost per instance: ms/decision x num_operations, from
   ``e8_cpu_timing.csv`` (T5.3 widened this to the full 43-instance set).
3. GNN-DRL, amortized cost: training cost / N instances solved so far, plus the full
   amortization curve as a function of N (:func:`amortization_curve`).
4. CP-SAT: wall-clock per instance, OPTIMAL vs. FEASIBLE-time-limited split
   (T7.1's ``certified_optimal``/``mip_gap`` distinction feeds this split directly).
5. GA: wall-clock per instance under its fixed time budget.
6. PDR: construction (rollout) time per instance.

Reference values this table should reproduce once run (tutor's own measurement, T5.1):
training (ft06) 2213s +/- 143s (5 seeds); inference 1.0-3.4 ms/decision (single-thread);
amortized ~51s/instance over N=43; CP-SAT 4.67s mean, 60.02s max; GA saturates the 60s
budget every time; PDR 0.023s.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class TrainingCostResult:
    """One-time training cost, aggregated over seeds (T2's multi-seed summaries feed
    this directly once they exist; a single-seed run degenerates to std=0.0, n=1)."""

    run_label: str
    mean_wall_clock_s: float
    std_wall_clock_s: float
    n_seeds: int
    per_seed_wall_clock_s: tuple[float, ...]


def load_training_costs(summary_json_paths: list[Path], *, run_label: str) -> TrainingCostResult:
    """Aggregate ``train_wall_clock_s`` across one run's ``summary_*_seed{N}.json`` files.

    Args:
        summary_json_paths: Every seed's summary JSON for one (instance, kappa, stage)
            configuration -- e.g. ``summary_dynamic_kappa0.5_D1_ft06_seed{0..4}.json``
            once T2.1's multi-seed sweep has run.
        run_label: Free-text label for the table row (e.g. ``"GNN-DRL train (ft06, D1)"``).

    Raises:
        ValueError: If ``summary_json_paths`` is empty, or any file lacks
            ``train_wall_clock_s``.
    """
    if not summary_json_paths:
        raise ValueError("no summary JSON paths given")
    values: list[float] = []
    for path in summary_json_paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "train_wall_clock_s" not in data:
            raise ValueError(f"{path}: missing 'train_wall_clock_s'")
        values.append(float(data["train_wall_clock_s"]))
    arr = np.array(values, dtype=float)
    return TrainingCostResult(
        run_label=run_label,
        mean_wall_clock_s=float(arr.mean()),
        std_wall_clock_s=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        n_seeds=len(arr),
        per_seed_wall_clock_s=tuple(values),
    )


def inference_cost_per_instance(
    e8_timing_df: pd.DataFrame, *, ms_per_decision_col: str = "rl_ms_per_decision_multithread"
) -> pd.DataFrame:
    """Total inference wall-clock per instance: ms/decision x num_operations.

    Args:
        e8_timing_df: ``experiments/results/week24/e8_cpu_timing.csv``, loaded.
        ms_per_decision_col: Which column to use -- ``rl_ms_per_decision_multithread``
            (default, matches the thread count every other E1/E8 timing number was
            measured under) or ``rl_ms_per_decision_singlethread`` (descriptive only,
            see ``e8_cpu_timing.py``'s module docstring for which manuscript claims
            use which column).

    Returns:
        ``e8_timing_df`` with an added ``total_inference_s`` column.
    """
    out = e8_timing_df.copy()
    out["total_inference_s"] = out[ms_per_decision_col] * out["num_operations"] / 1000.0
    return out


def amortization_curve(
    training_cost_s: float, per_instance_inference_s: pd.Series, *, offline_per_instance_s: float
) -> pd.DataFrame:
    """Cumulative cost of the GNN-DRL approach (train once + infer N times) vs. an
    always-reoptimize-from-scratch offline approach (e.g. CP-SAT/GA per instance),
    as a function of N -- Fig_costo_amortizado's underlying data (T5.2).

    Args:
        training_cost_s: One-time training wall-clock (mean over seeds).
        per_instance_inference_s: Per-instance RL inference cost (from
            :func:`inference_cost_per_instance`'s ``total_inference_s``), used in the
            order given -- callers should sort ascending or by instance-size for a
            monotone-looking curve, but this function does not resort.
        offline_per_instance_s: Mean per-instance cost of the offline comparator
            (e.g. CP-SAT's mean ``wall_clock_s`` from E1) -- assumed constant per
            instance for the crossover computation (a simplification stated here
            explicitly, not hidden: real CP-SAT time varies by instance size).

    Returns:
        One row per N in ``1..len(per_instance_inference_s)``: ``n_instances``,
        ``gnn_drl_cumulative_s`` (``training_cost_s + N * mean_inference_s_so_far``),
        ``offline_cumulative_s`` (``N * offline_per_instance_s``), and
        ``gnn_drl_cheaper`` (bool) -- the crossover point is the smallest N where this
        flips ``True`` and stays ``True`` for the rest of the curve (not guaranteed if
        per-instance inference cost is non-monotone; report the raw column, do not
        silently assume a single crossover).
    """
    inference_values = per_instance_inference_s.to_numpy(dtype=float)
    n = len(inference_values)
    cumulative_inference = np.cumsum(inference_values)
    ns = np.arange(1, n + 1)
    gnn_cumulative = training_cost_s + cumulative_inference
    offline_cumulative = ns * offline_per_instance_s
    return pd.DataFrame(
        {
            "n_instances": ns,
            "gnn_drl_cumulative_s": gnn_cumulative,
            "offline_cumulative_s": offline_cumulative,
            "gnn_drl_cheaper": gnn_cumulative < offline_cumulative,
        }
    )


@dataclass(frozen=True)
class SolverCostSummary:
    """Per-instance wall-clock summary for one offline method (cpsat/ga) or PDR."""

    method: str
    mean_wall_clock_s: float
    max_wall_clock_s: float
    n_instances: int
    n_certified_optimal: int | None = None  # cpsat only; None for ga/PDR


def cpsat_cost_summary(e1_df: pd.DataFrame) -> SolverCostSummary:
    """CP-SAT wall-clock per instance, split by certified-optimal status (T7.1's
    ``certified_optimal``/``mip_gap`` columns feed this split -- a FEASIBLE,
    time-limited row is not the same claim as an OPTIMAL one).

    Raises:
        ValueError: If no ``method == "cpsat"`` rows are present.
    """
    subset = e1_df[e1_df["method"] == "cpsat"]
    if subset.empty:
        raise ValueError("no cpsat rows in the given frame")
    is_optimal_col = "certified_optimal" if "certified_optimal" in subset.columns else "is_optimal"
    n_optimal = int(subset[is_optimal_col].sum())
    return SolverCostSummary(
        method="cpsat",
        mean_wall_clock_s=float(subset["wall_clock_s"].mean()),
        max_wall_clock_s=float(subset["wall_clock_s"].max()),
        n_instances=int(subset["instance"].nunique()),
        n_certified_optimal=n_optimal,
    )


def ga_cost_summary(e1_df: pd.DataFrame) -> SolverCostSummary:
    """GA wall-clock per instance under its fixed time budget.

    Raises:
        ValueError: If no ``method == "ga"`` rows are present.
    """
    subset = e1_df[e1_df["method"] == "ga"]
    if subset.empty:
        raise ValueError("no ga rows in the given frame")
    return SolverCostSummary(
        method="ga",
        mean_wall_clock_s=float(subset["wall_clock_s"].mean()),
        max_wall_clock_s=float(subset["wall_clock_s"].max()),
        n_instances=int(subset["instance"].nunique()),
    )


def pdr_cost_summary(e1_df: pd.DataFrame, *, methods: tuple[str, ...]) -> SolverCostSummary:
    """PDR construction (rollout) time per instance, pooled over ``methods``.

    Raises:
        ValueError: If no matching rows are present.
    """
    subset = e1_df[e1_df["method"].isin(methods)]
    if subset.empty:
        raise ValueError(f"no PDR rows for methods={methods!r} in the given frame")
    return SolverCostSummary(
        method="pdr",
        mean_wall_clock_s=float(subset["wall_clock_s"].mean()),
        max_wall_clock_s=float(subset["wall_clock_s"].max()),
        n_instances=int(subset["instance"].nunique()),
    )


def tabla_costo_computacional_markdown(
    *,
    training: TrainingCostResult | None,
    inference_df: pd.DataFrame | None,
    amortized_per_instance_s: float | None,
    cpsat: SolverCostSummary | None,
    ga: SolverCostSummary | None,
    pdr: SolverCostSummary | None,
) -> str:
    """T5.1's table: six rows, always separate, ``[PENDIENTE-CORRIDA]`` for any input
    not yet available (e.g. before ``train_dynamic.py``/``e8_cpu_timing.py`` have run).
    """
    lines = ["| Categoria | Metrica | Valor |", "|---|---|---|"]

    if training is not None:
        lines.append(
            f"| GNN-DRL | costo de entrenamiento (unico, {training.run_label}) | "
            f"{training.mean_wall_clock_s:.1f} s +/- {training.std_wall_clock_s:.1f} s (N={training.n_seeds}) |"
        )
    else:
        lines.append("| GNN-DRL | costo de entrenamiento (unico) | [PENDIENTE-CORRIDA: train_dynamic.py / train_hetero.py] |")

    if inference_df is not None and not inference_df.empty:
        mean_ms = float(inference_df["rl_ms_per_decision_multithread"].mean())
        mean_total_s = float(inference_df["total_inference_s"].mean())
        lines.append(
            f"| GNN-DRL | costo de inferencia por instancia (media sobre {len(inference_df)} instancias) | "
            f"{mean_ms:.3f} ms/decision, {mean_total_s:.3f} s/instancia |"
        )
    else:
        lines.append("| GNN-DRL | costo de inferencia por instancia | [PENDIENTE-CORRIDA: experiments/results/week24/e8_cpu_timing.py] |")

    if amortized_per_instance_s is not None:
        lines.append(f"| GNN-DRL | costo amortizado (entrenamiento / N instancias) | {amortized_per_instance_s:.2f} s/instancia |")
    else:
        lines.append("| GNN-DRL | costo amortizado | [PENDIENTE-CORRIDA: requiere entrenamiento + e8_cpu_timing.py] |")

    for label, summary in (("CP-SAT", cpsat), ("GA", ga), ("PDR", pdr)):
        if summary is not None:
            extra = f", {summary.n_certified_optimal}/{summary.n_instances} optimo certificado" if summary.n_certified_optimal is not None else ""
            lines.append(
                f"| {label} | tiempo de optimizacion/construccion por instancia | "
                f"media {summary.mean_wall_clock_s:.3f} s, max {summary.max_wall_clock_s:.3f} s "
                f"(N={summary.n_instances}{extra}) |"
            )
        else:
            lines.append(f"| {label} | tiempo de optimizacion/construccion por instancia | [PENDIENTE-CORRIDA: run_evaluation.py] |")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI: build Tabla_costo_computacional from whatever inputs are already on disk,
    marking any missing input as [PENDIENTE-CORRIDA] rather than failing outright
    (mirrors ``experiments/stats/cli.py --skip-missing``'s incremental-run philosophy).

        python -m experiments.cost_accounting \\
            --e1-csv experiments/results/week24/e1_seed0_baselines.csv \\
            --e8-timing-csv experiments/results/week24/e8_cpu_timing.csv \\
            --summary-glob "experiments/results/dynamic/summary_dynamic_kappa0.0_D1_ft06_seed*.json" \\
            --out experiments/results/stats/tables/costo_computacional.md
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-csv", type=Path, default=None)
    parser.add_argument("--e8-timing-csv", type=Path, default=None)
    parser.add_argument("--summary-glob", default=None, help="glob for one run's summary_*_seed*.json files")
    parser.add_argument("--pdr-methods", nargs="*", default=["spt", "lpt", "mwkr", "lwkr", "mor", "lor", "fifo", "edd", "cr"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    e1_df = pd.read_csv(args.e1_csv) if args.e1_csv and args.e1_csv.exists() else None
    e8_df = pd.read_csv(args.e8_timing_csv) if args.e8_timing_csv and args.e8_timing_csv.exists() else None

    training = None
    if args.summary_glob:
        paths = [Path(p) for p in glob.glob(args.summary_glob)]
        if paths:
            training = load_training_costs(paths, run_label=args.summary_glob)

    inference_df = inference_cost_per_instance(e8_df) if e8_df is not None else None

    amortized = None
    if training is not None and inference_df is not None and not inference_df.empty:
        amortized = training.mean_wall_clock_s / len(inference_df) + float(inference_df["total_inference_s"].mean())

    cpsat = ga = pdr = None
    if e1_df is not None:
        try:
            cpsat = cpsat_cost_summary(e1_df)
        except ValueError:
            pass
        try:
            ga = ga_cost_summary(e1_df)
        except ValueError:
            pass
        try:
            pdr = pdr_cost_summary(e1_df, methods=tuple(args.pdr_methods))
        except ValueError:
            pass

    markdown = tabla_costo_computacional_markdown(
        training=training, inference_df=inference_df, amortized_per_instance_s=amortized,
        cpsat=cpsat, ga=ga, pdr=pdr,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
