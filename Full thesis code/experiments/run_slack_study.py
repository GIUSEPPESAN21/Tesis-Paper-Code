"""Fase 1 (Recomendacion 2 del tutor), T1.2: genera los programas nominales de CP-SAT,
GA y las 9 PDR sobre el conjunto JSSP completo y emite
``results/datos_crudos/slack/slack_metrics.csv`` con
``instance, method, makespan, rpd_pct`` + las seis metricas de
``experiments.slack_analysis.schedule_slack_metrics``.

No se ejecuta en esta sesion (Regla Cero). Uso previsto:

    python -m experiments.run_slack_study \\
        --instances ft06 ft10 ft20 la01 ... la40 \\
        --methods cpsat ga pdr \\
        --seed 0 \\
        --out results/datos_crudos/slack/slack_metrics.csv

Presupuesto de tiempo: ver RUN_PLAN.md Fase 1 -- dominado por CP-SAT (hasta 60s/instancia
en las mayores) y GA (tiempo fijo por instancia), las 9 PDR son casi instantaneas.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.cpsat import compute_mip_gap  # noqa: E402
from baselines.dispatching_rules import RULES  # noqa: E402
from baselines.ga import GAConfig  # noqa: E402
from baselines.interface import BaselineResult, CPSATBaseline, GABaseline, PDRBaseline  # noqa: E402
from envs.loader import known_optimum, load_by_name  # noqa: E402
from experiments.slack_analysis import schedule_slack_metrics  # noqa: E402

NINE_PDR_NAMES = ["spt", "lpt", "mwkr", "lwkr", "mor", "lor", "fifo", "edd", "cr"]

#: Full bundled JSSP set: ft06/ft10/ft20 (Fisher-Thompson) + la01-la40 (Lawrence),
#: matching PreRegistration_FINAL.md Sec 1's frozen JSSP test universe.
DEFAULT_INSTANCES = ["ft06", "ft10", "ft20"] + [f"la{n:02d}" for n in range(1, 41)]

CSV_FIELDNAMES = [
    "instance", "method", "makespan", "rpd_pct",
    "total_slack", "mean_slack", "slack_ratio", "critical_path_length",
    "critical_op_fraction", "machine_idle_fraction",
    "n_operations", "wall_clock_s", "feasible", "status", "mip_gap",
]


def _rpd(makespan: float, optimum: int | None) -> float:
    if optimum is None or optimum <= 0:
        return float("nan")
    return round(100.0 * (makespan - optimum) / optimum, 3)


def run_slack_study(
    instance_names: list[str],
    method_names: list[str],
    *,
    seed: int,
    time_limit_s: float,
    num_workers: int,
) -> list[dict]:
    """Solve every ``(instance, method)`` pair once and compute its slack metrics.

    Args:
        instance_names: Bundled JSSP instance names.
        method_names: ``"cpsat"``, ``"ga"``, and/or ``"pdr"`` (expands to all nine
            PDRs) -- or an explicit PDR name.
        seed: GA/PDR seed.
        time_limit_s: CP-SAT/GA wall-clock budget.
        num_workers: CP-SAT search worker threads.

    Returns:
        One row (dict) per ``(instance, method)`` pair that solved without raising --
        a per-cell failure is skipped with a printed warning, matching
        ``run_evaluation.py``'s catch-and-continue hardening (never discard an
        already-computed sweep for one bad cell).
    """
    expanded_methods: list[str] = []
    for m in method_names:
        if m == "pdr":
            expanded_methods.extend(NINE_PDR_NAMES)
        else:
            expanded_methods.append(m)

    rows: list[dict] = []
    for instance_name in instance_names:
        instance = load_by_name(instance_name)
        try:
            optimum = known_optimum(instance_name)
        except KeyError:
            optimum = None

        for method in expanded_methods:
            try:
                if method == "cpsat":
                    baseline = CPSATBaseline(num_workers=num_workers)
                    result = baseline.solve(instance, seed=seed, time_limit_s=time_limit_s)
                elif method == "ga":
                    baseline = GABaseline(GAConfig())
                    result = baseline.solve(instance, seed=seed, time_limit_s=time_limit_s)
                elif method in RULES:
                    baseline = PDRBaseline(method)
                    result = baseline.solve(instance, seed=seed, time_limit_s=time_limit_s)
                else:
                    raise ValueError(f"unknown method {method!r}")

                row = {
                    "instance": instance_name,
                    "method": method,
                    "makespan": result.makespan,
                    "rpd_pct": _rpd(result.makespan, optimum),
                }
                metrics = schedule_slack_metrics(result.start_times, instance, check_feasible=False)
                row.update(
                    {
                        "total_slack": metrics.total_slack,
                        "mean_slack": metrics.mean_slack,
                        "slack_ratio": metrics.slack_ratio,
                        "critical_path_length": metrics.critical_path_length,
                        "critical_op_fraction": metrics.critical_op_fraction,
                        "machine_idle_fraction": metrics.machine_idle_fraction,
                        "n_operations": metrics.n_operations,
                        "wall_clock_s": round(result.wall_clock_time, 4),
                        "feasible": result.feasible,
                        "status": result.status,
                        "mip_gap": result.extra.get("mip_gap", float("nan")),
                    }
                )
                rows.append(row)
                print(f"{instance_name:>8} {method:>8} Cmax={result.makespan:<8.1f} slack_ratio={row['slack_ratio']:.4f}", flush=True)
            except Exception as exc:  # noqa: BLE001 -- catch-and-continue, see docstring
                print(f"FAILED  {instance_name:>8} {method:>8} {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                continue

    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        raise ValueError("no rows to write")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="*", default=DEFAULT_INSTANCES)
    parser.add_argument("--methods", nargs="*", default=["cpsat", "ga", "pdr"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=REPO_ROOT.parent / "results" / "datos_crudos" / "slack" / "slack_metrics.csv")
    args = parser.parse_args(argv)

    rows = run_slack_study(
        args.instances, args.methods, seed=args.seed, time_limit_s=args.time_limit, num_workers=args.num_workers
    )
    if not rows:
        raise SystemExit("no rows produced; every (instance, method) cell failed")
    write_csv(rows, args.out)
    print(f"\nwrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
