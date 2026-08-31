"""Baseline evaluation: run all nine PDRs, the GA, and CP-SAT over ft06, ft10, ft20, and
la01-la10, producing an RPD-vs-known-optimum / wall-clock-time results table.

Originally STEP3 of ``Baselines_TechSpec.md``'s Week 9 scope (PDRs + CP-SAT only); Week
12 (``GA_TechSpec.md``) adds ``GABaseline`` to the same sweep, using the fairness
principle both specs establish: the GA's ``time_limit_s`` equals CP-SAT's on the same
instance, not a separate GA-only budget. This remains a standalone benchmarking script,
not a reusable harness module (``baselines/harness.py`` is still out of scope --- see
``BASELINES_RESULTS_WEEK12.md``).

MK01-MK10/FJSP evaluation is not included: no real FJSP benchmark file loader is bundled
(``envs/fjsp_instance.py`` has a synthetic generator instead --- see its module
docstring), and this repo's own ``scripts/fetch_instances.py`` establishes the precedent
against hand-transcribing benchmark data from memory (see ``BASELINES_RESULTS_WEEK12.md``).
This script predates the Week 15 refactor's unified harness --- see ``run_evaluation.py``
at the repo root for the current entry point, which supports FJSP/GA/CP-SAT and
disruption sweeps uniformly; this file is kept as-is for the Week 9/12 result tables'
provenance.

Usage
-----
    python experiments/eval_baselines.py
    python experiments/eval_baselines.py --time-limit 60 --seed 0
    python experiments/eval_baselines.py --instances ft06 la01 la02
    python experiments/eval_baselines.py --ga-seeds 5   # multiple GA seeds per instance
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.interface import CPSATBaseline, GABaseline, PDRBaseline  # noqa: E402
from envs.loader import known_optimum, load_by_name  # noqa: E402

#: The nine Part B6 PDRs by their canonical registry name (excludes the "mopnr"/"mor"
#: duplicate entry in baselines.dispatching_rules.RULES -- see that dict's docstring).
NINE_PDR_NAMES = ["spt", "lpt", "mwkr", "lwkr", "mor", "lor", "fifo", "edd", "cr"]

DEFAULT_INSTANCES = ["ft06", "ft10", "ft20"] + [f"la{n:02d}" for n in range(1, 11)]

RESULTS_DIR = REPO_ROOT / "experiments" / "results"


def rpd_pct(makespan: float, optimum: int) -> float:
    """Relative Percentage Deviation vs. the known optimum, in percent."""
    return 100.0 * (makespan - optimum) / optimum


def _row(instance, instance_name: str, optimum: int, method: str, seed: int, result) -> dict:
    return {
        "instance": instance_name,
        "num_jobs": instance.num_jobs,
        "num_machines": instance.num_machines,
        "method": method,
        "seed": seed,
        "makespan": result.makespan,
        "known_optimum": optimum,
        "rpd_pct": round(rpd_pct(result.makespan, optimum), 3),
        "wall_clock_s": round(result.wall_clock_time, 4),
        "feasible": result.feasible,
        "status": result.status,
        "is_optimal": result.is_optimal,
    }


def run_evaluation(
    instance_names: list[str], seed: int, time_limit_s: float, ga_seeds: int = 1
) -> list[dict]:
    """Run every baseline over every instance, returning one result record each.

    ``time_limit_s`` is passed uniformly to every baseline's ``solve()`` --- PDR
    adapters accept and ignore it (a single deterministic pass is not meaningfully
    time-limited); ``CPSATBaseline``/``GABaseline`` both actually use it, and use the
    *same* value for a fair comparison (``GA_TechSpec.md`` §3). This uniform call is the
    entire point of the common ``Baseline`` interface (``baselines/interface.py``).

    GA is stochastic (unlike the deterministic-given-seed PDRs and CP-SAT's own
    internally-randomized-but-not-seeded search), so ``ga_seeds`` runs it that many
    times per instance (seeds ``0..ga_seeds-1``), each a separate row -- set to more than
    1 to see the GA's run-to-run variance on a given instance.
    """
    pdr_and_cpsat = [PDRBaseline(name) for name in NINE_PDR_NAMES] + [CPSATBaseline()]

    rows: list[dict] = []
    for instance_name in instance_names:
        instance = load_by_name(instance_name)
        optimum = known_optimum(instance_name)

        for baseline in pdr_and_cpsat:
            result = baseline.solve(instance, seed=seed, time_limit_s=time_limit_s)
            row = _row(instance, instance_name, optimum, baseline.name, seed, result)
            rows.append(row)
            print(
                f"{instance_name:>6}  {baseline.name:>6}  Cmax={result.makespan:<8.1f} "
                f"RPD={row['rpd_pct']:>7.2f}%  {result.wall_clock_time:>8.3f}s  "
                f"feasible={result.feasible}  status={result.status}",
                flush=True,
            )

        ga_baseline = GABaseline()  # population_size auto-scaled per instance (GA_TechSpec.md §2.1)
        for ga_seed in range(ga_seeds):
            result = ga_baseline.solve(instance, seed=ga_seed, time_limit_s=time_limit_s)
            row = _row(instance, instance_name, optimum, "ga", ga_seed, result)
            rows.append(row)
            print(
                f"{instance_name:>6}      ga  Cmax={result.makespan:<8.1f} "
                f"RPD={row['rpd_pct']:>7.2f}%  {result.wall_clock_time:>8.3f}s  "
                f"feasible={result.feasible}  status={result.status}  "
                f"(seed={ga_seed}, generations={result.extra['generations_run']})",
                flush=True,
            )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path, seed: int, time_limit_s: float, ga_seeds: int) -> None:
    ga_note = f"; GA additionally run with seeds 0-{ga_seeds - 1}" if ga_seeds > 1 else ""
    lines = [
        f"# Baseline results (CP-SAT/GA time limit {time_limit_s:.0f}s, seed {seed}{ga_note})",
        "",
        "| instance | jobs x machines | method | seed | makespan | known optimum | RPD (%) | "
        "wall-clock (s) | feasible | status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['instance']} | {row['num_jobs']}x{row['num_machines']} | {row['method']} | "
            f"{row['seed']} | {row['makespan']:.1f} | {row['known_optimum']} | {row['rpd_pct']:.2f} | "
            f"{row['wall_clock_s']:.3f} | {row['feasible']} | {row['status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=60.0, help="CP-SAT/GA wall-clock budget, seconds")
    parser.add_argument("--seed", type=int, default=0, help="PDR tie-break / CP-SAT seed")
    parser.add_argument("--ga-seeds", type=int, default=1, help="number of GA seeds (0..N-1) to run per instance")
    parser.add_argument("--instances", nargs="*", default=DEFAULT_INSTANCES)
    parser.add_argument("--out-prefix", default="baselines_week12")
    args = parser.parse_args(argv)

    rows = run_evaluation(args.instances, seed=args.seed, time_limit_s=args.time_limit, ga_seeds=args.ga_seeds)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{args.out_prefix}.csv"
    md_path = RESULTS_DIR / f"{args.out_prefix}.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, args.seed, args.time_limit, args.ga_seeds)

    print(f"\nwrote {len(rows)} rows to {csv_path.relative_to(REPO_ROOT)} and {md_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
