"""Regenerate ``data/known_optima.csv`` by proving optima with CP-SAT.

Rather than trusting a hand-transcribed literature table, this script solves every
bundled instance with :func:`baselines.cpsat.solve_cpsat` and writes the proven optima
it obtains.  Instances not closed within the time limit keep the literature value from
the existing CSV and are marked ``best_known``.

The script is *not* part of the test suite --- it is the provenance trail for the CSV,
which is committed so the tests stay fast and offline.

Usage
-----
    python scripts/verify_optima.py --time-limit 60
    python scripts/verify_optima.py --time-limit 300 --only la36 la37 la38 la39 la40
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.cpsat import solve_cpsat  # noqa: E402
from envs.loader import KNOWN_OPTIMA_CSV, available_instances, load_by_name  # noqa: E402

FAMILIES = {"ft": "fisher-thompson", "la": "lawrence", "ab": "adams-balas-zawack"}


def family_of(name: str) -> str:
    """Map an instance name to its benchmark family label."""
    return FAMILIES.get(name[:2], "unknown")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=60.0, help="seconds per instance")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these instances")
    args = parser.parse_args(argv)

    existing: dict[str, dict[str, str]] = {}
    if KNOWN_OPTIMA_CSV.exists():
        with KNOWN_OPTIMA_CSV.open(encoding="utf-8", newline="") as handle:
            existing = {row["name"]: row for row in csv.DictReader(handle)}

    names = args.only or available_instances()
    rows: list[dict[str, object]] = []
    for name in names:
        instance = load_by_name(name)
        result = solve_cpsat(instance, time_limit_s=args.time_limit, num_workers=args.workers)
        prior = existing.get(name)
        prior_value = int(prior["best_makespan"]) if prior else None

        if result.is_optimal and result.makespan is not None:
            makespan, status, reference = result.makespan, "proven", "cp-sat (this repo)"
            flag = "" if prior_value in (None, makespan) else f"  <-- CSV had {prior_value}"
        elif prior_value is not None:
            makespan, status = prior_value, prior["status"]
            reference = prior["reference"]
            flag = f"  (not closed in {args.time_limit}s; kept literature value)"
        else:  # pragma: no cover - only if a brand-new instance stays open
            makespan, status, reference = result.makespan or -1, "best_known", "cp-sat (open)"
            flag = "  (open)"

        print(
            f"{name:>6}  {instance.num_jobs:>2}x{instance.num_machines:<2}  "
            f"Cmax={makespan:<6} bound={result.best_bound:<6} {result.status:<9} "
            f"{result.wall_time:6.2f}s{flag}",
            flush=True,
        )
        rows.append(
            {
                "name": name,
                "family": family_of(name),
                "num_jobs": instance.num_jobs,
                "num_machines": instance.num_machines,
                "best_makespan": makespan,
                "status": status,
                "reference": reference,
            }
        )

    if args.only is None:
        with KNOWN_OPTIMA_CSV.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "name",
                    "family",
                    "num_jobs",
                    "num_machines",
                    "best_makespan",
                    "status",
                    "reference",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {KNOWN_OPTIMA_CSV.relative_to(REPO_ROOT)}")
    else:
        print("\n--only given: CSV left untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
