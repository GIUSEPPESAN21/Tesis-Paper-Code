"""Download the Brandimarte (``MK*``) FJSP benchmarks and their reference makespans.

Companion to ``scripts/fetch_instances.py`` (the classic-JSSP fetcher) and written to
the same design note: **we deliberately do not transcribe instance data by hand.**
Benchmark tables are large and a single typo silently invalidates every downstream
experiment, so the canonical files are fetched and the structural invariants are
asserted (see ``tests/test_fjsp_loader.py``).

Source
------
The Brandimarte instances are not distributed by the OR-Library.  This script uses the
``SchedulingLab/fjsp-instances`` mirror, which publishes the original ``.txt`` files
alongside an ``instances.json`` index carrying each instance's proven optimum or
best-known bounds --- so the reference-makespan table is fetched from the same source
as the data rather than typed in from a paper.

Usage
-----
    python scripts/fetch_fjsp_instances.py                  # mk01 (the sprint default)
    python scripts/fetch_fjsp_instances.py --instances mk01 mk02 mk03
    python scripts/fetch_fjsp_instances.py --all --force
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RAW_DIR = REPO_ROOT / "data" / "raw"
FJSP_INSTANCE_DIR = REPO_ROOT / "data" / "fjsp_instances"
FJSP_OPTIMA_CSV = REPO_ROOT / "data" / "fjsp_known_optima.csv"

MIRROR_BASE = "https://raw.githubusercontent.com/SchedulingLab/fjsp-instances/main"
INDEX_URL = f"{MIRROR_BASE}/instances.json"

#: Week 18 needs one small flexible instance; ``mk01`` (10 jobs x 6 machines, proven
#: optimum 40) is the smallest Brandimarte instance and the standard entry point.
DEFAULT_INSTANCES = ("mk01",)

CITATION = "Brandimarte (1993), via SchedulingLab/fjsp-instances"


def fetch_index(force: bool = False) -> list[dict]:
    """Fetch (and cache under ``data/raw/``) the mirror's instance index.

    Args:
        force: Re-download even when a cached copy exists.

    Returns:
        The decoded ``instances.json`` list.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cached = RAW_DIR / "fjsp_instances_index.json"
    if force or not cached.exists():
        print(f"[fetch] {INDEX_URL}")
        with urllib.request.urlopen(INDEX_URL, timeout=60) as response:
            cached.write_bytes(response.read())
    return json.loads(cached.read_text(encoding="utf-8"))


def fetch_instance(entry: dict, force: bool = False) -> Path:
    """Download one instance file into ``data/fjsp_instances/``.

    Args:
        entry: One record of the mirror index (needs ``name`` and ``path``).
        force: Re-download even when the file already exists.

    Returns:
        Path to the written instance file.
    """
    FJSP_INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    target = FJSP_INSTANCE_DIR / f"{entry['name']}.txt"
    if target.exists() and not force:
        print(f"[skip]  {target.name} already present")
        return target
    url = f"{MIRROR_BASE}/{entry['path']}"
    print(f"[fetch] {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        target.write_bytes(response.read())
    return target


def write_optima_csv(entries: list[dict]) -> None:
    """Write/merge ``data/fjsp_known_optima.csv`` from the mirror index.

    An entry with a proven ``optimum`` is recorded ``status="proven"``; one with only
    ``bounds.upper`` is recorded ``status="best_known"`` at that upper bound, which is
    the convention ``data/known_optima.csv`` already uses for the JSSP family.

    Args:
        entries: Mirror index records for the instances to record.
    """
    existing: dict[str, dict] = {}
    if FJSP_OPTIMA_CSV.exists():
        with FJSP_OPTIMA_CSV.open(encoding="utf-8", newline="") as handle:
            existing = {row["name"]: row for row in csv.DictReader(handle)}

    for entry in entries:
        optimum = entry.get("optimum")
        if optimum is not None:
            best, status = int(optimum), "proven"
        elif entry.get("bounds", {}).get("upper") is not None:
            best, status = int(entry["bounds"]["upper"]), "best_known"
        else:
            print(f"[warn]  {entry['name']}: no reference makespan in the index; skipped")
            continue
        existing[entry["name"]] = {
            "name": entry["name"],
            "family": entry["path"].split("/")[0],
            "num_jobs": str(entry["jobs"]),
            "num_machines": str(entry["machines"]),
            "best_makespan": str(best),
            "status": status,
            "reference": CITATION,
        }

    fields = ["name", "family", "num_jobs", "num_machines", "best_makespan", "status", "reference"]
    with FJSP_OPTIMA_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in sorted(existing):
            writer.writerow({key: existing[name][key] for key in fields})
    print(f"[write] {FJSP_OPTIMA_CSV.relative_to(REPO_ROOT)} ({len(existing)} rows)")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", default=list(DEFAULT_INSTANCES))
    parser.add_argument("--all", action="store_true", help="fetch every Brandimarte instance")
    parser.add_argument("--force", action="store_true", help="re-download cached files")
    args = parser.parse_args(argv)

    index = fetch_index(force=args.force)
    brandimarte = [e for e in index if e["path"].startswith("brandimarte/")]
    wanted = brandimarte if args.all else [e for e in brandimarte if e["name"] in set(args.instances)]

    missing = set(args.instances) - {e["name"] for e in wanted}
    if missing and not args.all:
        raise SystemExit(
            f"unknown instance(s) {sorted(missing)}. Available: "
            f"{sorted(e['name'] for e in brandimarte)}"
        )

    for entry in wanted:
        fetch_instance(entry, force=args.force)
    write_optima_csv(wanted)

    # Structural verification: the parser must accept everything just written.
    from envs.fjsp_loader import load_fjsp_by_name, real_operation_mask  # noqa: PLC0415

    for entry in wanted:
        instance = load_fjsp_by_name(entry["name"])
        n_real = int(real_operation_mask(instance).sum())
        print(
            f"[ok]    {instance.name}: {instance.num_jobs} jobs x {instance.ops_per_job} "
            f"(padded) ops, {instance.num_machines} machines, {n_real} real operations"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
