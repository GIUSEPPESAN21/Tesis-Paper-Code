"""Download and split the OR-Library job-shop benchmark file into per-instance files.

The OR-Library distributes all 82 classic JSSP instances (``abz``, ``ft``, ``la``,
``orb``, ``swv``, ``yn``) in a single text file, ``jobshop1.txt``, wrapped in prose
headers.  This script downloads that file once into ``data/raw/`` (kept verbatim for
provenance) and explodes it into one file per instance under ``data/instances/``,
using the plain "Taillard-style" specification consumed by :mod:`envs.loader`::

    <num_jobs> <num_machines>
    <machine> <duration> <machine> <duration> ...   # one line per job
    ...

Machines are 0-indexed, exactly as in the OR-Library original.

Design note
-----------
We deliberately *do not* transcribe instance data by hand into the repository.
Benchmark tables are large and a single typo silently invalidates every downstream
experiment.  Fetching from the canonical source and asserting structural invariants
(see ``tests/test_loader.py``) is the reproducible option.  The downloaded raw file is
committed alongside the split files so the pipeline can be re-verified offline.

Usage
-----
    python scripts/fetch_instances.py            # download if missing, then split
    python scripts/fetch_instances.py --force    # re-download even if cached
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
INSTANCE_DIR = REPO_ROOT / "data" / "instances"

OR_LIBRARY_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt"
RAW_FILENAME = "jobshop1.txt"

#: Only these families are exploded into ``data/instances/``.  Week 3 of the thesis
#: plan needs Fisher-Thompson and Lawrence; the remaining families stay available in
#: the cached raw file and can be enabled by extending this tuple.
WANTED_PREFIXES: tuple[str, ...] = ("ft", "la")

_INSTANCE_HEADER = re.compile(r"^\s*instance\s+(\S+)\s*$")
_DIMS = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")


def download_raw(force: bool = False) -> Path:
    """Fetch ``jobshop1.txt`` into ``data/raw/`` and return its path.

    Args:
        force: Re-download even when a cached copy already exists.

    Returns:
        Path to the cached raw OR-Library file.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / RAW_FILENAME
    if target.exists() and not force:
        print(f"[cache] {target.relative_to(REPO_ROOT)} already present")
        return target
    print(f"[fetch] {OR_LIBRARY_URL}")
    with urllib.request.urlopen(OR_LIBRARY_URL, timeout=60) as response:
        payload = response.read().decode("utf-8", errors="strict")
    target.write_text(payload, encoding="utf-8")
    print(f"[fetch] wrote {len(payload)} bytes to {target.relative_to(REPO_ROOT)}")
    return target


def split_instances(raw_path: Path, prefixes: tuple[str, ...] = WANTED_PREFIXES) -> list[str]:
    """Split the OR-Library bundle into one Taillard-format file per instance.

    The bundle alternates ``instance <name>`` headers with a free-text description
    line, a dimensions line and then one line per job.  We locate each header, skip
    forward to the first line that parses as ``<jobs> <machines>``, and read exactly
    ``num_jobs`` following non-empty data lines.

    Args:
        raw_path: Path to the cached ``jobshop1.txt``.
        prefixes: Instance-name prefixes to export.

    Returns:
        Sorted list of exported instance names.
    """
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    lines = raw_path.read_text(encoding="utf-8").splitlines()

    exported: list[str] = []
    for idx, line in enumerate(lines):
        header = _INSTANCE_HEADER.match(line)
        if header is None:
            continue
        name = header.group(1)
        if not name.startswith(prefixes):
            continue

        # Walk forward to the dimensions line, tolerating separator/description lines.
        cursor = idx + 1
        dims = None
        while cursor < len(lines):
            dims = _DIMS.match(lines[cursor])
            if dims is not None:
                break
            cursor += 1
        if dims is None:
            raise ValueError(f"no dimensions line found after 'instance {name}'")

        num_jobs, num_machines = int(dims.group(1)), int(dims.group(2))
        job_rows: list[str] = []
        cursor += 1
        while cursor < len(lines) and len(job_rows) < num_jobs:
            stripped = lines[cursor].strip()
            if stripped:
                job_rows.append(" ".join(stripped.split()))
            cursor += 1
        if len(job_rows) != num_jobs:
            raise ValueError(f"instance {name}: expected {num_jobs} job rows, got {len(job_rows)}")
        for row_no, row in enumerate(job_rows):
            tokens = row.split()
            if len(tokens) != 2 * num_machines:
                raise ValueError(
                    f"instance {name}, job {row_no}: expected {2 * num_machines} tokens, "
                    f"got {len(tokens)}"
                )

        out = INSTANCE_DIR / f"{name}.txt"
        out.write_text(f"{num_jobs} {num_machines}\n" + "\n".join(job_rows) + "\n", encoding="utf-8")
        exported.append(name)

    return sorted(exported)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download the raw bundle")
    args = parser.parse_args(argv)

    raw_path = download_raw(force=args.force)
    names = split_instances(raw_path)
    print(f"[split] exported {len(names)} instances -> {INSTANCE_DIR.relative_to(REPO_ROOT)}")
    print("[split] " + ", ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
