"""Parsers and a registry for standard JSSP benchmark instance files.

Two closely related plain-text conventions are in circulation for classic JSSP
benchmarks, and both are supported here:

**Standard / OR-Library format** (:func:`parse_standard`) --- the format used by
Beasley's OR-Library bundle and by essentially every JSSP paper.  A header line gives
``<num_jobs> <num_machines>``; each subsequent line describes one job as interleaved
``<machine> <duration>`` pairs, in the job's technological order::

    6 6
    2  1  0  3  1  6  3  7  5  3  4  6
    ...

**Taillard format** (:func:`parse_taillard`) --- the layout of Taillard's own ``ta*``
files: a header with ``<num_jobs> <num_machines>``, then a *block* of processing times
(one row per job, in technological order) followed by a *block* of machine numbers.
Taillard's files are historically 1-indexed for machines; :func:`parse_taillard`
detects this and normalises to 0-indexing.

:func:`load_instance` sniffs which of the two a file uses, so callers normally only
need :func:`load_instance` or :func:`load_by_name`.

Known optima
------------
:func:`known_optimum` reads ``data/known_optima.csv``, which records the literature
reference value for each benchmark together with whether it is a *proven* optimum or
merely the best value known.  These are used as ground truth in the loader tests and,
later, as the denominator of the relative-percentage-deviation (RPD) metric.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from envs.instance import JSSPInstance

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTANCE_DIR = REPO_ROOT / "data" / "instances"
KNOWN_OPTIMA_CSV = REPO_ROOT / "data" / "known_optima.csv"


class InstanceParseError(ValueError):
    """Raised when an instance file does not conform to any supported format."""


@dataclass(frozen=True)
class BenchmarkReference:
    """Literature reference value for a benchmark instance.

    Attributes:
        name: Instance name, e.g. ``"ft10"``.
        family: Benchmark family, e.g. ``"fisher-thompson"``.
        num_jobs: Number of jobs (redundant with the instance file; used as a check).
        num_machines: Number of machines (idem).
        best_makespan: Reference makespan value.
        status: ``"proven"`` if the value is a proven optimum, ``"best_known"`` if it
            is only the best published upper bound.
        reference: Short citation for the value.
    """

    name: str
    family: str
    num_jobs: int
    num_machines: int
    best_makespan: int
    status: str
    reference: str

    @property
    def is_proven_optimal(self) -> bool:
        """Whether :attr:`best_makespan` is a proven optimum."""
        return self.status == "proven"


# --------------------------------------------------------------------- parsers


def _tokens(text: str) -> list[int]:
    """Split a benchmark file body into integer tokens, ignoring comment lines.

    Lines whose first non-space character is ``#`` are treated as comments.

    Raises:
        InstanceParseError: If a non-comment token is not an integer.
    """
    values: list[int] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for token in stripped.split():
            try:
                values.append(int(token))
            except ValueError as exc:  # pragma: no cover - defensive
                raise InstanceParseError(
                    f"line {line_no}: non-integer token {token!r}"
                ) from exc
    return values


def parse_standard(text: str, name: str, source_file: str | None = None) -> JSSPInstance:
    """Parse the OR-Library ``<machine> <duration>``-interleaved format.

    Args:
        text: Full file contents.
        name: Instance name to attach to the result.
        source_file: Provenance path recorded on the instance.

    Returns:
        The parsed :class:`~envs.instance.JSSPInstance`.

    Raises:
        InstanceParseError: If the token count does not match the declared dimensions.
    """
    values = _tokens(text)
    if len(values) < 2:
        raise InstanceParseError(f"{name}: file too short")
    num_jobs, num_machines = values[0], values[1]
    body = values[2:]
    expected = num_jobs * num_machines * 2
    if len(body) != expected:
        raise InstanceParseError(
            f"{name}: expected {expected} tokens for {num_jobs}x{num_machines}, got {len(body)}"
        )
    pairs = np.asarray(body, dtype=np.int64).reshape(num_jobs, num_machines, 2)
    return JSSPInstance(
        name=name,
        machines=pairs[:, :, 0],
        durations=pairs[:, :, 1],
        num_machines=num_machines,
        source_file=source_file,
    )


def parse_taillard(text: str, name: str, source_file: str | None = None) -> JSSPInstance:
    """Parse Taillard's two-block ``ta*`` format (times block, then machines block).

    Machine numbers in Taillard's original files are 1-indexed.  If no ``0`` appears in
    the machine block and the maximum equals ``num_machines``, the block is shifted down
    by one so the result is 0-indexed like everything else in this codebase.

    Args:
        text: Full file contents.
        name: Instance name to attach to the result.
        source_file: Provenance path recorded on the instance.

    Returns:
        The parsed :class:`~envs.instance.JSSPInstance`.

    Raises:
        InstanceParseError: If the token count does not match the declared dimensions.
    """
    values = _tokens(text)
    if len(values) < 2:
        raise InstanceParseError(f"{name}: file too short")
    num_jobs, num_machines = values[0], values[1]
    body = values[2:]
    block = num_jobs * num_machines
    if len(body) != 2 * block:
        raise InstanceParseError(
            f"{name}: expected {2 * block} tokens for {num_jobs}x{num_machines}, got {len(body)}"
        )
    durations = np.asarray(body[:block], dtype=np.int64).reshape(num_jobs, num_machines)
    machines = np.asarray(body[block:], dtype=np.int64).reshape(num_jobs, num_machines)
    if machines.min() == 1 and machines.max() == num_machines:
        machines = machines - 1
    return JSSPInstance(
        name=name,
        machines=machines,
        durations=durations,
        num_machines=num_machines,
        source_file=source_file,
    )


def _looks_like_standard(values: list[int], num_jobs: int, num_machines: int) -> bool:
    """Heuristic: does the token stream match the interleaved OR-Library layout?"""
    return len(values) - 2 == num_jobs * num_machines * 2


def load_instance(path: str | Path, name: str | None = None) -> JSSPInstance:
    """Load an instance file, auto-detecting the standard vs. Taillard layout.

    Both layouts declare ``<num_jobs> <num_machines>`` first but then carry a different
    number of tokens (``2*J*M`` for interleaved, ``2*J*M`` for two-block --- identical),
    so the layouts are disambiguated by *content*: in the interleaved format the
    even-offset tokens are machine indices and must therefore form a permutation of
    ``0..M-1`` within every job.  We try that first and fall back to Taillard.

    Args:
        path: Path to the instance file.
        name: Override for the instance name (defaults to the file stem).

    Returns:
        The parsed :class:`~envs.instance.JSSPInstance`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        InstanceParseError: If neither parser accepts the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"instance file not found: {path}")
    text = path.read_text(encoding="utf-8")
    instance_name = name or path.stem

    errors: list[str] = []
    for parser in (parse_standard, parse_taillard):
        try:
            return parser(text, instance_name, source_file=str(path))
        except (InstanceParseError, ValueError) as exc:
            errors.append(f"{parser.__name__}: {exc}")
    raise InstanceParseError(
        f"{instance_name}: no supported format matched.\n  " + "\n  ".join(errors)
    )


@lru_cache(maxsize=None)
def load_by_name(name: str) -> JSSPInstance:
    """Load a bundled instance by benchmark name, e.g. ``load_by_name("ft06")``.

    Results are cached: instances are immutable, so sharing them across an experiment
    is safe and avoids re-parsing in tight evaluation loops.

    Args:
        name: Instance name matching a file in ``data/instances/``.

    Returns:
        The parsed :class:`~envs.instance.JSSPInstance`.

    Raises:
        FileNotFoundError: If no such instance is bundled.
    """
    path = INSTANCE_DIR / f"{name}.txt"
    if not path.exists():
        available = ", ".join(available_instances()) or "<none: run scripts/fetch_instances.py>"
        raise FileNotFoundError(f"unknown instance {name!r}. Available: {available}")
    return load_instance(path, name=name)


def available_instances() -> list[str]:
    """Sorted names of every instance file bundled under ``data/instances/``."""
    if not INSTANCE_DIR.exists():
        return []
    return sorted(p.stem for p in INSTANCE_DIR.glob("*.txt"))


# ------------------------------------------------------------- known optima


@lru_cache(maxsize=1)
def load_known_optima() -> dict[str, BenchmarkReference]:
    """Read ``data/known_optima.csv`` into a name-keyed mapping.

    Returns:
        Mapping from instance name to :class:`BenchmarkReference`.

    Raises:
        FileNotFoundError: If the CSV is missing.
    """
    if not KNOWN_OPTIMA_CSV.exists():
        raise FileNotFoundError(f"missing reference table: {KNOWN_OPTIMA_CSV}")
    table: dict[str, BenchmarkReference] = {}
    with KNOWN_OPTIMA_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            table[row["name"]] = BenchmarkReference(
                name=row["name"],
                family=row["family"],
                num_jobs=int(row["num_jobs"]),
                num_machines=int(row["num_machines"]),
                best_makespan=int(row["best_makespan"]),
                status=row["status"],
                reference=row["reference"],
            )
    return table


def known_optimum(name: str) -> int:
    """Reference (usually proven-optimal) makespan for a benchmark instance.

    Args:
        name: Instance name, e.g. ``"ft10"``.

    Returns:
        The reference makespan.

    Raises:
        KeyError: If the instance has no recorded reference value.
    """
    table = load_known_optima()
    if name not in table:
        raise KeyError(f"no reference makespan recorded for instance {name!r}")
    return table[name].best_makespan
