"""Parser and registry for the Brandimarte (``MK*``) flexible job-shop benchmarks.

Why this module exists
----------------------
``envs/fjsp_instance.py``'s own docstring records the state of play as of Week 15: *"No
real FJSP benchmark loader (Brandimarte ``MK*``, Hurink) is bundled this sprint"*, with
:func:`~envs.fjsp_instance.generate_fjsp_instance` standing in as a synthetic
substitute derived from the bundled ``ft``/``la`` JSSP data.  A synthetic instance is
enough to exercise the FJSP *code paths*, but it has no literature-proven optimum and
therefore no RPD denominator, so it cannot support the "compare the trained policy's
makespan against the baselines" claim the Week 18 sprint has to make on a real flexible
instance.  This module closes that gap for the Brandimarte family.

Format
------
The Brandimarte layout is one header line followed by one line per job::

    <num_jobs> <num_machines> [<avg_machines_per_operation>]
    <n_ops> (<n_eligible> (<machine> <duration>) x n_eligible) x n_ops
    ...

The optional third header token is a summary statistic, not data, and is ignored.
Machine indices appear 0-indexed in the canonical mirror this repo fetches from and
1-indexed in some older transcriptions; :func:`parse_brandimarte` detects which and
normalises to the 0-indexing every other module in this codebase uses.

Rectangularity padding (the one real modelling decision here)
--------------------------------------------------------------
Brandimarte jobs do **not** all have the same number of operations (``mk01``: six jobs
of 6 operations and four of 5).  :class:`~envs.fjsp_instance.FJSPInstance` is rectangular
by construction, and that is load-bearing well beyond this parser: the global operation
id ``op_id = job * ops_per_job + position`` is the action id in every environment, the
gene index in :class:`~baselines.ga.GeneticAlgorithm`'s chromosome, and the row index of
every adjacency matrix and start-time vector in the codebase.  Making it ragged is a
change to that shared indexing convention, not to one parser.

So short jobs are padded to ``ops_per_job = max_j n_ops(j)`` with **zero-duration
operations eligible on machine 0 only**, appended at the end of the job chain.  This is
exactly makespan-neutral, provided the consumer treats a zero-duration operation as
occupying no machine capacity --- which is what :class:`~envs.fjsp_env.FJSPEnv` does
explicitly (see its ``step``), and what CP-SAT's ``NoOverlap`` does for zero-size
intervals.  ``tests/test_fjsp_loader.py`` pins the neutrality claim by checking that
CP-SAT reaches the literature optimum on the padded instance.

:attr:`FJSPInstance.name` is unchanged by padding, and :func:`real_operation_mask` lets
any consumer that cares recover which operation ids are genuine.

Sources
-------
Brandimarte, P. (1993). *Routing and scheduling in a flexible job shop by tabu search.*
Annals of Operations Research, 41(3), 157-183.  Instance files and reference makespans
are fetched from the ``SchedulingLab/fjsp-instances`` mirror (see
``scripts/fetch_fjsp_instances.py``) rather than transcribed by hand, for the same
reason ``scripts/fetch_instances.py`` gives for the JSSP family: a single typo in a
hand-copied benchmark table silently invalidates every downstream experiment.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from envs.fjsp_instance import FJSPInstance

REPO_ROOT = Path(__file__).resolve().parents[1]
FJSP_INSTANCE_DIR = REPO_ROOT / "data" / "fjsp_instances"
FJSP_OPTIMA_CSV = REPO_ROOT / "data" / "fjsp_known_optima.csv"

#: Machine a padding (zero-duration) operation is nominally routed to.  Arbitrary: a
#: zero-duration operation never occupies the machine (see the module docstring).
PADDING_MACHINE = 0


class FJSPParseError(ValueError):
    """Raised when a file does not conform to the Brandimarte layout."""


@dataclass(frozen=True)
class FJSPReference:
    """Literature reference makespan for an FJSP benchmark instance.

    Attributes:
        name: Instance name, e.g. ``"mk01"``.
        family: Benchmark family, e.g. ``"brandimarte"``.
        num_jobs: Number of jobs (redundant with the instance file; used as a check).
        num_machines: Number of machines (idem).
        best_makespan: Reference makespan.
        status: ``"proven"`` if the value is a proven optimum, ``"best_known"`` if it is
            only the best published upper bound.
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


def _tokens(text: str) -> list[int]:
    """Split a Brandimarte file body into integer tokens, ignoring comment lines.

    Raises:
        FJSPParseError: If a non-comment token is not an integer.
    """
    values: list[int] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for token in stripped.split():
            try:
                values.append(int(float(token)))
            except ValueError as exc:
                raise FJSPParseError(f"line {line_no}: non-integer token {token!r}") from exc
    return values


def parse_brandimarte(
    text: str, name: str, source_file: str | None = None
) -> FJSPInstance:
    """Parse the Brandimarte ``MK*`` layout into a padded :class:`FJSPInstance`.

    Args:
        text: Full file contents.
        name: Instance name to attach to the result, e.g. ``"mk01"``.
        source_file: Provenance path (recorded only in this function's error messages;
            :class:`FJSPInstance` has no ``source_file`` field).

    Returns:
        The parsed instance, padded to rectangular per the module docstring.

    Raises:
        FJSPParseError: If the header is missing, a job row is truncated, or the token
            stream does not consume exactly.
    """
    values = _tokens(text)
    if len(values) < 2:
        raise FJSPParseError(f"{name}: file too short ({source_file})")

    num_jobs, num_machines = values[0], values[1]
    if num_jobs <= 0 or num_machines <= 0:
        raise FJSPParseError(f"{name}: bad header {num_jobs} x {num_machines}")

    # The optional third header token is an average-flexibility statistic.  It is
    # distinguishable from the first job's operation count only by trying both readings;
    # we try the 2-token header first and fall back, since the 3-token variant is rarer.
    for header_len in (2, 3):
        try:
            jobs = _parse_job_rows(values[header_len:], num_jobs, name)
        except FJSPParseError:
            continue
        return _build_padded_instance(jobs, name, num_machines)

    raise FJSPParseError(
        f"{name}: token stream does not parse as Brandimarte under either a 2- or "
        f"3-token header ({source_file})"
    )


def _parse_job_rows(
    body: list[int], num_jobs: int, name: str
) -> list[list[list[tuple[int, int]]]]:
    """Consume ``num_jobs`` job rows from ``body``.

    Returns:
        ``jobs[job][position]`` is the list of ``(machine, duration)`` alternatives.

    Raises:
        FJSPParseError: If the stream is truncated or does not consume exactly.
    """
    cursor = 0
    jobs: list[list[list[tuple[int, int]]]] = []

    def take() -> int:
        nonlocal cursor
        if cursor >= len(body):
            raise FJSPParseError(f"{name}: token stream truncated at offset {cursor}")
        value = body[cursor]
        cursor += 1
        return value

    for _ in range(num_jobs):
        n_ops = take()
        if n_ops <= 0:
            raise FJSPParseError(f"{name}: job declares {n_ops} operations")
        job: list[list[tuple[int, int]]] = []
        for _ in range(n_ops):
            n_eligible = take()
            if n_eligible <= 0:
                raise FJSPParseError(f"{name}: operation declares {n_eligible} alternatives")
            job.append([(take(), take()) for _ in range(n_eligible)])
        jobs.append(job)

    if cursor != len(body):
        raise FJSPParseError(
            f"{name}: {len(body) - cursor} trailing tokens after {num_jobs} job rows"
        )
    return jobs


def _build_padded_instance(
    jobs: list[list[list[tuple[int, int]]]], name: str, num_machines: int
) -> FJSPInstance:
    """Normalise machine indexing, pad to rectangular, and construct the instance.

    Raises:
        FJSPParseError: If a machine index is outside ``[0, num_machines)`` after
            normalisation, or an alternative list is empty.
    """
    all_machines = [m for job in jobs for op in job for m, _ in op]
    # Some transcriptions are 1-indexed; detect and shift, mirroring envs.loader's
    # parse_taillard.
    shift = 1 if (min(all_machines) == 1 and max(all_machines) == num_machines) else 0
    if shift == 0 and (min(all_machines) < 0 or max(all_machines) >= num_machines):
        raise FJSPParseError(
            f"{name}: machine indices [{min(all_machines)}, {max(all_machines)}] outside "
            f"[0, {num_machines}) and do not look 1-indexed either"
        )

    ops_per_job = max(len(job) for job in jobs)
    eligible_machines: list[list[list[int]]] = []
    durations: list[list[list[int]]] = []
    for job in jobs:
        job_machines: list[list[int]] = []
        job_durations: list[list[int]] = []
        for alternatives in job:
            machines_here: list[int] = []
            durations_here: list[int] = []
            for machine, duration in alternatives:
                machine -= shift
                if machine in machines_here:
                    # A duplicate machine for one operation is a transcription artefact;
                    # keep the cheaper alternative rather than failing the whole parse.
                    slot = machines_here.index(machine)
                    durations_here[slot] = min(durations_here[slot], duration)
                    continue
                machines_here.append(machine)
                durations_here.append(duration)
            job_machines.append(machines_here)
            job_durations.append(durations_here)
        for _ in range(ops_per_job - len(job)):
            job_machines.append([PADDING_MACHINE])
            job_durations.append([0])
        eligible_machines.append(job_machines)
        durations.append(job_durations)

    return FJSPInstance(
        name=name,
        num_jobs=len(jobs),
        ops_per_job=ops_per_job,
        num_machines=num_machines,
        eligible_machines=eligible_machines,
        durations=durations,
    )


def real_operation_mask(instance: FJSPInstance) -> np.ndarray:
    """Boolean mask of which global operation ids are genuine rather than padding.

    A padding operation is the loader's zero-duration, single-eligible-machine filler
    (see the module docstring).  Genuine zero-duration operations do not occur in the
    Brandimarte family, so the test is exact for instances this module parses.

    Args:
        instance: The FJSP instance.

    Returns:
        Boolean array of shape ``(num_operations,)``; ``True`` marks a real operation.
    """
    mask = np.ones(instance.num_operations, dtype=bool)
    for job in range(instance.num_jobs):
        for pos in range(instance.ops_per_job):
            durs = instance.durations[job][pos]
            if len(durs) == 1 and durs[0] == 0:
                mask[instance.op_id(job, pos)] = False
    return mask


def load_fjsp_instance(path: str | Path, name: str | None = None) -> FJSPInstance:
    """Load one Brandimarte-format file.

    Args:
        path: Path to the instance file.
        name: Override for the instance name (defaults to the file stem).

    Returns:
        The parsed :class:`FJSPInstance`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        FJSPParseError: If the file does not parse.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FJSP instance file not found: {path}")
    return parse_brandimarte(
        path.read_text(encoding="utf-8"), name or path.stem, source_file=str(path)
    )


@lru_cache(maxsize=None)
def load_fjsp_by_name(name: str) -> FJSPInstance:
    """Load a bundled FJSP instance by name, e.g. ``load_fjsp_by_name("mk01")``.

    Cached like :func:`~envs.loader.load_by_name`: instances are immutable.

    Args:
        name: Instance name matching a file in ``data/fjsp_instances/``.

    Returns:
        The parsed :class:`FJSPInstance`.

    Raises:
        FileNotFoundError: If no such instance is bundled.
    """
    path = FJSP_INSTANCE_DIR / f"{name}.txt"
    if not path.exists():
        available = ", ".join(available_fjsp_instances()) or (
            "<none: run scripts/fetch_fjsp_instances.py>"
        )
        raise FileNotFoundError(f"unknown FJSP instance {name!r}. Available: {available}")
    return load_fjsp_instance(path, name=name)


def available_fjsp_instances() -> list[str]:
    """Sorted names of every FJSP instance bundled under ``data/fjsp_instances/``."""
    if not FJSP_INSTANCE_DIR.exists():
        return []
    return sorted(p.stem for p in FJSP_INSTANCE_DIR.glob("*.txt"))


@lru_cache(maxsize=1)
def load_fjsp_known_optima() -> dict[str, FJSPReference]:
    """Read ``data/fjsp_known_optima.csv`` into a name-keyed mapping.

    Returns:
        Mapping from instance name to :class:`FJSPReference`.

    Raises:
        FileNotFoundError: If the CSV is missing.
    """
    if not FJSP_OPTIMA_CSV.exists():
        raise FileNotFoundError(f"missing reference table: {FJSP_OPTIMA_CSV}")
    table: dict[str, FJSPReference] = {}
    with FJSP_OPTIMA_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            table[row["name"]] = FJSPReference(
                name=row["name"],
                family=row["family"],
                num_jobs=int(row["num_jobs"]),
                num_machines=int(row["num_machines"]),
                best_makespan=int(row["best_makespan"]),
                status=row["status"],
                reference=row["reference"],
            )
    return table


def fjsp_known_optimum(name: str) -> int:
    """Reference makespan for an FJSP benchmark instance.

    Args:
        name: Instance name, e.g. ``"mk01"``.

    Returns:
        The reference makespan.

    Raises:
        KeyError: If the instance has no recorded reference value.
    """
    table = load_fjsp_known_optima()
    if name not in table:
        raise KeyError(f"no reference makespan recorded for FJSP instance {name!r}")
    return table[name].best_makespan
