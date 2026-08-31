"""Ingestion, schema validation, and derived columns for E1-E8 result CSVs.

Implements ``StatsAnalysis_TechSpec.md`` Sections 2-3. This module never re-derives
``rpd_pct``/``known_optimum`` (already computed by ``run_evaluation.py``); it only loads,
validates, tags, and adds the small set of columns Section 2.4 lists as needed downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Section 2.1's CSV schema, unchanged from ``run_evaluation.py``.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "instance",
    "family",
    "num_jobs",
    "num_machines",
    "method",
    "disruption",
    "seed",
    "makespan",
    "known_optimum",
    "rpd_pct",
    "wall_clock_s",
    "feasible",
    "status",
    "is_optimal",
    "violations_count",
    "timestamp",
)

#: Section 2.4: post-padding operation counts for the 7 bundled Brandimarte FJSP
#: instances (hardcoded, not recomputed from num_jobs/num_machines -- FJSP padding
#: makes op count non-multiplicative). Sourced from
#: ``envs.fjsp_loader.load_fjsp_by_name(name).num_operations`` at authoring time.
FJSP_NUM_OPERATIONS: dict[str, int] = {
    "mk01": 60,
    "mk02": 60,
    "mk03": 150,
    "mk04": 135,
    "mk05": 135,
    "mk06": 150,
    "mk07": 100,
}

#: RL rollout-mode convention (Section 2.2): seed==0 is the single confirmatory
#: deterministic rollout; seeds 1-4 are the four descriptive stochastic rollouts.
#: This convention applies ONLY to nominal (``disruption == "none"``) rows -- H1/H2/H5's
#: measurement protocol. Disruption-regime rows (E4/E5/E6/E7, PreRegistration_FINAL.md
#: Sec 4's grid table) use 30 equally-weighted replicate seeds instead, with no special
#: "deterministic seed 0" role at all (RI is a mean over all replicates).
RL_DETERMINISTIC_SEED = 0
RL_STOCHASTIC_SEEDS = frozenset({1, 2, 3, 4})
NOMINAL_DISRUPTION_LABEL = "none"


class DataContractError(ValueError):
    """Raised when an input CSV violates a contract this pipeline must not silently
    guess around (Section 2.2/2.1)."""


def validate_schema(df: pd.DataFrame, *, source: str = "<unknown>") -> None:
    """Assert ``df`` has every column ``REQUIRED_COLUMNS`` lists.

    Args:
        df: A single experiment's loaded rows.
        source: Path or label, for the error message.

    Raises:
        DataContractError: If any required column is missing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataContractError(f"{source}: missing required column(s) {missing}")


def validate_rl_rollout_convention(df: pd.DataFrame, *, source: str = "<unknown>") -> None:
    """Section 2.2's hard validation gate.

    For every ``(instance, disruption)`` block where ``method == "rl"`` appears:

    - ``disruption == "none"`` (nominal, H1/H2/H5's protocol): assert exactly one row
      has ``seed == 0`` and the remaining rows (if present) have ``seed in {1,2,3,4}``
      with no duplicates.
    - any other ``disruption`` value (a disruption regime, H3/H4's protocol per
      ``PreRegistration_FINAL.md`` Sec 4): no deterministic-seed requirement -- assert
      only that seeds within the block are non-duplicated (the grid specifies 30
      equally-weighted replicate seeds, all contributing to the RI mean alike).

    Raises:
        DataContractError: If the convention is violated for any block. Never falls
            back to "use the first row found".
    """
    rl_rows = df[df["method"] == "rl"]
    if rl_rows.empty:
        return
    for (instance, disruption), block in rl_rows.groupby(["instance", "disruption"]):
        seeds = block["seed"].tolist()
        if len(seeds) != len(set(seeds)):
            raise DataContractError(
                f"{source}: instance={instance!r} disruption={disruption!r} has "
                f"duplicate rl seeds: {sorted(seeds)}"
            )
        if disruption != NOMINAL_DISRUPTION_LABEL:
            continue
        n_det = sum(1 for s in seeds if s == RL_DETERMINISTIC_SEED)
        if n_det != 1:
            raise DataContractError(
                f"{source}: instance={instance!r} disruption={disruption!r} has "
                f"{n_det} rl row(s) with seed==0 (expected exactly 1); seeds present: "
                f"{sorted(seeds)}"
            )
        stochastic = [s for s in seeds if s != RL_DETERMINISTIC_SEED]
        bad = [s for s in stochastic if s not in RL_STOCHASTIC_SEEDS]
        if bad:
            raise DataContractError(
                f"{source}: instance={instance!r} disruption={disruption!r} has rl "
                f"seed(s) outside {{0}} u {{1,2,3,4}}: {bad}"
            )


def _num_operations_row(row: pd.Series) -> float:
    """Section 2.4's derivation rule, applied to one row."""
    if row["family"] == "brandimarte":
        return float(FJSP_NUM_OPERATIONS.get(row["instance"], np.nan))
    if row["family"] == "fjsp-synthetic":
        # No per-instance table exists for synthetic FJSP conversions (a fresh instance
        # per --fjsp-seed draw); out of scope here per Section 12 item 2 ("E8's primary
        # scope is JSSP-only, so this is low priority") -- left NaN, not silently
        # approximated by the multiplicative JSSP formula, which would be wrong.
        return float("nan")
    return float(row["num_jobs"]) * float(row["num_machines"])


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``num_operations``, ``ms_per_decision``, ``rpd_or_makespan``,
    ``uses_raw_makespan``, ``certified_optimal`` (Section 2.4). Returns a new
    DataFrame; does not mutate ``df``.
    """
    out = df.copy()
    out["num_operations"] = out.apply(_num_operations_row, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ms_per_decision"] = np.where(
            out["num_operations"] > 0,
            out["wall_clock_s"] / out["num_operations"] * 1000.0,
            np.nan,
        )
    rpd_is_null = out["rpd_pct"].isna()
    out["uses_raw_makespan"] = rpd_is_null
    out["rpd_or_makespan"] = np.where(rpd_is_null, out["makespan"], out["rpd_pct"])

    # Recomendacion 1/10 del tutor (T7.1): a row is "certified optimal" iff CP-SAT
    # itself reported status == "OPTIMAL" (equivalently mip_gap == 0.0), never merely
    # because a solution was found. mip_gap is only present on CSVs produced after this
    # column was added to run_evaluation.py's CSV_FIELDNAMES; older CSVs lack it and
    # get NaN here rather than failing schema validation (validate_schema only requires
    # REQUIRED_COLUMNS, which mip_gap is deliberately NOT part of, so pre-existing E1-E8
    # CSVs keep loading unchanged -- Regla 5, sin regresiones silenciosas).
    if "mip_gap" not in out.columns:
        out["mip_gap"] = np.nan
    out["certified_optimal"] = (out["status"] == "OPTIMAL") & (out["mip_gap"].fillna(0.0) == 0.0)
    return out


@dataclass(frozen=True)
class LoadedExperiment:
    """One experiment's rows plus provenance, for ``stats_manifest.json`` (Section 9.4)."""

    experiment_id: str
    paths: tuple[Path, ...]
    n_rows_loaded: int
    n_rows_infeasible_dropped: int


def load_results(
    experiment_csvs: dict[str, Path | list[Path]],
    *,
    skip_missing: bool = False,
) -> tuple[pd.DataFrame, list[LoadedExperiment]]:
    """Load every experiment's CSV(s), tag ``experiment_id``, validate, derive columns.

    Args:
        experiment_csvs: ``{"E1": path, "E2a": path, "E2b": path, ...}``. A value may be
            a single path or a list of paths (an experiment split across multiple CSVs,
            e.g. E1's seed-0 + RL-stochastic files, is concatenated before tagging).
        skip_missing: If ``True``, a missing path is skipped with a printed note
            (Section 11's ``--skip-missing``) instead of raising -- for running the
            pipeline incrementally as results land.

    Returns:
        ``(combined_df, provenance)``. ``combined_df`` has every required column plus
        the Section 2.4 derived columns and an ``experiment_id`` column. Rows with
        ``feasible == False`` are dropped (Section 2.1's validation gate: such a row
        must never enter a Wilcoxon/Friedman comparison) -- this is the ONLY row
        filtering this function performs; every other row is passed through.

    Raises:
        FileNotFoundError: If a path is missing and ``skip_missing`` is ``False``.
        DataContractError: If schema validation or the RL rollout convention fails for
            any experiment.
    """
    frames: list[pd.DataFrame] = []
    provenance: list[LoadedExperiment] = []

    for experiment_id, spec in experiment_csvs.items():
        paths = [spec] if isinstance(spec, (str, Path)) else list(spec)
        paths = [Path(p) for p in paths]
        existing = [p for p in paths if p.exists()]
        missing = [p for p in paths if not p.exists()]
        if missing:
            if not skip_missing:
                raise FileNotFoundError(f"{experiment_id}: missing CSV(s): {missing}")
            print(f"[stats.io] {experiment_id}: skipping, not yet run: {missing}")
        if not existing:
            continue

        parts = [pd.read_csv(p) for p in existing]
        df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        validate_schema(df, source=experiment_id)
        validate_rl_rollout_convention(df, source=experiment_id)

        n_before = len(df)
        df = df[df["feasible"].astype(bool)].reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"[stats.io] {experiment_id}: dropped {n_dropped} infeasible row(s)")

        df = df.copy()
        df["experiment_id"] = experiment_id
        frames.append(df)
        provenance.append(
            LoadedExperiment(
                experiment_id=experiment_id,
                paths=tuple(existing),
                n_rows_loaded=n_before,
                n_rows_infeasible_dropped=n_dropped,
            )
        )

    if not frames:
        return pd.DataFrame(columns=[*REQUIRED_COLUMNS, "experiment_id"]), provenance

    combined = pd.concat(frames, ignore_index=True)
    combined = add_derived_columns(combined)
    return combined, provenance
