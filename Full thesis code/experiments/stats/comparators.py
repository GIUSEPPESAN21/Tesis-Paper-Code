"""Recomendaciones 6/7 del tutor (Fase 4, T4.1/T4.2): comparadores PDR bien separados.

`report.py`'s H1/H3 accept/reject computation already implements a single "best PDR"
convention (lowest RPD/RI *per instance*, i.e. the oracle) and this module does **not**
change that -- `PreRegistration_FINAL.md`'s frozen decision rules are untouched (Regla
1). What this module adds is a set of *descriptive*, clearly-separated comparators that
were previously conflated under the single phrase "best PDR", so a reader of Chapter 6
can tell apart:

- :func:`pdr_global` -- one fixed rule, chosen by lowest MEAN RPD over the whole
  instance set. The only comparator actually deployable a priori (a real scheduler has
  to pick one rule before seeing which instance shows up).
- :func:`pdr_por_familia` -- best fixed rule per benchmark family (fisher-thompson,
  lawrence, brandimarte). Still deployable a priori, if the family is known in advance.
- :func:`oracle_best_per_instance` -- minimum RPD per instance, allowed to pick a
  *different* rule for every instance. This is an ORACLE, not a policy: every public
  name (column, legend, caption) this module produces for it says "oracle" explicitly,
  per T4.1's instruction, so it can never be quoted as if it were a deployable
  comparator.

T4.2 additionally separates H2's two readings of "does RL generalize" so acceptance
under one reading is never silently presented as acceptance under both:

- **Generalizacion relativa de rango**: Friedman-Nemenyi, RL not significantly worse
  than the best PDR's *average rank* (already computed in ``report.py::run_h2``).
- **Generalizacion competitiva absoluta**: RL's raw mean RPD vs. :func:`pdr_global`'s
  raw mean RPD, per tier, in percentage points -- :func:`h2_absolute_competitive_table`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FAMILY_DISPLAY_NAMES: dict[str, str] = {
    "fisher-thompson": "Fisher-Thompson",
    "lawrence": "Lawrence",
    "brandimarte": "Brandimarte",
}


@dataclass(frozen=True)
class PdrGlobalResult:
    """The single fixed PDR with lowest mean RPD over the scored instance set."""

    method: str
    mean_rpd_pct: float
    n_instances: int


def pdr_global(e1_df: pd.DataFrame, *, candidates: tuple[str, ...]) -> PdrGlobalResult:
    """The one rule deployable a priori: lowest mean RPD across every instance.

    Args:
        e1_df: E1 rows (one per instance/method), already restricted by the caller to
            the deterministic seed and disruption=="none" slice it wants to score.
        candidates: PDR names eligible for selection (usually the nine Part B6 rules).

    Raises:
        ValueError: If no row in ``e1_df`` matches ``candidates``.
    """
    subset = e1_df[e1_df["method"].isin(candidates)]
    if subset.empty:
        raise ValueError(f"no rows for candidates={candidates!r} in the given frame")
    means = subset.groupby("method")["rpd_pct"].mean()
    best = means.idxmin()
    n_instances = int(subset.loc[subset["method"] == best, "instance"].nunique())
    return PdrGlobalResult(method=str(best), mean_rpd_pct=float(means[best]), n_instances=n_instances)


@dataclass(frozen=True)
class PdrPorFamiliaResult:
    """The best fixed PDR within one benchmark family."""

    family: str
    method: str
    mean_rpd_pct: float
    n_instances: int


def pdr_por_familia(e1_df: pd.DataFrame, *, candidates: tuple[str, ...]) -> list[PdrPorFamiliaResult]:
    """Best fixed PDR per benchmark family (fisher-thompson/lawrence/brandimarte).

    Still a deployable-a-priori comparator (picking a rule once you know the family),
    distinct from both :func:`pdr_global` (one rule for everything) and
    :func:`oracle_best_per_instance` (a different rule per instance, an oracle).
    """
    results: list[PdrPorFamiliaResult] = []
    subset = e1_df[e1_df["method"].isin(candidates)]
    for family, group in subset.groupby("family"):
        means = group.groupby("method")["rpd_pct"].mean()
        if means.empty:
            continue
        best = means.idxmin()
        n_instances = int(group.loc[group["method"] == best, "instance"].nunique())
        results.append(
            PdrPorFamiliaResult(
                family=str(family), method=str(best), mean_rpd_pct=float(means[best]), n_instances=n_instances
            )
        )
    return results


def oracle_best_per_instance(e1_df: pd.DataFrame, *, candidates: tuple[str, ...]) -> pd.DataFrame:
    """Minimum RPD per instance among ``candidates`` -- an ORACLE, never a policy.

    Column names are deliberately prefixed ``oracle_`` (T4.1: "debe etiquetarse como
    oraculo y no como politica, en el nombre de la columna, en la leyenda de toda
    figura y en el pie de toda tabla") so a copy-paste into a figure legend or table
    caption cannot silently drop the disclaimer.

    Returns:
        One row per instance: ``instance``, ``oracle_winning_pdr``,
        ``oracle_best_per_instance_rpd_pct``.

    Raises:
        ValueError: If no row in ``e1_df`` matches ``candidates``.
    """
    subset = e1_df[e1_df["method"].isin(candidates)]
    if subset.empty:
        raise ValueError(f"no rows for candidates={candidates!r} in the given frame")
    idx = subset.groupby("instance")["rpd_pct"].idxmin()
    winners = subset.loc[idx, ["instance", "method", "rpd_pct"]].rename(
        columns={"method": "oracle_winning_pdr", "rpd_pct": "oracle_best_per_instance_rpd_pct"}
    )
    return winners.reset_index(drop=True).sort_values("instance").reset_index(drop=True)


def comparator_gap_summary(e1_df: pd.DataFrame, *, candidates: tuple[str, ...]) -> dict:
    """T4.1's headline: the pp gap between :func:`pdr_global` and the oracle, and how
    many distinct rules the oracle actually needs.

    Reference values this should reproduce on E1 (43 instances, seed 0), per the
    tutor's own measurement: ``pdr_global`` = ``fifo`` at 19.15% mean RPD; oracle at
    18.96% mean RPD, using 3 distinct rules (fifo, mor, mwkr); gap = 0.19pp -- small
    enough to report explicitly as evidence that H1's rejection does not hinge on which
    of the three comparators (global/per-familia/oracle) is used.
    """
    glob = pdr_global(e1_df, candidates=candidates)
    oracle_df = oracle_best_per_instance(e1_df, candidates=candidates)
    oracle_mean = float(oracle_df["oracle_best_per_instance_rpd_pct"].mean())
    return {
        "pdr_global_method": glob.method,
        "pdr_global_mean_rpd_pct": glob.mean_rpd_pct,
        "oracle_mean_rpd_pct": oracle_mean,
        "gap_pp": glob.mean_rpd_pct - oracle_mean,
        "oracle_n_distinct_pdrs": int(oracle_df["oracle_winning_pdr"].nunique()),
        "oracle_winning_pdrs": sorted(oracle_df["oracle_winning_pdr"].unique().tolist()),
    }


def tabla_comparadores_pdr_markdown(e1_df: pd.DataFrame, *, candidates: tuple[str, ...]) -> str:
    """T4.1's table: the three comparators, always separate, oracle always labeled."""
    glob = pdr_global(e1_df, candidates=candidates)
    familias = pdr_por_familia(e1_df, candidates=candidates)
    gap = comparator_gap_summary(e1_df, candidates=candidates)

    lines = [
        "| Comparador | Regla(s) | RPD medio (%) | N instancias |",
        "|---|---|---|---|",
        f"| pdr_global (unica regla, desplegable a priori) | {glob.method} | {glob.mean_rpd_pct:.4f} | {glob.n_instances} |",
    ]
    for f in familias:
        family_name = FAMILY_DISPLAY_NAMES.get(f.family, f.family)
        lines.append(
            f"| pdr_por_familia -- {family_name} | {f.method} | {f.mean_rpd_pct:.4f} | {f.n_instances} |"
        )
    lines.append(
        f"| **oraculo_best_per_instance** (NO es una politica desplegable; usa "
        f"{gap['oracle_n_distinct_pdrs']} reglas distintas: {', '.join(gap['oracle_winning_pdrs'])}) "
        f"| (varia por instancia) | {gap['oracle_mean_rpd_pct']:.4f} | -- |"
    )
    lines.append("")
    lines.append(
        f"Brecha pdr_global vs. oraculo: **{gap['gap_pp']:+.4f} pp** -- una brecha pequena indica que el "
        "veredicto de H1 no depende de cual de los tres comparadores se use."
    )
    return "\n".join(lines)


def h2_absolute_competitive_table(
    e3_df: pd.DataFrame, tiers: dict[str, tuple[str, ...]], *, pdr_global_method: str
) -> pd.DataFrame:
    """T4.2's second reading of H2: RL's raw mean RPD vs. ``pdr_global``'s raw mean
    RPD, per tier, in percentage points -- the "generalizacion competitiva absoluta"
    reading, reported alongside (never instead of) the rank-based reading already
    computed by ``report.py::run_h2``.

    Reference values this should reproduce (tutor's own measurement, RL - FIFO gap):
    +6.21pp (1.5-2x), +7.57pp (2.0x), +11.13pp (2.25x), +5.63pp (3.0x).

    Returns:
        One row per tier: ``tier``, ``rl_mean_rpd_pct``, ``pdr_global_mean_rpd_pct``,
        ``gap_pp`` (RL minus pdr_global -- positive means RL is worse in absolute RPD).
    """
    rows = []
    for tier_label, instances in tiers.items():
        tier_df = e3_df[e3_df["instance"].isin(instances)]
        rl_rpd = tier_df.loc[tier_df["method"] == "rl", "rpd_pct"]
        pdr_rpd = tier_df.loc[tier_df["method"] == pdr_global_method, "rpd_pct"]
        if rl_rpd.empty or pdr_rpd.empty:
            continue
        rl_mean = float(rl_rpd.mean())
        pdr_mean = float(pdr_rpd.mean())
        rows.append(
            {
                "tier": tier_label,
                "rl_mean_rpd_pct": rl_mean,
                "pdr_global_mean_rpd_pct": pdr_mean,
                "gap_pp": rl_mean - pdr_mean,
            }
        )
    return pd.DataFrame(rows)


def h2_dual_reading_markdown(
    h2_absolute_df: pd.DataFrame,
    rank_reading_by_tier: dict[str, str],
    *,
    pdr_global_method: str,
) -> str:
    """Merge the rank-based and absolute-competitive readings into one table (T4.2:
    "las dos lecturas juntas en la misma tabla para que la aceptacion de H2 no pueda
    leerse como competitividad absoluta").

    Args:
        h2_absolute_df: Output of :func:`h2_absolute_competitive_table`.
        rank_reading_by_tier: ``{tier_label: free-text rank-based verdict summary}``,
            e.g. from ``report.py``'s per-tier H2 ``HypothesisVerdict.notes``.
    """
    lines = [
        f"| Tier | Lectura de rango (Friedman-Nemenyi) | RL RPD medio (%) | "
        f"pdr_global ({pdr_global_method}) RPD medio (%) | Brecha RL-pdr_global (pp) |",
        "|---|---|---|---|---|",
    ]
    for _, row in h2_absolute_df.iterrows():
        rank_note = rank_reading_by_tier.get(row["tier"], "(no disponible)")
        lines.append(
            f"| {row['tier']} | {rank_note} | {row['rl_mean_rpd_pct']:.4f} | "
            f"{row['pdr_global_mean_rpd_pct']:.4f} | {row['gap_pp']:+.4f} |"
        )
    lines.append("")
    lines.append(
        "**Lectura correcta:** la aceptacion de H2 (columna de rango) significa que RL no es "
        "significativamente peor que la mejor PDR en rango promedio -- NO significa que RL tenga "
        "menor RPD absoluta. La columna de brecha (pp) es la lectura complementaria que evita esa "
        "confusion; ambas deben citarse juntas, nunca una sin la otra (T4.2)."
    )
    return "\n".join(lines)
