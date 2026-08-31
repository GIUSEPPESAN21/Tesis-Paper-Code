"""Critical-difference diagrams (Demsar 2006 style). ``StatsAnalysis_TechSpec.md`` Section 8."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_cd_diagram(
    average_ranks: dict[str, float],
    cd: float,
    *,
    title: str,
    output_path: Path,
) -> Path:
    """Demsar-style CD diagram: a horizontal axis of ranks, one tick per method, with a
    bracket connecting methods NOT significantly different from each other (rank
    difference <= cd).

    Args:
        average_ranks: method -> mean rank, from ``FriedmanResult``.
        cd: From :func:`~experiments.stats.omnibus.critical_difference`.
        title: e.g. ``"E3 -- la21-la25 tier (H2)"`` -- experiment + hypothesis, always,
            so a reader flipping through a folder of PNGs can identify each one without
            opening it.
        output_path: PNG, 300dpi minimum (thesis print quality).

    Returns:
        ``output_path``, unchanged, for chaining into ``report.py``'s manifest.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    methods = sorted(average_ranks, key=lambda m: average_ranks[m])
    ranks = [average_ranks[m] for m in methods]
    k = len(methods)

    fig, ax = plt.subplots(figsize=(max(6, k * 0.9), 2.5 + 0.3 * k))
    lo, hi = min(ranks) - 0.5, max(ranks) + 0.5
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.axhline(0.85, color="black", lw=1)
    ax.get_yaxis().set_visible(False)
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)

    for rank, method in zip(ranks, methods):
        ax.plot([rank, rank], [0.85, 0.90], color="black", lw=1)
        ax.text(rank, 0.94, f"{method}\n({rank:.2f})", ha="center", va="bottom", fontsize=8)

    # CD reference bar, drawn above the axis.
    cd_y = 0.55
    ax.plot([lo + 0.1, lo + 0.1 + cd], [cd_y, cd_y], color="black", lw=1.5)
    ax.text(lo + 0.1 + cd / 2, cd_y + 0.03, f"CD={cd:.3f}", ha="center", fontsize=8)

    # Group brackets: chain overlapping cliques of methods within `cd` of each other,
    # sorted by rank -- the standard Demsar-diagram construction.
    order = np.argsort(ranks)
    sorted_ranks = np.array(ranks)[order]
    sorted_methods = [methods[i] for i in order]
    groups: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(sorted_ranks)):
        if sorted_ranks[i] - sorted_ranks[start] > cd:
            if i - 1 > start:
                groups.append((start, i - 1))
            start = i
    if len(sorted_ranks) - 1 > start:
        groups.append((start, len(sorted_ranks) - 1))

    bracket_y = 0.72
    for lo_i, hi_i in groups:
        ax.plot(
            [sorted_ranks[lo_i], sorted_ranks[hi_i]],
            [bracket_y, bracket_y],
            color="black",
            lw=3,
            solid_capstyle="butt",
        )
        bracket_y -= 0.06

    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
