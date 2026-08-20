"""Descriptive statistics (mean/std) per (instance-or-family, method[, disruption level])
for a run_evaluation.py CSV. STEP4 of the Week 24 task brief explicitly wants descriptive
stats only -- no significance testing (that's next month's dedicated script) -- so this
is intentionally a pandas groupby, not a stats-test pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def summarize(csv_path: Path, group_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["rpd_pct"] = pd.to_numeric(df["rpd_pct"], errors="coerce")
    agg = df.groupby(group_cols).agg(
        n=("makespan", "size"),
        mean_makespan=("makespan", "mean"),
        std_makespan=("makespan", "std"),
        mean_rpd_pct=("rpd_pct", "mean"),
        std_rpd_pct=("rpd_pct", "std"),
        mean_wall_clock_s=("wall_clock_s", "mean"),
        pct_feasible=("feasible", lambda s: 100.0 * s.mean()),
    )
    return agg.round(3)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python descriptive_stats.py <csv> [group_col ...]")
    csv_path = Path(sys.argv[1])
    group_cols = sys.argv[2:] or ["method"]
    result = summarize(csv_path, group_cols)
    print(result.to_string())
    out_path = csv_path.with_name(csv_path.stem + "_descriptive.csv")
    result.to_csv(out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
