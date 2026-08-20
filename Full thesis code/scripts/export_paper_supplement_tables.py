"""Two supplementary tables for the IEEE conference paper, English, derived
from already-exported/already-cited results -- no independent recomputation.

Table S1: full per-method RPD (mean +/- SD, JSSP arm, N=43), i.e. every one of
the 9 PDR individually plus GA/CP-SAT/RL -- the paper's Table I and Section
V-A only report "best PDR per instance" and the RL/GA/CP-SAT aggregates, so a
reviewer asking "which specific dispatching rule is that, and how do the
others compare" has no table to point to. Derived by a pure column-mean/SD
reduction of ``results/tablas/Tabla_6.2.1_rpd_jssp_e1.csv`` (already the
thesis's own cited per-instance RPD table for E1) -- not recomputed from raw
CSVs.

Table S2: the H5 la40 measurement-anomaly sensitivity check that Chapter 6
Section 6.5.1 already performed narratively (Chapter6_FINAL.md) but the paper
draft only summarizes in one sentence (Section V-D). Reviewers of a
latency/robustness claim routinely ask for exactly this: what changes if the
one outlier is corrected. Sourced directly from
``results/tablas/Tabla_6.5.1_latencia_e8.csv``'s la38/la40 rows (same
provenance chain as Chapter 6) plus the single already-computed substitution
value (la40's CP-SAT wall-clock replaced by la38's, same size/difficulty
class, per Chapter6_FINAL.md Sec 6.5.1) -- again a transcription, not a new
computation.

Usage
-----
    python scripts/export_paper_supplement_tables.py [--out-dir ../figures/paper]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

TABLAS_DIR = REPO_ROOT.parent / "results" / "tablas"

PDR_ENGLISH_NOTE = {
    "spt": "Shortest Processing Time", "lpt": "Longest Processing Time",
    "mwkr": "Most Work Remaining", "lwkr": "Least Work Remaining",
    "mor": "Most Operations Remaining", "lor": "Least Operations Remaining",
    "fifo": "First In First Out", "edd": "Earliest Due Date", "cr": "Critical Ratio",
    "ga": "Genetic Algorithm", "cpsat": "CP-SAT (exact)", "rl": "RL (this work)",
}


def _write(df: pd.DataFrame, out_dir: Path, stem: str, caption: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    md_path = out_dir / f"{stem}.md"
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    md_path.write_text(f"<!-- {caption} -->\n\n{header}\n{sep}\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return [csv_path, md_path]


def build_table_s1(out_dir: Path) -> list[Path]:
    src = TABLAS_DIR / "Tabla_6.2.1_rpd_jssp_e1.csv"
    e1 = pd.read_csv(src)
    method_cols = [c for c in e1.columns if c != "instance"]

    rows = []
    for method in method_cols:
        vals = e1[method].astype(float)
        rows.append({
            "method": method.upper(),
            "type": "RL policy" if method == "rl" else ("Exact/metaheuristic" if method in ("ga", "cpsat") else "PDR"),
            "description": PDR_ENGLISH_NOTE.get(method, method),
            "mean_rpd_pct": round(float(vals.mean()), 2),
            "sd_rpd_pct": round(float(vals.std(ddof=1)), 2),
            "median_rpd_pct": round(float(vals.median()), 2),
            "n": len(vals),
        })
    df = pd.DataFrame(rows).sort_values("mean_rpd_pct").reset_index(drop=True)
    caption = ("Table S1. Full per-method RPD (%) on the JSSP arm (E1, N=43 instances, mean over "
               "seed-0 deterministic evaluation), all 9 PDR individually plus GA/CP-SAT/RL. Derived "
               "by column mean/SD of results/tablas/Tabla_6.2.1_rpd_jssp_e1.csv -- same numbers "
               "already cited in Chapter 6/7 of the thesis, reduced across instances, not recomputed.")
    return _write(df, out_dir, "Table_S1_rpd_full_by_method_jssp", caption)


def build_table_s2(out_dir: Path) -> list[Path]:
    src = TABLAS_DIR / "Tabla_6.5.1_latencia_e8.csv"
    e8 = pd.read_csv(src)
    e8.columns = [c.strip() for c in e8.columns]
    la38 = e8[e8["Instancia"] == "la38"].iloc[0] if "Instancia" in e8.columns else e8[e8.iloc[:, 0] == "la38"].iloc[0]
    la40 = e8[e8["Instancia"] == "la40"].iloc[0] if "Instancia" in e8.columns else e8[e8.iloc[:, 0] == "la40"].iloc[0]

    col = list(e8.columns)
    ms_rl_col, ms_cpsat_col, ratio_col, pass_col = col[2], col[3], col[4], col[5]

    la40_ms_rl = float(la40[ms_rl_col])
    la40_ms_cpsat_raw = float(la40[ms_cpsat_col])
    la40_ratio_raw = float(la40[ratio_col])
    la38_ms_cpsat = float(la38[ms_cpsat_col])
    la40_ratio_corrected = la38_ms_cpsat / la40_ms_rl

    rows = [
        {
            "reading": "Raw (as measured)",
            "la40_ms_per_decision_rl": round(la40_ms_rl, 3),
            "la40_ms_per_decision_cpsat": round(la40_ms_cpsat_raw, 3),
            "cpsat_over_rl_ratio": round(la40_ratio_raw, 2),
            "passes_1pct_threshold": bool(la40_ratio_raw >= 100),
            "h5_instances_passing_of_43": 1,
        },
        {
            "reading": "Sensitivity-corrected (la40 CP-SAT wall-clock substituted with la38's, same 15x15/225-op size class, both FEASIBLE under the identical 60s budget)",
            "la40_ms_per_decision_rl": round(la40_ms_rl, 3),
            "la40_ms_per_decision_cpsat": round(la38_ms_cpsat, 3),
            "cpsat_over_rl_ratio": round(la40_ratio_corrected, 2),
            "passes_1pct_threshold": bool(la40_ratio_corrected >= 100),
            "h5_instances_passing_of_43": 0,
        },
    ]
    df = pd.DataFrame(rows)
    caption = ("Table S2. H5 sensitivity check for the la40 CP-SAT measurement anomaly (raw wall-clock "
               "7216.67s vs. la38's 60.0554s on an identical-size, identical-status instance under the "
               "same 60s budget -- see thesis Chapter 6 Sec. 6.5.1 for the full anomaly audit). Under "
               "either reading H5 is rejected (0/43 or 1/43 pass the strict all-instances threshold); "
               "the correction makes the rejection more decisive, not less -- reported here so the "
               "paper's one-sentence summary (Sec. V-D) is independently checkable.")
    return _write(df, out_dir, "Table_S2_h5_la40_sensitivity", caption)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT.parent / "figures" / "paper")
    args = parser.parse_args(argv)

    written: list[Path] = []
    written += build_table_s1(args.out_dir)
    written += build_table_s2(args.out_dir)

    print(f"[export_paper_supplement_tables] wrote {len(written)} files to {args.out_dir}")
    for p in sorted(written):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
