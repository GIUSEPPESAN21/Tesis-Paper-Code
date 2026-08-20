"""CLI entrypoint. ``StatsAnalysis_TechSpec.md`` Section 11.

    python -m experiments.stats.cli \\
        --results-dir experiments/results \\
        --experiments E1 E2a E2b E3 E4 E5 E6 E7 E8 \\
        --out-dir experiments/results/stats \\
        [--skip-missing]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.stats.io import load_results
from experiments.stats.report import REGIMES, generate_report


def default_experiment_csvs(results_dir: Path) -> dict[str, list[Path]]:
    """The Week 24/25 CSV layout this thesis's own runs actually produced."""
    week24 = results_dir / "week24"
    week25 = results_dir / "week25"
    e4e5e6 = week24 / "e4e5e6"
    e4e5e6_rl = week24 / "e4e5e6_rl"
    dynamic = results_dir / "dynamic"

    csvs: dict[str, list[Path]] = {
        "E1": [week24 / "e1_seed0_baselines.csv", week24 / "e1_rl_stochastic.csv"],
        "E2a": [week24 / "e2_armA_synth_seed0.csv", week24 / "e2_armA_synth_stochastic.csv"],
        "E2b": [week24 / "e2_armB_mk_seed0.csv", week24 / "e2_armB_mk_stochastic.csv"],
        "E3": [week24 / "e3_generalization_la21_la40.csv"],
        "E8": [week25 / "e8_clean" / "e1_e8_clean_seed0_baselines.csv", week25 / "e8_clean" / "e1_e8_clean_rl_stochastic.csv"],
    }
    for axis, avails in (("E4", ("0.95", "0.90", "0.80")),):
        for level in avails:
            csvs[f"E4_breakdowns_A{level}"] = [
                e4e5e6 / f"e4_breakdowns_avail{level}.csv",
                e4e5e6_rl / f"e4_breakdowns_avail{level}_rl.csv",
            ]
    for level in ("0.1", "0.25", "0.5"):
        label = {"0.1": "0.10", "0.25": "0.25", "0.5": "0.50"}[level]
        csvs[f"E5_stochastic_s{label}"] = [
            e4e5e6 / f"e5_stochastic_sigma{level}.csv",
            e4e5e6_rl / f"e5_stochastic_sigma{level}_rl.csv",
        ]
    for level in ("0.7", "0.85", "0.95"):
        # experiment_id tag must be zero-padded to match REGIMES's labels exactly
        # (e.g. "E6_arrivals_u0.70", not "u0.7") -- the on-disk filename stays
        # unpadded (util0.7.csv), only the tag used for report.py's regime lookup
        # needs to agree with REGIMES, the same way the E5 loop above already does.
        label = {"0.7": "0.70", "0.85": "0.85", "0.95": "0.95"}[level]
        csvs[f"E6_arrivals_u{label}"] = [
            e4e5e6 / f"e6_arrivals_util{level}.csv",
            e4e5e6_rl / f"e6_arrivals_util{level}_rl.csv",
        ]
    for kappa in ("0.5", "1.0"):
        for regime_label, _disruption, _level, _candidates in REGIMES:
            csvs[f"E7_kappa{kappa}_{regime_label}"] = [dynamic / f"eval_kappa{kappa}_{regime_label}.csv"]
        csvs[f"E7_kappa{kappa}_nominal"] = [dynamic / f"eval_kappa{kappa}_nominal.csv"]

    return csvs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "experiments" / "results")
    parser.add_argument("--experiments", nargs="*", default=None, help="subset of experiment ids to load; default: all known")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-missing", action="store_true", help="skip not-yet-run experiments instead of failing")
    args = parser.parse_args(argv)

    out_dir = args.out_dir or (args.results_dir / "stats")
    all_csvs = default_experiment_csvs(args.results_dir)
    if args.experiments:
        wanted = set(args.experiments)
        csvs = {k: v for k, v in all_csvs.items() if k in wanted or k.split("_")[0] in wanted}
    else:
        csvs = all_csvs

    combined, provenance = load_results(csvs, skip_missing=args.skip_missing)
    if combined.empty:
        print("[stats.cli] no data loaded; nothing to report.")
        return 1

    result = generate_report(combined, provenance, out_dir)
    print(f"[stats.cli] wrote {len(result['manifest']['tables'])} table(s), "
          f"{len(result['manifest']['figures'])} figure(s) to {out_dir}")
    for v in result["verdicts"]:
        print(f"  {v.hypothesis:<28} {v.verdict.upper()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
