"""Build E3's CSV (size generalization, la21-la40) by extracting rows already produced
by E1's full 43-instance sweep, per FinalExperimentMatrix.md Sec 2.3/2.8: E3 reuses E1's
PDR + RL-deterministic rows for la21-la40 rather than re-running them. Also cross-checks
the 5 instances that already had a number from HPO_ABLATION_REPORT_WEEK21.md's zero-shot
table (la21=35.6%, la25=27.4%, la30=19.9%, la35=26.1%, la40=21.4%) against this run's RL
row for the same instance, as a sanity check that the checkpoint/eval path is consistent.
"""

from __future__ import annotations

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent
E1_CSV = RESULTS_DIR / "e1_seed0_baselines.csv"
OUT_CSV = RESULTS_DIR / "e3_generalization_la21_la40.csv"

TIER_INSTANCES = [f"la{n:02d}" for n in range(21, 41)]

#: Week 21's own zero-shot generalization numbers, for the cross-check (RPD %).
WEEK21_KNOWN = {"la21": 35.6, "la25": 27.4, "la30": 19.9, "la35": 26.1, "la40": 21.4}


def main() -> None:
    with E1_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    e3_rows = [r for r in rows if r["instance"] in TIER_INSTANCES]
    if not e3_rows:
        raise SystemExit(f"no rows found for {TIER_INSTANCES} in {E1_CSV} -- has E1 finished?")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(e3_rows[0].keys()))
        writer.writeheader()
        writer.writerows(e3_rows)
    print(f"wrote {len(e3_rows)} rows to {OUT_CSV}")

    print("\nCross-check vs Week 21 zero-shot RL numbers:")
    for name, week21_rpd in WEEK21_KNOWN.items():
        match = [r for r in e3_rows if r["instance"] == name and r["method"] == "rl"]
        if not match:
            print(f"  {name}: NO RL ROW FOUND")
            continue
        this_rpd = float(match[0]["rpd_pct"])
        delta = this_rpd - week21_rpd
        flag = "OK" if abs(delta) < 5.0 else "CHECK -- large discrepancy"
        print(f"  {name}: week21={week21_rpd:.1f}%  week24={this_rpd:.2f}%  delta={delta:+.2f}pp  [{flag}]")

    missing = sorted(set(TIER_INSTANCES) - {r["instance"] for r in e3_rows})
    if missing:
        print(f"\nWARNING: missing instances in E1 output: {missing}")

    methods_present = sorted({r["method"] for r in e3_rows})
    print(f"\nMethods present: {methods_present}")


if __name__ == "__main__":
    main()
