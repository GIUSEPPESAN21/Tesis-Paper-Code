"""Golden regression test: :mod:`experiments.stats.report`'s H3 computation must
reproduce ``experiments/results/week24/compute_h3_verdict.py``'s already-published,
already-defensible H3 result (the exact numbers ``EXPERIMENT_LOG_WEEK24.md`` reports)
when run over the same Week 24 CSVs, via the general-purpose pipeline rather than the
one-off script that originally produced them.

``StatsAnalysis_TechSpec.md`` Section 10: pin against already-published numbers so a
disagreement is caught as an implementation bug, not discovered later in Chapter 6.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.stats.io import load_results
from experiments.stats.report import REGIMES, run_h3

WEEK24 = Path(__file__).resolve().parents[1] / "experiments" / "results" / "week24"
E1_CSV = WEEK24 / "e1_merged.csv"
PDR_DIR = WEEK24 / "e4e5e6"
RL_DIR = WEEK24 / "e4e5e6_rl"
GOLDEN_JSON = WEEK24 / "h3_verdict.json"

REGIME_FILES = {
    "E4_breakdowns_A0.95": ("e4_breakdowns_avail0.95.csv", "e4_breakdowns_avail0.95_rl.csv"),
    "E4_breakdowns_A0.90": ("e4_breakdowns_avail0.90.csv", "e4_breakdowns_avail0.90_rl.csv"),
    "E4_breakdowns_A0.80": ("e4_breakdowns_avail0.80.csv", "e4_breakdowns_avail0.80_rl.csv"),
    "E5_stochastic_s0.10": ("e5_stochastic_sigma0.1.csv", "e5_stochastic_sigma0.1_rl.csv"),
    "E5_stochastic_s0.25": ("e5_stochastic_sigma0.25.csv", "e5_stochastic_sigma0.25_rl.csv"),
    "E5_stochastic_s0.50": ("e5_stochastic_sigma0.5.csv", "e5_stochastic_sigma0.5_rl.csv"),
    "E6_arrivals_u0.70": ("e6_arrivals_util0.7.csv", "e6_arrivals_util0.7_rl.csv"),
    "E6_arrivals_u0.85": ("e6_arrivals_util0.85.csv", "e6_arrivals_util0.85_rl.csv"),
    "E6_arrivals_u0.95": ("e6_arrivals_util0.95.csv", "e6_arrivals_util0.95_rl.csv"),
}

pytestmark = pytest.mark.skipif(
    not E1_CSV.exists() or not GOLDEN_JSON.exists(),
    reason="Week 24 result CSVs not present in this checkout",
)


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def combined_h3_frame():
    csvs = {"E1": E1_CSV}
    for regime_label, (pdr_file, rl_file) in REGIME_FILES.items():
        csvs[regime_label] = [PDR_DIR / pdr_file, RL_DIR / rl_file]
    combined, _ = load_results(csvs)
    return combined


class TestH3GoldenAgainstWeek24:
    def test_regime_labels_match(self, golden: dict) -> None:
        golden_ids = {r["regime_id"] for r in golden["per_regime"]}
        pipeline_labels = {label for label, *_ in REGIMES}
        assert golden_ids == pipeline_labels

    def test_per_regime_p_values_and_best_pdr_match(self, combined_h3_frame, golden: dict) -> None:
        pipeline_verdicts = {v.raw["regime"]: v for v in run_h3(combined_h3_frame) if "regime" in v.raw}
        golden_by_id = {r["regime_id"]: r for r in golden["per_regime"]}

        assert set(pipeline_verdicts) == set(golden_by_id)
        for regime_id, golden_row in golden_by_id.items():
            pv = pipeline_verdicts[regime_id]
            assert pv.raw["best_pdr"] == golden_row["best_pdr"], regime_id
            assert pv.test_result.p_value == pytest.approx(golden_row["p_value"], rel=1e-3, abs=1e-9), regime_id
            assert pv.verdict == ("accept" if golden_row["accept_h3_this_regime"] else "reject"), regime_id

    def test_overall_verdict_matches(self, combined_h3_frame, golden: dict) -> None:
        all_verdicts = run_h3(combined_h3_frame)
        overall = next(v for v in all_verdicts if v.hypothesis == "H3 (overall)")
        expected_n = golden["n_regimes_accepted"]
        assert overall.raw["n_accept"] == expected_n
        expected_label = (
            "accept" if golden["overall_verdict"] == "ACCEPT H3"
            else ("partial" if golden["overall_verdict"] == "PARTIAL SUPPORT FOR H3" else "reject")
        )
        assert overall.verdict == expected_label
