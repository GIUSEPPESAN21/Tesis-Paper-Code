"""Tests for scripts/generate_thesis_figures.py's Fig_holgura_vs_RI (T1.4).

Only exercises the graceful-skip paths (empty inputs) -- rendering a real figure with
synthetic-but-plausible data is covered manually when the pipeline actually runs;
these tests guard against a crash on missing/partial data, which is the failure mode
most likely to bite mid-Semana-24 when only some regimes have landed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_thesis_figures import generate_fig_holgura_vs_ri  # noqa: E402


class TestGenerateFigHolguraVsRi:
    def test_empty_slack_df_returns_no_files(self, tmp_path: Path) -> None:
        empty_slack = pd.DataFrame(columns=["instance", "method", "slack_ratio"])
        combined = pd.DataFrame(columns=["experiment_id", "instance", "method", "makespan", "disruption", "seed"])
        written = generate_fig_holgura_vs_ri(empty_slack, combined, tmp_path)
        assert written == []

    def test_missing_e1_data_returns_no_files(self, tmp_path: Path) -> None:
        slack_df = pd.DataFrame({"instance": ["ft06"], "method": ["fifo"], "slack_ratio": [0.1]})
        combined = pd.DataFrame(columns=["experiment_id", "instance", "method", "makespan", "disruption", "seed"])
        written = generate_fig_holgura_vs_ri(slack_df, combined, tmp_path)
        assert written == []
