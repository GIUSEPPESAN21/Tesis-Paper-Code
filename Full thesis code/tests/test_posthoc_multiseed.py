"""Tests for experiments/stats/posthoc_multiseed.py (Fase 2, T2.4)."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.stats.posthoc_multiseed import (
    POSTHOC_HEADER,
    load_summary_field,
    tabla_h4_multisemilla_posthoc_markdown,
)


def _write_summary(path: Path, **fields) -> Path:
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


class TestLoadSummaryField:
    def test_reads_field_from_each_file(self, tmp_path: Path) -> None:
        p1 = _write_summary(tmp_path / "s0.json", last_100_mean_makespan=100.0)
        p2 = _write_summary(tmp_path / "s1.json", last_100_mean_makespan=110.0)
        values = load_summary_field([p1, p2], "last_100_mean_makespan")
        assert values == [100.0, 110.0]

    def test_skips_missing_field(self, tmp_path: Path) -> None:
        p1 = _write_summary(tmp_path / "s0.json", last_100_mean_makespan=100.0)
        p2 = _write_summary(tmp_path / "s1.json", other_field=1)
        values = load_summary_field([p1, p2], "last_100_mean_makespan")
        assert values == [100.0]

    def test_skips_null_field(self, tmp_path: Path) -> None:
        p1 = _write_summary(tmp_path / "s0.json", last_100_mean_makespan=None)
        assert load_summary_field([p1], "last_100_mean_makespan") == []


class TestTablaH4MultisemillaPosthocMarkdown:
    def test_pendiente_corrida_when_no_files_match(self) -> None:
        md = tabla_h4_multisemilla_posthoc_markdown({0.5: "/nonexistent/path/*.json"})
        assert "PENDIENTE-CORRIDA" in md
        assert POSTHOC_HEADER in md

    def test_real_aggregate_when_files_present(self, tmp_path: Path) -> None:
        for i, val in enumerate([100.0, 110.0, 105.0]):
            _write_summary(tmp_path / f"summary_kappa0.5_seed{i}.json", last_100_mean_makespan=val)
        pattern = str(tmp_path / "summary_kappa0.5_seed*.json")
        md = tabla_h4_multisemilla_posthoc_markdown({0.5: pattern})
        assert "PENDIENTE-CORRIDA" not in md
        assert "n=3" in md

    def test_header_always_present_and_labeled_posthoc(self) -> None:
        md = tabla_h4_multisemilla_posthoc_markdown({0.0: "/nonexistent/*.json"})
        assert "POST-HOC" in md
        assert "PreRegistration_FINAL.md" in md
