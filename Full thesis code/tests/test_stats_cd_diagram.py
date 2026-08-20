"""Smoke tests for :mod:`experiments.stats.cd_diagram`."""

from __future__ import annotations

from pathlib import Path

from experiments.stats.cd_diagram import plot_cd_diagram


class TestPlotCdDiagram:
    def test_writes_a_nonempty_png(self, tmp_path: Path) -> None:
        average_ranks = {"rl": 1.5, "mwkr": 2.0, "spt": 4.5, "lpt": 4.0, "fifo": 3.0}
        out_path = tmp_path / "cd.png"
        result = plot_cd_diagram(average_ranks, cd=0.8, title="Test diagram", output_path=out_path)
        assert result == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        out_path = tmp_path / "nested" / "dir" / "cd.png"
        plot_cd_diagram({"a": 1.0, "b": 2.0}, cd=0.5, title="t", output_path=out_path)
        assert out_path.exists()

    def test_handles_two_methods(self, tmp_path: Path) -> None:
        out_path = tmp_path / "cd2.png"
        plot_cd_diagram({"a": 1.0, "b": 2.0}, cd=5.0, title="all in one group", output_path=out_path)
        assert out_path.exists()
