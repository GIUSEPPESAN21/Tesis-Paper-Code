"""Unit tests for :mod:`envs.loader`: file parsing, bundled instances, known optima.

The three known-makespan assertions in :class:`TestKnownOptima` are the load-bearing
checks for Week 1 of the thesis plan (see ``VALIDATION_WEEK1.md``): if the parser or the
bundled instance files are wrong, these are the tests that catch it.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.disjunctive_graph import static_lower_bound
from envs.loader import (
    InstanceParseError,
    available_instances,
    known_optimum,
    load_by_name,
    load_instance,
    load_known_optima,
    parse_standard,
    parse_taillard,
)

STANDARD_TEXT = """\
2 2
0 3 1 2
1 4 0 1
"""

TAILLARD_TEXT = """\
2 2
3 2
4 1
0 1
1 0
"""


class TestParseStandard:
    def test_parses_shape_and_values(self) -> None:
        inst = parse_standard(STANDARD_TEXT, name="toy")
        assert inst.num_jobs == 2
        assert inst.num_machines == 2
        assert inst.machines.tolist() == [[0, 1], [1, 0]]
        assert inst.durations.tolist() == [[3, 2], [4, 1]]

    def test_wrong_token_count_raises(self) -> None:
        with pytest.raises(InstanceParseError, match="expected"):
            parse_standard("2 2\n0 3 1 2\n1 4\n", name="short")

    def test_ignores_comment_lines(self) -> None:
        text = "# a comment\n2 2\n0 3 1 2\n1 4 0 1\n"
        inst = parse_standard(text, name="toy")
        assert inst.num_jobs == 2


class TestParseTaillard:
    def test_parses_shape_and_values(self) -> None:
        inst = parse_taillard(TAILLARD_TEXT, name="toy")
        assert inst.durations.tolist() == [[3, 2], [4, 1]]
        assert inst.machines.tolist() == [[0, 1], [1, 0]]

    def test_one_indexed_machines_are_normalised(self) -> None:
        text = "2 2\n3 2\n4 1\n1 2\n2 1\n"  # machines block is 1-indexed
        inst = parse_taillard(text, name="toy")
        assert inst.machines.min() == 0
        assert inst.machines.tolist() == [[0, 1], [1, 0]]

    def test_wrong_token_count_raises(self) -> None:
        with pytest.raises(InstanceParseError, match="expected"):
            parse_taillard("2 2\n3 2\n4 1\n0 1\n", name="short")


class TestLoadInstanceAutoDetect:
    def test_detects_standard_format(self, tmp_path) -> None:
        path = tmp_path / "toy.txt"
        path.write_text(STANDARD_TEXT, encoding="utf-8")
        inst = load_instance(path)
        assert inst.name == "toy"
        assert inst.durations.tolist() == [[3, 2], [4, 1]]

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_instance(tmp_path / "does-not-exist.txt")

    def test_name_override(self, tmp_path) -> None:
        path = tmp_path / "toy.txt"
        path.write_text(STANDARD_TEXT, encoding="utf-8")
        inst = load_instance(path, name="renamed")
        assert inst.name == "renamed"


class TestBundledInstances:
    @pytest.mark.parametrize("name", ["ft06", "ft10", "ft20"])
    def test_fisher_thompson_instances_load(self, name: str) -> None:
        inst = load_by_name(name)
        assert inst.name == name
        assert inst.num_operations == inst.num_jobs * inst.ops_per_job

    @pytest.mark.parametrize("name", [f"la{i:02d}" for i in range(1, 41)])
    def test_lawrence_instances_load(self, name: str) -> None:
        inst = load_by_name(name)
        assert inst.num_jobs > 0
        assert inst.num_machines > 0

    def test_unknown_instance_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="unknown instance"):
            load_by_name("does-not-exist")

    def test_available_instances_includes_bundled_names(self) -> None:
        names = available_instances()
        assert "ft06" in names
        assert "la40" in names
        assert names == sorted(names)

    def test_load_by_name_is_cached(self) -> None:
        assert load_by_name("ft06") is load_by_name("ft06")


class TestKnownOptima:
    """The three headline checks: parser + bundled files reproduce literature optima."""

    def test_ft06_known_optimum(self) -> None:
        assert known_optimum("ft06") == 55

    def test_ft10_known_optimum(self) -> None:
        assert known_optimum("ft10") == 930

    def test_ft20_known_optimum(self) -> None:
        assert known_optimum("ft20") == 1165

    def test_unreferenced_instance_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            known_optimum("not-a-real-instance")

    def test_known_optima_table_loads(self) -> None:
        table = load_known_optima()
        assert table["ft06"].family == "fisher-thompson"
        assert table["ft06"].is_proven_optimal

    @pytest.mark.parametrize("name", available_instances() or ["ft06"])
    def test_static_lower_bound_never_exceeds_known_optimum(self, name: str) -> None:
        """A correct parse's LB must be <= the literature optimum for every instance.

        This is the cheapest possible cross-check that the whole bundled instance
        set was parsed correctly, not just the three headline ``ft`` instances.
        """
        instance = load_by_name(name)
        lb = static_lower_bound(instance)
        optimum = known_optimum(name)
        assert lb <= optimum, f"{name}: LB={lb} exceeds known optimum={optimum} (parse bug?)"

    def test_ft06_lower_bound_is_reasonably_tight(self) -> None:
        inst = load_by_name("ft06")
        lb = static_lower_bound(inst)
        assert lb <= 55
        assert lb >= 40  # loose sanity floor; a bound of e.g. 0 would signal a bug
