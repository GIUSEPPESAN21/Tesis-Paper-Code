"""Unit tests for :mod:`envs.fjsp_loader` (the real Brandimarte ``MK*`` loader).

Written per ``FinalExperimentMatrix.md`` §5's amendment precondition: E2 Arm B (the real
``mk01``-``mk07`` FJSP arm) may not be used in any confirmatory result until this file
exists and passes, pinning the rectangular-padding neutrality claim the module's own
docstring promises (a zero-duration, single-eligible-machine padding operation must not
change the makespan CP-SAT finds).
"""

from __future__ import annotations

import pytest

from baselines.cpsat import solve_cpsat_fjsp
from envs.fjsp_loader import (
    FJSPParseError,
    available_fjsp_instances,
    fjsp_known_optimum,
    load_fjsp_by_name,
    load_fjsp_known_optima,
    real_operation_mask,
)

#: The 7 bundled Brandimarte instances and their known-optimum status, per
#: data/fjsp_known_optima.csv.
BUNDLED_INSTANCES = ["mk01", "mk02", "mk03", "mk04", "mk05", "mk06", "mk07"]
PROVEN_OPTIMAL = {"mk01", "mk03", "mk04"}


class TestAvailableInstances:
    def test_lists_exactly_the_seven_bundled_instances(self) -> None:
        assert available_fjsp_instances() == sorted(BUNDLED_INSTANCES)


#: KNOWN UPSTREAM DATA ISSUE (found while writing this file, 2026-07-31): the
#: SchedulingLab/fjsp-instances mirror's own reference index (``instances.json``,
#: consumed into ``data/fjsp_known_optima.csv``) records mk06 as 15 machines -- the
#: canonical Brandimarte (1993) figure -- but the mirror's own ``brandimarte/mk06.txt``
#: data file only ever uses machine indices 0-9 (10 distinct machines) and its own
#: header line literally reads ``10 10``. Re-fetching with --force reproduces the same
#: file, so this is not a transient download glitch or a local corruption -- the
#: mirror's metadata and its data file for mk06 genuinely disagree with each other.
#: Until this is resolved upstream (or the correct 15-machine mk06.txt is sourced from
#: elsewhere and substituted by hand, which this project deliberately avoids doing --
#: see the module docstring's "we do not transcribe by hand" rationale), mk06 is
#: excluded from any confirmatory RPD-vs-known-optimum comparison: comparing this
#: file's makespan against the literature's 58 optimum would silently compare two
#: different problem instances. See EXPERIMENT_LOG_WEEK24.md.
MK06_MACHINE_COUNT_MISMATCH = "mk06"


class TestParsing:
    @pytest.mark.parametrize(
        "name", [n for n in BUNDLED_INSTANCES if n != MK06_MACHINE_COUNT_MISMATCH]
    )
    def test_loads_without_error_and_matches_reference_shape(self, name: str) -> None:
        instance = load_fjsp_by_name(name)
        ref = load_fjsp_known_optima()[name]
        assert instance.num_jobs == ref.num_jobs
        assert instance.num_machines == ref.num_machines
        assert instance.name == name

    def test_mk06_known_upstream_machine_count_mismatch_is_still_present(self) -> None:
        """Documents (does not silently hide) the mismatch described above. If this
        test starts failing, the upstream mirror has been fixed -- remove the
        ``MK06_MACHINE_COUNT_MISMATCH`` exclusion above and re-enable mk06 in the
        parametrized shape test and in any confirmatory Arm B reporting."""
        instance = load_fjsp_by_name("mk06")
        ref = load_fjsp_known_optima()["mk06"]
        max_machine_used = max(
            m for job in instance.eligible_machines for op in job for m in op
        )
        assert instance.num_machines == 10, "mk06.txt header no longer reads 10 machines"
        assert max_machine_used == 9, "mk06.txt now uses an 11th+ machine index"
        assert ref.num_machines == 15, "known_optima.csv no longer claims 15 machines"

    @pytest.mark.parametrize("name", BUNDLED_INSTANCES)
    def test_every_operation_has_at_least_one_eligible_machine(self, name: str) -> None:
        instance = load_fjsp_by_name(name)
        for job in range(instance.num_jobs):
            for pos in range(instance.ops_per_job):
                assert len(instance.eligible_machines[job][pos]) >= 1

    @pytest.mark.parametrize("name", BUNDLED_INSTANCES)
    def test_machine_indices_are_zero_indexed_in_range(self, name: str) -> None:
        instance = load_fjsp_by_name(name)
        for job in range(instance.num_jobs):
            for pos in range(instance.ops_per_job):
                for machine in instance.eligible_machines[job][pos]:
                    assert 0 <= machine < instance.num_machines

    def test_unknown_instance_name_raises_filenotfound(self) -> None:
        with pytest.raises(FileNotFoundError, match="unknown FJSP instance"):
            load_fjsp_by_name("mk99")

    def test_malformed_text_raises_fjspparseerror(self) -> None:
        from envs.fjsp_loader import parse_brandimarte  # noqa: PLC0415

        with pytest.raises(FJSPParseError):
            parse_brandimarte("not a valid instance body", name="bad")


class TestRectangularPadding:
    @pytest.mark.parametrize("name", BUNDLED_INSTANCES)
    def test_padding_ops_are_zero_duration_single_machine(self, name: str) -> None:
        """Every operation ``real_operation_mask`` marks as padding must be exactly the
        loader's zero-duration, single-eligible-machine filler (module docstring)."""
        instance = load_fjsp_by_name(name)
        mask = real_operation_mask(instance)
        for job in range(instance.num_jobs):
            for pos in range(instance.ops_per_job):
                op = instance.op_id(job, pos)
                durs = instance.durations[job][pos]
                is_padding_shape = len(durs) == 1 and durs[0] == 0
                assert mask[op] == (not is_padding_shape)

    @pytest.mark.parametrize("name", BUNDLED_INSTANCES)
    def test_ops_per_job_matches_longest_job(self, name: str) -> None:
        instance = load_fjsp_by_name(name)
        mask = real_operation_mask(instance)
        # Every job must have at least one real (non-padding) operation at every
        # position up to its own true length, and the padded tail (if any) must be
        # entirely marked as non-real.
        for job in range(instance.num_jobs):
            real_positions = [
                pos for pos in range(instance.ops_per_job) if mask[instance.op_id(job, pos)]
            ]
            # Real positions for a job must be a contiguous prefix (padding is only
            # ever appended at the end of the job chain, per the module docstring).
            assert real_positions == list(range(len(real_positions)))


class TestKnownOptima:
    def test_status_distinguishes_proven_from_best_known(self) -> None:
        table = load_fjsp_known_optima()
        for name in PROVEN_OPTIMAL:
            assert table[name].is_proven_optimal, f"{name} should be status=proven"
        for name in set(BUNDLED_INSTANCES) - PROVEN_OPTIMAL:
            assert not table[name].is_proven_optimal, f"{name} should be status=best_known"

    def test_unknown_name_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            fjsp_known_optimum("mk99")


@pytest.mark.slow
class TestPaddingNeutralityViaCpsat:
    """Pins the module docstring's central claim: rectangularity padding is exactly
    makespan-neutral, because CP-SAT's ``NoOverlap`` treats a zero-size interval as
    occupying no machine capacity. If padding silently cost capacity, CP-SAT would never
    reach the literature-proven optimum on these instances; it does.
    """

    @pytest.mark.parametrize(
        "name,time_limit_s",
        [("mk01", 120.0), ("mk03", 300.0), ("mk04", 300.0)],
    )
    def test_cpsat_reaches_literature_optimum_on_padded_instance(
        self, name: str, time_limit_s: float
    ) -> None:
        instance = load_fjsp_by_name(name)
        optimum = fjsp_known_optimum(name)
        result = solve_cpsat_fjsp(instance, time_limit_s=time_limit_s, num_workers=8)
        assert result.is_optimal, f"{name}: status={result.status}, bound={result.best_bound}"
        assert result.status == "OPTIMAL"
        assert result.makespan == optimum, (
            f"{name}: CP-SAT found {result.makespan} but the proven literature optimum is "
            f"{optimum} -- padding may not be makespan-neutral"
        )
