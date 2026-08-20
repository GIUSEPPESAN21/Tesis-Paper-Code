"""Unit tests for :mod:`baselines.ga` (the Week 12 genetic algorithm baseline).

Covers ``Baselines_TechSpec.md`` §7's required list (decode always precedence-feasible,
POX preserves per-job operation counts, elitism monotonicity, respects ``time_limit_s``,
beats ``random_rule``) plus ``GA_TechSpec.md`` §6's additions (population-size band
tripwire, random immigrants replace the worst offspring not the elites, and the
``la05``-reaches-zero-RPD permanent regression test promoted from that spec's §4.2
tuning-confirmation step).

The FJSP-ready path (assignment vector, uniform crossover, reassignment mutation) is
exercised both against a hand-built ragged-eligibility
:class:`~envs.fjsp_instance.FJSPInstance` fixture (``fjsp_toy``) and, since the Week 15
refactor, against a real-benchmark-derived one
(:func:`~envs.fjsp_instance.generate_fjsp_instance`) -- see ``TestRunReturnsMachineOfOp``
and ``tests/test_fjsp_instance.py``. No real FJSP benchmark file loader (Brandimarte
``MK*``, Hurink) is bundled; that remains a research-scope decision for a later sprint.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

from baselines.dispatching_rules import run_rule
from baselines.ga import (
    GAConfig,
    GeneticAlgorithm,
    _random_job_partition,
    decode,
    default_population_size,
    pox,
    random_chromosome,
    reassignment_mutation,
    swap_mutation,
    uniform_crossover_assignment,
)
from envs.disjunctive_graph import is_schedule_feasible
from envs.fjsp_instance import FJSPInstance, generate_fjsp_instance
from envs.jssp_env import JSSPEnv
from envs.loader import known_optimum, load_by_name


@pytest.fixture
def fjsp_toy() -> FJSPInstance:
    """2 jobs x 2 ops, ragged eligible-machine sets (same fixture Week 9's CP-SAT FJSP
    tests use; known optimum 7, hand-verified in ``tests/test_cpsat.py``)."""
    return FJSPInstance(
        name="fjsp_toy",
        num_jobs=2,
        ops_per_job=2,
        num_machines=2,
        eligible_machines=[[[0, 1], [1]], [[0], [0, 1]]],
        durations=[[[3, 5], [2]], [[4], [1, 2]]],
    )


class TestDecode:
    def test_decode_is_always_precedence_feasible(self) -> None:
        instance = load_by_name("ft06")
        rng = np.random.default_rng(0)
        for _ in range(200):
            chromosome = random_chromosome(instance, rng)
            result = decode(instance, chromosome)
            feasible, violations = is_schedule_feasible(instance, result.start_times)
            assert feasible, violations

    def test_decode_never_beats_the_proven_optimum(self) -> None:
        """Regression guard for the exact bug found during implementation: computing
        makespan as ``max(start_times)`` instead of ``max(end_times)`` silently
        under-reports it, which briefly made the GA look like it beat ft06's proven
        optimum (55). Any random chromosome's decoded makespan must be >= 55.
        """
        instance = load_by_name("ft06")
        rng = np.random.default_rng(1)
        for _ in range(200):
            result = decode(instance, random_chromosome(instance, rng))
            assert result.makespan >= known_optimum("ft06")

    def test_fjsp_decode_respects_chosen_machine_and_precedence(self, fjsp_toy: FJSPInstance) -> None:
        rng = np.random.default_rng(0)
        for _ in range(50):
            chromosome = random_chromosome(fjsp_toy, rng)
            result = decode(fjsp_toy, chromosome)
            for job in range(fjsp_toy.num_jobs):
                for pos in range(fjsp_toy.ops_per_job - 1):
                    u, v = fjsp_toy.op_id(job, pos), fjsp_toy.op_id(job, pos + 1)
                    dur_u = fjsp_toy.durations[job][pos][chromosome.assignment[u]]
                    assert result.start_times[v] + 1e-9 >= result.start_times[u] + dur_u
                    assert result.machine_of_op[u] in fjsp_toy.eligible_machines[job][pos]


class TestPOX:
    def test_pox_preserves_per_job_operation_counts(self) -> None:
        instance = load_by_name("ft06")
        rng = np.random.default_rng(2)
        for _ in range(100):
            a = random_chromosome(instance, rng)
            b = random_chromosome(instance, rng)
            j1 = _random_job_partition(instance.num_jobs, rng)
            child = pox(a.sequence, b.sequence, j1, rng)
            counts = np.bincount(child, minlength=instance.num_jobs)
            assert (counts == instance.ops_per_job).all()

    def test_pox_child_decodes_feasibly(self) -> None:
        instance = load_by_name("ft06")
        rng = np.random.default_rng(3)
        for _ in range(50):
            a = random_chromosome(instance, rng)
            b = random_chromosome(instance, rng)
            j1 = _random_job_partition(instance.num_jobs, rng)
            child_seq = pox(a.sequence, b.sequence, j1, rng)
            result = decode(instance, type(a)(sequence=child_seq, assignment=None))
            feasible, violations = is_schedule_feasible(instance, result.start_times)
            assert feasible, violations


class TestMutationOperators:
    def test_swap_mutation_preserves_operation_counts(self) -> None:
        instance = load_by_name("ft06")
        rng = np.random.default_rng(4)
        chromosome = random_chromosome(instance, rng)
        mutated = swap_mutation(chromosome.sequence, rng)
        counts_before = np.bincount(chromosome.sequence, minlength=instance.num_jobs)
        counts_after = np.bincount(mutated, minlength=instance.num_jobs)
        np.testing.assert_array_equal(counts_before, counts_after)

    def test_reassignment_mutation_stays_within_eligible_machines(self, fjsp_toy: FJSPInstance) -> None:
        rng = np.random.default_rng(5)
        chromosome = random_chromosome(fjsp_toy, rng)
        for _ in range(50):
            mutated = reassignment_mutation(fjsp_toy, chromosome.assignment, rng)
            for job in range(fjsp_toy.num_jobs):
                for pos in range(fjsp_toy.ops_per_job):
                    op = fjsp_toy.op_id(job, pos)
                    assert 0 <= mutated[op] < len(fjsp_toy.eligible_machines[job][pos])
            chromosome.assignment = mutated

    def test_uniform_crossover_assignment_is_noop_for_jssp(self) -> None:
        rng = np.random.default_rng(6)
        assert uniform_crossover_assignment(None, None, rng) is None

    def test_uniform_crossover_assignment_picks_from_either_parent(self, fjsp_toy: FJSPInstance) -> None:
        rng = np.random.default_rng(7)
        a = random_chromosome(fjsp_toy, rng)
        b = random_chromosome(fjsp_toy, rng)
        child = uniform_crossover_assignment(a.assignment, b.assignment, rng)
        for gene, ga_, gb in zip(child, a.assignment, b.assignment):
            assert gene == ga_ or gene == gb


class TestGenerationLoop:
    @pytest.mark.parametrize("seed", range(5))
    def test_elitism_makespan_monotone_non_increasing(self, seed: int) -> None:
        instance = load_by_name("ft06")
        rng = np.random.default_rng(seed)
        cfg = GAConfig(population_size=20)
        ga = GeneticAlgorithm(cfg)

        population = [random_chromosome(instance, rng) for _ in range(cfg.population_size)]
        makespans = np.array([decode(instance, c).makespan for c in population])
        fitnesses = -makespans
        best_so_far = float(makespans.min())

        for _ in range(15):
            population, makespans, fitnesses = ga._one_generation(instance, population, makespans, fitnesses, rng)
            gen_best = float(makespans.min())
            assert gen_best <= best_so_far + 1e-9
            best_so_far = min(best_so_far, gen_best)

    def test_elites_pass_through_a_generation_unchanged(self) -> None:
        instance = load_by_name("ft06")
        rng = np.random.default_rng(8)
        cfg = GAConfig(population_size=20, elite_count=2)
        ga = GeneticAlgorithm(cfg)

        population = [random_chromosome(instance, rng) for _ in range(cfg.population_size)]
        makespans = np.array([decode(instance, c).makespan for c in population])
        fitnesses = -makespans
        expected_elites = [population[i] for i in np.argsort(makespans)[: cfg.elite_count]]

        new_population, _, _ = ga._one_generation(instance, population, makespans, fitnesses, rng)

        assert all(new_population[i] is expected_elites[i] for i in range(cfg.elite_count))

    def test_random_immigrants_replace_worst_offspring_only(self) -> None:
        """The replaced individuals are exactly the ``immigrant_fraction``-worst of the
        *offspring* (never the elites, which don't pass through ``_apply_random_immigrants``
        at all --- see :class:`TestGenerationLoop`'s companion elitism test).
        """
        instance = load_by_name("ft06")
        rng = np.random.default_rng(9)
        cfg = GAConfig(population_size=20, elite_count=2, immigrant_fraction=0.2)
        ga = GeneticAlgorithm(cfg)

        population = [random_chromosome(instance, rng) for _ in range(cfg.population_size)]
        makespans = np.array([decode(instance, c).makespan for c in population])
        fitnesses = -makespans

        offspring = ga._generate_offspring(instance, population, fitnesses, rng, cfg.population_size - cfg.elite_count)
        offspring_makespans = np.array([decode(instance, c).makespan for c in offspring])
        offspring_ids_before = [id(c) for c in offspring]

        expected_n_immigrants = max(1, round(cfg.immigrant_fraction * len(offspring)))
        expected_worst_idx = set(np.argsort(offspring_makespans)[::-1][:expected_n_immigrants].tolist())

        ga._apply_random_immigrants(instance, offspring, offspring_makespans.copy(), rng)

        changed_idx = {i for i, c in enumerate(offspring) if id(c) != offspring_ids_before[i]}
        assert changed_idx == expected_worst_idx


class TestWallClockBudget:
    @pytest.mark.parametrize("time_limit_s", [0.5, 1.5, 3.0])
    def test_ga_respects_time_limit(self, time_limit_s: float) -> None:
        instance = load_by_name("ft06")
        ga = GeneticAlgorithm(GAConfig(population_size=20))
        t0 = time.perf_counter()
        ga.run(instance, seed=0, time_limit_s=time_limit_s)
        elapsed = time.perf_counter() - t0
        assert elapsed <= time_limit_s + 1.0  # small fixed slack for one in-flight generation

    def test_run_requires_time_limit(self) -> None:
        instance = load_by_name("ft06")
        with pytest.raises(ValueError, match="time_limit_s"):
            GeneticAlgorithm().run(instance, seed=0, time_limit_s=None)


class TestGAQuality:
    def test_ga_beats_random_rule_median_on_ft06(self) -> None:
        env = JSSPEnv("ft06")
        random_makespans = [run_rule(env, "random", seed=s)["makespan"] for s in range(20)]
        median_random = statistics.median(random_makespans)

        ga = GeneticAlgorithm(GAConfig(population_size=40))
        result = ga.run(load_by_name("ft06"), seed=0, time_limit_s=10.0)

        assert result["makespan"] <= median_random

    def test_ga_reaches_zero_rpd_on_la05(self) -> None:
        """Permanent regression test (``GA_TechSpec.md`` §4.2/§6): median RPD over 5
        seeds at the full ``time_limit_s=60`` on ``la05`` is 0.0 -- the one instance
        where a single-pass PDR (FIFO) already matches the CP-SAT optimum exactly
        (``BaselineAnalysis.md`` §2), making it a low-bar sanity check. If this starts
        failing after an unrelated change, treat it as a decoder/operator regression,
        not a tuning drift.
        """
        instance = load_by_name("la05")
        optimum = known_optimum("la05")
        rpds = []
        for seed in range(5):
            ga = GeneticAlgorithm(GAConfig())  # default_population_size(50) == 80
            result = ga.run(instance, seed=seed, time_limit_s=60.0)
            rpds.append(100.0 * (result["makespan"] - optimum) / optimum)
        assert statistics.median(rpds) == pytest.approx(0.0, abs=1e-6)


class TestRunReturnsMachineOfOp:
    """``machine_of_op`` (Week 15): GABaseline needs it to build an FJSP-aware
    feasibility check, so GeneticAlgorithm.run() must surface it, not just decode()."""

    def test_jssp_machine_of_op_matches_instance_routing(self) -> None:
        instance = load_by_name("ft06")
        result = GeneticAlgorithm(GAConfig(population_size=20)).run(instance, seed=0, time_limit_s=2.0)
        assert "machine_of_op" in result
        np.testing.assert_array_equal(result["machine_of_op"], instance.flat_machines)

    def test_fjsp_machine_of_op_stays_within_eligible_sets(self) -> None:
        fjsp = generate_fjsp_instance(load_by_name("ft06"), seed=3)
        result = GeneticAlgorithm(GAConfig(population_size=20)).run(fjsp, seed=0, time_limit_s=3.0)
        for job in range(fjsp.num_jobs):
            for pos in range(fjsp.ops_per_job):
                op = fjsp.op_id(job, pos)
                assert int(result["machine_of_op"][op]) in fjsp.eligible_machines[job][pos]


class TestDefaultPopulationSize:
    def test_default_population_size_matches_tier_bands(self) -> None:
        assert default_population_size(36) == 80
        assert default_population_size(100) == 80
        assert default_population_size(101) == 60
        assert default_population_size(450) == 60
        assert default_population_size(451) == 40
