"""Genetic algorithm solver for JSSP, FJSP-ready via the machine-assignment chromosome.

Per ``Baselines_TechSpec.md`` §5 (encoding/operators) and ``GA_TechSpec.md`` (the
finalized hyperparameter defaults and the random-immigrants stagnation guard). The
assignment-vector chromosome, its decoder, and its operators run unmodified against
either :class:`~envs.instance.JSSPInstance` (fixed routing, ``assignment=None``) or
:class:`~envs.fjsp_instance.FJSPInstance` (ragged eligible-machine sets) --- see
:func:`is_fjsp` --- and are exercised against both a hand-built ragged fixture and a
real-benchmark-derived instance (:func:`~envs.fjsp_instance.generate_fjsp_instance`) in
``tests/test_ga.py``. No real FJSP benchmark file loader (Brandimarte ``MK*``, Hurink) is
bundled this sprint (see ``envs/fjsp_instance.py``'s module docstring); that is a
research-scope decision for a later week, not a blocker for this code path.

Decoding is the same list-scheduling rule :class:`~envs.jssp_env.JSSPEnv` uses
(``start = max(job_ready_time[job], machine_free_time[machine])``), so every decoded
chromosome is guaranteed semi-active-feasible by construction, cross-checkable against
:func:`~envs.disjunctive_graph.is_schedule_feasible` with zero special-casing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Union

import numpy as np

from envs.fjsp_instance import FJSPInstance
from envs.instance import JSSPInstance

Instance = Union[JSSPInstance, FJSPInstance]


def is_fjsp(instance: Instance) -> bool:
    """Duck-type: an :class:`~envs.fjsp_instance.FJSPInstance` has ``eligible_machines``."""
    return hasattr(instance, "eligible_machines")


def default_population_size(num_operations: int) -> int:
    """``population_size``, scaled down for larger instances (``GA_TechSpec.md`` §2.1).

    Bands match ``Section5.1_5.2.md`` Table 5.1.1 exactly: a starting rule, not a tuned
    law --- the two untested bands (60, 40) are a documented extrapolation, same status
    as that table's CP-SAT time-limit tiers.
    """
    if num_operations <= 100:  # "pequeña"/"moderada" bands, validated tiers
        return 80
    if num_operations <= 450:  # "intermedia" band (la11-la40), not yet executed
        return 60
    return 40  # "grande" band (ta, up to 2000 ops), not yet executed


@dataclass
class GAConfig:
    """Hyperparameters for :class:`GeneticAlgorithm`. Defaults are ``GA_TechSpec.md``
    Table 2.1, a documented starting configuration (standard permutation-GA practice),
    not a value derived from Part B7 --- see that spec's §7 for the full rationale.

    Attributes:
        population_size: ``None`` (default) auto-scales via
            :func:`default_population_size` on the instance's operation count.
        tournament_size: Individuals sampled (with replacement) per tournament.
        elite_count: Best individuals of generation ``g`` copied unmutated into ``g+1``.
        p_crossover: Probability POX is applied to a mating-pool pair; the complement
            copies the tournament winner forward unmodified.
        p_mutation_sequence: Per-individual (not per-gene) probability of exactly one
            swap mutation on the sequence gene.
        p_mutation_assignment: Same, for the reassignment mutation on the assignment
            gene (FJSP only; a no-op when ``assignment is None``).
        immigrant_fraction: Fraction of the *offspring* (not the elites) replaced each
            generation with freshly random chromosomes, a stagnation guard
            (``GA_TechSpec.md`` §5) not specified by ``Baselines_TechSpec.md`` §5.
    """

    population_size: int | None = None
    tournament_size: int = 3
    elite_count: int = 2
    p_crossover: float = 0.90
    p_mutation_sequence: float = 0.15
    p_mutation_assignment: float = 0.15
    immigrant_fraction: float = 0.05


@dataclass
class Chromosome:
    """One candidate schedule.

    Attributes:
        sequence: Shape ``(num_operations,)``; job index repeated ``ops_per_job`` times.
            The r-th occurrence of job ``j`` designates that job's r-th operation.
        assignment: Shape ``(num_operations,)``, an eligible-machine *index* (not a
            machine id) per global operation id, indexing into
            ``instance.eligible_machines[job][pos]``. ``None`` for classic JSSP (fixed
            routing) --- the gene is omitted entirely, not stored as a length-0 array.
    """

    sequence: np.ndarray
    assignment: np.ndarray | None = None


@dataclass
class DecodeResult:
    """Outcome of decoding one :class:`Chromosome` against an instance.

    Attributes:
        start_times: Start time per global operation id, same convention as
            ``JSSPEnv.start_times``.
        makespan: The decoded schedule's makespan.
        machine_of_op: The actual machine used per operation (== ``instance.machines``
            for JSSP; the chosen alternative for FJSP) --- diagnostic only, not part of
            the canonical schedule representation.
    """

    start_times: np.ndarray
    makespan: float
    machine_of_op: np.ndarray


def _machine_and_duration(
    instance: Instance, chromosome: Chromosome, job: int, pos: int, op_id: int
) -> tuple[int, float]:
    """Resolve the machine and duration operation ``(job, pos)`` actually runs with."""
    if chromosome.assignment is None:
        return int(instance.machines[job, pos]), float(instance.durations[job, pos])
    alt_index = int(chromosome.assignment[op_id])
    machine = instance.eligible_machines[job][pos][alt_index]
    duration = instance.durations[job][pos][alt_index]
    return int(machine), float(duration)


def decode(instance: Instance, chromosome: Chromosome) -> DecodeResult:
    """Left-to-right list-scheduling scan of ``chromosome.sequence``.

    For each occurrence of job ``j`` in the sequence, dispatch its next operation on the
    machine/duration :func:`_machine_and_duration` resolves, at
    ``start = max(job_ready_time[j], machine_free_time[machine])`` --- the same rule
    ``JSSPEnv.step()`` uses (``Baselines_TechSpec.md`` §5.2).
    """
    num_jobs = instance.num_jobs
    num_machines = instance.num_machines
    job_ready_time = np.zeros(num_jobs, dtype=np.float64)
    machine_free_time = np.zeros(num_machines, dtype=np.float64)
    job_next_pos = np.zeros(num_jobs, dtype=np.int64)

    n_ops = instance.num_operations
    start_times = np.full(n_ops, -1.0)
    end_times = np.full(n_ops, -1.0)
    machine_of_op = np.zeros(n_ops, dtype=np.int64)

    for job in chromosome.sequence:
        job = int(job)
        pos = int(job_next_pos[job])
        op_id = instance.op_id(job, pos)
        machine, duration = _machine_and_duration(instance, chromosome, job, pos, op_id)

        start = max(float(job_ready_time[job]), float(machine_free_time[machine]))
        end = start + duration

        start_times[op_id] = start
        end_times[op_id] = end
        machine_of_op[op_id] = machine
        job_ready_time[job] = end
        machine_free_time[machine] = end
        job_next_pos[job] += 1

    return DecodeResult(start_times=start_times, makespan=float(end_times.max()), machine_of_op=machine_of_op)


def random_chromosome(instance: Instance, rng: np.random.Generator) -> Chromosome:
    """A uniformly random permutation-with-repetition sequence, plus (FJSP only) a
    uniformly random eligible-machine assignment."""
    sequence = np.repeat(np.arange(instance.num_jobs), instance.ops_per_job)
    rng.shuffle(sequence)

    assignment = None
    if is_fjsp(instance):
        assignment = np.zeros(instance.num_operations, dtype=np.int64)
        for job in range(instance.num_jobs):
            for pos in range(instance.ops_per_job):
                op = instance.op_id(job, pos)
                num_alternatives = len(instance.eligible_machines[job][pos])
                assignment[op] = int(rng.integers(num_alternatives))
    return Chromosome(sequence=sequence, assignment=assignment)


def _random_job_partition(num_jobs: int, rng: np.random.Generator) -> set[int]:
    """A random, non-trivial (neither empty nor all) subset of ``range(num_jobs)``."""
    if num_jobs == 1:
        return {0}
    mask = rng.random(num_jobs) < 0.5
    if not mask.any():
        mask[int(rng.integers(num_jobs))] = True
    if mask.all():
        mask[int(rng.integers(num_jobs))] = False
    return set(np.flatnonzero(mask).tolist())


def pox(seq_a: np.ndarray, seq_b: np.ndarray, j1: set[int], rng: np.random.Generator) -> np.ndarray:
    """Precedence Operation Crossover: one child from two parent sequences.

    Every gene belonging to a job in ``j1`` keeps parent A's value in its original
    position; the remaining positions (jobs in ``J2 = complement(j1)``) are filled, in
    order, with parent B's ``J2`` genes. This invariant (`child` is always a valid
    permutation-with-repetition with the same per-job operation counts as the parents)
    is what guarantees precedence feasibility without a repair step
    (``Baselines_TechSpec.md`` §5.3). ``rng`` is accepted for interface symmetry with
    the other operators but unused: POX itself is deterministic given ``j1``.
    """
    del rng
    in_j1 = np.isin(seq_a, list(j1))
    child = np.empty_like(seq_a)
    child[in_j1] = seq_a[in_j1]
    child[~in_j1] = seq_b[~np.isin(seq_b, list(j1))]
    return child


def swap_mutation(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pick two positions uniformly at random and swap their values."""
    child = sequence.copy()
    if len(child) < 2:
        return child
    i, j = rng.choice(len(child), size=2, replace=False)
    child[i], child[j] = child[j], child[i]
    return child


def uniform_crossover_assignment(
    assignment_a: np.ndarray | None, assignment_b: np.ndarray | None, rng: np.random.Generator
) -> np.ndarray | None:
    """Per-gene independent 50/50 inheritance. No-op (returns ``None``) for classic JSSP."""
    if assignment_a is None:
        return None
    mask = rng.random(len(assignment_a)) < 0.5
    return np.where(mask, assignment_a, assignment_b)


def reassignment_mutation(instance: Instance, assignment: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pick an operation uniformly at random, replace its assigned machine with a
    different uniformly-random eligible machine (FJSP only)."""
    child = assignment.copy()
    op = int(rng.integers(len(child)))
    job, pos = instance.job_of(op), instance.position_of(op)
    num_alternatives = len(instance.eligible_machines[job][pos])
    if num_alternatives <= 1:
        return child  # no alternative machine to switch to
    current = int(child[op])
    choices = [a for a in range(num_alternatives) if a != current]
    child[op] = rng.choice(choices)
    return child


def tournament_select(
    population: list[Chromosome], fitnesses: np.ndarray, tournament_size: int, rng: np.random.Generator
) -> Chromosome:
    """Sample ``tournament_size`` individuals (with replacement) and return the fittest."""
    idx = rng.integers(0, len(population), size=tournament_size)
    winner = idx[int(np.argmax(fitnesses[idx]))]
    return population[int(winner)]


class GeneticAlgorithm:
    """GA solver: tournament selection + elitism, POX/uniform crossover, swap/reassignment
    mutation, random immigrants (``GA_TechSpec.md`` §5). Wall-clock-first stopping rule:
    the primary termination criterion is ``time_limit_s``, not a fixed generation count
    (``Baselines_TechSpec.md`` §5.6), for fair comparison against CP-SAT under the same
    time budget (``GA_TechSpec.md`` §3).
    """

    def __init__(self, config: GAConfig | None = None) -> None:
        self.config = config if config is not None else GAConfig()

    def run(self, instance: Instance, *, seed: int | None = None, time_limit_s: float | None = None) -> dict:
        """Run to the wall-clock budget and return the best schedule found.

        Args:
            instance: The instance to solve.
            seed: Governs population initialization and all stochastic operators.
                ``None``-safe (falls back to entropy), like ``envs/disruptions.py``'s
                ``spawn_disruption_rngs``.
            time_limit_s: Wall-clock budget in seconds. Required.

        Returns:
            Dict with ``start_times``, ``makespan``, ``machine_of_op`` (the actual
            machine each operation ran on --- ``instance.machines`` for JSSP, the
            chosen alternative for FJSP; see :attr:`DecodeResult.machine_of_op`),
            ``generations_run``, and ``best_makespan_history`` (the running best-so-far
            makespan after each generation --- monotone non-increasing by construction,
            thanks to elitism).

        Raises:
            ValueError: If ``time_limit_s`` is ``None``.
        """
        if time_limit_s is None:
            raise ValueError("GeneticAlgorithm.run requires an explicit time_limit_s")

        rng = np.random.default_rng(seed)
        cfg = self.config
        population_size = (
            cfg.population_size if cfg.population_size is not None else default_population_size(instance.num_operations)
        )

        population = [random_chromosome(instance, rng) for _ in range(population_size)]
        makespans = np.array([decode(instance, c).makespan for c in population])
        fitnesses = -makespans

        best_idx = int(np.argmin(makespans))
        best_chromosome = population[best_idx]
        best_makespan = float(makespans[best_idx])
        best_makespan_history = [best_makespan]

        deadline = time.monotonic() + time_limit_s
        generations_run = 0
        while time.monotonic() < deadline:
            population, makespans, fitnesses = self._one_generation(instance, population, makespans, fitnesses, rng)
            gen_best_idx = int(np.argmin(makespans))
            if makespans[gen_best_idx] < best_makespan:
                best_makespan = float(makespans[gen_best_idx])
                best_chromosome = population[gen_best_idx]
            best_makespan_history.append(best_makespan)
            generations_run += 1

        best_decode = decode(instance, best_chromosome)
        return {
            "start_times": best_decode.start_times,
            "makespan": best_makespan,
            "machine_of_op": best_decode.machine_of_op,
            "generations_run": generations_run,
            "best_makespan_history": best_makespan_history,
        }

    def _one_generation(
        self,
        instance: Instance,
        population: list[Chromosome],
        makespans: np.ndarray,
        fitnesses: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[list[Chromosome], np.ndarray, np.ndarray]:
        """One generation: elitism, tournament selection, POX/uniform crossover,
        swap/reassignment mutation, then random immigrants replace the worst offspring
        (``GA_TechSpec.md`` §5's pseudocode, implemented literally --- as the two
        sub-steps :meth:`_generate_offspring` and :meth:`_apply_random_immigrants`, kept
        separate so the immigrant-replacement behavior is independently testable)."""
        cfg = self.config
        n = len(population)

        elite_idx = np.argsort(makespans)[: cfg.elite_count]
        elites = [population[i] for i in elite_idx]
        elite_makespans = makespans[elite_idx]

        offspring = self._generate_offspring(instance, population, fitnesses, rng, n - cfg.elite_count)
        offspring_makespans = np.array([decode(instance, c).makespan for c in offspring])
        offspring, offspring_makespans = self._apply_random_immigrants(instance, offspring, offspring_makespans, rng)

        new_population = elites + offspring
        new_makespans = np.concatenate([elite_makespans, offspring_makespans])
        new_fitnesses = -new_makespans
        return new_population, new_makespans, new_fitnesses

    def _generate_offspring(
        self,
        instance: Instance,
        population: list[Chromosome],
        fitnesses: np.ndarray,
        rng: np.random.Generator,
        count: int,
    ) -> list[Chromosome]:
        """Tournament selection + POX/uniform crossover + swap/reassignment mutation,
        producing exactly ``count`` non-elite offspring."""
        cfg = self.config
        offspring: list[Chromosome] = []
        while len(offspring) < count:
            parent_a = tournament_select(population, fitnesses, cfg.tournament_size, rng)
            parent_b = tournament_select(population, fitnesses, cfg.tournament_size, rng)

            if rng.random() < cfg.p_crossover:
                j1 = _random_job_partition(instance.num_jobs, rng)
                child_sequence = pox(parent_a.sequence, parent_b.sequence, j1, rng)
                child_assignment = uniform_crossover_assignment(parent_a.assignment, parent_b.assignment, rng)
                child = Chromosome(sequence=child_sequence, assignment=child_assignment)
            else:
                child = Chromosome(
                    sequence=parent_a.sequence.copy(),
                    assignment=None if parent_a.assignment is None else parent_a.assignment.copy(),
                )

            if rng.random() < cfg.p_mutation_sequence:
                child.sequence = swap_mutation(child.sequence, rng)
            if child.assignment is not None and rng.random() < cfg.p_mutation_assignment:
                child.assignment = reassignment_mutation(instance, child.assignment, rng)

            offspring.append(child)
        return offspring

    def _apply_random_immigrants(
        self,
        instance: Instance,
        offspring: list[Chromosome],
        offspring_makespans: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[list[Chromosome], np.ndarray]:
        """Replace the worst ``immigrant_fraction`` of ``offspring`` (never the elites,
        which never pass through this method) with freshly random chromosomes ---
        ``GA_TechSpec.md`` §5's stagnation guard. Mutates and returns ``offspring``/
        ``offspring_makespans`` in place, for convenient testing of exactly which
        indices changed.
        """
        cfg = self.config
        n_immigrants = min(max(1, int(round(cfg.immigrant_fraction * len(offspring)))), len(offspring))
        worst_idx = np.argsort(offspring_makespans)[::-1][:n_immigrants]
        for i in worst_idx:
            offspring[i] = random_chromosome(instance, rng)
            offspring_makespans[i] = decode(instance, offspring[i]).makespan
        return offspring, offspring_makespans
