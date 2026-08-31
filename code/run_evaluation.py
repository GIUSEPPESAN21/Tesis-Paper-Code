"""Unified evaluation harness: instance set x methods x disruption profile -> one CSV.

Week 15 code-health sprint: this replaces ``experiments/eval_baselines.py`` as *the*
entry point for producing results the statistical analysis (Wilcoxon signed-rank,
Friedman, Nemenyi post-hoc) will consume, across every method family this codebase has
--- PDRs, GA, CP-SAT --- plus a schema-stable placeholder for the GNN+PPO policy the
next sprint trains. ``experiments/eval_baselines.py`` is kept as-is for its Week 9/12
result tables' provenance; it is not re-implemented on top of this module, to avoid
silently changing numbers a prior report already cited.

Usage
-----
    python run_evaluation.py
    python run_evaluation.py --instances ft06 la01 la02 --methods pdr ga cpsat rl
    python run_evaluation.py --disruption breakdowns --seeds 0 1 2 3 4
    python run_evaluation.py --fjsp --methods ga cpsat --instances ft06 la01

Disruption profiles and what "evaluate under disruption" means per method family
----------------------------------------------------------------------------------
``--disruption`` selects one of five :class:`~envs.disruptions.DisruptionConfig`
profiles: ``none`` (the regression-contract all-off default), ``breakdowns``,
``stochastic_times``, ``arrivals``, ``combined``. What running a method "under" one of
these means differs by method family, because a PDR is an *online* policy and GA/CP-SAT
are *offline* solvers:

Naming note (Recomendaciones 1/10 del tutor): E4/E5/E6 are **nine separate regimes of
three disruption classes** (``MULTIPLE_DISRUPTION_CLASSES`` below) -- three levels each
of breakdowns, stochastic processing times, and arrivals, evaluated as independent
single-axis sweeps, never all three at once unless ``--disruption combined`` is
explicitly requested. Describing E4/E5/E6 as "simultaneous disruptions" is wrong: only
``combined`` turns all three axes on in the same run, and it is a separate, secondary
profile with its own parametrized severities (``--combined-severity``), not what E4/E5/E6
measure.

* **PDR** --- already an online ``(env, rng) -> op_id`` policy; evaluating it under
  disruption is a direct rollout against a live
  :class:`~envs.dynamic_jssp_env.DynamicJSSPEnv`, exactly as
  ``tests/test_dynamic_env.py`` already does. No new machinery needed.
* **GA / CP-SAT** --- both only ever see the *nominal* instance; neither has any notion
  of "solve while machines might break." The standard way to make an offline solver
  evaluable under disruption is *predictive-reactive execution*
  (:func:`run_predictive_reactive`): solve the nominal instance once to get a planned
  dispatch order, then execute that order against a live, disrupted
  ``DynamicJSSPEnv`` -- dispatching the next still-legal planned operation whenever
  possible, and falling back to the already-tested, arrival-safe ``lwkr`` rule for
  anything the plan never anticipated (chiefly: a dynamically arrived job, which by
  construction cannot appear in a plan computed before it existed). This is glue
  necessary to make the four method families comparable at all under disruption, not a
  new scheduling algorithm -- see ``REFACTOR_NOTES_WEEK15.md`` for the full rationale.
* **RL** --- without ``--rl-policy``, :class:`~baselines.interface.RLPlaceholderBaseline`
  returns the same "not implemented" row regardless of disruption, so the CSV schema is
  correct even before a policy exists. With ``--rl-policy <checkpoint.zip>`` the slot is
  filled by :class:`~baselines.interface.TrainedRLBaseline`, which --- like a PDR, and
  unlike GA/CP-SAT --- rolls the policy out directly against a freshly constructed
  environment. Because the trained policy's parameter count is independent of instance
  size, one adapter serves every instance in a sweep. It is restricted to
  ``--disruption none``; see ``GNN_PPO_Architecture_Spec_v2.md`` §8.3 and
  :func:`validate_methods_for_disruption` for why that guard is explicit rather than
  silent.

FJSP (``--fjsp``)
------------------
Evaluates flexible instances with ``--methods`` restricted to the families that support
machine eligibility sets (``ga``, ``cpsat`` --- routed to
:class:`~baselines.interface.CPSATFJSPBaseline` --- and, since Week 18, ``rl`` via
:class:`~envs.fjsp_env.FJSPEnv`'s joint ``(operation, machine)`` action space). An
instance name matching a real bundled flexible benchmark (``data/fjsp_instances/``,
e.g. ``mk01``) is loaded directly and carries its literature optimum, so its rows get a
real ``rpd_pct``; any other name is converted to a synthetic
:class:`~envs.fjsp_instance.FJSPInstance` via
:func:`~envs.fjsp_instance.generate_fjsp_instance` and leaves ``known_optimum``/
``rpd_pct`` blank, since a synthetic instance has no literature-proven optimum. PDR
methods are rejected at argument-resolution time with a clear error (see
``baselines/interface.py``'s module docstring for why). ``--fjsp`` and ``--disruption``
other than ``none`` are mutually exclusive: ``DynamicJSSPEnv`` models fixed routing only,
so there is no live env to roll a PDR or a predictive-reactive replay against once
machine eligibility is ragged.

CSV schema
----------
One row per ``(instance, method, disruption, seed)``, long format (every stats routine
downstream -- a paired Wilcoxon test between two methods, a Friedman test ranking all
methods per instance-block, Nemenyi's post-hoc pairwise comparison -- pivots this shape
directly, no reshaping needed): ``instance, family, num_jobs, num_machines, method,
disruption, seed, makespan, known_optimum, rpd_pct, wall_clock_s, feasible, status,
is_optimal, violations_count, timestamp``. ``known_optimum``/``rpd_pct`` are blank for
FJSP rows: a synthetically generated instance has no literature-proven optimum (its true
optimum is generally *better* than the base JSSP instance's, since routing flexibility
only helps).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from baselines.cpsat import dispatch_order_from_starts  # noqa: E402
from baselines.dispatching_rules import RULES, run_rule  # noqa: E402
from baselines.ga import GAConfig, is_fjsp  # noqa: E402
from baselines.interface import (  # noqa: E402
    Baseline,
    BaselineResult,
    CPSATBaseline,
    CPSATFJSPBaseline,
    GABaseline,
    PDRBaseline,
    RLPlaceholderBaseline,
    TrainedRLBaseline,
)
from envs.disruptions import DisruptionConfig  # noqa: E402
from envs.dynamic_jssp_env import DynamicJSSPEnv  # noqa: E402
from envs.fjsp_instance import FJSPInstance, generate_fjsp_instance  # noqa: E402
from envs.fjsp_loader import (  # noqa: E402
    available_fjsp_instances,
    fjsp_known_optimum,
    load_fjsp_by_name,
)
from envs.instance import JSSPInstance  # noqa: E402
from envs.loader import known_optimum, load_by_name  # noqa: E402

Instance = Union[JSSPInstance, FJSPInstance]

RESULTS_DIR = REPO_ROOT / "experiments" / "results"

#: Nine Part B6 PDRs by canonical registry name (excludes the "mopnr"/"mor" duplicate
#: key in baselines.dispatching_rules.RULES -- see that dict's own docstring).
NINE_PDR_NAMES = ["spt", "lpt", "mwkr", "lwkr", "mor", "lor", "fifo", "edd", "cr"]

DEFAULT_INSTANCES = ["ft06", "ft10", "ft20"] + [f"la{n:02d}" for n in range(1, 11)]

#: ``--methods`` group shorthands, expanded before resolution.
METHOD_GROUPS = {
    "pdr": NINE_PDR_NAMES,
    "all": NINE_PDR_NAMES + ["ga", "cpsat", "rl"],
}

DISRUPTION_CHOICES = ["none", "breakdowns", "stochastic_times", "arrivals", "combined"]

#: The three disruption classes E4/E5/E6 each sweep three levels of, independently.
#: "Nueve regimenes de tres clases de disrupcion" -- see the module docstring's naming
#: note. Never "simultaneous" unless the profile is explicitly "combined".
MULTIPLE_DISRUPTION_CLASSES = ("breakdowns", "stochastic_times", "arrivals")

#: Parametrized severities for ``--disruption combined`` (Recomendacion 1/10, T7.2):
#: three levels so the "combined" profile can be swept the same way E4/E5/E6 sweep
#: their single axes, rather than being pinned to whatever ``--availability``/
#: ``--proc-sigma``/``--utilization`` happen to default to. Levels mirror the frozen
#: D4 grid (``PreRegistration_FINAL.md`` Sec. 4): low = mildest level on each axis
#: (A=0.95, sigma=0.10, rho=0.70), medium = middle level, high = harshest level
#: (A=0.80, sigma=0.50, rho=0.95) -- (availability, proc_sigma, utilization) tuples.
COMBINED_SEVERITY_LEVELS: dict[str, tuple[float, float, float]] = {
    "low": (0.95, 0.10, 0.70),
    "medium": (0.90, 0.25, 0.85),
    "high": (0.80, 0.50, 0.95),
}

#: Arrival-safe fallback rule for predictive-reactive replay (see module docstring).
#: Must NOT be spt/lpt/mwkr -- those index the base instance's fixed-shape arrays
#: directly and are unsafe once a dynamically arrived job's operation is legal (see
#: baselines/dispatching_rules.py's module docstring).
_FALLBACK_RULE_NAME = "lwkr"

#: PDR names that are unsafe to roll out live once arrivals are enabled -- as of T8.1
#: (Recomendacion del tutor: ampliar cobertura de PDR bajo llegadas), only edd/cr
#: remain unsafe: due-date synthesis for a dynamically arrived job is not implemented
#: (see baselines/due_dates.py and baselines/dispatching_rules.py's
#: ``_reject_arrived_ops``), which both rules raise their own NotImplementedError for.
#: spt/lpt/mwkr were unsafe before T8.1 (they indexed the base instance's fixed-shape
#: arrays directly, a raw IndexError the instant an arrived job's operation became
#: legal) and are now arrival-safe via ``env.duration_per_op()``/``env.machine_per_op()``/
#: ``env.remaining_work_per_op()``. Rejected here at argument-resolution time so an
#: --disruption arrivals/combined sweep with edd/cr fails fast with one clear message
#: instead of crashing (or silently erroring) partway through.
ARRIVAL_UNSAFE_PDR_NAMES = {"edd", "cr"}

_FAMILY_PREFIXES = {"ft": "fisher-thompson", "la": "lawrence", "ab": "adams-balas-zawack"}


def family_of(instance_name: str) -> str:
    """Benchmark family label for a CSV row, mirroring ``scripts/verify_optima.py``."""
    if instance_name.endswith("_fjsp"):
        return "fjsp-synthetic"
    if instance_name.startswith("mk"):
        return "brandimarte"
    return _FAMILY_PREFIXES.get(instance_name[:2], "unknown")


def known_optimum_for(instance: Instance) -> int | None:
    """Reference makespan for RPD, or ``None`` if none is recorded.

    A *synthetic* FJSP instance has no literature optimum (its true optimum is generally
    better than its base JSSP instance's, since routing flexibility only helps), which is
    why the FJSP branch returned ``None`` unconditionally before Week 18.  A *real*
    Brandimarte instance does have one, recorded in ``data/fjsp_known_optima.csv``, so
    those rows now carry a proper RPD like every JSSP row --- without which an RL-vs-GA
    comparison on ``mk01`` would have no normalised column to compare on.
    """
    if is_fjsp(instance):
        try:
            return fjsp_known_optimum(instance.name)
        except (KeyError, FileNotFoundError):
            return None
    try:
        return known_optimum(instance.name)
    except KeyError:
        return None


def build_disruption_config(
    profile: str,
    *,
    availability: float,
    proc_sigma: float,
    utilization: float,
    max_dynamic_jobs: int,
    combined_severity: str | None = None,
) -> DisruptionConfig:
    """Map a named ``--disruption`` profile to a concrete :class:`DisruptionConfig`.

    Args:
        combined_severity: Only consulted when ``profile == "combined"``. One of
            :data:`COMBINED_SEVERITY_LEVELS` (``"low"``/``"medium"``/``"high"``), which
            overrides ``availability``/``proc_sigma``/``utilization`` with the matching
            D4-grid triple so ``combined`` can be swept at three parametrized
            severities (T7.2). ``None`` (default) keeps the pre-existing behaviour of
            using the explicit ``--availability``/``--proc-sigma``/``--utilization``
            values as-is -- this is a backward-compatible additive flag, not a breaking
            change to any already-run ``combined`` sweep.
    """
    if profile == "none":
        return DisruptionConfig()
    if profile == "breakdowns":
        return DisruptionConfig(target_availability=availability)
    if profile == "stochastic_times":
        return DisruptionConfig(processing_time_sigma=proc_sigma)
    if profile == "arrivals":
        return DisruptionConfig(target_utilization=utilization, max_dynamic_jobs=max_dynamic_jobs)
    if profile == "combined":
        if combined_severity is not None:
            if combined_severity not in COMBINED_SEVERITY_LEVELS:
                raise ValueError(
                    f"unknown --combined-severity {combined_severity!r}. "
                    f"Known: {sorted(COMBINED_SEVERITY_LEVELS)}"
                )
            availability, proc_sigma, utilization = COMBINED_SEVERITY_LEVELS[combined_severity]
        return DisruptionConfig(
            target_availability=availability,
            processing_time_sigma=proc_sigma,
            target_utilization=utilization,
            max_dynamic_jobs=max_dynamic_jobs,
        )
    raise ValueError(f"unknown disruption profile {profile!r}. Known: {DISRUPTION_CHOICES}")


def load_instances(names: list[str], *, fjsp: bool, fjsp_seed: int) -> list[Instance]:
    """Load bundled instances by name, optionally converting each to FJSP.

    Under ``--fjsp``, a name that matches a *real* bundled flexible benchmark
    (``data/fjsp_instances/``, e.g. ``mk01``) is loaded directly; anything else is a
    JSSP name converted to a synthetic FJSP variant by
    :func:`~envs.fjsp_instance.generate_fjsp_instance`, exactly as before.  Real
    benchmarks take precedence because they are the ones with a literature optimum to
    report RPD against.
    """
    real_fjsp = set(available_fjsp_instances())
    instances: list[Instance] = []
    for name in names:
        if fjsp and name in real_fjsp:
            instances.append(load_fjsp_by_name(name))
            continue
        base = load_by_name(name)
        instances.append(generate_fjsp_instance(base, seed=fjsp_seed) if fjsp else base)
    return instances


def resolve_methods(
    method_specs: list[str],
    *,
    fjsp: bool,
    num_workers: int,
    population_size: int | None,
    rl_policy: str | None = None,
    rl_deterministic: bool = True,
) -> list[Baseline]:
    """Expand ``--methods`` (rule names, ``"ga"``/``"cpsat"``/``"rl"``, or a group
    shorthand) into concrete :class:`~baselines.interface.Baseline` adapters.

    Args:
        method_specs: The raw ``--methods`` values.
        fjsp: Whether the run is over FJSP instances.
        num_workers: CP-SAT search worker threads.
        population_size: GA population size override.
        rl_policy: Path to a trained policy checkpoint.  When given, ``"rl"`` resolves to
            :class:`~baselines.interface.TrainedRLBaseline` instead of
            :class:`~baselines.interface.RLPlaceholderBaseline` --- the one-line swap
            ``GNN_PPO_Architecture_Spec_v2.md`` §8.2 specifies.  The placeholder is kept
            as the no-checkpoint default so the CSV schema stays stable for a run that
            has no trained policy to point at yet.

    Raises:
        SystemExit: If a name is unrecognised, or a PDR name is requested together
            with ``fjsp=True`` (see the module docstring for why PDRs have no FJSP
            adapter).
    """
    expanded: list[str] = []
    for spec in method_specs:
        expanded.extend(METHOD_GROUPS.get(spec, [spec]))
    seen: set[str] = set()
    names = [n for n in expanded if not (n in seen or seen.add(n))]

    baselines: list[Baseline] = []
    for name in names:
        if name in RULES:
            if fjsp:
                raise SystemExit(
                    f"--methods {name!r} is a PDR and --fjsp was given: priority "
                    "dispatching rules have no machine-selection rule for FJSP "
                    "(see baselines/interface.py's module docstring). Use ga/cpsat."
                )
            baselines.append(PDRBaseline(name))
        elif name == "ga":
            baselines.append(GABaseline(GAConfig(population_size=population_size)))
        elif name == "cpsat":
            baselines.append(
                CPSATFJSPBaseline(num_workers=num_workers)
                if fjsp
                else CPSATBaseline(num_workers=num_workers)
            )
        elif name == "rl":
            baselines.append(
                TrainedRLBaseline(policy_path=rl_policy, deterministic=rl_deterministic)
                if rl_policy
                else RLPlaceholderBaseline()
            )
        else:
            raise SystemExit(
                f"unknown method {name!r}. Known: {sorted(RULES)} + ['ga', 'cpsat', 'rl'], "
                f"or a group name in {sorted(METHOD_GROUPS)}"
            )
    return baselines


def validate_methods_for_disruption(methods: list[Baseline], disruption_name: str) -> None:
    """Fail fast if a resolved method cannot run live under ``disruption_name``.

    Raises:
        SystemExit: If ``disruption_name`` enables arrivals (``"arrivals"``/``"combined"``)
            and a resolved :class:`~baselines.interface.PDRBaseline` is one of
            :data:`ARRIVAL_UNSAFE_PDR_NAMES`.
    """
    # Week 24 E4-E6 wiring fix: TrainedRLBaseline.solve_dynamic rolls the trained policy
    # out live against DynamicJSSPEnv (baselines/interface.py's
    # _DynamicNodeFeatureAdapter reconciles the one genuine observation-contract
    # difference -- see that class's docstring for the full diff and the quantified,
    # documented limitation this implies versus a policy retrained on the dynamic
    # contract directly). No longer blocked here; --fjsp+disruption is still blocked
    # earlier in main() since DynamicJSSPEnv models fixed routing only.
    if disruption_name not in ("arrivals", "combined"):
        return
    unsafe = [b.name for b in methods if isinstance(b, PDRBaseline) and b.name in ARRIVAL_UNSAFE_PDR_NAMES]
    if unsafe:
        raise SystemExit(
            f"--methods {unsafe} cannot run under --disruption {disruption_name!r}: these "
            "PDRs are not arrival-safe (see baselines/dispatching_rules.py's module "
            f"docstring). Use one of {sorted(RULES.keys() - ARRIVAL_UNSAFE_PDR_NAMES)} instead."
        )


def run_pdr_online(rule_name: str, instance: Instance, disruption_config: DisruptionConfig, seed: int) -> BaselineResult:
    """Roll a PDR out live against a disrupted :class:`DynamicJSSPEnv`."""
    env = DynamicJSSPEnv(instance, disruption_config)
    t0 = time.perf_counter()
    rollout = run_rule(env, rule_name, seed=seed)
    wall_clock_time = time.perf_counter() - t0
    return BaselineResult(
        instance=env.instance.name,
        method=rule_name,
        start_times=env.start_times.copy(),
        makespan=float(rollout["makespan"]),
        wall_clock_time=wall_clock_time,
        feasible=bool(rollout["feasible"]),
        violations=list(rollout["violations"]),
        status="OK",
        is_optimal=False,
        extra={"dispatch_order": rollout["dispatch_order"]},
    )


def run_rl_online(
    baseline: TrainedRLBaseline, instance: Instance, disruption_config: DisruptionConfig, seed: int
) -> BaselineResult:
    """Roll the trained policy out live against a disrupted :class:`DynamicJSSPEnv`,
    like a PDR (:func:`run_pdr_online`) -- unlike GA/CP-SAT, which have no online
    ``(env) -> action`` mapping and so need :func:`run_predictive_reactive` instead."""
    return baseline.solve_dynamic(instance, disruption_config, seed=seed)


#: T6.2 (Recomendacion 9 del tutor): three execution policies for offline methods
#: (ga/cpsat) under disruption. "right_shift" is the pre-T6.2 predictive-reactive
#: replay (unchanged default); "periodic"/"event_triggered" route through
#: envs.rescheduling's genuine reoptimization policies (T6.1) instead.
EXECUTION_POLICY_CHOICES = ["right_shift", "periodic", "event_triggered"]


def run_reoptimizing(
    baseline: Baseline,
    instance: Instance,
    disruption_config: DisruptionConfig,
    seed: int,
    *,
    execution_policy: str,
    reopt_period: float,
    reopt_time_limit_s: float,
) -> BaselineResult:
    """T6.2: execute ``baseline`` (ga/cpsat) under ``execution_policy in
    {"periodic", "event_triggered"}`` via ``envs.rescheduling`` (T6.1) instead of
    the one-shot predictive-reactive replay :func:`run_predictive_reactive` uses.
    Records the extra reoptimization cost in ``extra`` -- T6.2's own instruction that
    "la reoptimizacion recupera calidad pero deja de ser gratuita" must be measurable,
    not just asserted.

    Raises:
        ValueError: If ``baseline.name`` is not ``"ga"`` or a CP-SAT variant, or
            ``execution_policy`` is not one of the two reoptimizing policies.
    """
    from envs.rescheduling import EventTrigger, RescheduleResult, event_triggered_reopt, periodic_reopt

    solver: str
    if baseline.name == "ga":
        solver = "ga"
    elif baseline.name.startswith("cpsat"):
        solver = "cpsat"
    else:
        raise ValueError(f"run_reoptimizing only supports ga/cpsat baselines, got {baseline.name!r}")

    if execution_policy == "periodic":
        result: RescheduleResult = periodic_reopt(
            instance, disruption_config, seed, period=reopt_period, solver=solver, budget_s=reopt_time_limit_s
        )
    elif execution_policy == "event_triggered":
        result = event_triggered_reopt(
            instance, disruption_config, seed,
            trigger=EventTrigger(on_breakdown=True, on_arrival=True, deviation_threshold=None),
            solver=solver, budget_s=reopt_time_limit_s,
        )
    else:
        raise ValueError(f"unknown reoptimizing execution_policy {execution_policy!r}")

    return BaselineResult(
        instance=instance.name,
        method=baseline.name,
        start_times=result.start_times,
        makespan=result.makespan,
        wall_clock_time=result.reoptimization_wall_clock_s,
        feasible=result.feasible,
        violations=result.violations,
        status="OK",
        is_optimal=False,
        extra={
            "execution_policy": execution_policy,
            "n_reoptimizations": result.n_reoptimizations,
            "reoptimization_wall_clock_s": result.reoptimization_wall_clock_s,
        },
    )


def run_predictive_reactive(
    baseline: Baseline,
    instance: Instance,
    disruption_config: DisruptionConfig,
    seed: int,
    time_limit_s: float,
) -> BaselineResult:
    """Solve ``instance`` offline, then execute the resulting plan reactively under
    disruption. See the module docstring for the full rationale."""
    t0 = time.perf_counter()
    offline = baseline.solve(instance, seed=seed, time_limit_s=time_limit_s)
    planned = list(dispatch_order_from_starts(instance, offline.start_times))
    num_planned = len(planned)

    env = DynamicJSSPEnv(instance, disruption_config)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    fallback_rule = RULES[_FALLBACK_RULE_NAME]

    terminated = False
    info: dict = {}
    while not terminated:
        legal = set(np.flatnonzero(env.action_masks()).tolist())
        action = next((op for op in planned if op in legal), None)
        if action is not None:
            planned.remove(action)
        else:
            action = fallback_rule(env, rng)
        _, _, terminated, _, info = env.step(action)

    wall_clock_time = time.perf_counter() - t0
    return BaselineResult(
        instance=env.instance.name,
        method=baseline.name,
        start_times=env.start_times.copy(),
        makespan=float(info["makespan"]),
        wall_clock_time=wall_clock_time,
        feasible=bool(info["schedule_feasible"]),
        violations=list(info["violations"]),
        status="OK",
        is_optimal=False,
        extra={"offline_makespan": offline.makespan, "planned_ops": num_planned},
    )


def evaluate_one(
    baseline: Baseline,
    instance: Instance,
    disruption_name: str,
    disruption_config: DisruptionConfig,
    seed: int,
    time_limit_s: float,
    *,
    execution_policy: str = "right_shift",
    reopt_period: float = 20.0,
    reopt_time_limit_s: float = 10.0,
) -> BaselineResult:
    """Dispatch to the right execution strategy for ``(baseline, disruption_name)``.

    Args:
        execution_policy: T6.2. Only consulted for ga/cpsat under a non-``"none"``
            disruption (PDR/RL are already online -- an execution policy is meaningless
            for them, so this argument is silently ignored in those branches, matching
            how ``--rl-sample``/etc. are already method-specific flags in this CLI).
    """
    if disruption_name == "none":
        return baseline.solve(instance, seed=seed, time_limit_s=time_limit_s)
    if baseline.name == "rl" and isinstance(baseline, TrainedRLBaseline):
        return run_rl_online(baseline, instance, disruption_config, seed)
    if baseline.name == "rl":
        # RLPlaceholderBaseline (no --rl-policy given): no live env to roll out against,
        # same schema-stable "not implemented" row it returns under --disruption none.
        return baseline.solve(instance, seed=seed, time_limit_s=time_limit_s)
    if baseline.name in RULES:
        return run_pdr_online(baseline.name, instance, disruption_config, seed)
    if execution_policy != "right_shift":
        return run_reoptimizing(
            baseline, instance, disruption_config, seed,
            execution_policy=execution_policy, reopt_period=reopt_period, reopt_time_limit_s=reopt_time_limit_s,
        )
    return run_predictive_reactive(baseline, instance, disruption_config, seed, time_limit_s)


#: Fixed column order for every row ``build_row`` produces -- pinned as a module
#: constant (rather than derived from ``rows[0].keys()`` at write time) so incremental
#: writes can open the CSV and write the header before the first row exists.
CSV_FIELDNAMES: list[str] = [
    "instance",
    "family",
    "num_jobs",
    "num_machines",
    "method",
    "disruption",
    "seed",
    "makespan",
    "known_optimum",
    "rpd_pct",
    "wall_clock_s",
    "feasible",
    "status",
    "is_optimal",
    "violations_count",
    "timestamp",
    "mip_gap",
    "execution_policy",
    "n_reoptimizations",
    "reoptimization_wall_clock_s",
]


def build_row(instance: Instance, method: str, disruption: str, seed: int, result: BaselineResult, timestamp: str) -> dict:
    """One standardized CSV row. See the module docstring for the column list.

    ``status``/``is_optimal``/``mip_gap`` are three distinct things (Recomendacion
    1/10 del tutor): ``status`` is the raw solver/method status string (``"OK"`` for
    PDR/GA/RL, CP-SAT's own status name for cpsat/cpsat_fjsp); ``is_optimal`` is
    ``True`` only when ``status == "OPTIMAL"``; ``mip_gap`` (CP-SAT rows only, ``NaN``
    otherwise -- ``result.extra`` has no ``"mip_gap"`` key for non-CP-SAT methods) is
    the quantitative residual gap on a non-optimal CP-SAT incumbent. A downstream table
    or figure caption must never read ``feasible=True`` or ``status="FEASIBLE"`` as
    "exacto-optimo" -- only ``is_optimal=True`` (equivalently ``mip_gap == 0.0``)
    licenses that label.
    """
    optimum = known_optimum_for(instance)
    rpd = (
        round(100.0 * (result.makespan - optimum) / optimum, 3)
        if (optimum is not None and np.isfinite(result.makespan))
        else float("nan")
    )
    return {
        "instance": instance.name,
        "family": family_of(instance.name),
        "num_jobs": instance.num_jobs,
        "num_machines": instance.num_machines,
        "method": method,
        "disruption": disruption,
        "seed": seed,
        "makespan": result.makespan,
        "known_optimum": optimum if optimum is not None else "",
        "rpd_pct": rpd,
        "wall_clock_s": round(result.wall_clock_time, 4),
        "feasible": result.feasible,
        "status": result.status,
        "is_optimal": result.is_optimal,
        "violations_count": len(result.violations),
        "timestamp": timestamp,
        "mip_gap": result.extra.get("mip_gap", float("nan")),
        "execution_policy": result.extra.get("execution_policy", "right_shift"),
        "n_reoptimizations": result.extra.get("n_reoptimizations", 0),
        "reoptimization_wall_clock_s": result.extra.get("reoptimization_wall_clock_s", 0.0),
    }


def run_evaluation(
    instances: list[Instance],
    methods: list[Baseline],
    disruption_name: str,
    disruption_config: DisruptionConfig,
    seeds: list[int],
    time_limit_s: float,
    *,
    verbose: bool = True,
    out_path: Path | None = None,
    execution_policy: str = "right_shift",
    reopt_period: float = 20.0,
    reopt_time_limit_s: float = 10.0,
) -> tuple[list[dict], list[dict]]:
    """Evaluate every ``(instance, method, seed)`` combination and return CSV rows.

    Hardened per ``E7_E8_FinalPlan.md`` §2.1 precondition #4 /
    ``EXPERIMENT_LOG_WEEK24.md`` Finding #3: previously, a single exception anywhere in
    the sweep (e.g. a checkpoint-mismatch ``ValueError`` on cell 400 of 540) propagated
    out of this function, so ``main()`` never reached ``write_csv`` and the *entire*
    sweep's output -- including hours of already-completed, valid rows -- was
    discarded. Now: (a) each cell is individually caught, so one failure only skips
    that one row and the sweep continues; (b) if ``out_path`` is given, every
    successfully computed row is written to disk immediately (and flushed), so a crash
    partway through -- in this function, or from something external like the process
    being killed -- loses at most the in-flight cell, not everything computed so far.

    Args:
        out_path: If given, rows are appended to this CSV incrementally as they are
            computed (header written once, upfront). If ``None``, no incremental write
            happens and the caller is responsible for calling ``write_csv`` on the
            returned rows (e.g. tests that only want the in-memory rows).

    Returns:
        ``(rows, failures)``. ``failures`` is one dict per ``(instance, method, seed)``
        cell that raised, with ``instance``/``method``/``disruption``/``seed``/
        ``error_type``/``error`` keys -- never silently dropped, always both printed as
        it happens and returned for the caller to log/inspect.
    """
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []
    failures: list[dict] = []

    handle = None
    writer = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        handle = out_path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

    try:
        for instance in instances:
            for baseline in methods:
                for seed in seeds:
                    try:
                        result = evaluate_one(
                            baseline, instance, disruption_name, disruption_config, seed, time_limit_s,
                            execution_policy=execution_policy, reopt_period=reopt_period,
                            reopt_time_limit_s=reopt_time_limit_s,
                        )
                        row = build_row(instance, baseline.name, disruption_name, seed, result, timestamp)
                    except Exception as exc:  # noqa: BLE001 -- deliberate catch-and-continue (see docstring)
                        failures.append(
                            {
                                "instance": instance.name,
                                "method": baseline.name,
                                "disruption": disruption_name,
                                "seed": seed,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        print(
                            f"FAILED    {instance.name:>10} {baseline.name:>10} disruption={disruption_name:<16} "
                            f"seed={seed:<3} {type(exc).__name__}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue

                    rows.append(row)
                    if writer is not None:
                        writer.writerow(row)
                        handle.flush()
                    if verbose:
                        rpd_display = f"{row['rpd_pct']:.2f}%" if row["rpd_pct"] == row["rpd_pct"] else "n/a"
                        print(
                            f"{row['instance']:>10} {row['method']:>10} disruption={disruption_name:<16} "
                            f"seed={seed:<3} Cmax={row['makespan']:<10.1f} RPD={rpd_display:>8} "
                            f"{row['wall_clock_s']:>8.3f}s feasible={row['feasible']} status={row['status']}",
                            flush=True,
                        )
    finally:
        if handle is not None:
            handle.close()

    if failures:
        print(
            f"\n{len(failures)} cell(s) failed and were skipped (see FAILED lines above); "
            f"{len(rows)} row(s) succeeded.",
            flush=True,
        )
    return rows, failures


def write_csv(rows: list[dict], path: Path) -> None:
    """Write ``rows`` (uniform-keyed dicts) to ``path`` as a header + data CSV.

    Raises:
        ValueError: If ``rows`` is empty --- an empty CSV has no header to validate,
            which would make ``tests/test_run_evaluation.py`` pass vacuously.
    """
    if not rows:
        raise ValueError("no rows to write; check --instances/--methods/--seeds are non-empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--instances", nargs="*", default=DEFAULT_INSTANCES, help="bundled instance names")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["mwkr", "spt", "fifo", "ga", "cpsat", "rl"],
        help="rule names from baselines.dispatching_rules.RULES, 'ga', 'cpsat', 'rl', "
        f"or a group shorthand ({sorted(METHOD_GROUPS)})",
    )
    parser.add_argument("--disruption", choices=DISRUPTION_CHOICES, default="none")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0], help="seeds to evaluate each method/instance with")
    parser.add_argument("--time-limit", type=float, default=60.0, help="CP-SAT/GA wall-clock budget, seconds")
    parser.add_argument("--num-workers", type=int, default=8, help="CP-SAT search worker threads")
    parser.add_argument("--population-size", type=int, default=None, help="GA population size (default: auto-scaled)")
    parser.add_argument(
        "--fjsp",
        action="store_true",
        help="evaluate FJSP variants: real bundled benchmarks by name (e.g. mk01), "
        "otherwise synthetic conversions of the named JSSP instances (ga/cpsat/rl only)",
    )
    parser.add_argument(
        "--rl-policy",
        default=None,
        help="path to a trained policy .zip (experiments/train_hetero.py); makes "
        "--methods rl use the trained policy instead of the schema-stable placeholder",
    )
    parser.add_argument(
        "--rl-sample",
        action="store_true",
        help="sample the trained policy's action distribution instead of taking the "
        "argmax action, so repeated --seeds produce genuinely different stochastic "
        "rollouts (see baselines.interface.TrainedRLBaseline's deterministic flag)",
    )
    parser.add_argument("--fjsp-seed", type=int, default=0, help="seed for the synthetic FJSP generator")
    parser.add_argument("--availability", type=float, default=0.8, help="breakdowns: target per-machine availability")
    parser.add_argument("--proc-sigma", type=float, default=0.25, help="stochastic_times: lognormal sigma")
    parser.add_argument("--utilization", type=float, default=0.6, help="arrivals: target shop utilization")
    parser.add_argument("--max-dynamic-jobs", type=int, default=20, help="arrivals: cap on arriving jobs")
    parser.add_argument(
        "--combined-severity",
        choices=sorted(COMBINED_SEVERITY_LEVELS),
        default=None,
        help="combined: pick one of three parametrized severities (D4-grid levels) "
        "instead of the raw --availability/--proc-sigma/--utilization values, so the "
        "multiple_disruption_classes co-occur at a matched low/medium/high level "
        "(T7.2). Ignored for any --disruption other than combined.",
    )
    parser.add_argument(
        "--execution-policy", choices=EXECUTION_POLICY_CHOICES, default="right_shift",
        help="T6.2 (Recomendacion 9): how ga/cpsat react to disruption. 'right_shift' "
        "(default, unchanged): one-shot predictive-reactive replay. 'periodic'/"
        "'event_triggered': genuine reoptimization via envs.rescheduling (T6.1). "
        "Ignored for pdr/rl (already online).",
    )
    parser.add_argument("--reopt-period", type=float, default=20.0, help="execution-policy periodic: simulated-time interval between reoptimizations")
    parser.add_argument("--reopt-time-limit", type=float, default=10.0, help="execution-policy periodic/event_triggered: residual-solve wall-clock budget per reoptimization")
    parser.add_argument("--out", default=None, help="output CSV path (default: experiments/results/run_evaluation.csv)")
    parser.add_argument("--quiet", action="store_true", help="suppress per-row progress printing")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.fjsp and args.disruption != "none":
        parser.error(
            "--fjsp only supports --disruption none: DynamicJSSPEnv models fixed "
            "routing only, so there is no live env to roll a disrupted FJSP episode "
            "against (see envs/fjsp_instance.py's module docstring)."
        )

    disruption_config = build_disruption_config(
        args.disruption,
        availability=args.availability,
        proc_sigma=args.proc_sigma,
        utilization=args.utilization,
        max_dynamic_jobs=args.max_dynamic_jobs,
        combined_severity=args.combined_severity,
    )
    instances = load_instances(args.instances, fjsp=args.fjsp, fjsp_seed=args.fjsp_seed)
    methods = resolve_methods(
        args.methods,
        fjsp=args.fjsp,
        num_workers=args.num_workers,
        population_size=args.population_size,
        rl_policy=args.rl_policy,
        rl_deterministic=not args.rl_sample,
    )
    validate_methods_for_disruption(methods, args.disruption)

    out_path = Path(args.out) if args.out else RESULTS_DIR / "run_evaluation.csv"
    rows, failures = run_evaluation(
        instances,
        methods,
        args.disruption,
        disruption_config,
        args.seeds,
        args.time_limit,
        verbose=not args.quiet,
        out_path=out_path,
        execution_policy=args.execution_policy,
        reopt_period=args.reopt_period,
        reopt_time_limit_s=args.reopt_time_limit,
    )

    if not rows:
        # Every cell failed: out_path has a header and no data rows. write_csv's own
        # empty-rows guard would reject this even though a file already exists on disk
        # (written incrementally above), so raise here with the actual failure count
        # rather than a generic "no rows" message.
        raise SystemExit(f"all {len(failures)} cell(s) failed; see FAILED lines above. No usable rows in {out_path}")

    suffix = f" ({len(failures)} cell(s) failed and were skipped, see above)" if failures else ""
    print(f"\nwrote {len(rows)} rows to {out_path}{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
