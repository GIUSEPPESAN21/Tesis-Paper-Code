"""Priority dispatching rules (PDRs) as reference policies over :class:`~envs.jssp_env.JSSPEnv`.

Every rule is a function ``(env, rng) -> op_id`` that picks one operation from the
currently legal set.  They serve three purposes in this thesis:

1. **Sanity checks** --- a PDR-generated schedule must satisfy every JSSP constraint, so
   running the rules is the cheapest end-to-end test of the environment's transition.
2. **Baselines** --- the classic comparison point for any learned dispatcher.  Week 9
   completes the full 9-rule PDR suite required by Part B6 (SPT, LPT, MWKR, LWKR, MOR,
   LOR, FIFO, EDD, CR); see ``Baselines_TechSpec.md`` and ``BASELINES_RESULTS_WEEK9.md``.
3. **Behavioural reference** --- a learned policy that cannot beat SPT is not learning.

Rules implemented here operate on the *frontier*: at any decision epoch each unfinished
job exposes exactly one dispatchable operation, and a rule ranks those.

Arrival-safety (``Baselines_TechSpec.md`` §3.2)
------------------------------------------------
``spt_rule``, ``lpt_rule``, and ``mwkr_rule`` index the *base* ``JSSPInstance``'s
fixed-shape arrays directly by global operation id, which cannot cover an operation
belonging to a dynamically arrived job (see ``VALIDATION_WEEK6.md`` decision #2) --- they
are frozen as-is, unchanged this sprint. The four new rules (LWKR, LOR, EDD, CR) instead
use ``env.remaining_work_per_op()`` (an additive method on both
:class:`~envs.jssp_env.JSSPEnv` and :class:`~envs.dynamic_jssp_env.DynamicJSSPEnv`) and
pure ``job_of``/``position_of`` arithmetic, so they work unmodified on both environments
for every job present at ``reset()`` time. Due-date synthesis for dynamically *arrived*
jobs is not implemented this sprint (see ``baselines/due_dates.py``); this only matters
if EDD/CR are run against a ``DynamicJSSPEnv`` with arrivals actually enabled, which no
evaluation in this sprint does (STEP3 covers static ``ft``/``la`` instances only).
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from baselines.due_dates import ensure_due_dates
from envs.jssp_env import JSSPEnv


class DispatchingRule(Protocol):
    """Callable protocol for a dispatching rule."""

    def __call__(self, env: JSSPEnv, rng: np.random.Generator) -> int:
        """Return the global operation id to dispatch next."""
        ...


def _legal_ops(env: JSSPEnv) -> np.ndarray:
    """Global ids of the currently dispatchable operations.

    Raises:
        RuntimeError: If called on a terminated episode.
    """
    legal = np.flatnonzero(env.action_masks())
    if legal.size == 0:
        raise RuntimeError("no legal action: episode already terminated")
    return legal


def _argmin_with_tiebreak(
    legal: np.ndarray, keys: np.ndarray, rng: np.random.Generator
) -> int:
    """Pick the legal operation with the smallest key, breaking ties uniformly at random.

    Random tie-breaking matters: benchmark instances have many equal processing times,
    and deterministic ``argmin`` tie-breaking silently biases a rule toward low job
    indices, which inflates or deflates its measured performance.
    """
    best = keys == keys.min()
    candidates = legal[best]
    return int(candidates[rng.integers(len(candidates))]) if candidates.size > 1 else int(candidates[0])


def random_rule(env: JSSPEnv, rng: np.random.Generator) -> int:
    """Uniformly random legal dispatch. The weakest meaningful baseline."""
    legal = _legal_ops(env)
    return int(legal[rng.integers(len(legal))])


def spt_rule(env: JSSPEnv, rng: np.random.Generator) -> int:
    """Shortest Processing Time: dispatch the frontier operation with the smallest duration."""
    legal = _legal_ops(env)
    return _argmin_with_tiebreak(legal, env.instance.flat_durations[legal], rng)


def lpt_rule(env: JSSPEnv, rng: np.random.Generator) -> int:
    """Longest Processing Time: the mirror image of :func:`spt_rule`."""
    legal = _legal_ops(env)
    return _argmin_with_tiebreak(legal, -env.instance.flat_durations[legal], rng)


def fifo_rule(env: JSSPEnv, rng: np.random.Generator) -> int:
    """First In First Out: dispatch whichever frontier operation could start earliest.

    Ties (very common early in an episode, when several jobs are ready at time 0) are
    broken at random.
    """
    legal = _legal_ops(env)
    ready = np.array(
        [env.job_ready_time[env.instance.job_of(int(op))] for op in legal], dtype=np.float64
    )
    return _argmin_with_tiebreak(legal, ready, rng)


def mwkr_rule(env: JSSPEnv, rng: np.random.Generator) -> int:
    """Most Work Remaining: prioritise the job with the largest unscheduled workload.

    Consistently one of the strongest single-pass PDRs for makespan on the ``ft``/``la``
    families, and the usual PDR that learned dispatchers are asked to beat.
    """
    legal = _legal_ops(env)
    remaining = env.instance.remaining_work().ravel()[legal]
    return _argmin_with_tiebreak(legal, -remaining, rng)


def mopnr_rule(env: JSSPEnv, rng: np.random.Generator) -> int:
    """Most Operations Remaining (a.k.a. MOR, Part B6): prioritise the job with the most
    unscheduled operations. Registered under both ``"mopnr"`` (this function's original
    name) and ``"mor"`` (the Part B6 name) in :data:`RULES` --- see that dict's docstring.
    """
    legal = _legal_ops(env)
    ops_per_job = env.instance.ops_per_job
    remaining = np.array(
        [ops_per_job - env.instance.position_of(int(op)) for op in legal], dtype=np.float64
    )
    return _argmin_with_tiebreak(legal, -remaining, rng)


def lwkr_rule(env: JSSPEnv, rng: np.random.Generator) -> int:
    """Least Work Remaining: the mirror image of :func:`mwkr_rule`.

    Arrival-safe: uses ``env.remaining_work_per_op()`` rather than indexing the base
    instance's fixed-shape ``remaining_work()`` array directly (see the module
    docstring).
    """
    legal = _legal_ops(env)
    remaining = env.remaining_work_per_op()[legal]
    return _argmin_with_tiebreak(legal, remaining, rng)


def lor_rule(env: JSSPEnv, rng: np.random.Generator) -> int:
    """Least Operations Remaining: the mirror image of :func:`mopnr_rule`.

    Arrival-safe by construction: pure ``job_of``/``position_of`` arithmetic, exactly
    like :func:`mopnr_rule`.
    """
    legal = _legal_ops(env)
    ops_per_job = env.instance.ops_per_job
    remaining = np.array(
        [ops_per_job - env.instance.position_of(int(op)) for op in legal], dtype=np.float64
    )
    return _argmin_with_tiebreak(legal, remaining, rng)


def _reject_arrived_ops(env: JSSPEnv, legal: np.ndarray, rule_name: str) -> None:
    """Raise a clear error instead of a confusing ``IndexError`` if ``legal`` contains an
    operation belonging to a dynamically arrived job.

    ``edd_rule``/``cr_rule`` are safe for every job present at ``reset()`` time (static
    ``JSSPEnv``, or ``DynamicJSSPEnv`` with breakdowns/stochastic durations but arrivals
    disabled --- both exercised by this sprint's tests), but due-date synthesis for a
    dynamically *arrived* job (release time = actual arrival time,
    ``Baselines_TechSpec.md`` §4.2) is out of scope this sprint, and ``cr_rule``
    additionally looks up an operation's machine via the base instance's fixed-shape
    ``flat_machines``. Both would otherwise fail with a bare ``IndexError`` deep inside
    array indexing the moment an arrived job's operation became legal.
    """
    if int(legal.max()) >= env.instance.num_operations:
        raise NotImplementedError(
            f"{rule_name} does not support dynamically arrived jobs this sprint "
            "(due-date synthesis for arrivals is out of scope; see "
            "baselines/due_dates.py and BASELINES_RESULTS_WEEK9.md). Use lwkr/lor/mor "
            "(or any of the six original rules) against DynamicJSSPEnv with arrivals "
            "enabled instead."
        )


def edd_rule(env: JSSPEnv, rng: np.random.Generator) -> int:
    """Earliest Due Date: prioritise the job with the nearest synthesized due date.

    Due dates (``env.due_dates``) are synthesized via the Total Work Content rule on
    first use and cached on ``env`` --- see :func:`baselines.due_dates.ensure_due_dates`.
    """
    legal = _legal_ops(env)
    _reject_arrived_ops(env, legal, "edd_rule")
    due_dates = ensure_due_dates(env)
    keys = np.array(
        [due_dates[env.instance.job_of(int(op))] for op in legal], dtype=np.float64
    )
    return _argmin_with_tiebreak(legal, keys, rng)


def cr_rule(env: JSSPEnv, rng: np.random.Generator) -> int:
    """Critical Ratio: ``(due_date - t_now) / remaining_work``; smallest ratio first.

    ``t_now`` for the candidate operation is resolved, per ``Baselines_TechSpec.md``
    §9.2, as ``max(job_ready_time[job], machine_free_time[machine])`` --- the operation's
    own earliest feasible start, i.e. the instant it *would* begin if dispatched now.
    ``JSSPEnv`` has no global simulation clock by design (Section 4.3.2), so this proxy is
    what makes CR well-defined without adding new environment state; a live simulated
    clock (as ``DynamicJSSPEnv`` actually has) would be the more literal reading of Part
    B6's ``(d_j - t)``, but is not used here for consistency across both environments.
    """
    legal = _legal_ops(env)
    _reject_arrived_ops(env, legal, "cr_rule")
    due_dates = ensure_due_dates(env)
    remaining = env.remaining_work_per_op()[legal]
    keys = np.empty(legal.shape, dtype=np.float64)
    for i, op in enumerate(legal):
        op = int(op)
        job = env.instance.job_of(op)
        machine = int(env.instance.flat_machines[op])
        t_now = max(float(env.job_ready_time[job]), float(env.machine_free_time[machine]))
        keys[i] = (due_dates[job] - t_now) / max(float(remaining[i]), 1e-9)
    return _argmin_with_tiebreak(legal, keys, rng)


#: Registry of the rules available by name. Eleven keys covering the nine Part B6 PDRs
#: (``"mopnr"``/``"mor"`` are two keys for the *same* callable --- the function keeps its
#: original name per ``Baselines_TechSpec.md`` §3.1, "keep the name, add `mor` as a
#: registry alias, do not rename the function" --- so ``len(RULES) == 11`` but
#: ``len(set(RULES.values())) == 10`` (nine unique PDRs + random), which is the
#: tripwire ``tests/test_env.py``/``tests/test_dynamic_env.py`` actually assert.
RULES: dict[str, Callable[[JSSPEnv, np.random.Generator], int]] = {
    "random": random_rule,
    "spt": spt_rule,
    "lpt": lpt_rule,
    "fifo": fifo_rule,
    "mwkr": mwkr_rule,
    "mopnr": mopnr_rule,
    "mor": mopnr_rule,
    "lwkr": lwkr_rule,
    "lor": lor_rule,
    "edd": edd_rule,
    "cr": cr_rule,
}

#: Rules that cannot be made arrival-safe and are therefore excluded from ``RULES``.
#: Empty this sprint --- all four new rules passed the arrival-safety requirement (§3.2)
#: --- but exported for forward-compatibility per ``Baselines_TechSpec.md`` §8.
STATIC_ONLY_RULES: dict[str, Callable[[JSSPEnv, np.random.Generator], int]] = {}


def run_rule(
    env: JSSPEnv, rule: str | Callable[[JSSPEnv, np.random.Generator], int], seed: int = 0
) -> dict[str, object]:
    """Roll out one full episode under a dispatching rule.

    Args:
        env: The environment (reset internally).
        rule: A rule name from :data:`RULES` or a callable following
            :class:`DispatchingRule`.
        seed: Seed for both the environment reset and the tie-breaking RNG.

    Returns:
        Dict with ``instance``, ``rule``, ``makespan``, ``return`` (undiscounted sum of
        rewards), ``feasible``, ``violations`` and ``dispatch_order``.

    Raises:
        KeyError: If ``rule`` is a name not present in :data:`RULES`.
    """
    fn = RULES[rule] if isinstance(rule, str) else rule
    rule_name = rule if isinstance(rule, str) else getattr(rule, "__name__", "custom")

    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    info: dict = {}
    while not terminated:
        action = fn(env, rng)
        _, reward, terminated, _, info = env.step(action)
        total_reward += reward

    return {
        "instance": env.instance.name,
        "rule": rule_name,
        "makespan": info["makespan"],
        "return": total_reward,
        "feasible": info["schedule_feasible"],
        "violations": info["violations"],
        "dispatch_order": env.dispatch_order,
    }
