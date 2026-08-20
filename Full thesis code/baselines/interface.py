"""Common interface shared by every baseline scheduling method.

``Baseline.solve(instance, seed=None, time_limit_s=None) -> BaselineResult`` is the one
call signature ``run_evaluation.py`` (and, before it, ``experiments/eval_baselines.py``)
uses uniformly across every method family --- the nine PDRs (:class:`PDRBaseline`), the
genetic algorithm (:class:`GABaseline`), and CP-SAT (:class:`CPSATBaseline`) --- regardless
of what happens inside (a constructive rollout for a PDR, a population search for the GA,
an exact solve for CP-SAT).

FJSP (machine eligibility sets)
--------------------------------
``GABaseline`` and :class:`CPSATFJSPBaseline` both accept an
:class:`~envs.fjsp_instance.FJSPInstance` in place of a
:class:`~envs.instance.JSSPInstance` --- :class:`~baselines.ga.GeneticAlgorithm` already
implements the assignment-vector chromosome for either instance type (see
:func:`~baselines.ga.is_fjsp`), and :class:`CPSATFJSPBaseline` wraps
:func:`~baselines.cpsat.solve_cpsat_fjsp`'s ``OptionalIntervalVar`` model. ``PDRBaseline``
has no FJSP counterpart this sprint: a priority dispatching rule additionally needs a
*machine-selection* rule once an operation has more than one eligible machine, and
inventing one is a research decision, not a refactor --- see ``PDRBaseline.solve``'s
``TypeError`` for the explicit guard, and ``REFACTOR_NOTES_WEEK15.md`` for the full
scoping note.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from baselines.cpsat import is_fjsp_schedule_feasible, solve_cpsat, solve_cpsat_fjsp
from baselines.dispatching_rules import RULES, run_rule
from baselines.ga import GAConfig, GeneticAlgorithm, Instance, is_fjsp
from envs.disjunctive_graph import is_schedule_feasible
from envs.fjsp_instance import FJSPInstance
from envs.instance import JSSPInstance
from envs.jssp_env import JSSPEnv, NUM_NODE_FEATURES

if TYPE_CHECKING:  # pragma: no cover - import cost only paid by type checkers
    from envs.base_env import SchedulingEnv
    from envs.disruptions import DisruptionConfig


@dataclass(frozen=True)
class BaselineResult:
    """Uniform output of any baseline method.

    Attributes:
        instance: ``instance.name``.
        method: Short identifier, e.g. ``"spt"``, ``"cpsat"``.
        start_times: Start time per global operation id, same convention as
            ``JSSPEnv.start_times``. The single canonical schedule representation across
            every method, so ``is_schedule_feasible(instance, result.start_times)`` can
            be called identically regardless of which method produced it.
        makespan: Best makespan found (``float("inf")`` if no feasible schedule found).
        wall_clock_time: Seconds, measured by the adapter itself.
        feasible: Result of an independent :func:`~envs.disjunctive_graph.is_schedule_feasible`
            check.
        violations: Empty iff ``feasible``.
        status: ``"OK"`` (PDR always), or CP-SAT's solver status
            (``"OPTIMAL"``/``"FEASIBLE"``/``"INFEASIBLE"``/``"UNKNOWN"``).
        is_optimal: ``True`` only when ``status == "OPTIMAL"``; always ``False`` for PDR.
        extra: Method-specific detail (PDR: ``dispatch_order``; CP-SAT: ``best_bound``).
    """

    instance: str
    method: str
    start_times: np.ndarray
    makespan: float
    wall_clock_time: float
    feasible: bool
    violations: list[str]
    status: str
    is_optimal: bool
    extra: dict = field(default_factory=dict)


class Baseline(Protocol):
    """Callable protocol every concrete baseline adapter satisfies."""

    name: str  # short identifier used as BaselineResult.method and in result tables

    def solve(
        self,
        instance: Instance,
        *,
        seed: int | None = None,
        time_limit_s: float | None = None,
    ) -> BaselineResult:
        """Solve ``instance`` and return a :class:`BaselineResult`.

        Args:
            instance: The instance to solve --- a :class:`~envs.instance.JSSPInstance`
                for every adapter, or an :class:`~envs.fjsp_instance.FJSPInstance` for
                ``GABaseline``/:class:`CPSATFJSPBaseline` (``PDRBaseline`` rejects it,
                see its own ``solve``).
            seed: Governs stochastic tie-breaking. ``None``-safe (falls back to entropy).
            time_limit_s: Wall-clock budget. Required (non-``None``) for CP-SAT --- see
                :meth:`CPSATBaseline.solve`. Ignored by PDR adapters (a single
                deterministic pass is not meaningfully time-limited) but still accepted
                in the signature so every ``Baseline`` is call-compatible.
        """
        ...


class PDRBaseline:
    """Adapter running one priority dispatching rule over :class:`JSSPEnv`.

    Args:
        rule_name: A key of :data:`~baselines.dispatching_rules.RULES`, e.g. ``"mwkr"``.

    Raises:
        KeyError: If ``rule_name`` is not registered in ``RULES``.
    """

    def __init__(self, rule_name: str) -> None:
        if rule_name not in RULES:
            raise KeyError(f"unknown rule {rule_name!r}. Available: {sorted(RULES)}")
        self.name = rule_name

    def solve(
        self,
        instance: Instance,
        *,
        seed: int | None = None,
        time_limit_s: float | None = None,
    ) -> BaselineResult:
        """Roll out this rule to completion. ``time_limit_s`` is accepted and ignored.

        Raises:
            TypeError: If ``instance`` is an :class:`~envs.fjsp_instance.FJSPInstance`.
                A PDR ranks *among the legal frontier operations*; it has no rule for
                *which eligible machine* to run one on once that set has more than one
                member, and inventing one is a new algorithm, not a refactor (see the
                module docstring). Use ``GABaseline``/``CPSATFJSPBaseline`` for FJSP.
        """
        if is_fjsp(instance):
            raise TypeError(
                f"PDRBaseline({self.name!r}) does not support FJSPInstance: priority "
                "dispatching rules have no machine-selection rule for an operation with "
                "more than one eligible machine. Use GABaseline or CPSATFJSPBaseline "
                "for FJSP instances instead."
            )
        env = JSSPEnv(instance)
        actual_seed = seed if seed is not None else int(np.random.SeedSequence().entropy % 2**31)

        t0 = time.perf_counter()
        rollout = run_rule(env, self.name, seed=actual_seed)
        wall_clock_time = time.perf_counter() - t0

        return BaselineResult(
            instance=env.instance.name,
            method=self.name,
            start_times=env.start_times.copy(),
            makespan=float(rollout["makespan"]),
            wall_clock_time=wall_clock_time,
            feasible=bool(rollout["feasible"]),
            violations=list(rollout["violations"]),
            status="OK",
            is_optimal=False,
            extra={"dispatch_order": rollout["dispatch_order"]},
        )


class CPSATBaseline:
    """Adapter around :func:`~baselines.cpsat.solve_cpsat` (the JSSP path only).

    Args:
        num_workers: CP-SAT search worker threads.
        log_progress: Stream the solver log to stdout.
    """

    def __init__(self, num_workers: int = 8, log_progress: bool = False) -> None:
        self.name = "cpsat"
        self.num_workers = num_workers
        self.log_progress = log_progress

    def solve(
        self,
        instance: JSSPInstance,
        *,
        seed: int | None = None,
        time_limit_s: float | None = None,
    ) -> BaselineResult:
        """Solve to (proven) minimum makespan. ``seed`` is unused: CP-SAT's search is
        internally randomized but not exposed as a public seed by this wrapper.

        Raises:
            ValueError: If ``time_limit_s`` is ``None``.
        """
        if time_limit_s is None:
            raise ValueError("CPSATBaseline.solve requires an explicit time_limit_s")

        result = solve_cpsat(
            instance,
            time_limit_s=time_limit_s,
            num_workers=self.num_workers,
            log_progress=self.log_progress,
        )

        feasible = result.start_times is not None
        violations: list[str] = [] if feasible else ["CP-SAT found no feasible solution"]
        start_times = result.start_times
        if feasible:
            feasible, violations = is_schedule_feasible(instance, start_times)
        else:
            start_times = np.full(instance.num_operations, -1.0)

        return BaselineResult(
            instance=instance.name,
            method=self.name,
            start_times=start_times,
            makespan=float(result.makespan) if result.makespan is not None else float("inf"),
            wall_clock_time=result.wall_time,
            feasible=feasible,
            violations=violations,
            status=result.status,
            is_optimal=result.is_optimal,
            extra={"best_bound": result.best_bound, "mip_gap": result.mip_gap},
        )


class CPSATFJSPBaseline:
    """Adapter around :func:`~baselines.cpsat.solve_cpsat_fjsp` (the FJSP path).

    The eligible-machine-set counterpart to :class:`CPSATBaseline`: same
    ``OptionalIntervalVar``-per-alternative model, same call signature, distinct
    ``name`` (``"cpsat_fjsp"``) so a ``run_evaluation.py`` results CSV can tell the two
    apart even when both appear in the same ``method`` column.

    Args:
        num_workers: CP-SAT search worker threads.
        log_progress: Stream the solver log to stdout.
    """

    def __init__(self, num_workers: int = 8, log_progress: bool = False) -> None:
        self.name = "cpsat_fjsp"
        self.num_workers = num_workers
        self.log_progress = log_progress

    def solve(
        self,
        instance: FJSPInstance,
        *,
        seed: int | None = None,
        time_limit_s: float | None = None,
    ) -> BaselineResult:
        """Solve to (proven) minimum makespan over the eligible-machine routing space.

        No external re-check of ``feasible`` is done here (unlike
        :meth:`CPSATBaseline.solve`'s ``is_schedule_feasible`` call): ``solve_cpsat_fjsp``
        already asserts feasibility internally against the *chosen* machine per
        operation before returning, a strictly stronger guarantee than an external
        recheck could add without also plumbing that assignment back out.

        Raises:
            ValueError: If ``time_limit_s`` is ``None``.
        """
        if time_limit_s is None:
            raise ValueError("CPSATFJSPBaseline.solve requires an explicit time_limit_s")

        result = solve_cpsat_fjsp(
            instance,
            time_limit_s=time_limit_s,
            num_workers=self.num_workers,
            log_progress=self.log_progress,
        )

        feasible = result.start_times is not None
        violations: list[str] = [] if feasible else ["CP-SAT found no feasible solution"]
        start_times = result.start_times if feasible else np.full(instance.num_operations, -1.0)

        return BaselineResult(
            instance=instance.name,
            method=self.name,
            start_times=start_times,
            makespan=float(result.makespan) if result.makespan is not None else float("inf"),
            wall_clock_time=result.wall_time,
            feasible=feasible,
            violations=violations,
            status=result.status,
            is_optimal=result.is_optimal,
            extra={"best_bound": result.best_bound, "mip_gap": result.mip_gap},
        )


class GABaseline:
    """Adapter around :class:`~baselines.ga.GeneticAlgorithm`.

    Accepts either a :class:`~envs.instance.JSSPInstance` (fixed routing) or an
    :class:`~envs.fjsp_instance.FJSPInstance` (eligible-machine sets) --- see
    :func:`~baselines.ga.is_fjsp`, which :class:`~baselines.ga.GeneticAlgorithm`
    already dispatches on internally; this adapter only needs to pick the matching
    feasibility checker for :class:`BaselineResult`.

    Args:
        config: A :class:`~baselines.ga.GAConfig`. Defaults to ``GAConfig()`` (the
            ``GA_TechSpec.md`` Table 2.1 defaults, with population size auto-scaled by
            instance size).
        **overrides: Individual ``GAConfig`` fields to override, so ``GABaseline(**{"population_size": 40})``
            (or, more naturally, ``GABaseline(population_size=40)``) works without
            constructing a ``GAConfig`` by hand first --- the adapter-construction style
            ``Baselines_TechSpec.md`` §2.2 specifies (``GABaseline(**ga_config)``).
    """

    def __init__(self, config: GAConfig | None = None, **overrides) -> None:
        self.name = "ga"
        base_config = config if config is not None else GAConfig()
        self.config = replace(base_config, **overrides) if overrides else base_config
        self._ga = GeneticAlgorithm(self.config)

    def solve(
        self,
        instance: Instance,
        *,
        seed: int | None = None,
        time_limit_s: float | None = None,
    ) -> BaselineResult:
        """Run the GA to the wall-clock budget and return the best schedule found.

        Raises:
            ValueError: If ``time_limit_s`` is ``None``.
        """
        if time_limit_s is None:
            raise ValueError("GABaseline.solve requires an explicit time_limit_s")

        t0 = time.perf_counter()
        result = self._ga.run(instance, seed=seed, time_limit_s=time_limit_s)
        wall_clock_time = time.perf_counter() - t0

        if is_fjsp(instance):
            machine_assignment = {op: int(m) for op, m in enumerate(result["machine_of_op"])}
            feasible, violations = is_fjsp_schedule_feasible(
                instance, result["start_times"], machine_assignment
            )
        else:
            feasible, violations = is_schedule_feasible(instance, result["start_times"])

        return BaselineResult(
            instance=instance.name,
            method=self.name,
            start_times=result["start_times"],
            makespan=float(result["makespan"]),
            wall_clock_time=wall_clock_time,
            feasible=feasible,
            violations=violations,
            status="OK",
            is_optimal=False,
            extra={
                "generations_run": result["generations_run"],
                "best_makespan_history": result["best_makespan_history"],
            },
        )


class RLPlaceholderBaseline:
    """Schema-stable stand-in for the GIN+PPO policy this sprint's refactor exists to
    receive next month.

    ``run_evaluation.py`` accepts ``"rl"`` as a method today so the results CSV's
    columns, dtypes, and ``(instance, method, disruption, seed)`` row shape are already
    exactly what the trained policy will produce --- swapping this adapter for a real one
    later changes zero downstream code (the CLI, the CSV schema, the
    Wilcoxon/Friedman/Nemenyi analysis that reads it). ``solve`` never raises: every
    field of :class:`BaselineResult` is populated with an explicit "not implemented"
    marker (``status="NOT_IMPLEMENTED"``, ``makespan=nan``, ``feasible=False``) rather
    than a crash, so a mixed-method evaluation run completes and the placeholder rows
    are simply filterable/ignorable in the resulting CSV.
    """

    def __init__(self) -> None:
        self.name = "rl"

    def solve(
        self,
        instance: Instance,
        *,
        seed: int | None = None,
        time_limit_s: float | None = None,
    ) -> BaselineResult:
        """Return a placeholder :class:`BaselineResult`. Never raises."""
        return BaselineResult(
            instance=instance.name,
            method=self.name,
            start_times=np.full(instance.num_operations, -1.0),
            makespan=float("nan"),
            wall_clock_time=0.0,
            feasible=False,
            violations=["RL policy not yet trained -- placeholder row for CSV schema stability"],
            status="NOT_IMPLEMENTED",
            is_optimal=False,
            extra={},
        )


class _DynamicNodeFeatureAdapter(gym.ObservationWrapper):
    """Adapts :class:`~envs.dynamic_jssp_env.DynamicJSSPEnv`'s observation contract down
    to the frozen static-trained checkpoint's expected input.

    ``DynamicJSSPEnv`` (as of the E4-E6 wiring fix) already emits ``machine_features``
    and ``adj_eligibility`` matching :class:`~envs.jssp_env.JSSPEnv`'s contract exactly
    --- see that module's ``NUM_MACHINE_FEATURES``/``MACHINE_FEATURE_NAMES``. The one
    genuine difference left is ``node_features``: the dynamic env appends a real,
    dynamic-only ``is_preempted`` column (index 8) that the static contract's
    ``NUM_NODE_FEATURES`` (8) has no room for --- the checkpoint's first linear layer is
    hard-wired to width 8 (verified directly against the saved ``state_dict``), so
    feeding it 9 columns is a shape error, not a soft mismatch.

    This wrapper's only job is dropping that one column before the policy sees it. It
    deliberately does **not** touch ``DynamicJSSPEnv`` itself (that environment's own
    observation contract stays exactly as tested by ``tests/test_dynamic_env.py``,
    unaffected by whichever old or new checkpoint happens to be evaluated against it).

    Honest limitation this implies (documented here, not hidden): the loaded policy
    cannot see which operations are currently preempted. It cannot select an illegal
    action because of this (masking is independent, enforced by the environment itself),
    but its choices among legal actions may be weaker specifically around resumed
    operations under breakdowns, since it has no way to distinguish a resumed op from a
    freshly-dispatchable one. A policy retrained against the full 9-feature contract
    would not have this limitation.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        base_space = env.observation_space
        node_shape = base_space["node_features"].shape
        adapted_spaces = dict(base_space.spaces)
        adapted_spaces["node_features"] = spaces.Box(
            low=0.0, high=np.inf, shape=(node_shape[0], NUM_NODE_FEATURES), dtype=np.float32
        )
        self.observation_space = spaces.Dict(adapted_spaces)

    def observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Drop the dynamic-only ``is_preempted`` column (last column) from
        ``node_features``; every other key passes through unchanged."""
        adapted = dict(observation)
        adapted["node_features"] = observation["node_features"][:, :NUM_NODE_FEATURES]
        return adapted


@dataclass
class TrainedRLBaseline:
    """Wraps a trained heterogeneous GNN+PPO policy as a :class:`Baseline` adapter.

    The real thing :class:`RLPlaceholderBaseline` was holding the ``"rl"`` slot for, per
    ``GNN_PPO_Architecture_Spec_v2.md`` §8.  Because the Week 18 policy is *size-agnostic*
    (§3.4 --- the encoder and both heads have parameter counts independent of
    ``num_operations`` and ``num_machines``), **one** adapter instance serves every
    instance ``run_evaluation.py`` iterates over: the saved ``state_dict`` is loaded once
    and re-wrapped per distinct instance shape, rather than needing a policy per size.

    Like a PDR --- and unlike GA/CP-SAT --- a trained policy is fundamentally an
    ``(env) -> action`` mapping, so ``solve`` constructs the matching environment and
    rolls out to termination, reading ``start_times`` off the environment exactly as
    :class:`PDRBaseline` does.  ``time_limit_s`` is accepted and ignored for the same
    reason it is for a PDR: a single deterministic rollout is not meaningfully
    time-limited.

    Args:
        policy_path: Path to a ``MaskablePPO.save``-produced ``.zip``, as written by
            ``experiments/train_hetero.py``.
        deterministic: Take the argmax legal action (``True``, the default) or sample.
        n_rollouts: Number of rollouts per ``solve`` call; the best (minimum makespan) is
            returned.  ``1`` with ``deterministic=True`` is the honest "no search"
            setting and the default.  Values ``> 1`` only make sense with
            ``deterministic=False``, since repeated greedy rollouts are identical.
        device: Torch device for inference.
        name: Method identifier; ``"rl"`` so the results CSV schema is unchanged.

    Raises:
        FileNotFoundError: If ``policy_path`` does not exist (raised on first ``solve``,
            not at construction, so a resolved-but-unused adapter never fails a sweep).
    """

    policy_path: str
    deterministic: bool = True
    n_rollouts: int = 1
    device: str = "cpu"
    name: str = "rl"

    def __post_init__(self) -> None:
        self._state_dict: dict | None = None
        self._policy_kwargs: dict | None = None
        self._head_kind: str | None = None
        self._policies: dict[tuple[int, int, int], object] = {}

    # ------------------------------------------------------------------ internals

    def _load_source_policy(self) -> None:
        """Load the checkpoint once, caching its weights, kwargs and head kind."""
        if self._state_dict is not None:
            return
        from sb3_contrib import MaskablePPO  # noqa: PLC0415 (heavy import, call-site only)

        path = Path(self.policy_path)
        if not path.exists():
            raise FileNotFoundError(f"trained policy checkpoint not found: {path}")
        model = MaskablePPO.load(str(path), device=self.device)
        self._state_dict = {k: v.clone() for k, v in model.policy.state_dict().items()}
        self._policy_kwargs = {
            key: value
            for key, value in model.policy_kwargs.items()
            if key not in ("net_arch",)
        }
        # Infer which actor head the checkpoint carries from the first layer's fan-in:
        # PerNodeActorHead consumes one node embedding, PerPairActorHead consumes two
        # concatenated.  Cheaper and more robust than re-deriving it from the saved
        # action space, and it is what gives the clear error message below.
        first_weight = self._state_dict["action_net.mlp.0.weight"]
        hidden_dim = int(self._state_dict["value_net.critic.mlp.0.weight"].shape[1]) // 2
        self._head_kind = "per_node" if first_weight.shape[1] == hidden_dim else "per_pair"

    def _build_env(self, instance: Instance) -> "SchedulingEnv":
        """Construct the environment matching ``instance``'s problem variant."""
        if is_fjsp(instance):
            from envs.fjsp_env import FJSPEnv  # noqa: PLC0415 (cycle-free)

            return FJSPEnv(instance)
        return JSSPEnv(instance)

    def _policy_for(self, env: "SchedulingEnv"):
        """Return a policy wrapping ``env``'s spaces, loading the cached weights into it.

        Raises:
            ValueError: If the checkpoint's actor head does not match the problem variant
                ``env`` represents --- a JSSP-trained per-node head cannot score
                ``(operation, machine)`` pairs, and saying so plainly beats letting
                ``load_state_dict`` fail with a shape message mid-sweep (the same
                fail-loudly-and-early discipline ``validate_methods_for_disruption``
                already applies).
        """
        from models.policy import PerNodeMaskableActorCriticPolicy  # noqa: PLC0415

        self._load_source_policy()
        num_ops = int(env.observation_space["node_features"].shape[0])
        num_machines = int(env.observation_space["machine_features"].shape[0])
        n_actions = int(env.action_space.n)

        needed = "per_node" if n_actions == num_ops else "per_pair"
        if needed != self._head_kind:
            raise ValueError(
                f"{self.policy_path} carries a {self._head_kind!r} actor head but "
                f"{env.instance.name} needs a {needed!r} one: a policy trained on "
                f"{'JSSP (fixed routing)' if self._head_kind == 'per_node' else 'FJSP (flexible routing)'} "
                f"cannot be evaluated on the other variant. Train a separate policy, or "
                f"pass the matching checkpoint."
            )

        key = (num_ops, num_machines, n_actions)
        if key not in self._policies:
            policy = PerNodeMaskableActorCriticPolicy(
                env.observation_space,
                env.action_space,
                lambda _: 0.0,  # inference only; no optimizer step is ever taken
                **(self._policy_kwargs or {}),
            )
            policy.load_state_dict(self._state_dict, strict=True)
            policy.set_training_mode(False)
            self._policies[key] = policy
        return self._policies[key]

    def _rollout(self, env: "SchedulingEnv", policy, seed: int | None) -> dict:
        """Run one episode to termination and return its outcome."""
        obs, _ = env.reset(seed=seed)
        terminated, info = False, {}
        while not terminated:
            action, _ = policy.predict(
                obs, action_masks=env.action_masks(), deterministic=self.deterministic
            )
            obs, _, terminated, _, info = env.step(int(action))
        return {
            "makespan": float(info["makespan"]),
            "start_times": env.start_times.copy(),
            "dispatch_order": env.dispatch_order,
            "machine_of_op": (
                env.machine_of_op.copy() if hasattr(env, "machine_of_op") else None
            ),
        }

    def _build_dynamic_env(
        self, instance: Instance, disruption_config: "DisruptionConfig"
    ) -> _DynamicNodeFeatureAdapter:
        """Construct a disrupted :class:`~envs.dynamic_jssp_env.DynamicJSSPEnv`, wrapped
        so its observation matches this checkpoint's contract (see
        :class:`_DynamicNodeFeatureAdapter`).

        Raises:
            TypeError: If ``instance`` is an FJSP instance --- ``DynamicJSSPEnv`` models
                fixed routing only (see ``run_evaluation.py``'s module docstring).
        """
        if is_fjsp(instance):
            raise TypeError(
                "TrainedRLBaseline.solve_dynamic does not support FJSPInstance: "
                "DynamicJSSPEnv models fixed routing only."
            )
        from envs.dynamic_jssp_env import DynamicJSSPEnv  # noqa: PLC0415 (cycle-free)

        return _DynamicNodeFeatureAdapter(DynamicJSSPEnv(instance, disruption_config))

    def _rollout_dynamic(
        self, env: _DynamicNodeFeatureAdapter, policy, seed: int | None
    ) -> dict:
        """Run one disrupted episode to termination, reading feasibility from the
        environment's own dynamic-aware checker (``info["schedule_feasible"]``) rather
        than re-deriving it against the static instance, mirroring
        ``run_evaluation.py::run_pdr_online``'s convention for the same reason: a
        schedule that covers dynamically-arrived jobs has no fixed-size static
        counterpart to check against.

        ``gymnasium.Wrapper`` (this version) only proxies ``action_space``/
        ``observation_space``, not arbitrary attributes or methods -- unlike
        ``TrainedRLBaseline._rollout``, which relies on ``env.action_masks()``/
        ``env.start_times``/``env.dispatch_order`` resolving directly on an unwrapped
        env, this method reaches through ``env.unwrapped`` explicitly for those.
        """
        raw_env = env.unwrapped
        obs, _ = env.reset(seed=seed)
        terminated, info = False, {}
        while not terminated:
            action, _ = policy.predict(
                obs, action_masks=raw_env.action_masks(), deterministic=self.deterministic
            )
            obs, _, terminated, _, info = env.step(int(action))
        return {
            "makespan": float(info["makespan"]),
            "start_times": raw_env.start_times.copy(),
            "dispatch_order": raw_env.dispatch_order,
            "feasible": bool(info["schedule_feasible"]),
            "violations": list(info["violations"]),
        }

    # ------------------------------------------------------------------ public API

    def solve_dynamic(
        self,
        instance: Instance,
        disruption_config: "DisruptionConfig",
        *,
        seed: int | None = None,
        time_limit_s: float | None = None,
    ) -> BaselineResult:
        """Roll the trained policy out live against a disrupted
        :class:`~envs.dynamic_jssp_env.DynamicJSSPEnv`, like a PDR --- see
        :class:`_DynamicNodeFeatureAdapter` for the one input-contract adaptation this
        requires and the limitation it implies. ``time_limit_s`` is accepted for
        call-compatibility and ignored, same as :meth:`solve`.

        Args:
            instance: A :class:`~envs.instance.JSSPInstance` (FJSP is not supported ---
                ``DynamicJSSPEnv`` models fixed routing only).
            disruption_config: The disruption regime to evaluate under.
            seed: Seeds the environment, the disruption generators, and (with
                ``deterministic=False``) the rollout sampling.
            time_limit_s: Accepted for ``Baseline`` call-compatibility and ignored.

        Returns:
            A :class:`BaselineResult` whose ``feasible``/``violations`` come from the
            environment's own dynamic-aware feasibility check.

        Raises:
            FileNotFoundError: If the checkpoint is missing.
            ValueError: If the checkpoint's head does not match the problem variant.
            TypeError: If ``instance`` is an FJSP instance.
        """
        env = self._build_dynamic_env(instance, disruption_config)
        policy = self._policy_for(env)

        t0 = time.perf_counter()
        rollouts = [
            self._rollout_dynamic(env, policy, None if seed is None else seed + i)
            for i in range(max(self.n_rollouts, 1))
        ]
        wall_clock_time = time.perf_counter() - t0
        best = min(rollouts, key=lambda r: r["makespan"])

        return BaselineResult(
            instance=instance.name,
            method=self.name,
            start_times=best["start_times"],
            makespan=best["makespan"],
            wall_clock_time=wall_clock_time,
            feasible=best["feasible"],
            violations=best["violations"],
            status="OK",
            is_optimal=False,
            extra={"dispatch_order": best["dispatch_order"], "n_rollouts": len(rollouts)},
        )

    def solve(
        self,
        instance: Instance,
        *,
        seed: int | None = None,
        time_limit_s: float | None = None,
    ) -> BaselineResult:
        """Roll the trained policy out on ``instance``. ``time_limit_s`` is ignored.

        Args:
            instance: A :class:`~envs.instance.JSSPInstance` or an
                :class:`~envs.fjsp_instance.FJSPInstance`; the matching environment is
                constructed automatically.
            seed: Seeds the environment and, with ``deterministic=False``, the rollout
                sampling.
            time_limit_s: Accepted for ``Baseline`` call-compatibility and ignored.

        Returns:
            A :class:`BaselineResult` whose ``feasible`` field comes from the same
            independent checker every other adapter uses for that instance type.

        Raises:
            FileNotFoundError: If the checkpoint is missing.
            ValueError: If the checkpoint's head does not match the problem variant.
        """
        env = self._build_env(instance)
        policy = self._policy_for(env)

        t0 = time.perf_counter()
        rollouts = [
            self._rollout(env, policy, None if seed is None else seed + i)
            for i in range(max(self.n_rollouts, 1))
        ]
        wall_clock_time = time.perf_counter() - t0
        best = min(rollouts, key=lambda r: r["makespan"])

        extra: dict = {"dispatch_order": best["dispatch_order"], "n_rollouts": len(rollouts)}
        if is_fjsp(instance):
            machine_assignment = {
                op: int(machine) for op, machine in enumerate(best["machine_of_op"])
            }
            feasible, violations = is_fjsp_schedule_feasible(
                instance, best["start_times"], machine_assignment
            )
            extra["machine_of_op"] = best["machine_of_op"]
        else:
            feasible, violations = is_schedule_feasible(instance, best["start_times"])

        return BaselineResult(
            instance=instance.name,
            method=self.name,
            start_times=best["start_times"],
            makespan=best["makespan"],
            wall_clock_time=wall_clock_time,
            feasible=feasible,
            violations=violations,
            status="OK",
            is_optimal=False,
            extra=extra,
        )
