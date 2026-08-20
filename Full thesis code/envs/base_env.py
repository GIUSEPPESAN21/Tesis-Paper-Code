"""Shared base for every scheduling Gymnasium environment in this codebase.

:class:`~envs.jssp_env.JSSPEnv` and :class:`~envs.dynamic_jssp_env.DynamicJSSPEnv`
model the same MDP family --- dispatch operations of a disjunctive graph under action
masking --- but their internal state representations are genuinely different (fixed-size
NumPy arrays with an analytic list-scheduling transition vs. growable arrays driven by a
real future-event list). Forcing ``step()``/``_observation()``/episode-state allocation
into one shared implementation would distort one of the two designs for no benefit, so
this base class factors out only what is actually identical between them: instance
resolution, the ``reward_scale`` convention, and the ``static_lb``/``dispatch_order``
read surface every dispatching rule and :class:`~baselines.interface.Baseline` adapter
relies on to treat either environment interchangeably.

Before this refactor, ``DynamicJSSPEnv`` subclassed ``JSSPEnv`` but deliberately skipped
its ``__init__`` (``gym.Env.__init__(self)``, not ``super().__init__()``) and overrode
every other method --- an inheritance relationship that bought nothing but implicit
reuse of two properties, at the cost of a misleading class hierarchy. Both environments
now inherit from this class instead, so what they share is stated once, explicitly, and
what they don't is no longer implied by a false ``is-a`` relationship.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.disjunctive_graph import static_lower_bound
from envs.instance import JSSPInstance
from envs.loader import load_by_name


class SchedulingEnv(gym.Env[dict[str, np.ndarray], np.int64]):
    """Common contract and shared setup for a disjunctive-graph scheduling environment.

    Subclasses own their episode state and observation/transition logic; this class
    only provides:

    * :meth:`_init_common` --- the instance-resolution / reward-scale / horizon
      bookkeeping every concrete environment's ``__init__`` needs, in one place.
    * :attr:`static_lb` / :attr:`dispatch_order` --- read-only properties every
      baseline and test uses to treat any concrete environment interchangeably.
    * Documented method stubs (:meth:`action_masks`, :meth:`makespan`,
      :meth:`remaining_work_per_op`, :meth:`schedule_dataframe_records`) describing the
      contract :mod:`baselines` code is written against, so the shared interface is
      readable in one file instead of only implied by duck typing across two.
    """

    metadata = {"render_modes": ["human"]}

    def _init_common(
        self,
        instance: JSSPInstance | str,
        reward_scale: float | None,
        include_machine_edges: bool,
    ) -> None:
        """Resolve the instance and set the handful of attributes both environments
        compute identically: :attr:`instance`, :attr:`include_machine_edges`,
        :attr:`_static_lb`, :attr:`reward_scale`, and :attr:`_horizon`.

        Args:
            instance: A :class:`~envs.instance.JSSPInstance`, or an instance name such
                as ``"ft06"`` to be loaded from ``data/instances/``.
            reward_scale: ``None`` uses ``1 / static_lower_bound(instance)`` so returns
                are size-normalised; pass ``1.0`` for raw time-unit rewards.
            include_machine_edges: Whether ``adj_machine`` carries real edges or is
                returned all-zero (the "no disjunctive edges" ablation).

        Raises:
            FileNotFoundError: If ``instance`` names an instance that is not bundled.
        """
        self.instance: JSSPInstance = (
            load_by_name(instance) if isinstance(instance, str) else instance
        )
        self.include_machine_edges = include_machine_edges
        self._static_lb = static_lower_bound(self.instance)
        self.reward_scale = (
            float(reward_scale) if reward_scale is not None else 1.0 / self._static_lb
        )
        self._horizon = float(max(self.instance.total_processing_time, 1))

    @property
    def static_lb(self) -> int:
        """Instance-level lower bound; the best makespan this env could conceivably reach."""
        return self._static_lb

    @property
    def dispatch_order(self) -> list[int]:
        """Operation ids in the order they were dispatched this episode."""
        return list(self._dispatch_order)

    # ------------------------------------------------------- shared method contract
    #
    # Every concrete environment overrides all four; declared here (rather than left
    # purely implicit) so the interface baselines/dispatching rules are written against
    # -- action_masks() -> bool mask, makespan() -> float, remaining_work_per_op() ->
    # per-op remaining work, schedule_dataframe_records() -> Gantt/CSV rows -- has one
    # documented home instead of only existing as convention across two files.

    def action_masks(self) -> np.ndarray:
        """Boolean legality mask over the action space. See the concrete subclass."""
        raise NotImplementedError

    def makespan(self) -> float:
        """Makespan of the schedule built so far. See the concrete subclass."""
        raise NotImplementedError

    def remaining_work_per_op(self) -> np.ndarray:
        """Remaining work per active operation. See the concrete subclass."""
        raise NotImplementedError

    def schedule_dataframe_records(self) -> list[dict[str, float]]:
        """Per-operation schedule records for Gantt plots / CSV export. See the
        concrete subclass."""
        raise NotImplementedError
