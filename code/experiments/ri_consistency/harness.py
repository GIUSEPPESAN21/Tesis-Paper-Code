"""Consistent predictive-reactive harness for the RI-consistency study.

All eleven methods are executed in ONE event-driven simulator (envs.dynamic_jssp_env.DynamicJSSPEnv),
and every method's nominal makespan is obtained from the SAME simulator with all disruption axes off,
so the nominal and the perturbed makespan of each method always come from the same execution semantics.

* Dispatching rules: online, non-delay (only operations whose machine is idle and whose job
  predecessor has completed at the current decision epoch are eligible).
* CP-SAT and GA: the committed (offline) schedule is executed under right-shift repair: the planned
  per-machine operation sequence is never reordered; each planned operation starts as soon as its job
  predecessor has completed and its machine is free after the previous planned operation. Operations of
  dynamically arrived jobs (absent from any offline plan) are inserted only into machine idle time,
  i.e. when the machine is free and its next planned operation is not ready, and are chosen by LWKR.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from envs.dynamic_jssp_env import DynamicJSSPEnv            # noqa: E402
from envs.disruptions import DisruptionConfig               # noqa: E402
from envs.events import OpState, MachineState               # noqa: E402
from baselines.dispatching_rules import RULES               # noqa: E402

MAX_DYNAMIC_JOBS = 20
REGIMES = {
    'none': DisruptionConfig(),
    'B_A0.95': DisruptionConfig(target_availability=0.95),
    'B_A0.90': DisruptionConfig(target_availability=0.90),
    'B_A0.80': DisruptionConfig(target_availability=0.80),
    'S_s0.10': DisruptionConfig(processing_time_sigma=0.10),
    'S_s0.25': DisruptionConfig(processing_time_sigma=0.25),
    'S_s0.50': DisruptionConfig(processing_time_sigma=0.50),
    'A_u0.70': DisruptionConfig(target_utilization=0.70, max_dynamic_jobs=MAX_DYNAMIC_JOBS),
    'A_u0.85': DisruptionConfig(target_utilization=0.85, max_dynamic_jobs=MAX_DYNAMIC_JOBS),
    'A_u0.95': DisruptionConfig(target_utilization=0.95, max_dynamic_jobs=MAX_DYNAMIC_JOBS),
}


def machine_sequences(instance, start_times):
    """Per-machine planned sequence (operation ids sorted by planned start, ties by id)."""
    mach = instance.flat_machines
    seq = {m: [] for m in range(instance.num_machines)}
    for op in sorted(range(instance.num_operations), key=lambda o: (float(start_times[o]), o)):
        seq[int(mach[op])].append(op)
    return seq


class RightShiftEnv(DynamicJSSPEnv):
    """DynamicJSSPEnv whose legal set enforces a fixed per-machine planned sequence."""

    def __init__(self, instance, disruption_config, sequences):
        self._seq = {m: list(s) for m, s in sequences.items()}
        self._ptr = {m: 0 for m in self._seq}
        super().__init__(instance, disruption_config)

    def _allocate_episode_state(self, seed):
        if hasattr(self, '_ptr'):
            self._ptr = {m: 0 for m in self._seq}
        super()._allocate_episode_state(seed)

    def action_masks(self):
        base = super().action_masks()
        if getattr(self, '_raw_next', False):  # legality check of the action being dispatched
            self._raw_next = False
            return base
        out = np.zeros_like(base)
        legal = np.flatnonzero(base)
        if legal.size == 0:
            return out
        planned_ready = {}
        for m, s in self._seq.items():
            p = self._ptr[m]
            if p < len(s) and base[s[p]]:
                out[s[p]] = True
                planned_ready[m] = True
        for op in legal:
            if op >= self.base_num_ops:  # dynamically arrived job: fill idle machine time only
                m = int(self._flat_machines[op])
                if not planned_ready.get(m, False):
                    out[op] = True
        return out

    def step(self, action):
        action = int(action)
        assert self.action_masks()[action], 'action not legal under the planned sequence'
        if action < self.base_num_ops:
            m = int(self._flat_machines[action])
            assert self._seq[m][self._ptr[m]] == action
            self._ptr[m] += 1
        self._raw_next = True
        return super().step(action)


def run_pdr(rule, instance, cfg, seed):
    env = DynamicJSSPEnv(instance, cfg)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    f = RULES[rule]
    term, info = False, {}
    while not term:
        _, _, term, _, info = env.step(f(env, rng))
    assert info['schedule_feasible'], (rule, instance.name, info['violations'][:3])
    return float(info['makespan'])


def run_planned(sequences, instance, cfg, seed):
    env = RightShiftEnv(instance, cfg, sequences)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    lwkr = RULES['lwkr']
    term, info = False, {}
    while not term:
        legal = np.flatnonzero(env.action_masks())
        planned = [o for o in legal if o < env.base_num_ops]
        if planned:
            a = min(planned)
        else:
            a = lwkr(env, rng)
            assert env.action_masks()[a]
        _, _, term, _, info = env.step(a)
    assert info['schedule_feasible'], (instance.name, info['violations'][:3])
    return float(info['makespan'])
