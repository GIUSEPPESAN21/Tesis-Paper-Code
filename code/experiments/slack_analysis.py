"""Fase 1 (Recomendacion 2 del tutor): medir la holgura explicitamente, T1.1.

El mecanismo central de la tesis -- los programas mas compactos son mas fragiles
porque tienen menos holgura -- hoy es una interpretacion narrativa, no una medicion.
Este modulo calcula holgura (slack) via el metodo del camino critico (CPM) sobre el
grafo disyuntivo YA RESUELTO por un schedule concreto: se reutiliza
``envs.disjunctive_graph.build_precedence_adjacency`` para las aristas de precedencia
(Regla del tutor T1.1: "no reimplementes la topologia") y se anaden las aristas de
orden de maquina inducidas por el orden real de despacho de ``start_times`` -- una vez
fijado ese orden, el grafo disyuntivo resuelto es un DAG y el CPM estandar aplica.

No se ejecuta nada de este modulo en esta sesion (Regla Cero); queda listo para que
``experiments/run_slack_study.py`` (T1.2) lo invoque sobre los programas nominales de
CP-SAT/GA/9 PDR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.disjunctive_graph import build_precedence_adjacency, is_schedule_feasible
from envs.instance import JSSPInstance


@dataclass(frozen=True)
class SlackMetrics:
    """The six slack/criticality metrics T1.1 asks for, one struct per (instance,
    method) schedule."""

    total_slack: float
    mean_slack: float
    slack_ratio: float
    critical_path_length: float
    critical_op_fraction: float
    machine_idle_fraction: float
    makespan: float
    n_operations: int


def _machine_order_edges(instance: JSSPInstance, start_times: np.ndarray) -> list[tuple[int, int]]:
    """Directed edges ``prev -> next`` for each machine, ordered by ``start_times`` --
    the disjunctive arcs, resolved (oriented) by this particular schedule's dispatch
    order, per operation on shared machines."""
    edges: list[tuple[int, int]] = []
    flat_machines = instance.flat_machines
    for machine in range(instance.num_machines):
        op_ids = np.flatnonzero(flat_machines == machine)
        if op_ids.size <= 1:
            continue
        ordered = op_ids[np.argsort(start_times[op_ids], kind="stable")]
        for prev, nxt in zip(ordered[:-1], ordered[1:]):
            edges.append((int(prev), int(nxt)))
    return edges


def _topological_order(n: int, successors: list[list[int]]) -> list[int]:
    """Kahn's algorithm. Raises ValueError if the combined graph has a cycle --
    which, for a genuinely feasible schedule, should never happen (precedence +
    a single fixed per-machine order is acyclic by construction); a cycle here means
    ``start_times`` did not actually come from a feasible schedule."""
    in_degree = [0] * n
    for u in range(n):
        for v in successors[u]:
            in_degree[v] += 1
    queue = [u for u in range(n) if in_degree[u] == 0]
    order: list[int] = []
    while queue:
        u = queue.pop()
        order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    if len(order) != n:
        raise ValueError(
            "combined precedence+machine-order graph has a cycle -- start_times does "
            "not correspond to a feasible schedule (see envs.disjunctive_graph.is_schedule_feasible)"
        )
    return order


def schedule_slack_metrics(
    start_times: np.ndarray, instance: JSSPInstance, *, check_feasible: bool = True
) -> SlackMetrics:
    """CPM-based slack metrics for one concrete schedule of ``instance``.

    Args:
        start_times: Start time per global operation id, shape ``(num_operations,)`` --
            same convention as ``envs.disjunctive_graph.is_schedule_feasible`` and
            every ``BaselineResult.start_times`` in this codebase.
        instance: The JSSP instance the schedule was built for.
        check_feasible: If ``True`` (default), reject an infeasible schedule up front
            with a clear message rather than let CPM silently produce nonsense on
            malformed input. Callers doing a bulk sweep that already validated
            feasibility upstream may pass ``False`` to skip the redundant check.

    Returns:
        A :class:`SlackMetrics`.

    Raises:
        ValueError: If ``check_feasible`` and the schedule is infeasible, or if the
            combined precedence+machine-order graph contains a cycle.
    """
    if check_feasible:
        feasible, violations = is_schedule_feasible(instance, start_times)
        if not feasible:
            raise ValueError(f"start_times is not a feasible schedule for {instance.name}: {violations[:3]}")

    n = instance.num_operations
    durations = instance.flat_durations
    precedence_adj = build_precedence_adjacency(instance)
    successors: list[list[int]] = [list(np.flatnonzero(precedence_adj[u])) for u in range(n)]
    for prev, nxt in _machine_order_edges(instance, start_times):
        successors[prev].append(nxt)

    order = _topological_order(n, successors)

    es = np.zeros(n, dtype=np.float64)
    for u in order:
        for v in successors[u]:
            candidate = es[u] + durations[u]
            if candidate > es[v]:
                es[v] = candidate
    ef = es + durations
    makespan = float(ef.max()) if n > 0 else 0.0

    lf = np.full(n, makespan, dtype=np.float64)
    for u in reversed(order):
        for v in successors[u]:
            candidate = lf[v] - durations[v]
            if candidate < lf[u]:
                lf[u] = candidate
    ls = lf - durations

    slack = ls - es
    # Numerical guard: CPM slack is exactly >=0 by construction on a correct DAG;
    # floating-point durations (stochastic-time studies) can produce a slack of
    # -1e-9-ish on a truly-zero-slack op. Clip rather than let a spurious tiny
    # negative slack corrupt critical_op_fraction's zero-slack count.
    slack = np.clip(slack, 0.0, None)

    total_slack = float(slack.sum())
    mean_slack = total_slack / n if n > 0 else 0.0
    slack_ratio = total_slack / (makespan * instance.num_machines) if makespan > 0 else 0.0
    critical_op_fraction = float(np.isclose(slack, 0.0, atol=1e-6).sum()) / n if n > 0 else 0.0

    total_processing_time = float(instance.machine_totals.sum())
    machine_idle_fraction = (
        1.0 - total_processing_time / (makespan * instance.num_machines) if makespan > 0 else 0.0
    )

    return SlackMetrics(
        total_slack=total_slack,
        mean_slack=mean_slack,
        slack_ratio=slack_ratio,
        critical_path_length=makespan,
        critical_op_fraction=critical_op_fraction,
        machine_idle_fraction=machine_idle_fraction,
        makespan=makespan,
        n_operations=n,
    )
