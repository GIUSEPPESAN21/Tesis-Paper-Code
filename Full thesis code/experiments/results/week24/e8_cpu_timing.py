"""E8 -- decision-time timing, CPU-only secondary pass (FinalExperimentMatrix.md Sec 2.8).

GPU/training-hardware timing (the "primary" pass) is computed post-hoc from E1's own
wall_clock_s column (see build_e8_from_e1.py). This script produces the descriptive
"CPU-only" comparison point on a 4-instance subset (ft06, la05, la20, la35), against
CP-SAT's own wall-clock, as a ratio -- no significance test, per the matrix's own
instruction.

Honest scope note (see EXPERIMENT_LOG_WEEK24.md): TrainedRLBaseline's policy is
constructed raw (baselines/interface.py::_policy_for) and never moved off its default
construction device, so in this codebase's current implementation the "GPU" and
"CPU-only" passes both actually execute inference on CPU -- there is no code path today
that runs the trained policy's forward pass on the GPU. To still produce a meaningful
descriptive ratio (rather than two identical numbers under a misleading GPU/CPU label),
this script instead contrasts unconstrained multi-threaded CPU inference (this machine's
default: 16 logical cores available to torch) against single-threaded CPU inference
(torch.set_num_threads(1)) -- a fast/slow CPU proxy for the GPU-machine-vs-weaker-laptop
comparison the matrix intends, labelled accordingly rather than mislabelled as GPU/CPU.

T5.3 (Recomendacion 8 del tutor): full 43-instance JSSP coverage (``ft06``-``la40``),
not the 4-instance ``SUBSET`` this script originally shipped with -- H5's decision rule
and the amortized-cost figure (``Fig_costo_amortizado``, ``scripts/generate_thesis_figures.py``)
both need every instance, not a spot-check subset. The single-thread/multi-thread split
was already implemented (``rl_ms_per_decision_singlethread``/``_multithread``); this
sprint only widens the instance set and documents, below, which of the two columns feeds
which manuscript claim.

Which column feeds which claim (document here, not scattered across chapters):
- ``rl_ms_per_decision_multithread`` -- primary column for H5's accept/reject
  computation (``PreRegistration_FINAL.md`` Sec 6) and Chapter 6.5's headline
  ms/decision numbers: this machine's default thread count is what every other
  timing number in E1/E8 (``run_evaluation.py``'s ``wall_clock_s``) was measured
  under, so the comparison stays apples-to-apples.
- ``rl_ms_per_decision_singlethread`` -- descriptive-only, cited in Chapter 6.5's
  "does the picture change under different hardware" discussion
  (``PreRegistration_FINAL.md`` Amendment 2) and in ``Tabla_costo_computacional``'s
  footnote, never in the H5 accept/reject computation itself.
- ``singlethread_vs_multithread_ratio`` -- descriptive only, the size of the
  thread-count effect itself (not a claim about RL vs. CP-SAT).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch

from baselines.cpsat import solve_cpsat
from baselines.interface import TrainedRLBaseline
from envs.loader import load_by_name

REPO_ROOT = Path(__file__).resolve().parents[3]
CKPT = REPO_ROOT / "experiments/results/curriculum_full/checkpoints/curriculum_stage19_la20_final.zip"
OUT_CSV = Path(__file__).resolve().parent / "e8_cpu_timing.csv"

#: T5.3: full JSSP set (was a 4-instance SUBSET: ft06, la05, la20, la35). Kept as a
#: separate constant (not re-derived from run_evaluation.DEFAULT_INSTANCES, which only
#: covers la01-la10) so this script's scope is explicit and independently auditable.
ALL_JSSP_INSTANCES = ["ft06", "ft10", "ft20"] + [f"la{n:02d}" for n in range(1, 41)]

#: Backward-compatible alias -- some already-written notebooks/scripts may still
#: import SUBSET by that name; kept pointing at the full set rather than removed
#: outright, per Regla 5 (sin regresiones silenciosas por renombrado).
SUBSET = ALL_JSSP_INSTANCES


def time_rl(instance_name: str, n_threads: int | None, n_reps: int = 5) -> dict:
    if n_threads is not None:
        torch.set_num_threads(n_threads)
    instance = load_by_name(instance_name)
    baseline = TrainedRLBaseline(policy_path=str(CKPT), deterministic=True)
    times = []
    for rep in range(n_reps):
        t0 = time.perf_counter()
        result = baseline.solve(instance, seed=rep)
        times.append(time.perf_counter() - t0)
    n_ops = instance.num_operations
    mean_s = sum(times) / len(times)
    return {
        "instance": instance_name,
        "num_operations": n_ops,
        "mean_wall_clock_s": round(mean_s, 5),
        "ms_per_decision": round(1000.0 * mean_s / n_ops, 4),
        "makespan": result.makespan,
    }


def time_cpsat(instance_name: str, time_limit_s: float = 30.0) -> dict:
    instance = load_by_name(instance_name)
    t0 = time.perf_counter()
    result = solve_cpsat(instance, time_limit_s=time_limit_s)
    wall = time.perf_counter() - t0
    return {"instance": instance_name, "cpsat_wall_clock_s": round(wall, 4), "cpsat_status": result.status}


def main() -> None:
    rows = []
    for name in SUBSET:
        multi = time_rl(name, n_threads=None)
        single = time_rl(name, n_threads=1)
        cpsat = time_cpsat(name)
        row = {
            "instance": name,
            "num_operations": multi["num_operations"],
            "rl_ms_per_decision_multithread": multi["ms_per_decision"],
            "rl_ms_per_decision_singlethread": single["ms_per_decision"],
            "rl_makespan": multi["makespan"],
            "cpsat_wall_clock_s": cpsat["cpsat_wall_clock_s"],
            "cpsat_status": cpsat["cpsat_status"],
            "singlethread_vs_multithread_ratio": round(
                single["ms_per_decision"] / multi["ms_per_decision"], 3
            ),
        }
        rows.append(row)
        print(row, flush=True)

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
