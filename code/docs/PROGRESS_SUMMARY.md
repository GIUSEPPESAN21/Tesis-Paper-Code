# Progress Summary — E7/E8 Sprint (COMPLETE, 2026-08-01)

Task: execute `../../thesis/E7_E8_Plan.md` + `../../thesis/StatsAnalysis_TechSpec.md` in full —
E7 disruption-aware training (kappa in {0, 0.5, 1.0}), E8 latency benchmark,
full statistical pipeline over E1-E8, `../../thesis/RESULTS_REPORT.md`.

**All 11 tasks completed. See `../../thesis/RESULTS_REPORT.md` for the actual results,
hypothesis-by-hypothesis verdict table, and full artifact manifest.** This file is now
a historical record of how the sprint was executed, not a resume point.

## What happened this session (resuming from the 2026-07-31 checkpoint below)

1. **E8 clean re-measurement** — confirmed complete (516 deterministic + 172
   RL-stochastic rows, `E8_CLEAN_DONE` sentinel present); had finished unattended
   overnight as intended.
2. **Stats test suite** — run for the first time: 84/84 passed, no bugs found in the
   core `io.py`/`metrics.py`/`pairwise.py`/`omnibus.py`/`cd_diagram.py` modules.
3. **`train_dynamic.py` smoke test** — already run in an earlier pass this same day
   (caught a hidden_dim/num_layers checkpoint-shape mismatch, fixed before real budget
   was spent — see the dated comment in `train_dynamic.py`'s CLI arg definitions).
4. **E7 training, for real** — 4 full training runs (kappa in {0.5, 1.0} x stage
   {D1=ft06, D2=la01}), 100k timesteps each, ~33-36 min/stage, CPU-only, 8-core
   machine. **Important environment note:** background processes launched via the
   Bash tool's own `run_in_background` do NOT reliably survive session/turn
   boundaries on this setup (two training runs were silently killed mid-run this way,
   with no error — just a truncated log). Switched to `nohup ... & disown` from a
   script file (matching how `run_e8_clean.sh` had already survived) for all four real
   training runs; all four then ran to completion cleanly. **If resuming any future
   long-running background work on this machine/harness, use nohup+disown from a
   script file, not the tool's bare `run_in_background` flag, for anything that needs
   to survive more than one turn.**
5. **E7 evaluation grid** — 9 regimes x 30 seeds x 2 instances x 2 kappa values, plus
   nominal (`--disruption none`) pass, run via `experiments/results/week25/run_e7_grid.sh`
   (written this session). 1084/1084 rows, 100% feasible.
6. **Stats pipeline run over E1-E8** — found and fixed two real bugs before trusting
   any number into the report (both detailed in `../../thesis/RESULTS_REPORT.md` §8):
   - `cli.py`'s E6-arrivals experiment-ID tag used `"u0.7"` instead of the zero-padded
     `"u0.70"` `REGIMES` expects, silently dropping that regime from H3/H4.
   - `report.py::run_h5` pooled deterministic and `--rl-sample` stochastic RL rows
     together, inflating RL's reported latency (didn't change the verdict, but was a
     real methodological inconsistency vs. the rest of the pipeline's `seed==0`
     convention).
   Also extended `report.py` with `generate_chapter6_artifacts()` — the original
   implementation only produced the H1-H5 verdict table + H2's CD diagrams; the tech
   spec's §9.3/9.4 additional RPD/RI/latency summary tables and E3/E4-E6/E8 figures
   did not exist yet. Added and wired in.
7. **`../../thesis/RESULTS_REPORT.md` written** — every verdict traced to a real statistic,
   two REJECT results (H4, H5) investigated and diagnosed with concrete root causes
   rather than taken at face value (H4: domain-randomized training never saw a
   nominal episode, nominal RPD collapsed; H5: CP-SAT's near-instant solve on small
   instances produces an artificially tiny per-decision figure, RL wins decisively
   only at the largest instances — both explained with real numbers, not hand-waved).
8. **Full regression suite**: 752 passed, 0 failed, run twice (before and after all
   `experiments/stats/` changes).

## Historical checkpoint below (2026-07-31, superseded — kept for provenance)

<details>
<summary>Original checkpoint text</summary>

### What was DONE and tested as of 2026-07-31

- **E7 precondition 1** — `envs/dynamic_jssp_env.py`: added `kappa` param and
  `-kappa*planned_start_deviation*reward_scale` term in `step()`. Bit-for-bit
  no-op at kappa=0 verified by `TestKappaReward` in `tests/test_dynamic_env.py`.
- **E7 precondition 2** — `models/policy.py::build_hetero_maskable_ppo()` now
  takes `device: str = "cpu"`, passed through to `MaskablePPO`.
- **E7 precondition 4** — `run_evaluation.py` hardened: per-cell try/except so
  one failing cell doesn't discard the sweep, incremental CSV flush, failure
  count surfaced instead of a silent empty run. Covered by
  `TestCsvHardening` in `tests/test_run_evaluation.py`.
- `AblationPlan.md` and `../../thesis/PreRegistration.md` amended with dated notes
  documenting the E7 training design decisions and the GPU/CPU framing
  collapse (Amendment 2), respectively.

### What was WRITTEN but not yet executed as of 2026-07-31

- `experiments/train_dynamic.py` (E7 precondition 3) — written, never run once.
- `experiments/stats/` pipeline — all 8 modules written, test suite never run.

### E8 clean re-measurement — was running in background, unattended, intentionally.

</details>

No results have been fabricated or assumed at any point in this sprint; every REJECT
and ACCEPT verdict in `../../thesis/RESULTS_REPORT.md` is the literal output of a real
statistical test against real data, and every anomalous-looking result was traced to a
concrete, stated root cause before being written down.
