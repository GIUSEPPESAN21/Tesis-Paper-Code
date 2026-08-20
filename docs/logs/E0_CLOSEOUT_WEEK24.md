# E0 — Validación DES: Close-Out Summary (Week 24)

**Status:** Closed, no new compute scheduled, per `../docs/planificacion/FinalExperimentMatrix.md` §2.0
("Chapter 6.1 needs a validation narrative, not a new experiment"). This document
consolidates `VALIDATION_WEEK1.md` and `VALIDATION_WEEK6.md` into the one-page summary
Chapter 6.1 needs, plus notes the one honest coverage gap.

---

## What E0 claims and where it's proven

E0 supports Chapter 6.1's claim that the discrete-event simulation core (static and
dynamic) is a faithful, constraint-correct implementation of the JSSP/FJSP model this
thesis studies — the foundation every later experiment (E1–E8) stands on.

| Claim | Evidence | Result |
|---|---|---|
| Instance loader reproduces literature optima | `tests/test_loader.py::TestKnownOptima`; independently re-derived via CP-SAT (`scripts/verify_optima.py`) | ft06=55, ft10=930, ft20=1165, all CP-SAT-proven (`bound == Cmax`). 104/104 passed. |
| Static env respects precedence + machine-capacity constraints | `tests/test_env.py::TestConstraintSafety` — 6 rules × 2 instances × 3 seeds (36 combos) rolled out and independently re-checked via `is_schedule_feasible` | 36/36 feasible, Cmax ≥ static lower bound in every case. 51/51 passed. |
| GIN+PPO pipeline runs end-to-end and learns | `experiments/train_ft06.py`, 20k timesteps | 560 episodes, zero crashes, mean makespan 98.5→67.8, best-seen 57 (2 above proven optimum 55). |
| Dynamic env's 3 disruption generators are correctly calibrated | `tests/test_disruptions.py`, N=5000 draws/assertion, calibrated against closed-form targets | Breakdown availability within 0.01% of target `A_k` ∈ {0.95, 0.90, 0.80}; processing-time variance within 2.6% of `exp(s²)-1`; arrival rate within 0.8% of target `λ_a`. 25/25 passed. |
| Dynamic env respects constraints under breakdowns + stochastic times | `tests/test_dynamic_env.py::TestConstraintSafetyUnderBreakdown` — 6 rules × 2 instances × 3 seeds, re-checked via `is_schedule_feasible_dynamic` | 36/36 feasible, including a 60-seed stress sweep with all 3 disruption axes on simultaneously. 88/88 passed. |
| Dynamic env terminates correctly under job arrivals | `tests/test_dynamic_env.py::TestDynamicArrivals` | Episodes terminate exactly when every arrived job (not just the base instance) completes; step count bounded by `max_possible_ops` across 8 seeds. |
| Dynamic env reduces exactly to the static env when disruptions are off | `test_default_config_matches_static_env_exactly` (12 rule × instance combos) | Reward, termination, makespan, and every dispatch-status-dependent feature match step-for-step; the 3 completion-gated features converge to an exact match by the terminal step (proved by construction, not just observed). |

## Test suite state as of Week 24 (re-run fresh for this close-out)

```
$ pytest -q -m "not slow"
589 passed, 3 failed, 1 deselected in 357.80s
```

The 3 failures are new since Week 21's "593 passing" baseline, all in
`tests/test_hetero_policy.py`, all the same root cause: `build_hetero_maskable_ppo`
constructs a policy via `MaskablePPO(...)` directly rather than the raw
`PerNodeMaskableActorCriticPolicy(...)` construction path, and SB3's own device="auto"
resolution ends up splitting the model across cuda/cpu on this machine (GPU driver
reports "Driver/library version mismatch" via `nvidia-smi`, yet `torch.cuda.is_available()`
returns `True`, and something in that partially-cuda environment leaves the model's
`action_net` weight on cpu while `obs_to_tensor` moves inputs to cuda). This is a
**test/training-helper environment issue, not a modeling or environment-correctness
bug**: confirmed by direct reproduction that `baselines.interface.TrainedRLBaseline`'s
own evaluation path (raw policy construction, explicit `device="cpu"`) does not hit this
code path and runs correctly end-to-end (see `../docs/logs/EXPERIMENT_LOG_WEEK24.md`). Flagged for a
follow-up fix (pin `device="cpu"` explicitly wherever `MaskablePPO`/`build_hetero_maskable_ppo`
is constructed in a CPU-only or mixed-driver environment) but does not block any Week 24
evaluation, since none of E1/E2/E3/E8 go through that construction path.

## The one honest gap

D4's original E0 scope named an "M/M/1 analytic-queue case" check, and no such test
exists in the suite. This was **not** built this sprint, per §2.0's own recommendation
("flagging it as optional, not blocking" — the closed-form M/M/1 comparison is a nice-to-have
sanity net for the arrival-process calibration already covered statistically above, not
a load-bearing validation result). If the committee asks for it specifically, it is a
single cheap unit test (simulate a long single-server queue, compare mean queue length
and wait time against the M/M/1 closed form) — not attempted this window so as not to
displace E1–E3/E8 compute time.

## Conclusion

E0 is closed. The static and dynamic environments, the instance loader, and the base
training pipeline are validated to the standard Chapter 6.1 needs: literature-optima
reproduction, constraint safety under both nominal and all three disruption axes,
statistically calibrated disruption generators, and a proven exact-reduction regression
contract between the dynamic and static environments. No further compute is scheduled
against E0 this sprint.
