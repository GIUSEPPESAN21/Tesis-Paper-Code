# Analysis plan: consistency of the conventional robustness index (RI) in dynamic job-shop scheduling

Status: fixed before any experiment of this study was run. Any later change is recorded in the
"Deviations" section at the end of this file, with its date and reason.

## 1. Research question
Does the conventional, self-normalised robustness index
RI = (E[C_max | perturbed] − C_max | nominal) / C_max | nominal
rank scheduling methods consistently with the quality they actually deliver under disruption?

Writing r = (C_nom − C*)/C* and q = (E[C_pert] − C*)/C* for the nominal and realized relative
deviations from a certified optimum C*, RI = (q − r)/(1 + r) holds identically, so at equal realized
quality q the index is strictly decreasing in r (∂RI/∂r = −(1+q)/(1+r)² < 0). The study measures
how large this effect is in practice when nominal and perturbed makespans of every method come from
the same execution semantics.

## 2. Design
* Instances (16, OR-Library, certified optima): ft06, ft10, ft20, la01, la02, la06, la07, la11,
  la12, la16, la17, la18, la21, la22, la26, la27 (6×6 to 20×10).
* Methods (11): CP-SAT (exact; 60 s wall-clock, 2 workers, random_seed 1), GA (60 s, seed 0), and
  nine dispatching rules (SPT, LPT, MWKR, LWKR, MOR, LOR, FIFO, EDD, CR). EDD and CR are excluded
  under dynamic arrivals because due dates of arriving jobs are undefined (9 methods in those regimes).
* Execution: one event-driven simulator (envs/dynamic_jssp_env.py). Rules run online (non-delay).
  CP-SAT and GA commit to their offline schedule, executed under right-shift repair (fixed
  per-machine sequence; operations of arriving jobs fill machine idle time, chosen by LWKR).
* Nominal reference: each method's makespan in the same simulator with all disruptions off (seed 0).
* Regimes (9): machine breakdowns with target availability A ∈ {0.95, 0.90, 0.80} (exponential
  time-to-failure, lognormal time-to-repair, MTTR = 2 × mean processing time, σ_r = 0.5,
  preempt-resume); stochastic processing times with unit-mean lognormal shape s ∈ {0.10, 0.25, 0.50};
  dynamic Poisson arrivals with target utilisation ρ ∈ {0.70, 0.85, 0.95} (at most 20 arriving jobs,
  cut-off at 80% of the total base processing time).
* Replications: 30 seeds (0–29) per instance × method × regime.

## 3. Metrics
r (nominal RPD), q (realized RPD of the mean perturbed makespan), RI, CVaR90 of the perturbed
makespan (mean of the worst 3 of 30) expressed as RPD from C*. Under arrivals C* is the base-instance
optimum, so absolute q values are inflated and only between-method comparisons are interpreted.

## 4. Hypotheses and decision rules (α = 0.05, Holm correction across the 9 regimes)
* H1 (self-normalisation bias). In each regime, the Spearman correlation across methods between mean
  nominal RPD and mean RI is negative and significant (exact permutation p). Supported in a regime if
  ρ < 0 and Holm-adjusted p < 0.05.
* H2 (rank inconsistency). In each regime, the Kendall τ_b between the method ranking by mean RI and
  the ranking by mean realized RPD is not positive (τ_b ≤ 0). Reported with its permutation p-value.
* H3 (cost of selecting by RI). For each instance and regime, Δ = q(method with the lowest RI) −
  q(method with the lowest q). Supported if the 95% bootstrap confidence interval (10,000 resamples
  of instances) of the mean Δ lies above 0 in a regime.
* H4 (price of optimality). In each regime, the exact solver's realized RPD is compared with that of
  the best dispatching rule of that regime (lowest mean realized RPD among the rules) by an exact
  Wilcoxon signed-rank test over the 16 instances; the direction and Holm-adjusted p are reported.
* Descriptive: Friedman test with Nemenyi critical difference on realized RPD and on RI per regime;
  three-comparator reporting (mean rule, best rule per regime, per-instance oracle rule).

## 5. Deviations
(none)
