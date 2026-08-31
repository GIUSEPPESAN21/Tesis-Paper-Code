# Statistical Analysis Pipeline — Technical Specification

**Status:** Specification only. No code in this repository implements this yet — nothing here has been executed. This document is written to be handed directly to an implementer (human or AI-assisted) for implementation once E1–E8 results exist.
**Companion documents this spec is bound by:** `PreRegistration.md` (frozen 2026-07-29T05:22:26Z UTC — every statistical test, alternative hypothesis, and accept/reject threshold below is copied from there, not reinvented), `../docs/planificacion/Step 4 Thesis.md` Part D6 (exact formulas for RPD/RI/effect size/CD), `../docs/planificacion/FinalExperimentMatrix.md` (which experiments produce which CSVs, on what schedule), `REFACTOR_NOTES_WEEK15.md` / `run_evaluation.py` (the input CSV schema this pipeline consumes).
**Consumer:** whoever executes E1–E8 (Semana 24/27 per `../docs/planificacion/FinalExperimentMatrix.md`) and Chapter 6/7 of the thesis.

---

## 1. Purpose and scope

This pipeline turns the long-format CSVs `run_evaluation.py` (the Week 15 unified harness) produces for E1–E8 into: (a) every summary statistic Chapter 6 needs (RPD, robustness index, decision latency), (b) the confirmatory hypothesis tests H1–H5 as `PreRegistration.md` defines them — exact statistic, p-value, effect size, decision-rule verdict — and (c) publication-ready tables and critical-difference diagrams.

**Explicitly out of scope for this document:** running any experiment, generating any CSV, or changing `run_evaluation.py`'s schema. This is a light-week planning document; Semana 24/27 execute against it.

**Design principle:** every accept/reject computation in `PreRegistration.md` is reproduced here as a *named function with the hypothesis's exact wording pinned into its docstring and its default arguments*, not as a generic "run a Wilcoxon test" utility a caller has to remember to configure correctly. Getting a sign or an `alternative=` argument backwards on H1's accept/reject rule is the single highest-consequence bug this pipeline could contain — per the master plan's own words for the sprint that runs this code, "this is what the thesis defense will be tested on most rigorously." §7 below treats this as the primary design constraint, not an afterthought.

---

## 2. Input contract

### 2.1 CSV schema (from `run_evaluation.py`, unchanged)

One row per `(instance, method, disruption, seed)`, long format:

```
instance, family, num_jobs, num_machines, method, disruption, seed, makespan,
known_optimum, rpd_pct, wall_clock_s, feasible, status, is_optimal,
violations_count, timestamp
```

Column notes the ingestion layer (§3.1) must honor, not re-derive differently:

- `rpd_pct` and `known_optimum` are **already computed by the harness** for JSSP rows and for the real-Brandimarte FJSP rows (`mk01`–`mk07`, per `../docs/planificacion/FinalExperimentMatrix.md` §5's amendment). They are **blank** for synthetic FJSP rows (no literature optimum exists) — the pipeline must not attempt to fill these in from `known_optimum` columns of unrelated files; a blank `rpd_pct` means "use raw makespan comparison for this row," per `PreRegistration.md` §1's own Arm-A instruction.
- `status` is `"OK"` (PDR, always), or CP-SAT's solver status (`OPTIMAL`/`FEASIBLE`/`INFEASIBLE`/`UNKNOWN`). `is_optimal` is `True` only when `status == "OPTIMAL"`.
- `feasible` is an independently-checked boolean (`envs.disjunctive_graph.is_schedule_feasible`), not merely "the solver said so." **A row with `feasible == False` must never enter a Wilcoxon/Friedman comparison** — see §3.2's validation gate.
- `disruption` ∈ `{none, breakdowns, stochastic_times, arrivals, combined}`. E4/E5/E6 each use exactly one non-`none` value; `combined` is descriptive-only per `PreRegistration.md` §4, never part of a confirmatory H3 test.

### 2.2 Known gap: RL rows do not currently distinguish deterministic from stochastic rollouts

`PreRegistration.md` §2 requires, per JSSP instance: **one deterministic rollout** (the value that enters the confirmatory Wilcoxon test) **plus four stochastic rollouts, seeds 1–4** (descriptive only — reported as mean ± std, excluded from the accept/reject computation). `baselines.interface.TrainedRLBaseline` fixes `deterministic` at *construction* time, not per call, so producing both groups requires two separate `run_evaluation.py` invocations (one with a deterministic-configured adapter, one with a stochastic one across `--seeds 1 2 3 4`) whose output CSVs get concatenated — and **the resulting merged CSV has no column that says which rows came from which invocation.**

This pipeline cannot silently guess. §3.1 specifies the required convention and a hard validation check:

- **Convention (must be enforced by whoever runs E1/E2, documented here so the stats code can assume it):** `seed == 0` is reserved, for `method == "rl"` rows only, for the single deterministic rollout. `seed ∈ {1, 2, 3, 4}` are the four stochastic descriptive rollouts. This mirrors the CSV's existing `--seeds 0` default and needs no schema change.
- **Validation (§3.2, `validate_rl_rollout_convention`):** for every `(instance, disruption)` block where `method == "rl"` appears, assert exactly one row has `seed == 0` and the remaining rows (if present) have `seed ∈ {1,2,3,4}` with no duplicates. Raise a loud, specific `DataContractError` if violated — do not fall back to "use the first row found," which would silently pick an arbitrary rollout as the confirmatory one.
- **Recommendation flagged, not actioned here:** a `rollout_mode: Literal["deterministic", "stochastic"]` column on the harness side would remove this fragility entirely. That is a `run_evaluation.py` change, out of scope for this spec — note it as an open item (§12) for whoever runs Semana 24, since it is cheap to add before E1 executes and expensive to retrofit after.

### 2.3 Multi-file ingestion

Each experiment (E1–E8) produces its own CSV (and, per `../docs/planificacion/FinalExperimentMatrix.md` §2.2, E2 produces two — Arm A synthetic, Arm B real-Brandimarte). The pipeline must:

1. Load each experiment's CSV(s) independently, tagged with an `experiment_id` (`"E1"`...`"E8"`, `"E2a"`/`"E2b"` for the two FJSP arms) added as a derived column at load time — **not** read from the CSV, since `run_evaluation.py` has no notion of experiment IDs.
2. Never silently concatenate across experiments with different `disruption` values into one Friedman/Wilcoxon call — each hypothesis's test operates on exactly the instance/disruption slice `PreRegistration.md` specifies (§6 below pins this per hypothesis).
3. Keep E1's rows and E8's timing columns as **the same rows** (per `../docs/planificacion/FinalExperimentMatrix.md` §2.8, E1 and E8 share one execution pass) — `wall_clock_s` on an E1 row *is* E8's raw timing input; §5.3 derives ms/decision from it rather than expecting a separate E8 CSV to exist.

### 2.4 Derived columns required at ingestion (`experiments/stats/io.py`)

| Derived column | Formula | Needed for |
|---|---|---|
| `num_operations` | JSSP rows: `num_jobs * num_machines`. FJSP rows: look up `ops_per_job * num_jobs` from a small static table (7 Brandimarte instances — hardcode from `data/fjsp_instances/`, do not recompute from `num_jobs`/`num_machines` alone, since FJSP padding makes this non-multiplicative) | E8 decision latency |
| `ms_per_decision` | `wall_clock_s / num_operations * 1000` | E8 |
| `rpd_or_makespan` | `rpd_pct` if not null, else raw `makespan` (flagged with a `uses_raw_makespan: bool` sibling column) | Uniform handling of Arm-A FJSP rows (no RPD) alongside everything else that has one |
| `experiment_id` | Set by the loader per §2.3, not read from file | Routing to the correct hypothesis test |

---

## 3. Module layout

```
experiments/stats/
    __init__.py
    io.py              # load_results(), validate_schema(), validate_rl_rollout_convention()
    metrics.py          # compute_rpd(), compute_ri(), compute_decision_latency(), summary_stats()
    pairwise.py          # wilcoxon_signed_rank() generic + h1_test(), h3_regime_test(), h4_test(), h5_check()
    omnibus.py           # friedman_test(), nemenyi_posthoc(), critical_difference()
    cd_diagram.py        # plot_cd_diagram()
    tables.py            # to_markdown_table(), to_latex_table()
    report.py            # orchestrates all of the above into the H1-H5 verdict table + Chapter 6 tables/figures
    cli.py               # `python -m experiments.stats.cli --experiment-dir experiments/results`
tests/
    test_stats_io.py
    test_stats_metrics.py
    test_stats_pairwise.py
    test_stats_omnibus.py
    test_stats_cd_diagram.py
    test_stats_report_golden.py   # regression-pins against already-published Week 21 numbers, see §10
```

Follow the codebase's existing conventions (`envs/fjsp_loader.py` is the reference example): `from __future__ import annotations`, `@dataclass(frozen=True)` for result containers, Google-style `Args`/`Returns`/`Raises` docstrings, no bare `except`, functions raise a specific exception class rather than returning `None` on error.

---

## 4. Dependencies

- `scipy.stats` — `wilcoxon`, `friedmanchisquare`, `norm` (already used elsewhere in the codebase per `PreRegistration.md` §2's own note that this matches `tests/test_disruptions.py`'s tooling).
- `statsmodels.stats.libqsturng.qsturng` — for the Nemenyi critical-difference studentized-range quantile (§6.2). **Use this, not a hand-transcribed table**, for the primary computation; a small hardcoded reference table (§6.2) is included only as an offline cross-check / sanity assertion, not the production code path, because a transcription error in a hardcoded critical-value table is exactly the class of silent, hard-to-catch bug this pipeline exists to avoid.
- `matplotlib` — CD diagrams and any other figures (already a project dependency per `training_curves_*.png` outputs).
- `pandas` — long-format CSV handling, pivoting for Friedman blocks. (If the project prefers to stay numpy-only per its existing style, this is a one-line swap in `io.py`; flagged as an implementation choice, not a hard requirement.)

---

## 5. Metric functions (`metrics.py`)

All formulas below are copied verbatim from `../docs/planificacion/Step 4 Thesis.md` D6 — do not re-derive.

### 5.1 RPD

$$\text{RPD} = 100 \cdot \frac{C_{\max}^{\text{method}} - C^{*}}{C^{*}}$$

```python
def compute_rpd(makespan: float, optimum: float) -> float:
    """RPD (%) of a makespan against a reference optimum/best-known value.

    Args:
        makespan: Achieved makespan.
        optimum: Reference C* (proven optimum or best-known upper bound).

    Returns:
        RPD as a percentage. 0.0 means the reference was matched exactly.

    Raises:
        ValueError: If optimum <= 0.
    """
```

Already computed by `run_evaluation.py` for rows with a known reference; this function exists for (a) the FJSP Arm-B `status="best_known"` rows where an RPD is still meaningful but should be reported alongside its provenance flag (§2.4's `rpd_or_makespan`), and (b) E3's fresh tier baselines and E8's cross-checks, computed independently of the harness for verification.

### 5.2 Robustness Index (RI)

$$RI = \frac{\mathbb{E}[C_{\max}^{\text{pert}}] - C_{\max}^{\text{nom}}}{C_{\max}^{\text{nom}}}$$

```python
@dataclass(frozen=True)
class RobustnessResult:
    """RI plus the dispersion measures PreRegistration.md §4 requires alongside it."""
    ri: float
    std: float
    cvar_90: float          # CVaR_0.9 of the perturbed makespan distribution
    n_replicates: int
    nominal_makespan: float

def compute_ri(perturbed_makespans: np.ndarray, nominal_makespan: float) -> RobustnessResult:
    """Robustness index for one (instance, method, regime) cell.

    Args:
        perturbed_makespans: One value per replicate seed (expect len == 30 per the
            frozen D4 grid, but do not hard-fail on a different count -- log a warning
            instead, since a partially-completed sweep should still be inspectable).
        nominal_makespan: C_max under --disruption none for the same (instance, method).

    Returns:
        RobustnessResult with ri, std, cvar_90 (mean of the worst 10% of
        perturbed_makespans, i.e. worst 3 of 30 at full replication), n_replicates.

    Raises:
        ValueError: If nominal_makespan <= 0 or perturbed_makespans is empty.
    """
```

**Nominal-makespan source, pinned explicitly:** for the RL arm, `nominal_makespan` is the E1 deterministic-rollout makespan (§2.2's `seed==0` convention) for the same instance — do **not** recompute a separate nominal run for E4–E6; reuse E1's row. For PDR arms, `nominal_makespan` is the E1 CSV's PDR row for that instance (also already computed, no `--disruption none` re-run needed for E4–E6).

### 5.3 Decision latency

```python
def compute_decision_latency(wall_clock_s: float, num_operations: int) -> float:
    """Milliseconds per decision epoch.

    Args:
        wall_clock_s: Total episode wall-clock time (the harness's wall_clock_s column).
        num_operations: Number of decision epochs in one complete schedule (see §2.4's
            derivation rule -- this is num_jobs * num_machines for JSSP, a static
            per-instance lookup for the seven bundled FJSP instances).

    Returns:
        Milliseconds per decision, for direct comparison against CP-SAT's per-decision
        figure (BASELINES_RESULTS_WEEK12.md's wall_clock_s divided the same way).
    """
```

### 5.4 Summary statistics

```python
def summary_stats(values: np.ndarray) -> dict:
    """Mean, median, std, min, max, n. Median is D6's preferred summary for latency
    (avoids one slow-outlier episode dominating the number); mean +/- std is the
    convention used elsewhere (RPD, RI)."""
```

---

## 6. Pairwise comparisons — Wilcoxon signed-rank (`pairwise.py`)

### 6.1 Generic primitive

```python
@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float          # W, as reported by scipy
    p_value: float             # two-sided, always computed regardless of `alternative`
    p_value_one_sided: float | None  # populated iff alternative != "two-sided"
    z: float                   # signed z, derived per the procedure below -- not scipy's internal value
    effect_size_r: float       # z / sqrt(n_effective)
    n_nominal: int              # len(x) before zero-difference pairs are dropped
    n_effective: int            # pairs actually used (n_nominal minus exact ties)
    median_diff: float
    method: str                 # scipy's chosen method ("exact" or "approx"), recorded for the paper's methods section
    all_zero: bool               # True if every paired difference was exactly 0

def wilcoxon_signed_rank(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
) -> WilcoxonResult:
    """Paired Wilcoxon signed-rank test on (x - y), pinned parameters for reproducibility.

    Args:
        x, y: Paired samples, same length, same (instance, seed) order -- callers MUST
            sort/align both arrays by a shared key before calling this (see
            `align_paired` helper below); this function does not do it implicitly,
            because silent misalignment is a worse failure mode than a loud one.
        alternative: "two-sided" (default, matches H1/H3's two-sided frozen tests) or
            "less"/"greater" for the one-sided variants some ablation-style comparisons
            use descriptively (Week 21's Axis 2 reward-density test reported both).
        zero_method: "wilcox" (drop exact-zero-difference pairs; scipy default and the
            convention used throughout this pipeline) -- pinned explicitly rather than
            left to scipy's version-dependent default.

    Returns:
        WilcoxonResult. If all differences are exactly zero (this DOES happen -- see
        docs/logs/HPO_ABLATION_REPORT_WEEK21.md's reference-vs-gat_hybrid comparison, W=0,
        p=1.0, identical results in every seed), return statistic=0, p_value=1.0, z=0,
        effect_size_r=0, all_zero=True -- do not raise. scipy's own wilcoxon() raises
        ValueError on all-zero input under some zero_method/mode combinations; this
        wrapper must catch that specific case and return the degenerate-but-valid
        result above rather than propagate the exception.

    Raises:
        ValueError: len(x) != len(y), or either array is empty.
    """
```

**Computing `z` (the part that needs to be exact):** scipy's `wilcoxon()` does not expose a `.zstatistic` attribute across all supported scipy versions/methods, and the "exact" method (used automatically for small, tie-free `N`, which is most of this thesis's `N=5`-seed ablation-scale comparisons) has no native z-statistic at all — it's a permutation-based p-value. Compute `z` uniformly, regardless of which internal method scipy chose, by inverting the reported **two-sided** p-value through the standard normal quantile function, then re-attaching the correct sign:

```python
from scipy.stats import norm

magnitude = norm.ppf(1 - p_two_sided / 2)
sign = 1.0 if median_diff > 0 else (-1.0 if median_diff < 0 else 0.0)
z = sign * magnitude
```

This is the standard "effect size back-computed from a reported p-value" procedure used when the underlying test statistic isn't itself on a z-scale (exactly the situation here), and it is what `r = Z/√N` in `PreRegistration.md` §2/D6 refers to. Document this derivation in the function's docstring so a reader of the eventual results chapter — or a reviewer — can see exactly how `r` was obtained, not just cite the formula.

### 6.2 Per-hypothesis wrappers (`pairwise.py`, continued)

Each function below hard-codes the exact comparison `PreRegistration.md` specifies — the direction of subtraction, the alternative, and the accept/reject threshold are not caller-configurable, on purpose.

```python
def h1_jssp_test(rpd_best_pdr: np.ndarray, rpd_rl_deterministic: np.ndarray) -> HypothesisVerdict:
    """PreRegistration.md Section 2, JSSP arm. N=43, paired by instance.
    Computes (RPD_bestPDR - RPD_RL); accept iff p < 0.05 AND median diff > 0."""

def h1_fjsp_test(makespan_best_comparator: np.ndarray, makespan_rl: np.ndarray) -> HypothesisVerdict:
    """Section 2, FJSP arm. Restricted to GA/CP-SAT comparators (no PDR has an FJSP
    machine-selection rule). Run once per FJSP arm (synthetic and real-Brandimarte,
    per docs/planificacion/FinalExperimentMatrix.md's amendment) -- report separately, never pooled."""

def h3_regime_test(ri_pdr: np.ndarray, ri_rl: np.ndarray, regime_label: str) -> HypothesisVerdict:
    """Section 4. One call per one of the 9 regimes (3 breakdown levels + 3 sigma
    levels + 3 utilization levels), paired by (instance x seed). Comparator is the
    single best-RI PDR for that regime, matching H1/H3's "best PDR" convention --
    caller must have already selected that PDR's column before calling this, this
    function does not do best-of selection itself. Accept per-regime iff p<0.05 AND
    median(RI_PDR - RI_RL) > 0. H3 overall verdict (accept / reject / partial) is
    computed by report.py from the 9 per-regime verdicts, per Section 4's conjunctive
    rule -- not inside this function."""

def h4_test(ri_kappa: np.ndarray, ri_kappa0: np.ndarray, rpd_nominal_kappa: np.ndarray,
            rpd_nominal_kappa0: np.ndarray, kappa: float) -> HypothesisVerdict:
    """Section 5. Two independent checks per kappa in {0.5, 1.0}:
    (a) Wilcoxon on delta_RI = RI_kappa - RI_kappa0, paired by (instance x seed) x
        regime; accept-component iff p<0.05 AND median delta_RI < 0, in >= 5 of 9
        regimes (majority, NOT all 9 -- this differs from H3's conjunctive rule, see
        Section 5's own note on why).
    (b) delta_RPD_nominal = mean(rpd_nominal_kappa) - mean(rpd_nominal_kappa0),
        compared against the fixed margin delta=2 percentage points -- a THRESHOLD
        comparison, not a significance test (Section 5 is explicit that this asymmetry
        -- test for (a), margin for (b) -- is deliberate).
    Returns one of four labeled outcomes: "accept", "robustness gained at an
    unacceptable nominal cost" ((a) holds, (b) doesn't), "no measurable robustness
    benefit at this kappa" ((b) holds, (a) doesn't), "reject" (neither holds) --
    Section 5's own vocabulary, reproduced exactly so Chapter 6/7 can quote it."""

def h5_check(ms_per_decision_rl: np.ndarray, ms_per_decision_cpsat: np.ndarray) -> HypothesisVerdict:
    """Section 6. NOT a significance test -- Section 6 is explicit ("no test required").
    Accept iff, for every instance, median(ms/decision_RL) < 0.01 * median(ms/decision_CPSAT).
    Also compute and return the ratio ms/decision_CPSAT / ms/decision_RL per instance,
    for the "does the ratio widen with size" descriptive figure Section 6 asks for."""
```

`HypothesisVerdict` is a shared dataclass: `hypothesis: str`, `verdict: Literal["accept","reject","partial","inconclusive","blocked"]`, `test_result: WilcoxonResult | None`, `notes: str` (free text for the labeled H4 outcomes and H2/H3's per-tier/per-regime inconclusive cases), `raw: dict` (everything needed to regenerate the table row without re-running the test).

---

## 7. Omnibus comparisons — Friedman + Nemenyi (`omnibus.py`)

### 7.1 Friedman test

```python
@dataclass(frozen=True)
class FriedmanResult:
    chi_square: float
    df: int
    p_value: float
    k_methods: int
    n_blocks: int
    average_ranks: dict[str, float]   # method name -> mean rank across blocks

def friedman_test(pivoted: pd.DataFrame) -> FriedmanResult:
    """Friedman test over a (n_blocks x k_methods) matrix.

    Args:
        pivoted: Rows = blocks (instances), columns = methods, values = the metric
            being ranked (RPD for E1/E3, RI for a multi-method robustness comparison).
            Every cell must be present -- see the missing-data policy below.

    Returns:
        FriedmanResult. Ranks are computed method-averaged across blocks, ascending
        (rank 1 = best/lowest RPD), matching D6's convention.

    Raises:
        ValueError: If `pivoted` has any missing (NaN) cell -- Friedman requires a
            complete block design. Do not impute; the caller (report.py) must decide
            whether to drop the offending instance-block entirely or report the
            comparison as unavailable for that experiment, and must say which choice
            it made in the resulting report, not silently drop rows here.
    """
```

**Missing-data policy, stated once, applied everywhere:** an instance where any one method in the comparison is infeasible/failed makes that whole instance-block unusable for Friedman (which needs a complete design). `report.py` drops such blocks and prints exactly which instances were dropped and why, in both the console log and the generated Markdown table's footnote — never a silent `n_blocks` shrinkage a reader can't trace.

### 7.2 Nemenyi post-hoc and critical difference

```python
def critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """CD = q_alpha * sqrt(k*(k+1) / (6*n)), Demsar (2006) / D6's exact formula.

    q_alpha is the studentized range statistic (infinite df) divided by sqrt(2),
    computed via statsmodels.stats.libqsturng.qsturng(1 - alpha, k, np.inf) / sqrt(2).

    Cross-check (not the production path): for alpha=0.05, standard published q_alpha
    values are k=2: 1.960, k=3: 2.343, k=4: 2.569, k=5: 2.728, k=6: 2.850, k=7: 2.949,
    k=8: 3.031, k=9: 3.102, k=10: 3.164 (Demsar 2006, Table 5(b)). Assert the computed
    qsturng value matches this table to 2 decimal places wherever k<=10 as a unit test
    (tests/test_stats_omnibus.py) -- if it doesn't, that is a dependency/version bug to
    fix before trusting any CD diagram this pipeline produces.

    Args:
        k: Number of methods compared.
        n: Number of blocks (instances).
        alpha: Family-wise significance level, 0.05 throughout this thesis.

    Returns:
        The critical difference in average-rank units.
    """

def nemenyi_posthoc(friedman_result: FriedmanResult, cd: float) -> pd.DataFrame:
    """Pairwise significant-difference flags: |rank_i - rank_j| > cd.

    Returns a k x k boolean DataFrame (method names as both index and columns),
    True where the pair is significantly different at the family-wise alpha.
    Only call this if friedman_result.p_value < 0.05 -- report.py must check this
    and, if not rejected, report the comparison as "not significant, no post-hoc run"
    rather than running Nemenyi anyway on a non-significant omnibus result.
    """
```

---

## 8. Critical-difference diagrams (`cd_diagram.py`)

```python
def plot_cd_diagram(
    average_ranks: dict[str, float],
    cd: float,
    *,
    title: str,
    output_path: Path,
) -> Path:
    """Demsar-style CD diagram: a horizontal axis of ranks, one tick per method, with
    a bar/bracket connecting methods NOT significantly different from each other
    (rank difference <= cd). Matplotlib, no seaborn dependency (matches the project's
    existing training-curve plots, which are plain matplotlib).

    Args:
        average_ranks: method -> mean rank, from FriedmanResult.
        cd: from critical_difference().
        title: e.g. "E3 -- la21-la25 tier (H2)" -- experiment + hypothesis, always,
            so a reader flipping through a folder of PNGs can identify each one
            without opening it.
        output_path: PNG, 300dpi minimum (thesis print quality).

    Returns:
        output_path, unchanged, for chaining into report.py's manifest.
    """
```

One diagram per Friedman-significant comparison: each of E3's 4 size tiers (if significant), and any Axis-1-style encoder comparison the committee asks to see extended past the Week 21 ablation's `ft06`/`la16` pair. Non-significant comparisons still get a table (§9) but not a CD diagram — a CD diagram implies a rejected omnibus null; drawing one for a non-significant Friedman result would visually overstate the finding.

---

## 9. Output formats (`tables.py`)

### 9.1 Markdown

GitHub-flavored tables, one function producing the exact row shapes Chapter 6 needs directly (not a generic DataFrame-to-markdown dump the writer has to reshape by hand):

```python
def hypothesis_verdict_table(verdicts: list[HypothesisVerdict]) -> str:
    """The single table RESULTS_REPORT.md leads with: one row per H1-H5 (H1 split
    into JSSP/FJSP-Arm-A/FJSP-Arm-B sub-rows, H3 split into per-regime + overall, H4
    split per kappa), columns: Hypothesis | Test | Statistic | p-value | Effect size |
    Verdict | Notes."""

def rpd_summary_table(df: pd.DataFrame, group_by: list[str]) -> str:
    """Per-instance or per-tier RPD table: method columns, instance rows, mean +/- std
    where multiple seeds exist."""
```

### 9.2 LaTeX

```python
def to_latex_table(markdown_or_df, *, caption: str, label: str) -> str:
    """Booktabs style (\\toprule/\\midrule/\\bottomrule, no vertical rules), matching
    the formatting convention already implied by docs/planificacion/Step 4 Thesis.md's own table style.
    label is auto-prefixed tab:ch6- so cross-references in the LaTeX thesis source are
    consistent without hand-editing."""
```

### 9.3 Figures

CD diagrams (§8) plus, reusing the project's existing plotting conventions from `training_curves_*.png`: RPD-vs-instance-size scatter (E3), RI-vs-disruption-level line plot per axis (E4/E5/E6), ms/decision-vs-instance-size log-scale plot with the CP-SAT ratio overlaid (E8, directly serving `PreRegistration.md` §6's "does the ratio widen" figure).

### 9.4 File naming and layout

```
experiments/results/stats/
    tables/
        h1_h5_verdict.md        h1_h5_verdict.tex
        e1_rpd_summary.md       ...
    figures/
        cd_diagram_e3_tier1.png
        cd_diagram_e3_tier2.png
        ...
        e8_latency_vs_size.png
    stats_manifest.json          # every table/figure produced, source CSV(s), git-ish
                                  # provenance (timestamp, row counts in, rows dropped
                                  # and why) -- so a committee question about any single
                                  # number is traceable back to its exact inputs
```

---

## 10. Validation and self-tests

Beyond ordinary unit tests (one per function above, edge cases: empty arrays, all-tied Wilcoxon input, single-block Friedman, `k=1`), pin two **golden-value regression tests** against numbers `../docs/logs/HPO_ABLATION_REPORT_WEEK21.md` already published — if this pipeline's `friedman_test`/`wilcoxon_signed_rank` don't reproduce these exactly, the implementation has a bug, full stop:

```python
def test_friedman_matches_week21_ft06_ablation():
    """Reference/homogeneous/gat_hybrid RPDs from docs/logs/HPO_ABLATION_REPORT_WEEK21.md's
    ft06 ablation table (5 seeds each). Expect chi_square == 2.0000, p == 0.3679
    (to 4 decimals), df == 2."""

def test_wilcoxon_matches_week21_reward_density():
    """Dense vs sparse RPDs from the same report's Axis 2 table (5 seeds). Expect
    statistic (W) == 0, one-sided p == 0.0312, two-sided p == 0.0625."""

def test_wilcoxon_handles_all_zero_difference():
    """Reference vs gat_hybrid: both arms identical in every seed (all-zero diffs).
    Expect all_zero=True, statistic=0, p_value=1.0, effect_size_r=0.0 -- not a raised
    exception. This is not a hypothetical edge case; it already occurred in Week 21's
    own data."""
```

---

## 11. CLI and orchestration (`report.py` / `cli.py`)

```
python -m experiments.stats.cli \
    --results-dir experiments/results \
    --experiments E1 E2a E2b E3 E4 E5 E6 E7 E8 \
    --out-dir experiments/results/stats \
    [--skip-missing]   # if an experiment's CSV doesn't exist yet (e.g. mid-Semana-24,
                        # E4-E6 still blocked per docs/planificacion/FinalExperimentMatrix.md Finding B),
                        # skip it and note "not yet run" in the verdict table rather
                        # than fail the whole pipeline -- this needs to be runnable
                        # incrementally as results land through Semana 24/27, not only
                        # once at the very end
```

`report.py`'s top-level function iterates H1 → H5 in `PreRegistration.md`'s own order, calling the matching wrapper from §6.2, and for H2/H3 additionally the omnibus machinery from §7, writing every table/figure as it goes and recording the run in `stats_manifest.json`. Each hypothesis's block in the generated report literally reproduces `PreRegistration.md`'s "Decision rule (exact, fixed in advance)" quote alongside the computed verdict, so a reader never has to flip between two documents to check the test actually matches what was promised.

---

## 12. Open items (not blocking this spec, but flag before Semana 24 starts execution)

1. **RL rollout-mode column** (§2.2) — recommend adding `rollout_mode` to the harness's CSV schema before E1 runs, rather than relying on the `seed==0` convention. Cheap now, expensive to retrofit after CSVs exist.
2. **FJSP `num_operations` lookup table** (§2.4) — needs the 7 Brandimarte instances' post-padding operation counts hardcoded or computed once from `envs.fjsp_loader.load_fjsp_by_name(name).num_operations` at ingestion time; either is fine, just needs to exist before E8's FJSP-adjacent numbers (if any) are computed. E8's primary scope per `../docs/planificacion/FinalExperimentMatrix.md` is JSSP-only, so this is low priority.
3. **`statsmodels` as a new dependency** — confirm it's acceptable to add (it is a standard, widely-used package; `qsturng` specifically has no lighter-weight substitute for an exact studentized-range quantile). If rejected, fall back to the hardcoded table in §7.2 for `k<=10` only, and hard-fail with a clear error for any comparison needing `k>10` until the dependency question is resolved.
4. **`pandas` as a new dependency** (§4) — flagged as a style choice, not a requirement; the codebase currently leans numpy+dataclasses (`envs/fjsp_loader.py`, `baselines/interface.py`). Either is workable; pick one and be consistent across `io.py`/`omnibus.py`.
