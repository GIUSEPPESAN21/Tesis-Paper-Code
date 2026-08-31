› **Target venue:** IEEE CASE (Conference on Automation Science and Engineering) or Winter Simulation Conference (WSC). **Format note:** content is structured for IEEE two-column conversion (section/subsection hierarchy, table/figure placement suggested inline); LaTeX typesetting to be handled separately. **Target length:** ~6–8 pages IEEE two-column (≈5,500–6,500 words body text).

---

# Does Size Generalization Imply Disruption Robustness? A Pre-Registered Study of GNN–PPO Scheduling Policies for the Dynamic Flexible Job-Shop Problem

**Joseph Sanchez**
UNIMINUTO — Corporación Universitaria Minuto de Dios

---

## Abstract

Graph neural network (GNN) policies trained with deep reinforcement learning (DRL) have been shown to generalize to job-shop instances larger than those seen during training, and, separately, to be trainable for robustness against a single type of stochastic disruption. Whether these two properties — size generalization and multi-disruption robustness — co-emerge in the same policy has not, to our knowledge, been tested jointly under a statistically rigorous, pre-registered protocol. We train a relational-GIN, PPO-based scheduling policy for the dynamic, stochastic flexible job-shop problem (FJSP) and pre-register five hypotheses (H1–H5) with frozen decision rules before running any experiment, evaluating against nine priority dispatching rules (PDR), a genetic algorithm (GA), and CP-SAT across the Fisher–Thompson, Lawrence, and Brandimarte benchmark families. Four of five hypotheses are rejected. The policy does not match the best dispatching rule on static instances (H1, RPD 24.82% vs. 18.96%, $N=43$, $p=4.03\times10^{-5}$), is not robust under nine disruption regimes spanning machine breakdowns, stochastic processing times, and dynamic arrivals (H3, $p<1.7\times10^{-11}$ in all nine), and a disruption-aware reward term does not improve robustness without collapsing nominal performance (H4, +46.6 to +106.0 percentage points of nominal optimality-gap cost). Decision latency is 1–2 orders of magnitude below CP-SAT's on large instances but does not clear the strict pre-registered threshold on every instance (H5). Size generalization, in contrast, is accepted (H2): the policy's rank among ten methods is never significantly worse than the best dispatching rule's across four size tiers (Friedman–Nemenyi, $1.5\times$–$3.0\times$ training size), even though its absolute optimality gap is not competitive. We argue this pattern — architectural transfer of *rank* without transfer of *competitiveness*, and no automatic transfer of robustness at all — is itself the paper's central, defensible finding: generalization and robustness are separable properties of GNN–PPO scheduling policies, not a package deal, and we provide the first quantitative, root-cause-diagnosed evidence of that separation.

**Index Terms** — deep reinforcement learning, graph neural networks, flexible job-shop scheduling, dynamic scheduling, robustness, size generalization, pre-registration.

---

## I. Introduction

Discrete manufacturing scheduling — assigning operations to machines over time — remains one of the hardest decision problems to automate robustly on a real shop floor, precisely because real shops are neither static nor deterministic: machines fail, processing times deviate from their nominal estimates, and jobs arrive without full advance notice. Exact methods (constraint programming, MILP) guarantee optimality only at a computational cost that grows prohibitively with instance size; priority dispatching rules (PDR) and metaheuristics trade solution quality for speed; and neither family adapts natively to events that occur *during* schedule execution.

Deep reinforcement learning (DRL) combined with graph neural networks (GNN) has emerged as an alternative: instead of resolving each instance from scratch, a policy is trained once and applied, in amortized fashion, to the partial state of a shop represented as a graph. Zhang et al.'s L2D [1] showed that a GIN+PPO policy trained on small instances can outperform classical PDR and approach CP-SAT's solution quality on static job-shop instances of increasing size, with decision times orders of magnitude below an exact solver's. Subsequent work extended this paradigm to the flexible job-shop problem (FJSP) via heterogeneous graph representations that capture both operation-operation and operation-machine relations [2], [3].

However, a systematic 2015–2026 literature review (Section II) shows that the large majority of this line of work evaluates policies under static, deterministic conditions, or introduces *at most one type of disruption at a time* — either machine breakdowns, or dynamic arrivals, or stochastic processing times, but rarely more than one simultaneously, and never jointly with a size-generalization claim under a common statistical protocol. Two 2024–2026 threads come closest, each covering one axis: Smit et al. [4] demonstrate strong size generalization under a *single* source of uncertainty (stochastic processing times); Lassoued et al. [5] combine *two* disruption types (breakdowns and arrivals) but report no size-transfer experiment and use no graph encoder at all. **No work we are aware of tests both axes — size generalization and multi-disruption robustness — jointly, under a pre-registered, statistically powered protocol.** This is the gap this paper addresses.

We train a relational-GIN, PPO-based policy for the dynamic and stochastic FJSP and pre-register five hypotheses (H1: static competitiveness; H2: size generalization; H3: robustness under disruption; H4: whether a disruption-aware reward term improves robustness without a large nominal-performance cost; H5: decision-latency advantage) with decision rules frozen before any experiment was run, following the same anti-HARKing discipline increasingly expected in empirical machine learning research. Our central contribution is not a positive result — four of the five hypotheses are rejected — but a **diagnosed, statistically rigorous characterization of exactly where and why the joint objective fails**: generalization transfers architecturally (H2 accepted, via the same permutation-invariance argument that motivates GNN generalization theory [6]), while robustness does not emerge as a byproduct of that same architecture (H3 rejected) and cannot be induced by a naïvely-trained disruption-aware reward term without an unacceptable competitiveness cost (H4 rejected, with a diagnosed root cause: domain-randomized training that never exposes the policy to an undisturbed episode). We hold this pattern — not a claimed win — as the paper's strongest, most defensible contribution, precisely because it is the first quantitative evidence, under a controlled statistical protocol, that these two properties do not co-emerge for free.

---

## II. Related Work

**Foundational GNN+RL for static JSSP (2020–2021).** L2D [1] pairs a disjunctive-graph representation with a GIN encoder and a PPO policy, reporting 13.6–29.2% optimality gaps and outperforming SPT/MWKR/MOPNR dispatching rules on 87.5–92% of evaluated instances, with a policy trained on 20×20/30×20 instances retaining a 9.4% gap when evaluated zero-shot on 100×20 instances — far below the best PDR's 33.7–54.9% gap at that size. Park et al. [7] independently formalize a similar approach with a 14.7% mean relative error. Both are restricted to static, deterministic JSSP with fixed routing.

**Heterogeneous graphs for FJSP (2022–2023).** Song et al. [2] extend the disjunctive graph with explicit machine nodes and heterogeneous attention for FJSP; Lei et al. [3] propose a multi-action, multi-PPO scheme over Brandimarte instances. Both report qualitative superiority over PDR baselines; exact figures were not independently verifiable from open-access sources at the time of this review.

**2024–2026 frontier — the two axes in isolation.** Smit et al.'s SPM-DAN [4] extends a heterogeneous DAN architecture with a scenario-processing module that aggregates sampled realizations of stochastic processing times; trained on 20×10 instances, it maintains a $-7.10\%$ to $-7.07\%$ gap against CP-SAT when transferred, without retraining, to 30×10/40×10 — a markedly stronger absolute generalization result than ours (Section VI), obtained under a *single* uncertainty source and with an architecture purpose-built for it. Infantes et al.'s Wheatley [8] similarly targets generalization and scalability for JSSP under uncertain durations. On the robustness side without a generalization claim, Lassoued et al. [5] combine machine breakdowns (Weibull time-to-failure) and dynamic arrivals (Gamma process) over a colored Petri-net representation with a masked-PPO policy — no graph encoder, no size-transfer experiment — reporting 1.1–3% improvement over the best heuristic under breakdowns alone, up to 8% combined. This is, to our knowledge, the closest prior demonstration that a simple PPO policy *can* beat PDR under multi-type disruption **when the training budget is not shared with a size-generalization objective** — a contrast we return to in Section VI.

**Disruption-aware reward shaping, single event type.** Zhang et al. [9] propose a multi-agent D3QN scheme with disruption-aware reward shaping for FJSP with dynamic transport times — the closest precedent to our $\kappa$-weighted instability penalty (Section III-D), but designed and evaluated for one disruption type in a non-GNN, non-PPO architecture.

**The homogeneous/heterogeneous debate.** A near-universal assumption since 2022 is that FJSP requires a heterogeneous graph representation to scale. Hoss et al. [10], in 2026, directly challenge this, proposing a feature-based homogenization that achieves linear complexity without sacrificing scalability — evidence that heterogeneity is not strictly necessary for size generalization, and a point of reference for our own architectural choice (Section III-B).

**Gap.** Across this literature, eleven of twelve technical works we identified isolate a single uncertainty axis by design, even where the underlying architecture could in principle support more than one. No work jointly demonstrates, under one statistically powered protocol, that a GNN–PPO policy generalizes in size **and** is robust to multiple simultaneous disruption types. This is the gap our pre-registered study is designed to test directly, rather than assume.

---

## III. Method

### A. Problem Formulation

We address the dynamic, stochastic FJSP: a set of jobs $J=\{1,\dots,n\}$, each a fixed-precedence sequence of operations $O_{j,h}$, each processable on any machine in an eligible subset $M_{j,h}\subseteq M$ with machine-dependent processing time $p_{j,h,k}$. Three sources of uncertainty extend the deterministic base problem: (i) dynamic job arrivals, modeled as a Poisson process calibrated to a target utilization $\rho$; (ii) stochastic processing times, a log-normal multiplier of unit mean over the nominal duration; (iii) machine breakdowns, with exponential time-to-failure and log-normal time-to-repair, parameterized by a target steady-state availability $A_k$. The environment is realized as a discrete-event simulation exposed through a Gymnasium-compatible interface, with each decision epoch corresponding to a `step()` call and a legality mask recomputed at every step.

### B. Policy Architecture

The state at decision epoch $t$ is a heterogeneous graph $G_t=(O_t\cup M,\ E^{\text{prec}}\cup E^{\text{mach}}\cup E^{\text{elig}})$ over operation nodes $O_t$ and machine nodes $M$, connected by three typed relations: intra-job precedence ($E^{\text{prec}}$), shared-machine competition among undispatched operations ($E^{\text{mach}}$), and operation-machine eligibility ($E^{\text{elig}}$). We adopt a **relational GIN** — a single node type with three separate linear transformations per edge type, aggregated by sum before a GIN update — rather than the fully homogeneous graph of [1] or the fully heterogeneous, attention-based graph of [2]. This is a deliberate midpoint of the homogeneous/heterogeneous spectrum: more expressive than a single shared adjacency because it distinguishes the opposing semantics of each relation (a predecessor constrains, a successor is constrained, a machine competitor contests the same resource), without the parameter and compute cost of full heterogeneity, and consistent with Hoss et al.'s [10] 2026 evidence that heterogeneity is not a strict precondition for scalable generalization. Sum aggregation, not mean/max, is used throughout, following the expressiveness argument of Xu et al. [6] (1-WL-equivalent discriminative power) — critical for the policy to distinguish, e.g., a lightly- from a heavily-loaded machine.

Message passing at layer $l$ for an operation node $v$ and a machine node $k$:
$$h_v^{(l)} = \text{MLP}_O^{(l)}\Big((1{+}\epsilon^{(l)})h_v^{(l-1)} + W_{\text{pred}}\!\!\sum_{u\to v}\!h_u^{(l-1)} + W_{\text{succ}}\!\!\sum_{v\to u}\!h_u^{(l-1)} + W_{\text{mach}}\!\!\sum_{u\sim v}\!h_u^{(l-1)} + W_{\text{elig}}\,h_{m(v)}^{(l-1)}\Big)$$
$$h_k^{(l)} = \text{MLP}_M^{(l)}\Big((1{+}\epsilon_M^{(l)})h_k^{(l-1)} + W_{\text{cand}}\!\!\!\sum_{v:m(v)=k}\!\!\!h_v^{(l-1)}\Big)$$

A **per-node actor head**, shared across all operation embeddings, produces one logit per candidate operation regardless of instance size: $\text{logit}(o)=\text{MLP}_{\text{actor}}(h_o^{(L)})$, softmax-normalized over legal actions only (mask applied before log-probability computation, not as a reward penalty [11]). The actor's parameter count is, by construction, independent of the number of operations $N$ — the architectural property underlying the size-generalization hypothesis (H2). The critic consumes a mean-pooled graph embedding. Training uses `MaskablePPO` [12] with a clipped surrogate objective and GAE($\lambda$), $\gamma=1$ (justified by the reward's telescoping-sum structure, below).

### C. Baselines

Nine PDR (SPT, LPT, MWKR, LWKR, MOR, LOR, FIFO, EDD, CR — the last two requiring synthesized due dates via the Total Work Content rule, $d_j=r_j+1.5\cdot\sum_h p_{j,h}$, since the benchmark families used carry no published due dates); a genetic algorithm (permutation-with-repetition encoding, POX crossover, swap mutation, tournament selection with elitism, population 80 scaled down for larger instances, a 5% random-immigrant rate against premature convergence); and CP-SAT [13] as the exact/near-exact reference, all under a wall-clock budget matched to CP-SAT's (60 s) for fair comparison.

### D. Disruption-Aware Reward

The base reward is dense and potential-based: $r_t=(H(s_t)-H(s_{t+1}))\times\text{reward\_scale}$, where $H(s)$ is a state-dependent lower-bound estimate of the makespan (optimistic job-chain propagation ignoring machine contention on undispatched operations), guaranteeing $r_t\le0$ and a telescoping return $\sum_t r_t=(H(s_0)-C_{\max})\times\text{reward\_scale}$ — maximizing undiscounted return is thus exactly minimizing makespan [14]. To test H4, we extend this with an instability penalty: $r_t=(LB_{t-1}-LB_t)\times\text{reward\_scale}-\kappa\cdot\Delta_{\text{inest}}\times\text{reward\_scale}$, where $\Delta_{\text{inest}}$ is the deviation from planned start times and $\kappa\in\{0,0.5,1.0\}$ is bit-for-bit a no-op at $\kappa=0$ (unit-tested).

---

## IV. Experimental Design

Five hypotheses were pre-registered with frozen accept/reject rules before any policy evaluation was run, an increasingly-expected but still rare discipline in this literature (Section VI, Table I column "Statistical rigor"). **H1** (static competitiveness): accept if the policy's median RPD is below the *best PDR per instance* (not a fixed named rule) with Wilcoxon $p<0.05$. **H2** (size generalization): accept if, across size tiers $1.5\times$–$3.0\times$ the training size, the policy's Friedman rank is never significantly worse than the best PDR's (Nemenyi critical difference). **H3** (robustness): accept only if the robustness index $RI=(\mathbb{E}[C_{\max}^{\text{pert}}]-C_{\max}^{\text{nom}})/C_{\max}^{\text{nom}}$ is significantly below the best PDR's in *every one* of nine disruption regimes (three levels each of breakdowns, stochastic times, arrivals) — a conjunctive rule chosen deliberately over a majority rule. **H4**: accept for a given $\kappa$ only if (a) $\Delta RI<0$ significantly in $\ge5/9$ regimes **and** (b) nominal RPD does not increase by more than 2 percentage points — both conditions required. **H5**: accept only if median decision latency is below 1% of CP-SAT's for *every* instance across the full size range.

Evaluation spans 43 JSSP instances (Fisher–Thompson `ft06/ft10/ft20`; Lawrence `la01`–`la40`) and two FJSP arms (43 synthetic instances; 7 of 8 Brandimarte instances — `mk06` excluded after an upstream data-mirror inconsistency, 15 vs. 10 machines, was traced during instance loading). All pairwise comparisons use Wilcoxon's exact signed-rank test (confirmed via the pipeline's own method log, not the normal approximation, even at $N=43$); joint comparisons use Friedman with Nemenyi post-hoc and critical-difference diagrams [15]; effect sizes ($r=Z/\sqrt N$) are reported throughout.

---

## V. Results

**Table I** summarizes all five verdicts.

**TABLE I. Hypothesis verdicts.**

| Hyp. | Test | Statistic | Verdict |
|---|---|---|---|
| H1 (JSSP) | Wilcoxon, $N{=}43$ | $W{=}133$, $p{=}4.03{\times}10^{-5}$, $r{=}{-}0.626$ | **REJECT** |
| H1 (FJSP synth.) | Wilcoxon, $N{=}43$ | $W{=}0$, $p{=}1.12{\times}10^{-8}$, $r{=}{-}0.871$ | **REJECT** |
| H1 (FJSP Brandim.) | Wilcoxon, $N{=}7$ | $W{=}0$, $p{=}0.0156$, $r{=}{-}0.914$ | **REJECT** |
| H2 (4 tiers) | Friedman+Nemenyi | rank gap $\le1.00 <$ CD $=6.06$ in all tiers | **ACCEPT** |
| H3 (9 regimes) | Wilcoxon $\times$9 | $p<1.7{\times}10^{-11}$ in all 9 | **REJECT** |
| H4 ($\kappa$=0.5) | Wilcoxon $\times$9 + $\Delta$RPD | 0/9 sig.; $\Delta$RPD$_{\text{nom}}$=+106.0pp | **REJECT** |
| H4 ($\kappa$=1.0) | Wilcoxon $\times$9 + $\Delta$RPD | 1/9 sig.; $\Delta$RPD$_{\text{nom}}$=+46.6pp | **REJECT** |
| H5 (latency) | threshold, $N{=}43$ | 42/43 instances fail | **REJECT** |

### A. H1 — Static Competitiveness (Rejected)

Mean RPD over 43 JSSP instances: RL 24.82%, best-PDR-per-instance 18.96%, GA 5.52%, CP-SAT $\approx$0%. The policy beats the per-instance best PDR on only 8/43 instances (18.6%) and loses on all 43 FJSP-synthetic and all 7 Brandimarte instances. We note explicitly that the GA — under the same 60 s wall-clock budget as CP-SAT — outperforms both RL and the best PDR by an even larger margin than the PDR comparison alone suggests; we return to this asymmetry (training-time compute vs. evaluation-time search) in Section VI.

### B. H2 — Size Generalization (Accepted)

Across four size tiers ($1.5\times$ to $3.0\times$ training size, `la21`–`la40`), Friedman rejects equal performance decisively in each tier ($p<3.4\times10^{-6}$), and in each tier the policy's mean rank is not significantly different from the best PDR's ($\Delta$ rank 0.20–1.00, Nemenyi $CD=6.06$) — critical-difference diagrams place RL in the same statistically indistinguishable cluster as `fifo`, `mwkr`, `mor`, `spt` in all four tiers. This is an acceptance of **rank-relative**, not absolute, competitiveness: RL's mean RPD (23.9–32.9%) remains numerically above `fifo`'s (12.7–26.8%) at every tier. The size-RPD relationship is non-monotonic — the $3.0\times$ tier has the *lowest* mean RPD of the four, below even the $1.5\times$ tier — a genuine, corroborated finding, not a sampling artifact.

### C. H3/H4 — Robustness (Rejected)

The trained policy's robustness index is worse than the best PDR's in all nine disruption regimes, with large, consistent effect sizes ($r\approx-0.87$ throughout). The arrivals regime is the most severe ($RI\approx+1.65$ to $+1.82$) — the axis furthest from the policy's training distribution, since it presents genuinely unseen mid-episode job insertions, whereas breakdowns and stochastic times only perturb the timing of operations the policy already expects. Extending the reward with the $\kappa\cdot\Delta_{\text{inest}}$ instability penalty does not rescue robustness: at $\kappa{=}0.5$, all nine regimes show *significantly worse* robustness (not better); at $\kappa{=}1.0$, only one of nine regimes improves significantly, short of the pre-registered $\ge5/9$ threshold. Critically, both $\kappa$ values collapse nominal (undisturbed) performance catastrophically (+106.0 and +46.6 percentage points of RPD). We diagnose the root cause: training under domain-randomized sampling across all nine regimes, **without ever including an undisturbed episode**, is sufficient to explain the nominal collapse independent of whatever the $\Delta_{\text{inest}}$ term itself contributes — a finding with direct, transferable implications for any attempt to generalize single-disruption-type reward shaping [9] to a multi-disruption training regime.

### D. H5 — Decision Latency (Rejected under strict threshold)

RL's per-decision latency is essentially flat (4–18 ms) across the full instance-size range, while CP-SAT's total solve time is highly variable — near-instant on small, easy instances and saturating its 60 s budget on large or structurally hard ones. Under the strict "for every instance" criterion, 42/43 instances fail the 1% threshold, driven largely by small instances where CP-SAT's near-instantaneous solve produces an artificially tiny per-decision denominator — a construct-validity limitation of dividing a solver's one-shot solve time by operation count, not evidence that RL is slow. One large instance's CP-SAT measurement (`la40`, 7216.7 s vs. a structurally comparable instance's 60.1 s) was traced to a likely measurement anomaly via a raw-log audit; a sensitivity check substituting a comparable instance's value still rejects H5, and the rejection is *more*, not less, decisive under that correction — we report this diagnostic explicitly as evidence of the audit rigor applied throughout, not to explain away a result. For genuinely large, hard instances that saturate CP-SAT's time budget, RL retains a real 1–2 order-of-magnitude latency advantage — a qualitative finding consistent with H5's motivating intuition even though the literal threshold is not met.

---

## VI. Discussion

**H1 vs. L2D.** Our comparator (minimum RPD across nine PDR, evaluated per instance) is intrinsically harder to beat than L2D's three fixed rules (SPT/MWKR/MOPNR) — notably, `fifo`, the dominant rule in 93% of our instances, is not among L2D's comparators — so the raw win-rate figures (18.6% vs. L2D's 87.5–92%) are not directly comparable across papers. Training budget likely also differs by orders of magnitude, though L2D does not report a verifiable figure. H1's rejection should be read as "this instantiation, under this budget, is not competitive," not as a categorical claim about the architecture.

**H2 vs. L2D and SPM-DAN.** L2D also reports its generalization gap (9.4% at $4$–$5\times$ training size) staying far below the best PDR's gap at that size (33.7–54.9%) — the same qualitative pattern we observe, but starting from a far more competitive base policy, so L2D's generalization preserves absolute competitiveness where ours preserves only rank. This is consistent with, and illustrates cleanly, a reading of GNN generalization theory [6] in which size-invariance is a *multiplicative* property acting on whatever competitiveness the base policy already has, not a mechanism that improves it. SPM-DAN [4] generalizes far more strongly in absolute terms ($-7.10\%$ vs. our $23.9$–$32.9\%$) but under a single uncertainty source and with a scenario-processing module purpose-built for that axis — suggestive evidence that a dedicated architectural mechanism for the uncertainty axis being generalized outperforms a generic relational encoder without one.

**H3/H4 as the first joint test of the generalization-robustness combination.** Lassoued et al. [5] show that a plain PPO policy *can* beat PDR under combined breakdown-and-arrival disruption — when robustness is the only objective, with no size-generalization requirement sharing the same training budget and architecture. Our policy, pursuing both objectives simultaneously, loses decisively to the same class of PDR under disruption. Read together, this is the first quantitative evidence we are aware of that the joint objective is not free: a policy whose capacity and training regime are spent entirely on one axis can beat PDR on that axis; a policy splitting that budget across two axes loses decisively on the robustness axis. This does not prove the joint objective is unattainable — no architectural ablation isolating the relational encoder's specific contribution was run — but it is, to our knowledge, the first controlled demonstration that the tension is real and measurable, not merely assumed. The mechanism we diagnose for H4 (domain-randomized training without nominal exposure) is itself a transferable lesson for the broader disruption-aware reward-shaping literature [9], which has not previously needed to confront a multi-disruption training regime.

**Threats to validity.** Single training seed per $\kappa$ condition limits the precision (not the direction) of the H4 magnitudes; a structural asymmetry between $\kappa{=}0$ (a reused, statically-trained policy evaluated through an adapter) and $\kappa{>}0$ (trained natively against the dynamic contract) means $\Delta RI$ is not a fully clean isolation of the reward term's effect; all evaluation is simulation-only, with three parametric disruption generators (exponential/log-normal breakdowns, log-normal processing-time noise, Poisson arrivals) whose fidelity to real shop-floor disruption processes is assumed, not validated against field data.

**Practical implication.** The contrast between H1/H3's quality weakness and H5's qualitative speed strength motivates, but does not itself evaluate, a hybrid deployment pattern: a fast reactive policy as an immediate decision bridge following a disruption event, while an exact or metaheuristic re-optimizer computes a higher-quality replacement plan in parallel — assigning each method the role its own results support, rather than requiring one policy to be simultaneously the fastest and the best.

---

## VII. Conclusion

We pre-registered and tested five hypotheses about whether a relational-GIN, PPO-based scheduling policy can jointly generalize across instance sizes and remain robust to multiple simultaneous disruption types. Four of five hypotheses are rejected, one is accepted, and — critically — every rejection is diagnosed to a concrete, actionable mechanism rather than left as an unexplained negative result: a legality-semantics mismatch between static training and dynamic evaluation (H3), a domain-randomized training regime that never exposes the policy to undisturbed operation (H4), and a per-decision latency metric with a construct-validity weakness for one-shot solvers (H5). The accepted hypothesis (H2) demonstrates that size-invariant architecture transfers rank-relative competitiveness even from a weak base policy — direct empirical support for permutation-invariance-based GNN generalization theory, independent of whether the underlying policy is strong. Taken together, we hold this pattern — architectural transfer of one property, no automatic transfer of the other, with diagnosed causes for the gap — as the paper's central, defensible contribution: the first statistically rigorous, pre-registered evidence that size generalization and multi-disruption robustness are separable properties of GNN–PPO scheduling policies, not a package deal. Future work includes an architectural ablation isolating the relational encoder's contribution to each axis independently, a corrected domain-randomization training regime that preserves nominal-episode exposure, and multi-seed replication of the $\kappa$ ablation.

---

## References

[1] C. Zhang, W. Song, Z. Cao, J. Zhang, P. S. Tan, C. Xu, "Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning," *Proc. NeurIPS*, 2020.

[2] W. Song, X. Chen, Q. Li, Z. Cao, "Flexible Job-Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning," *IEEE Trans. Ind. Informat.*, vol. 19, no. 2, pp. 1600–1610, 2023.

[3] K. Lei, P. Guo, W. Zhao, Y. Wang, L. Qian, X. Meng, L. Tang, "A multi-action deep reinforcement learning framework for flexible job-shop scheduling problem," *Expert Syst. Appl.*, vol. 205, art. 117796, 2022.

[4] I. G. Smit, Y. Wu, P. Troubil, Y. Zhang, W. P. M. Nuijten, "Neural Combinatorial Optimization for Stochastic Flexible Job Shop Scheduling Problems," *Proc. AAAI*, 2025.

[5] S. Lassoued, S. Lier, A. Schwung, "Policy-Based Reinforcement Learning with Action Masking for Dynamic Job Shop Scheduling under Uncertainty: Handling Random Arrivals and Machine Failures," arXiv:2601.09293, 2026.

[6] K. Xu, W. Hu, J. Leskovec, S. Jegelka, "How Powerful are Graph Neural Networks?" *Proc. ICLR*, 2019.

[7] J. Park, J. Chun, S. H. Kim, Y. Kim, J. Park, "Learning to schedule job-shop problems: representation and policy learning using graph neural network and reinforcement learning," *Int. J. Prod. Res.*, vol. 59, no. 11, pp. 3360–3377, 2021.

[8] G. Infantes, S. Roussel, P. Pereira, A. Jacquet, E. Benazera, "Learning to Solve Job Shop Scheduling under Uncertainty," *Proc. CPAIOR*, 2024.

[9] L. Zhang, Y. Yan, C. Yang, Y. Hu, "Dynamic flexible job-shop scheduling by multi-agent reinforcement learning with reward-shaping," *Adv. Eng. Informat.*, vol. 62, art. 102872, 2024.

[10] J. Hoss, M. Link, N. Klarmann, "Scalable Production Scheduling: Linear Complexity via Unified Homogeneous Graphs," arXiv:2604.23841, 2026.

[11] S. Huang, S. Ontañón, "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms," *Proc. 35th Int. FLAIRS Conf.*, 2022.

[12] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, N. Dormann, "Stable-Baselines3: Reliable Reinforcement Learning Implementations," *J. Mach. Learn. Res.*, vol. 22, no. 268, pp. 1–8, 2021.

[13] L. Perron, F. Didier, S. Gay, "The CP-SAT-LP Solver," *Proc. 29th Int. Conf. Principles Pract. Constraint Program. (CP 2023)*, LIPIcs, vol. 280, art. 3, 2023.

[14] A. Y. Ng, D. Harada, S. Russell, "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping," *Proc. ICML*, 1999, pp. 278–287.

[15] J. Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets," *J. Mach. Learn. Res.*, vol. 7, pp. 1–30, 2006.

[16] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.

[17] R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018.

[18] M. L. Pinedo, *Scheduling: Theory, Algorithms, and Systems*, 6th ed. Springer, 2022.

[19] B. Roy, B. Sussmann, "Les problèmes d'ordonnancement avec contraintes disjonctives," Note DS n.9 bis, SEMA, 1964.

[20] A. M. Law, *Simulation Modeling and Analysis*, 6th ed. McGraw Hill, 2024.

[21] R. G. Sargent, "Verification and Validation of Simulation Models," *Proc. 2011 Winter Simulation Conf.*, IEEE, 2011, pp. 183–198.

[22] I. G. Smit, J. Zhou, R. Reijnen, Y. Wu, J. Chen, C. Zhang, Z. Bukhsh, Y. Zhang, W. Nuijten, "Graph Neural Networks for Job Shop Scheduling Problems: A Survey," *Comput. Oper. Res.*, 2024.

---

## Author's Note on Framing (not for inclusion in submission)

The single strongest, most defensible claim to lead with — and the one this draft is built around — is **not** "our policy generalizes" (H2, true but weak: rank-equivalence, not absolute competitiveness, and the least novel of the five results relative to prior work like L2D) and **not** a disguised negative-results dump (four rejections presented apologetically). It is the **structural/diagnostic claim**: *generalization and robustness are empirically separable properties of GNN–PPO scheduling policies, demonstrated here for the first time under a pre-registered, statistically powered, jointly-tested protocol, with root causes identified for every rejection.* This framing is defensible for three concrete reasons worth keeping in mind for reviewer response: (1) it does not depend on RL "winning" any comparison, so it cannot be undermined by a reviewer pointing out RL loses to PDR/GA — that loss is part of the finding, not a threat to it; (2) it is grounded in the pre-registration timestamp and frozen decision rules, which preempts the most common desk-reject reason for negative-result RL papers (suspicion of post-hoc rationalization); (3) it directly fills the specific, citable gap identified in Section II (Table-style gap analysis omitted here for length but available in the source thesis, Chapter 3 §3.6) — reviewers at WSC in particular tend to respond well to a clearly-stated gap-to-hypothesis-to-verdict chain. If reviewer feedback pushes toward a more "positive" framing, the fallback second-strongest claim is H2 itself (rank-preserving generalization from a weak base policy as evidence for a multiplicative theory of GNN size-invariance) — weaker as a headline but easier to defend to a reviewer uncomfortable with a majority-negative results paper.
