# AI in Industrial Engineering: Trend Investigation, Thesis Topic Recommendation, and Full Methodological Plan

> **Important tooling note (read first):** Live web search and web fetch were unavailable during this investigation. Consequently, exact 2023–2026 publication counts, growth percentages, current leaderboard numbers, DOIs/URLs, and verbatim quotes could **not** be verified against live sources. All paper identities (e.g., Zhang et al. "Learning to Dispatch," NeurIPS 2020; Song et al. FJSP-GNN, IEEE TII 2023; Bergmann et al. MVTec AD, CVPR 2019; Saxena & Goebel C-MAPSS, PHM 2008), dataset contents, and software-tool identities below are drawn from established domain knowledge and were cross-checked against an independent knowledge scaffold. **Every citation, URL, and citation count must be click-verified before use in the thesis.** Treat this as a rigorously reasoned expert plan, not a live-sourced bibliometric audit.

## TL;DR
- The single highest-momentum, best-fit topic for a simulation-only master's thesis at the intersection of AI + (Manufacturing/Production & Quality/Maintenance) is **Deep Reinforcement Learning (DRL) for Job-Shop and Flexible Job-Shop Scheduling (JSSP/FJSP)** — specifically **graph-neural-network policies for *dynamic/stochastic* scheduling** — mature enough to have standard benchmarks and open Gym environments, yet with clearly stated open gaps (generalization, dynamic events, multi-objective, baselining rigor).
- Strong runner-ups: (2) **diffusion/GAN synthetic-data augmentation for few-shot surface-defect detection** (Quality); (3) **RUL/PHM on the C-MAPSS turbofan benchmark with calibrated uncertainty quantification** (Maintenance).
- All recommended paths run entirely on public benchmarks + simulation (Taillard/DMU/OR-Library instances and JSSEnv/schlably/JobShopLib Gym environments, or C-MAPSS / MVTec AD), require only a single consumer GPU, and target venues like *Journal of Manufacturing Systems*, *IEEE T-ASE*, *Computers & Industrial Engineering*, IEEE CASE, and the Winter Simulation Conference.

## Key Findings
1. Across 2023–2026, four themes dominate AI-in-industrial-engineering publishing: **predictive maintenance/RUL**, **deep-learning visual defect/anomaly detection**, **digital twins**, and **DRL for production scheduling**. Predictive maintenance and defect detection are the most **saturated** (largest volume, hardest to claim novelty). DRL scheduling and generative/synthetic data are the **highest-momentum, least-saturated** clusters with clear open gaps.
2. DRL scheduling is uniquely suited to a simulation-only thesis because **the "data" IS the simulation**: standard instance libraries (Taillard, Demirkol/DMU, Lawrence, OR-Library) and open Gymnasium environments (JSSEnv, schlably, JobShopLib) provide reproducible experimentation with no real-plant data.
3. The dominant method family is a **disjunctive-graph state encoded by a Graph Neural Network with a Proximal Policy Optimization (PPO) policy**, established by Zhang et al.'s "Learning to Dispatch" (L2D, NeurIPS 2020) and extended to the flexible job shop by Song et al. (IEEE TII 2023).
4. Recurring stated gaps: **poor size-generalization; weak handling of dynamic/stochastic events** (machine breakdown, stochastic processing times, new job arrivals); **single-objective (makespan) bias; and non-standardized baselining** against OR-Tools CP-SAT and metaheuristics.
5. The recommended thesis narrows to: **a GNN-based DRL scheduler for dynamic FJSP under stochastic disruptions, benchmarked for generalization and compared rigorously against dispatching rules, a genetic algorithm, and CP-SAT** — a defensible, publishable, resource-light contribution.

## Details

# DOCUMENT 1 — Topic Investigation, Subject Primer & Research Question

## Part 1 — Bibliometric / Trend Investigation

### 1.1 The macro-landscape (2023–2026)
The literature on AI applied to Industrial Engineering has consolidated around a handful of high-volume themes, visible in the flow of systematic reviews in *Journal of Manufacturing Systems*, *Journal of Intelligent Manufacturing*, *Computers & Industrial Engineering*, *IEEE Transactions on Automation Science and Engineering (T-ASE)*, *IEEE Transactions on Industrial Informatics (TII)*, and *Reliability Engineering & System Safety*. The dominant clusters and their fit to your constraints:

| Theme | Volume | Growth trajectory | Saturation | Sim/synthetic-data feasibility | Fit (Mfg / Q&M) |
|---|---|---|---|---|---|
| Predictive maintenance / PHM / RUL | Very high | Steady, mature | High (saturated) | Excellent (C-MAPSS is simulated) | Maintenance |
| DL surface-defect / visual inspection & anomaly detection | Very high | Growing | High (saturated) | Good (synthetic images, MVTec) | Quality |
| Digital twins | High | Explosive but review-heavy | Medium; often conceptual | Good | Both |
| **DRL for production scheduling (JSSP/FJSP)** | High | **Fast growth since 2020** | **Low–medium (open gaps)** | **Excellent (benchmark + Gym envs)** | Manufacturing |
| Generative AI / LLMs in manufacturing | Emerging | Fastest relative growth | Very low; mostly position papers | Mixed | Both |
| Synthetic data (GANs/diffusion) for industrial data | Emerging | Fast growth | Low | Excellent by definition | Quality/Maintenance |
| Physics-informed NN (PINNs) in PHM/process | Emerging | Growing | Low–medium | Good | Maintenance |
| GNNs for scheduling/production systems | Emerging | Growing (backbone of DRL scheduling) | Low | Excellent | Manufacturing |
| Few-shot / self-supervised defect detection | Emerging | Growing | Low–medium | Good | Quality |
| Transfer learning / domain adaptation | High | Steady | Medium | Good | Both |
| Industry 5.0 / human-centric AI | High (conceptual) | Growing | Conceptual | Weak for a technical thesis | Both |

**Interpretation.** For a master's thesis needing a *defensible novelty claim* under a *simulation-only* constraint, the ideal zone is a theme that is (a) high-momentum, (b) not yet saturated, (c) natively simulation-based, and (d) equipped with standard benchmarks so results are comparable to prior work. **DRL for scheduling** sits precisely in that zone; **synthetic-data defect detection** and **RUL with uncertainty quantification** are the strong runners-up.

### 1.2 Theme-by-theme characterization

**Predictive maintenance / PHM / RUL.** Among the largest and most mature AI-in-manufacturing literatures. Organized around the NASA C-MAPSS turbofan degradation benchmark (Saxena & Goebel, PHM 2008) and bearing datasets (CWRU, FEMTO-ST/PRONOSTIA, IMS). Dominant methods: CNN and LSTM regressors, then Transformers and, recently, physics-informed and uncertainty-aware models. Seminal works: Babu et al. (deep CNN for RUL, DASFAA 2016); Zheng et al. (LSTM for RUL, IEEE ICPHM 2017). Venues: *Reliability Engineering & System Safety*, *Mechanical Systems and Signal Processing*, IEEE TII/TIE, PHM Society. **Saturation is high** — beating C-MAPSS state-of-the-art by a small margin is a weak thesis claim. Open gaps: uncertainty quantification, domain adaptation across operating conditions, data scarcity.

**Deep-learning surface-defect detection & industrial anomaly detection.** Very large volume, driven by the MVTec AD benchmark (Bergmann et al., CVPR 2019) and successors MVTec LOCO (IJCV 2022) and VisA (Zou et al., ECCV 2022). Manufacturing-specific defect sets: NEU surface defect database (steel strip, 6 classes, 1,800 images), DAGM 2007 (synthetic textures), Severstal steel defect (Kaggle 2019). Dominant methods: unsupervised anomaly detection (PatchCore, PaDiM, student-teacher/EfficientAD) plus supervised segmentation. **Saturation is high** on MVTec AD (scores near ceiling). Open gaps: few-shot/self-supervised learning, logical/contextual anomalies, domain shift, and **synthetic-data generation to overcome label scarcity** — the last is where genuine novelty remains.

**Digital twins.** Explosive publication growth but review-heavy and often conceptual; a rigorous technical thesis needs a concrete modeling core (e.g., a simulation-based digital twin driving a DRL scheduler), which folds back into scheduling or PHM.

**DRL for production scheduling (JSSP/FJSP).** The fastest-growing *technical* theme with clean open gaps. Catalyzed by Zhang et al.'s "Learning to Dispatch" (L2D, NeurIPS 2020), which introduced a disjunctive-graph state with a Graph Isomorphism Network encoder trained by PPO to construct schedules end-to-end. Park et al. (IJPR 2021) formalized GNN+RL for JSSP; Song et al. (IEEE TII 2023) extended GNN+DRL to the flexible job shop via a heterogeneous graph. Attention/pointer-network approaches derive from Vinyals et al. (Pointer Networks, NeurIPS 2015) and Kool et al. ("Attention, Learn to Solve Routing Problems!", ICLR 2019). Venues: IEEE T-ASE, IEEE TII, *International Journal of Production Research*, *Computers & Industrial Engineering*, *Journal of Manufacturing Systems*; methods at NeurIPS/ICML/ICLR. **Saturation is low–medium** with widely stated gaps.

**Generative AI / LLMs and synthetic data.** Newest and fastest relative growth; LLM work is mostly position/framework papers, but **GAN/diffusion synthetic data for defect and PHM datasets** is technically substantive and simulation-native.

### 1.3 The most-researched vs. the most-defensible
- **Most-researched right now (highest raw volume):** predictive maintenance/RUL and DL defect/anomaly detection.
- **Highest-momentum with defensible open gaps:** DRL for scheduling (especially dynamic/stochastic FJSP) and synthetic-data generation for quality inspection.
- **Best single choice for THIS user:** DRL for dynamic FJSP — high-momentum, not saturated, natively simulation-based, squarely in Manufacturing & Production, with a strong secondary tie to Quality & Maintenance when disruptions such as machine breakdown are modeled.

### 1.4 Recurring stated research gaps
- **DRL scheduling:** generalization to unseen instance sizes/distributions; dynamic/stochastic events (breakdowns, stochastic processing times, dynamic job arrivals); multi-objective beyond makespan (tardiness, energy); standardized baselining against OR-Tools CP-SAT and metaheuristics; sim-to-real; scalability/training cost.
- **RUL/PHM:** uncertainty quantification, domain adaptation, data scarcity, interpretability.
- **Defect detection:** label scarcity/imbalance, domain shift, logical anomalies, real-time/edge deployment, synthetic-data realism.

### 1.5 Benchmark datasets and open simulation environments

**Scheduling (recommended path):**
- **Taillard (1993)** benchmark instances (ta01–ta80) — canonical JSSP/flow-shop set (EJOR 64(2):278–285).
- **Demirkol, Mehta & Uzsoy (1998)** DMU instances (EJOR 109(1):137–141).
- **Lawrence (1984)** la01–la40; **Fisher–Thompson** ft06/ft10/ft20 classics.
- **FJSP-specific:** Brandimarte (MK01–MK10); Hurink (edata/rdata/vdata).
- **OR-Library** (J.E. Beasley) hosts JSSP instances.
- **Open Gym/Gymnasium environments:** JSSEnv (Tassel et al., 2021, arXiv:2104.03760, `prosysscience/JSSEnv`); schlably (Fraunhofer IML / TU Dortmund, SoftwareX 2023, `tmdt-buw/schlably`); JobShopLib (Gymnasium env with graph representations); the "Job Shop Scheduling Benchmark" consolidation codebase by Reijnen et al. (2023).
- **Solver/metaheuristic baselines:** Google OR-Tools CP-SAT; genetic algorithms; priority dispatching rules (SPT, MWKR, MOR, FIFO, LWKR).

**Maintenance/PHM:** NASA C-MAPSS (FD001–FD004) and N-CMAPSS; PHM Society data challenges; FEMTO-ST/PRONOSTIA bearing (IEEE PHM 2012); IMS bearing (Univ. of Cincinnati); CWRU bearing.

**Quality/defect:** MVTec AD; MVTec LOCO; VisA; DAGM 2007; NEU surface defect; Severstal steel defect; SECOM (UCI, 590 features, pass/fail); Bosch Production Line Performance (Kaggle).

**Industrial time-series anomaly:** SMD (OmniAnomaly, KDD 2019); SWaT and WADI (SUTD iTrust testbeds).

**Simulation & synthetic-data tools:** SimPy and salabim (open-source Python DES); AnyLogic (multi-method, free PLE); Siemens Tecnomatix Plant Simulation; Arena; NVIDIA Isaac Sim + Omniverse Replicator (physics sim + domain-randomized synthetic vision data); Blender + BlenderProc for synthetic labeled defect images.

## Part 2 — Recommendation and Topic Selection

### 2.1 Recommended topic
**"Generalizable Deep Reinforcement Learning for Dynamic Flexible Job-Shop Scheduling under Stochastic Disruptions."**
A graph-neural-network DRL policy (GNN encoder + PPO) that schedules a *flexible* job shop online, explicitly trained and evaluated for (a) generalization across instance sizes/distributions and (b) robustness to stochastic events (machine breakdowns, stochastic processing times, dynamic job arrivals), benchmarked rigorously against dispatching rules, a metaheuristic (GA), and OR-Tools CP-SAT.

**Justification against the six criteria:**
- (a) *Currency/momentum:* one of the fastest-growing technical themes since 2020; dynamic/stochastic FJSP is explicitly named as under-explored.
- (b) *Simulation-only feasibility:* the environment IS the data — Taillard/DMU/Brandimarte instances plus a DES layer (SimPy/JSSEnv/schlably) generate all training and test episodes; no real plant data needed.
- (c) *Fit:* core Manufacturing & Production; disruptions bring in Quality & Maintenance (breakdown/repair dynamics).
- (d) *Defensible gap:* generalization + dynamic-stochastic robustness + standardized baselining are all recurring, explicitly stated gaps.
- (e) *Compute:* trainable on a single consumer GPU (RTX-class, 8–12 GB) in hours-to-days; scheduling GNNs are lightweight.
- (f) *Publication potential:* fits IEEE T-ASE, *Computers & Industrial Engineering*, *Journal of Manufacturing Systems*, IEEE CASE, WSC.

### 2.2 Runner-up topics
1. **Diffusion/GAN synthetic-data augmentation for few-shot surface-defect detection** (Quality). Simulation-native (you generate the data), high novelty, uses MVTec AD/NEU/DAGM. Second because image-generation training is heavier on GPU and evaluating "does synthetic data actually help" is methodologically subtle.
2. **RUL estimation on C-MAPSS with calibrated uncertainty quantification** (Maintenance). Fully simulated benchmark, clean metrics. Second because the benchmark is saturated; novelty must come from UQ/calibration, not raw accuracy.
3. **Simulation-optimization via surrogate models / Bayesian optimization for a DES production line** (Manufacturing). Elegant and simulation-native, but a smaller literature and narrower publication surface.

## Part 3 — Introduction to the Subject (Primer on DRL for Scheduling)

### 3.1 Conceptual foundations and terminology
- **Job-Shop Scheduling Problem (JSSP):** *n* jobs, each a fixed sequence of operations, processed on *m* machines; each operation needs a specific machine for a fixed time; minimize an objective (classically **makespan**, C_max). NP-hard.
- **Flexible JSSP (FJSP):** each operation can run on any of a *set* of eligible machines (routing flexibility) → adds a machine-assignment decision on top of sequencing; harder and more realistic.
- **Dynamic/stochastic scheduling:** events unfold over time — new jobs arrive, machines break down, processing times are random — requiring online (re)scheduling.
- **Disjunctive graph:** the standard JSSP representation; nodes = operations, conjunctive arcs = precedence within a job, disjunctive arcs = machine-sharing constraints. Solving = orienting disjunctive arcs.
- **MDP framing:** *state* = current partial schedule/graph; *action* = choose next operation (and, in FJSP, machine); *reward* = shaped from makespan/tardiness; *policy* = neural network selecting actions.
- **Dispatching rules:** simple priority heuristics (SPT = shortest processing time, MWKR = most work remaining, MOR, FIFO) — the classic baselines an RL agent must beat.

### 3.2 Historical evolution
Exact methods (branch-and-bound, MILP, constraint programming) solve small instances optimally but do not scale. Metaheuristics (genetic algorithms, tabu search, simulated annealing, ACO) dominated the 1990s–2010s for larger instances. Priority dispatching rules gave fast, myopic online decisions. Since ~2018, **end-to-end DRL** reframed schedule construction as sequential decision-making: Pointer Networks (Vinyals 2015) and attention models (Kool 2019) for routing; then Zhang et al.'s L2D (NeurIPS 2020) brought GNN+PPO to JSSP; Park et al. (IJPR 2021) and Song et al. (IEEE TII 2023, FJSP) established the GNN+DRL paradigm now standard.

### 3.3 Core technical methods / model families
- **State encoders:** Graph Neural Networks (GIN, GAT, heterogeneous GNNs for FJSP) on the disjunctive graph; alternatively attention/Transformer encoders.
- **RL algorithms:** PPO (most common, stable), plus DQN, A2C/A3C; REINFORCE with baselines for construction.
- **Two design families:** (1) *end-to-end construction* — policy directly builds the schedule operation by operation; (2) *dispatching-rule selection* — agent picks among priority rules at each decision epoch (easier, natural for dynamic/online settings).
- **Reward shaping:** dense intermediate rewards (e.g., negative incremental makespan) vs. sparse terminal reward.

### 3.4 Evaluation metrics and reporting
- **Objective values:** makespan C_max; total/weighted tardiness; flow time; machine utilization; energy (for multi-objective).
- **Optimality/quality gap:** percent gap to best-known or to CP-SAT optimum: (obj − obj*) / obj*.
- **Generalization:** train on small instances, test on larger/unseen distributions; report gap degradation.
- **Efficiency:** inference/decision time; training wall-clock; sample efficiency.
- **Robustness (dynamic):** performance under injected breakdowns / stochastic times vs. deterministic baseline.
- **Statistical reporting:** multiple seeds, mean ± std, and non-parametric significance tests (Wilcoxon signed-rank pairwise; Friedman + Nemenyi across multiple algorithms) — the field standard for comparing scheduling algorithms.

### 3.5 Open problems and controversies
- Do learned policies truly generalize, or overfit to training distributions? (Central controversy.)
- Are DRL schedulers actually better than a well-tuned GA or CP-SAT given equal compute? (Baselining rigor is contested.)
- Reward shaping vs. sparse reward; credit assignment on long horizons.
- Sim-to-real transfer remains largely unaddressed.

### 3.6 Prioritized reading list (suggested order)
1. **Survey first** — a recent (2023–2024) survey of DRL for job-shop/flexible-job-shop scheduling in IEEE T-ASE / *Journal of Intelligent Manufacturing* / *Computers & Industrial Engineering* (for taxonomy and gap map).
2. **Zhang et al., L2D, NeurIPS 2020** (arXiv:2010.12367) — the seminal GNN+PPO end-to-end method; read with the `zcaicaros/L2D` code.
3. **Park et al., IJPR 2021** — GNN+RL formalization for JSSP.
4. **Song et al., IEEE TII 2023** — heterogeneous-GNN DRL for FJSP (your closest baseline).
5. **Kool et al., ICLR 2019** and **Vinyals et al., NeurIPS 2015** — attention/pointer foundations.
6. **Tassel et al., 2021** (JSSEnv) and **schlably** (SoftwareX 2023) — the environments you will build on.
7. **Reijnen et al., 2023** — unified job-shop benchmarking codebase.
8. **Textbooks:** Pinedo, *Scheduling: Theory, Algorithms, and Systems*; Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.).
9. **Courses/tutorials:** Hugging Face Deep RL Course; OpenAI Spinning Up; Stanford CS234.
10. **Libraries:** Stable-Baselines3 / CleanRL (PPO), PyTorch Geometric (GNNs), Gymnasium, Google OR-Tools (CP-SAT baseline), SimPy (DES layer).

### 3.7 Target journals and conferences
Journals: *Journal of Manufacturing Systems*; IEEE T-ASE; IEEE TII; *Computers & Industrial Engineering*; *International Journal of Production Research*; *Journal of Intelligent Manufacturing*; IISE Transactions. Conferences: IEEE CASE; Winter Simulation Conference (WSC); IFAC MIM; NeurIPS/ICML workshops on ML for combinatorial optimization.

## Part 4 — Research Question

**Primary RQ:** *Can a graph-neural-network deep-reinforcement-learning policy trained on small, deterministic flexible-job-shop instances generalize to larger instances and remain robust to stochastic disruptions (machine breakdowns, stochastic processing times, dynamic job arrivals), outperforming priority dispatching rules and matching or approaching metaheuristic/CP-SAT solution quality at a fraction of the decision-time cost?*

**Sub-questions & hypotheses:**
- **SQ1:** How does the size-generalization gap scale as test instances grow beyond training size? **H1:** the learned policy's optimality gap degrades gracefully (sub-linearly) and stays below tuned dispatching rules.
- **SQ2:** How much does explicit training on stochastic disruptions improve robustness vs. a deterministically trained policy? **H2:** disruption-aware training reduces makespan/tardiness degradation under injected disruptions by a statistically significant margin.
- **SQ3:** How does the DRL policy compare to OR-Tools CP-SAT and a GA on quality vs. decision time? **H3:** DRL is far faster at decision time with a modest, quantifiable quality gap that shrinks under tight time budgets and dynamic conditions.
- **SQ4 (optional):** Does a multi-objective reward (makespan + tardiness) yield Pareto-competitive schedules? **H4:** a scalarized/weighted reward produces non-dominated trade-offs versus single-objective baselines.

**Contribution/novelty:** A systematic, reproducible study of *generalization + disruption-robustness* for GNN-DRL FJSP with standardized baselining against dispatching rules, GA, and CP-SAT on public instances — directly addressing the three most-cited gaps, entirely with simulation-generated data.

**Scope/delimitations:** FJSP (not general resource-constrained project scheduling); objectives limited to makespan (+ optional tardiness); disruptions limited to the three named types; single-agent PPO; simulation-only (no sim-to-real deployment).

**Gap mapping:** RQ → generalization gap; SQ2 → dynamic/stochastic gap; SQ3 → baselining-rigor gap; SQ4 → multi-objective gap.

---

# DOCUMENT 2 — Methodological Development Plan

## M1. Research design and framing
Quantitative, experimental, computational research within a **Design Science Research (DSR)** frame: the "artifact" is the DRL scheduling policy; evaluation is controlled simulation experimentation. Positivist/empirical stance; hypotheses tested via controlled factorial experiments with statistical inference.

## M2. Data collection strategy (simulation)
- **Instances:** use public benchmark libraries — Taillard (ta01–ta80), Demirkol/DMU, Lawrence (la), Fisher–Thompson — for JSSP, and standard FJSP sets (Brandimarte MK01–MK10; Hurink edata/rdata/vdata) for flexibility. Generate additional synthetic instances with a controlled random generator (varying n jobs, m machines, processing-time distribution, flexibility ratio) for training and generalization tests.
- **Simulation layer:** wrap instances in a discrete-event environment — extend JSSEnv/schlably or build a SimPy-based Gymnasium env — to inject dynamic events: machine breakdowns (exponential time-to-failure + repair), stochastic processing times (log-normal multiplier), Poisson dynamic job arrivals.
- **Scenario design (experimental factors):** instance size (small/medium/large), flexibility ratio, disruption regime (none / breakdowns / stochastic times / arrivals / combined), objective (makespan / makespan+tardiness).
- **Volume:** train on thousands of episodes (PPO for millions of environment steps); evaluate on held-out benchmark instances with ≥30 stochastic replications each for variance estimation.
- **Simulation V&V:** *verification* — unit-test that generated schedules respect precedence and machine constraints; reproduce known optimal makespans on ft06/ft10 with CP-SAT. *Validation* — confirm dispatching-rule baselines match published values; confirm CP-SAT recovers best-known makespans on Taillard instances.
- **Justifying simulated data:** benchmark instances are the field's accepted ground truth; CP-SAT provides optimal/near-optimal references, so simulated results are directly comparable to prior published work.

## M3. Preprocessing pipeline
- **State featurization:** node features (operation processing time, remaining work, machine eligibility, ready time), normalized (min-max or z-score per feature); build disjunctive/heterogeneous graph per decision step.
- **Sequencing/windowing:** MDP episodes = full instance schedules; no manual windowing needed (unlike time-series).
- **Class/label handling:** RL has no labels; ensure reward scaling is stable (reward normalization / advantage normalization in PPO).
- **Splitting:** train on one distribution (e.g., ≤15×15), validate on a disjoint set, **test on strictly larger/unseen instances and on all benchmark libraries** — this out-of-distribution split is the core of the generalization claim.

## M4. Model development
- **Baselines:** priority dispatching rules (SPT, MWKR, MOR, FIFO, LWKR); a genetic algorithm; OR-Tools CP-SAT (with time budgets); and the reproduced L2D / Song-et-al. GNN-DRL as the learning baseline.
- **Proposed model:** heterogeneous-GNN (operation + machine nodes) encoder → PPO actor-critic; ablate encoder (GIN vs. GAT), reward (dense vs. sparse, single- vs. multi-objective), and disruption-aware vs. deterministic training.
- **Hyperparameter tuning:** grid/Bayesian search (Optuna) over learning rate, GNN layers/width, PPO clip, entropy coefficient, discount; select on validation set only.

## M5. Experimental design
- **Factors/levels:** as in M2; full or fractional factorial.
- **Replications:** ≥5 training seeds per configuration; ≥30 stochastic evaluation replications per instance.
- **Randomization:** seed control for env, torch, numpy; randomized instance order.
- **Power:** with 30 replications and typical effect sizes, Wilcoxon has ample power; report effect sizes (e.g., median difference, Vargha–Delaney A12).

## M6. Evaluation
- **Metrics:** makespan, optimality gap to CP-SAT, tardiness (if multi-objective), decision time, training time, generalization-gap curves.
- **Significance testing:** Wilcoxon signed-rank for pairwise DRL-vs-baseline; Friedman test + Nemenyi post-hoc across multiple algorithms (with critical-difference diagrams) — the field-standard protocol.
- **Cross-validation:** across instance libraries and seeds.
- **Sensitivity/robustness:** vary disruption intensity; stress-test on largest instances; ablation on reward and encoder.

## M7. Tooling and software stack
Python; PyTorch + PyTorch Geometric (GNNs); Gymnasium; Stable-Baselines3 or CleanRL (PPO); Google OR-Tools CP-SAT (baseline/oracle); SimPy (DES layer) and/or JSSEnv/schlably; Optuna (HPO); NumPy/pandas; Matplotlib/Seaborn; scipy.stats/scikit-posthocs (statistics); Weights & Biases or MLflow (experiment tracking); Git + DVC (versioning).

## M8. Computational resources
Single consumer GPU (8–12 GB, RTX-class) is sufficient; scheduling GNNs are lightweight. Full experiment suite runs in hours-to-days. Free-tier cloud (Colab/Kaggle) is a viable fallback. This is squarely student-feasible.

## M9. Reproducibility
Fixed seeds; pinned environment (conda/requirements + Docker); config-driven runs (Hydra); experiment tracking; public GitHub repo with instances, env, and scripts; archived on Zenodo with a DOI; report hardware and wall-clock.

## M10. Threats to validity
- **Internal:** hyperparameter unfairness → tune all learning baselines equally; seed variance → multiple seeds + CIs.
- **External:** overfitting to Taillard-style distributions → test on multiple libraries and synthetic distributions; simulation-only → state explicitly, no real-plant claim.
- **Construct:** makespan-only bias → include tardiness; reward-shaping artifacts → ablate.
- **Conclusion:** multiple-comparison inflation → Friedman/Nemenyi with correction; report effect sizes not just p-values.

## M11. Ethical considerations
Low risk (no human/personal data). Note broader implications: automation of scheduling decisions, labor impact, and the responsibility to report negative results honestly (avoid cherry-picking baselines).

## M12. Timeline (≈9 months)

| Phase | Months | Activities | Deliverables |
|---|---|---|---|
| 1. Foundations & lit review | 1–1.5 | Survey reading, gap map, reproduce L2D | Literature review chapter; running baseline |
| 2. Environment build & V&V | 1.5–3 | SimPy/Gym env, disruption injection, verification | Validated simulation env; unit tests |
| 3. Baselines | 3–4 | Dispatching rules, GA, CP-SAT harness | Baseline results table |
| 4. Model development | 4–6 | GNN-DRL, reward/encoder ablations, HPO | Trained models; ablation results |
| 5. Experiments & analysis | 6–8 | Generalization + disruption experiments, stats | Full results; significance analysis |
| 6. Writing & submission | 8–9 | Thesis writing, conference paper | Thesis draft; paper submission |

## M13. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| DRL fails to beat tuned GA/CP-SAT | Medium | Medium | Frame contribution as decision-time & robustness advantage; report honestly as a finding |
| Training instability | Medium | Medium | Use SB3/CleanRL defaults, reward normalization, more seeds |
| Env bugs (invalid schedules) | Medium | High | Rigorous verification vs. CP-SAT; constraint assertions |
| Scope creep (too many factors) | High | Medium | Freeze factor set early; multi-objective as optional |
| Compute overruns | Low | Medium | Cap instance sizes; cloud fallback |

## M14. Expected outputs & thesis structure
Chapters: (1) Introduction & RQ; (2) Literature review & gap; (3) Methodology (this plan); (4) Simulation environment & V&V; (5) Baselines; (6) Proposed GNN-DRL model; (7) Experiments & results; (8) Discussion & threats to validity; (9) Conclusions & future work. Target paper: a conference paper at IEEE CASE or WSC, extended to a journal (IEEE T-ASE / *Computers & Industrial Engineering*).

## Recommendations
1. **Commit to DRL for dynamic FJSP** as the thesis topic. Start in Phase 1 by reproducing Zhang et al.'s L2D on Taillard instances — **if you cannot reproduce a published baseline within ~3 weeks, that is your go/no-go checkpoint.**
2. **Lock scope early:** makespan + optional tardiness; three disruption types; single-agent PPO. Add multi-objective only if Phases 1–4 finish on schedule.
3. **Baseline rigor is your novelty moat:** always compare against tuned dispatching rules, a GA, and OR-Tools CP-SAT with matched time budgets, using Friedman/Nemenyi statistics. This alone distinguishes your thesis from the many DRL-scheduling papers that under-baseline.
4. **Fallback triggers:** if DRL cannot approach CP-SAT quality after Phase 4, pivot the contribution to *robustness-under-disruption and decision-time* (still publishable); if the whole DRL path stalls, switch to Runner-up #2 (synthetic-data defect detection on MVTec/NEU), which reuses much of the tooling.
5. **Publish the environment** as open source — a clean, disruption-capable FJSP Gym env is itself a citable contribution and strengthens the reproducibility story.

**Benchmarks/thresholds that would change the recommendation:** (i) if a fresh 2025–2026 Scopus/Semantic Scholar search shows the exact "generalization + dynamic FJSP + GNN-DRL" combination has been thoroughly published in the last 12 months, tighten novelty to a specific under-covered disruption type or multi-objective setting; (ii) if L2D reproduction fails by the Phase-1 checkpoint, switch to the dispatching-rule-selection design family (simpler, more robust) or to Runner-up #2/#3.

## Caveats
- **Tooling limitation:** live web search and fetch were unavailable, so exact 2023–2026 publication counts, growth percentages, current benchmark leaderboard numbers, DOIs/URLs, and verbatim quotes could not be verified. All paper identities (Zhang L2D NeurIPS 2020; Song FJSP-GNN IEEE TII 2023; Bergmann MVTec CVPR 2019; Saxena C-MAPSS PHM 2008), dataset contents, and tool identities are drawn from established domain knowledge and should be click-verified before citation.
- **Recency risk:** the 2025–2026 frontier (especially LLM-agent scheduling and diffusion-based synthetic data) is evolving fast; confirm the newest surveys before finalizing the gap claim.
- **Novelty check:** before committing, run a Scopus/Semantic Scholar search for "generalization" + "dynamic flexible job shop" + "deep reinforcement learning" (2024–2026) to ensure the specific combination remains open.
- **Baselining honesty:** DRL schedulers do not always beat CP-SAT/GA on raw quality; the defensible, honest contribution is the generalization/robustness/decision-time trade-off, not necessarily a new best makespan.