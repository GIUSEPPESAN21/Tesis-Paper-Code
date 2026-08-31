> **Deliverable for:** Thesis framework document ("../docs/planificacion/Step 4 Thesis.md"), Part F (bibliography) and Part C (systematic review / gap table).
> **Date of verification:** 2026-07-27.
> **Method note (read before using this document):** No direct API/login access to Scopus, Web of Science, or IEEE Xplore's subscriber search interfaces was available in this environment. All verification below was done via web search, DOI resolution (doi.org), publisher pages (ScienceDirect, Springer, Taylor & Francis, INFORMS PubsOnline, IEEE Xplore's public pages), arXiv, Semantic Scholar, DBLP, and GitHub repositories. This is a strong proxy for bibliographic accuracy (DOIs, page numbers, author names are independently confirmed against publisher metadata), but it is **not equivalent to a native Scopus/WoS/IEEE Xplore export**, and the PRISMA hit counts in Task 2 are explicitly approximate, not official database counts. Anywhere this matters, it is flagged inline.

---

# TASK 1 — Bibliographic Verification

## 1.1 Change log (corrections vs. original Part F)

| Ref. | Field changed | Original (draft) | Corrected | Confidence |
|---|---|---|---|---|
| [3] | Edition/year | Pinedo, *Scheduling*, 5th ed., 2016 | A 6th ed. exists: Springer, 2022, DOI 10.1007/978-3-031-05921-6. Keep 5th ed. only if you are specifically citing 2016 content; otherwise cite 6th ed. | HIGH |
| [11] | URL | `http://people.brunel.ac.uk/~mastjjb/jeb/info.html` | `https://www.brunel.ac.uk/~mastjjb/jeb/info.html` (live-fetched; last updated Feb. 2018 per page). Add an accessed-date, since it's a web resource. | HIGH |
| [26] | Full author list | K. Lei, P. Guo, Y. Wang, X. Wu, W. Zhao | **Kun Lei, Peng Guo, Wenchao Zhao, Yi Wang, Linmao Qian, Xiangyin Meng, Liansheng Tang** (7 authors — original list both misordered names and included a non-existent author "X. Wu," while omitting Qian, Meng, and Tang) | HIGH |
| [29] | Full author list | R. Reijnen, K. van Straaten, Z. Bukhsh, Y. Zhang | **Robbert Reijnen, Igor G. Smit, Hongxiang Zhang, Yaoxin Wu, Zaharah Bukhsh, Yingqian Zhang** (6 authors — "K. van Straaten" does not appear on the actual author list) | HIGH |
| [30] | Edition/year | Law, *Simulation Modeling and Analysis*, 5th ed., 2015 | A 6th ed. exists: McGraw Hill, Feb. 2024, ISBN 978-1264268245. Cite 6th ed. if writing after 2024. | HIGH |
| [31] | Full citation detail | "Proc. Winter Simulation Conference, 2011" (no editors/pages/DOI) | R. G. Sargent, "Verification and Validation of Simulation Models," in *Proc. 2011 Winter Simulation Conf.*, S. Jain, R. R. Creasey, J. Himmelspach, K. P. White, M. Fu, Eds., IEEE, 2011, pp. 183–198, doi:10.1109/WSC.2011.6147750 | HIGH |
| [33] | Citation form | "OR-Tools (CP-SAT), Google, https://developers.google.com/optimization" | Update to current version per Google's own citation guidance ("OR-Tools v9.x. L. Perron and V. Furnon. https://developers.google.com/optimization/") **and/or** cite the peer-reviewed solver paper with a real DOI: L. Perron, F. Didier, S. Gay, "The CP-SAT-LP Solver," in *Proc. 29th Int. Conf. on Principles and Practice of Constraint Programming (CP 2023)*, LIPIcs vol. 280, art. 3, 2023, doi:10.4230/LIPIcs.CP.2023.3 | HIGH |

**Two corrections already flagged in the draft were independently re-verified and confirmed accurate — no further change needed:**
- [23] Author order for L2D is **Zhang, Song, Cao**, not "Zhang, Cao, Song" — confirmed directly against the NeurIPS 2020 proceedings page and arXiv.
- [28] *schlably*'s affiliation is the **TMDT group at Bergische Universität Wuppertal (BUW)**, not Fraunhofer IML / TU Dortmund — confirmed via the institute's own site and the `tmdt-buw` GitHub organization.

**All other entries ([1], [2], [4]–[10], [12]–[22], [24], [25], [27], [32], [34]) were verified as drafted with no corrections required** (DOIs, volumes, issues, pages, and author names all matched publisher/DOI metadata exactly). Two minor residual notes, not treated as errors:
- [4] The claim that Carlier & Pinson (1989) first solved ft10 is corroborated by secondary literature but not independently checked against the primary 1989 text itself — treat as MEDIUM confidence specifically on that historical claim (the bibliographic data itself is HIGH confidence).
- [34] Place of publication is cited inconsistently across secondary sources as "Paris" vs. "Montrouge" (SEMA's offices were in the Paris region, at Montrouge) — no primary copy of the 1964 note was locatable to confirm the imprint line. Not flagged as an error.

## 1.2 Corrected reference list (IEEE format)

**Fundamentos de scheduling y complejidad**

[1] R. L. Graham, E. L. Lawler, J. K. Lenstra, A. H. G. Rinnooy Kan, "Optimization and approximation in deterministic sequencing and scheduling: a survey," *Annals of Discrete Mathematics*, vol. 5, pp. 287–326, 1979, doi:10.1016/S0167-5060(08)70356-X.

[2] M. R. Garey, D. S. Johnson, R. Sethi, "The complexity of flowshop and jobshop scheduling," *Mathematics of Operations Research*, vol. 1, no. 2, pp. 117–129, 1976, doi:10.1287/moor.1.2.117.

[3] M. L. Pinedo, *Scheduling: Theory, Algorithms, and Systems*, 6th ed., Springer, 2022, doi:10.1007/978-3-031-05921-6. *(Corrected from 5th ed./2016 — cite 5th ed. only if intentionally referencing that edition's content.)*

[4] J. Carlier, É. Pinson, "An Algorithm for Solving the Job-Shop Problem," *Management Science*, vol. 35, no. 2, pp. 164–176, 1989, doi:10.1287/mnsc.35.2.164.

**Benchmarks**

[5] H. Fisher, G. L. Thompson, "Probabilistic learning combinations of local job-shop scheduling rules," in *Industrial Scheduling*, J. F. Muth and G. L. Thompson, Eds., Englewood Cliffs, NJ: Prentice-Hall, 1963, pp. 225–251.

[6] S. Lawrence, "Resource constrained project scheduling: an experimental investigation of heuristic scheduling techniques," Tech. Rep., Graduate School of Industrial Administration, Carnegie-Mellon Univ., 1984.

[7] E. Taillard, "Benchmarks for basic scheduling problems," *European Journal of Operational Research*, vol. 64, no. 2, pp. 278–285, 1993, doi:10.1016/0377-2217(93)90182-M.

[8] E. Demirkol, S. Mehta, R. Uzsoy, "Benchmarks for shop scheduling problems," *European Journal of Operational Research*, vol. 109, no. 1, pp. 137–141, 1998, doi:10.1016/S0377-2217(97)00019-2.

[9] P. Brandimarte, "Routing and scheduling in a flexible job shop by tabu search," *Annals of Operations Research*, vol. 41, no. 3, pp. 157–183, 1993, doi:10.1007/BF02023073.

[10] J. Hurink, B. Jurisch, M. Thole, "Tabu search for the job-shop scheduling problem with multi-purpose machines," *OR Spektrum*, vol. 15, no. 4, pp. 205–215, 1994, doi:10.1007/BF01719451.

[11] J. E. Beasley, "OR-Library," Brunel Univ. [Online]. Available: https://www.brunel.ac.uk/~mastjjb/jeb/info.html. *(Corrected URL; accessed 27 Jul. 2026.)*

**RL y GNN (fundamentos)**

[12] R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., Cambridge, MA: MIT Press, 2018.

[13] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.

[14] J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," arXiv:1506.02438, 2015/2016 (ICLR 2016 poster).

[15] V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, pp. 529–533, 2015, doi:10.1038/nature14236.

[16] T. N. Kipf, M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," in *Proc. ICLR*, 2017, arXiv:1609.02907.

[17] W. L. Hamilton, R. Ying, J. Leskovec, "Inductive Representation Learning on Large Graphs," in *Proc. NIPS (NeurIPS)*, 2017, pp. 1024–1034, arXiv:1706.02216.

[18] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio, "Graph Attention Networks," in *Proc. ICLR*, 2018, arXiv:1710.10903.

[19] K. Xu, W. Hu, J. Leskovec, S. Jegelka, "How Powerful are Graph Neural Networks?," in *Proc. ICLR*, 2019, arXiv:1810.00826.

**NCO / DRL para scheduling**

[20] O. Vinyals, M. Fortunato, N. Jaitly, "Pointer Networks," in *Proc. NeurIPS*, 2015, pp. 2692–2700, arXiv:1506.03134.

[21] W. Kool, H. van Hoof, M. Welling, "Attention, Learn to Solve Routing Problems!," in *Proc. ICLR*, 2019, arXiv:1803.08475; repo: github.com/wouterkool/attention-learn-to-route.

[22] Y.-D. Kwon, J. Choo, B. Kim, I. Yoon, Y. Gwon, S. Min, "POMO: Policy Optimization with Multiple Optima for Reinforcement Learning," in *Proc. NeurIPS*, 2020, arXiv:2010.16011.

[23] C. Zhang, W. Song, Z. Cao, J. Zhang, P. S. Tan, C. Xu, "Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning," in *Proc. NeurIPS*, 2020, arXiv:2010.12367; repo: github.com/zcaicaros/L2D.

[24] J. Park, J. Chun, S. H. Kim, Y. Kim, J. Park, "Learning to schedule job-shop problems: representation and policy learning using graph neural network and reinforcement learning," *International Journal of Production Research*, vol. 59, no. 11, pp. 3360–3377, 2021, doi:10.1080/00207543.2020.1870013.

[25] W. Song, X. Chen, Q. Li, Z. Cao, "Flexible Job-Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning," *IEEE Transactions on Industrial Informatics*, vol. 19, no. 2, pp. 1600–1610, 2023, doi:10.1109/TII.2022.3189725.

[26] K. Lei, P. Guo, W. Zhao, Y. Wang, L. Qian, X. Meng, L. Tang, "A multi-action deep reinforcement learning framework for flexible Job-shop scheduling problem," *Expert Systems with Applications*, vol. 205, art. 117796, 2022, doi:10.1016/j.eswa.2022.117796. *(Corrected author list.)*

**Entornos y benchmarks de código**

[27] P. Tassel, M. Gebser, K. Schekotihin, "A Reinforcement Learning Environment For Job-Shop Scheduling," arXiv:2104.03760, 2021; repo: github.com/prosysscience/JSSEnv.

[28] C. Waubert de Puiseau, J. Peters, C. Dörpelkus, H. Tercan, T. Meisen, "schlably: A Python framework for deep reinforcement learning based scheduling experiments," *SoftwareX*, vol. 22, art. 101383, 2023, doi:10.1016/j.softx.2023.101383; repo: github.com/tmdt-buw/schlably.

[29] R. Reijnen, I. G. Smit, H. Zhang, Y. Wu, Z. Bukhsh, Y. Zhang, "Job Shop Scheduling Benchmark: Environments and Instances for Learning and Non-learning Methods," arXiv:2308.12794, 2023; repo: github.com/ai-for-decision-making-tue/Job_Shop_Scheduling_Benchmark_Environments_and_Instances. *(Corrected author list.)*

**Simulación y estadística**

[30] A. M. Law, *Simulation Modeling and Analysis*, 6th ed., McGraw Hill, 2024. *(Corrected from 5th ed./2015; cite 5th ed. only if intentionally referencing 2015 content.)*

[31] R. G. Sargent, "Verification and Validation of Simulation Models," in *Proc. 2011 Winter Simulation Conf.*, S. Jain, R. R. Creasey, J. Himmelspach, K. P. White, M. Fu, Eds., IEEE, 2011, pp. 183–198, doi:10.1109/WSC.2011.6147750. *(Corrected: added editors, pages, DOI.)*

[32] J. Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets," *Journal of Machine Learning Research*, vol. 7, pp. 1–30, 2006.

**Herramientas**

[33] L. Perron, V. Furnon, "OR-Tools," Google, v9.x, 2025. [Online]. Available: https://developers.google.com/optimization/. For the underlying CP-SAT-LP solver as a peer-reviewed reference: L. Perron, F. Didier, S. Gay, "The CP-SAT-LP Solver," in *Proc. 29th Int. Conf. on Principles and Practice of Constraint Programming (CP 2023)*, LIPIcs, vol. 280, art. 3, 2023, doi:10.4230/LIPIcs.CP.2023.3.

**Fundamento del grafo disyuntivo**

[34] B. Roy, B. Sussmann, "Les problèmes d'ordonnancement avec contraintes disjonctives," Note DS n.º 9 bis, SEMA, Montrouge, 1964.

---

# TASK 2 — PRISMA Systematic Search (2018–2026)

**Access caveat (read first):** This environment has no subscriber login for Scopus, Web of Science, or IEEE Xplore's native search interface, so the literal search strings could not be executed against those three databases as a real PRISMA protocol requires. The counts below were built from open proxies — arXiv, Semantic Scholar-style web search, Google-indexed publisher pages, and (where renderable) IEEE Xplore's public result pages — and are approximate, not official per-database exports. **Before the thesis defense, re-run the four literal strings directly in Scopus/WoS/IEEE Xplore** (campus library access or UNIMINUTO's subscription) to get authoritative hit counts; treat the numbers below as a directional pre-screen, not the final PRISMA count for the thesis document.

## 2.1 Approximate hit counts by search string (proxy sources, 2018–2026)

| Search string | Records surfaced via proxy search | Estimated true universe (Scopus/WoS/IEEE scale) |
|---|---|---|
| 1 — RL/DRL/GNN + job shop/FJSP + scheduling (core) | ~30–40 distinct titles surfaced | Likely 150–250+ |
| 2 — RL + job shop + dynamic/stochastic/breakdown/disruption/rescheduling | ~25–30 distinct titles surfaced | Likely 80–120+ |
| 3 — NCO/GNN + generalization + scheduling/combinatorial optimization | ~15–20 distinct titles surfaced | Likely 40–60 (scheduling-specific) |
| 4 — LLM agent + scheduling/job shop/combinatorial optimization | ~15–18 distinct titles surfaced | Likely 20–35 (genuinely small, emerging niche) |

## 2.2 Approximate PRISMA flow

```
Identified (proxy search, all 4 strings, 2018–2026)  ≈ 85–108 records surfaced
        │
        ▼
After de-duplication across strings                   ≈ 55 unique records
        │
        ▼
Screened by title/abstract (on-topic, in date range)   ≈ 40 passed
        │
        ▼
Included (met inclusion criteria: peer-reviewed or
arXiv w/ code or citation impact; RL/GNN or LLM-agent
applied to JSSP/FJSP; quantitative benchmarks reported) ≈ 28–30 records
        │
        ▼
Shortlisted as highest-relevance 2024–2026 additions    14 records (below)
```

Excluded at screening: non-manufacturing scheduling (cloud/grid job scheduling), pure metaheuristic papers with no RL/GNN/LLM component, papers without quantitative benchmark comparison, duplicate conference/journal versions of the same arXiv preprint.

**Note:** these numbers are weighted toward 2024–2026 recall (per the specific goal of finding what you had not yet identified) and likely undercount older (2018–2023) String 1 records that are already well represented in your existing Part C narrative and Part F bibliography.

## 2.3 Shortlist — most relevant 2024–2026 papers not yet in your reference list

Confidence flag: items marked **(unverified author list)** were found via search snippets only; their author names were not independently confirmed against a full-text or publisher metadata page and should be checked directly before citing.

**String 1 — core GNN + DRL for FJSP**

[35] I. G. Smit, J. Zhou, R. Reijnen, Y. Wu, J. Chen, C. Zhang, Z. Bukhsh, Y. Zhang, W. Nuijten, "Graph Neural Networks for Job Shop Scheduling Problems: A Survey," *Computers & Operations Research*, 2024, arXiv:2406.14096. — The essential up-to-date survey for positioning your Part B8/C2 narrative; not a competing novelty claim.

[36] G. Liu, M. Liao, W. Chen, Z. Zhang, H. Wan, Y. Lin, "Multi-Faceted Dynamic Flexible Job Shop Scheduling via Heterogeneous Graph Transformer and Deep Reinforcement Learning," *Expert Systems with Applications*, vol. 303, art. 130532, 2025/2026 (in press). — Heterogeneous graph transformer + DRL for multiple simultaneous dynamic factors. **Overlap risk: moderate-high** — read in full; does not explicitly test size-generalization or use disruption-aware reward shaping, but is close on the "multiple dynamic factors" axis.

[37] "Schedule repair for flexible job shops under machine breakdowns by deep reinforcement learning" **(unverified author list)**, ScienceDirect record S0360835225004024, 2025. — Explicitly frames the generalization + disruption-robustness gap; schedule-repair (post-hoc rescheduling) rather than an end-to-end size-generalizing policy. **Overlap risk: moderate.**

**String 2 — dynamic/stochastic/disruption**

[38] S. Lassoued, S. Lier, A. Schwung, "Policy-Based Reinforcement Learning with Action Masking for Dynamic Job Shop Scheduling under Uncertainty: Handling Random Arrivals and Machine Failures," arXiv:2601.09293, 2026. — Combines two disruption types (arrivals + failures) via maskable PPO over a Petri-net state; no GNN, no explicit cross-size generalization study. **Overlap risk: moderate-high on the multi-disruption-type axis** — read before finalizing your novelty claim.

[39] R. Chen, Y. Mei, F. Zhang, M. Zhang, "DyRo-MCTS: A Robust Monte Carlo Tree Search Approach to Dynamic Job Shop Scheduling," arXiv:2509.21902, 2025. — Robustness-aware MCTS over offline-learned policies; adjacent framing, planning-based not GNN+DRL end-to-end. **Overlap risk: low-moderate.**

[40] "Dynamic flexible job-shop scheduling by multi-agent reinforcement learning with reward-shaping" **(unverified author list)**, *Advanced Engineering Informatics*, 2024, doi:10.1016/j.aei.2024.102872. — Reward-shaping multi-agent DRL for a single disruption type (random job arrivals). **Overlap risk: high on the reward-shaping axis specifically** — closest existing match to your disruption-aware reward mechanism; must be read and explicitly differentiated (single disruption type vs. your multi-disruption-type design).

[41] Related contrastive reward-shaping paper (DCMAD3QN framework) **(unverified author list)**, *Expert Systems with Applications*, 2025, ScienceDirect record S0957417425025680. — Same reward-shaping niche as [40]; single disruption type. **Overlap risk: high**, same reasoning as [40].

[42] I. G. Smit, Y. Wu, P. Troubil, Y. Zhang, W. P. M. Nuijten, "Neural Combinatorial Optimization for Stochastic Flexible Job Shop Scheduling Problems," in *Proc. AAAI*, 2025, arXiv:2412.14052. — Attention-based scenario processing for stochastic processing times; explicitly claims generalizability across scenario counts/distributions. **Overlap risk: high — arguably the single closest existing paper to your core novelty claim.** Its "stochastic" element is processing-time/scenario variability rather than discrete disruption events (breakdowns, urgent arrivals) with reward shaping — this is exactly the differentiator you should state explicitly in your literature review.

**String 3 — generalization / size-generalization**

[43] "Learning to Optimize Job Shop Scheduling Under Structural Uncertainty" (UP-AAC method) **(unverified author list)**, arXiv:2601.21389, 2026. — Addresses stochastic routing-path uncertainty, a distinct uncertainty type from machine/arrival disruptions. **Overlap risk: low-moderate.**

[44] "RobustGANTT" (exact arXiv ID not independently confirmed — verify directly before citing) — GNN scheduler trained on small networks generalizing to networks ~100× larger. **Overlap risk: low** (not JSSP/FJSP-specific) but a useful methodological precedent for your size-generalization argument.

**String 4 — LLM agents for scheduling (emerging niche, almost entirely 2025–2026 preprints)**

[45] H. Abgaryan, T. Cazenave, A. Harutyunyan, "Starjob: Dataset for LLM-Driven Job Shop Scheduling," arXiv:2503.01877, 2025. — 130k-instance supervised dataset for fine-tuning LLMs as direct schedulers; reports gains over PDRs and over L2D. **Overlap risk: low** (different paradigm — supervised LLM fine-tuning, not RL) but essential for your Part C2 "frontera 2024–2026" LLM discussion.

[46] S. Cao, Y. Yuan, "ReflecSched: Solving Dynamic Flexible Job-Shop Scheduling via LLM-Powered Hierarchical Reflection," arXiv:2508.01724, 2025 (v3, 2026). — LLM agent with distilled "strategic experience" for non-myopic dynamic FJSP decisions; benchmarked on "GEN-Bench." **Overlap risk: low-moderate** — the clearest String 2 + String 4 crossover found; cite both as an LLM-agent example and as a dynamic-FJSP comparator.

[47] J. Zhang, C. Luo, Z. Su, Q. Zhang, et al. **(unverified full author list)**, "NS4S: Neighborhood Search for Scheduling Problems via Large Language Models," in *Proc. IJCAI*, 2025. — LLM-guided neighborhood search operator design. **Overlap risk: low.**

[48] "Large Language Model-Assisted Deep Reinforcement Learning from Human Feedback for Job Shop Scheduling" **(unverified author list)**, *Machines* (MDPI), vol. 13, no. 5, art. 361, 2025. — LLM as few-shot reward-function designer for a downstream DRL scheduler. **Overlap risk: moderate on the reward-shaping axis** — an LLM-assisted (not disruption-aware) reward-design approach; cite to differentiate from your mechanism.

Also surfaced but more tangential, useful as tracking resources rather than direct citations: "DynaSchedBench: Calibrated Dynamic Scheduling Benchmarks and Observability Paradox in LLM-based Scheduling Agents" (arXiv:2605.27566, 2026, author list unresolved from snippets), and the curated GitHub list `github.com/ai4co/awesome-fm4co` for tracking this fast-moving niche.

### Bottom line on novelty

No single paper found in this pre-screen combines all three elements of your thesis's claim — GNN-based **size generalization** + robustness to **multiple distinct disruption types** + an explicit **disruption-aware reward-shaping mechanism** — in one FJSP framework. The three papers you should read in full before finalizing your novelty statement, because a reviewer is most likely to raise them, are **[38]** (multi-disruption-type, no GNN/no size-generalization test), **[40]/[41]** (reward-shaping, single disruption type), and **[42]** (size-generalization + stochasticity, but scenario-based rather than discrete-event disruptions). This is consistent with — and sharpens — the draft's own Recommendation #2 ("if an identical paper appears, reorient toward the differential").

---

# TASK 3 — Gap-Analysis Table Update

| Gap identified (original) | Evidence (original) | Research question (original) | Update after 2024–2026 search | Status |
|---|---|---|---|---|
| Little joint evaluation of size-generalization **and** robustness under disruptions | Most work evaluates static, or one disruption type at a time | **PP:** Can a GNN+PPO policy generalize in size **and** be robust to breakdowns/stochastic times/simultaneous arrivals? | [38] covers two disruption types (arrivals + failures) but no GNN/no size-generalization test; [42] covers size-generalization + stochastic scenarios but not discrete disruption events. No paper combines both axes. | **Gap holds — still open.** Recommend explicitly citing [38] and [42] in Ch. 3 to preempt the "why not already done" question. |
| Disruption-aware rewards underexplored | Static makespan-bound reward dominates | **SP1:** Does a disruption-aware reward improve robustness without harming nominal optimality? | [40] and [41] (2024–2025) both do reward-shaping for disruption, but each targets a **single** disruption type (random arrivals); your proposed κ-weighted multi-disruption reward (D3 in Part D) is not superseded but its differentiation must now be made explicit against these two papers. | **Gap narrows, does not close.** Recommend adding one paragraph in Ch. 3/Ch. 4 explicitly contrasting your reward formulation against [40]/[41] on "single vs. multi disruption-type." |
| Limited simultaneous comparison against CP-SAT + GA + PDRs | Comparisons partial | **SP2:** What is the real gap vs. the CP-SAT ceiling and vs. GA/PDRs in each regime? | No 2024–2026 paper found that changes this picture; still a genuine methodological gap in the literature broadly, and your D4/D7 pre-registered design already addresses it directly. | **Gap holds — unaffected by new search.** |
| Weak reproducibility / statistical protocol | Few papers use Friedman/Nemenyi/CD | **SP3:** Are differences statistically significant, with effect sizes reported? | No 2024–2026 paper found that changes this picture. | **Gap holds — unaffected by new search.** |
| *(New, emerging)* LLM agents for scheduling | Almost entirely 2025–2026 preprints ([45]–[48]) | *(Not previously a research question in your framework)* | This is a genuinely new, fast-moving sub-area, but none of [45]–[48] combine LLM agents with your specific novelty claim (GNN + size-generalization + multi-disruption reward). It is adjacent, not competing. | **New territory noted, no action required** unless you want to add a short "LLM agents as an alternative paradigm" paragraph to your discussion (Ch. 7) for completeness. |

### Overall verdict on the novelty claim

Your thesis's core differential — **joint** size-generalization and multi-disruption-type robustness, achieved via a disruption-aware reward mechanism on a GNN+PPO policy — still occupies open territory as of this pre-screen. The three closest papers ([38], [40]/[41], [42]) each cover one axis of your claim but not the combination. This matches the draft document's own Recommendation #2, and this search provides the evidence to act on it: **narrow and sharpen** your novelty statement in Chapter 1/3 to explicitly name the single-axis-vs-joint distinction, rather than claiming the combination is entirely unaddressed in the literature.

---

# Outstanding actions before the thesis defense

1. Re-run the four literal PRISMA strings directly in Scopus, Web of Science, and IEEE Xplore (via UNIMINUTO library access) to replace the approximate counts in Task 2 with authoritative per-database numbers, and complete a full-text eligibility pass (this pre-screen only went to title/abstract screening).
2. Read [38], [40], [41], and [42] in full and add the explicit differentiation paragraph described in the gap table.
3. Verify the author lists flagged **(unverified author list)** for [37], [40], [41], [43], and any DynaSchedBench citation, directly against publisher/arXiv pages before adding them to your formal reference list.
4. Decide whether to update [3] and [30] to their newer editions (6th ed. each) or keep the 5th editions intentionally, and update [11] and [33] as shown.
