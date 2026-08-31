# Roadmap — work already identified as pending by the thesis and the paper

This is not a wishlist. Every item below is quoted or closely paraphrased from a
specific section of `thesis/Thesis.md` (Chapter 8.6, "Trabajo futuro") or
`paper/PaperCIE_LaTeX/manuscript/en/main.tex` (§7, "Threats to validity") —
work the authors themselves already flagged as the natural next step, not a
speculative addition. Each item cites its source so the claim can be checked
against the primary text.

## From the paper's own "Threats to validity" (`paper/PaperCIE_LaTeX`, §Threats to validity)

1. **Full offline-method arm (CP-SAT/GA) across the entire disruption grid.**
   Currently CP-SAT and the GA "enter only through the descriptive comparison
   ... at one instance, one regime and five seeds." The paper states directly:
   *"A full offline-method arm across the grid is the natural next experiment
   and would sharpen [the exact-optimality result] from an observation into a
   test."*

2. **Multi-seed training campaign for the GNN-PPO policy.** *"Multi-seed
   training infrastructure is implemented and reported in the accompanying
   repository, but the multi-seed campaign had not completed at the time of
   writing and no multi-seed number is claimed here."* The learned policy's
   position in the results table currently carries no estimate of training
   variance.

3. **Expand instance coverage in the disruption grid beyond `ft06`/`la01`.**
   *"The nine regimes are executed on two instances, ft06 and la01 ... the
   method-level findings ... should be read as holding for these two
   instances and this disruption model rather than as a general ranking."*

4. **Validate the realized-quality reference point under dynamic arrivals.**
   *"RQ under dynamic arrivals is referenced to the optimum of the nominal
   instance rather than of the realised one. Comparisons within a regime
   remain valid; absolute magnitudes there are not optimality gaps."* A
   dedicated study measuring realized quality against the optimum of the
   instance actually executed (not the nominal one) would close this gap.
   Related, narrower point already pre-registered but never run (see thesis
   item 7 below): the expected-tardiness metric for the arrivals regime.

(Also noted in the same section, lower priority: only 5 of 10 methods are
compared under the arrivals regime — `paper/PaperCIE_LaTeX` argues this is
anti-conservative for the learned policy specifically, not neutral.)

## From the thesis's own Chapter 8.6 ("Trabajo futuro"), by horizon

### Short term — before defense

1. Reproduce L2D (and Song et al., if source access resolves) inside this
   thesis's own pipeline, on the same instances/protocol as H1 — converts
   Table 7.1 from a comparison against published figures into a controlled
   one.
2. Re-measure `la40` in isolation (Chapter 6, §6.5.1) — resolves the one
   pending data ambiguity in H5.
3. Obtain the full text of Song et al. and Lei et al. to replace the
   "unconfirmed" cells in Table 7.1 with verified figures.

### Medium term — direct methodological extensions

4. Architectural ablation per `docs/planificacion/AblationPlan.md` (Axes
   1-2): homogeneous/L2D vs. relational (this thesis) vs. heterogeneous
   (Song et al./Liu et al.) — the only identified way to separate whether
   H1's rejection traces to the joint generalization+robustness objective or
   to the specific relational architecture choice.
5. Test the domain-randomization correction from Chapter 7 (§7.4.3):
   retrain κ=0.5/κ=1.0 mixing in a non-zero fraction of undisturbed episodes,
   re-evaluate H4 under the same pre-registered decision rule.
6. Repeat H4 training with multiple seeds (3-5 minimum) to report κ results
   as intervals, not point estimates — corrects the most-cited internal-validity
   threat in Chapter 7 (§7.7.1). *(Same underlying gap as paper item 2 above.)*
7. Compute the expected-tardiness metric E[ΣT_j] for the arrivals regime
   (E6) — already specified in `thesis/PreRegistration.md` §4 but never
   executed; gives a second, C_max/RI-independent reading of performance
   under dynamic arrivals.
8. Extend size generalization to other instance families (Taillard, Hurink),
   not just within the Lawrence family — tests whether H2 holds as
   generalization *across distributions*, not only *in size within one
   distribution*.

### Long term — motivated by the central theoretical finding

9. Design and evaluate a hybrid bridge-reactive/re-optimizer architecture —
   motivated by the contrast between this policy's quality weakness (H1, H3)
   and speed strength (H5). Requires a substitution criterion (when the exact
   re-optimizer replaces the bridge decision) and evaluation of the combined
   system, not each component separately.
10. Investigate whether a dedicated scenario-processing module (SPM-DAN-style,
    per Smit et al.) improves generalization under uncertainty for this
    relational architecture — Chapter 7 §7.3 documents SPM-DAN achieving a
    substantially better generalization gap under a single uncertainty type
    than this thesis's architecture achieves under its multi-perturbation
    regime.
11. Partial sim-to-real validation — even limited (e.g. contrasting this
    thesis's three disruption generators against real time-to-failure/
    time-to-repair records from a collaborating shop) — first step toward
    closing the thesis's most fundamental external-validity limitation: no
    conclusion here has been checked against a real industrial disruption
    process.

## Overlaps between the two sources

Paper item 2 and thesis item 6 are the same underlying gap (multi-seed
training for the perturbation-aware policy) stated in two documents written
for two audiences — not double work. Paper item 3/thesis item 8 (instance
breadth) similarly overlap but at different scopes: the paper's point is
about the *disruption* grid (ft06/la01 only), the thesis's is about the
*static generalization* grid (Lawrence family only) — both are real, both
listed above under their own numbering to preserve the source's own framing.
