# Paper package — *Robustness indices reward poor nominal schedules*

Target venue: **Computers & Industrial Engineering** (Elsevier, Q1). Manuscript
prepared in the Elsevier CAS single-column format (`cas-sc`), the format C&IE
asks for at submission.

## What is here

```
manuscript/en/main.pdf     English manuscript (17 pp, submission-ready)
manuscript/es/main.pdf     Spanish manuscript (18 pp)
manuscript/{en,es}/        full LaTeX sources: main.tex, refs.bib, tables, figures
analysis/                  the re-analysis that produced every number
  analyze.py               metric panel, RI artifact, rank reversal, Friedman/Nemenyi
  analyze2.py              corrected effect sizes, per-method tables, headline numbers
  analyze3.py              Proposition-1 audit, offline-vs-online, clique membership
  tables.py                emits analysis/tex/T1..T5 (+_es) and macros.tex
  figures.py               emits figures/ and figures_es/ (PDF + 600-dpi PNG)
  out/                     every intermediate CSV/JSON, all script-generated
figures/, figures_es/      the six publication figures, per language
results/datos_crudos/      the raw per-run records the whole paper rests on
run_all.sh                 regenerates everything end to end
```

## Reproducing

```bash
./run_all.sh
```

Recomputes the analysis from the raw CSVs, regenerates the tables, the prose
macros and the figures in both languages, and compiles both PDFs. Requires
Python (pandas, numpy, scipy, matplotlib) and a TeX Live installation with
`texlive-lang-spanish` for the Spanish build.

## Why no number in the prose is typed by hand

Every numeral the manuscripts state is either inside a `\input`-ed table
(`analysis/tex/T*.tex`) or expanded from a macro (`analysis/tex/macros.tex`).
Both files are written by `analysis/tables.py` from the computed outputs, which
in turn come from the raw per-run CSVs. Editing a number by hand is therefore
impossible without the next `run_all.sh` overwriting it — which is the point.

## What the paper claims, and on what evidence

| Claim | Evidence | Strength |
|---|---|---|
| RI is strictly decreasing in nominal deviation at fixed realised performance | Proposition 1 — an identity, verified numerically to 9.4e-6 over 150 cells | analytic |
| RI is negatively rank-correlated with nominal deviation | 9/9 regimes; ρ −0.985 to −0.948, p ≤ 3e-5 in the six ten-method regimes | strong |
| RI orders methods opposite to realised quality | Kendall τ negative in 9/9; the two criteria never agree | strong |
| Following RI costs 15.8–49.3 pp of deviation | direct computation from the same runs | strong |
| GNN-PPO ties the best rule in every regime, first in five | Friedman + Nemenyi, 60 blocks/regime | moderate — one training seed, two instances |
| Nominal optimality is largely dissipated by right-shift execution | CP-SAT/GA vs the field on ft06, A=0.90 | descriptive only — n=4 |

The limits are stated in §7.4 of the paper, not hidden: two instances in the
disruption grid, one training checkpoint, four rules rather than nine under
arrivals, and CP-SAT/GA present only descriptively.

## Before submitting

1. **Verify the references.** `refs.bib` is a working bibliography; confirm every
   DOI and page range against the publisher record.
2. **Add your ORCID** to `\author[1]{...}[orcid=...]` in `manuscript/en/main.tex`.
3. **Decide the Spanish version's role.** C&IE publishes in English; the Spanish
   manuscript is for the thesis and for local dissemination.
4. **Point the data-availability statement at a real URL** (Zenodo DOI or the
   public GitHub repository) once the artefact is deposited.
