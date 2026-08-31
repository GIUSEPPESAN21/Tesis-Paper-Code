#!/usr/bin/env bash
# Regenerates every number, table, figure and PDF in this paper from the raw
# per-run CSVs in results/datos_crudos/. Nothing in the manuscript is entered
# by hand: prose numerals come from analysis/tex/macros.tex, tables from
# analysis/tex/T*.tex, figures from figures/ and figures_es/.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/5] statistical re-analysis"
python3 analysis/analyze.py  > analysis/out/log_analyze.txt
python3 analysis/analyze2.py > analysis/out/log_analyze2.txt
python3 analysis/analyze3.py > analysis/out/log_analyze3.txt

echo "[2/5] LaTeX tables and prose macros"
python3 analysis/tables.py > /dev/null

echo "[3/5] figures (English and Spanish)"
python3 analysis/figures.py > /dev/null
FIGLANG=es python3 analysis/figures.py > /dev/null

echo "[4/5] syncing generated assets into the manuscript trees"
cp analysis/tex/macros.tex          manuscript/en/
cp analysis/tex/macros.tex          manuscript/es/
cp analysis/tex/T{1,2,3,4,5}.tex    manuscript/en/
cp analysis/tex/T{1,2,3,4,5}_es.tex manuscript/es/
cp figures/*.pdf                    manuscript/en/figures/
cp figures_es/*.pdf                 manuscript/es/figures/

echo "[5/5] compiling both manuscripts"
for d in en es; do
  ( cd "manuscript/$d"
    pdflatex -interaction=nonstopmode main.tex > /dev/null
    bibtex main > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode main.tex > /dev/null
    pdflatex -interaction=nonstopmode main.tex > /dev/null
    echo "  manuscript/$d/main.pdf  ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pp)" )
done
echo "done."
