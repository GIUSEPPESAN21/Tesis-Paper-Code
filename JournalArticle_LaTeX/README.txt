Journal article - LaTeX source
================================
Same manuscript as ../JournalArticle_DRAFT.docx, typeset in LaTeX.

Files:
  main.tex          - manuscript source (article class, natbib/apalike author-year citations)
  references.bib    - BibTeX bibliography (42 entries)
  figures/*.pdf      - vector figures (H1-H5 results)
  JournalArticle.pdf - pre-compiled output (19 pages)

To recompile:
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex

Tested with TeX Live (pdflatex + bibtex), no errors or unresolved references/citations.
No specific target journal was specified, so this uses a generic single-column
Q1-journal layout (Letter, 1in margins, Times, natbib author-year). Swap in a
journal's own class file (e.g. elsarticle for Elsevier journals) if needed --
content and structure will carry over with minimal edits.
