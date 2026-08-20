"""Export the already-validated stats-pipeline tables as thesis-numbered CSV/MD/TeX.

Converts ``experiments/results/stats/tables/*.md`` (the pipeline's own output,
covered by the 84-test stats suite) into ``results/tablas/Tabla_X.Y.Z_*.{csv,md,tex}``
-- a direct format conversion, not a recomputation, so there is zero risk of the
thesis-numbered copies drifting from the numbers already cited in
``Chapter6_FINAL.md``.

Usage
-----
    python scripts/export_thesis_tables.py [--out-dir ../results/tablas]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

SOURCE_DIR = REPO_ROOT / "experiments" / "results" / "stats" / "tables"

#: (source stem, thesis-numbered stem, caption)
MAPPING: tuple[tuple[str, str, str], ...] = (
    ("h1_h5_verdict", "Tabla_6.0_veredicto_hipotesis",
     "Veredicto por hipotesis (H1-H5), resumen ejecutivo del capitulo"),
    ("e1_rpd_summary", "Tabla_6.2.1_rpd_jssp_e1",
     "RPD (%) por instancia y metodo, brazo JSSP estatico (E1, N=43)"),
    ("comparadores_pdr_e1", "Tabla_6.2.2_comparadores_pdr_e1",
     "pdr_global, pdr_por_familia y oraculo_best_per_instance (RPD medio, %), separados (T4.1, Recomendacion 6)"),
    ("e3_rpd_summary", "Tabla_6.3.2_rpd_generalizacion_e3",
     "RPD medio +/- desviacion estandar por tier de generalizacion y metodo (E3)"),
    ("e3_h2_dual_reading", "Tabla_6.3.3_h2_lectura_dual",
     "H2: lectura de rango (Friedman-Nemenyi) vs. lectura competitiva absoluta (RPD vs. pdr_global), por tier (T4.2, Recomendacion 7)"),
    ("e4e5e6e7_ri_summary", "Tabla_6.4.1_ri_por_regimen",
     "Indice de Robustez (RI) por regimen y serie -- mejor PDR, kappa en {0, 0.5, 1.0} (E4-E7)"),
    ("e8_latency_summary", "Tabla_6.5.1_latencia_e8",
     "Tiempo de decision (ms/epoca, mediana), RL vs. CP-SAT, las 43 instancias (E8)"),
    ("costo_computacional", "Tabla_6.5.2_costo_computacional",
     "Costo computacional homogeneo: entrenamiento, inferencia, amortizado (GNN-DRL), CP-SAT, GA, PDR (T5.1, Recomendacion 8)"),
)


def _markdown_table_to_df(markdown: str) -> pd.DataFrame:
    lines = [line for line in markdown.strip().splitlines() if line.strip().startswith("|")]
    header = [c.strip().strip("*") for c in lines[0].strip("|").split("|")]
    rows = [[c.strip().strip("*") for c in line.strip("|").split("|")] for line in lines[2:]]
    return pd.DataFrame(rows, columns=header)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT.parent / "results" / "tablas")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for source_stem, thesis_stem, caption in MAPPING:
        md_path = SOURCE_DIR / f"{source_stem}.md"
        tex_path = SOURCE_DIR / f"{source_stem}.tex"
        if not md_path.exists():
            print(f"[export_thesis_tables] SKIP {source_stem}: {md_path} not found")
            continue

        markdown = md_path.read_text(encoding="utf-8")
        df = _markdown_table_to_df(markdown)

        csv_out = args.out_dir / f"{thesis_stem}.csv"
        df.to_csv(csv_out, index=False, encoding="utf-8")

        md_out = args.out_dir / f"{thesis_stem}.md"
        md_out.write_text(f"<!-- {caption} -->\n\n{markdown}\n", encoding="utf-8")

        if tex_path.exists():
            tex_out = args.out_dir / f"{thesis_stem}.tex"
            tex_out.write_text(tex_path.read_text(encoding="utf-8"), encoding="utf-8")
            written.append(tex_out)

        written += [csv_out, md_out]

    print(f"[export_thesis_tables] wrote {len(written)} files to {args.out_dir}")
    for p in sorted(written):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
