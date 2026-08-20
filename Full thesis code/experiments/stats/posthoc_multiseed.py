"""Fase 2 (T2.4): generador de results/tablas/Tabla_H4_multisemilla_posthoc.{csv,md,tex}.

**Analisis post-hoc y exploratorio, explicitamente etiquetado como tal (Regla 1).**
El veredicto pre-registrado de H4 (``PreRegistration_FINAL.md`` Sec 5,
``FINAL_RESULTS_REPORT.md`` Sec 6.4: REJECT para kappa=0.5 y kappa=1.0) permanece
publicado sin cambios. Esta tabla NO recomputa ni reinterpreta ese veredicto -- agrega,
sobre las semillas de entrenamiento multiples que T2.1 permite producir, la misma
metrica (RI, RPD nominal) que motivo el REJECT, para que el capitulo de discusion
pueda decir si el REJECT es estable entre semillas o si una semilla particular fue
inusualmente mala. Los valores numericos aqui son SIEMPRE de una corrida real
(``experiments.stats.seeds.aggregate_over_seeds`` sobre datos ya existentes) o quedan
como ``[PENDIENTE-CORRIDA]`` -- nunca inventados.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.stats.seeds import aggregate_over_seeds, format_mean_ci  # noqa: E402

POSTHOC_HEADER = (
    "**ANALISIS POST-HOC Y EXPLORATORIO.** El veredicto pre-registrado de H4 "
    "(`PreRegistration_FINAL.md` Sec 5) permanece publicado SIN CAMBIOS: REJECT para "
    "kappa=0.5 y kappa=1.0 (`FINAL_RESULTS_REPORT.md` Sec 6.4). Esta tabla no "
    "recomputa ese veredicto -- agrega la misma metrica sobre semillas de "
    "entrenamiento adicionales (T2.1) para evaluar la ESTABILIDAD del hallazgo, no su "
    "validez confirmatoria."
)


def load_summary_field(summary_paths: list[Path], field: str) -> list[float]:
    """Read one numeric field from each ``summary_*_seed{N}.json``, skipping (with a
    printed note) any file missing that field rather than failing the whole aggregate.
    """
    values: list[float] = []
    for path in summary_paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if field not in data or data[field] is None:
            print(f"[posthoc_multiseed] {path}: missing/null field {field!r}, skipping")
            continue
        values.append(float(data[field]))
    return values


def tabla_h4_multisemilla_posthoc_markdown(
    metric_glob_by_kappa: dict[float, str], *, field: str = "last_100_mean_makespan"
) -> str:
    """T2.4's table: one row per kappa, ``field`` aggregated over every matching
    ``summary_*_seed*.json`` found by that kappa's glob pattern.

    Args:
        metric_glob_by_kappa: ``{kappa: glob_pattern}``, e.g.
            ``{0.0: "experiments/results/dynamic/summary_dynamic_kappa0.0_D1_ft06_seed*.json",
              0.5: "...kappa0.5_D1_ft06_seed*.json", 1.0: "...kappa1.0_D1_ft06_seed*.json"}``.
        field: Which ``summary_*.json`` field to aggregate (default:
            ``last_100_mean_makespan``, the closest existing point estimate to a
            per-seed "how good did training end up" summary).

    Returns:
        Markdown, starting with :data:`POSTHOC_HEADER`. A kappa with no matching files
        yet gets a ``[PENDIENTE-CORRIDA]`` row instead of being silently omitted.
    """
    lines = [POSTHOC_HEADER, "", f"| kappa | {field} (media +/- DE, IC95, n) | Semillas encontradas |", "|---|---|---|"]
    for kappa, pattern in sorted(metric_glob_by_kappa.items()):
        paths = [Path(p) for p in glob.glob(pattern)]
        if not paths:
            lines.append(f"| {kappa} | [PENDIENTE-CORRIDA: {pattern}] | 0 |")
            continue
        values = load_summary_field(paths, field)
        if not values:
            lines.append(f"| {kappa} | [PENDIENTE-CORRIDA: ningun archivo tenia {field!r}] | 0 |")
            continue
        agg = aggregate_over_seeds(values)
        lines.append(f"| {kappa} | {format_mean_ci(agg)} | {agg.n_seeds} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI:

        python -m experiments.stats.posthoc_multiseed \\
            --kappa-glob 0.0 "experiments/results/dynamic/summary_dynamic_kappa0.0_D1_ft06_seed*.json" \\
            --kappa-glob 0.5 "experiments/results/dynamic/summary_dynamic_kappa0.5_D1_ft06_seed*.json" \\
            --kappa-glob 1.0 "experiments/results/dynamic/summary_dynamic_kappa1.0_D1_ft06_seed*.json" \\
            --out results/tablas/Tabla_H4_multisemilla_posthoc
    """
    import argparse

    from experiments.stats.tables import to_latex_table

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kappa-glob", nargs=2, action="append", metavar=("KAPPA", "GLOB"), required=True)
    parser.add_argument("--field", default="last_100_mean_makespan")
    parser.add_argument("--out", type=Path, default=REPO_ROOT.parent / "results" / "tablas" / "Tabla_H4_multisemilla_posthoc")
    args = parser.parse_args(argv)

    metric_glob_by_kappa = {float(k): pattern for k, pattern in args.kappa_glob}
    md = tabla_h4_multisemilla_posthoc_markdown(metric_glob_by_kappa, field=args.field)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    args.out.with_suffix(".tex").write_text(
        to_latex_table(md, caption="H4 multi-semilla (post-hoc, exploratorio -- veredicto pre-registrado sin cambios)", label="h4-multisemilla-posthoc"),
        encoding="utf-8",
    )
    print(f"wrote {args.out}.{{md,tex}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
