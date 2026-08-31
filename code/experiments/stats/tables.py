"""Output formats: Markdown and LaTeX tables. ``StatsAnalysis_TechSpec.md`` Section 9."""

from __future__ import annotations

import pandas as pd

from experiments.stats.pairwise import HypothesisVerdict


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def hypothesis_verdict_table(verdicts: list[HypothesisVerdict]) -> str:
    """The single table ``FINAL_RESULTS_REPORT.md`` leads with: one row per H1-H5 (H1
    split into JSSP/FJSP-Arm-A/FJSP-Arm-B sub-rows, H3 split into per-regime + overall,
    H4 split per kappa), columns: Hypothesis | Test | Statistic | p-value | Effect size
    | Verdict | Notes.
    """
    lines = [
        "| Hypothesis | Test | Statistic | p-value | Effect size (r) | Verdict | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in verdicts:
        if v.test_result is not None:
            test_name = f"Wilcoxon ({v.test_result.method})"
            statistic = _fmt(v.test_result.statistic)
            p_value = _fmt(v.test_result.p_value)
            effect = _fmt(v.test_result.effect_size_r)
        else:
            test_name = "--"
            statistic = "--"
            p_value = "--"
            effect = "--"
        lines.append(
            f"| {v.hypothesis} | {test_name} | {statistic} | {p_value} | {effect} | "
            f"**{v.verdict.upper()}** | {v.notes} |"
        )
    return "\n".join(lines)


def rpd_summary_table(df: pd.DataFrame, group_by: list[str], *, value_col: str = "rpd_pct") -> str:
    """Per-instance or per-tier RPD table: method columns, instance rows, mean +/- std
    where multiple seeds exist.
    """
    grouped = df.groupby(group_by + ["method"])[value_col].agg(["mean", "std", "count"]).reset_index()

    def fmt_cell(row) -> str:
        if row["count"] > 1 and pd.notna(row["std"]):
            return f"{row['mean']:.2f} +/- {row['std']:.2f}"
        return f"{row['mean']:.2f}"

    grouped["cell"] = grouped.apply(fmt_cell, axis=1)
    pivot = grouped.pivot(index=group_by, columns="method", values="cell")
    pivot = pivot.reset_index()

    header = "| " + " | ".join(str(c) for c in pivot.columns) + " |"
    sep = "|" + "|".join(["---"] * len(pivot.columns)) + "|"
    rows = []
    for _, row in pivot.iterrows():
        rows.append("| " + " | ".join(str(v) if pd.notna(v) else "--" for v in row) + " |")
    return "\n".join([header, sep, *rows])


def to_latex_table(markdown_or_df, *, caption: str, label: str) -> str:
    """Booktabs style (``\\toprule``/``\\midrule``/``\\bottomrule``, no vertical rules).
    ``label`` is auto-prefixed ``tab:ch6-`` so cross-references in the LaTeX thesis
    source are consistent without hand-editing.
    """
    if isinstance(markdown_or_df, pd.DataFrame):
        df = markdown_or_df
    else:
        df = _markdown_table_to_df(markdown_or_df)

    col_spec = "l" * len(df.columns)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:ch6-{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(str(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cells = [str(v).replace("_", r"\_").replace("%", r"\%") for v in row]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _markdown_table_to_df(markdown: str) -> pd.DataFrame:
    lines = [line for line in markdown.strip().splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        raise ValueError("to_latex_table: markdown input does not look like a pipe table")
    header = [cell.strip().strip("*") for cell in lines[0].strip("|").split("|")]
    data_lines = lines[2:]  # skip the |---|---| separator row
    rows = []
    for line in data_lines:
        cells = [cell.strip().strip("*") for cell in line.strip("|").split("|")]
        rows.append(cells)
    return pd.DataFrame(rows, columns=header)
