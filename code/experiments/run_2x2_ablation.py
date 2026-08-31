"""Fase 3 (Recomendacion 5 del tutor), T3.1: ablacion 2x2 entorno x recompensa.

Hoy kappa mezcla dos cosas -- el cambio de recompensa y el cambio de regimen de
entrenamiento -- porque la celda kappa=0 se entreno ESTATICAMENTE (JSSPEnv) y se
evaluo via el adaptador dinamico, mientras las celdas kappa>0 se entrenaron
NATIVAMENTE contra DynamicJSSPEnv. Eso impide atribuir el efecto observado (REJECT de
H4, ver FINAL_RESULTS_REPORT.md Sec 6.4) al entorno de entrenamiento o a la
recompensa. Las cuatro celdas:

    A: entorno ESTATICO,  recompensa NOMINAL (kappa=0) -- el checkpoint YA EXISTENTE
       (curriculum_stage19_la20_final.zip o equivalente), reutilizado, no reentrenado.
    B: entorno ESTATICO,  recompensa ROBUSTA (kappa>0)  -- NUEVO. Entrena via
       train_dynamic.py --training-regime none --kappa <k>: DynamicJSSPEnv con
       disrupciones apagadas en TODO episodio (ya validado bit-identico a JSSPEnv,
       tests/test_dynamic_env.py::TestRegressionAgainstStatic). Nota de diseno, no
       verificada por ejecucion (queda en RUN_PLAN.md como supuesto a confirmar en la
       primera corrida): sin averias, `Delta_inest` (envs/dynamic_jssp_env.py
       `_earliest_start_estimates_dynamic`) ya no puede caer por debajo del techo de
       reparacion de maquina (nunca hay maquina rota), pero SI puede seguir siendo
       distinto de cero por contencion de maquina ordinaria o por que la propia
       politica elija despachar mas tarde de lo "optimo" -- no es necesariamente un
       no-op exacto. Si B resulta estadisticamente indistinguible de A, es evidencia
       de que el termino es casi-degenerado sin disrupcion real; si no lo es, el
       termino sigue teniendo efecto incluso sin disrupcion (tambien un resultado
       reportable, no descartado de antemano).
    C: entorno DINAMICO,  recompensa NOMINAL (kappa=0)  -- LA CELDA QUE HOY NO EXISTE
       (T3.1's "critico"). Entrena via train_dynamic.py --training-regime
       domain_randomized --kappa 0.0: ve los 9 regimenes (+ episodios nominales via
       --nominal-episode-fraction, T3.2) pero sin el termino Delta_inest.
    D: entorno DINAMICO,  recompensa ROBUSTA (kappa>0)  -- las corridas E7 ya
       existentes (kappa=0.5, kappa=1.0), reutilizadas, no reentrenadas.

Este script NO se ejecuta en esta sesion (Regla Cero). Por defecto imprime el plan
(dry-run); pasar --execute lo corre de verdad -- decision del usuario, no de esta
sesion.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "ablation_2x2"


@dataclass(frozen=True)
class AblationCell:
    label: str  # "A", "B", "C", "D"
    env_type: str  # "estatico" | "dinamico"
    reward_type: str  # "nominal" | "robusta"
    kappa: float
    training_regime: str  # "none" | "domain_randomized"
    reuse_checkpoint: str | None  # non-None -> do not retrain, reuse this checkpoint


def build_cells(
    *,
    static_checkpoint: str,
    kappa_robust: float,
    kappa_d_checkpoint: str | None,
    instance: str,
    nominal_episode_fraction: float,
) -> list[AblationCell]:
    """The four cells, per this module's docstring. Cells A and D reuse existing
    checkpoints (never retrained); cells B and C are new training runs."""
    return [
        AblationCell("A", "estatico", "nominal", 0.0, "none", reuse_checkpoint=static_checkpoint),
        AblationCell("B", "estatico", "robusta", kappa_robust, "none", reuse_checkpoint=None),
        AblationCell("C", "dinamico", "nominal", 0.0, "domain_randomized", reuse_checkpoint=None),
        AblationCell(
            "D", "dinamico", "robusta", kappa_robust, "domain_randomized",
            reuse_checkpoint=kappa_d_checkpoint,
        ),
    ]


def training_command(
    cell: AblationCell, *, instance: str, init_checkpoint: str, timesteps: int, seeds: list[int], nominal_episode_fraction: float
) -> list[str] | None:
    """Exact ``train_dynamic.py`` invocation for one cell, or ``None`` if the cell
    reuses an existing checkpoint (no training needed)."""
    if cell.reuse_checkpoint is not None:
        return None
    cmd = [
        sys.executable, str(REPO_ROOT / "experiments" / "train_dynamic.py"),
        "--instance", instance, "--kappa", str(cell.kappa),
        "--init-checkpoint", init_checkpoint,
        "--stage", f"ablation2x2_{cell.label}",
        "--timesteps", str(timesteps),
        "--training-regime", cell.training_regime,
        "--seeds", *[str(s) for s in seeds],
        "--output-dir", str(RESULTS_DIR),
    ]
    if cell.training_regime == "domain_randomized":
        cmd += ["--nominal-episode-fraction", str(nominal_episode_fraction)]
    return cmd


def evaluation_command(cell: AblationCell, *, checkpoint: str, instances: list[str], disruption: str) -> list[str]:
    """Exact ``run_evaluation.py`` invocation to score one cell's checkpoint under
    ``disruption`` (call once per regime the cell needs scored on, per Fase 4's
    ``pdr_global``/oraculo comparators and Fase 8's realized-quality metric)."""
    return [
        sys.executable, str(REPO_ROOT / "run_evaluation.py"),
        "--instances", *instances,
        "--methods", "rl",
        "--rl-policy", checkpoint,
        "--disruption", disruption,
        "--out", str(RESULTS_DIR / f"eval_cell{cell.label}_{disruption}.csv"),
    ]


def print_plan(cells: list[AblationCell], commands: dict[str, list[str] | None]) -> None:
    print("Plan de ablacion 2x2 (entorno x recompensa) -- T3.1\n")
    for cell in cells:
        print(f"Celda {cell.label}: entorno={cell.env_type}, recompensa={cell.reward_type}, kappa={cell.kappa}")
        cmd = commands[cell.label]
        if cmd is None:
            print(f"  -> reutiliza checkpoint existente: {cell.reuse_checkpoint}")
        else:
            print(f"  -> {' '.join(cmd)}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance", default="ft06")
    parser.add_argument("--static-checkpoint", required=True, help="cell A's existing static-trained checkpoint")
    parser.add_argument("--kappa-d-checkpoint", default=None, help="cell D's existing E7 kappa>0 checkpoint (omit to retrain)")
    parser.add_argument("--kappa-robust", type=float, default=0.5, help="kappa value for cells B/D")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4], help="T2.1: multi-seed per cell")
    parser.add_argument("--nominal-episode-fraction", type=float, default=0.2)
    parser.add_argument("--execute", action="store_true", help="actually run the commands (default: print the plan only)")
    args = parser.parse_args(argv)

    cells = build_cells(
        static_checkpoint=args.static_checkpoint,
        kappa_robust=args.kappa_robust,
        kappa_d_checkpoint=args.kappa_d_checkpoint,
        instance=args.instance,
        nominal_episode_fraction=args.nominal_episode_fraction,
    )

    commands: dict[str, list[str] | None] = {}
    for cell in cells:
        init_ckpt = args.static_checkpoint  # every cell warm-starts from the static checkpoint
        commands[cell.label] = training_command(
            cell, instance=args.instance, init_checkpoint=init_ckpt, timesteps=args.timesteps,
            seeds=args.seeds, nominal_episode_fraction=args.nominal_episode_fraction,
        )

    print_plan(cells, commands)

    if not args.execute:
        print("(dry-run: pasa --execute para correr de verdad)")
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        cmd = commands[cell.label]
        if cmd is None:
            continue
        print(f"[run_2x2_ablation] launching cell {cell.label}: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[run_2x2_ablation] cell {cell.label} FAILED (exit {result.returncode})", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
