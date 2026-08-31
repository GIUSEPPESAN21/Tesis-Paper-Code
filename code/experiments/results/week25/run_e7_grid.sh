#!/bin/bash
# E7 evaluation grid: 9 regimes x 30 seeds x {ft06, la01} x {kappa=0.5, kappa=1.0},
# plus nominal (--disruption none) RPD pass for the H4 (b) margin check.
#
# Output paths/names MUST match experiments/stats/cli.py::default_experiment_csvs
# exactly: experiments/results/dynamic/eval_kappa{K}_{regime_label}.csv, where
# regime_label matches experiments/stats/report.py::REGIMES's labels verbatim
# (e.g. "E4_breakdowns_A0.95"), and eval_kappa{K}_nominal.csv for the nominal pass --
# these are the experiment_id tags run_h4_full() looks for, not a display convention.
#
# kappa=0's arm is NOT re-run here: PreRegistration_FINAL.md Section 5 and
# E7_E8_FinalPlan.md Section 2.3 both specify kappa=0 reuses the SAME already-trained
# curriculum_stage19_la20_final.zip checkpoint and its already-accepted-as-final E4-E6
# results (experiments/results/week24/e4e5e6_rl/*.csv) for RI, and E1's existing
# deterministic-rollout row for nominal RPD -- not a fourth training/eval pass.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
PY="./.venv/Scripts/python.exe"

SEEDS=$(seq 0 29)
KAPPA="${1:?usage: run_e7_grid.sh <kappa> <checkpoint.zip>}"
CKPT="${2:?usage: run_e7_grid.sh <kappa> <checkpoint.zip>}"
OUT="experiments/results/dynamic"
mkdir -p "$OUT"

echo "=== E7 kappa=${KAPPA}: E4-analogue breakdowns ==="
declare -A AVAIL_LABEL=( ["0.95"]="A0.95" ["0.90"]="A0.90" ["0.80"]="A0.80" )
for avail in 0.95 0.90 0.80; do
  echo "--- availability=$avail ---"
  "$PY" -u run_evaluation.py --instances ft06 la01 --methods rl \
    --disruption breakdowns --availability "$avail" --seeds $SEEDS \
    --rl-policy "$CKPT" \
    --out "$OUT/eval_kappa${KAPPA}_E4_breakdowns_${AVAIL_LABEL[$avail]}.csv" --quiet
done

echo "=== E7 kappa=${KAPPA}: E5-analogue stochastic_times ==="
declare -A SIGMA_LABEL=( ["0.1"]="s0.10" ["0.25"]="s0.25" ["0.5"]="s0.50" )
for sigma in 0.1 0.25 0.5; do
  echo "--- sigma=$sigma ---"
  "$PY" -u run_evaluation.py --instances ft06 la01 --methods rl \
    --disruption stochastic_times --proc-sigma "$sigma" --seeds $SEEDS \
    --rl-policy "$CKPT" \
    --out "$OUT/eval_kappa${KAPPA}_E5_stochastic_${SIGMA_LABEL[$sigma]}.csv" --quiet
done

echo "=== E7 kappa=${KAPPA}: E6-analogue arrivals ==="
declare -A UTIL_LABEL=( ["0.7"]="u0.70" ["0.85"]="u0.85" ["0.95"]="u0.95" )
for util in 0.7 0.85 0.95; do
  echo "--- utilization=$util ---"
  "$PY" -u run_evaluation.py --instances ft06 la01 --methods rl \
    --disruption arrivals --utilization "$util" --seeds $SEEDS \
    --rl-policy "$CKPT" \
    --out "$OUT/eval_kappa${KAPPA}_E6_arrivals_${UTIL_LABEL[$util]}.csv" --quiet
done

echo "=== E7 kappa=${KAPPA}: nominal (--disruption none), deterministic seed 0, for H4(b) delta_RPD_nominal ==="
"$PY" -u run_evaluation.py --instances ft06 la01 --methods rl \
  --disruption none --seeds 0 \
  --rl-policy "$CKPT" \
  --out "$OUT/eval_kappa${KAPPA}_nominal.csv" --quiet

echo "E7_GRID_KAPPA${KAPPA}_DONE"
