#!/bin/bash
set -uo pipefail
cd "/home/aiaindustrial/Downloads/THESIS MASTER'S/Full thesis code"
source .venv-linux/bin/activate

SEEDS=$(seq 0 29)
OUT=experiments/results/week24/e4e5e6_rl
CKPT="experiments/results/curriculum_full/checkpoints/curriculum_stage19_la20_final.zip"

echo "=== E4 breakdowns (RL) ==="
for avail in 0.95 0.90 0.80; do
  echo "--- availability=$avail ---"
  python -u run_evaluation.py --instances ft06 la01 --methods rl \
    --disruption breakdowns --availability "$avail" --seeds $SEEDS \
    --rl-policy "$CKPT" \
    --out "$OUT/e4_breakdowns_avail${avail}_rl.csv" --quiet
done

echo "=== E5 stochastic_times (RL) ==="
for sigma in 0.1 0.25 0.5; do
  echo "--- sigma=$sigma ---"
  python -u run_evaluation.py --instances ft06 la01 --methods rl \
    --disruption stochastic_times --proc-sigma "$sigma" --seeds $SEEDS \
    --rl-policy "$CKPT" \
    --out "$OUT/e5_stochastic_sigma${sigma}_rl.csv" --quiet
done

echo "=== E6 arrivals (RL -- no arrival-safety restriction, unlike PDRs) ==="
for util in 0.7 0.85 0.95; do
  echo "--- utilization=$util ---"
  python -u run_evaluation.py --instances ft06 la01 --methods rl \
    --disruption arrivals --utilization "$util" --seeds $SEEDS \
    --rl-policy "$CKPT" \
    --out "$OUT/e6_arrivals_util${util}_rl.csv" --quiet
done

echo "E4E5E6_RL_ALL_DONE"
