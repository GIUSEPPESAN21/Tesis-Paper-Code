#!/bin/bash
set -uo pipefail
cd "/home/aiaindustrial/Downloads/THESIS MASTER'S/Full thesis code"
source .venv-linux/bin/activate

SEEDS=$(seq 0 29)
OUT=experiments/results/week24/e4e5e6

echo "=== E4 breakdowns ==="
for avail in 0.95 0.90 0.80; do
  echo "--- availability=$avail ---"
  python -u run_evaluation.py --instances ft06 la01 --methods pdr \
    --disruption breakdowns --availability "$avail" --seeds $SEEDS \
    --out "$OUT/e4_breakdowns_avail${avail}.csv" --quiet
done

echo "=== E5 stochastic_times ==="
for sigma in 0.1 0.25 0.5; do
  echo "--- sigma=$sigma ---"
  python -u run_evaluation.py --instances ft06 la01 --methods pdr \
    --disruption stochastic_times --proc-sigma "$sigma" --seeds $SEEDS \
    --out "$OUT/e5_stochastic_sigma${sigma}.csv" --quiet
done

echo "=== E6 arrivals (arrival-safe PDRs only: lwkr mor lor fifo) ==="
for util in 0.7 0.85 0.95; do
  echo "--- utilization=$util ---"
  python -u run_evaluation.py --instances ft06 la01 --methods lwkr mor lor fifo \
    --disruption arrivals --utilization "$util" --seeds $SEEDS \
    --out "$OUT/e6_arrivals_util${util}.csv" --quiet
done

echo "=== descriptive-only: GA/CP-SAT under one regime, one instance (NOT full replication, per matrix guidance) ==="
python -u run_evaluation.py --instances ft06 --methods ga cpsat \
  --disruption breakdowns --availability 0.90 --seeds 0 1 2 3 4 --time-limit 30 \
  --out "$OUT/descriptive_ga_cpsat_breakdowns90_ft06.csv" --quiet

echo "E4E5E6_ALL_DONE"
