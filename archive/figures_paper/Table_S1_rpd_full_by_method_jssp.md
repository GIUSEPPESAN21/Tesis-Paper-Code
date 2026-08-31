<!-- Table S1. Full per-method RPD (%) on the JSSP arm (E1, N=43 instances, mean over seed-0 deterministic evaluation), all 9 PDR individually plus GA/CP-SAT/RL. Derived by column mean/SD of results/tablas/Tabla_6.2.1_rpd_jssp_e1.csv -- same numbers already cited in Chapter 6/7 of the thesis, reduced across instances, not recomputed. -->

| method | type | description | mean_rpd_pct | sd_rpd_pct | median_rpd_pct | n |
|---|---|---|---|---|---|---|
| CPSAT | Exact/metaheuristic | CP-SAT (exact) | 0.02 | 0.13 | 0.0 | 43 |
| GA | Exact/metaheuristic | Genetic Algorithm | 5.52 | 5.31 | 5.15 | 43 |
| FIFO | PDR | First In First Out | 19.15 | 11.1 | 21.11 | 43 |
| RL | RL policy | RL (this work) | 24.82 | 12.92 | 25.48 | 43 |
| MOR | PDR | Most Operations Remaining | 39.83 | 16.91 | 39.02 | 43 |
| MWKR | PDR | Most Work Remaining | 42.59 | 16.19 | 44.2 | 43 |
| SPT | PDR | Shortest Processing Time | 268.23 | 114.01 | 283.39 | 43 |
| LPT | PDR | Longest Processing Time | 281.88 | 116.91 | 290.02 | 43 |
| EDD | PDR | Earliest Due Date | 387.34 | 177.0 | 383.84 | 43 |
| LWKR | PDR | Least Work Remaining | 391.91 | 177.61 | 383.84 | 43 |
| LOR | PDR | Least Operations Remaining | 401.15 | 174.56 | 423.11 | 43 |
| CR | PDR | Critical Ratio | 430.31 | 183.62 | 417.81 | 43 |
