<!-- Table S2. H5 sensitivity check for the la40 CP-SAT measurement anomaly (raw wall-clock 7216.67s vs. la38's 60.0554s on an identical-size, identical-status instance under the same 60s budget -- see thesis Chapter 6 Sec. 6.5.1 for the full anomaly audit). Under either reading H5 is rejected (0/43 or 1/43 pass the strict all-instances threshold); the correction makes the rejection more decisive, not less -- reported here so the paper's one-sentence summary (Sec. V-D) is independently checkable. -->

| reading | la40_ms_per_decision_rl | la40_ms_per_decision_cpsat | cpsat_over_rl_ratio | passes_1pct_threshold | h5_instances_passing_of_43 |
|---|---|---|---|---|---|
| Raw (as measured) | 18.085 | 32074.086 | 1773.49 | True | 1 |
| Sensitivity-corrected (la40 CP-SAT wall-clock substituted with la38's, same 15x15/225-op size class, both FEASIBLE under the identical 60s budget) | 18.085 | 266.913 | 14.76 | False | 0 |
