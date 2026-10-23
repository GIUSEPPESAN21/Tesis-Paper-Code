"""Exploratory sensitivity analysis of the dispatching-rule implementation
(Section 4.5 and Tables 6-7 of the COR manuscript; NOT part of the pre-registration).

Place this file in code/experiments/sensitivity/ and run from the repository's code/ folder:
    python experiments/sensitivity/COR_sensitivity_analysis.py
Outputs (written next to this file): pdr_nondelay.csv, dyn_nominal.csv, h3_consistent.csv,
and the H1/H2/H3 figures reported in Section 4.5.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path.insert(0, str(CODE))
from baselines.dispatching_rules import run_rule                      # noqa: E402
from envs.jssp_env import JSSPEnv                                     # noqa: E402
from envs.dynamic_jssp_env import DynamicJSSPEnv                      # noqa: E402
from envs.disruptions import DisruptionConfig                         # noqa: E402
from envs.loader import load_by_name, known_optimum                   # noqa: E402

W24 = CODE / "experiments" / "results" / "week24"
RULES = ["spt", "lpt", "mwkr", "lwkr", "mor", "lor", "fifo", "edd", "cr"]
INSTANCES = ["ft06", "ft10", "ft20"] + [f"la{i:02d}" for i in range(1, 41)]
NEMENYI_Q = 3.164  # q_0.05 for k = 10 methods (Demsar, 2006)


class NonDelayEnv(JSSPEnv):
    """Static env restricted to the non-delay set (Giffler & Thompson, 1960): only the
    frontier operations whose earliest start equals the minimum earliest start are legal."""
    def action_masks(self):
        mask = super().action_masks()
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return mask
        mach = self.machine_per_op()
        est = np.array([max(self.job_ready_time[self.instance.job_of(int(o))],
                            self.machine_free_time[int(mach[o])]) for o in legal])
        out = np.zeros_like(mask)
        out[legal[est <= est.min() + 1e-9]] = True
        return out


def rpd(ms, opt):
    return 100.0 * (ms - opt) / opt


def table6():
    rows = []
    for name in INSTANCES:
        inst, opt = load_by_name(name), known_optimum(name)
        for variant, env_cls in [("semi_active", JSSPEnv), ("non_delay", NonDelayEnv)]:
            for rule in RULES:
                r = run_rule(env_cls(inst), rule, seed=0)
                assert r["feasible"]
                rows.append(dict(instance=name, variant=variant, rule=rule, rpd=rpd(r["makespan"], opt)))
        for rule in RULES:  # event-driven simulator without disruption
            r = run_rule(DynamicJSSPEnv(inst, DisruptionConfig()), rule, seed=0)
            rows.append(dict(instance=name, variant="event_driven_nominal", rule=rule, rpd=rpd(r["makespan"], opt)))
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "pdr_nondelay.csv", index=False)
    t = d.groupby(["rule", "variant"]).rpd.mean().unstack().round(2)
    print("Table 6 (mean RPD %, 43 instances, seed 0):\n", t.sort_values("non_delay"))
    gap = (t["non_delay"] - t["event_driven_nominal"]).abs().max()
    print(f"max |non-delay - event-driven nominal| = {gap:.2e}  (should be 0)")
    return d


def h1_h2(d):
    nd = d[d.variant == "non_delay"].pivot(index="instance", columns="rule", values="rpd")
    e1 = pd.read_csv(W24 / "e1_merged.csv"); e1 = e1[e1.seed == 0]
    rl = e1[e1.method == "rl"].set_index("instance")["rpd_pct"]
    diff = (nd.min(axis=1) - rl).dropna()
    w = stats.wilcoxon(diff); wa = stats.wilcoxon(diff, method="approx")
    r = -stats.norm.isf(wa.pvalue / 2) / np.sqrt(len(diff))
    print(f"H1 non-delay: best-rule mean {nd.min(axis=1).mean():.2f}, policy {rl.mean():.2f}, "
          f"policy better on {(diff > 0).sum()}/{len(diff)}, W={w.statistic:.0f}, p={w.pvalue:.2e}, r={r:.2f}")
    e3 = pd.read_csv(W24 / "e3_generalization_la21_la40.csv"); e3 = e3[e3.seed == 0]
    rl3 = e3[e3.method == "rl"].set_index("instance")["rpd_pct"]
    cd = NEMENYI_Q * np.sqrt(10 * 11 / (6 * 5))
    for lo in (21, 26, 31, 36):
        ins = [f"la{i}" for i in range(lo, lo + 5)]
        m = nd.loc[ins].copy(); m["rl"] = rl3.loc[ins].values
        avg = m.rank(axis=1).mean(); best = avg.drop("rl").idxmin()
        fp = stats.friedmanchisquare(*[m[c] for c in m]).pvalue
        print(f"H2 la{lo}-la{lo+4}: Friedman p={fp:.1e}, rank gap {avg['rl'] - avg[best]:.2f} vs CD {cd:.2f} (best {best})")
    ins = [f"la{i}" for i in range(21, 41)]
    d3 = rl3.loc[ins] - nd.loc[ins].min(axis=1); w3 = stats.wilcoxon(d3)
    print(f"H2 pooled: policy minus best non-delay rule {d3.mean():.2f} pp, policy better on {(d3 < 0).sum()}/20, "
          f"W={w3.statistic:.0f}, p={w3.pvalue:.1e}")


def h3(d):
    spec = importlib.util.spec_from_file_location("h3", W24 / "compute_h3_verdict.py")
    h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    nominal = h.load_nominal_makespans()
    ev = d[d.variant == "event_driven_nominal"]
    consistent = dict(nominal)
    for (inst, method) in nominal:
        if method != "rl":
            x = ev[(ev.instance == inst) & (ev.rule == method)].rpd.iloc[0]
            consistent[(inst, method)] = known_optimum(inst) * (1 + x / 100)
    out = []
    for rid, pdr_csv, rl_csv, _axis, _lvl, cand in h.REGIMES:
        for label, nom in [("pre_registered", nominal), ("consistent", consistent)]:
            pdr = h.compute_ri_per_row(pd.read_csv(h.PDR_DIR / pdr_csv), nom)
            rlr = h.compute_ri_per_row(pd.read_csv(h.RL_DIR / rl_csv), nom)
            best, best_ri = h.best_pdr_for_regime(pdr, cand)
            m = pdr[pdr.method == best].merge(rlr, on=["instance", "seed"], suffixes=("_pdr", "_rl"))
            diff = m.ri_pdr - m.ri_rl; w = stats.wilcoxon(diff)
            r = np.sign(diff.median()) * stats.norm.isf(w.pvalue / 2) / np.sqrt(len(diff))
            # realized quality (cross-method comparable)
            bq = pdr.groupby("method").rpd_pct.median()[cand].idxmin()
            m2 = pdr[pdr.method == bq].merge(rlr, on=["instance", "seed"], suffixes=("_pdr", "_rl"))
            d2 = m2.rpd_pct_pdr - m2.rpd_pct_rl
            out.append(dict(regime=rid, nominal=label, best_rule=best, median_ri_rule=round(best_ri, 3),
                            median_ri_policy=round(rlr.ri.median(), 3), median_diff=round(diff.median(), 3),
                            W=w.statistic, p=w.pvalue, r=round(r, 2), reject=not (w.pvalue < .05 and diff.median() > 0),
                            best_realized_rule=bq, realized_median_diff_pp=round(d2.median(), 2),
                            realized_p=stats.wilcoxon(d2).pvalue))
    o = pd.DataFrame(out); o.to_csv(HERE / "h3_consistent.csv", index=False)
    pd.set_option("display.width", 220); print(o.to_string(index=False))


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    data = table6(); h1_h2(data); h3(data)
