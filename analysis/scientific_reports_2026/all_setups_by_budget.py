#!/usr/bin/env python3
"""
Figure-4 analysis (all 8 SSL setups vs supervised, paired t-test, Holm over the 8 setups)
repeated at every epoch-matched budget: SSL at (pretext p, downstream d) vs supervised at p + d epochs.

Outputs <out>.csv with one row per (scenario, p, d, setup) and prints, per scenario, a table of
pretext x downstream cells giving the number of setups significantly better (+) / worse (-) than supervised.

Usage: python all_setups_by_budget.py srep_paper_results_final_checkpoints.xlsx out_stem
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

XLSX, OUT = sys.argv[1], sys.argv[2]
SHEETS = {"CIFAR-10": ("cifar10_extensions", "cifar10_extensions_1100"),
          "CIFAR-100": ("cifar100_extensions", "cifar100_extensions_1100")}
SETUPS = [("freeze", False, "original"), ("freeze", False, "alternative"),
          ("freeze", True, "original"), ("freeze", True, "alternative"),
          ("finetune", False, "original"), ("finetune", False, "alternative"),
          ("finetune", True, "original"), ("finetune", True, "alternative")]
PRETEXT = [100, 200, 300, 400, 500]
DOWN = [100, 200, 300, 400, 500, 600]


def load(sheet):
    df = pd.read_excel(XLSX, sheet_name=sheet).iloc[:, :8].copy()
    df.columns = ["proj", "da", "mode", "pct", "run", "pre", "down", "acc"]
    df["pre"] = pd.to_numeric(df["pre"], errors="coerce")
    df["acc"] = df["acc"].astype(float) * 100
    return df


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); r = 0
    for k, i in enumerate(o):
        r = max(r, (m - k) * p[i]); adj[i] = min(1, r)
    return adj


def label(mode, proj, da):
    return f"{'evolved' if proj else 'fixed'} proj / {'new' if da == 'alternative' else 'orig'} DA / {'FT' if mode == 'finetune' else 'frozen'}"


rows = []
for ds, (sh_ssl, sh_sup) in SHEETS.items():
    ssl = load(sh_ssl); sup = load(sh_sup)
    for pct in (10, 100):
        s = sup[(sup["mode"] == "supervised") & (sup["pct"] == pct)].pivot(index="run", columns="down", values="acc")
        for p in PRETEXT:
            for d in DOWN:
                T = p + d
                tests = []
                for mode, proj, da in SETUPS:
                    g = ssl[(ssl["mode"] == mode) & (ssl["proj"] == proj) & (ssl["da"] == da) & (ssl["pct"] == pct)
                            & (ssl["pre"] == p) & (ssl["down"] == d)].set_index("run")["acc"].sort_index()
                    b = s[T].reindex(g.index)
                    diff = (g - b).values
                    t, pv = stats.ttest_rel(g.values, b.values)
                    tests.append(dict(dataset=ds, labelled_pct=pct, pretext=p, downstream=d, total_epochs=T,
                                      setup=label(mode, proj, da), ssl_mean=g.mean(), sup_mean=b.mean(),
                                      mean_diff=diff.mean(), t=t, p=pv))
                for te, pa in zip(tests, holm([x["p"] for x in tests])):
                    te["p_holm"] = pa; te["sig_better"] = pa < 0.05 and te["mean_diff"] > 0
                    te["sig_worse"] = pa < 0.05 and te["mean_diff"] < 0
                rows.extend(tests)

df = pd.DataFrame(rows)
df.to_csv(OUT + ".csv", index=False)
for (ds, pct), g in df.groupby(["dataset", "labelled_pct"]):
    cnt = g.groupby(["pretext", "downstream"]).agg(better=("sig_better", "sum"), worse=("sig_worse", "sum"))
    tab = cnt.apply(lambda r: f"+{int(r.better)}/-{int(r.worse)}", axis=1).unstack("downstream")
    print(f"\n{ds}, {pct}% labels — setups significantly better / worse than supervised (of 8), rows = pretext, cols = downstream")
    print(tab.to_string())
