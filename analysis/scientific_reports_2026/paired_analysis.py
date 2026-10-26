#!/usr/bin/env python3
"""
Paired statistical comparison of SSL and supervised neuroevolution after training extension.

Reproduces Table 5, Figure 4 and Supplementary Appendix D of
"Assessing the role of label availability in self-supervised and supervised neuroevolution".

Input
-----
srep_paper_results_final_checkpoints.xlsx with the per-checkpoint extension logs:
    cifar10_extensions,  cifar100_extensions        SSL runs: one row per (setup, run, pretext epochs, downstream epochs)
    cifar10_extensions_1100, cifar100_extensions_1100  supervised runs: one row per (run, epoch), 100..1100
Columns (first eight, in order): evolvable projector, data augmentation, downstream mode,
    % labelled data, run, pretext epochs, downstream epochs, test accuracy (fraction).

Runs are paired by run ID: run k of every SSL setup and run k of the supervised baseline
share the same numerical seed: run k uses seed k (see the README sheet of the results file).

Checkpoint rules
----------------
peak   : per-run maximum test accuracy over all logged checkpoints
         (SSL: 5 pretext x 6 downstream = 30 checkpoints; supervised: 11 checkpoints).
final  : fixed, predefined checkpoint — SSL 500 pretext + 600 downstream epochs; supervised 1100 epochs.

Statistics
----------
For every (SSL setup, supervised baseline) pair within a scenario (dataset x label fraction):
    d_k   = acc_SSL(run k) - acc_SUP(run k),                 k = 1..10
    mean paired difference, 95% CI = t_{0.975, n-1} * sd(d) / sqrt(n)
    two-sided paired t-test (scipy.stats.ttest_rel)
Family-wise error across the 8 setups of a scenario is controlled with the Holm step-down
procedure at alpha = 0.05. The four scenarios are treated as separate families.

Three comparisons are reported, as in Appendix D:
    primary     : SSL <rule> vs supervised <rule>   (same rule on both sides)
    adverse     : SSL final vs supervised peak      (bound working against SSL)
    favourable  : SSL peak  vs supervised final     (bound working in favour of SSL)

Usage
-----
    python paired_analysis.py srep_paper_results_final_checkpoints.xlsx --rule peak  --out results_peak
    python paired_analysis.py srep_paper_results_final_checkpoints.xlsx --rule final --out results_final   # reported

Outputs (in --out directory)
----------------------------
    per_run.csv      one row per (dataset, label %, setup, run): peak and final accuracy, checkpoint of the peak
    paired_tests.csv one row per (scenario, setup, comparison): mean diff, CI, t, p, Holm-adjusted p, significance
    summary.txt      human-readable table

Dependencies: pandas, numpy, scipy, openpyxl.  Tested with pandas 2.x, scipy 1.1x.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
SHEETS = {
    "CIFAR-10": ("cifar10_extensions", "cifar10_extensions_1100"),
    "CIFAR-100": ("cifar100_extensions", "cifar100_extensions_1100"),
}
LABEL_FRACTIONS = (10, 100)
# (downstream mode, projector evolved?, data augmentation) in the order used in Figure 4
SETUPS = [
    ("freeze", False, "original"), ("freeze", False, "alternative"),
    ("freeze", True, "original"), ("freeze", True, "alternative"),
    ("finetune", False, "original"), ("finetune", False, "alternative"),
    ("finetune", True, "original"), ("finetune", True, "alternative"),
]


def load_sheet(xlsx, sheet):
    df = pd.read_excel(xlsx, sheet_name=sheet).iloc[:, :8].copy()
    df.columns = ["proj", "da", "mode", "pct", "run", "pre", "down", "acc"]
    df["pre"] = pd.to_numeric(df["pre"], errors="coerce")  # '-' for supervised -> NaN
    df["acc"] = df["acc"].astype(float) * 100.0
    df["run"] = df["run"].astype(int)
    return df


def per_run_values(df):
    """Per-run peak and final accuracy for one setup. Returns DataFrame indexed by run."""
    out = []
    final_pre = df["pre"].max()  # NaN for supervised
    final_down = df["down"].max()
    for run, g in df.groupby("run"):
        i_peak = g["acc"].idxmax()
        sel = g[g["down"] == final_down]
        if not np.isnan(final_pre):
            sel = sel[sel["pre"] == final_pre]
        if len(sel) != 1:
            sys.exit(f"expected exactly one final checkpoint for run {run}, found {len(sel)}")
        out.append(dict(run=run, peak=g.loc[i_peak, "acc"], peak_pre=g.loc[i_peak, "pre"],
                        peak_down=g.loc[i_peak, "down"], final=sel["acc"].iloc[0],
                        n_checkpoints=len(g)))
    return pd.DataFrame(out).set_index("run").sort_index()


def holm(pvals):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_test(a, b):
    """a, b: aligned Series (same run index). Returns dict of paired statistics."""
    b = b.reindex(a.index)
    if b.isna().any():
        sys.exit("unpaired runs: supervised baseline missing a run ID present in the SSL setup")
    d = (a - b).values
    n = len(d)
    t, p = stats.ttest_rel(a.values, b.values)
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return dict(n=n, mean_diff=d.mean(), ci_low=d.mean() - half, ci_high=d.mean() + half,
                sd_diff=d.std(ddof=1), t=t, p=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--rule", choices=["peak", "final"], default="peak",
                    help="checkpoint rule for the primary comparison")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    per_run_rows, test_rows, lines = [], [], []
    for ds, (sh_ssl, sh_sup) in SHEETS.items():
        ssl = load_sheet(args.xlsx, sh_ssl)
        sup = load_sheet(args.xlsx, sh_sup)
        for pct in LABEL_FRACTIONS:
            sup_runs = per_run_values(sup[(sup["mode"] == "supervised") & (sup["pct"] == pct)])
            for _, r in sup_runs.iterrows():
                per_run_rows.append(dict(dataset=ds, labelled_pct=pct, paradigm="supervised", setup="supervised (1100 ep.)",
                                         run=int(r.name), peak=r["peak"], peak_epoch=int(r["peak_down"]),
                                         final=r["final"], n_checkpoints=int(r["n_checkpoints"])))
            scenario = []
            for mode, proj, da in SETUPS:
                q = ssl[(ssl["mode"] == mode) & (ssl["proj"] == proj) & (ssl["da"] == da) & (ssl["pct"] == pct)]
                ssl_runs = per_run_values(q)
                name = f"{'evolved' if proj else 'fixed'} projector / {'new' if da == 'alternative' else 'original'} DA / {'finetuned' if mode == 'finetune' else 'frozen'}"
                for _, r in ssl_runs.iterrows():
                    per_run_rows.append(dict(dataset=ds, labelled_pct=pct, paradigm="SSL", setup=name, run=int(r.name),
                                             peak=r["peak"], peak_epoch=f"{int(r['peak_pre'])}+{int(r['peak_down'])}",
                                             final=r["final"], n_checkpoints=int(r["n_checkpoints"])))
                comps = {
                    "primary": (ssl_runs[args.rule], sup_runs[args.rule]),
                    "adverse_to_SSL": (ssl_runs["final"], sup_runs["peak"]),
                    "favourable_to_SSL": (ssl_runs["peak"], sup_runs["final"]),
                }
                for comp, (a, b) in comps.items():
                    st = paired_test(a, b)
                    st.update(dataset=ds, labelled_pct=pct, setup=name, comparison=comp,
                              ssl_mean=a.mean(), ssl_sd=a.std(ddof=1), sup_mean=b.mean(), sup_sd=b.std(ddof=1))
                    scenario.append(st)
            # Holm within scenario, separately for each comparison family (8 tests each)
            for comp in ("primary", "adverse_to_SSL", "favourable_to_SSL"):
                fam = [s for s in scenario if s["comparison"] == comp]
                for s, pa in zip(fam, holm([s["p"] for s in fam])):
                    s["p_holm"] = pa
                    s["significant"] = pa < ALPHA
            test_rows.extend(scenario)

            prim = [s for s in scenario if s["comparison"] == "primary"]
            lines.append(f"\n{ds}, {pct}% labels — rule: {args.rule}; supervised {prim[0]['sup_mean']:.2f} ± {prim[0]['sup_sd']:.2f}")
            for s in prim:
                lines.append(f"  {s['setup']:45s} SSL {s['ssl_mean']:6.2f} ± {s['ssl_sd']:5.2f}   "
                             f"Δ {s['mean_diff']:+6.2f} [{s['ci_low']:+6.2f}, {s['ci_high']:+6.2f}]   "
                             f"t = {s['t']:6.2f}  p = {s['p']:.4f}  p_Holm = {s['p_holm']:.4f}{'  *' if s['significant'] else ''}")

    pd.DataFrame(per_run_rows).to_csv(os.path.join(args.out, "per_run.csv"), index=False)
    cols = ["dataset", "labelled_pct", "setup", "comparison", "n", "ssl_mean", "ssl_sd", "sup_mean", "sup_sd",
            "mean_diff", "ci_low", "ci_high", "sd_diff", "t", "p", "p_holm", "significant"]
    pd.DataFrame(test_rows)[cols].to_csv(os.path.join(args.out, "paired_tests.csv"), index=False)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
