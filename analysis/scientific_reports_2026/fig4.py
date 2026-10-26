import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"   # matplotlib default; matches the manuscript figures
from matplotlib.lines import Line2D

XLSX = sys.argv[1]
OUT = sys.argv[2]          # output stem
MODE = sys.argv[3]         # 'peak' or 'final'
SSL_PRE = float(sys.argv[4]) if len(sys.argv) > 4 else None    # fixed checkpoint override (SSL pretext epochs)
SSL_DOWN = float(sys.argv[5]) if len(sys.argv) > 5 else None   # SSL downstream epochs
SUP_EP = float(sys.argv[6]) if len(sys.argv) > 6 else None     # supervised epochs

def load(sheet):
    df = pd.read_excel(XLSX, sheet_name=sheet).iloc[:, :8]
    df.columns = ["proj", "da", "mode", "pct", "run", "pre", "down", "acc"]
    return df

def per_run(df, mode):
    """Return Series indexed by run: peak over checkpoints or final checkpoint."""
    if mode == "peak":
        return df.groupby("run")["acc"].max() * 100
    pre = df["pre"].replace("-", np.nan).astype(float)
    is_sup = pre.isna().all()
    last_pre = pre.max() if SSL_PRE is None else SSL_PRE
    last_down = df["down"].max() if (SSL_DOWN is None or is_sup) else SSL_DOWN
    if is_sup and SUP_EP is not None:
        last_down = SUP_EP
    sel = df[df["down"] == last_down]
    if not is_sup:
        sel = sel[pre.loc[sel.index] == last_pre]
    return sel.set_index("run")["acc"].sort_index() * 100

SETUPS = [  # (frozen?, proj evolved?, da) in figure order top->bottom
    ("freeze", False, "original"), ("freeze", False, "alternative"),
    ("freeze", True, "original"), ("freeze", True, "alternative"),
    ("finetune", False, "original"), ("finetune", False, "alternative"),
    ("finetune", True, "original"), ("finetune", True, "alternative"),
]

results = {}
for ds, sh_ssl, sh_sup in [("CIFAR-10", "cifar10_extensions", "cifar10_extensions_1100"),
                           ("CIFAR-100", "cifar100_extensions", "cifar100_extensions_1100")]:
    ssl = load(sh_ssl); sup = load(sh_sup)
    for pct in (10, 100):
        s = sup[(sup["mode"] == "supervised") & (sup["pct"] == pct)]
        sup_runs = per_run(s, MODE)
        rows = []
        for mode, proj, da in SETUPS:
            q = ssl[(ssl["mode"] == mode) & (ssl["proj"] == proj) & (ssl["da"] == da) & (ssl["pct"] == pct)]
            ssl_runs = per_run(q, MODE)
            d = (ssl_runs - sup_runs.reindex(ssl_runs.index)).values
            t, p = stats.ttest_rel(ssl_runs.values, sup_runs.reindex(ssl_runs.index).values)
            ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            rows.append(dict(mode=mode, proj=proj, da=da, ssl_mean=ssl_runs.mean(), ssl_sd=ssl_runs.std(ddof=1),
                             diff=d, mean=d.mean(), ci=ci, p=p))
        # Holm correction over the 8 tests of the scenario
        ps = np.array([r["p"] for r in rows]); order = np.argsort(ps); m = len(ps)
        adj = np.empty(m); running = 0
        for k, i in enumerate(order):
            running = max(running, (m - k) * ps[i]); adj[i] = min(1, running)
        for r, a in zip(rows, adj): r["p_holm"] = a; r["sig"] = a < 0.05
        results[(ds, pct)] = dict(rows=rows, sup_mean=sup_runs.mean(), sup_sd=sup_runs.std(ddof=1))

# ---- print table
for key, v in results.items():
    print(f"\n{key[0]}, {key[1]}% labels — supervised {v['sup_mean']:.2f} ± {v['sup_sd']:.2f}")
    for r in v["rows"]:
        print(f"  {r['mode']:8s} proj={'evolved' if r['proj'] else 'fixed':7s} {r['da']:11s} "
              f"SSL {r['ssl_mean']:6.2f} ± {r['ssl_sd']:5.2f}  Δ {r['mean']:+6.2f} [±{r['ci']:.2f}]  "
              f"p={r['p']:.4f} holm={r['p_holm']:.4f} {'*' if r['sig'] else ''}")

# ---- figure
GREY, BLUE, RUN_GREY, RUN_BLUE = "#7f7f7f", "#1f77b4", "#c7c7c7", "#a6cee3"
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
panels = [("CIFAR-10", 10, "(a)"), ("CIFAR-10", 100, "(b)"), ("CIFAR-100", 10, "(c)"), ("CIFAR-100", 100, "(d)")]
ylabels = ["original DA", "new DA"] * 4
rng = np.random.default_rng(0)
for ax, (ds, pct, tag) in zip(axes.flat, panels):
    v = results[(ds, pct)]
    ys = np.arange(8)[::-1]
    for y, r in zip(ys, v["rows"]):
        col = GREY if r["mode"] == "freeze" else BLUE
        rcol = RUN_GREY if r["mode"] == "freeze" else RUN_BLUE
        jitter = rng.uniform(-0.18, 0.18, len(r["diff"]))
        ax.scatter(r["diff"], y + jitter, s=7, color=rcol, zorder=2, linewidths=0)
        off = r["diff"][r["diff"] < -25]
        if len(off):
            ax.text(-24.6, y + 0.14, f"◂{len(off)} run at {off.min():.0f}", fontsize=5.5, color=col, ha="left", va="bottom")
        ax.errorbar(r["mean"], y, xerr=r["ci"], fmt="none", ecolor=col, elinewidth=1.6, capsize=3, capthick=1.6, zorder=3)
        ax.scatter([r["mean"]], [y], s=46, facecolor=col if r["sig"] else "white", edgecolor=col, linewidths=1.6, zorder=4)
    ax.axvline(0, color="#555555", lw=1, zorder=1)
    ax.axhline(3.5, color="#dddddd", lw=0.8, zorder=0)
    ax.set_yticks(ys); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_ylim(-0.7, 7.7); ax.set_xlim(-25, 33)
    ax.set_xticks([-20, -10, 0, 10, 20, 30])
    ax.grid(axis="x", color="#eeeeee", lw=0.8, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=8); ax.tick_params(axis="y", length=0)
    ax.set_title(f"{tag}  {ds}, {pct}% labels", fontsize=9, fontweight="bold", loc="left", pad=14)
    ax.text(0, 1.02, f"supervised reference: {v['sup_mean']:.1f} ± {v['sup_sd']:.1f}%", transform=ax.transAxes,
            fontsize=7.5, color="#555555", ha="left", va="bottom")
    if ax in (axes[0, 0], axes[1, 0]):
        for y0, lab, col in [(6.5, "fixed\nprojector", GREY), (4.5, "evolved\nprojector", GREY),
                             (2.5, "fixed\nprojector", BLUE), (0.5, "evolved\nprojector", BLUE)]:
            ax.annotate(lab, xy=(-0.41, y0), xycoords=("axes fraction", "data"), fontsize=6.8, color=col,
                        ha="center", va="center", annotation_clip=False)
        for y0, lab, col in [(5.5, "FROZEN", GREY), (1.5, "FINETUNED", BLUE)]:
            ax.annotate(lab, xy=(-0.6, y0), xycoords=("axes fraction", "data"), fontsize=8, color=col,
                        ha="center", va="center", rotation=90, fontweight="bold", annotation_clip=False)
fig.supxlabel("Δ test accuracy, SSL − supervised (percentage points)", fontsize=9, y=0.085)
handles = [Line2D([], [], marker="o", color="k", ls="", ms=7, label="significant after Holm (α = 0.05)"),
           Line2D([], [], marker="o", mfc="white", mec="k", color="k", ls="", ms=7, label="not significant"),
           Line2D([], [], marker="o", color=RUN_GREY, ls="", ms=4, label="individual paired run (n = 10)")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.55, 0.0))
fig.subplots_adjust(left=0.25, right=0.98, top=0.92, bottom=0.17, hspace=0.45, wspace=0.3)
fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".png", dpi=200)
print("saved", OUT)
