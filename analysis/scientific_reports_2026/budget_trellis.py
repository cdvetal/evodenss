#!/usr/bin/env python3
"""
Trellis of epoch-matched comparisons: one small heatmap per (SSL setup, scenario).
Within each heatmap: x = downstream epochs (100..600), y = pretext epochs (100..500).
Cell colour = mean paired difference SSL - supervised at the same total epochs (pp, n = 10 seeds);
dot = significant after Holm correction over the eight setups at that budget (alpha = 0.05);
black frame = the Table 5 comparison (500 pretext + 600 downstream vs 1100 supervised epochs).
Reads the CSV written by all_setups_by_budget.py.

Usage: python budget_trellis.py all_setups_by_budget.csv out_stem
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"   # matplotlib default; matches the manuscript figures
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

CSV, OUT = sys.argv[1], sys.argv[2]
df = pd.read_csv(CSV)
SETUPS = ["fixed proj / orig DA / frozen", "fixed proj / new DA / frozen", "evolved proj / orig DA / frozen",
          "evolved proj / new DA / frozen", "fixed proj / orig DA / FT", "fixed proj / new DA / FT",
          "evolved proj / orig DA / FT", "evolved proj / new DA / FT"]
ROWLAB = ["fixed projector\noriginal DA", "fixed projector\nnew DA", "evolved projector\noriginal DA",
          "evolved projector\nnew DA"] * 2
SCEN = [("CIFAR-10", 10), ("CIFAR-10", 100), ("CIFAR-100", 10), ("CIFAR-100", 100)]
PRETEXT = [100, 200, 300, 400, 500]; DOWN = [100, 200, 300, 400, 500, 600]
vmax = 25
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax); cmap = "RdBu"
GREY, BLUE = "#7f7f7f", "#1f77b4"

fig, axes = plt.subplots(8, 4, figsize=(6.93, 7.9))   # Scientific Reports template: \textwidth = 6.93 in
for i, s in enumerate(SETUPS):
    for j, (ds, pct) in enumerate(SCEN):
        ax = axes[i, j]
        g = df[(df["dataset"] == ds) & (df["labelled_pct"] == pct) & (df["setup"] == s)]
        M = g.pivot(index="pretext", columns="downstream", values="mean_diff").loc[PRETEXT, DOWN].values
        S = g.pivot(index="pretext", columns="downstream", values="p_holm").loc[PRETEXT, DOWN].values < 0.05
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest", origin="lower")
        yy, xx = np.where(S); ax.scatter(xx, yy, s=14, color="black", edgecolor="white", linewidths=0.5, zorder=3)
        ax.add_patch(Rectangle((5 - 0.5, 4 - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.3, zorder=4))
        # white hairlines between cells
        for k in range(1, 6): ax.axvline(k - 0.5, color="white", lw=0.8)
        for k in range(1, 5): ax.axhline(k - 0.5, color="white", lw=0.8)
        ax.set_xticks(range(6)); ax.set_yticks(range(5))
        ax.set_xticklabels([str(d) for d in DOWN] if i == 7 else [], fontsize=6.3, rotation=45)
        ax.set_yticklabels([str(p) for p in PRETEXT] if j == 0 else [], fontsize=7)
        ax.tick_params(length=0, pad=2)
        for sp in ax.spines.values(): sp.set_visible(False)
        if i == 0:
            ax.set_title(f"{ds}\n{pct}% labels", fontsize=8.5, fontweight="bold", pad=6)
        if j == 0:
            ax.text(-0.9, 0.5, ROWLAB[i], transform=ax.transAxes, ha="center", va="center", fontsize=7.5,
                    color=GREY if i < 4 else BLUE)

# section labels for the two representation policies
fig.text(0.02, 0.745, "FROZEN", rotation=90, ha="center", va="center", fontsize=9.5, color=GREY, fontweight="bold")
fig.text(0.02, 0.355, "FINETUNED", rotation=90, ha="center", va="center", fontsize=9.5, color=BLUE, fontweight="bold")

fig.text(0.62, 0.118, "downstream epochs", ha="center", fontsize=8.5)
fig.text(0.205, 0.55, "pretext epochs", rotation=90, ha="center", va="center", fontsize=8.5)
cax = fig.add_axes([0.37, 0.082, 0.5, 0.012])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
cb.set_label("Δ test accuracy, SSL − supervised at the same total epochs (percentage points)", fontsize=8)
cb.ax.tick_params(labelsize=8)
cb.ax.text(-0.01, 0.5, "supervised better  ◀", transform=cb.ax.transAxes, ha="right", va="center", fontsize=8, color="#b2182b")
cb.ax.text(1.01, 0.5, "▶  SSL better", transform=cb.ax.transAxes, ha="left", va="center", fontsize=8, color="#2166ac")
handles = [Line2D([], [], marker="o", color="black", ls="", ms=4.5, label="significant after Holm (α = 0.05, eight setups)"),
           Line2D([], [], marker="s", mfc="none", mec="black", ls="", ms=7, label="Table 5 comparison (500 + 600 vs 1100)")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.58, 0.0), columnspacing=1.2)
fig.subplots_adjust(left=0.265, right=0.985, top=0.95, bottom=0.16, hspace=0.3, wspace=0.14)
fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".png", dpi=200)
print("saved", OUT)
