"""Appendix figures and tables (same paper style as the main trade-off figure).

  gain_quality.pdf        approximate-gain ratio vs worst-case alpha (UniPROT Fig. 2 analogue)
  w1_vs_size.pdf          W1 to the training set vs coreset size, mean +- std over splits
  per_size_auc.tex        downstream AUC per dataset and coreset size
  per_size_dd.tex         downstream demographic disparity per dataset and coreset size
  clustering_cost.tex     clustering cost per dataset and coreset size (FWC Table 4)
  eps_effect.tex          budget feasibility, realised parity violation, AUC and DD vs eps

Usage: python extra_figs.py   (reads results/ and results_extra/, writes report/)
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analyze import paper_rc, PAPER, NAMES, TITLES, ORDER, DS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report")
os.makedirs(OUT, exist_ok=True)

df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(HERE, "results", "*_run*.csv"))],
               ignore_index=True)
d0 = df[df["rw"] == False]


TEX_TITLES = {k: v.replace("&", r"\&") for k, v in TITLES.items()}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def size_label(ds, s):
    return f"{s * 100:g}\\%"


# ---------------------------------------------------------------------------
def fig_gain_quality():
    g = pd.read_csv(os.path.join(HERE, "results_extra", "gain_quality.csv"))
    paper_rc()
    strata_col = {"D=0,Y=0": "#0072B2", "D=0,Y=1": "#D55E00", "D=1,Y=0": "#009E73", "D=1,Y=1": "#CC79A7"}
    strata_ls = {"D=0,Y=0": "-", "D=0,Y=1": "--", "D=1,Y=0": "-.", "D=1,Y=1": ":"}
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6), squeeze=False)
    for ax, ds in zip(axes.ravel(), DS):
        x = g[g["dataset"] == ds]
        cand = x[x["kind"] == "candidate"]
        if len(cand):
            ax.scatter(cand["progress"], cand["ratio"], s=4, c="#b5b5b5", alpha=0.45, lw=0, zorder=1)
        for st, sub in x[x["kind"] == "selected"].groupby("stratum"):
            ax.plot(sub["progress"], sub["ratio"], color=strata_col[st], ls=strata_ls[st], lw=1.1, zorder=3)
        amin = x["alpha"].min()
        ax.axhline(amin, color="black", lw=0.9, ls=(0, (4, 2)), zorder=2)
        ax.text(0.98, amin + 0.025, rf"worst-case $\alpha={amin:.2f}$", ha="right", va="bottom", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 1)
        ax.set_title(TITLES[ds], pad=4)
        if ax in axes[-1]:
            ax.set_xlabel("Greedy progress $t/\\kappa_s$")
    for row in axes:
        row[0].set_ylabel(r"$\hat f(j\,|\,P)\,/\,f(j\,|\,P)$")
    handles = [Line2D([], [], color=strata_col[s], ls=strata_ls[s], lw=1.1,
                      label=f"selected, $D={s[2]},\\,Y={s[6]}$") for s in strata_col]
    handles += [Line2D([], [], marker="o", ls="none", color="#b5b5b5", ms=3.5, label="random candidates"),
                Line2D([], [], color="black", ls=(0, (4, 2)), lw=0.9, label=r"bound $\alpha$ (Lemma 5)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True, fancybox=False,
               edgecolor="black", handlelength=2.0, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.885), w_pad=0.8, h_pad=0.6)
    save(fig, "gain_quality")
    sel = g[g["kind"] == "selected"]
    return (sel.groupby("dataset")["ratio"].agg(["min", "median"]),
            g.groupby("dataset")["alpha"].min(),
            g[g["kind"] == "candidate"].groupby("dataset")["ratio"].min())


# ---------------------------------------------------------------------------
def fig_w1_vs_size():
    paper_rc()
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6), squeeze=False)
    per_run = d0.groupby(["dataset", "method", "size", "run"])["W1"].mean().reset_index()
    a = per_run.groupby(["dataset", "method", "size"])["W1"].agg(["mean", "std"]).reset_index()
    for ax, ds in zip(axes.ravel(), DS):
        x = a[a["dataset"] == ds]
        sizes = sorted(x["size"].unique())
        pos = {s: i for i, s in enumerate(sizes)}
        offs = np.linspace(-0.18, 0.18, len(ORDER))
        for k, meth in enumerate(ORDER):
            sub = x[x["method"] == meth].sort_values("size")
            col, mk, ls, ms, z = PAPER[meth]
            xs = [pos[s] + offs[k] for s in sub["size"]]
            ax.errorbar(xs, sub["mean"], yerr=sub["std"], color=col, marker=mk, ls=ls, lw=0.9,
                        ms=ms * 0.8, mec="black" if meth == "FC-POT" else "white", mew=0.5,
                        capsize=1.5, elinewidth=0.7, zorder=z)
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([f"{s * 100:g}%" for s in sizes])
        ax.set_title(TITLES[ds], pad=4)
        ax.grid(axis="x", visible=False)
        if ax in axes[-1]:
            ax.set_xlabel("Coreset size (fraction of training set)")
    for row in axes:
        row[0].set_ylabel(r"$W_1$ to training set $\downarrow$")
    handles = [Line2D([], [], color=PAPER[m][0], marker=PAPER[m][1], ls=PAPER[m][2], lw=0.9,
                      ms=PAPER[m][3] * 0.8, mec="black" if m == "FC-POT" else "white", mew=0.5,
                      label=NAMES[m]) for m in ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True, fancybox=False,
               edgecolor="black", handlelength=2.2, columnspacing=1.3)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=0.8, h_pad=0.6)
    save(fig, "w1_vs_size")


# ---------------------------------------------------------------------------
def per_size_table(metric, name, higher_better, nd=3, scale=None):
    """Methods x (dataset, size); entries are means over 10 splits (pooled over eps)."""
    per_run = d0.groupby(["dataset", "method", "size", "run"])[metric].mean().reset_index()
    a = per_run.groupby(["dataset", "method", "size"])[metric].mean()
    cols = [(ds, s) for ds in DS for s in sorted(d0[d0["dataset"] == ds]["size"].unique())]
    best = {}
    for c in cols:
        vals = {m: a.get((c[0], m, c[1])) for m in ORDER}
        vals = {m: v for m, v in vals.items() if v is not None and not np.isnan(v)}
        best[c] = (max if higher_better else min)(vals, key=vals.get)
    head1 = " & ".join(rf"\multicolumn{{3}}{{c}}{{{TEX_TITLES[ds]}}}" for ds in DS)
    rules = " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(DS)))
    head2 = " & ".join(f"{s * 100:g}\\%" for _, s in cols)
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             " & " + head1 + r"\\", rules, "Method & " + head2 + r"\\", r"\midrule"]
    for m in ORDER:
        cells = []
        for c in cols:
            v = a.get((c[0], m, c[1]))
            if scale:
                v = v / scale[c[0]]
            txt = f"{v:.{nd}f}"
            cells.append(r"\textbf{" + txt + "}" if best[c] == m else txt)
        nm = r"\method{}" if m == "FC-POT" else NAMES[m]
        lines.append(nm + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def eps_table():
    fc = df[(df["method"] == "FC-POT") & (df["rw"] == False)]
    fw = df[(df["method"] == "FWC") & (df["rw"] == False)]
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r" & & \multicolumn{4}{c}{\method{}} & \multicolumn{2}{c}{FWC}\\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
             r"Dataset & $\epsilon$ & Feasible & Viol.\ (max) & AUC & DD & Viol.\ (max) & AUC / DD\\",
             r"\midrule"]
    for ds in DS:
        for i, e in enumerate([0.01, 0.05, 0.1]):
            a = fc[(fc["dataset"] == ds) & (np.isclose(fc["eps"], e))]
            b = fw[(fw["dataset"] == ds) & (np.isclose(fw["eps"], e))]
            feas = a["budget_feasible"].astype(bool).mean() * 100
            lines.append((TEX_TITLES[ds] if i == 0 else "") +
                         f" & {e:g} & {feas:.0f}\\% & {a['parity_viol'].max():.3f} & {a['auc'].mean():.3f}"
                         f" & {a['dd'].mean():.3f} & {b['parity_viol'].max():.3f}"
                         f" & {b['auc'].mean():.3f} / {b['dd'].mean():.3f}\\\\")
        lines.append(r"\midrule" if ds != DS[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(os.path.join(OUT, "eps_effect.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    q, al, cand_min = fig_gain_quality()
    print("selected-ratio min/median:\n", q.round(3), "\nalpha min:\n", al.round(3), "\ncandidate min:\n", cand_min.round(3))
    fig_w1_vs_size()
    per_size_table("auc", "per_size_auc", True)
    per_size_table("dd", "per_size_dd", False)
    cc_scale = {"adult": 1e4, "german": 1e3, "crime": 1e3, "drug": 1e3}
    per_size_table("clust_cost", "clustering_cost", False, nd=2, scale=cc_scale)
    eps_table()
    print("wrote", OUT)
