"""Aggregate FC-POT experiment results into FWC-style tables and trade-off plots.

Usage: python analyze.py [results_dir] [out_dir]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "report")
os.makedirs(OUT, exist_ok=True)

ORDER = ["FC-POT", "FWC", "Stratified", "UniPROT", "Uniform", "K-Means", "K-Medoids"]
# Validated categorical slots 1-3 (all-pairs safe) for our method and the two
# closest comparators; remaining baselines are neutral gray, identified by marker.
STYLE = {"FC-POT": ("#2a78d6", "o"), "FWC": ("#eb6834", "s"), "UniPROT": ("#1baf7a", "D"),
         "Stratified": ("#8a8984", "P"), "Uniform": ("#8a8984", "x"),
         "K-Means": ("#8a8984", "^"), "K-Medoids": ("#8a8984", "v")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DS = ["adult", "german", "crime", "drug"]


def label(r):
    return r["method"] + ("" if pd.isna(r["eps"]) else f" (ε={r['eps']:g})")


def load():
    files = glob.glob(os.path.join(RES, "*_run*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["label"] = df.apply(label, axis=1)
    return df


def agg(df, metric):
    g = df.groupby(["dataset", "label", "method", "eps", "size"], dropna=False)[metric]
    return g.agg(["mean", "std", "count"]).reset_index()


def fmt(mu, sd, scale=1.0, nd=3):
    return f"{mu / scale:.{nd}f} ± {sd / scale:.{nd}f}"


def table(df, metric, rw=False, nd=3):
    """Rows: method label, columns: dataset x size (FWC Tables 3-4 layout)."""
    d = df[df["rw"] == rw]
    a = agg(d, metric)
    rows = []
    labels = sorted(a["label"].unique(), key=lambda s: (ORDER.index(s.split(" (")[0]), s))
    for lab in labels:
        row = {"Method": lab}
        for ds in DS:
            sub = a[(a["dataset"] == ds) & (a["label"] == lab)]
            for _, r in sub.sort_values("size").iterrows():
                row[f"{ds} {r['size'] * 100:g}%"] = fmt(r["mean"], 0 if pd.isna(r["std"]) else r["std"], nd=nd)
        rows.append(row)
    return pd.DataFrame(rows)


def pareto(points):
    """points: list of (dd, auc); return frontier (min dd, max auc)."""
    pts = sorted(points)
    front, best = [], -np.inf
    for dd, auc in pts:
        if auc > best:
            front.append((dd, auc)); best = auc
    return front


# Paper style: Okabe-Ito colours (colour-blind safe) + distinct markers and line styles.
PAPER = {  # method: (colour, marker, linestyle, marker size, zorder)
    "FC-POT":     ("#D55E00", "*", "-",  11, 6),
    "FWC":        ("#0072B2", "s", "-",  6,  5),
    "UniPROT":    ("#009E73", "D", "--", 5.5, 4),
    "Stratified": ("#CC79A7", "P", ":",  6.5, 3),
    "K-Means":    ("#E69F00", "^", "-.", 6.5, 3),
    "K-Medoids":  ("#56B4E9", "v", "-.", 6.5, 3),
    "Uniform":    ("#555555", "o", ":",  5.5, 2),
}
NAMES = {"FC-POT": "FC-POT (ours)", "FWC": "FWC", "UniPROT": "UniPROT", "Stratified": "Stratified",
         "K-Means": "$k$-means", "K-Medoids": "$k$-medoids", "Uniform": "Uniform"}
TITLES = {"adult": "Adult", "german": "German Credit", "crime": "Communities & Crime", "drug": "Drug"}


def paper_rc():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5,
        "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8, "axes.edgecolor": "black", "axes.facecolor": "white",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "axes.grid": True, "grid.color": "#d9d9d9", "grid.linestyle": "--", "grid.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def tradeoff_plot(df, rw_baselines):
    """One trajectory per method across coreset sizes (small -> large, marker grows),
    each point averaged over 10 splits and over eps where applicable."""
    paper_rc()
    present = [d for d in DS if d in set(df["dataset"])]
    # Sized for the paper's text width so fonts print at their nominal size.
    nr, nc = (2, 2) if len(present) == 4 else (1, len(present))
    fig, axes = plt.subplots(nr, nc, figsize=(6.5, 2.35 * nr + 0.55), squeeze=False)
    for ax, ds in zip(axes.ravel(), present):
        d = df[df["dataset"] == ds]
        ours = d["method"].isin(["FC-POT", "FWC"])
        if rw_baselines:   # FC-POT/FWC as is; every other method with Kamiran-Calders reweighing
            use = d[(ours & (d["rw"] == False)) | (~ours & (d["rw"] == True))]
        else:
            use = d[d["rw"] == False]
        a = use.groupby(["method", "size"])[["auc", "dd"]].mean().reset_index()
        sizes = sorted(a["size"].unique())
        grow = {sz: 0.75 + 0.25 * i for i, sz in enumerate(sizes)}
        fr = pareto(list(zip(a["dd"], a["auc"])))
        ax.plot([q[0] for q in fr], [q[1] for q in fr], ls=(0, (4, 2)), color="black", lw=0.7,
                alpha=0.55, zorder=1)
        for meth in ORDER[::-1]:
            sub = a[a["method"] == meth].sort_values("size")
            if sub.empty:
                continue
            col, mk, ls, ms, z = PAPER[meth]
            for _, r in sub.iterrows():
                ax.plot(r["dd"], r["auc"], marker=mk, color=col, ms=ms * grow[r["size"]],
                        mec="black" if meth == "FC-POT" else "white", mew=0.5, ls="none", zorder=z + 0.5)
        ax.set_title(TITLES[ds], pad=4)
        if ax in axes[-1]:
            ax.set_xlabel(r"Demographic disparity $\downarrow$")
        ax.margins(x=0.08, y=0.10)
        ax.set_axisbelow(True)
    for row in axes:
        row[0].set_ylabel(r"Test AUC $\uparrow$")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=PAPER[m][0], marker=PAPER[m][1], ls="none",
                      ms=PAPER[m][3] * (1.0 if m != "FC-POT" else 0.95),
                      mec="black" if m == "FC-POT" else "white", mew=0.5, label=NAMES[m])
               for m in ORDER]
    handles.append(Line2D([], [], color="black", ls=(0, (4, 2)), lw=0.7, alpha=0.55, label="Pareto front"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True, fancybox=False,
               edgecolor="black", framealpha=1.0, handlelength=1.6, columnspacing=1.3,
               bbox_to_anchor=(0.5, 1.0), borderpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.885 if nr == 1 else 0.90), w_pad=0.8, h_pad=0.6)
    name = "tradeoff_reweighed" if rw_baselines else "tradeoff"
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def latex_summary(df):
    """Per dataset and method: mean +- std over the 10 splits of the per-split
    average over coreset sizes (and eps, where applicable)."""
    d = df[df["rw"] == False]
    present = [x for x in DS if x in set(d["dataset"])]
    per_run = d.groupby(["dataset", "method", "run"])[["auc", "dd", "W1", "parity_viol"]].mean()
    g = per_run.groupby(["dataset", "method"]).agg(["mean", "std"])
    lines = [r"\begin{tabular}{ll" + "c" * 4 + "}", r"\toprule",
             r"Dataset & Method & AUC $\uparrow$ & DD $\downarrow$ & $W_1$ $\downarrow$ & Parity viol. $\downarrow$\\", r"\midrule"]
    for ds in present:
        best = {m: (g.loc[ds][(m, "mean")].idxmax() if m == "auc" else g.loc[ds][(m, "mean")].idxmin())
                for m in ["auc", "dd", "W1"]}
        for i, meth in enumerate([m for m in ORDER if (ds, m) in g.index]):
            cells = []
            for m in ["auc", "dd", "W1", "parity_viol"]:
                mu, sd = g.loc[(ds, meth), (m, "mean")], g.loc[(ds, meth), (m, "std")]
                c = f"{mu:.3f}\\,{{\\scriptsize$\\pm${sd:.3f}}}" if m != "W1" else f"{mu:.2f}\\,{{\\scriptsize$\\pm${sd:.2f}}}"
                if m in best and best[m] == meth:
                    c = r"\textbf{" + c + "}"
                cells.append(c)
            name = r"\method{}" if meth == "FC-POT" else meth
            lines.append((ds.capitalize() if i == 0 else "") + f" & {name} & " + " & ".join(cells) + r"\\")
        lines.append(r"\midrule" if ds != present[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(os.path.join(OUT, "summary_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    df = load()
    print("runs per dataset:", df.groupby("dataset")["run"].nunique().to_dict())
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        for metric, title, nd in [("W1", "l1-Wasserstein distance to the training set (lower is better)", 3),
                                  ("clust_cost", "Clustering cost (lower is better)", 1),
                                  ("auc", "Downstream AUC (higher is better)", 3),
                                  ("dd", "Downstream demographic disparity (lower is better)", 3),
                                  ("parity_viol", "Coreset parity violation max|p(y|d)/pT(y)-1|", 3)]:
            f.write(f"## {title}\n\n")
            f.write(table(df, metric, nd=nd).to_markdown(index=False))
            f.write("\n\n")
        f.write("## Downstream AUC / DD with Kamiran-Calders reweighing of baselines\n\n")
        f.write(table(df, "auc", rw=True).to_markdown(index=False) + "\n\n")
        f.write(table(df, "dd", rw=True).to_markdown(index=False) + "\n\n")
        f.write("## Selection time (s)\n\n")
        f.write(table(df, "secs", nd=1).to_markdown(index=False) + "\n")
    latex_summary(df)
    tradeoff_plot(df, rw_baselines=False)
    tradeoff_plot(df, rw_baselines=True)
    print("wrote", OUT)
