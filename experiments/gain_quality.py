"""Quality of the approximate marginal gain (Lemma 5), the analogue of UniPROT Fig. 2.

For one split per dataset (eps = 0.05; coreset 10%, or 1% for Adult), run the
per-stratum FC-POT greedy with approximate gains. At every step record
    ratio = fhat(j*|P) / f(j*|P)
for the selected element j*, where f is computed exactly (network simplex), and,
on the small datasets, the same ratio for 20 random unselected candidates.
Also record the worst-case bound alpha of Lemma 5 for each stratum.

Output: results_extra/gain_quality.csv
"""
import os
import numpy as np
import pandas as pd
import ot
from sklearn.model_selection import train_test_split

import fcpot_exp as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_extra")
os.makedirs(OUT, exist_ok=True)


def pot_value(Csel, b, beta):
    """f(P) = max <beta - C, G>, rows = 1, cols <= b (exact, dummy-row network simplex)."""
    k = Csel.shape[0]
    if k == 0:
        return 0.0, b.copy()
    kappa = b.sum()
    dummy = max(kappa - k, 0.0)
    a = np.concatenate([np.ones(k), [dummy]]) if dummy > 1e-12 else np.ones(k)
    M = np.vstack([Csel, np.full((1, len(b)), beta)]) if dummy > 1e-12 else Csel
    G = ot.emd(a, b, M, numItermax=int(1e7))
    val = float(((beta - Csel) * G[:k]).sum())
    resid = np.maximum(G[k], 0.0) if dummy > 1e-12 else np.zeros_like(b)
    return val, resid


def fill_val(k_row, caps, mass=1.0, descending=True):
    o = np.argsort(-k_row if descending else k_row, kind="stable")
    cs = np.cumsum(caps[o])
    take = np.clip(mass - (cs - caps[o]), 0, caps[o])
    return float((k_row[o] * take).sum())


def run_stratum(Zc, Zt, kappa, rng, n_cand):
    nt = Zt.shape[0]
    b = np.full(nt, kappa / nt)
    C = E.l1_cdist(Zc, Zt)
    beta = C.max() + 1.0
    K = beta - C
    alpha = min(fill_val(K[j], b, descending=False) / fill_val(K[j], b) for j in range(len(Zc)))
    P, rows = [], []
    fP, r = 0.0, b.copy()
    for t in range(int(kappa)):
        avail = np.setdiff1d(np.arange(len(Zc)), P)
        fhat = np.array([fill_val(K[j], r) for j in avail])
        jstar = int(avail[np.argmax(fhat)])
        f_new, r_new = pot_value(C[P + [jstar]], b, beta)
        rows.append(dict(step=t, kind="selected", ratio=fhat.max() / (f_new - fP)))
        if n_cand:
            for j in rng.choice(avail[avail != jstar], size=min(n_cand, len(avail) - 1), replace=False):
                fj, _ = pot_value(C[P + [int(j)]], b, beta)
                rows.append(dict(step=t, kind="candidate", ratio=fill_val(K[j], r) / (fj - fP)))
        P.append(jstar); fP, r = f_new, r_new
    return rows, alpha


def main():
    recs = []
    for ds, frac, cap, n_cand in [("drug", 0.10, 3000, 20), ("german", 0.10, 3000, 20),
                                  ("crime", 0.10, 3000, 20), ("adult", 0.01, 1500, 0)]:
        Xdf, D, Y = E.load(ds)
        idx = np.arange(len(D))
        tr, _ = train_test_split(idx, train_size=0.75, random_state=0)
        tr, _ = train_test_split(tr, train_size=0.9, random_state=0)
        X = E.encode(Xdf, tr)
        Xtr, Dtr, Ytr = X[tr], D[tr], Y[tr]
        m = int(round(frac * len(tr)))
        budgets, _ = E.fair_budgets(Dtr, Ytr, m, 0.05, Ytr.mean())
        Z = E.Zmat(Xtr, Dtr, Ytr)
        rng = np.random.default_rng(0)
        for (d, y), k in budgets.items():
            pool = np.flatnonzero((Dtr == d) & (Ytr == y))
            if k < 2:
                continue
            tgt = pool if len(pool) <= cap else rng.choice(pool, cap, replace=False)
            # candidate pool capped for exact evaluation on Adult (stochastic-greedy regime)
            src = pool if len(pool) <= 3000 else rng.choice(pool, 3000, replace=False)
            rows, alpha = run_stratum(Z[src], Z[tgt], k, rng, n_cand)
            for rw in rows:
                rw.update(dataset=ds, stratum=f"D={d},Y={y}", kappa_s=k, alpha=alpha,
                          progress=rw["step"] / max(k - 1, 1))
            recs += rows
            print(ds, (d, y), "kappa_s", k, "alpha %.3f" % alpha,
                  "min selected ratio %.3f" % min(r["ratio"] for r in rows if r["kind"] == "selected"), flush=True)
    pd.DataFrame(recs).to_csv(os.path.join(OUT, "gain_quality.csv"), index=False)


if __name__ == "__main__":
    main()
