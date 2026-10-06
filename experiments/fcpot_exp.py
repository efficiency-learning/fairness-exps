"""FC-POT experiments following the protocol of Fair Wasserstein Coresets
(Xiong et al., NeurIPS 2024; arXiv:2311.05436, Sec. 7 and App. C.2).

Datasets, protected attributes, outcomes, coreset sizes, downstream MLP,
splits, metrics (AUC, demographic disparity, l1-Wasserstein distance,
clustering cost) and the Kamiran-Calders reweighing comparison follow FWC.

Usage:
    python fcpot_exp.py --dataset drug --runs 0-9 --out results/
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import ot
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack as sp_vstack
from scipy.special import logsumexp
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

SIZES = {"adult": [0.005, 0.01, 0.02], "german": [0.05, 0.10, 0.20],
         "crime": [0.05, 0.10, 0.20], "drug": [0.05, 0.10, 0.20]}
EPS = [0.01, 0.05, 0.1]


# ---------------------------------------------------------------------------
# Data (FWC App. C.2 definitions)
# ---------------------------------------------------------------------------
def load(name):
    """Return (X_df features, D in {0,1}, Y in {0,1}); categorical columns are object dtype."""
    if name == "adult":
        cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                "hours-per-week", "native-country", "income"]
        df = pd.read_csv(os.path.join(DATA, "adult-all.csv"), header=None, names=cols,
                         skipinitialspace=True)
        D = (df["sex"] == "Male").astype(int).values
        Y = (df["income"].str.strip().str.startswith(">50K")).astype(int).values
        X = df.drop(columns=["fnlwgt", "sex", "income"])
    elif name == "german":
        df = pd.read_csv(os.path.join(DATA, "german_data_credit.csv"))
        D = (df["sex"] == "male").astype(int).values
        Y = (df["class-label"] == 1).astype(int).values           # credit approved
        X = df.drop(columns=["sex", "class-label"])
    elif name == "crime":
        df = pd.read_csv(os.path.join(DATA, "communities_crime.csv"))
        D = (df["racepctblack"] > df["racepctblack"].median()).astype(int).values
        Y = (df["ViolentCrimesPerPop"] > df["ViolentCrimesPerPop"].mean()).astype(int).values
        X = df.drop(columns=["state", "communityname", "fold", "Black", "class",
                             "ViolentCrimesPerPop", "racepctblack"])
    elif name == "drug":
        df = pd.read_csv(os.path.join(DATA, "drug_consumption.data"), header=None)
        D = (df[2] > 0).astype(int).values                       # 0.48246 = female
        Y = (df[18] != "CL0").astype(int).values                  # ever used cannabis
        X = df[[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]].copy()
        X.columns = ["age", "education", "country", "ethnicity", "N", "E", "O", "A", "C", "Imp", "SS"]
    else:
        raise ValueError(name)
    return X.reset_index(drop=True), D, Y


def encode(Xdf, fit_idx):
    """One-hot categoricals, standardize everything using statistics of fit_idx."""
    cat = [c for c in Xdf.columns if Xdf[c].dtype == object or str(Xdf[c].dtype).startswith("str")]
    Xe = pd.get_dummies(Xdf, columns=cat, dtype=float).astype(float).values
    mu = Xe[fit_idx].mean(0); sd = Xe[fit_idx].std(0); sd[sd < 1e-12] = 1.0
    return (Xe - mu) / sd


def sq_cdist(A, B):
    """Squared Euclidean distances via the Gram form (memory O(|A||B|))."""
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


def nearest(Z, M, chunk=4000):
    return np.concatenate([sq_cdist(Z[s:s + chunk], M).argmin(1) for s in range(0, len(Z), chunk)])


def Zmat(X, D, Y):
    """Point representation used for OT costs: (D, X, Y), as in FWC."""
    return np.hstack([D[:, None].astype(float), X, Y[:, None].astype(float)])


def l1_cdist(A, B):
    out = np.empty((A.shape[0], B.shape[0]), dtype=np.float64)
    chunk = max(1, int(1.5e7 // max(1, B.shape[0] * B.shape[1])))
    for s in range(0, A.shape[0], chunk):
        out[s:s + chunk] = np.abs(A[s:s + chunk, None, :] - B[None, :, :]).sum(-1)
    return out


# ---------------------------------------------------------------------------
# Fair budgets (eq. fwc-dp in the paper)
# ---------------------------------------------------------------------------
def fair_budgets(D, Y, m, eps, pT1):
    """Integer budgets kappa_(d,y) with sum m, kappa_(d,y) <= n_(d,y), satisfying
    (1-eps) pT(y) kappa_d <= kappa_(d,y) <= (1+eps) pT(y) kappa_d, closest to
    proportional. Returns (dict, feasible flag)."""
    n = len(D)
    nd = {d: int((D == d).sum()) for d in (0, 1)}
    ndy = {(d, y): int(((D == d) & (Y == y)).sum()) for d in (0, 1) for y in (0, 1)}
    pT = {1: pT1, 0: 1 - pT1}

    def split(kd, d):
        lo = max(math.ceil((1 - eps) * pT[1] * kd - 1e-9), kd - math.floor((1 + eps) * pT[0] * kd + 1e-9))
        hi = min(math.floor((1 + eps) * pT[1] * kd + 1e-9), kd - math.ceil((1 - eps) * pT[0] * kd - 1e-9))
        lo = max(lo, kd - ndy[(d, 0)], 0); hi = min(hi, ndy[(d, 1)], kd)
        if lo > hi:
            return None
        k1 = int(min(max(round(pT[1] * kd), lo), hi))
        return {(d, 1): k1, (d, 0): kd - k1}

    k0_prop = m * nd[0] / n
    best = None
    for k0 in sorted(range(0, m + 1), key=lambda k: abs(k - k0_prop)):
        k1 = m - k0
        if k0 > nd[0] or k1 > nd[1] or k0 < 1 or k1 < 1:
            continue
        s0, s1 = split(k0, 0), split(k1, 1)
        if s0 is not None and s1 is not None:
            best = {**s0, **s1}; break
        if abs(k0 - k0_prop) > 0.25 * m:
            break
    if best is not None:
        return best, True
    # integer-infeasible: closest proportional rounding (violation reported later)
    k0 = int(round(k0_prop)); out = {}
    for d, kd in ((0, k0), (1, m - k0)):
        k1 = int(min(max(round(pT[1] * kd), kd - ndy[(d, 0)]), ndy[(d, 1)]))
        out[(d, 1)] = k1; out[(d, 0)] = kd - k1
    return out, False


# ---------------------------------------------------------------------------
# POT greedy (UniPROT / FC-POT per stratum)
# ---------------------------------------------------------------------------
def _fill_cost(Csorted, rsorted):
    """Min-cost unit-mass fill: Csorted ascending per row, rsorted caps in same order."""
    cs = np.cumsum(rsorted, axis=1)
    take = np.clip(1.0 - (cs - rsorted), 0.0, rsorted)
    return (take * Csorted).sum(1)


class POTSolver:
    """max <K,G>, rows (selected) = 1, cols <= b; exact (network simplex with a
    dummy row) for small problems, warm-started entropic dual ascent otherwise."""

    def __init__(self, b, beta, lam, exact_limit=float('inf'), iters=60):
        self.b = b; self.beta = beta; self.lam = lam
        self.exact_limit = exact_limit; self.iters = iters
        self.g = np.zeros_like(b)

    def residual(self, Csel):
        k_sel = Csel.shape[0]
        kappa = self.b.sum()
        if k_sel == 0:
            return self.b.copy()
        if k_sel * len(self.b) <= self.exact_limit:
            dummy = kappa - k_sel
            a = np.concatenate([np.ones(k_sel), [max(dummy, 0.0)]])
            M = np.vstack([Csel, np.full((1, len(self.b)), self.beta)])
            if dummy <= 1e-12:
                return np.zeros_like(self.b)
            G = ot.emd(a, self.b, M, numItermax=int(1e7))
            return np.maximum(G[-1], 0.0)
        # entropic dual coordinate ascent: rows equality, columns inequality (g >= 0)
        K = (self.beta - Csel) / self.lam
        g = self.g / self.lam
        logb = np.log(self.b)
        for _ in range(self.iters):
            f = logsumexp(K - g[None, :], axis=1)
            g = np.maximum(0.0, logsumexp(K - f[:, None], axis=0) - logb)
        self.g = g * self.lam
        col = np.exp(logsumexp(K - f[:, None] - g[None, :], axis=0))
        return np.maximum(self.b - col, 0.0)


def pot_greedy(Zc, Zt, kappa, rng, lam_frac=0.01, stoch_eps=0.01, full_limit=1.2e8, log=None):
    """Greedy selection of kappa candidates (rows of Zc) for the uniform target Zt
    with approximate marginal gains (paper eq. approx). Returns local indices."""
    nc, nt = Zc.shape[0], Zt.shape[0]
    kappa = int(min(kappa, nc))
    if kappa <= 0:
        return np.array([], dtype=int)
    b = np.full(nt, kappa / nt)
    precompute = nc * nt <= full_limit
    if precompute:
        Cfull = l1_cdist(Zc, Zt).astype(np.float32); cmax = float(Cfull.max())
        order_full = np.argsort(Cfull, axis=1).astype(np.int32)
        Csorted_full = np.take_along_axis(Cfull, order_full, axis=1)
    else:
        cmax = max(l1_cdist(Zc[s:s + 512], Zt).max() for s in range(0, nc, 512))
    beta = cmax + 1.0
    solver = POTSolver(b, beta, lam=lam_frac * cmax)
    selected, Csel_rows = [], []
    avail = np.ones(nc, bool)
    r = b.copy()
    # stochastic greedy (Mirzasoleiman et al., 2015) on large pools, as in UniPROT
    pool_size = min(nc, int(math.ceil(nc / kappa * math.log(1 / stoch_eps)))) if nc > 3000 else nc
    ratios = []
    for t in range(kappa):
        cand = np.flatnonzero(avail)
        if pool_size < len(cand):
            cand = rng.choice(cand, size=pool_size, replace=False)
        if precompute:
            cost = _fill_cost(Csorted_full[cand], r[order_full[cand]])
            Crow = None
        else:
            Cc = l1_cdist(Zc[cand], Zt)
            o = np.argsort(Cc, axis=1)
            cost = _fill_cost(np.take_along_axis(Cc, o, 1), r[o])
        j = int(np.argmin(cost))
        jstar = int(cand[j])
        selected.append(jstar); avail[jstar] = False
        Csel_rows.append(Cfull[jstar].astype(np.float64) if precompute else Cc[j])
        if t + 1 < kappa:
            r = solver.residual(np.vstack(Csel_rows))
    return np.array(selected, dtype=int)


# ---------------------------------------------------------------------------
# Coreset methods. Each returns (Xhat, Dhat, Yhat, weights summing to 1).
# ---------------------------------------------------------------------------
def m_uniform(X, D, Y, m, rng):
    idx = rng.choice(len(D), size=m, replace=False)
    return X[idx], D[idx], Y[idx], np.full(m, 1 / m)


def m_stratified(X, D, Y, budgets, rng):
    idx = []
    for (d, y), k in budgets.items():
        pool = np.flatnonzero((D == d) & (Y == y))
        idx += list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    idx = np.array(idx)
    return X[idx], D[idx], Y[idx], np.full(len(idx), 1 / len(idx))


TARGET_CAP = 3000


def m_fcpot(X, D, Y, budgets, rng, target_cap=None):
    target_cap = target_cap or TARGET_CAP
    """FC-POT: per-stratum UniPROT greedy (strata (D,Y); source = target = stratum)."""
    Z = Zmat(X, D, Y)
    idx = []
    for (d, y), k in budgets.items():
        pool = np.flatnonzero((D == d) & (Y == y))
        if k <= 0 or len(pool) == 0:
            continue
        tgt = pool if len(pool) <= target_cap else rng.choice(pool, size=target_cap, replace=False)
        loc = pot_greedy(Z[pool], Z[tgt], k, rng)
        idx += list(pool[loc])
    idx = np.array(idx)
    return X[idx], D[idx], Y[idx], np.full(len(idx), 1 / len(idx))


def m_uniprot(X, D, Y, m, rng, target_cap=None):
    target_cap = target_cap or TARGET_CAP
    Z = Zmat(X, D, Y)
    n = len(D)
    tgt = np.arange(n) if n <= target_cap else rng.choice(n, size=target_cap, replace=False)
    idx = pot_greedy(Z, Z[tgt], m, rng)
    return X[idx], D[idx], Y[idx], np.full(len(idx), 1 / len(idx))


def m_kmeans(X, D, Y, m, rng):
    Z = Zmat(X, D, Y)
    km = KMeans(n_clusters=m, n_init=1, random_state=int(rng.integers(1 << 31))).fit(Z)
    C = km.cluster_centers_
    w = np.bincount(km.labels_, minlength=m).astype(float)
    keep = w > 0
    Dh = (C[:, 0] >= 0.5).astype(int); Yh = (C[:, -1] >= 0.5).astype(int)
    return C[keep, 1:-1], Dh[keep], Yh[keep], w[keep] / w.sum()


def m_kmedoids(X, D, Y, m, rng, iters=10):
    """Alternating k-medoids (Euclidean), k-means++ initialisation."""
    Z = Zmat(X, D, Y)
    _, med = kmeans_plusplus(Z, n_clusters=m, random_state=int(rng.integers(1 << 31)))
    med = np.array(med)
    for _ in range(iters):
        lab = nearest(Z, Z[med])
        new = med.copy()
        for c in range(m):
            mem = np.flatnonzero(lab == c)
            if len(mem) == 0:
                continue
            if len(mem) > 1500:
                mem_s = rng.choice(mem, 1500, replace=False)
            else:
                mem_s = mem
            dsum = np.sqrt(sq_cdist(Z[mem_s], Z[mem_s])).sum(1)
            new[c] = mem_s[np.argmin(dsum)]
        if np.array_equal(new, med):
            break
        med = new
    lab = nearest(Z, Z[med])
    w = np.bincount(lab, minlength=m).astype(float)
    keep = w > 0
    idx = med[keep]
    return X[idx], D[idx], Y[idx], w[keep] / w.sum()


def m_fwc(X, D, Y, m, eps, pT1, rng, iters=30, counts=None):
    """Fair Wasserstein Coresets (Xiong et al. 2024), majorization-minimization.
    Problem (8) is solved exactly through its class-aggregated reformulation:
    because the parity constraints only involve class totals of the weights,
    each data point sends its mass only to its nearest representative of each
    class, so (8) reduces to an LP with n x |D||Y| variables."""
    n = len(D)
    classes = [(d, y) for d in (0, 1) for y in (0, 1) if ((D == d) & (Y == y)).any()]
    G = len(classes)
    # representatives per class proportional to the data (FWC Step 1)
    raw = {g: m * ((D == g[0]) & (Y == g[1])).mean() for g in classes}
    cnt = {g: max(1, int(round(v))) for g, v in raw.items()}
    while sum(cnt.values()) > m:
        g = max(cnt, key=lambda g: cnt[g] - raw[g]); cnt[g] -= 1
    while sum(cnt.values()) < m:
        g = max(cnt, key=lambda g: raw[g] - cnt[g]); cnt[g] += 1
    if counts is not None:          # e.g. FWC's LLM setup: equal positives per gender
        cnt = {g: counts[g] for g in classes}
    cls_of, Xh = [], []
    for gi, g in enumerate(classes):
        pool = np.flatnonzero((D == g[0]) & (Y == g[1]))
        init = rng.choice(pool, size=cnt[g], replace=len(pool) < cnt[g])
        Xh.append(X[init]); cls_of += [gi] * cnt[g]
    Xh = np.vstack(Xh).copy(); cls_of = np.array(cls_of)
    Dh = np.array([classes[g][0] for g in cls_of]); Yh = np.array([classes[g][1] for g in cls_of])
    Z = Zmat(X, D, Y)
    pT = {1: pT1, 0: 1 - pT1}
    # fairness rows on class totals T_g:  A T >= 0
    A = []
    for d in (0, 1):
        gd = [i for i, g in enumerate(classes) if g[0] == d]
        for gi in gd:
            y = classes[gi][1]
            row_hi = np.zeros(G); row_lo = np.zeros(G)
            for gj in gd:
                row_hi[gj] += (1 + eps) * pT[y]; row_lo[gj] -= (1 - eps) * pT[y]
            row_hi[gi] -= 1; row_lo[gi] += 1
            A += [row_hi, row_lo]
    A = np.array(A)
    prev = np.inf
    for it in range(iters):
        Zh = Zmat(Xh, Dh, Yh)
        C = l1_cdist(Z, Zh)
        cig = np.empty((n, G)); jstar = np.empty((n, G), dtype=int)
        for gi in range(G):
            cols = np.flatnonzero(cls_of == gi)
            sub = C[:, cols]; a = sub.argmin(1)
            jstar[:, gi] = cols[a]; cig[:, gi] = sub[np.arange(n), a]
        # LP: min sum c_ig Q_ig, sum_g Q_ig = 1/n, A T >= 0 with T_g = sum_i Q_ig
        nv = n * G
        rows_eq = coo_matrix((np.ones(nv), (np.repeat(np.arange(n), G), np.arange(nv))), shape=(n, nv))
        rr, cc, vv = [], [], []
        for k in range(A.shape[0]):
            for gi in range(G):
                if A[k, gi] != 0:
                    rr.append(np.full(n, k)); cc.append(np.arange(n) * G + gi); vv.append(np.full(n, -A[k, gi]))
        A_ub = coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(A.shape[0], nv))
        res = linprog(cig.ravel(), A_ub=A_ub.tocsr(), b_ub=np.zeros(A.shape[0]), A_eq=rows_eq.tocsr(),
                      b_eq=np.full(n, 1 / n), bounds=(0, None), method="highs")
        Q = res.x.reshape(n, G)
        obj = res.fun
        # weighted coordinate-wise median update (L1 cost), D-hat and Y-hat fixed
        nz = np.argwhere(Q > 1e-15)
        P_i = nz[:, 0]; P_j = jstar[nz[:, 0], nz[:, 1]]; P_w = Q[nz[:, 0], nz[:, 1]]
        order = np.argsort(P_j, kind="stable")
        P_i, P_j, P_w = P_i[order], P_j[order], P_w[order]
        bounds_ = np.searchsorted(P_j, np.arange(m + 1))
        for j in range(m):
            s, e = bounds_[j], bounds_[j + 1]
            if e <= s:
                continue
            pts = X[P_i[s:e]]; w = P_w[s:e]
            o = np.argsort(pts, axis=0)
            ws = w[o]; cw = np.cumsum(ws, axis=0)
            k = (cw >= 0.5 * w.sum()).argmax(0)
            Xh[j] = np.take_along_axis(pts, o, 0)[k, np.arange(pts.shape[1])]
        if prev - obj <= 1e-7 * max(1.0, abs(prev)):
            break
        prev = obj
    theta = np.bincount(P_j, weights=P_w, minlength=m)
    keep = theta > 1e-12
    return Xh[keep], Dh[keep], Yh[keep], theta[keep] / theta[keep].sum()


# ---------------------------------------------------------------------------
# Evaluation (FWC App. C.2)
# ---------------------------------------------------------------------------
def kamiran_calders(D, Y, w):
    """Reweighing weights w(d,y) = P(d)P(y)/P(d,y) under the coreset weights w."""
    out = w.copy()
    for d in (0, 1):
        for y in (0, 1):
            msk = (D == d) & (Y == y)
            pdy = w[msk].sum()
            if pdy > 0:
                out[msk] *= w[D == d].sum() * w[Y == y].sum() / pdy
    return out / out.sum()


def train_mlp(Xc, Dc, Yc, wc, Xva, Dva, Yva, seed, max_epochs=500, patience=10):
    Xin = np.hstack([Xc, Dc[:, None]]); Xv = np.hstack([Xva, Dva[:, None]])
    if len(np.unique(Yc)) < 2:
        return None
    clf = MLPClassifier(hidden_layer_sizes=(20,), activation="relu", solver="adam",
                        learning_rate_init=1e-3, batch_size=32, random_state=seed)
    sw = wc * len(wc)
    best, best_state, bad = np.inf, None, 0
    for ep in range(max_epochs):
        clf.partial_fit(Xin, Yc, sample_weight=sw, classes=np.array([0, 1]))
        loss = log_loss(Yva, clf.predict_proba(Xv)[:, 1], labels=[0, 1])
        if loss < best - 1e-6:
            best, bad = loss, 0
            best_state = ([c.copy() for c in clf.coefs_], [b.copy() for b in clf.intercepts_])
        else:
            bad += 1
            if bad >= patience:
                break
    clf.coefs_, clf.intercepts_ = best_state
    return clf


def downstream(clf, Xte, Dte, Yte):
    if clf is None:
        return 0.5, 0.0
    Xin = np.hstack([Xte, Dte[:, None]])
    p = clf.predict_proba(Xin)[:, 1]
    yhat = (p >= 0.5).astype(int)
    dd = abs(yhat[Dte == 1].mean() - yhat[Dte == 0].mean())
    return roc_auc_score(Yte, p), dd


def coreset_metrics(Xc, Dc, Yc, wc, Ztr_eval, Ztr_full, pT1):
    Zc = Zmat(Xc, Dc, Yc)
    M = l1_cdist(Zc, Ztr_eval)
    W = float(ot.emd2(wc, np.full(len(Ztr_eval), 1 / len(Ztr_eval)), M, numItermax=int(1e7)))
    cc = 0.0
    for s in range(0, len(Ztr_full), 4000):
        cc += sq_cdist(Ztr_full[s:s + 4000], Zc).min(1).sum()
    # parity violation of the coreset (FWC eq. 3): max |p(y|d)/pT(y) - 1|
    pv = 0.0
    for d in (0, 1):
        wd = wc[Dc == d].sum()
        if wd <= 0:
            pv = max(pv, 1.0); continue
        for y in (0, 1):
            pT = pT1 if y == 1 else 1 - pT1
            pv = max(pv, abs(wc[(Dc == d) & (Yc == y)].sum() / wd / pT - 1))
    return W, cc, pv


# ---------------------------------------------------------------------------
def run_one(name, run, out_dir, methods):
    Xdf, D, Y = load(name)
    n = len(D)
    idx = np.arange(n)
    tr, te = train_test_split(idx, train_size=0.75, random_state=run, stratify=None)
    tr, va = train_test_split(tr, train_size=0.9, random_state=run)
    X = encode(Xdf, tr)
    Xtr, Dtr, Ytr = X[tr], D[tr], Y[tr]
    pT1 = Ytr.mean()
    rng_eval = np.random.default_rng(10_000 + run)
    Ztr = Zmat(Xtr, Dtr, Ytr)
    Ztr_eval = Ztr if len(tr) <= 5000 else Ztr[rng_eval.choice(len(tr), 5000, replace=False)]
    rows = []

    def record(method, frac, eps, core, secs, extra=None):
        Xc, Dc, Yc, wc = core
        W, cc, pv = coreset_metrics(Xc, Dc, Yc, wc, Ztr_eval, Ztr, pT1)
        clf = train_mlp(Xc, Dc, Yc, wc, X[va], D[va], Y[va], seed=run)
        auc, dd = downstream(clf, X[te], D[te], Y[te])
        base = dict(dataset=name, run=run, method=method, size=frac, eps=eps, m=len(wc),
                    W1=W, clust_cost=cc, parity_viol=pv, auc=auc, dd=dd, secs=secs, rw=False)
        if extra:
            base.update(extra)
        rows.append(base)
        if method not in ("FC-POT", "FWC"):
            clf2 = train_mlp(Xc, Dc, Yc, kamiran_calders(Dc, Yc, wc), X[va], D[va], Y[va], seed=run)
            auc2, dd2 = downstream(clf2, X[te], D[te], Y[te])
            rows.append({**base, "auc": auc2, "dd": dd2, "rw": True})
        print(f"[{name} r{run}] {method:12s} size={frac:<6} eps={eps!s:5} m={len(wc):4d} "
              f"W1={W:8.3f} pv={pv:.3f} AUC={auc:.3f} DD={dd:.3f} ({secs:.1f}s)", flush=True)

    for frac in SIZES[name]:
        m = max(4, int(round(frac * len(tr))))
        rng = np.random.default_rng(1000 * run + int(frac * 1000))
        if "uniform" in methods:
            t = time.time(); core = m_uniform(Xtr, Dtr, Ytr, m, rng); record("Uniform", frac, None, core, time.time() - t)
        if "kmeans" in methods:
            t = time.time(); core = m_kmeans(Xtr, Dtr, Ytr, m, rng); record("K-Means", frac, None, core, time.time() - t)
        if "kmedoids" in methods:
            t = time.time(); core = m_kmedoids(Xtr, Dtr, Ytr, m, rng); record("K-Medoids", frac, None, core, time.time() - t)
        if "uniprot" in methods:
            t = time.time(); core = m_uniprot(Xtr, Dtr, Ytr, m, rng); record("UniPROT", frac, None, core, time.time() - t)
        fc_cache = {}
        for eps in EPS:
            budgets, feas = fair_budgets(Dtr, Ytr, m, eps, pT1)
            bkey = tuple(sorted(budgets.items()))
            ex = {"budget_feasible": feas}
            if "stratified" in methods:
                t = time.time(); core = m_stratified(Xtr, Dtr, Ytr, budgets, rng)
                record("Stratified", frac, eps, core, time.time() - t, ex)
            if "fcpot" in methods:
                t = time.time()
                if bkey not in fc_cache:          # same budgets => same selection problem
                    fc_cache[bkey] = (m_fcpot(Xtr, Dtr, Ytr, budgets, rng), time.time() - t)
                core, secs = fc_cache[bkey]
                record("FC-POT", frac, eps, core, secs, ex)
            if "fwc" in methods:
                t = time.time(); core = m_fwc(Xtr, Dtr, Ytr, m, eps, pT1, rng)
                record("FWC", frac, eps, core, time.time() - t)
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{name}_run{run}.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runs", default="0-9")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--methods", default="uniform,kmeans,kmedoids,uniprot,stratified,fcpot,fwc")
    ap.add_argument("--target_cap", type=int, default=3000)
    a = ap.parse_args()
    TARGET_CAP = a.target_cap
    lo, hi = (int(x) for x in a.runs.split("-")) if "-" in a.runs else (int(a.runs), int(a.runs))
    for r in range(lo, hi + 1):
        run_one(a.dataset, r, a.out, set(a.methods.split(",")))
