"""Numerical sanity checks for the results of the compact FC-POT draft
(Lemmas 1-5, Proposition 1, Theorem 1, Remark 1).

Each check builds small random instances, computes the quantities in the
statement exactly (LP / brute force), and asserts the claimed relation.
Run:  python fcpot_sanity_checks.py
"""
import itertools
import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(0)
TOL = 1e-7


def _lp(K, row_rhs, col_rhs, row_eq, col_eq):
    m, n = K.shape
    A_row = np.kron(np.eye(m), np.ones(n))
    A_col = np.kron(np.ones(m), np.eye(n))
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    (A_eq if row_eq else A_ub).append(A_row); (b_eq if row_eq else b_ub).append(row_rhs)
    (A_eq if col_eq else A_ub).append(A_col); (b_eq if col_eq else b_ub).append(col_rhs)
    res = linprog(-K.ravel(),
                  A_ub=np.vstack(A_ub) if A_ub else None, b_ub=np.concatenate(b_ub) if b_ub else None,
                  A_eq=np.vstack(A_eq) if A_eq else None, b_eq=np.concatenate(b_eq) if b_eq else None,
                  bounds=(0, None), method="highs")
    assert res.status == 0, res.message
    return -res.fun, res.x.reshape(m, n)


def f_val(K, a, b, P):
    """POT surrogate (eq. f): rows <= a*1_P, columns <= b."""
    ind = np.zeros(len(a)); ind[list(P)] = 1
    return _lp(K, a * ind, b, False, False)


def h_val(K, a, nu, P):
    """Exact objective (eq. h): OT(a*1_P, a(P) nu)."""
    ind = np.zeros(len(a)); ind[list(P)] = 1
    if not P:
        return 0.0
    return _lp(K, a * ind, a[list(P)].sum() * nu, True, True)[0]


def fill(k, caps, mass, descending=True):
    order = np.argsort(-k if descending else k, kind="stable")
    v, left = np.zeros_like(k), mass
    for l in order:
        t = min(caps[l], left); v[l] = t; left -= t
        if left <= 1e-15:
            break
    return float(k @ v)


def inst(m, n):
    C = rng.random((m, n))
    K = (C.max() + 0.1) - C          # K = beta - C > 0
    return K


def subsets(m):
    for r in range(m + 1):
        yield from (frozenset(S) for S in itertools.combinations(range(m), r))


# ---------------------------------------------------------------------------
def check_lemma1(trials=15, m=6, n=5):
    """h: nonneg, h(empty)=0, monotone, super-additive (general capacities)."""
    for _ in range(trials):
        K = inst(m, n); a = rng.uniform(0.2, 1.5, m); nu = rng.dirichlet(np.ones(n))
        H = {S: h_val(K, a, nu, S) for S in subsets(m)}
        assert H[frozenset()] == 0
        for A in H:
            assert H[A] >= -TOL
            for B in H:
                if A <= B:
                    assert H[A] <= H[B] + TOL
                if not (A & B):
                    assert H[A | B] >= H[A] + H[B] - 1e-6
    return "Lemma 1 (h nonneg, monotone, super-additive; general a): OK"


def check_lemma2(trials=15, m=6, n=5):
    """f: normalized, monotone, submodular for arbitrary a, B nu; saturation."""
    worst = np.inf
    for _ in range(trials):
        K = inst(m, n); a = rng.uniform(0.2, 1.5, m)
        nu = rng.dirichlet(np.ones(n)); B = rng.uniform(0.5, 6.0)
        Fv = {S: f_val(K, a, B * nu, S)[0] for S in subsets(m)}
        assert abs(Fv[frozenset()]) < TOL
        for A in Fv:
            for C in Fv:
                if A <= C:
                    assert Fv[A] <= Fv[C] + TOL
                gap = Fv[A] + Fv[C] - Fv[A | C] - Fv[A & C]
                worst = min(worst, gap); assert gap >= -1e-6
            if a[list(A)].sum() <= B and A:             # (iv) saturation
                _, G = f_val(K, a, B * nu, A)
                assert np.allclose(G.sum(1)[list(A)], a[list(A)], atol=1e-7)
    return f"Lemma 2 (f normalized, monotone, submodular, saturating): OK; min gap {worst:.1e}"


def check_lemma3(trials=40, m=7, n=6):
    """f >= h when a(P) <= B; f = h when a(P) = B (general a)."""
    for _ in range(trials):
        K = inst(m, n); a = rng.uniform(0.2, 1.5, m); nu = rng.dirichlet(np.ones(n))
        P = list(rng.choice(m, size=rng.integers(1, m), replace=False))
        aP = a[P].sum()
        B = aP * rng.uniform(1.0, 2.0)
        assert f_val(K, a, B * nu, P)[0] >= h_val(K, a, nu, P) - 1e-7
        assert abs(f_val(K, a, aP * nu, P)[0] - h_val(K, a, nu, P)) < 1e-7
    return "Lemma 3 (f >= h; f = h at a(P) = B, general a): OK"


def greedy(K, kappa, nu, approx, cands=None):
    m = K.shape[0]; a = np.ones(m); b = kappa * nu
    cands = list(range(m)) if cands is None else list(cands)
    P = []
    for _ in range(kappa):
        val, G = f_val(K, a, b, P); r = b - G.sum(0)
        rest = [j for j in cands if j not in P]
        score = {j: (fill(K[j], r, 1.0) if approx else f_val(K, a, b, P + [j])[0] - val) for j in rest}
        P.append(max(rest, key=score.get))
    return P


def check_lemmas4_5(trials=15, m=8, n=6, kappa=3):
    worst = {False: np.inf, True: np.inf}
    for _ in range(trials):
        K = inst(m, n); nu = rng.dirichlet(np.ones(n)); a = np.ones(m)
        alpha = min(fill(K[j], kappa * nu, 1.0, False) / fill(K[j], kappa * nu, 1.0, True) for j in range(m))
        OPT = max(h_val(K, a, nu, list(S)) for S in itertools.combinations(range(m), kappa))
        # approximate-gain sandwich (Lemma 5, S2-S5) at random P
        P = list(rng.choice(m, size=rng.integers(0, kappa), replace=False))
        val, G = f_val(K, a, kappa * nu, P); r = kappa * nu - G.sum(0)
        for j in set(range(m)) - set(P):
            gain = f_val(K, a, kappa * nu, P + [j])[0] - val
            fh = fill(K[j], r, 1.0)
            assert alpha * gain - 1e-7 <= fh <= gain + 1e-7
        for approx in (False, True):
            Phat = greedy(K, kappa, nu, approx)
            assert len(Phat) == kappa
            fP = f_val(K, a, kappa * nu, Phat)[0]
            assert abs(fP - h_val(K, a, nu, Phat)) < 1e-7          # f = h at |P| = kappa
            bound = (1 - np.exp(-alpha)) if approx else (1 - 1 / np.e)
            assert fP >= bound * OPT - 1e-7
            worst[approx] = min(worst[approx], fP / OPT)
    return (f"Lemma 4 (1-1/e) and Lemma 5 (1-e^-alpha, sandwich): OK; "
            f"worst h(P)/OPT exact {worst[False]:.3f}, approx {worst[True]:.3f}")


def check_theorem1(trials=6):
    """Strata (D,Y); FWC parity budgets; per-stratum greedy is exactly fair and
    within (1-e^-alpha_min) of the best stratified fair coreset."""
    eps = 0.1
    for _ in range(trials):
        m, kappa = 12, 4
        D = rng.integers(0, 2, m); Y = rng.integers(0, 2, m)
        strata = sorted(set(zip(D, Y)))
        pT = {y: (Y == y).mean() for y in (0, 1)}
        # feasible budgets satisfying FWC rows (eq. fwc-dp) by brute force
        sizes = {s: int(sum((D == s[0]) & (Y == s[1]))) for s in strata}
        feas = []
        for kv in itertools.product(*[range(sizes[s] + 1) for s in strata]):
            if sum(kv) != kappa:
                continue
            ks = dict(zip(strata, kv)); ok = True
            for d in (0, 1):
                tot = sum(ks.get((d, y), 0) for y in (0, 1))
                for y in (0, 1):
                    c = ks.get((d, y), 0)
                    if tot and not ((1 - eps) * pT[y] * tot - 1e-12 <= c <= (1 + eps) * pT[y] * tot + 1e-12):
                        ok = False
            if ok:
                feas.append(ks)
        if not feas:
            continue
        ks = feas[0]
        X = rng.random((m, 3))
        total_hat, total_opt = 0.0, 0.0
        Phat = []
        alpha_min = 1.0
        for s in strata:
            idx = np.where((D == s[0]) & (Y == s[1]))[0]
            k = ks[s]
            if k == 0:
                continue
            C = np.abs(X[idx][:, None, :] - X[idx][None, :, :]).sum(-1)
            Ks = (C.max() + 0.1) - C
            nu = np.ones(len(idx)) / len(idx)
            alpha_min = min(alpha_min, min(fill(Ks[j], k * nu, 1.0, False) / fill(Ks[j], k * nu, 1.0, True)
                                           for j in range(len(idx))))
            loc = greedy(Ks, k, nu, approx=True)
            Phat += list(idx[loc])
            total_hat += h_val(Ks, np.ones(len(idx)), nu, loc)
            total_opt += max(h_val(Ks, np.ones(len(idx)), nu, list(S))
                             for S in itertools.combinations(range(len(idx)), k))
        # (i) exact fairness of the returned set
        for d in (0, 1):
            tot = sum(1 for i in Phat if D[i] == d)
            for y in (0, 1):
                c = sum(1 for i in Phat if D[i] == d and Y[i] == y)
                if tot:
                    assert (1 - eps) * pT[y] * tot - 1e-9 <= c <= (1 + eps) * pT[y] * tot + 1e-9
        # (ii) guarantee
        assert total_hat >= (1 - np.exp(-alpha_min)) * total_opt - 1e-7
    return "Theorem 1 (stratified FWC parity: exactly fair, (1-e^-alpha_min) guarantee): OK"


def check_remark_free_weights():
    K = np.ones((2, 2)); b = np.ones(2)

    def free_val(P):
        cap = [1.0 if i in P else 0.0 for i in range(2)]
        c = np.concatenate([-K.ravel(), np.zeros(2)])
        A_eq = np.array([[1, 1, 0, 0, -1, 0], [0, 0, 1, 1, 0, -1], [0, 0, 0, 0, 1, -1]], float)
        A_ub = np.array([[1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0]], float)
        res = linprog(c, A_ub=A_ub, b_ub=b, A_eq=A_eq, b_eq=np.zeros(3),
                      bounds=[(0, None)] * 4 + [(0, cap[0]), (0, cap[1])], method="highs")
        return -res.fun
    v = {P: free_val(P) for P in [(), (0,), (1,), (0, 1)]}
    assert abs(v[(0,)]) < TOL and abs(v[(1,)]) < TOL and abs(v[(0, 1)] - 2) < TOL
    assert abs(f_val(K, np.ones(2), b, [0])[0] - 1) < TOL and abs(f_val(K, np.ones(2), b, [0, 1])[0] - 2) < TOL
    return "Remark 1 (free-weight coupling breaks submodularity; f does not): OK"


if __name__ == "__main__":
    for chk in (check_lemma1, check_lemma2, check_lemma3, check_lemmas4_5,
                check_theorem1, check_remark_free_weights):
        print(chk())
