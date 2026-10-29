"""Cluster statistics: Student t quantiles (no SciPy) and a stratum-preserving cluster bootstrap."""
import math

import numpy as np


def _betacf(a, b, x):
    # Continued fraction of the regularized incomplete beta (modified Lentz).
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ArithmeticError("incomplete beta did not converge")


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_quantile(p, df):
    """Quantile of Student's t with ``df`` degrees of freedom (bisection on the CDF)."""
    if not 0 < p < 1 or df <= 0:
        raise ValueError("need 0 < p < 1 and df > 0")
    if p == 0.5:
        return 0.0
    lo, hi = -1.0, 1.0
    while t_cdf(lo, df) > p:
        lo *= 2
    while t_cdf(hi, df) < p:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


def resample_indices(strata, B, seed):
    """B x n parent indices; each resample keeps every stratum's parent count (with replacement)."""
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    n = strata.size
    out = np.empty((B, n), dtype=np.int64)
    col = 0
    for s in sorted(set(strata.tolist())):
        members = np.flatnonzero(strata == s)
        k = members.size
        out[:, col:col + k] = members[rng.integers(0, k, size=(B, k))]
        col += k
    return out


def upper_bound(values, strata, level, indices, weights=None):
    """One-sided upper bound at ``level`` for the mean of per-parent ``values``.

    The larger (more conservative) of the stratum-preserving cluster bootstrap quantile and the
    Student t bound with n - S degrees of freedom. ``weights="equal"``: the estimand is the mean of the
    stratum means (every stratum weighs the same whatever its parent count); the bootstrap weights each
    resampled parent by 1 / (S n_s) and the t bound uses the stratified standard error with
    Satterthwaite degrees of freedom.
    """
    if weights == "equal":
        return _upper_bound_equal(values, strata, level, indices)
    if weights is not None:
        raise ValueError(f"unknown stratum weighting {weights!r}")
    v = np.asarray(values, dtype=np.float64)
    n, S = v.size, len(set(np.asarray(strata).tolist()))
    mean = float(v.mean())
    boot = float(np.quantile(v[indices].mean(axis=1), level))
    df = n - S
    if df <= 0:
        # No residual degrees of freedom: no bound exists (never substitute a df).
        return {"estimate": mean, "ucb": float("inf"), "ucb_bootstrap": boot, "ucb_t": float("inf"), "df": df,
                "n_parents": n, "sd_parents": None}
    sd = float(v.std(ddof=1))
    tb = mean + t_quantile(level, df) * sd / math.sqrt(n)
    return {"estimate": mean, "ucb": max(boot, tb), "ucb_bootstrap": boot, "ucb_t": tb, "df": df,
            "n_parents": n, "sd_parents": sd}


def _upper_bound_equal(values, strata, level, indices):
    v = np.asarray(values, dtype=np.float64)
    st = np.asarray(strata)
    names = sorted(set(st.tolist()))
    S = len(names)
    groups = {s: v[st == s] for s in names}
    counts = {s: g.size for s, g in groups.items()}
    mean = float(np.mean([g.mean() for g in groups.values()]))
    # resample_indices lays out each resample stratum by stratum in sorted order: weight its columns alike
    colw = np.concatenate([np.full(counts[s], 1.0 / (S * counts[s])) for s in names])
    boot = float(np.quantile((v[indices] * colw).sum(axis=1), level))
    base = {"estimate": mean, "ucb_bootstrap": boot, "n_parents": int(v.size), "weights": "equal",
            "parents_per_stratum": counts}
    if any(c < 2 for c in counts.values()):
        # A stratum with one parent has no variance estimate: no bound exists.
        return dict(base, ucb=float("inf"), ucb_t=float("inf"), df=0, sd_parents=None)
    a = {s: float(groups[s].var(ddof=1)) / (S * S * counts[s]) for s in names}
    se = math.sqrt(sum(a.values()))
    denom = sum(a[s] ** 2 / (counts[s] - 1) for s in names)
    df = (sum(a.values()) ** 2 / denom) if denom > 0 else float(v.size - S)
    tb = mean + t_quantile(level, df) * se
    return dict(base, ucb=max(boot, tb), ucb_t=tb, df=df, sd_parents=None, se=se)
