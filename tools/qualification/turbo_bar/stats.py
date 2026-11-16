"""Paired statistics: CI = envelope of Student t and percentile bootstrap; PASS/FAIL/INCONCLUSIVE verdicts."""
import math

import numpy as np

from tools.qualification.option_qual.stats import t_quantile


def ci(d, level=0.95, B=10000, seed=0):
    """Two-sided CI of the mean: union (wider envelope) of the t interval and a percentile bootstrap."""
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    mean = float(d.mean()) if n else float("nan")
    if n < 2:
        return mean, float("-inf"), float("inf")
    se = float(d.std(ddof=1)) / math.sqrt(n)
    t = t_quantile(1 - (1 - level) / 2, n - 1)
    lo, hi = mean - t * se, mean + t * se
    rng = np.random.default_rng(seed)
    means = np.empty(B)
    step = max(1, 4_000_000 // n)
    for s in range(0, B, step):
        k = min(step, B - s)
        means[s:s + k] = d[rng.integers(0, n, size=(k, n))].mean(axis=1)
    a = (1 - level) / 2
    blo, bhi = np.quantile(means, [a, 1 - a])
    return mean, float(min(lo, blo)), float(max(hi, bhi))


def verdict(d, limit=0.0, level=0.95, B=10000, seed=0):
    """PASS iff CI_hi < limit; FAIL iff CI_lo > limit; otherwise INCONCLUSIVE (never PASS when the CI crosses)."""
    d = np.asarray(d, dtype=np.float64)
    n = int(d.size)
    out = {"n": n, "limit": limit}
    if n < 3 or not np.isfinite(d).all():
        out.update(mean=float(d.mean()) if n else None, lo=None, hi=None, verdict="INCONCLUSIVE",
                   n_needed="unbounded", message="enlarge sample")
        return out
    mean, lo, hi = ci(d, level, B, seed)
    out.update(mean=mean, lo=lo, hi=hi)
    if hi < limit:
        out.update(verdict="PASS", n_needed=None, message="CI upper bound below limit")
    elif lo > limit:
        out.update(verdict="FAIL", n_needed=None, message="CI lower bound above limit")
    else:
        gap = abs(limit - mean)
        need = "unbounded" if gap == 0 else max(n + 1, math.ceil(n * (((hi - lo) / 2) / gap) ** 2))
        out.update(verdict="INCONCLUSIVE", n_needed=need, message="enlarge sample")
    return out


def paired_binary(a, b, limit=0.0, **kw):
    """a, b: 0/1 success per item; d = a - b. Reports rates and discordant counts."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired arrays differ in length")
    out = verdict(a - b, limit, **kw)
    out.update(rate_a=float(a.mean()) if a.size else None, rate_b=float(b.mean()) if b.size else None,
               a_only=int(((a == 1) & (b == 0)).sum()), b_only=int(((a == 0) & (b == 1)).sum()),
               both=int(((a == 1) & (b == 1)).sum()), neither=int(((a == 0) & (b == 0)).sum()))
    return out


def t_ci(samples, level=0.95):
    x = np.asarray(samples, dtype=np.float64)
    n = x.size
    m = float(x.mean())
    if n < 2:
        return m, float("-inf"), float("inf")
    h = t_quantile(1 - (1 - level) / 2, n - 1) * float(x.std(ddof=1)) / math.sqrt(n)
    return m, m - h, m + h
