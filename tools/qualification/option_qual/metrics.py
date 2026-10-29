"""Per-position full-vocabulary comparison of a reference row and a candidate row.

All arithmetic is float64. ``overlap = sum_v min(p_r, p_c)`` equals ``1 - TV``: the largest
probability that the two models emit the same token at this shared history under any coupling.
It is not greedy equality and says nothing about histories that were not scored.
"""
import numpy as np

FIELDS = ("overlap", "tv", "kl_rc", "kl_cr", "agree", "gap_r", "nll_r", "nll_c",
          "top1_r", "top1_c", "top1_r_is_label", "top1_c_is_label", "nonfinite_r", "nonfinite_c")
SELF_FIELDS = ("nll", "top1", "is_label", "nonfinite")
NEG_TOLERANCE = 1e-12


def log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    m = z.max()
    shifted = z - m
    return shifted - np.log(np.exp(shifted).sum())


def _kl(lp, lq, p):
    # p * (log p - log q); terms with p == 0 contribute exactly 0.
    terms = np.where(p > 0, p * (lp - lq), 0.0)
    value = float(terms.sum())
    if value < 0:
        if value < -NEG_TOLERANCE:
            raise ArithmeticError(f"negative KL {value}")
        value = 0.0
    return value


def top_two_gap(lp):
    if lp.size < 2:
        return float("inf")
    a, b = np.partition(lp, lp.size - 2)[-2:]
    return float(max(a, b) - min(a, b))


def compare(z_r, z_c, label):
    """Metrics of one shared position; ``label`` is the token the row is scored against."""
    z_r = np.asarray(z_r)
    z_c = np.asarray(z_c)
    if z_r.shape != z_c.shape or not 0 <= label < z_r.size:
        raise ValueError("row shapes differ or label outside vocabulary")
    bad_r = not np.isfinite(z_r).all()
    bad_c = not np.isfinite(z_c).all()
    out = dict.fromkeys(FIELDS, float("nan"))
    out.update(nonfinite_r=int(bad_r), nonfinite_c=int(bad_c), agree=0, top1_r=-1, top1_c=-1,
               top1_r_is_label=0, top1_c_is_label=0)
    if bad_r or bad_c:
        return out
    lp, lq = log_softmax(z_r), log_softmax(z_c)
    p, q = np.exp(lp), np.exp(lq)
    t_r, t_c = int(np.argmax(z_r)), int(np.argmax(z_c))  # lowest ID on ties
    overlap = float(np.minimum(p, q).sum())
    out.update(overlap=overlap, tv=1.0 - overlap, kl_rc=_kl(lp, lq, p), kl_cr=_kl(lq, lp, q),
               top1_r_is_label=int(t_r == label), top1_c_is_label=int(t_c == label),
               agree=int(t_r == t_c), gap_r=top_two_gap(lp), nll_r=float(-lp[label]),
               nll_c=float(-lq[label]), top1_r=t_r, top1_c=t_c)
    return out


def self_metrics(z, label):
    z = np.asarray(z)
    if not np.isfinite(z).all():
        return dict(nll=float("nan"), top1=-1, is_label=0, nonfinite=1)
    lp = log_softmax(z)
    t = int(np.argmax(z))
    return dict(nll=float(-lp[label]), top1=t, is_label=int(t == label), nonfinite=0)
