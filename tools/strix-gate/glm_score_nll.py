#!/usr/bin/env python3
"""GLM scored prefill gate: compares two teacher-forced score captures of the same prompt.

The GLM rank writes the last N prompt rows' logits when run with STRIX_GLM_PF_SCORE_LAST=N and
STRIX_GLM_PF_SCORE_OUT=<dir>/<name>-tf: one file per rank, <name>-tf-r0.f32 and <name>-tf-r1.f32, each N rows of
its vocabulary slice (rank 0 first). Row i sits at prompt position len - N + i. The offset-1 target (row i
predicts token len - N + i + 1) is the scored next-token rule; offset 0 is printed as a sanity line only.

  glm_score_nll.py --dir DIR --ref NAME --test NAME --prompt PROMPT.u32 [--rows 256] [--json OUT]

Prints NLL ref/test, dNLL (test - ref), mean KL(ref || test) and top-1 agreement over the scored rows.
"""
import argparse
import json
import sys

import numpy as np


def log_softmax(name, d, rows):
    parts = [np.fromfile(f"{d}/{name}-tf-r{k}.f32", dtype="<f4") for k in (0, 1)]
    for p in parts:
        if p.size % rows:
            raise SystemExit(f"{name}: capture size {p.size} is not a multiple of {rows} rows")
    x = np.concatenate([p.reshape(rows, -1).astype(np.float64) for p in parts], axis=1)
    x -= x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def compare(a, b, toks, rows):
    out = {}
    n = len(toks)
    if n < rows + 1:
        raise SystemExit(f"prompt has {n} tokens, fewer than scored rows + 1")
    for off in (0, 1):
        tgt = toks[n - rows + off: n - rows + off + rows - off].astype(np.int64)
        r = np.arange(len(tgt))
        na, nb = -a[r, tgt], -b[r, tgt]
        out[f"offset{off}"] = {"rows": int(len(tgt)), "nll_ref": float(na.mean()), "nll_test": float(nb.mean()),
                               "dnll": float(nb.mean() - na.mean())}
    pa = np.exp(a)
    kl = (pa * (a - b)).sum(axis=1)
    out["kl_mean"] = float(kl.mean())
    out["top1"] = float((a.argmax(1) == b.argmax(1)).mean())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--json")
    o = ap.parse_args(argv)
    toks = np.fromfile(o.prompt, dtype="<u4")
    res = compare(log_softmax(o.ref, o.dir, o.rows), log_softmax(o.test, o.dir, o.rows), toks, o.rows)
    res["prompt_tokens"] = int(len(toks))
    for off in (0, 1):
        r = res[f"offset{off}"]
        print(f"target offset {off}: rows {r['rows']} NLL ref {r['nll_ref']:.4f} test {r['nll_test']:.4f} dNLL {r['dnll']:+.4f}")
    print(f"KL mean {res['kl_mean']:.5f} top1 {res['top1'] * 100:.1f}% prompt tokens {res['prompt_tokens']}")
    if o.json:
        with open(o.json, "w", encoding="utf-8", newline="") as f:
            json.dump(res, f, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
