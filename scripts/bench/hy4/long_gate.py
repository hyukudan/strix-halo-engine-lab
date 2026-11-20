#!/usr/bin/env python3
"""Hy4-flash long-prompt FAST gate v2 (prompt windows past what one ticket can reference). Kit: long_gate.sh.

The reference is BASE (strix-hy4-rank --gate-ref: verify rows, each row bitwise the position-by-position step) over a
whole teacher-forced document, built in pieces across tickets (state images). Each cut N scores 200 held-out rows: the
logits at positions N-1 .. N+198 against the document's next ids.
  candidate: --prompt-rows windows over [0, N), then 199 teacher-forced steps, routes REPLAYED from the reference
  null:      the reference form from the reference's image at N - 1024, routes replayed + a one-ulp nudge (shorter than
             the candidate's N + 199 rows, so its KL / flips can only be smaller: the limits it sets are stricter)
Statistics: tools/strix-gate/gate_v2.py (window_stats / verdict; one implementation for every engine). Flips: the
shared route tap's local flips (the arm's own ids vs the replayed ones) of the candidate over the null's.

  long_gate.py routes-pos OUT.npz FILE.rte:START [...]          reference route files -> per (layer, position) table
  long_gate.py replay-arm POS.npz OUT.rte N ROWS STEPS           candidate's call schedule (shared window rule)
  long_gate.py replay-chunks POS.npz OUT.rte P0 P1 R             reference-form schedule [P0, P1) in R-row chunks
  long_gate.py gate DIR MANIFEST [--gate-v2 PATH] [--json F]     verdict; MANIFEST lines: NAME CLASS DOC N
Files under DIR: DOC.ids (document), ref-DOC-*.f32 + .pos (reference rows), arm-DOC-N.f32 / .err, null-DOC-N.f32 / .err.
"""
import argparse
import importlib.util
import json
import pathlib
import re
import sys

import numpy as np

MAGIC = b"STXROUT1"
RECEIPT = re.compile(rb"^strix-route-file/v1 (\{.*\})\s*$", re.M)
ALIGN, MIN_ROWS = 8, 16   # src/hy4/hip/hy4_hip_plan.hpp kPromptRule (strix/session/prompt_windows.hpp)


def read_routes(path):
    data = pathlib.Path(path).read_bytes()
    if data[:8] != MAGIC or int.from_bytes(data[8:12], "little") != 1:
        raise SystemExit(f"{path}: not a strix-route-file/v1")
    at, out = 16, []
    while at < len(data):
        layer, rows, k, _ = np.frombuffer(data, "<u4", 4, at).tolist()
        n = rows * k
        ids = np.frombuffer(data, "<u4", n, at + 16).reshape(rows, k)
        w = np.frombuffer(data, "<f4", n, at + 16 + 4 * n).reshape(rows, k)
        out.append((layer, ids, w))
        at += 16 + 8 * n
    return out


def routes_pos(a):
    """Per layer, rows are appended in position order by any causal schedule (each call of a layer covers the next rows)."""
    per = {}
    starts = []
    for spec in a.files:
        path, start = spec.rsplit(":", 1)
        start = int(start)
        starts.append(start)
        cursor = {}
        for layer, ids, w in read_routes(path):
            c = cursor.get(layer, start)
            d = per.setdefault(layer, {})
            for r in range(ids.shape[0]):
                d[c + r] = (ids[r], w[r])
            cursor[layer] = c + ids.shape[0]
        ends = set(cursor.values())
        if len(ends) != 1:
            raise SystemExit(f"{path}: layers end at different positions {sorted(ends)}")
        print(f"{path}: positions {start}..{ends.pop()} layers {len(cursor)}")
    layers = sorted(per)
    lo = min(min(d) for d in per.values())
    hi = max(max(d) for d in per.values()) + 1
    k = next(iter(per[layers[0]].values()))[0].size
    ids = np.zeros((len(layers), hi - lo, k), np.uint32)
    w = np.zeros((len(layers), hi - lo, k), np.float32)
    have = np.zeros((len(layers), hi - lo), bool)
    for li, layer in enumerate(layers):
        for p, (i, x) in per[layer].items():
            ids[li, p - lo], w[li, p - lo], have[li, p - lo] = i, x, True
    np.savez(a.out, layers=np.asarray(layers, np.uint32), lo=lo, ids=ids, w=w, have=have)
    print(f"routes-pos {a.out}: layers {len(layers)} positions {lo}..{hi} top_k {k} complete {bool(have.all())}")


def write_schedule(pos_path, out, segments):
    z = np.load(pos_path)
    layers, lo, ids, w, have = z["layers"], int(z["lo"]), z["ids"], z["w"], z["have"]
    k = ids.shape[2]
    blob = bytearray(MAGIC + (1).to_bytes(4, "little") + (0).to_bytes(4, "little"))
    for p0, rows in segments:
        a, b = p0 - lo, p0 - lo + rows
        if a < 0 or b > ids.shape[1] or not have[:, a:b].all():
            raise SystemExit(f"positions {p0}..{p0 + rows} are not all in the reference routes")
        for li, layer in enumerate(layers.tolist()):
            blob += np.asarray([layer, rows, k, 0], "<u4").tobytes()
            blob += ids[li, a:b].astype("<u4").tobytes() + w[li, a:b].astype("<f4").tobytes()
    pathlib.Path(out).write_bytes(bytes(blob))
    print(f"{out}: {len(segments)} segments x {len(layers)} layers, positions {segments[0][0]}..{segments[-1][0] + segments[-1][1]}")


def window_plan(pos, count, rmax):
    """strix::session::prompt_windows::plan with the Hy4 rule {align 8, min rows 16}: (lead steps, windows, tail)."""
    if not (rmax >= MIN_ROWS and rmax % ALIGN == 0):
        return count, [], 0
    lead = min((ALIGN - pos % ALIGN) % ALIGN, count)
    at, left, wins = pos + lead, count - lead, []
    while left >= MIN_ROWS:
        rows = min(left // ALIGN * ALIGN, rmax)
        wins.append((at, rows))
        at += rows
        left -= rows
    return lead, wins, left


def replay_arm(a):
    lead, wins, tail = window_plan(0, a.n, a.rows)
    seg = [(p, 1) for p in range(lead)] + wins
    end = wins[-1][0] + wins[-1][1] if wins else lead
    seg += [(p, 1) for p in range(end, end + tail)]
    seg += [(a.n + s, 1) for s in range(a.steps)]
    write_schedule(a.pos, a.out, seg)


def replay_chunks(a):
    seg, p = [], a.p0
    while p < a.p1:
        r = min(a.r, a.p1 - p)
        seg.append((p, r))
        p += r
    write_schedule(a.pos, a.out, seg)


def load_gate_v2(path):
    spec = importlib.util.spec_from_file_location("gate_v2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows_of(path, positions=None):
    data = np.fromfile(path, np.float32)
    pos = None
    pp = pathlib.Path(str(path) + ".pos")
    if pp.exists():
        pos = np.loadtxt(pp, dtype=np.int64, ndmin=1)
    if positions is not None and pos is None:
        pos = np.asarray(positions, np.int64)
    n = pos.size
    if data.size % n:
        raise SystemExit(f"{path}: {data.size} floats is not {n} rows")
    return pos, data.reshape(n, data.size // n)


def compare(ref, cand, label):
    r = ref.astype(np.float64)
    c = cand.astype(np.float64)
    lr = r - r.max() - np.log(np.exp(r - r.max()).sum())
    lc = c - c.max() - np.log(np.exp(c - c.max()).sum())
    kl = float((np.exp(lr) * (lr - lc)).sum())
    return kl, int(ref.argmax()) == int(cand.argmax()), -lr[label], -lc[label], int(ref.argmax()) == label, int(cand.argmax()) == label


def receipt(path):
    m = RECEIPT.findall(pathlib.Path(path).read_bytes())
    if not m:
        raise SystemExit(f"{path}: no strix-route-file/v1 receipt")
    j = json.loads(m[-1])
    if not j.get("in", {}).get("complete"):
        raise SystemExit(f"{path}: replay incomplete {j.get('in')}")
    return j["local_flips"]


def gate(a):
    g = load_gate_v2(a.gate_v2)
    d = pathlib.Path(a.dir)
    ref_rows = {}
    for f in sorted(d.glob("ref-*.f32")):
        doc = f.name.split("-")[1]
        pos, x = rows_of(f)
        for p, row in zip(pos.tolist(), x):
            ref_rows[(doc, p)] = row
    windows, flips, lines = [], [], []
    for line in pathlib.Path(a.manifest).read_text().split("\n"):
        if not line.strip() or line.startswith("#"):
            continue
        name, cls, doc, n = line.split()[:4]
        n = int(n)
        ids = np.loadtxt(d / f"{doc}.ids", dtype=np.int64, ndmin=1)
        want = list(range(n - 1, n + 199))
        _, arm = rows_of(d / f"arm-{doc}-{n}.f32", want)
        npos, null = rows_of(d / f"null-{doc}-{n}.f32")
        if npos.tolist() != want or arm.shape[0] != len(want):
            raise SystemExit(f"{name}: arm / null rows are not positions {want[0]}..{want[-1]}")
        st = {k: [] for k in ("kl", "agree", "nr", "nc", "hr", "hc", "nkl", "nagree", "nn")}
        for i, p in enumerate(want):
            ref = ref_rows[(doc, p)]
            lab = int(ids[p + 1])
            kl, ag, nr, nc, hr, hc = compare(ref, arm[i], lab)
            nkl, nag, _, nn, _, _ = compare(ref, null[i], lab)
            for k, v in zip(st, (kl, ag, nr, nc, hr, hc, nkl, nag, nn)):
                st[k].append(v)
        w = g.window_stats(name, st["kl"], st["agree"], st["nr"], st["nc"], [float(np.mean(st["nkl"]))],
                           np.asarray(st["hr"], bool), np.asarray(st["hc"], bool), null_top1s=[float(np.mean(st["nagree"]))],
                           null_nlls=[np.asarray(st["nn"])], cls=cls)
        fa, fn = receipt(d / f"arm-{doc}-{n}.err"), receipt(d / f"null-{doc}-{n}.err")
        f = {"cand": {"slots": fa["slots"], "ids": fa["ids"], "rate": fa["ids"] / fa["slots"]},
             "null": {"slots": fn["slots"], "ids": fn["ids"], "rate": fn["ids"] / fn["slots"]}}
        w["flips"] = f
        windows.append(w)
        flips.append(f)
        lines.append(g.window_line(w, f))
    for x in lines:
        print(x)
    # R1: the shared flip-ratio check is defined on FREE-running arms against a null of the same length. Here both arms
    # replay (local flips: the arm's own ids vs the replayed ones) and the null is shorter, so the ratio is reported per
    # window and judged by --r1 (default: report only; the R1 evidence is the ratio's stationarity against the cuts
    # inside the range a free-running ruling already qualified).
    print("local flips (arm / null, rate): " + "; ".join(
        f"{w['name']} {f['cand']['rate']:.4f}/{f['null']['rate']:.4f}={f['cand']['rate'] / max(f['null']['rate'], 1e-12):.2f}"
        for w, f in zip(windows, flips)))
    try:
        pooled = g.verdict(windows, flips if a.r1 else None)
    except g.GateError as e:
        print(f"no verdict: {e}")
        return 2
    print(g.pooled_line(pooled))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"pooled": {k: v for k, v in pooled.items()},
                                                    "windows": [{k: v for k, v in w.items() if k != "_rows"} for w in windows]},
                                                   default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return g.EXIT[pooled["state"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("routes-pos")
    s.add_argument("out")
    s.add_argument("files", nargs="+")
    s = sub.add_parser("replay-arm")
    s.add_argument("pos")
    s.add_argument("out")
    s.add_argument("n", type=int)
    s.add_argument("rows", type=int)
    s.add_argument("steps", type=int)
    s = sub.add_parser("replay-chunks")
    s.add_argument("pos")
    s.add_argument("out")
    s.add_argument("p0", type=int)
    s.add_argument("p1", type=int)
    s.add_argument("r", type=int)
    s = sub.add_parser("gate")
    s.add_argument("dir")
    s.add_argument("manifest")
    s.add_argument("--gate-v2", default=str(pathlib.Path(__file__).resolve().parents[3] / "tools/strix-gate/gate_v2.py"))
    s.add_argument("--json")
    s.add_argument("--r1", action="store_true", help="apply the shared flip-ratio limit to the local flips")
    a = ap.parse_args(argv)
    return {"routes-pos": routes_pos, "replay-arm": replay_arm, "replay-chunks": replay_chunks, "gate": gate}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
