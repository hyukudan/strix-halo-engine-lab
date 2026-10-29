"""Frozen corpus manifest: parents (statistical units), cases, SEARCH/HOLDOUT split.

usage:
  python3 -m tools.qualification.option_qual.manifest build --sources SOURCES.jsonl --out DIR \
      --tokenizer bin/strix-tokenize --model MODEL.gguf --model-receipt RECEIPT \
      --lengths 8192,32768 --cont 256 --stop-ids 248044,248046 --salt TEXT --search-per-stratum 4
  python3 -m tools.qualification.option_qual.manifest verify DIR

SOURCES.jsonl has one item per line: source_id, parent_id, stratum, lang, kind (text|chat),
license, origin, path (relative to the JSONL's directory). A text item is a UTF-8 file; a chat item is
JSON {"messages": [{"role": "system|user|assistant", "content": str}, ...]}. Items of a parent are
joined in source_id order. Every case of a parent shares the parent's split, and lengths of one
parent are nested prefixes of the same token stream.
"""
import argparse
from collections import defaultdict
import re
import json
from pathlib import Path

from .common import VERSION, atomic, canonical, file_sha, read_id_text, sha, write_id_text

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
# Every assistant message carries the empty reasoning block that the serving template puts before a
# non-thinking answer, so each scored continuation starts where a served answer starts.
ASSISTANT_HEAD = "<think>\n\n</think>\n\n"
MARKUP = re.compile(r"<\|[^\s<>|]{1,64}\|>|</?think>")
ROLES = ("system", "user", "assistant")
SPLITS = ("search", "holdout")


def load_sources(path):
    path = Path(path)
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        missing = {"source_id", "parent_id", "stratum", "lang", "kind", "license", "origin", "path"} - set(r)
        if missing or r["kind"] not in ("text", "chat"):
            raise ValueError(f"{path}:{n}: bad source record ({sorted(missing) or r['kind']})")
        r["file"] = (path.parent / r["path"]).resolve()
        items.append(r)
    ids = [r["source_id"] for r in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate source_id")
    parents = defaultdict(list)
    for r in items:
        parents[r["parent_id"]].append(r)
    for pid, group in parents.items():
        keys = {(r["stratum"], r["lang"], r["kind"]) for r in group}
        if len(keys) != 1:
            raise ValueError(f"parent {pid} mixes stratum/lang/kind {sorted(keys)}")
        group.sort(key=lambda r: r["source_id"])
    return dict(parents)


def render_chat(messages):
    out = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role not in ROLES or not isinstance(content, str):
            raise ValueError("chat message needs role system/user/assistant and string content")
        content = content.strip()
        if not content or MARKUP.search(content):
            raise ValueError("chat content is empty or contains special-token or reasoning markup")
        head = ASSISTANT_HEAD if role == "assistant" else ""
        out.append(f"{IM_START}{role}\n{head}{content}{IM_END}\n")
    return "".join(out)


def _find(seq, sub, start):
    n = len(sub)
    for i in range(start, len(seq) - n + 1):
        if seq[i:i + n] == sub:
            return i
    return -1


def assistant_spans(ids, header, end, expected):
    """Token spans [a, b) of assistant contents: after each assistant header, up to the end marker."""
    spans, i = [], 0
    while True:
        i = _find(ids, header, i)
        if i < 0:
            break
        a = i + len(header)
        b = _find(ids, end, a)
        if b < 0:
            raise ValueError("assistant content without end marker")
        spans.append((a, b))
        i = b
    if len(spans) != expected:
        raise ValueError(f"found {len(spans)} assistant spans, expected {expected} "
                         "(a content start merged with its header?)")
    return spans


class _Tok:
    def __init__(self, tokenizer):
        self.t = tokenizer
        self.plain = getattr(tokenizer, "plain", tokenizer.encode)
        self.encode = tokenizer.encode


def parent_stream(tok, group):
    """Token IDs of one parent plus, for chat, the assistant content spans."""
    kind = group[0]["kind"]
    if kind == "text":
        text = "\n\n".join(Path(r["file"]).read_text(encoding="utf-8") for r in group)
        return tok.plain(text), None
    msgs = []
    for r in group:
        msgs.extend(json.loads(Path(r["file"]).read_text(encoding="utf-8"))["messages"])
    ids = tok.encode(render_chat(msgs))
    header = tok.encode(f"{IM_START}assistant\n{ASSISTANT_HEAD}")
    return ids, assistant_spans(ids, header, tok.encode(IM_END), sum(m["role"] == "assistant" for m in msgs))


def cut_text(ids, length, cont):
    if len(ids) < length + cont:
        return None
    return length, ids[:length], ids[length:length + cont]


def cut_chat(ids, spans, length, cont, cont_min, min_fraction):
    """Prompt = everything before the last assistant content that starts at or before ``length``
    (and not before ``min_fraction * length``) and has at least ``cont_min`` tokens."""
    best = None
    for a, b in spans:
        if a <= length and b - a >= cont_min and a >= min_fraction * length:
            best = (a, b)
    if best is None:
        return None
    a, b = best
    return a, ids[:a], ids[a:min(b, a + cont)]


def assign_splits(parents, salt, search_per_stratum, eligible=None):
    """Parents without any case (``eligible`` excludes them) never take a SEARCH slot."""
    by_stratum = defaultdict(list)
    for pid, info in parents.items():
        if eligible is None or pid in eligible:
            by_stratum[info["stratum"]].append(pid)
    split = {pid: "unused" for pid in parents}
    for stratum, pids in by_stratum.items():
        order = sorted(pids, key=lambda p: sha(f"{salt}\0{p}".encode()))
        for k, pid in enumerate(order):
            split[pid] = "search" if k < search_per_stratum else "holdout"
    return split


def build(sources, out, tokenizer, lengths, cont=256, cont_min=128, stop_ids=(), salt="",
          search_per_stratum=4, min_fraction=0.75):
    if not salt:
        raise ValueError("a nonempty salt fixed before any result is required")
    out = Path(out)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    groups = load_sources(sources)
    tok = _Tok(tokenizer)
    stop = set(int(s) for s in stop_ids)
    parents, cases = {}, []
    for pid in sorted(groups):
        group = groups[pid]
        ids, spans = parent_stream(tok, group)
        g0 = group[0]
        parents[pid] = {"parent_id": pid, "stratum": g0["stratum"], "lang": g0["lang"], "kind": g0["kind"],
                        "tokens": len(ids), "source_ids": [r["source_id"] for r in group],
                        "source_sha256": [file_sha(r["file"]) for r in group],
                        "source_paths": [r["path"] for r in group],
                        "licenses": sorted({r["license"] for r in group}),
                        "origins": [r["origin"] for r in group]}
        for length in lengths:
            cut = cut_text(ids, length, cont) if spans is None else \
                cut_chat(ids, spans, length, cont, cont_min, min_fraction)
            if cut is None:
                continue
            prompt_len, prompt, continuation = cut
            if stop & set(continuation):
                raise ValueError(f"{pid} L{length}: continuation contains a stop ID")
            cid = f"{pid}.L{length}"
            pf, cf = out / "cases" / f"{cid}.prompt", out / "cases" / f"{cid}.cont"
            write_id_text(pf, prompt)
            write_id_text(cf, continuation)
            cases.append({"case_id": cid, "parent_id": pid, "stratum": g0["stratum"], "lang": g0["lang"],
                          "kind": g0["kind"], "nominal_length": length, "prompt_tokens": prompt_len,
                          "cont_tokens": len(continuation), "prompt_file": f"cases/{cid}.prompt",
                          "cont_file": f"cases/{cid}.cont", "prompt_sha256": file_sha(pf),
                          "cont_sha256": file_sha(cf)})
    split = assign_splits(parents, salt, search_per_stratum, {c["parent_id"] for c in cases})
    for pid in parents:
        parents[pid]["split"] = split[pid]
    for c in cases:
        c["split"] = split[c["parent_id"]]
    identity = getattr(tokenizer, "identity", {"backend": type(tokenizer).__name__})
    manifest = {"schema": 1, "kit": VERSION, "tokenizer": identity,
                "fake_tokenizer": identity.get("backend", "").startswith("fake"),
                "params": {"lengths": list(lengths), "cont": cont, "cont_min": cont_min,
                           "min_prompt_fraction": min_fraction, "stop_ids": sorted(stop), "salt": salt,
                           "search_per_stratum": search_per_stratum,
                           "sources_sha256": file_sha(sources)},
                "parents": [parents[p] for p in sorted(parents)], "cases": cases}
    data = canonical(manifest)
    atomic(out / "manifest.json", data)
    atomic(out / "manifest.sha256", (sha(data) + "\n").encode())
    return manifest, sha(data)


def load(directory):
    directory = Path(directory)
    data = (directory / "manifest.json").read_bytes()
    return json.loads(data), sha(data)


def verify(directory, sources_root=None, case_ids=None):
    """Re-hash the manifest and every case file (and source item when the root is given)."""
    directory = Path(directory)
    manifest, digest = load(directory)
    recorded = (directory / "manifest.sha256").read_text().strip()
    if digest != recorded:
        raise ValueError("manifest.json does not match manifest.sha256")
    wanted = set(case_ids) if case_ids is not None else None
    for c in manifest["cases"]:
        if wanted is not None and c["case_id"] not in wanted:
            continue
        for key in ("prompt", "cont"):
            if file_sha(directory / c[key + "_file"]) != c[key + "_sha256"]:
                raise ValueError(f"{c['case_id']}: {key} file changed")
        if len(read_id_text(directory / c["cont_file"])) != c["cont_tokens"]:
            raise ValueError(f"{c['case_id']}: continuation length changed")
    if sources_root is not None:
        for p in manifest["parents"]:
            for path, digest_ in zip(p["source_paths"], p["source_sha256"]):
                if file_sha(Path(sources_root) / path) != digest_:
                    raise ValueError(f"{p['parent_id']}: source {path} changed")
    return manifest, digest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sources", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--tokenizer")
    b.add_argument("--model")
    b.add_argument("--model-receipt")
    b.add_argument("--lengths", required=True)
    b.add_argument("--cont", type=int, default=256)
    b.add_argument("--cont-min", type=int, default=128)
    b.add_argument("--stop-ids", default="")
    b.add_argument("--salt", required=True)
    b.add_argument("--search-per-stratum", type=int, default=4)
    b.add_argument("--fake-tokenizer-for-tests", action="store_true", help=argparse.SUPPRESS)
    v = sub.add_parser("verify")
    v.add_argument("dir")
    v.add_argument("--sources-root")
    a = p.parse_args(argv)
    if a.cmd == "verify":
        m, digest = verify(a.dir, a.sources_root)
        print(json.dumps({"manifest_sha256": digest, "cases": len(m["cases"]), "parents": len(m["parents"])}))
        return
    if a.fake_tokenizer_for_tests:
        from tools.qualification.longctx_fidelity.tok import FakeTokenizer
        tokenizer = FakeTokenizer()
    else:
        if not (a.tokenizer and a.model and a.model_receipt):
            p.error("--tokenizer, --model and --model-receipt are required")
        from tools.qualification.longctx_fidelity.tok import StrixTokenize
        tokenizer = StrixTokenize(a.tokenizer, a.model, a.model_receipt)
    lengths = [int(x) for x in a.lengths.split(",") if x]
    stops = [int(x) for x in a.stop_ids.split(",") if x]
    m, digest = build(a.sources, a.out, tokenizer, lengths, a.cont, a.cont_min, stops, a.salt, a.search_per_stratum)
    counts = defaultdict(int)
    for c in m["cases"]:
        counts[f"{c['split']}/{c['stratum']}/L{c['nominal_length']}"] += 1
    print(json.dumps({"manifest_sha256": digest, "parents": len(m["parents"]), "cases": dict(sorted(counts.items()))}))


if __name__ == "__main__":
    main()
