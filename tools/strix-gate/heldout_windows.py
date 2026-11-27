#!/usr/bin/env python3
"""Held-out gate windows (README "Held-out window set"): one shared, reusable text set for every engine's gate v2 run.

The set is data (heldout/<set>/spec.json): documents built from text already on disk that is not model output
(Wikipedia dump shards, system library sources, FAQ files), each with a content class (prose-en, prose-es, code, chat)
and a context band. Every model tokenizes the same documents with its own tokenizer and cuts the first N tokens of each
document as one window, so the scored rows (the window's last rows) are model-independent text.

  docs     --spec S --root DATA --out DIR      build DIR/<doc>.txt (prose, code) and DIR/<doc>.turns.json (chat) +
                                               DIR/sources.tsv (every input: file / shard row, title, sha256)
  render   --spec S --docs DIR --model M       chat documents rendered with the model's template (spec "templates")
  cut      --spec S --ids DIR --model M --band B --tokens N --out DIR2
                                               DIR/<doc>.<M>.ids (any separator) -> DIR2/<doc>-<N>.ids, DIR2/windows.tsv
                                               (NAME TOKENS IDS [ARMS]), DIR2/classes.json, DIR2/provenance.tsv
  prefilter --dumps D --manifest M --ref ARM --head-rows R [--max-greedy 0.95] --out M2
                                               reference-only pre-filter: drops every window whose labels equal the
                                               reference top-1 on more than --max-greedy of its scored rows (gate v2 R3
                                               needs held-out labels); writes M2 and prints the drop count. Never reads a
                                               candidate arm.
"""
import argparse
import gzip
import hashlib
import json
import pathlib
import re
import sys


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


def load_spec(path):
    spec = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    names = [d["id"] for d in spec["docs"]]
    if len(set(names)) != len(names):
        raise SystemExit("spec: duplicate document id")
    return spec


def read_text(path):
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


class Wiki:
    """Consumes qualifying articles of one shard in row order: every article feeds at most one document."""

    def __init__(self, path, start_row, min_chars, skip_prefixes):
        import pyarrow.parquet as pq  # only the docs step needs it
        self.name, self.table = path.name, pq.read_table(path, columns=["id", "title", "text"])
        self.row, self.min_chars, self.skip = start_row, min_chars, tuple(skip_prefixes)

    def take(self):
        while self.row < self.table.num_rows:
            r = self.row
            self.row += 1
            title, text = self.table["title"][r].as_py(), self.table["text"][r].as_py()
            if len(text) >= self.min_chars and not title.startswith(self.skip):
                return r, self.table["id"][r].as_py(), title, text
        raise SystemExit(f"{self.name}: out of qualifying articles")


def paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def faq_pairs(text):
    """Question blocks (ending in '?', or 'Q:' / '=== ' headings) and the blocks that answer them."""
    pairs, q, a = [], None, []
    for block in paragraphs(text):
        first = block.splitlines()[0].strip()
        if first.endswith("?") or first.startswith(("Q:", "Q.", "=== ")):
            if q and a:
                pairs.append((q, "\n\n".join(a)))
            q, a = block.lstrip("=Q:. ").strip(), []
        elif q:
            a.append(block)
    if q and a:
        pairs.append((q, "\n\n".join(a)))
    return pairs


def wiki_turns(title, text, lang, phrases):
    paras = paragraphs(text)
    turns, ask = [], phrases[lang]
    for i in range(0, len(paras), 3):
        q = ask["first"].format(title=title) if i == 0 else ask["more"].format(title=title)
        turns.append((q, "\n\n".join(paras[i:i + 3])))
    return turns


def build_docs(a):
    spec, root, out = load_spec(a.spec), pathlib.Path(a.root), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wikis = {k: Wiki(root / v["file"], v["start_row"], v["min_chars"], v.get("skip_title_prefixes", []))
             for k, v in spec["wiki"].items()}
    faq = []
    for f in spec.get("faq_files", []):
        faq.extend(faq_pairs(read_text(root / f)))
    sources = ["doc\tkind\tsource\trow\tid\ttitle\tsha256"]
    for d in spec["docs"]:
        target = int(d["chars"])
        if d["kind"] == "wiki":
            parts = []
            while sum(len(p) for p in parts) < target:
                r, wid, title, text = wikis[d["lang"]].take()
                parts.append(f"{title}\n\n{text.strip()}")
                sources.append(f"{d['id']}\twiki\t{wikis[d['lang']].name}\t{r}\t{wid}\t{title}\t{sha(text)}")
            (out / f"{d['id']}.txt").write_text("\n\n".join(parts) + "\n", encoding="utf-8", newline="\n")
        elif d["kind"] == "files":
            parts = []
            for f in d["files"]:
                text = read_text(root / f)
                parts.append(text)
                sources.append(f"{d['id']}\tfile\t{f}\t-\t-\t-\t{sha(text)}")
            body = "\n\n".join(parts)
            if len(body) < target:
                raise SystemExit(f"{d['id']}: {len(body)} chars < {target}")
            (out / f"{d['id']}.txt").write_text(body, encoding="utf-8", newline="\n")
        elif d["kind"] == "chat":
            turns, langs, n_faq = [], d["wiki_langs"], int(d.get("faq_pairs", 0))
            for q, ans in faq[:n_faq]:
                turns.append((q, ans))
            if n_faq:
                sources.append(f"{d['id']}\tfaq\t{','.join(spec['faq_files'])}\t0..{n_faq}\t-\t-\t{sha(json.dumps(faq[:n_faq]))}")
            faq = faq[n_faq:]
            k = 0
            while sum(len(q) + len(x) for q, x in turns) < target:
                lang = langs[k % len(langs)]
                k += 1
                r, wid, title, text = wikis[lang].take()
                turns.extend(wiki_turns(title, text, lang, spec["chat_phrases"]))
                sources.append(f"{d['id']}\twiki-chat\t{wikis[lang].name}\t{r}\t{wid}\t{title}\t{sha(text)}")
            (out / f"{d['id']}.turns.json").write_text(json.dumps(turns, ensure_ascii=False, indent=0), encoding="utf-8", newline="\n")
        else:
            raise SystemExit(f"{d['id']}: unknown kind {d['kind']}")
    (out / "sources.tsv").write_text("\n".join(sources) + "\n", encoding="utf-8", newline="\n")
    print(f"docs {len(spec['docs'])} sources {len(sources) - 1} -> {out}")


def render(a):
    spec, docs = load_spec(a.spec), pathlib.Path(a.docs)
    t = spec["templates"][a.model]
    for d in spec["docs"]:
        if d["kind"] != "chat":
            continue
        turns = json.loads((docs / f"{d['id']}.turns.json").read_text(encoding="utf-8"))
        text = t["begin"] + "".join(t["turn"].format(user=q, assistant=x) for q, x in turns)
        (docs / f"{d['id']}.{a.model}.txt").write_text(text, encoding="utf-8", newline="\n")
    print(f"rendered chat documents for {a.model}")


def cut(a):
    spec, ids_dir, out = load_spec(a.spec), pathlib.Path(a.ids), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = [d for d in spec["docs"] if a.band in d["bands"]]
    if a.only:
        keep = set(a.only.split(","))
        docs = [d for d in docs if d["id"] in keep]
    man, classes, prov = [], {}, ["name\tdoc\tclass\tlang\ttokens\tdoc_tokens\tids_sha256"]
    for d in docs:
        ids = re.findall(r"\d+", (ids_dir / f"{d['id']}.{a.model}.ids").read_text(encoding="utf-8"))
        if len(ids) < a.tokens:
            raise SystemExit(f"{d['id']}: {len(ids)} tokens < {a.tokens}")
        w = ids[:a.tokens]
        name = f"{d['id']}-{a.tokens}"
        (out / f"{name}.ids").write_text(" ".join(w) + "\n", encoding="utf-8", newline="\n")
        man.append(f"{name}\t{a.tokens}\t{(out / f'{name}.ids').resolve()}" + (f"\t{a.arms}" if a.arms else ""))
        classes[name] = d["class"]
        prov.append(f"{name}\t{d['id']}\t{d['class']}\t{d.get('lang', '-')}\t{a.tokens}\t{len(ids)}\t{sha(' '.join(w))}")
    (out / "windows.tsv").write_text("\n".join(man) + "\n", encoding="utf-8", newline="\n")
    (out / "classes.json").write_text(json.dumps(classes, indent=0) + "\n", encoding="utf-8", newline="\n")
    (out / "provenance.tsv").write_text("\n".join(prov) + "\n", encoding="utf-8", newline="\n")
    print(f"{a.model} band {a.band}: {len(man)} windows of {a.tokens} tokens -> {out}")


def prefilter(a):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from tools.qualification.option_qual.reduce import reduce_pair  # the gate's own head reader
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from gate_v2 import read_manifest
    manifest = read_manifest(a.manifest)
    lines = [x for x in pathlib.Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.split()]
    kept, report, dropped = [], [], 0
    for line in lines:
        case = line.split()[0]
        path = pathlib.Path(a.dumps) / f"{case}-{a.ref}.f32"
        if not path.exists():
            report.append(f"{case}\tmissing\t-")
            continue
        rows, _ = reduce_pair(path, path, "head", ids=manifest[case], rows=a.head_rows)
        greedy = float(rows["top1_r_is_label"].mean())
        drop = greedy > a.max_greedy
        dropped += drop
        report.append(f"{case}\t{'DROP' if drop else 'keep'}\t{greedy:.4f}\t{len(rows['label'])}")
        if not drop:
            kept.append(line)
    pathlib.Path(a.out).write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(report))
    print(f"prefilter {a.ref}: {len(lines)} windows, dropped {dropped} (label == reference top-1 > {a.max_greedy}), "
          f"kept {len(kept)}, missing {sum(r.split()[1] == 'missing' for r in report)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("docs")
    d.add_argument("--spec", required=True)
    d.add_argument("--root", required=True)
    d.add_argument("--out", required=True)
    r = sub.add_parser("render")
    r.add_argument("--spec", required=True)
    r.add_argument("--docs", required=True)
    r.add_argument("--model", required=True)
    c = sub.add_parser("cut")
    for k in ("--spec", "--ids", "--model", "--band", "--out"):
        c.add_argument(k, required=True)
    c.add_argument("--tokens", type=int, required=True)
    c.add_argument("--arms", default="", help="optional 4th manifest column (the quality tool's arm list)")
    c.add_argument("--only", default="", help="comma list of document ids")
    f = sub.add_parser("prefilter")
    for k in ("--dumps", "--manifest", "--ref", "--out"):
        f.add_argument(k, required=True)
    f.add_argument("--head-rows", type=int, default=64)
    f.add_argument("--max-greedy", type=float, default=0.95)
    a = p.parse_args(argv)
    {"docs": build_docs, "render": render, "cut": cut, "prefilter": prefilter}[a.cmd](a)


if __name__ == "__main__":
    main()
