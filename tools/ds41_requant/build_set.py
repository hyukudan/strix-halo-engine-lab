#!/usr/bin/env python3
"""Builds the requantization gate's token sequences with the model's tokenizer (BOS 0 before every segment).

  eval.u32 / eval.segs        prose-en, prose-es, code, math: 448 tokens per document (scored teacher-forced)
  long.u32 / long.segs        one 8192-token stream of other documents (long-context buckets by position)
  calib.u32 / calib.segs      calibration (imatrix) documents, disjoint from both
  greedy.prompts              "name gen ids" lines for --greedy (the reference builds greedy.u32 / greedy.segs)
usage: build_set.py TOKENIZER.json SET_DIR OUT_DIR
"""
import json, os, sys
import numpy as np
from tokenizers import Tokenizer

EVAL = [("prose-en", "en-carbonate-platform"), ("prose-en", "en-earth-caught-fire"), ("prose-es", "es-viento-metal"),
        ("prose-es", "es-consejo-aragon"), ("code", "code-py-difflib"), ("code", "code-cpp-verify"), ("code", "code-hip-qgemv"),
        ("math", "math-en"), ("math", "math-es")]
LONG = ["de-bosch", "fr-lyon-airport", "chat-en-article", "it-pompa", "chat-es-article", "pt-golfinho", "chat-code-fix",
        "ja-edo-bakufu", "zh-alcoholism"]
CALIB = ["en-b25-mitchell", "es-utopia", "code-rust-ahocorasick", "code-py-campaign", "chat-es-explain", "ru-scott"]
BOS = 0


def main():
    tok = Tokenizer.from_file(sys.argv[1])
    src, out = sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    text = lambda n: open(os.path.join(src, n + ".txt"), encoding="utf-8").read()
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids

    def write(name, items):  # items: (category, doc, ids)
        seq, segs = [], []
        for cat, doc, ids in items:
            segs.append(f"{cat} {doc} {len(seq)} {len(ids) + 1}")
            seq += [BOS] + ids
        np.array(seq, dtype="<u4").tofile(os.path.join(out, name + ".u32"))
        open(os.path.join(out, name + ".segs"), "w").write("\n".join(segs) + "\n")
        print(name, len(seq), "tokens")

    write("eval", [(c, d, enc(text(d))[:447]) for c, d in EVAL])
    items, n = [], 0
    for d in LONG:
        ids = enc(text(d))[: 8192 - n - 1]
        items.append(("long", d, ids))
        n += len(ids) + 1
        if n >= 8192:
            break
    write("long", items)
    write("calib", [("calib", d, enc(text(d))[:127]) for d in CALIB])
    prompts = json.load(open(os.path.join(src, "greedy-prompts.json"), encoding="utf-8"))
    with open(os.path.join(out, "greedy.prompts"), "w") as f:
        for name, p in prompts:
            f.write(f"{name} 64 {','.join(str(i) for i in [BOS] + enc(p))}\n")
    print("greedy", len(prompts), "prompts")


if __name__ == "__main__":
    main()
