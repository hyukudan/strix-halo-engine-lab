#!/usr/bin/env python3
"""Pool full-vocabulary teacher receipts, preserving per-language arm gates."""
import argparse
import json
import math
from pathlib import Path

LANGUAGES = ("en", "es", "fr", "de", "it", "pt", "zh", "ja")


def quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no full-vocabulary rows")
    position = (len(ordered) - 1) * probability
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def load(paths):
    docs = {}
    identities = set()
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.startswith("{"):
                continue
            record = json.loads(line)
            if "model_hash" in record:
                identities.add((record["identity"], record["model_hash"], record["rope_hash"], record["prefill_identity"]))
            if record.get("metric") != "quality":
                continue
            key = record["parent"]
            if key in docs:
                # Resuming a completed document may change only its wall time.
                a = {k: v for k, v in record.items() if k != "seconds"}
                b = {k: v for k, v in docs[key].items() if k != "seconds"}
                if a != b:
                    raise ValueError("conflicting repeated parent")
            docs[key] = record
    if len(identities) != 1 or len(docs) != 40:
        raise ValueError("NO-REFERENCE: requires one frozen identity and all 40 parents")
    for language in LANGUAGES:
        if sum(d["language"] == language for d in docs.values()) != 5:
            raise ValueError("NO-REFERENCE: missing language parents")
    for record in docs.values():
        if (not 0 <= record["matches"] <= record["rows"] or record["rows"] != record["nll_positions"] + 1
                or not record["kl_full"] or not record["teacher_sha256"]
                or not all(math.isfinite(x) and x >= -1e-12 for x in record["kl_full"])
                or not math.isfinite(record["sum_dnll"])):
            raise ValueError("invalid quality receipt")
    return docs, next(iter(identities))


def pool(docs):
    values = list(docs)
    kl = [value for d in values for value in d["kl_full"]]
    rows = sum(d["rows"] for d in values)
    nll = sum(d["nll_positions"] for d in values)
    return dict(parents=len(values), rows=rows, nll_positions=nll, full_vocab_rows=len(kl),
                top1=sum(d["matches"] for d in values) / rows,
                mean_kl=math.fsum(kl) / len(kl), p99_kl=quantile(kl, 0.99),
                dnll=math.fsum(d["sum_dnll"] for d in values) / nll)


def summarize(docs):
    out = {lang: pool(d for d in docs.values() if d["language"] == lang) for lang in LANGUAGES}
    out["pooled"] = pool(docs.values())
    return out


def compare(reference, candidate):
    if reference.keys() != candidate.keys():
        raise ValueError("parent sets differ")
    for key in reference:
        for field in ("teacher_sha256", "language", "rows", "nll_positions"):
            if reference[key][field] != candidate[key][field]:
                raise ValueError("teacher/prefix identity mismatch")
    base, arm = summarize(reference), summarize(candidate)
    gates = {lang: dict(kl=arm[lang]["mean_kl"] <= base[lang]["mean_kl"],
                       p99=arm[lang]["p99_kl"] <= base[lang]["p99_kl"],
                       top1=arm[lang]["top1"] >= base[lang]["top1"],
                       nll=arm[lang]["dnll"] <= base[lang]["dnll"])
             for lang in LANGUAGES}
    # These are observed pilot gates; neither confidence bounds nor long-context
    # or full release qualification are inferred from 40 sampled parents.
    required = all(g["kl"] and g["top1"] for g in gates.values())
    return dict(metrics=arm, language_gates=gates, observed_kl_top1_pass=required,
                observed_all_metrics_pass=all(all(g.values()) for g in gates.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", action="append", required=True, metavar="LABEL=JSONL[,JSONL]")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, identity = load(args.reference)
    result = dict(reference_identity=identity, reference=summarize(reference), candidates={}, qualified=False,
                  reference_full_logits="FP16 self-normalized; top1/NLL use FP32 teacher statistics")
    for value in args.candidate:
        label, paths = value.split("=", 1)
        candidate, identity = load([Path(p) for p in paths.split(",")])
        record = compare(reference, candidate)
        record["identity"] = identity
        result["candidates"][label] = record
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value["observed_kl_top1_pass"] for key, value in result["candidates"].items()}))


if __name__ == "__main__":
    main()
