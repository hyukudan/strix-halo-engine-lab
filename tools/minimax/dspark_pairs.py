#!/usr/bin/env python3
"""Evaluate a frozen eight-block DSpark campaign; never infers missing cells or exactness."""
import argparse
import json
import math
import statistics


def intervals(rows, expected):
    # Each key is [content, concurrency, endpoint, population]; population includes original and pooled.
    keys = [tuple(key) for key in expected]
    if len(set(keys)) != len(keys) or any(len(key) != 4 for key in keys):
        raise ValueError("duplicate or malformed frozen manifest key")
    groups = {key: {} for key in keys}
    for row in rows:
        key = tuple(row["key"])
        if key not in groups:
            raise ValueError("unexpected cell")
        block = row["block"]
        if type(block) is not int or block not in range(8) or block in groups[key]:
            raise ValueError("duplicate or invalid independent block")
        off, auto = row["off"], row["auto"]
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in (off, auto)):
            raise ValueError("finite positive throughput required")
        if row.get("complete") is not True or row.get("exact") is not True:
            raise ValueError("incomplete or inexact pair is unqualified; replace the whole pair")
        groups[key][block] = auto - off
    result = []
    for key, blocks in groups.items():
        if set(blocks) != set(range(8)):
            raise ValueError("missing independent block: " + str(key))
        ds = [blocks[i] for i in range(8)]
        mean = statistics.mean(ds)
        half = 2.3646242510103 * statistics.stdev(ds) / math.sqrt(8)  # t(.975, 7)
        upper = mean + half
        result.append({"key": key, "n": 8, "df": 7, "mean": mean, "lower": mean - half, "upper": upper,
                       "decision": "reject_slower" if upper < 0 else "not_demonstrated_slower"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="frozen JSON list of required four-part keys")
    parser.add_argument("pairs", help="JSONL: key, block 0..7, off, auto, complete=true, exact=true")
    args = parser.parse_args()
    with open(args.manifest, encoding="utf-8") as f:
        expected = json.load(f)
    if not expected:
        raise ValueError("empty campaign")
    with open(args.pairs, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    result = intervals(rows, expected)
    print(json.dumps(result, indent=2))
    return int(any(cell["decision"] == "reject_slower" for cell in result))


if __name__ == "__main__":
    raise SystemExit(main())
