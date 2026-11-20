#!/usr/bin/env python3
"""Summarise strix-dist-gpu-probe JSON lines (rank 0 view) from stdin: one line per size.

Columns: bytes, pair-max p50 (the exchange as both ranks see it), rank-0 p50 / p90 / p99, exchanges over 1 ms, CPU us per
exchange (all threads). Non-JSON lines are ignored; errors and refusals are echoed (first 3).
"""
import json
import sys


def main() -> int:
    errors = []
    print("bytes pair_p50 p50 p90 p99 over1ms cpu_us")
    for line in sys.stdin:
        line = line.strip()
        if not line.startswith("{"):
            low = line.lower()
            if ("error" in low or "refus" in low) and len(errors) < 3:
                errors.append(line[:200])
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "bytes" not in r:
            continue
        print(f'{r["bytes"]} {r.get("pair_max_p50_us", -1):.1f} {r.get("p50_us", -1):.1f} {r.get("p90_us", -1):.1f} '
              f'{r.get("p99_us", -1):.1f} {r.get("over_1ms", -1)} {r.get("cpu_us", -1):.0f}')
    for e in errors:
        print("!", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
