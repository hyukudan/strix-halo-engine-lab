#!/usr/bin/env python3
"""Export the all-on lever data of docs/optimizations.json to data/profile/all_on.json.

Every catalogue row that is an explicit lever (or a lever that cannot be classified) carries an "all_on" object:
  lever, class (exact|fast|turbo|kv|unclassified), gate (pass|pending|fail|none), engines, backends, phase,
  tokens (optional [min, max] prompt or context band), env + value or args, incompatible_with, requires,
  default_tiers (tiers where the lever already runs by default inside its band), quality, reason (unclassified).
The shared resolver (src/profile/all_on.cpp) embeds the exported file; nothing else is maintained by hand.

  export_all_on.py            rewrite data/profile/all_on.json
  export_all_on.py --check    exit 1 when the file differs from the catalogue
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CATALOGUE = ROOT / "docs" / "optimizations.json"
EXPORT = ROOT / "data" / "profile" / "all_on.json"
CLASSES = {"exact", "fast", "turbo", "kv", "unclassified"}
GATES = {"pass", "pending", "fail", "none"}
PHASES = {"any", "prefill", "decode"}


def build() -> str:
    rows = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    out, seen, keys = [], set(), set()
    for row in rows:
        key = (row["engine"], row["name"])
        if key in keys:
            raise SystemExit(f"duplicate catalogue row (engine, name) {key}")
        keys.add(key)
        item = row.get("all_on")
        if item is None:
            continue
        name = item["lever"]
        if name in seen:
            raise SystemExit(f"duplicate all_on lever {name}")
        seen.add(name)
        if item["class"] not in CLASSES or item["gate"] not in GATES or item["phase"] not in PHASES:
            raise SystemExit(f"{name}: unknown class, gate or phase")
        if item["class"] not in ("unclassified", "kv") and not item["backends"]:
            raise SystemExit(f"{name}: a classified lever needs backends")
        if item["class"] not in ("unclassified", "kv") and bool(item.get("env")) == bool(item.get("args")):
            raise SystemExit(f"{name}: a classified lever needs exactly one of env and args")
        if item.get("identity", "pending") != "pending" or (item.get("identity") and item["class"] != "exact"):
            raise SystemExit(f"{name}: identity may only be pending, on an exact lever")
        out.append({"row": row["name"], **item})
    for item in out:
        for other in item.get("incompatible_with", []) + item.get("requires", []):
            if other not in seen:
                raise SystemExit(f"{item['lever']}: unknown lever {other}")
    by_name = {item["lever"]: item for item in out}
    for item in out:
        for other in item.get("incompatible_with", []):
            if item["lever"] not in by_name[other].get("incompatible_with", []):
                raise SystemExit(f"{item['lever']}: incompatible with {other}, but {other} does not name it")
    lines = ",\n".join("  " + json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in out)
    return "[\n" + lines + "\n]\n"


def main() -> int:
    text = build()
    if "--check" in sys.argv:
        if not EXPORT.exists() or EXPORT.read_bytes().decode("utf-8") != text:
            print("data/profile/all_on.json is stale: run tools/ci/export_all_on.py", file=sys.stderr)
            return 1
        return 0
    EXPORT.parent.mkdir(parents=True, exist_ok=True)
    EXPORT.write_bytes(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
