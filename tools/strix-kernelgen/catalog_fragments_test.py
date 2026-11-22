"""Pass the package declarations directly to the compiled, SDK-free lookup fixture."""

import json
from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[2]
FRAGMENTS = {
    "data/catalog/mix.toml": 1,
    "src/cuda/core/select_catalog.toml": 12,
    "data/catalog/cuda_decode.toml": 38,
}


def main():
    lines = []
    for fragment, expected in FRAGMENTS.items():
        with (ROOT / fragment).open("rb") as stream:
            records = tomllib.load(stream)["instance"]
        if len(records) != expected:
            raise ValueError(f"{fragment}: expected {expected} declarations")
        for row in records:
            for weight, revision in row["weights"]:
                fields = (row["op"], row["form"], weight, revision, row["act"], row["max_rows"],
                          row["identity"], row["launcher"], row["source"], row["mode"],
                          int(row.get("capturable", True)), row.get("cost", "roofline"),
                          row.get("workspace", "site"))
                lines.append(" ".join(json.dumps(field) for field in fields))
    subprocess.run([sys.argv[1]], input="\n".join(lines) + "\n", text=True, check=True)


if __name__ == "__main__":
    main()
