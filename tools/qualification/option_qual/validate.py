"""Validate one engine logits file against its forced continuation, in a child process.

usage: python3 -m tools.qualification.option_qual.validate LOGITS CONTINUATION_FILE ROWS
       python3 -m tools.qualification.option_qual.validate --sha FILE

Prints {"rows", "problem", "sha256"} (or {"sha256"}) as JSON. The runner starts this as a separate process so that a
slow validation or hash can be stopped at the invocation deadline (a budget stop, never a failure).
"""
import json
from pathlib import Path
import struct
import sys

from .common import file_sha, read_id_text
from .run import check_forced, count_step_rows


def validate(path, cont_file, expected):
    rows, problem, digest = 0, None, None
    try:
        rows = count_step_rows(path)
        if rows != expected:
            problem = f"{rows} logged rows != continuation {expected}"
        else:
            problem = check_forced(path, read_id_text(cont_file))
        if problem is None:
            digest = file_sha(path)
    except (OSError, ValueError, struct.error) as e:
        problem = f"{type(e).__name__}: {e}"
    return {"rows": rows, "problem": problem, "sha256": digest}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--sha":
        print(json.dumps({"sha256": file_sha(argv[1]) if Path(argv[1]).is_file() else None}))
        return 0
    if len(argv) != 3:
        print(__doc__.splitlines()[2], file=sys.stderr)
        return 2
    print(json.dumps(validate(argv[0], argv[1], int(argv[2]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
