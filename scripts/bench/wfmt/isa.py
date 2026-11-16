#!/usr/bin/env python3
"""Conservative ISA gate. Any changed operation/resource stream requires review."""
import argparse
import json
import pathlib
import re
import subprocess


def normalise(text, symbols):
    lines, selected = [], not symbols
    for line in text.splitlines():
        # Keep resource lines and opcode/modifier text; strip only addresses/encodings.
        symbol = re.search(r"<([^>]+)>:", line)
        cuda_symbol = re.search(r"Function\s*:\s*(\S+)", line)
        if symbol or cuda_symbol:
            name = (symbol or cuda_symbol).group(1)
            selected = not symbols or any(item in name for item in symbols)
            if selected:
                lines.append("FUNCTION " + name)
            continue
        if not selected:
            continue
        line = re.sub(r"^\s*[0-9a-f]+:\s*(?:[0-9a-f]{2}\s+)*", "", line)
        line = re.sub(r"/\*[0-9a-f]+\*/", "", line)
        line = re.sub(r"//\s*[0-9A-Fa-f]+:.*$", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line and not re.search(r"file format|Fatbin|^code for|^arch =|^identifier =", line):
            lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True, type=pathlib.Path)
    parser.add_argument("--new", required=True, type=pathlib.Path)
    parser.add_argument("--backend", required=True, choices=("hip", "cuda"))
    parser.add_argument("--symbols", required=True, type=pathlib.Path,
                        help="JSON array of required symbol substrings")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--tool", help="llvm-objdump or cuobjdump path")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "wfmt.isa@1", "pass": False, "backend": args.backend}
    try:
        symbols = json.loads(args.symbols.read_text())
        if not isinstance(symbols, list) or not symbols or not all(isinstance(s, str) and s for s in symbols):
            raise ValueError("nonempty required symbol array expected")
        tool = args.tool or ("llvm-objdump" if args.backend == "hip" else "cuobjdump")
        options = (["--disassemble", "--mcpu=gfx1151"] if args.backend == "hip" else
                   ["--dump-sass", "--dump-resource-usage"])
        streams = []
        for name, binary in (("old", args.old), ("new", args.new)):
            output = subprocess.check_output([tool] + options + [str(binary)], text=True,
                                             stderr=subprocess.STDOUT)
            (args.out / (name + ".isa.log")).write_text(output)
            if any(symbol not in output for symbol in symbols):
                raise ValueError(f"required symbol missing in {name} disassembly")
            stream = normalise(output, symbols)
            if not stream:
                raise ValueError(f"empty {name} operation stream")
            (args.out / (name + ".normalised.txt")).write_text("\n".join(stream) + "\n")
            streams.append(stream)
        summary["old_bytes"], summary["new_bytes"] = args.old.stat().st_size, args.new.stat().st_size
        summary["pass"] = streams[0] == streams[1]
        if not summary["pass"]:
            summary["error"] = "instruction/resource stream changed: explanation, byte gate and performance receipts required"
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        summary["error"] = str(error)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
