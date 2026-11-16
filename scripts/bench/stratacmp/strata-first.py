#!/usr/bin/env python3
"""Collect native Strata first-window logits at frozen TURBO corpus selections, offline."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading

import quality
from tools.qualification.turbo_bar.common import canon_sha, sha256_file
from tools.qualification.turbo_bar import manifest as mf


def free_gb():
    if os.name == "nt":
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(x, ctypes.c_ulonglong) for x in
                ("total", "available", "total_page", "available_page", "total_virtual", "available_virtual", "extended")]
        status = Status(); status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("cannot query available RAM")
        return status.available / 2**30
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30


def engine_args(config, context):
    fixed = {"--gpu": "0", "--kv": "fp16", "--suffix-draft": "0", "--lookup-max-t": "0",
             "--prompt-cache": "0", "--conversation-cache-mib": "0", "--prefill": "auto",
             "--max-context": str(context)}
    args = list(config["args"])
    result = [config["exe"], "--serve"]
    i = 0
    while i < len(args):
        if args[i] in fixed or args[i] == "--short-read":
            if i + 1 == len(args):
                raise ValueError("missing engine argument value")
            i += 2
        elif args[i] == "--serve":
            i += 1
        else:
            result.append(args[i]); i += 1
    for key, value in fixed.items():
        result.extend([key, value])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("config", "manifest", "corpus", "case", "out"):
        p.add_argument("--" + option, required=True)
    p.add_argument("--timeout", type=float, default=120)
    a = p.parse_args()
    frozen = mf.load(a.manifest)
    corpus = Path(a.corpus)
    doc = json.loads((corpus / "corpus.json").read_text())
    if not frozen.get("frozen") or canon_sha(doc) != frozen["protocol"]["corpus_sha256"]:
        raise ValueError("corpus differs from frozen manifest")
    case = next(c for c in doc["cases"] if c["case"] == a.case)
    for key in ("prompt_ids", "positions"):
        if sha256_file(corpus / case[key]) != case[key + "_sha256"]:
            raise ValueError("changed frozen case payload")
    ids = quality.inputs(corpus / case["prompt_ids"])
    positions = [int(x) for x in (corpus / case["positions"]).read_text().split()]
    if not positions or positions != sorted(set(positions)) or positions[0] < 0 or positions[-1] >= len(ids)-1:
        raise ValueError("invalid labeled selections")
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    argv = engine_args(config, len(ids) + 8)
    env = dict(os.environ, **config.get("env", {}))
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0", STRATA_ROUTE_TAIL_SKIP="0",
               STRATA_PREFILL_CPU_SHARE="0", STRATA_DUMP_FIRST_LOGITS=str(out / "first"))
    if free_gb() < 20:
        raise RuntimeError("free RAM below 20 GiB")
    records = []
    with open(out / "engine.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                text=True, encoding="utf-8", errors="replace", cwd=config.get("cwd"), env=env)
        timer = threading.Timer(a.timeout, proc.kill); timer.start()
        try:
            for line in proc.stdout:
                if line.startswith("READY"):
                    break
            else:
                raise RuntimeError("engine ended before READY")
            for index, position in enumerate(positions):
                request = "GEN 1 " + ",".join(map(str, ids[:position+1])) + "\n"
                proc.stdin.write(request); proc.stdin.flush()
                for line in proc.stdout:
                    if line.startswith(("DONE", "ERR")):
                        break
                else:
                    raise RuntimeError("engine ended before request completion")
                if not line.startswith("DONE") or "nonfinite" in line.lower():
                    raise RuntimeError("failed first-logit request")
                records.append({"index": index, "position": position, "prompt_tokens": position+1,
                                "label": ids[position+1], "request_sha256": hashlib.sha256(request.encode()).hexdigest(),
                                "reply": line.strip()})
            proc.stdin.write("QUIT\n"); proc.stdin.flush()
            if proc.wait(timeout=10) != 0:
                raise RuntimeError("engine exit failure")
        finally:
            timer.cancel()
            if proc.poll() is None:
                proc.kill(); proc.wait()
    result = quality.convert_first(out / "first", corpus / case["prompt_ids"], corpus / case["positions"], out / "logits.bin")
    result.update(case=case["case"], manifest_core_sha256=frozen["frozen"]["core_sha256"],
                  binary_sha256=sha256_file(config["exe"]), args=argv, requests=records,
                  effective={"gpu": 0, "kv": "fp16", "prefix_cache": False, "drafting": False,
                             "context_capacity": len(ids)+8, "condition_every_prefix_token": True})
    (out / "index.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"case": case["case"], "rows": len(positions), "complete": True}))


if __name__ == "__main__":
    main()
