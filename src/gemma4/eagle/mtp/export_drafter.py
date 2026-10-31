"""Package an MTP assistant export into the release layout <model dir>/drafter-mtp/.

usage:
  python export_drafter.py EXPORT_DIR OUT_MODEL_DIR [--base BASE_ASSISTANT_DIR] [--license APACHE_LICENSE_TXT] [--force]

EXPORT_DIR is a fine-tune export in the original Google layout (e.g. runs/<name>/export-best:
model.safetensors, config.json, generation_config.json, LINEAGE.md). The script writes OUT_MODEL_DIR/drafter-mtp/ with
  config.json, generation_config.json, model.safetensors   copied byte for byte
  LINEAGE.md                                               copied (or a minimal one when absent)
  LICENSE                                                  Apache License 2.0 text (base model licence)
  NOTICE                                                   attribution: Google Gemma base, drafter fine-tune,
                                                           teacher = NVIDIA Nemotron 3.5 Bolt E4B outputs (OpenMDW-1.1)
  MANIFEST.json                                            sha256/size of every file, source paths, header summary
It refuses a checkpoint whose safetensors header (keys, dtypes, shapes) differs from --base (default: the Google
assistant), or whose config.json is not gemma4_assistant with backbone_hidden_size 2560. The engine loads the result
with --draft-mtp OUT_MODEL_DIR/drafter-mtp, or resolves it from the model directory (eagle::mtp::resolve_drafter).
Licence wording is a packaging aid, not legal advice; the maintainers sign off the release notices.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import sys

DEFAULT_BASE = "gemma-4-E4B-it-assistant"  # directory of the original Google assistant
NOTICE = """Drafter: Gemma-4-E4B-it MTP assistant (Gemma4AssistantForCausalLM), fine-tuned as a drafter.

Base model: google/gemma-4-E4B-it-assistant, Copyright Google LLC, licensed under the Apache License, Version 2.0
(see LICENSE; https://ai.google.dev/gemma/docs/gemma_4_license).

Modification: fine-tuned as a speculative-decoding drafter for NVIDIA Nemotron 3.5 Bolt E4B. Same tensor keys, shapes
and dtypes as the base model; token_ordering unchanged. Details: LINEAGE.md.

Training signal: outputs (greedy tokens, top-32 logits, hidden states and KV) of NVIDIA Nemotron 3.5 Bolt E4B, licensed
under OpenMDW-1.1. No target model tensor is contained in these weights.
"""


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return {k: (v["dtype"], tuple(v["shape"])) for k, v in h.items() if k != "__metadata__"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def default_license():
    try:
        import transformers  # Apache-2.0; its distribution ships the licence text
        root = os.path.dirname(os.path.dirname(transformers.__file__))
        for d in os.listdir(root):
            if d.startswith("transformers-") and d.endswith(".dist-info"):
                for name in ("LICENSE", "licenses/LICENSE"):
                    p = os.path.join(root, d, name)
                    if os.path.exists(p) and "Apache License" in open(p, encoding="utf-8", errors="replace").read(400):
                        return p
    except ImportError:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("out_model_dir")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--license", default=None, help="Apache License 2.0 text file")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    src = a.export_dir
    model = os.path.join(src, "model.safetensors")
    if not os.path.exists(model):
        sys.exit(f"no model.safetensors in {src}")
    cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    if cfg.get("model_type") != "gemma4_assistant" or cfg.get("backbone_hidden_size") != 2560:
        sys.exit("config.json must be gemma4_assistant with backbone_hidden_size 2560")
    got = header(model)
    base_model = os.path.join(a.base, "model.safetensors")
    if os.path.exists(base_model):
        want = header(base_model)
        if got != want:
            diff = sorted(set(got.items()) ^ set(want.items()))[:6]
            sys.exit(f"safetensors layout differs from the base assistant: {diff}")
    else:
        print(f"warning: base {base_model} not found; layout not compared", file=sys.stderr)
    lic = a.license or default_license()
    if not lic or "Apache License" not in open(lic, encoding="utf-8", errors="replace").read(400):
        sys.exit("need the Apache License 2.0 text (--license PATH)")
    out = os.path.join(a.out_model_dir, "drafter-mtp")
    if os.path.exists(out):
        if not a.force:
            sys.exit(f"{out} exists (use --force)")
        shutil.rmtree(out)
    os.makedirs(out)
    for name in ("model.safetensors", "config.json", "generation_config.json"):
        if os.path.exists(os.path.join(src, name)):
            shutil.copyfile(os.path.join(src, name), os.path.join(out, name))
    if os.path.exists(os.path.join(src, "LINEAGE.md")):
        shutil.copyfile(os.path.join(src, "LINEAGE.md"), os.path.join(out, "LINEAGE.md"))
    else:
        open(os.path.join(out, "LINEAGE.md"), "w", encoding="utf-8").write(
            "# Lineage\n\nBase: google gemma-4-E4B-it-assistant (Apache-2.0). Fine-tuned as a drafter for "
            "NVIDIA Nemotron 3.5 Bolt E4B (teacher outputs only).\n")
    text = open(lic, encoding="utf-8").read()
    start = text.rfind("\n", 0, text.index("Apache License")) + 1  # drop any third-party copyright preamble
    open(os.path.join(out, "LICENSE"), "w", encoding="utf-8", newline="\n").write(text[start:])
    open(os.path.join(out, "NOTICE"), "w", encoding="utf-8").write(NOTICE)
    files = {n: {"sha256": sha256(os.path.join(out, n)), "bytes": os.path.getsize(os.path.join(out, n))}
             for n in sorted(os.listdir(out))}
    manifest = {"layout": "drafter-mtp/v1", "engine_identity": "g4-mtp-v1", "source_export": os.path.abspath(src),
                "base": os.path.abspath(a.base), "license_source": os.path.abspath(lic), "tensors": len(got),
                "files": files}
    json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({"out": out, "tensors": len(got), "model_sha256": files["model.safetensors"]["sha256"]}))


if __name__ == "__main__":
    main()
