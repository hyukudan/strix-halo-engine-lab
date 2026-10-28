"""CPU helpers shared by the phased exactness gates."""
import json
import shutil

ARMS = ("identity-baseline", "identity-off", "identity-on-unnamed")

def compare_identity(out):
    arms = [json.loads((out / (name + ".identity.json")).read_text()) for name in ARMS]
    if not arms[0] == arms[1] == arms[2]:
        raise RuntimeError("OFF/baseline or ON unnamed/OFF identity failed")

def restore_turn1(out):
    # Turn 2 may automatically replace the published A. B must restore the original turn 1.
    shutil.copyfile(out / "checkpoint/session.bin", out / "sessions/A/session.bin")

def http_identity(response):
    return {"tokens": response["strix"]["token_ids"], "choices": response["choices"],
            "usage": response["usage"], "sampling": response["strix"].get("sampling")}

def numerical_digests(result, log):
    digests = {}
    def visit(value, path=""):
        if isinstance(value, dict):
            for key, item in value.items():
                if ("digest" in key or "fnv" in key) and isinstance(item, (str, int)):
                    digests[path + key] = item
                else:
                    visit(item, path + key + ".")
        elif isinstance(value, list):
            for index, item in enumerate(value): visit(item, path + str(index) + ".")
    visit(result)
    for line in log.splitlines():
        if line.startswith("tiled prompt: "):
            visit(json.loads(line[len("tiled prompt: "):]), "prompt.")
    if not digests: raise RuntimeError("identity arm lacks numerical digest evidence")
    return digests
