"""CPU test of scripts/bench/cuda/qwen_fast/gate.sh (no GPU, no model).

--arms: the stub --tf records the profile it was launched with and fails, so run_arm stops right after the launch: the test
sees which arms a selection starts, in which order, and that bad selections / a compare without sealed arms refuse with rc 2.
Contract check: with STUB_CFG set the stub writes that effective-config.json into its --out dir and succeeds, and the seal
tool is a no-op stub, so the arm's resolved-contract check runs for real (rc 0 accepted, rc 2 refused).
Run: python tests/qwen_fast_gate_arms_tests.py   (needs bash on PATH; Git Bash on Windows)
"""
import copy
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "bench" / "cuda" / "qwen_fast" / "gate.sh"
BASH = shutil.which("bash") or r"C:\Program Files\Git\bin\bash.exe"

I8 = "gsq-iq-i8mma-turbo-cuda-v1"
BF16 = "gsq-iq-bf16mma-cuda-v1"
FORMS = "qwen-cuda;prefill=fast;forms=hc_mma,qsa_mma;moe_route_bf16=0;ids=qwen-cuda-hc-f16mma-v2,qwen-cuda-qsa-bf16mma-v2"
RULE = f"prefill_rows>=512:{I8},else:{BF16}"
# The effective config a nested TURBO arm prints (--profile turbo, default expert_turbo_rows), reduced to the fields
# gate.sh reads.
TURBO = {
    "entry": "serve", "profile": "turbo",
    "arithmetic_contract": FORMS + "+experts[" + RULE + "]",
    "form_identities": "qwen-cuda-hc-f16mma-v2,qwen-cuda-qsa-bf16mma-v2",
    "expert_arithmetic": I8,
    "expert_regime": {"prefill_blocks_min_rows": 512, "prefill_blocks": I8, "windows_and_smaller_blocks": BF16, "rule": RULE},
    "forms": {"hc_mma": True, "mtp_mma": False, "qsa_mma": True, "moe_mma": False, "moe_route_bf16": False},
    "turbo_bundle": {"id": "qwen-cuda-turbo-bundle-v2"},
    "executions": [{"prompt": "p", "run": 0, "rows": 32640,
                    "forms": {"hc_mma": True, "mtp_mma": False, "qsa_mma": True, "moe_mma": False, "moe_route_bf16": False}}],
}


def posix(p):
    s = str(p).replace("\\", "/")
    return "/" + s[0].lower() + s[2:] if len(s) > 1 and s[1] == ":" else s


def run(tmp, *args, cfg=None):
    stub = tmp / "tf_stub.sh"
    stub.write_text('#!/bin/bash\nfor ((i=1;i<=$#;i++)); do [ "${!i}" = --profile ] && { j=$((i+1)); '
                    'echo "${!j}" >> "$(dirname "$0")/launched"; }; [ "${!i}" = --out ] && { j=$((i+1)); out=${!j}; }; done\n'
                    '[ -n "${STUB_CFG:-}" ] && { cp "$STUB_CFG" "$out/effective-config.json"; exit 0; }\nexit 1\n', newline="\n")
    stub.chmod(0o755)
    seal = tmp / "seal_stub.py"
    seal.write_text("import sys\nsys.exit(0)\n")
    env = dict(os.environ, PYTHON=sys.executable, GATE_PY=str(seal))
    env.pop("STUB_CFG", None)
    if cfg is not None:
        (tmp / "cfg.json").write_text(json.dumps(cfg))
        env["STUB_CFG"] = posix(tmp / "cfg.json")
    cmd = [BASH, posix(GATE), "--tf", posix(stub), "--out", posix(tmp / "out"), "--prompt", "p=x.u32", *args]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    launched = (tmp / "launched").read_text().split() if (tmp / "launched").exists() else []
    return r.returncode, launched, r.stdout + r.stderr


def case(name, args, want_rc, want_launched, want_text=None, prep=None, cfg=None):
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        if prep:
            prep(tmp)
        rc, launched, text = run(tmp, *args, cfg=cfg)
        ok = rc == want_rc and launched == want_launched and (want_text is None or want_text in text)
        print(f"{'PASS' if ok else 'FAIL'} {name}: rc {rc} launched {launched}" + ("" if ok else f"\n{text[-600:]}"))
        return ok


def turbo(edit):
    c = copy.deepcopy(TURBO)
    edit(c)
    return c


def flat(c):  # the historical form: one expert arithmetic everywhere (no expert_regime), e.g. --expert-turbo-rows off
    del c["expert_regime"]
    c["arithmetic_contract"] = FORMS + "+" + I8


def main():
    t = ["--arms", "turbo"]
    results = [
        # default = previous behaviour: the exact arm is launched first (the stub fails it, so the gate stops there)
        case("default starts exact", [], 2, ["exact"]),
        case("--arms exact", ["--arms", "exact"], 2, ["exact"]),
        case("--arms fast,compare starts only fast", ["--arms", "fast,compare"], 2, ["fast"]),
        case("unknown step refused", ["--arms", "exact,bogus"], 2, [], "unknown step 'bogus'"),
        case("empty selection refused", ["--arms", ","], 2, [], "no step selected"),
        case("compare without sealed arms refused", ["--arms", "compare"], 2, [], "is missing"),
        case("compare with only exact sealed refused", ["--arms", "compare"], 2, [], "out/fast is missing",
             prep=lambda t: (t / "out" / "exact").mkdir(parents=True)),
        # resolved-contract check: the nested TURBO contract (+experts[rule]) is accepted, mismatches are refused
        case("nested turbo contract accepted", t, 0, ["turbo"], "+experts[" + RULE + "]", cfg=TURBO),
        case("flat turbo contract (rows off) accepted", t, 0, ["turbo"], "+" + I8 + " |", cfg=turbo(flat)),
        case("regime rows differ from contract refused", t, 2, ["turbo"], "does not match resolved arithmetic contract",
             cfg=turbo(lambda c: c.update(arithmetic_contract=FORMS + "+experts[" + RULE.replace("512", "1024") + "]"))),
        case("regime with flat contract refused", t, 2, ["turbo"], "does not match resolved arithmetic contract",
             cfg=turbo(lambda c: c.update(arithmetic_contract=FORMS + "+" + I8))),
        case("flat config with bracket contract refused", t, 2, ["turbo"], "does not match resolved arithmetic contract",
             cfg=turbo(lambda c: (flat(c), c.update(arithmetic_contract=FORMS + "+experts[" + RULE + "]")))),
        case("regime prefill arithmetic != expert_arithmetic refused", t, 2, ["turbo"], "invalid expert_regime",
             cfg=turbo(lambda c: c["expert_regime"].update(prefill_blocks=BF16))),
        case("regime rule inconsistent with its fields refused", t, 2, ["turbo"], "invalid expert_regime",
             cfg=turbo(lambda c: c["expert_regime"].update(prefill_blocks_min_rows=1024))),
        case("regime rows 0 refused", t, 2, ["turbo"], "invalid expert_regime",
             cfg=turbo(lambda c: c["expert_regime"].update(prefill_blocks_min_rows=0, rule=f"prefill_rows>=0:{I8},else:{BF16}"))),
        case("regime narrow == wide refused", t, 2, ["turbo"], "invalid expert_regime",
             cfg=turbo(lambda c: c["expert_regime"].update(windows_and_smaller_blocks=I8, rule=f"prefill_rows>=512:{I8},else:{I8}"))),
        case("regime with incumbent experts refused", t, 2, ["turbo"], "invalid expert_regime",
             cfg=turbo(lambda c: c.update(expert_arithmetic="incumbent"))),
        # arm label = arithmetic: --profile turbo with --expert-turbo off resolves the FAST experts (refused under turbo);
        # a fast arm that resolved the int8 experts (e.g. --expert-turbo on) is refused under fast
        case("turbo arm with FAST experts refused", t, 2, ["turbo"], "turbo arm did not resolve the int8 experts",
             cfg=turbo(lambda c: (c.pop("expert_regime"), c.update(expert_arithmetic=BF16, arithmetic_contract=FORMS + "+" + BF16)))),
        case("fast arm with FAST experts accepted", ["--arms", "fast"], 0, ["fast"], "+" + BF16 + " |",
             cfg=turbo(lambda c: (c.pop("expert_regime"), c.update(profile="fast", expert_arithmetic=BF16, arithmetic_contract=FORMS + "+" + BF16)))),
        case("fast arm with int8 experts refused", ["--arms", "fast"], 2, ["fast"], "fast arm resolved the int8 experts",
             cfg=turbo(lambda c: (flat(c), c.update(profile="fast")))),
    ]
    print(f"qwen_fast_gate_arms: {sum(results)}/{len(results)} PASS")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
