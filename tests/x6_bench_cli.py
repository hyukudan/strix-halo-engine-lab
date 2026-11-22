import json
import subprocess
import sys


def run(*args, code):
    result = subprocess.run([sys.argv[1], *args], capture_output=True, text=True, timeout=5, check=False)
    if result.returncode != code:
        raise AssertionError((args, code, result.returncode, result.stdout, result.stderr))
    return result


assert "usage: strix-bench launch" in run("--help", code=0).stdout
assert json.loads(run("exchange", code=0).stdout) == {
    "schema": "strix-bench/v1", "bench": "exchange", "status": "unsupported",
    "reason": "single-device runner has no exchange channel",
}
for args, error in [
    (("unknown",), "unknown bench"),
    (("launch", "--nodes"), "missing value"),
    (("launch", "--nodes", "1"), "invalid value"),
    (("launch", "--nodes", "2s"), "invalid value"),
    (("launch", "--repeats", "1", "--repeats", "2"), "duplicate option"),
    (("launch", "--width", "4"), "unknown option"),
    (("byte-rate", "--bytes", "16,"), "invalid value"),
    (("byte-rate", "--bytes", "17"), "aligned"),
    (("byte-rate", "--bytes", "32,32"), "distinct"),
    (("entry",), "requires --entry"),
    (("entry", "--entry", "id", "--rows", "33"), "invalid value"),
    (("entry", "--entry", "id", "--peak-tops", "nan"), "invalid value"),
    (("exchange", "--repeats", "2"), "takes no options"),
]:
    assert error in run(*args, code=2).stderr

for args in [
    ("launch", "--nodes", "256", "--repeats", "1"),
    ("byte-rate", "--bytes", "16,1073741824"),
    ("entry", "--entry", "cuda-sm120-norm-rms-v1", "--width", "8192", "--rows", "32", "--byte-gbps", "3.5e2"),
]:
    assert "CUDA bench support was not built" in run(*args, code=1).stderr

print("x6_bench_cli: PASS parse and CPU dispatch")
