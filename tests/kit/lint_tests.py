"""Exercise lint classification and baseline matching with synthetic sources."""

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools" / "lint"))
from check import baseline_rows, scan, write_report  # noqa: E402


def main(report):
    fixtures = HERE / "seeded" / "lint"
    files = {
        "include/strix/graph/plan.hpp": (fixtures / "shared.txt").read_text(encoding="utf-8"),
        "include/strix/qwen/binding.hpp": "",
        "src/qwen/private.cpp": (fixtures / "model.txt").read_text(encoding="utf-8"),
        "src/core/test.hip": '#include <hip/hip_runtime.h>\n#include "strix/qwen/private.hpp"\n',
        "src/cuda/core/test.cuh": '#include "../../cuda_core/legacy.cuh"\n',
        "src/cuda_core/legacy.cuh": "",
    }
    findings = scan(files.__getitem__, set(files))
    layers = [item.message for item in findings if item.lint == "layering"]
    assert layers == [
        "L3 includes model header include/strix/qwen/binding.hpp",
        "L3 includes SDK header hip/hip_runtime.h",
        "L3 includes L6 header include/strix/serve/api.hpp",
        "L3 exposes SDK handle hipStream_t",
        "L1 includes model header include/strix/qwen/private.hpp",
        "CUDA kernel homes cross into src/cuda_core/legacy.cuh",
        "model defines a private kernel", "model defines a private resolve", "model defines a private plan",
        "model defines a private load",
    ], layers
    constants = [item.message for item in findings if item.lint == "no_lab_constants"]
    assert constants == [
        "literal topology or link value for kWorldSize",
        "literal topology or link value for link_bandwidth_gbps",
        "literal node name in shared code", "literal network address in shared code",
        "literal host path in shared code", "literal model name in shared code", "model namespace in shared code",
    ], constants
    baseline = {"baseline": "synthetic", "findings": baseline_rows(findings)}
    files["include/strix/graph/plan.hpp"] = "// line moved\n" + files["include/strix/graph/plan.hpp"]
    files["include/strix/graph/plan.hpp"] += "constexpr auto duplicate_host = \"lab-node87.local\";\n"
    after = scan(files.__getitem__, set(files))
    summary = write_report(after, baseline, report)
    assert summary == "R12 lint report: 18 findings, 1 new, 0 baseline entries resolved", summary
    text = report.read_text(encoding="utf-8")
    assert "[new] literal node name" in text and "lab-node87.local" not in text
    print("lint_rules: PASS layering, no_lab_constants, advisory baseline")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
