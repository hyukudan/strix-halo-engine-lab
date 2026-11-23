"""Guard shared-source coverage of split backend workflows."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    ".github/workflows/backends.yml": [
        "src/core/**",
        "src/exec/**",
        "src/attn_core/**",
        "src/moe_core/**",
        "src/ops/**",
        "src/session/**",
        "src/serve/**",
        "src/spec/**",
        "src/vision/**",
        "include/strix/**",
        "cmake/**",
        "CMakeLists.txt",
        "tests/backend_workflow_paths_tests.py",
        ".github/workflows/backends.yml",
        ".github/workflows/backend-intel.yml",
        "src/cuda/**",
        "src/cuda_core/**",
        "src/qwen35/**",
        "src/inkling2/**",
        "apps/inkling2_mini*",
        "tests/cuda_core/**",
        "tests/inkling2/**"
    ],
    ".github/workflows/backend-intel.yml": [
        "src/core/**",
        "src/exec/**",
        "src/attn_core/**",
        "src/moe_core/**",
        "src/ops/**",
        "src/session/**",
        "src/serve/**",
        "src/spec/**",
        "src/vision/**",
        "include/strix/**",
        "cmake/**",
        "CMakeLists.txt",
        "tests/backend_workflow_paths_tests.py",
        ".github/workflows/backends.yml",
        ".github/workflows/backend-intel.yml",
        "src/intel/**",
        "src/vulkan/**",
        "src/metal/**",
        "src/gemma4/**",
        "src/hy4/**",
        "tests/intel_*",
        "tests/gpu_port_host_tests.cpp",
        "tests/vulkan*",
        "tests/metal*"
    ]
}

def check(path, required):
    content = (ROOT / path).read_text(encoding="utf-8")
    match = re.search(r"(?m)^  pull_request:\n    paths:\n((?:      - '[^'\n]+'\n)+)", content)
    if not match:
        raise AssertionError(f"{path}: missing explicit PR path list")
    paths = set(re.findall(r"(?m)^      - '([^']+)'$", match.group(1)))
    missing = set(required) - paths
    if missing:
        raise AssertionError(f"{path}: missing PR paths: {sorted(missing)}")
    if any(p.startswith("!") for p in paths):
        raise AssertionError(f"{path}: exclusion patterns require explicit review")

if __name__ == "__main__":
    for path, required in REQUIRED.items():
        check(path, required)
    print("backend path-filter contract: ok")
