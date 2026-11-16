"""Extract unchanged host dispatch/control code for CPU-only regression tests.
CUDA kernels are not compiled or emulated; route and ownership behavior is exercised.
"""
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
def extract(path, begin, end):
    text = (root / path).read_text()
    return text[text.index(begin):text.index(end, text.index(begin))]
(out / "dispatch.inc").write_text(extract("src/cuda_core/dense/dense.cu", "cudaError_t dense_rows(", "cudaError_t embedding_rows("))
(out / "mma_support.inc").write_text(extract("src/cuda_core/dense/dense_mma.cu", "bool dense_mma_supported(", "cudaError_t dense_mma_launch("))
text = (root / "src/cuda_core/dense/nvfp4.cu").read_text()
parts = []
for begin, end in [("using WorkspaceKey", "template <"), ("cudaError_t ensure_ws(", "__global__ void amax_kernel_bf16"), ("cudaError_t nvfp4_reserve_workspace(", "namespace {")]:
    a = text.index(begin)
    parts.append(text[a:text.index(end, a)])
(out / "workspace.inc").write_text("\n".join(parts))
text = (root / "apps/cuda_qwen35_decode_main.cu").read_text()
a = text.index("#ifdef _WIN32", text.index("struct Clock")) if "struct Clock" in text else text.index("#ifdef _WIN32")
opening = text[a:text.index("#endif", a)+len("#endif")]
a = text.index("#ifdef _WIN32", a+len(opening))
closing = text[a:text.index("#endif", a)+len("#endif")]
(out / "sampler.inc").write_text("#undef _WIN32\n#define _popen forbidden_windows_popen\n#define _pclose forbidden_windows_pclose\n#define popen qwen35_guard_popen\n#define pclose qwen35_guard_pclose\nstatic FILE* popen(const char*, const char*) { return stdout; }\nstatic int pclose(FILE*) { return 0; }\nstatic int portable_sampler() {\n" + opening + "\n" + closing + "\nreturn p ? 0 : 1;\n}\n")
(out / "maxrel.inc").write_text(extract("tests/cuda_core/cuda_core_test.cu", "static double maxrel(", "static std::uint32_t* dpos("))
