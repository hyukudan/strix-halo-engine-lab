"""Integration checks for host paths whose enclosing engines require GPU kernels."""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
hip = (root / "src/qwen35/hip/engine.hip").read_text()
assert "g_wide" not in hip
impl = hip[hip.index("struct HipEngine::Impl"):]
assert "WideScratch wide;" in impl
destructor = impl[impl.index("~Impl()"):impl.index("template <class T>")]
assert destructor.index("hipStreamSynchronize") < destructor.index("hipFree")
for member in ("codes", "deltas", "g", "u"):
    assert f"fr(wide.{member})" in destructor
for line in impl.splitlines():
    if "linear(" in line or "glu(" in line:
        assert line.rstrip().endswith(", stream, wide);")

cuda = (root / "src/qwen35/cuda/engine.cu").read_text()
assert cuda.index("pf.sl.clear();") < cuda.index("allocations.clear();")
for name in ("CudaEngine::from_gguf(", "CudaEngine::synthetic("):
    factory = cuda[cuda.index(name):]
    assert factory.index("opt.prefill_slots < 1 || opt.prefill_slots > 2") < factory.index("P.opt = opt;")
assert "prefill effective slots:" in cuda and "prefill-admission:" in cuda
assert "dense_big_fast8_supported()" in cuda
big = (root / "src/cuda_core/dense/dense_big.cu").read_text()
assert "get_table(S, mode," in big
for marker, guard in (("cudaError_t dense_big_split(", "f8"), ("cudaError_t dense_big_gemm(", "a.f8")):
    function = big[big.index(marker):]
    assert function.index(f"if ({guard} && !dense_big_fast8_supported())") < function.index("Scratch& S")
print("engine ownership, dispatch wiring, slot bounds and FAST8 guards: PASS")
