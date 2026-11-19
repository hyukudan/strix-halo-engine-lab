#pragma once
// Frozen reference of the rows tests (tests/hip_dense_decode_rows_tests.hip): main's decode attention kernel and the
// input fill, in a translation unit that CMake always compiles with STRIX_HIP_EXACT_FP_OPTIONS (the options of the
// engine translation unit, src/qwen35/hip/engine.hip). Every build of the rows test, whatever its own flags, compares
// against the bits the engine build of main computes.
// Self-contained (no attn_core include): the header's non-template kernels would be defined twice, and a frozen
// reference must not move when the shared header does.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace rows_v0 {
// The fields of attn_core::dense_decode::Params, same order (the test converts and static_asserts the size).
struct Params {
    unsigned rows, heads, kv_heads, q_stride, rot_dim, pos0;
    float eps, scale;
    double theta;
    bool gated;
};
// Hash fill in [-2, 2): q_n, q_in (f32), K / V cache (bf16); the same values in every build.
void fill(float* qn, std::size_t nq, float* qin, std::size_t nqin, std::uint16_t* k, std::uint16_t* v, std::size_t nkv);
// Main's kernel, grid (rows, heads).
void attend(const Params& p, const float* qn, const float* qin, const std::uint16_t* k,
            const std::uint16_t* v, float* ctx, hipStream_t s);
}  // namespace rows_v0
