#pragma once
// Tile shapes of the exact prefill HC kernels (qwen_hc.cu: hc_projection_kernel, hc_output_kernel), host-visible so the kernels and
// the CPU DAG test (tests/qwen_hc_dag_tests.cpp) use one definition. The per-element FP32 chains do not depend on these shapes; the
// test checks that every (row, output) is owned exactly once and that the injection tree equals the low-output tree.
#include <cstdint>

namespace strix::cuda::qwen_hc::exact_shape {

constexpr std::uint32_t kStreams = 4U, kHidden = 2560U, kWide = 10240U, kRank = 320U;  // = qwen_hc.hpp (static_assert there)

// Projection: CTA = kProjWarps warps x kProjOutputs outputs x kProjRows rows; window lane = hardware lane; tile = 5 chunks of 128.
constexpr std::uint32_t kProjRows = 16U, kProjOutputs = 8U, kProjWarps = 8U, kProjItems = kProjRows * kProjOutputs;
constexpr std::uint32_t kProjLowBlocks = kRank / (kProjWarps * kProjOutputs);  // 5 low-output blocks, then 1 injection block
// Injection block: warp w owns rows kProjInjRows * w + (0 .. kProjInjRows - 1) x the kStreams injection outputs.
constexpr std::uint32_t kProjInjRows = kProjRows / kProjWarps, kProjInjItems = kProjInjRows * kStreams;
static_assert(kProjInjRows * kProjWarps == kProjRows && kProjInjItems <= 32U, "injection block: rows split over the warps");

// Projection chunk pipeline of one CTA (kProjChunks chunks of 128 features), shared by hc_projection_kernel and the CPU test so the order
// is checked on the host. The x tile and the weight words of chunk k + 1 are requested at the top of chunk k; the coefficients of chunk
// k + 1 are decoded at the tail of chunk k, after its FFMA body. Every wait on a global load therefore sits one whole chunk body after
// the load was issued, whatever scoreboard the compiler assigns. (Decoding at the top made the decode's scoreboard wait also cover the
// x request issued just before it: one exposed L2 round trip per chunk.) Schedule only: every chunk sees the same operands and the
// FMA order per (row, output) is unchanged.
constexpr std::uint32_t kProjChunks = kWide / 128U;
#if defined(__CUDACC__)
#define STRIX_QWEN_HC_HD __host__ __device__ __forceinline__
#pragma nv_exec_check_disable
#else
#define STRIX_QWEN_HC_HD inline
#endif
template <class Request, class Decode, class Store, class Body, class Barrier>
STRIX_QWEN_HC_HD void proj_pipeline(Request&& request, Decode&& decode, Store&& store, Body&& body, Barrier&& barrier) {
    request(0U);
    decode();
    store(0U);
    barrier();
    for (std::uint32_t chunk = 0; chunk < kProjChunks; ++chunk) {
        const bool more = chunk + 1U < kProjChunks;
        if (more) request(chunk + 1U);  // x tile -> registers, weight words -> registers
        body(chunk);                    // coefficients of chunk, x tile buffer chunk & 1
        if (more) {
            decode();             // coefficients of chunk + 1 (their words were requested before the body)
            store(chunk + 1U);    // x tile -> buffer (chunk + 1) & 1, last read by body(chunk - 1) before the previous barrier
        }
        barrier();
    }
}

// Output: thread = one feature, kOutRows rows x kStreams gate chains; lows staged as [kRank][kOutLowPitch] floats.
constexpr std::uint32_t kOutRows = 32U, kOutThreads = 128U;
// 36: every k row stays 16-B aligned for the float4 broadcast reads; the staging writes (consecutive lanes = consecutive k) spread
// over 8 banks (4-way) instead of one bank (32-way at pitch 32).
constexpr std::uint32_t kOutLowPitch = kOutRows + 4U;
// Fused epilogue: the chains leave the registers in halves of kOutHalfRows rows into thread-private float4 slots (index
// r * kOutThreads + thread) of the idle lows buffer; a rolled loop handles kOutMixRows rows per iteration.
constexpr std::uint32_t kOutHalfRows = kOutRows / 2U, kOutMixRows = 2U;
constexpr std::uint32_t kOutStagePerThread = kRank * kOutRows / kOutThreads;  // 80 staging loads per thread
static_assert(kOutHalfRows * kOutThreads * 4U <= kRank * kOutLowPitch, "epilogue half must fit the lows buffer");
static_assert((kOutLowPitch * 4U) % 16U == 0U && kOutHalfRows % kOutMixRows == 0U, "lows rows must stay float4-aligned");
static_assert((kRank * kOutRows) % kOutThreads == 0U, "staging covers the tile exactly");
// Staging element i of thread t: idx = t + kOutThreads * i, row r = idx / kRank, k = idx % kRank, stored at lows[k][r].

}  // namespace strix::cuda::qwen_hc::exact_shape
