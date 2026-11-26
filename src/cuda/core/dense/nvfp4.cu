// TURBO weights: NVFP4 (e2m1 + e4m3/16 + FP32 global) converter and decode GEMV rows 1..32. Layout/orders: nvfp4.hpp.
// Native FP4 conversion and decode require an architecture-specific image.
#include "strix/cuda_core/nvfp4.hpp"
#include "rows_fp4.cuh"
#include "../../../core/rows_plan.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/weight/formats/scalar_float.hpp"
#include "strix/weight/registry.hpp"


#include <cuda_fp16.h>

#include "strix/decode_weight_sweep.hpp"
#include "strix/lever.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>

#include "../common.cuh"
#include "../../arch/capability.hpp"
#include "../../arch/native_nvfp4.hpp"
#include "strix/split_k_policy.hpp"

namespace strix::cuda_core {
namespace {

__device__ __forceinline__ void e2m1_word(std::uint32_t w, std::uint32_t (&o)[4]) {
#if defined(STRIX_CUDA_NATIVE_NVFP4)
    asm("{\n .reg .b8 b0, b1, b2, b3;\n mov.b32 {b0, b1, b2, b3}, %4;\n"
        " cvt.rn.f16x2.e2m1x2 %0, b0;\n cvt.rn.f16x2.e2m1x2 %1, b1;\n cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        " cvt.rn.f16x2.e2m1x2 %3, b3;\n}"
        : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3])
        : "r"(w));
#else
    (void)w;
    o[0] = o[1] = o[2] = o[3] = 0;
    __trap();
#endif
}
// two e4m3 bytes (low 16 bits of v) -> f16x2 (low = byte 0)
__device__ __forceinline__ std::uint32_t e4m3x2_to_f16x2(std::uint32_t v) {
#if defined(STRIX_CUDA_NATIVE_NVFP4)
    std::uint32_t r;
    asm("{\n .reg .b16 lo, hi;\n mov.b32 {lo, hi}, %1;\n cvt.rn.f16x2.e4m3x2 %0, lo;\n}" : "=r"(r) : "r"(v));
    return r;
#else
    (void)v;
    __trap();
    return 0;
#endif
}
__device__ __forceinline__ std::uint32_t f32_to_e4m3(float v) {
#if defined(STRIX_CUDA_NATIVE_NVFP4)
    std::uint32_t r;
    asm("{\n .reg .b16 h;\n cvt.rn.satfinite.e4m3x2.f32 h, 0f00000000, %1;\n cvt.u32.u16 %0, h;\n}" : "=r"(r) : "f"(v));
    return r & 0xFFu;
#else
    (void)v;
    __trap();
    return 0;
#endif
}
// (hi, lo) -> byte, hi in the upper nibble
__device__ __forceinline__ std::uint32_t f32x2_to_e2m1x2(float hi, float lo) {
#if defined(STRIX_CUDA_NATIVE_NVFP4)
    std::uint32_t r;
    asm("{\n .reg .b8 h;\n cvt.rn.satfinite.e2m1x2.f32 h, %1, %2;\n cvt.u32.u8 %0, h;\n}" : "=r"(r) : "f"(hi), "f"(lo));
    return r & 0xFFu;
#else
    (void)hi;
    (void)lo;
    __trap();
    return 0;
#endif
}
__device__ __forceinline__ std::uint32_t pack_f16x2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const std::uint32_t*>(&h);
}
__device__ __forceinline__ std::uint32_t hmul2(std::uint32_t a, std::uint32_t b) {
    std::uint32_t r;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
__device__ __forceinline__ std::uint32_t prmt(std::uint32_t a, std::uint32_t sel) {
    std::uint32_t r;
    asm("prmt.b32 %0, %1, %1, %2;" : "=r"(r) : "r"(a), "r"(sel));
    return r;
}
__device__ __forceinline__ void mma16816(float (&c)[4], std::uint32_t a0, std::uint32_t a1, std::uint32_t a2,
                                         std::uint32_t a3, std::uint32_t b0, std::uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------------------------------------------------------- GEMV
constexpr int kWarpsPerBlock = 4;
// split-K policy (strix/split_k_policy.hpp): depends on the shape and the process-wide profile, never on rows
std::uint32_t g_target_warps = split_k::kExactTarget;  // EXACT target (test hook: nvfp4_set_split_target)
split_k::Profile g_split_profile = split_k::Profile::exact;
split_k::Device g_split_device{};

constexpr int kSlab = 4;  // K chunks (of 128) of activations staged in shared memory per barrier pair

struct Nvfp4Decoder {
    static constexpr unsigned kAlign = 128;

    struct Raw {
        uint4 d0, d1, sc;
    };

    // Weight stream: read once, keep out of L1 so the activation slab stays resident.
    static __device__ __forceinline__ uint4 ld_stream(const void* p) {
        uint4 r;
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
        return r;
    }

    static __device__ __forceinline__ Raw load(const unsigned char* rec, int lane) {
        Raw r;
        r.d0 = ld_stream(rec + lane * 16);
        r.d1 = ld_stream(rec + 512 + lane * 16);
        r.sc = ld_stream(rec + 1024 + (lane >> 2) * 16);
        return r;
    }

    // xs layout (fragment ready): [chunk in slab][rb][group j][lane] -> uint2 {b0, b1} (f16x2 pairs of that lane).
    template <int RB, int MT>
    static __device__ __forceinline__ void compute_chunk(const Raw (&L)[MT], const bool (&on)[MT], const uint2* xs, int lane,
                                                  float (&acc)[MT][RB][4]) {
#pragma unroll
        for (int jp = 0; jp < 4; ++jp) {  // group pair (2jp, 2jp+1)
            std::uint32_t a[MT][2][4];     // [mt][group in pair][a0..a3]
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                if (!on[mt]) continue;
                const std::uint32_t w0 = jp == 0 ? L[mt].d0.x : jp == 1 ? L[mt].d0.y : jp == 2 ? L[mt].d0.z : L[mt].d0.w;
                const std::uint32_t w1 = jp == 0 ? L[mt].d1.x : jp == 1 ? L[mt].d1.y : jp == 2 ? L[mt].d1.z : L[mt].d1.w;
                const std::uint32_t s0w = jp < 2 ? L[mt].sc.x : L[mt].sc.y;
                const std::uint32_t s1w = jp < 2 ? L[mt].sc.z : L[mt].sc.w;
                const std::uint32_t sh = (jp & 1) * 16;
                const std::uint32_t s0 = e4m3x2_to_f16x2(s0w >> sh);
                const std::uint32_t s1 = e4m3x2_to_f16x2(s1w >> sh);
                std::uint32_t o0[4], o1[4];
                e2m1_word(w0, o0);
                e2m1_word(w1, o1);
                const std::uint32_t s0a = prmt(s0, 0x1010), s0b = prmt(s0, 0x3232);
                const std::uint32_t s1a = prmt(s1, 0x1010), s1b = prmt(s1, 0x3232);
                // group 2jp: a0 = (h0,p0) a1 = (h1,p0) a2 = (h0,p1) a3 = (h1,p1)
                a[mt][0][0] = hmul2(o0[0], s0a);
                a[mt][0][1] = hmul2(o1[0], s1a);
                a[mt][0][2] = hmul2(o0[1], s0a);
                a[mt][0][3] = hmul2(o1[1], s1a);
                a[mt][1][0] = hmul2(o0[2], s0b);
                a[mt][1][1] = hmul2(o1[2], s1b);
                a[mt][1][2] = hmul2(o0[3], s0b);
                a[mt][1][3] = hmul2(o1[3], s1b);
            }
#pragma unroll
            for (int gi = 0; gi < 2; ++gi) {
                uint2 b[RB];
#pragma unroll
                for (int rb = 0; rb < RB; ++rb) b[rb] = xs[(rb * 8 + 2 * jp + gi) * 32 + lane];
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    if (!on[mt]) continue;
#pragma unroll
                    for (int rb = 0; rb < RB; ++rb)
                        mma16816(acc[mt][rb], a[mt][gi][0], a[mt][gi][1], a[mt][gi][2], a[mt][gi][3], b[rb].x, b[rb].y);
                }
            }
        }
    }
};

// grid (ceil(ntiles / (4 MT)), ks, nw): the 4 warps of a block take 4 MT consecutive output tiles and share the
// activation slab through shared memory; blockIdx.y selects the K range. part = [z][ks][rows][out] FP32; when DIRECT
// (ks == 1, single weight) the epilogue y = acc * gscale (+ resid) is written straight to y.
struct Job {
    const unsigned char* w[2];  // [1] != null: fused gate/up (silu(w0 x) * (w1 x))
    After<float> part;          // split-K partials [z][ks][rows][out]
    After<float> y;
    After<const float> resid;   // non-constant buffers: reachable only after pdl_wait (common.cuh)
    float gs[2];
    std::uint32_t out, ks, g0;  // g0 = first grid.x block of this job
};
struct Jobs {
    static constexpr int capacity = 4;
    Job j[capacity];
    int n;
};

struct SplitJob : Job {
    After<const float> x;
    const unsigned char* scales;
    std::uint32_t in;
};

struct SplitJobs {
    static constexpr int capacity = core::rows::kMaxJobs;
    SplitJob j[capacity];
    int n;
};

__host__ __device__ unsigned split_member(const SplitJobs& jobs, std::uint32_t block_x) {
    unsigned member = 0;
    while (member + 1 < static_cast<unsigned>(jobs.n) && block_x >= jobs.j[member + 1].g0) ++member;
    return member;
}

struct Mxfp8Decoder {
    static constexpr unsigned kGroup = 32, kAlign = 16, kChunk = kGroup;

    struct Raw {
        uint4 codes[2];
        unsigned char scale;
    };

    static __device__ __forceinline__ Raw load(const unsigned char* w, const unsigned char* scales,
                                                std::size_t row, std::uint32_t in, std::uint32_t group) {
        const auto* codes = reinterpret_cast<const uint4*>(w + row * in + group * kGroup);
        return {{codes[0], codes[1]}, scales[row * (in / kGroup) + group]};
    }

    template <int RB, int MT>
    static __device__ __forceinline__ void dot(const unsigned char* w, const unsigned char* scales, const float* x,
                                               std::uint32_t in, std::uint32_t rows, std::uint32_t bx, int warp, int lane,
                                               std::uint32_t c0, std::uint32_t c1, const bool (&on)[MT],
                                               float (&acc)[MT][RB][4]) {
        const auto g = static_cast<unsigned>(lane >> 2);
        const auto t = static_cast<unsigned>(lane & 3);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            if (!on[mt]) continue;
            const std::size_t tile = (bx * kWarpsPerBlock + warp) * MT + mt;
#pragma unroll
            for (unsigned half = 0; half < 2; ++half) {
                const std::size_t row = tile * 16 + half * 8 + g;
                for (std::uint32_t c = c0; c < c1; ++c) {
                    const Raw raw = load(w, scales, row, in, c);
                    const float scale = weight::scalar_float::e8m0(raw.scale);
                    float sums[RB][2]{};
#pragma unroll
                    for (unsigned v = 0; v < 2; ++v) {
                        const uint4 word = raw.codes[v];
                        const unsigned words[4] = {word.x, word.y, word.z, word.w};
#pragma unroll
                        for (unsigned j = 0; j < 4; ++j) {
#pragma unroll
                            for (unsigned b = 0; b < 4; ++b) {
                                const float value = weight::scalar_float::e4m3(
                                    static_cast<unsigned char>(words[j] >> (8 * b)));
                                const auto k = c * kGroup + v * 16 + j * 4 + b;
#pragma unroll
                                for (unsigned rb = 0; rb < RB; ++rb) {
#pragma unroll
                                    for (unsigned p = 0; p < 2; ++p) {
                                        const auto n = rb * 8 + 2 * t + p;
                                        if (n < rows)
                                            sums[rb][p] = __fadd_rn(sums[rb][p], __fmul_rn(value, x[static_cast<std::size_t>(n) * in + k]));
                                    }
                                }
                            }
                        }
                    }
#pragma unroll
                    for (unsigned rb = 0; rb < RB; ++rb) {
#pragma unroll
                        for (unsigned p = 0; p < 2; ++p)
                            acc[mt][rb][2 * half + p] = __fadd_rn(acc[mt][rb][2 * half + p], __fmul_rn(sums[rb][p], scale));
                    }
                }
            }
        }
    }
};

template <int RB, int MT>
__global__ void __launch_bounds__(kWarpsPerBlock * 32) fp4_gemv(const Jobs jobs, const After<const float> xa, std::uint32_t in, std::uint32_t rows) {
    extern __shared__ uint2 xs_dyn[];
    // PDL (scheduling only): this block's first weight records (constant data) are requested BEFORE pdl_wait, so the
    // weight stream overlaps the predecessor's tail. x, part and y are touched only after pdl_wait, and the dependent
    // kernel is released (pdl_trigger) only after it too (L2 footprint, common.cuh); After<> enforces the wait. The arithmetic below is unchanged.
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ji = 0;
#pragma unroll
    for (int q = 1; q < 4; ++q)
        if (q < jobs.n && blockIdx.x >= jobs.j[q].g0) ji = q;
    const Job& J = jobs.j[ji];
#include "rows_fp4_gemv.cuh"
}

template <int RB, int MT>
__global__ void __launch_bounds__(kWarpsPerBlock * 32) grouped_nvfp4(const SplitJobs jobs, std::uint32_t rows) {
    extern __shared__ uint2 xs_dyn[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const SplitJob& job = jobs.j[split_member(jobs, blockIdx.x)];
    const Job& J = job;
    const std::uint32_t in = job.in;
    const auto xa = job.x;
#include "rows_fp4_gemv.cuh"
}

template <int RB>
__global__ void __launch_bounds__(kWarpsPerBlock * 32) grouped_mxfp8(const SplitJobs jobs, std::uint32_t rows) {
    constexpr int MT = 1;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const SplitJob& J = jobs.j[split_member(jobs, blockIdx.x)];
    const std::uint32_t ks = J.ks, out = J.out, kidx = blockIdx.y;
    if (kidx >= ks) return;
    const std::uint32_t bx = blockIdx.x - J.g0, in = J.in;
    const std::uint32_t nch = in / Mxfp8Decoder::kChunk;
    const std::uint32_t c0 = static_cast<std::uint32_t>(static_cast<std::uint64_t>(kidx) * nch / ks);
    const std::uint32_t c1 = static_cast<std::uint32_t>(static_cast<std::uint64_t>(kidx + 1) * nch / ks);
    const bool on[MT] = {bx * kWarpsPerBlock + warp < out / 16};
    const PdlWaited waited = pdl_wait();
    pdl_trigger();
    const float* const x = J.x(waited);
    float* const part = J.part(waited);
    float* const y = J.y(waited);
    const float* const resid = J.resid(waited);
    const float gs0 = 1.0f;
    float acc[MT][RB][4]{};
    Mxfp8Decoder::dot<RB, MT>(J.w[0], J.scales, x, in, rows, bx, warp, lane, c0, c1, on, acc);
#include "rows_fp4_store.cuh"
}

// grid (blocks over the largest job, njobs): finalises split-K and fused jobs.
template <class Table>
__device__ __forceinline__ void reduce_split_k(const Table& jobs, std::uint32_t rows) {
    const PdlWaited waited = pdl_wait();
    pdl_trigger();  // after the wait: L2 footprint (common.cuh)
    const auto& J = jobs.j[blockIdx.y];
    if (J.ks <= 1 && J.w[1] == nullptr) return;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<std::size_t>(J.out) * rows) return;
    const std::size_t stride = static_cast<std::size_t>(J.out) * rows;
    const float* part = J.part(waited);
    const float* const resid = J.resid(waited);
    float s = part[idx];
    for (std::uint32_t k = 1; k < J.ks; ++k) s = __fadd_rn(s, part[k * stride + idx]);
    float v = __fmul_rn(s, J.gs[0]);
    if (J.w[1] != nullptr) {
        const std::size_t off = static_cast<std::size_t>(J.ks) * stride;
        float u = part[off + idx];
        for (std::uint32_t k = 1; k < J.ks; ++k) u = __fadd_rn(u, part[off + k * stride + idx]);
        v = __fmul_rn(cc_silu(v), __fmul_rn(u, J.gs[1]));
    } else if (resid != nullptr) {
        v = __fadd_rn(resid[idx], v);
    }
    J.y(waited)[idx] = v;
}

__global__ void fp4_reduce(const Jobs jobs, std::uint32_t rows) {
    reduce_split_k(jobs, rows);
}

__global__ void grouped_reduce(const SplitJobs jobs, std::uint32_t rows) {
    reduce_split_k(jobs, rows);
}

// ------------------------------------------------------------------------------------------------- weight read-ahead
// Rule and geometry: strix/decode_weight_sweep.hpp (read-ahead). The GEMV's block (bx, k, z) reads, for its 4 MT tiles
// m = bx * 4 MT + t, records [k nch / ks, (k + 1) nch / ks) (fp4_gemv above): that is readahead_piece with
// tiles_per_block = 4 MT. One thread per piece issues one bulk L2 prefetch of the piece's contiguous records. No
// register, shared or global result is written: the consumer GEMV computes exactly what it computes without it.
struct ReadaheadArgs {
    ::strix::decode::ReadaheadGrid g;
    const unsigned char* w[::strix::decode::kReadaheadMaxJobs][2];
    unsigned long long pieces;  // lead blocks x tiles_per_block
};

__global__ void fp4_readahead(const ReadaheadArgs a) {
#if __CUDA_ARCH__ >= 900
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < a.pieces;
         i += stride) {
        const ::strix::decode::ReadaheadPiece p =
            ::strix::decode::readahead_piece(a.g, i / a.g.tiles_per_block, static_cast<unsigned>(i % a.g.tiles_per_block));
        if (p.c1 <= p.c0) continue;
        const unsigned char* src =
            a.w[p.job][p.plane] + (static_cast<std::size_t>(p.tile) * a.g.chunks + p.c0) * kNvfp4RecordBytes;
        const unsigned bytes = (p.c1 - p.c0) * kNvfp4RecordBytes;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
    }
#endif
}

// STRIX_CUDA_WEIGHT_READAHEAD: bitwise (prefetch only); default ON (gain plus neutral). No-model bench, RTX PRO 6000 at
// 528 W: 4/4 configurations bitwise and faster (rows 8 -4.2 / -6.9 %, rows 16 -1.2 / -3.1 % per 3-set verify chain);
// Qwen dense 27B served decode ABAB x3, ids equal: B/A 1.0044 (B ahead in 5 of 6 runs). STRIX_CUDA_WEIGHT_READAHEAD=0 off.
// Rows 24 gains (-3.3 / -2.7 %), rows 32 loses (+2.1 / +1.6 %): consumers above 24 rows get no side node.
inline constexpr ::strix::lever::Decl readahead_lever{
    .name = "cuda_weight_readahead", .env = "STRIX_CUDA_WEIGHT_READAHEAD", .exactness = ::strix::lever::Exactness::bitwise,
    .default_on = true, .source = "tests/cuda_core/nvfp4_test.cu --readahead (bitwise, -1.2..-6.9 %); served ABAB B/A 1.0044"};
int g_readahead_mode = -1;
thread_local int t_readahead_mode = -1;  // nvfp4_thread_readahead: overrides the global mode on this thread
std::atomic<std::uint64_t> g_readahead_last{0};

bool readahead_on() {
    if (t_readahead_mode >= 0) return t_readahead_mode == 1;
    if (g_readahead_mode >= 0) return g_readahead_mode == 1;
    static const bool on = ::strix::lever::select(readahead_lever, {}).on;
    return on;
}

// Per (device, stream): the last GEMV node captured and its capture id.
struct ReadaheadSite {
    unsigned long long capture = 0;
    cudaGraphNode_t gemv = nullptr;
};
std::mutex readahead_mutex;
std::map<std::pair<int, cudaStream_t>, ReadaheadSite> readahead_sites;

cudaError_t capture_info(cudaStream_t s, cudaStreamCaptureStatus* st, unsigned long long* id, cudaGraph_t* graph,
                         const cudaGraphNode_t** deps, std::size_t* ndeps) {
#if CUDART_VERSION >= 13000
    return cudaStreamGetCaptureInfo(s, st, id, graph, deps, nullptr, ndeps);
#else
    return cudaStreamGetCaptureInfo(s, st, id, graph, deps, ndeps);
#endif
}

// Before a GEMV is captured: when the previous GEMV on this stream belongs to the same capture, add the read-ahead node
// (depends on that GEMV only, nothing depends on it: a leaf of the graph, so the stream order is unchanged).
// Consumer rows: the rule's stream-bound tiles are RB 1..3 (rows <= 24, launch_by_rows). RB4 (rows 25..32) is bound by
// its row work: no-model verify chain, RTX PRO 6000 at 528 W, bitwise 4/4: rows 24 -3.3 / -2.7 % with the side node,
// rows 32 +2.1 / +1.6 % (tests/cuda_core/nvfp4_test.cu --readahead 24 / 32), so RB4 consumers get none
// (kNvfp4ReadaheadMaxRows, cuda_core/nvfp4.hpp).
cudaError_t readahead_graph(cudaStream_t s, cudaGraph_t* graph, cudaGraphNode_t* after, int* l2) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    unsigned long long id = 0;
    const cudaGraphNode_t* deps = nullptr;
    std::size_t ndeps = 0;
    cudaError_t e = capture_info(s, &st, &id, graph, &deps, &ndeps);
    if (e != cudaSuccess || st != cudaStreamCaptureStatusActive) return e;
    int device = 0;
    if ((e = cudaGetDevice(&device)) != cudaSuccess) return e;
    {
        std::lock_guard<std::mutex> lock(readahead_mutex);
        const auto it = readahead_sites.find({device, s});
        if (it != readahead_sites.end() && it->second.capture == id) *after = it->second.gemv;
    }
    if (*after == nullptr) return cudaSuccess;
    if ((e = cudaDeviceGetAttribute(l2, cudaDevAttrL2CacheSize, device)) != cudaSuccess) return e;
    return cudaSuccess;
}

template <class Args>
cudaError_t add_readahead_node(cudaGraph_t graph, cudaGraphNode_t after, void (*kernel)(Args), Args& a,
                               unsigned long long pieces, unsigned long long bytes) {
    void* args[] = {&a};
    cudaKernelNodeParams p{};
    p.func = reinterpret_cast<void*>(kernel);
    p.gridDim = dim3(static_cast<unsigned>(std::min<unsigned long long>(64, (pieces + 255) / 256)));
    p.blockDim = dim3(256);
    p.sharedMemBytes = 0;
    p.kernelParams = args;
    cudaGraphNode_t node = nullptr;
    if (const auto e = cudaGraphAddKernelNode(&node, graph, &after, 1, &p); e != cudaSuccess) return e;
    g_readahead_last.store(bytes);
    return cudaSuccess;
}

static_assert(kNvfp4ReadaheadMaxRows == 24, "read-ahead row bound = the RB3 tile's last row count (launch_by_rows)");
cudaError_t readahead_before(const Jobs& jobs, std::uint32_t in, std::uint32_t rows, int tiles_per_block,
                             std::uint32_t gx, std::uint32_t maxks, bool fused, cudaStream_t s) {
    if (!readahead_on() || !::strix::decode::readahead_rows_ok(rows, kNvfp4ReadaheadMaxRows)) return cudaSuccess;
    cudaGraph_t graph = nullptr;
    cudaGraphNode_t after = nullptr;
    int l2 = 0;
    const auto e = readahead_graph(s, &graph, &after, &l2);
    if (e != cudaSuccess || after == nullptr) return e;
    ReadaheadArgs a{};
    a.g.jobs = static_cast<unsigned>(jobs.n);
    a.g.chunks = in >> 7;
    a.g.tiles_per_block = static_cast<unsigned>(tiles_per_block);
    a.g.gx = gx;
    a.g.gy = maxks;
    a.g.gz = fused ? 2U : 1U;
    unsigned long long total = 0;
    for (int q = 0; q < jobs.n; ++q) {
        const Job& J = jobs.j[q];
        a.g.tiles[q] = J.out / 16;
        a.g.ks[q] = J.ks;
        a.g.planes[q] = J.w[1] != nullptr ? 2U : 1U;
        a.w[q][0] = J.w[0];
        a.w[q][1] = J.w[1];
        total += static_cast<unsigned long long>(a.g.planes[q]) * nvfp4_bytes(J.out, in);
    }
    const unsigned long long budget = ::strix::decode::readahead_budget(total, static_cast<unsigned long long>(l2));
    const unsigned long long lead = ::strix::decode::readahead_lead_blocks(a.g, kNvfp4RecordBytes, budget);
    a.pieces = lead * a.g.tiles_per_block;
    if (a.pieces == 0) return cudaSuccess;
    unsigned long long bytes = 0;
    for (unsigned long long b = 0; b < lead; ++b) bytes += ::strix::decode::readahead_block_records(a.g, b) * kNvfp4RecordBytes;
    return add_readahead_node(graph, after, fp4_readahead, a, a.pieces, bytes);
}

// After a GEMV kernel was captured: remember its node (the stream's single dependency right after the launch).
cudaError_t readahead_after(cudaStream_t s) {
    if (!readahead_on()) return cudaSuccess;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    unsigned long long id = 0;
    cudaGraph_t graph = nullptr;
    const cudaGraphNode_t* deps = nullptr;
    std::size_t ndeps = 0;
    cudaError_t e = capture_info(s, &st, &id, &graph, &deps, &ndeps);
    if (e != cudaSuccess || st != cudaStreamCaptureStatusActive) return e;
    int device = 0;
    if ((e = cudaGetDevice(&device)) != cudaSuccess) return e;
    std::lock_guard<std::mutex> lock(readahead_mutex);
    readahead_sites[{device, s}] = ReadaheadSite{id, ndeps == 1 ? deps[0] : nullptr};
    return cudaSuccess;
}

// The incumbent read-ahead recipe with a K stride per job. A piece is one
// tile in dispatch order, and prefetches only the consumer's exact K interval.
struct RowsReadahead {
    SplitJobs jobs;
    unsigned gx, tiles_per_block, chunk_in;
    unsigned long long pieces;
};

struct PrefetchPiece {
    unsigned member, tile, c0, c1;
};

__host__ __device__ PrefetchPiece rows_prefetch_piece(const RowsReadahead& a, unsigned long long block,
                                                      unsigned t) {
    const auto x = static_cast<unsigned>(block % a.gx);
    const auto k = static_cast<unsigned>(block / a.gx);
    const auto member = split_member(a.jobs, x);
    const auto& job = a.jobs.j[member];
    const auto tile = (x - job.g0) * a.tiles_per_block + t;
    if (k >= job.ks || tile >= job.out / 16) return {};
    const auto chunks = job.in / a.chunk_in;
    return {member, tile, static_cast<unsigned>(static_cast<std::uint64_t>(k) * chunks / job.ks),
            static_cast<unsigned>(static_cast<std::uint64_t>(k + 1) * chunks / job.ks)};
}

template <bool Nvfp4>
__global__ void rows_readahead(const RowsReadahead a) {
#if __CUDA_ARCH__ >= 900
    auto i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (; i < a.pieces; i += stride) {
        const auto p = rows_prefetch_piece(a, i / a.tiles_per_block, static_cast<unsigned>(i % a.tiles_per_block));
        if (p.c0 == p.c1) continue;
        const auto& job = a.jobs.j[p.member];
        if constexpr (Nvfp4) {
            const auto records = static_cast<std::size_t>(p.tile) * (job.in / a.chunk_in) + p.c0;
            const auto* src = job.w[0] + records * kNvfp4RecordBytes;
            const auto bytes = (p.c1 - p.c0) * kNvfp4RecordBytes;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
        } else {
            const auto bytes = (p.c1 - p.c0) * a.chunk_in;
            for (unsigned o = 0; o < 16; ++o) {
                const auto* src = job.w[0] + (static_cast<std::size_t>(p.tile) * 16 + o) * job.in + p.c0 * a.chunk_in;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
            }
        }
    }
#endif
}

template <bool Nvfp4>
cudaError_t rows_readahead_before(RowsReadahead a, unsigned gy, std::uint32_t rows, cudaStream_t s) {
    if (!readahead_on() || !decode::readahead_rows_ok(rows, kNvfp4ReadaheadMaxRows)) return cudaSuccess;
#if !defined(STRIX_CUDA_NATIVE_NVFP4)
    if (!cuda::arch::current_device_at_least<9>()) return cudaSuccess;
#endif
    cudaGraph_t graph = nullptr;
    cudaGraphNode_t after = nullptr;
    int l2 = 0;
    const auto status = readahead_graph(s, &graph, &after, &l2);
    if (status != cudaSuccess || after == nullptr) return status;

    constexpr unsigned record_bytes = Nvfp4 ? kNvfp4RecordBytes : 16 * Mxfp8Decoder::kChunk;
    unsigned long long total_bytes = 0;
    for (int q = 0; q < a.jobs.n; ++q) {
        const auto& job = a.jobs.j[q];
        total_bytes += static_cast<unsigned long long>(job.out / 16) * (job.in / a.chunk_in) * record_bytes;
    }
    const auto budget = decode::readahead_budget(total_bytes, static_cast<unsigned long long>(l2));
    const auto blocks = static_cast<unsigned long long>(a.gx) * gy;
    unsigned long long bytes = 0;
    for (unsigned long long block = 0; block < blocks; ++block) {
        unsigned long long block_bytes = 0;
        for (unsigned t = 0; t < a.tiles_per_block; ++t) {
            const auto p = rows_prefetch_piece(a, block, t);
            block_bytes += static_cast<unsigned long long>(p.c1 - p.c0) * record_bytes;
        }
        if (block_bytes > budget - bytes) break;
        bytes += block_bytes;
        a.pieces += a.tiles_per_block;
        if (bytes == budget) break;
    }
    if (bytes == 0) return cudaSuccess;
    return add_readahead_node(graph, after, rows_readahead<Nvfp4>, a, a.pieces, bytes);
}

using WorkspaceKey = std::pair<int, cudaStream_t>;
std::mutex workspace_mutex;
std::map<WorkspaceKey, std::unique_ptr<Nvfp4Workspace>> workspaces;

template <int RB, int MT>
cudaError_t launch_gemv(Jobs& jobs, const float* x, std::uint32_t in, std::uint32_t rows, cudaStream_t s) {
    std::uint32_t gx = 0, maxks = 1, maxout = 0;
    bool fused = false, any_split = false;
    for (int q = 0; q < jobs.n; ++q) {
        jobs.j[q].g0 = gx;
        gx += (jobs.j[q].out / 16 + kWarpsPerBlock * MT - 1) / (kWarpsPerBlock * MT);
        maxks = std::max(maxks, jobs.j[q].ks);
        maxout = std::max(maxout, jobs.j[q].out);
        fused = fused || jobs.j[q].w[1] != nullptr;
        any_split = any_split || jobs.j[q].ks > 1;
    }
    // Both launches carry the programmatic-serialization attribute (off switch: set_programmatic_launch(false) gives the
    // previous fully serialised form; same kernels, same bits).
    cudaError_t e = readahead_before(jobs, in, rows, kWarpsPerBlock * MT, gx, maxks, fused, s);
    if (e != cudaSuccess) return e;
    e = launch_pdl_smem(fp4_gemv<RB, MT>, dim3(gx, maxks, fused ? 2 : 1), dim3(kWarpsPerBlock * 32),
                        static_cast<std::size_t>(kSlab) * RB * 2048, s, jobs, x, in, rows);
    if (e == cudaSuccess) e = readahead_after(s);
    if (e != cudaSuccess || (!any_split && !fused)) return e;
    const std::size_t n = static_cast<std::size_t>(maxout) * rows;
    return launch_pdl(fp4_reduce, dim3(static_cast<unsigned>((n + 255) / 256), static_cast<unsigned>(jobs.n)), dim3(256), s,
                      jobs, rows);
}

cudaError_t launch_by_rows(Jobs& jobs, const float* x, std::uint32_t in, std::uint32_t rows, cudaStream_t s) {
    if (rows <= 8) return launch_gemv<1, 4>(jobs, x, in, rows, s);
    if (rows <= 16) return launch_gemv<2, 4>(jobs, x, in, rows, s);
    if (rows <= 24) return launch_gemv<3, 2>(jobs, x, in, rows, s);
    return launch_gemv<4, 2>(jobs, x, in, rows, s);
}

bool ok(const DenseWeight& w, std::uint32_t rows) {
    return w.data != nullptr && w.format == WeightFormat::nvfp4 && rows >= 1 && rows <= kNvfp4MaxRows &&
           nvfp4_shape_ok(w.out, w.in);
}

cudaError_t ensure_ws(std::size_t bytes, cudaStream_t stream, float** data = nullptr) {
    int device = 0;
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    std::lock_guard<std::mutex> lock(workspace_mutex);
    auto& owner = workspaces[{device, stream}];
    if (!owner) owner = std::make_unique<Nvfp4Workspace>();
    error = owner->reserve(bytes);
    if (error == cudaSuccess && data) *data = owner->data();
    return error;
}

// ----------------------------------------------------------------------------------------------------- converter
__global__ void amax_kernel_bf16(const std::uint16_t* src, std::size_t n, std::uint32_t* amax) {
    float m = 0.f;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        m = fmaxf(m, fabsf(__uint_as_float(static_cast<std::uint32_t>(src[i]) << 16)));
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if ((threadIdx.x & 31) == 0) atomicMax(amax, __float_as_uint(m));
}
__global__ void amax_kernel_f32(const float* src, std::size_t n, std::uint32_t* amax) {
    float m = 0.f;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        m = fmaxf(m, fabsf(src[i]));
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if ((threadIdx.x & 31) == 0) atomicMax(amax, __float_as_uint(m));
}

// one thread per (output row o, group G of 16 inputs)
template <bool BF16>
__global__ void convert_kernel(const void* src, std::uint32_t out, std::uint32_t in, const std::uint32_t* amax_bits,
                               unsigned char* dst, float* gs_out) {
    const std::size_t ngroups = static_cast<std::size_t>(out) * (in >> 4);
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const float amax_t = __uint_as_float(*amax_bits);
    const float gs = amax_t > 0.f ? __fdiv_rn(amax_t, 6.0f * 448.0f) : 1.0f;
    if (idx == 0) *gs_out = gs;
    if (idx >= ngroups) return;
    const std::uint32_t G = static_cast<std::uint32_t>(idx % (in >> 4));
    const std::uint32_t o = static_cast<std::uint32_t>(idx / (in >> 4));
    float w[16];
    if constexpr (BF16) {
        const uint4* p = reinterpret_cast<const uint4*>(static_cast<const std::uint16_t*>(src) + static_cast<std::size_t>(o) * in + G * 16);
        const uint4 a = __ldg(p), b = __ldg(p + 1);
        const std::uint32_t u[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w[2 * i] = bf16_lo(u[i]);
            w[2 * i + 1] = bf16_hi(u[i]);
        }
    } else {
        const float4* p = reinterpret_cast<const float4*>(static_cast<const float*>(src) + static_cast<std::size_t>(o) * in + G * 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 v = __ldg(p + i);
            w[4 * i] = v.x; w[4 * i + 1] = v.y; w[4 * i + 2] = v.z; w[4 * i + 3] = v.w;
        }
    }
    float am = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) am = fmaxf(am, fabsf(w[i]));
    const std::uint32_t sb = f32_to_e4m3(__fdiv_rn(am, __fmul_rn(6.0f, gs)));
    const float sd = __half2float(__ushort_as_half(static_cast<unsigned short>(e4m3x2_to_f16x2(sb) & 0xFFFFu)));
    const float den = __fmul_rn(sd, gs);
    std::uint32_t code[16];
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
        if (sd > 0.f) {
            const std::uint32_t b = f32x2_to_e2m1x2(__fdiv_rn(w[i + 1], den), __fdiv_rn(w[i], den));
            code[i] = b & 0xF;
            code[i + 1] = b >> 4;
        } else {
            code[i] = code[i + 1] = 0;
        }
    }
    const std::uint32_t m = o >> 4, r = o & 15, h = r >> 3, gg = r & 7, c = G >> 3, j = G & 7;
    unsigned char* rec = dst + (static_cast<std::size_t>(m) * (in >> 7) + c) * kNvfp4RecordBytes;
    rec[1024 + gg * 16 + h * 8 + j] = static_cast<unsigned char>(sb);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const std::uint32_t lane = static_cast<std::uint32_t>(gg * 4 + t);
        // pair p=0: inputs 2t,2t+1 ; p=1: inputs 2t+8, 2t+9
        rec[h * 512 + lane * 16 + 4 * (j >> 1) + 2 * (j & 1) + 0] = static_cast<unsigned char>(code[2 * t] | (code[2 * t + 1] << 4));
        rec[h * 512 + lane * 16 + 4 * (j >> 1) + 2 * (j & 1) + 1] = static_cast<unsigned char>(code[2 * t + 8] | (code[2 * t + 9] << 4));
    }
}

template <bool BF16>
cudaError_t quantize(const void* src, std::uint32_t out, std::uint32_t in, void* dst, float* gscale, cudaStream_t s) {
    if (!nvfp4_shape_ok(out, in) || src == nullptr || dst == nullptr || gscale == nullptr) return cudaErrorInvalidValue;
    if (const auto e = cuda::arch::native_nvfp4_status(); e != cudaSuccess) return e;
    std::uint32_t* d_amax = nullptr;
    float* d_gs = nullptr;
    cudaError_t e = cudaMallocAsync(reinterpret_cast<void**>(&d_amax), 8, s);
    if (e != cudaSuccess) return e;
    d_gs = reinterpret_cast<float*>(d_amax + 1);
    cudaMemsetAsync(d_amax, 0, 4, s);
    const std::size_t n = static_cast<std::size_t>(out) * in;
    const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((n + 255) / 256, 4096));
    if constexpr (BF16) amax_kernel_bf16<<<blocks, 256, 0, s>>>(static_cast<const std::uint16_t*>(src), n, d_amax);
    else amax_kernel_f32<<<blocks, 256, 0, s>>>(static_cast<const float*>(src), n, d_amax);
    const std::size_t ng = static_cast<std::size_t>(out) * (in >> 4);
    convert_kernel<BF16><<<static_cast<unsigned>((ng + 127) / 128), 128, 0, s>>>(src, out, in, d_amax, static_cast<unsigned char*>(dst), d_gs);
    e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaMemcpyAsync(gscale, d_gs, 4, cudaMemcpyDeviceToHost, s);
    if (e == cudaSuccess) e = cudaStreamSynchronize(s);
    cudaFreeAsync(d_amax, s);
    return e;
}

}  // namespace

bool nvfp4_supported() { return cuda::arch::native_nvfp4_status() == cudaSuccess; }

void nvfp4_set_split_target(std::uint32_t warps) { g_target_warps = warps ? warps : split_k::kExactTarget; }

cudaError_t nvfp4_device_key(split_k::Device* key) {
    int dev = 0, sms = 0, major = 0, minor = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e == cudaSuccess) e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (e == cudaSuccess) e = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (e == cudaSuccess) e = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    if (e != cudaSuccess) return e;
    *key = {static_cast<std::uint32_t>(sms), static_cast<std::uint32_t>(10 * major + minor)};
    return cudaSuccess;
}

cudaError_t nvfp4_set_split_profile(bool fast) {
    // the key is recorded under EXACT too (reporting only: target() ignores the device under EXACT)
    split_k::Device d{};
    if (const cudaError_t e = nvfp4_device_key(&d); e != cudaSuccess) return e;
    g_split_device = d;
    g_split_profile = fast ? split_k::Profile::fast : split_k::Profile::exact;
    return cudaSuccess;
}

const char* nvfp4_split_identity() { return split_k::identity(g_split_profile, g_split_device).data(); }

split_k::Device nvfp4_split_device() { return g_split_device; }
void nvfp4_set_readahead(int mode) { g_readahead_mode = mode < 0 ? -1 : (mode > 0 ? 1 : 0); }
int nvfp4_thread_readahead(int mode) {
    const int prev = t_readahead_mode;
    t_readahead_mode = mode < 0 ? -1 : (mode > 0 ? 1 : 0);
    return prev;
}
std::uint64_t nvfp4_take_readahead_bytes() { return g_readahead_last.exchange(0); }

std::uint32_t nvfp4_split_k(std::uint32_t out, std::uint32_t in) {
    return split_k::factor(split_k::target(g_split_profile, g_split_device, out, in, g_target_warps), out, in);
}

cudaError_t nvfp4_quantize_bf16(const void* bf16, std::uint32_t out, std::uint32_t in, void* dst, float* gscale,
                                cudaStream_t s) {
    return quantize<true>(bf16, out, in, dst, gscale, s);
}
cudaError_t nvfp4_quantize_f32(const float* src, std::uint32_t out, std::uint32_t in, void* dst, float* gscale,
                               cudaStream_t s) {
    return quantize<false>(src, out, in, dst, gscale, s);
}

cudaError_t nvfp4_reserve_workspace(std::size_t bytes, cudaStream_t stream) { return ensure_ws(bytes, stream); }
void nvfp4_release_workspace(cudaStream_t stream) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return;
    std::lock_guard<std::mutex> lock(workspace_mutex);
    workspaces.erase({device, stream});
}

namespace {
// Shared by the single, fused and multi entry points: builds the job table (split-K partial buffers carved from the
// workspace) and launches. ks per job is nvfp4_split_k(out, in), independent of rows.
cudaError_t run_jobs(Jobs& jobs, const float* x, std::uint32_t in, std::uint32_t rows, cudaStream_t s) {
    if (const auto e = cuda::arch::native_nvfp4_status(); e != cudaSuccess) return e;
    std::size_t need = 0;
    for (int q = 0; q < jobs.n; ++q) {
        Job& J = jobs.j[q];
        J.ks = nvfp4_split_k(J.out, in);
        if (J.ks > 1 || J.w[1] != nullptr) need += static_cast<std::size_t>(J.w[1] ? 2 : 1) * J.ks * rows * J.out * 4;
    }
    float* workspace = nullptr;
    cudaError_t e = need ? ensure_ws(need, s, &workspace) : cudaSuccess;
    if (e != cudaSuccess) return e;
    std::size_t off = 0;
    for (int q = 0; q < jobs.n; ++q) {
        Job& J = jobs.j[q];
        J.part = nullptr;
        if (J.ks > 1 || J.w[1] != nullptr) {
            J.part = workspace + off / 4;
            off += static_cast<std::size_t>(J.w[1] ? 2 : 1) * J.ks * rows * J.out * 4;
        }
    }
    return launch_by_rows(jobs, x, in, rows, s);
}
}  // namespace

cudaError_t nvfp4_dense_rows_multi(const Nvfp4Job* jobs, std::uint32_t n, const float* x, std::uint32_t rows, cudaStream_t s) {
    if (n == 0 || n > 4) return cudaErrorInvalidValue;
    Jobs J{};
    J.n = static_cast<int>(n);
    for (std::uint32_t q = 0; q < n; ++q) {
        const DenseWeight& w = jobs[q].w;
        if (!ok(w, rows) || w.in != jobs[0].w.in) return cudaErrorInvalidValue;
        J.j[q].w[0] = static_cast<const unsigned char*>(w.data);
        J.j[q].w[1] = nullptr;
        J.j[q].y = jobs[q].y;
        J.j[q].resid = jobs[q].residual;
        J.j[q].gs[0] = w.gscale;
        J.j[q].gs[1] = 1.0f;
        J.j[q].out = w.out;
    }
    return run_jobs(J, x, jobs[0].w.in, rows, s);
}

cudaError_t nvfp4_dense_rows(const DenseWeight& w, const float* x, float* y, std::uint32_t rows, const float* residual,
                             cudaStream_t s) {
    const Nvfp4Job job{w, y, residual};
    return nvfp4_dense_rows_multi(&job, 1, x, rows, s);
}

cudaError_t nvfp4_gate_up_silu_rows(const DenseWeight& gate, const DenseWeight& up, const float* x, float* y,
                                    std::uint32_t rows, cudaStream_t s) {
    if (!ok(gate, rows) || !ok(up, rows) || gate.out != up.out || gate.in != up.in) return cudaErrorInvalidValue;
    Jobs J{};
    J.n = 1;
    J.j[0].w[0] = static_cast<const unsigned char*>(gate.data);
    J.j[0].w[1] = static_cast<const unsigned char*>(up.data);
    J.j[0].y = y;
    J.j[0].resid = nullptr;
    J.j[0].gs[0] = gate.gscale;
    J.j[0].gs[1] = up.gscale;
    J.j[0].out = gate.out;
    return run_jobs(J, x, gate.in, rows, s);
}

namespace {

namespace rows = core::rows;

bool aligned(const void* p, std::uintptr_t bytes) {
    return reinterpret_cast<std::uintptr_t>(p) % bytes == 0;
}

bool valid_rows_launch(const rows::Launch& launch) {
    for (unsigned q = 0; q < launch.tiles.count; ++q) {
        const auto& job = launch.jobs[q];
        const bool bf16 = job.w.has_format(weight::kBf16WeightStorage);
        const bool q8 = job.w.has_format(weight::kQ8WeightStorage);
        if (launch.form == rows::Form::row_tile && !bf16 && !q8) return false;
        const auto alignment = launch.form == rows::Form::nvfp4_split_k ? Nvfp4Decoder::kAlign :
                               q8 && job.in % 256 != 0 ? 2U : Mxfp8Decoder::kAlign;
        if (!aligned(job.w.data, alignment) || !aligned(job.x.data, 16) || !aligned(job.y.data, alignof(float)))
            return false;
        if (job.epilogue != 0 &&
            !(job.epilogue == kRowsAddResidual && job.aux[0].data && aligned(job.aux[0].data, alignof(float))) &&
            !(job.epilogue == kRowsRoundBf16 && bf16)) return false;
    }
    return true;
}

std::size_t split_workspace(const rows::Launch& launch, std::uint32_t count) {
    std::size_t bytes = 0;
    for (unsigned q = 0; q < launch.tiles.count; ++q) {
        const auto ks = launch.splits[q];
        if (ks > 1)
            bytes += static_cast<std::size_t>(ks) * count * launch.jobs[q].out * sizeof(float);
    }
    return bytes;
}

template <bool Nvfp4>
cudaError_t run_split_rows(const rows::Launch& launch, float* scratch, std::uint32_t count, cudaStream_t stream) {
    RowsReadahead ahead{};
    ahead.tiles_per_block = launch.tiles_per_block;
    ahead.chunk_in = launch.chunk_in;
    auto& jobs = ahead.jobs;
    jobs.n = static_cast<int>(launch.tiles.count);
    unsigned max_ks = 1, max_out = 0;
    for (unsigned q = 0; q < launch.tiles.count; ++q) {
        const auto& job = launch.jobs[q];
        auto& j = jobs.j[q];
        j.w[0] = static_cast<const unsigned char*>(job.w.data);
        j.scales = static_cast<const unsigned char*>(job.w.scales);
        j.in = job.in;
        j.x = static_cast<const float*>(job.x.data);
        j.y = const_cast<float*>(static_cast<const float*>(job.y.data));
        j.resid = job.epilogue == kRowsAddResidual ? static_cast<const float*>(job.aux[0].data) : nullptr;
        j.gs[0] = Nvfp4 ? job.w.outer_scale : 1.0f;
        j.out = job.out;
        j.ks = launch.splits[q];
        j.g0 = ahead.gx;
        ahead.gx += rows::output_blocks(job.out / rows::kTileOut, launch.tiles_per_block);
        max_ks = std::max(max_ks, j.ks);
        max_out = std::max(max_out, j.out);
        if (j.ks > 1) {
            j.part = scratch;
            scratch += static_cast<std::size_t>(j.ks) * count * j.out;
        }
    }
    auto status = rows_readahead_before<Nvfp4>(ahead, max_ks, count, stream);
    if (status != cudaSuccess) return status;

    const dim3 grid(ahead.gx, max_ks), block(kWarpsPerBlock * 32);
    const auto launch_rows = [&]<int RB>() {
        if constexpr (Nvfp4) {
            constexpr int MT = RB <= 2 ? 4 : 2;
            return launch_pdl_smem(grouped_nvfp4<RB, MT>, grid, block, kSlab * RB * 2048, stream, jobs, count);
        } else {
            return launch_pdl(grouped_mxfp8<RB>, grid, block, stream, jobs, count);
        }
    };
    if (count <= 8) status = launch_rows.template operator()<1>();
    else if (count <= 16) status = launch_rows.template operator()<2>();
    else if (count <= 24) status = launch_rows.template operator()<3>();
    else status = launch_rows.template operator()<4>();
    if (status == cudaSuccess) status = readahead_after(stream);
    if (status != cudaSuccess || max_ks == 1) return status;

    const auto n = static_cast<std::size_t>(max_out) * count;
    return launch_pdl(grouped_reduce, dim3(static_cast<unsigned>((n + 255) / 256), launch.tiles.count),
                      dim3(256), stream, jobs, count);
}

}  // namespace

std::size_t rows_split_k_workspace(const core::rows::Plan& plan) {
    std::size_t bytes = 0;
    for (const auto& launch : plan.launches())
        bytes = std::max(bytes, split_workspace(launch, plan.rows()));
    return bytes;
}

cudaError_t cuda_rows_grouped(const core::rows::Plan& plan, float* workspace, std::size_t workspace_bytes,
                              cudaStream_t stream) {
    const auto needed = rows_split_k_workspace(plan);
    if (plan.launches().empty() || plan.prologue() != 0 || needed > workspace_bytes ||
        (needed != 0 && (!workspace || !aligned(workspace, alignof(float))))) return cudaErrorInvalidValue;

    // Admit all jobs before submitting any work, including the step-1 forms.
    for (const auto& launch : plan.launches())
        if (!valid_rows_launch(launch)) return cudaErrorInvalidValue;
    for (const auto& launch : plan.launches()) {
        if (launch.form != rows::Form::nvfp4_split_k) continue;
        if (const auto e = cuda::arch::native_nvfp4_status(); e != cudaSuccess) return e;
        break;
    }

    for (std::size_t i = 0; i < plan.launches().size(); ++i) {
        const auto& launch = plan.launches()[i];
        cudaError_t status = cudaSuccess;
        switch (launch.form) {
        case rows::Form::row_tile:
            status = dense_mma3_rows_grouped(plan.select_launch(i), stream);
            break;
        case rows::Form::nvfp4_split_k:
            status = run_split_rows<true>(launch, workspace, plan.rows(), stream);
            break;
        case rows::Form::mxfp8_split_k:
            status = run_split_rows<false>(launch, workspace, plan.rows(), stream);
            break;
        }
        if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
}

namespace {

const rows::Launch* mxfp8_launch(const rows::RowsPlan& plan, std::size_t index) {
    if (index >= plan.launches().size()) return nullptr;
    const auto& launch = plan.launches()[index].launch;
    if (launch.form != rows::Form::mxfp8_split_k) return nullptr;
    constexpr unsigned max_grid_y = 65535;
    for (unsigned q = 0; q < launch.tiles.count; ++q)
        if (launch.splits[q] > max_grid_y) return nullptr;
    return &launch;
}

}  // namespace

std::size_t mxfp8_rows_workspace(const rows::RowsPlan& plan, std::size_t index) {
    const auto* launch = mxfp8_launch(plan, index);
    return launch ? split_workspace(*launch, plan.rows()) : 0;
}

cudaError_t cuda_mxfp8_rows(const rows::RowsPlan& plan, std::size_t index, float* workspace,
                            std::size_t workspace_bytes, cudaStream_t stream) {
    const auto* launch = mxfp8_launch(plan, index);
    if (!launch || !valid_rows_launch(*launch)) return cudaErrorInvalidValue;
    const auto needed = split_workspace(*launch, plan.rows());
    if (needed > workspace_bytes || (needed != 0 && (!workspace || !aligned(workspace, alignof(float)))))
        return cudaErrorInvalidValue;
    return run_split_rows<false>(*launch, workspace, plan.rows(), stream);
}

// ------------------------------------------------------------------------------------------------- host reference
float nvfp4_host_e4m3(unsigned char b) {
    const int e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? std::ldexp(static_cast<float>(m), -9) : std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
    if (e == 15 && m == 7) v = std::nanf("");
    return (b & 0x80) ? -v : v;
}

static const float kE2m1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

unsigned char nvfp4_host_quantize_group(const float* w16, float gs, unsigned char* codes16) {
    float am = 0.f;
    for (int i = 0; i < 16; ++i) am = std::max(am, std::fabs(w16[i]));
    const float target = am / (6.0f * gs);
    // e4m3 RN-even with saturation at 448: search the 126 finite non-negative codes for the nearest (ties to even code)
    int best = 0;
    float bd = std::fabs(target - 0.0f);
    for (int c = 1; c < 127; ++c) {
        const float d = std::fabs(target - nvfp4_host_e4m3(static_cast<unsigned char>(c)));
        if (d < bd || (d == bd && (c & 1) == 0)) {
            bd = d;
            best = c;
        }
    }
    const float sd = nvfp4_host_e4m3(static_cast<unsigned char>(best));
    const float den = sd * gs;
    for (int i = 0; i < 16; ++i) {
        if (!(sd > 0.f)) {
            codes16[i] = 0;
            continue;
        }
        const float v = w16[i] / den, a = std::fabs(v);
        int c = a <= 0.25f ? 0 : a < 0.75f ? 1 : a <= 1.25f ? 2 : a < 1.75f ? 3 : a <= 2.5f ? 4 : a < 3.5f ? 5 : a <= 5.0f ? 6 : 7;
        if (std::signbit(v)) c |= 8;
        codes16[i] = static_cast<unsigned char>(c);
    }
    return static_cast<unsigned char>(best);
}

float nvfp4_host_dequant(const unsigned char* tiles, std::uint32_t in, std::uint32_t o, std::uint32_t k) {
    const std::uint32_t m = o >> 4, r = o & 15, h = r >> 3, g = r & 7, c = k >> 7, kl = k & 127, j = kl >> 4, kk = kl & 15;
    const std::uint32_t t = (kk & 7) >> 1, p = kk >> 3, lane = g * 4 + t;
    const unsigned char* rec = tiles + (static_cast<std::size_t>(m) * (in >> 7) + c) * kNvfp4RecordBytes;
    const unsigned char byte = rec[h * 512 + lane * 16 + 4 * (j >> 1) + 2 * (j & 1) + p];
    const unsigned nib = (kk & 1) ? (byte >> 4) : (byte & 15);
    const float v = kE2m1[nib & 7] * ((nib & 8) ? -1.0f : 1.0f);
    return v * nvfp4_host_e4m3(rec[1024 + g * 16 + h * 8 + j]);
}

// ---------------------------------------------------------------------------------------------- expansion (prefill GEMM)
namespace {
// one warp per record; lane (g, t) holds row 8h + g of output tile m, groups j = 0..7, pairs p = 0, 1.
__global__ void __launch_bounds__(256) expand_f16_kernel(const unsigned char* w, std::uint32_t out, std::uint32_t in, __half* dst) {
    const std::uint32_t rec_id = blockIdx.x * 8 + (threadIdx.x >> 5);
    const std::uint32_t nch = in >> 7, ntiles = out >> 4;
    if (rec_id >= ntiles * nch) return;
    const std::uint32_t m = rec_id / nch, c = rec_id % nch;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const unsigned char* rec = w + static_cast<std::size_t>(rec_id) * kNvfp4RecordBytes;
    const uint4 d[2] = {*reinterpret_cast<const uint4*>(rec + lane * 16), *reinterpret_cast<const uint4*>(rec + 512 + lane * 16)};
    const uint4 sc = *reinterpret_cast<const uint4*>(rec + 1024 + g * 16);
    const std::uint32_t scw[4] = {sc.x, sc.y, sc.z, sc.w};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const std::uint32_t dw[4] = {d[h].x, d[h].y, d[h].z, d[h].w};
        __half* row = dst + static_cast<std::size_t>(m * 16 + 8 * h + g) * in + (static_cast<std::size_t>(c) << 7);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            std::uint32_t o[4];
            e2m1_word(dw[i], o);
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int j = 2 * i + (b >> 1), p = b & 1;
                const int sb = 8 * h + j;
                const std::uint32_t s2 = e4m3x2_to_f16x2((scw[sb >> 2] >> (8 * (sb & 3))) & 0xFFFFu);
                const std::uint32_t v = hmul2(o[b], prmt(s2, 0x1010));
                *reinterpret_cast<std::uint32_t*>(row + 32 * (j >> 1) + 8 * t + 4 * (j & 1) + 2 * p) = v;
            }
        }
    }
}
}  // namespace

cudaError_t nvfp4_expand_f16(const DenseWeight& w, void* dst_f16, cudaStream_t stream) {
    if (w.format != WeightFormat::nvfp4 || !nvfp4_shape_ok(w.out, w.in)) return cudaErrorInvalidValue;
    if (const auto e = cuda::arch::native_nvfp4_status(); e != cudaSuccess) return e;
    const std::uint32_t recs = (w.out >> 4) * (w.in >> 7);
    expand_f16_kernel<<<(recs + 7) / 8, 256, 0, stream>>>(static_cast<const unsigned char*>(w.data), w.out, w.in, static_cast<__half*>(dst_f16));
    return cudaGetLastError();
}


namespace {
// FAST/TURBO e4m3 expansion (src/cuda/core/dense/dense_big.cu big_kernel_f8): one warp per record, lane (g, t) writes 32 contiguous
// bytes per half-tile row: position 32t + 4j + 2p + e of a 128-input block holds input 16j + 8p + 2t + e. Value = e2m1 * e4m3 / 8
// rounded to e4m3 (the 1/8 keeps 6 * 448 inside the e4m3 range; the GEMM epilogue multiplies it back together with gscale).
__global__ void __launch_bounds__(256) expand_e4m3_kernel(const unsigned char* w, std::uint32_t out, std::uint32_t in, unsigned char* dst) {
    const std::uint32_t rec_id = blockIdx.x * 8 + (threadIdx.x >> 5);
    const std::uint32_t nch = in >> 7, ntiles = out >> 4;
    if (rec_id >= ntiles * nch) return;
    const std::uint32_t m = rec_id / nch, c = rec_id % nch;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const unsigned char* rec = w + static_cast<std::size_t>(rec_id) * kNvfp4RecordBytes;
    const uint4 d[2] = {*reinterpret_cast<const uint4*>(rec + lane * 16), *reinterpret_cast<const uint4*>(rec + 512 + lane * 16)};
    const uint4 sc = *reinterpret_cast<const uint4*>(rec + 1024 + g * 16);
    const std::uint32_t scw[4] = {sc.x, sc.y, sc.z, sc.w};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const std::uint32_t dw[4] = {d[h].x, d[h].y, d[h].z, d[h].w};
        unsigned char* row = dst + static_cast<std::size_t>(m * 16 + 8 * h + g) * in + (static_cast<std::size_t>(c) << 7) + 32 * t;
        std::uint32_t ob[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            std::uint32_t o[4];
            e2m1_word(dw[i], o);
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int j = 2 * i + (b >> 1);
                const int sb = 8 * h + j;
                const std::uint32_t s2 = e4m3x2_to_f16x2((scw[sb >> 2] >> (8 * (sb & 3))) & 0xFFFFu);
                std::uint32_t v = hmul2(o[b], prmt(s2, 0x1010));
                v = hmul2(v, 0x30003000u);  // * 0.125 (exact in f16)
                std::uint32_t r;
#if defined(STRIX_CUDA_NATIVE_NVFP4)
                asm("{\n .reg .b16 h;\n cvt.rn.satfinite.e4m3x2.f16x2 h, %1;\n cvt.u32.u16 %0, h;\n}" : "=r"(r) : "r"(v));
#else
                (void)v;
                r = 0;
                __trap();
#endif
                const int k = 4 * i + b;  // pair index in the thread's 16 pairs
                if (k & 1) ob[k >> 1] |= (r & 0xFFFFu) << 16;
                else ob[k >> 1] = r & 0xFFFFu;
            }
        }
        uint4* o4 = reinterpret_cast<uint4*>(row);
        o4[0] = make_uint4(ob[0], ob[1], ob[2], ob[3]);
        o4[1] = make_uint4(ob[4], ob[5], ob[6], ob[7]);
    }
}
}  // namespace

cudaError_t nvfp4_expand_e4m3(const DenseWeight& w, void* dst, cudaStream_t stream) {
    if (w.format != WeightFormat::nvfp4 || !nvfp4_shape_ok(w.out, w.in)) return cudaErrorInvalidValue;
    if (const auto e = cuda::arch::native_nvfp4_status(); e != cudaSuccess) return e;
    const std::uint32_t recs = (w.out >> 4) * (w.in >> 7);
    expand_e4m3_kernel<<<(recs + 7) / 8, 256, 0, stream>>>(static_cast<const unsigned char*>(w.data), w.out, w.in, static_cast<unsigned char*>(dst));
    return cudaGetLastError();
}

}  // namespace strix::cuda_core
