// CUDA QSA layer core, bit-identical to the HIP single-A8-profile tiled prefill (see include/strix/cuda/qwen_qsa.hpp).
// HIP references: src/hip_indexed_attention_launcher.hip (H: line numbers below), ISA
// the gfx1151 disassembly of hip_indexed_attention_launcher (S: line numbers).
// Every FP operation is an explicit IEEE intrinsic (__fadd_rn, __fmul_rn, __fmaf_rn, __fdiv_rn) in the HIP order.
#include "../../core/attn_rowgroup_launch.cuh"
#include "../../core/select_rows.cuh"
#include "qsa_fused_order.hpp"
#include "strix/env_flags.hpp"
#include "strix/cuda/core/decode_rows.hpp"
#include "strix/lever.hpp"
#include "strix/cuda/qwen_qsa.hpp"
#include "qwen_qsa_rowgroup.hpp"
#include "qwen_qsa_segments.hpp"
#include "strix/cuda/vram_guard.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace strix::cuda::qwen_qsa {
namespace {

namespace am = strix::cuda::amdmath;
namespace selection = strix::cuda::core::select;

constexpr unsigned kFull = 0xffffffffU;
#define kInf __uint_as_float(0x7f800000U)

__device__ __forceinline__ float bf16f(std::uint16_t b) { return __uint_as_float(static_cast<std::uint32_t>(b) << 16U); }
// H:328-343 float_to_bf16: (bits + 0x7fff + lsb) >> 16, no NaN special case.
__device__ __forceinline__ std::uint16_t to_bf16(float v) {
    const auto b = __float_as_uint(v);
    return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U);
}
__device__ __forceinline__ float round_bf16(float v) { return bf16f(to_bf16(v)); }

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("qwen_qsa: ") + what + ": " + cudaGetErrorString(e));
}

// ------------------------------------------------------------------------------------------------------------------
// Indexer projection: linear_to_float_rows_kernel<bf16,16> (H:3566-3591; S:182274 = ..._kernelILNS0_10WeightKindE2ELj16E..).
// Per (row, output o): thread t (256) keeps sum = fma(act[t+256k], w[o][t+256k], sum), k = 0..9 ascending from 0
// (S:182477-182494 v_fmac_f32 vSUM, vW, vACT: contracted); then the LDS tree strides 128, 64, ..., 1
// (scratch[t] += scratch[t+stride], S:182536-), round_bf16. One addition commutes, so any (lower, upper) operand order.
// CUDA: block = 8 rows x a slice of 64 outputs, 4 outputs per pass; the 8 activation rows staged once in shared memory
// (BF16), each activation read from shared memory feeds 4 FMAs (4 weight rows in registers); strides 128/64/32 in shared
// memory, 16..1 by shuffles (4 (row, output) pairs per warp).
constexpr unsigned kIdxRows = 8, kIdxOuts = 4, kIdxOutputs = kIndexQWidth + kIndexDim, kIdxSlice = 64;
constexpr unsigned kIdxPairs = kIdxRows * kIdxOuts;  // 32
constexpr std::size_t kIdxSmem = kIdxRows * kHidden * sizeof(std::uint16_t) + kIdxPairs * 256U * sizeof(float);
// Decode form (rows <= 8): kIdxWide outputs per CTA (same per-output arithmetic), 640/kIdxWide CTAs instead of 10.
constexpr unsigned kIdxWide = 4;
static_assert(kIdxSlice % kIdxOuts == 0 && kIdxOutputs % kIdxSlice == 0 && kIdxWide % kIdxOuts == 0 && kIdxOutputs % kIdxWide == 0);

template <unsigned kSlice>
__global__ __launch_bounds__(256) void indexer_kernel(const std::uint16_t* __restrict__ hidden, std::uint32_t rows,
    const std::uint16_t* __restrict__ wq, const std::uint16_t* __restrict__ wk, float* __restrict__ iq, float* __restrict__ ik) {
    extern __shared__ __align__(16) unsigned char smem[];
    auto* act = reinterpret_cast<std::uint16_t*>(smem);
    auto* red = reinterpret_cast<float*>(smem + kIdxRows * kHidden * sizeof(std::uint16_t));  // [pair = r*4+oo][256]
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5U;
    const std::uint32_t r0 = blockIdx.x * kIdxRows;
    const unsigned nr = min(kIdxRows, rows - r0);
    {
        const auto* src = reinterpret_cast<const uint4*>(hidden + static_cast<std::uint64_t>(r0) * kHidden);
        auto* dst = reinterpret_cast<uint4*>(act);
        const unsigned n = nr * kHidden / 8U, total = kIdxRows * kHidden / 8U;
        for (unsigned i = t; i < total; i += 256U) dst[i] = i < n ? src[i] : make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    for (unsigned o0 = blockIdx.y * kSlice; o0 < (blockIdx.y + 1U) * kSlice; o0 += kIdxOuts) {
        float w[kIdxOuts][10];
#pragma unroll
        for (unsigned oo = 0; oo < kIdxOuts; ++oo) {
            const unsigned o = o0 + oo;
            const std::uint16_t* wrow = o < kIndexQWidth ? wq + static_cast<std::uint64_t>(o) * kHidden
                                                         : wk + static_cast<std::uint64_t>(o - kIndexQWidth) * kHidden;
#pragma unroll
            for (unsigned k = 0; k < 10U; ++k) w[oo][k] = bf16f(__ldg(wrow + t + 256U * k));
        }
#pragma unroll 2
        for (unsigned r = 0; r < kIdxRows; ++r) {
            float sum[kIdxOuts];
#pragma unroll
            for (unsigned oo = 0; oo < kIdxOuts; ++oo) sum[oo] = 0.0F;
#pragma unroll
            for (unsigned k = 0; k < 10U; ++k) {
                const float a = bf16f(act[r * kHidden + t + 256U * k]);
#pragma unroll
                for (unsigned oo = 0; oo < kIdxOuts; ++oo) sum[oo] = __fmaf_rn(a, w[oo][k], sum[oo]);
            }
#pragma unroll
            for (unsigned oo = 0; oo < kIdxOuts; ++oo) red[(r * kIdxOuts + oo) * 256U + t] = sum[oo];
        }
        __syncthreads();
#pragma unroll
        for (unsigned level = 0; level < 3U; ++level) {
            const unsigned half = 128U >> level, shift = 7U - level;
            for (unsigned i = t; i < kIdxPairs * half; i += 256U) {
                const unsigned pr = i >> shift, j = i & (half - 1U);
                red[pr * 256U + j] = __fadd_rn(red[pr * 256U + j], red[pr * 256U + j + half]);
            }
            __syncthreads();
        }
#pragma unroll
        for (unsigned pp = 0; pp < kIdxPairs / 8U; ++pp) {
            const unsigned pr = warp * (kIdxPairs / 8U) + pp, r = pr / kIdxOuts, o = o0 + pr % kIdxOuts;
            float v = red[pr * 256U + lane];
#pragma unroll
            for (unsigned d = 16U; d > 0U; d >>= 1U) v = __fadd_rn(v, __shfl_down_sync(kFull, v, d));
            if (lane == 0U && r < nr) {
                const auto row = static_cast<std::uint64_t>(r0 + r);
                if (o < kIndexQWidth) iq[row * kIndexQWidth + o] = round_bf16(v);
                else ik[row * kIndexDim + (o - kIndexQWidth)] = round_bf16(v);
            }
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------------------------------------------------------------
// norm_rope_body (H:865-996), kernel norm_rope_rows_kernel<f32,64> (S:314600-315315). One block per (head, row), one
// thread per feature:
//   red[f] = x*x (S:314638 v_mul), LDS tree strides hd/2..1 (S:314662 v_add);
//   mean = red/hd + eps (S:314676-314694: correctly rounded v_div sequence, then v_add); inv = rsqrtf(mean) (S:314696-314701
//   OCML: scale by 2^24 below 2^-126, v_rsq, scale 2^12) = amd_rsqrtf;
//   y = round_bf16((x*inv)*w) (S:314704, 314707: two v_mul);
//   RoPE f < 64, pair = f % 32: res = a*cos (S:315272 v_mul) then f >= 32: fma(b, sin, a*cos) (S:315293 v_fmac),
//   f < 32: fma(-b, sin, a*cos) (S:315302 v_fma -v0); b = y[f -/+ 32]; round_bf16.
__device__ __forceinline__ void norm_rope_kernel_body(float* __restrict__ values, std::uint64_t row_stride, std::uint32_t head_dim,
    std::uint32_t head_stride, const float* __restrict__ norm, std::uint64_t first_position, const float2* __restrict__ cs,
    am::Tables tables) {
    extern __shared__ float nsm[];
    float* source = nsm;
    float* reduction = nsm + head_dim;
    const auto head = blockIdx.x, row = blockIdx.y, f = threadIdx.x;
    const auto position = first_position + row;
    float* v = values + row * row_stride + static_cast<std::uint64_t>(head) * head_stride;
    const float input = v[f];
    source[f] = input;
    reduction[f] = __fmul_rn(input, input);
    __syncthreads();
    for (std::uint32_t stride = head_dim / 2U; stride > 0; stride >>= 1U) {
        if (f < stride) reduction[f] = __fadd_rn(reduction[f], reduction[f + stride]);
        __syncthreads();
    }
    const float mean = __fadd_rn(__fdiv_rn(reduction[0], static_cast<float>(head_dim)), kEpsilon);
    const float inv = am::amd_rsqrtf(tables, mean);
    const float y = round_bf16(__fmul_rn(__fmul_rn(input, inv), norm[f]));
    source[f] = y;
    __syncthreads();
    float result = y;
    if (f < kRotaryDim) {
        const auto pair = f % kRotaryPairs;
        const float2 c = cs[position * kRotaryPairs + pair];
        const float ac = __fmul_rn(y, c.x);
        result = f < kRotaryPairs ? __fmaf_rn(-source[f + kRotaryPairs], c.y, ac) : __fmaf_rn(source[f - kRotaryPairs], c.y, ac);
    }
    v[f] = round_bf16(result);
}
__global__ void norm_rope_kernel(float* __restrict__ values, std::uint64_t row_stride, std::uint32_t head_dim,
    std::uint32_t head_stride, const float* __restrict__ norm, std::uint64_t first_position, const float2* __restrict__ cs,
    am::Tables tables) {
    norm_rope_kernel_body(values, row_stride, head_dim, head_stride, norm, first_position, cs, tables);
}

// ------------------------------------------------------------------------------------------------------------------
// store_rows_kernel<Rows> (H:3647-3671): raw index key, K, V -> BF16 (exact for BF16-valued inputs).
__device__ __forceinline__ void store_kernel_body(const float* __restrict__ ik, const float* __restrict__ k, const float* __restrict__ v,
    std::uint64_t first_position, std::uint16_t* __restrict__ raw, std::uint16_t* __restrict__ kv) {
    const auto r = blockIdx.y;
    const auto f = blockIdx.x * blockDim.x + threadIdx.x;
    const auto position = first_position + r;
    if (f < kIndexDim) raw[position * kIndexDim + f] = to_bf16(ik[static_cast<std::uint64_t>(r) * kIndexDim + f]);
    if (f < kKvWidth) {
        const auto base = position * kKvWidth * 2ULL;
        kv[base + f] = to_bf16(k[static_cast<std::uint64_t>(r) * kKvWidth + f]);
        kv[base + kKvWidth + f] = to_bf16(v[static_cast<std::uint64_t>(r) * kKvWidth + f]);
    }
}
__global__ void store_kernel(const float* __restrict__ ik, const float* __restrict__ k, const float* __restrict__ v,
    std::uint64_t first_position, std::uint16_t* __restrict__ raw, std::uint16_t* __restrict__ kv) {
    store_kernel_body(ik, k, v, first_position, raw, kv);
}

// ------------------------------------------------------------------------------------------------------------------
// block_pool_kernel<f32> (H:4172-4208 via the listing at H:3973-4010; S:317565-318324). One block per 4-token block,
// 128 threads:
//   mean = 0 + k0 + k1 + k2 + k3 (S:317611 v_add chain), mean /= 4 (correctly rounded div S:317616-317631), round_bf16;
//   sq = mean*mean (S:317641), LDS tree 64..1; inv = rsqrtf(sq/128 + eps) (S:317677-317705);
//   round_bf16((mean*inv)*w) (S:317706, 317709); RoPE at position 4*block: fma(+-b, sin, a*cos) (S:318275, 318296,
//   318305) exactly as norm_rope; round_bf16.
// block = the pooled block this CTA computes (the launchers pass first_block + blockIdx.x, or a grid-stride index)
__device__ __forceinline__ void block_pool_kernel_body(const std::uint16_t* __restrict__ raw, std::uint32_t block,
    const float* __restrict__ key_norm, const float2* __restrict__ cs, am::Tables tables, float* __restrict__ pooled_keys) {
    __shared__ float pooled[kIndexDim], scratch[kIndexDim];
    const auto f = threadIdx.x;
    const auto token_start = static_cast<std::uint64_t>(block) * kRatio;
    float mean = 0.0F;
    for (std::uint32_t o = 0; o < kRatio; ++o) mean = __fadd_rn(mean, bf16f(raw[(token_start + o) * kIndexDim + f]));
    mean = __fdiv_rn(mean, static_cast<float>(kRatio));
    mean = round_bf16(mean);
    pooled[f] = mean;
    scratch[f] = __fmul_rn(mean, mean);
    __syncthreads();
    for (std::uint32_t stride = kIndexDim / 2U; stride > 0; stride >>= 1U) {
        if (f < stride) scratch[f] = __fadd_rn(scratch[f], scratch[f + stride]);
        __syncthreads();
    }
    const float inv = am::amd_rsqrtf(tables, __fadd_rn(__fdiv_rn(scratch[0], static_cast<float>(kIndexDim)), kEpsilon));
    const float y = round_bf16(__fmul_rn(__fmul_rn(mean, inv), key_norm[f]));
    pooled[f] = y;
    __syncthreads();
    float rotated = y;
    if (f < kRotaryDim) {
        const auto pair = f % kRotaryPairs;
        const float2 c = cs[token_start * kRotaryPairs + pair];
        const float ac = __fmul_rn(y, c.x);
        rotated = f < kRotaryPairs ? __fmaf_rn(-pooled[f + kRotaryPairs], c.y, ac) : __fmaf_rn(pooled[f - kRotaryPairs], c.y, ac);
    }
    pooled_keys[static_cast<std::uint64_t>(block) * kIndexDim + f] = round_bf16(rotated);
}
__global__ void block_pool_kernel(const std::uint16_t* __restrict__ raw, std::uint32_t first_block,
    const float* __restrict__ key_norm, const float2* __restrict__ cs, am::Tables tables, float* __restrict__ pooled_keys) {
    block_pool_kernel_body(raw, first_block + blockIdx.x, key_norm, cs, tables, pooled_keys);
}

// ------------------------------------------------------------------------------------------------------------------
// Lane-xor reduce-scatter qsa_scatter<P> (H:3882-3913): step D = 16 (permlanex16 identity = lane ^ 16), 8, 4, 2, 1
// (DPP row_xmask = lane ^ D) halves the carried values, the lower lane keeping the lower half; each step adds the
// partials of lanes L and L ^ D (one addition: operand order irrelevant). Per value this is the 32-lane tree
// x_L + x_{L^16}, then ^8, ^4, ^2, ^1, i.e. lane 0's shfl_down 16..1 tree.
template <unsigned P, unsigned Step = 0U> __device__ __forceinline__ void scatter(float (&v)[P], unsigned lane) {
    if constexpr (Step < 5U) {
        constexpr unsigned D = 16U >> Step, Count = (P >> Step) > 0U ? (P >> Step) : 1U;
        if constexpr (Count > 1U) {
            constexpr unsigned H = Count / 2U;
            const bool upper = (lane & D) != 0U;
#pragma unroll
            for (unsigned j = 0; j < H; ++j) {
                const float low = __shfl_xor_sync(kFull, v[j], D), high = __shfl_xor_sync(kFull, v[j + H], D);
                v[j] = upper ? __fadd_rn(v[j + H], high) : __fadd_rn(v[j], low);
            }
        } else {
            v[0] = __fadd_rn(v[0], __shfl_xor_sync(kFull, v[0], D));
        }
        scatter<P, Step + 1U>(v, lane);
    }
}
template <unsigned P> __device__ __forceinline__ unsigned scatter_head(unsigned lane) {
    unsigned head = 0, weight = P / 2U;
#pragma unroll
    for (unsigned D = 16U; D > 0U && weight > 0U; D >>= 1U, weight >>= 1U) if (lane & D) head += weight;
    return head;
}

// ------------------------------------------------------------------------------------------------------------------
constexpr unsigned kScoreRowsPerWave = 4;
inline constexpr strix::lever::Decl score_tree_lever{
    .name = "qwen_qsa_score_tree", .env = "STRIX_QRTX_QSA_SCORE_TREE", .exactness = strix::lever::Exactness::bitwise,
    .default_on = true, .source = "tests/cuda/qwen_qsa_score_tree_test.cu: byte equality vs frozen scorers"};
bool score_tree_enabled() {
    static const bool on = strix::lever::select(score_tree_lever, {}).on;
    return on;
}
// The tree body reads the query rows and pooled keys as float4; the shuffle scorer reads floats. A base pointer that is
// not 16-B aligned gets the shuffle scorer (same bits) instead of a misaligned vector load.
bool tree_aligned(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0U; }
constexpr unsigned kTreeRows = 64, kTreeSeg = 256;
static_assert(kMaxSegments == ScoreMap::capacity);
constexpr selection::ScoreSpec kBlockScoreSpec{kTopK, kIndexHeads, kIndexDim, kRatio};
constexpr selection::PickSpec kBlockPickSpec{kTopK, kRatio, selection::KeyOrder::eligible_float};

template <class Map>
void enqueue_block_scores(selection::SegmentRows<Map> input, selection::SegmentScoreShape shape, bool tree,
    cudaStream_t stream, unsigned shuffle_blocks = 0) {
    constexpr auto relu = selection::ScoreForm::relu_tree;
    if (tree) {
        check(selection::enqueue_scores<relu, selection::ScoreSchedule::row_tree>(input, shape, kBlockScoreSpec, stream),
            "block score tree launch");
    } else {
        check(selection::enqueue_scores<relu, selection::ScoreSchedule::warp_shuffle>(input, shape, kBlockScoreSpec, stream,
            shuffle_blocks), "block score launch");
    }
}

// ------------------------------------------------------------------------------------------------------------------
// attention_score_rows_group256_kernel<64,12,16> (H:3981-4050, contract off; S:383040). One warp per (row, 12-head
// group = KV head, 16-slot run). Lane l holds q features l+32j (j=0..7) of the 12 heads; per slot k from the KV row;
// p_j = q*k; p[j] += p[j+4] (j<4); p0 += p2; p1 += p3; partial = p0 + p1; scatter<16> (heads 12..15 zero); lane with
// head h (lane & 1 == 0) writes partial / sqrtf(256) (= /16, exact). = the 256-feature tree strides 128..1.
constexpr unsigned kG = 12, kSlots = 16;
__global__ __launch_bounds__(256) void attention_score_kernel(const float* __restrict__ q_gate, std::uint32_t rows,
    std::uint64_t first_position, const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv,
    float* __restrict__ all_scores, std::uint32_t slots) {
    const unsigned lane = threadIdx.x & 31U;
    const auto wave = blockIdx.x * 8U + threadIdx.x / 32U;
    const auto groups = (slots + kSlots - 1U) / kSlots;
    constexpr unsigned head_groups = kHeads / kG;
    const auto r = wave / (head_groups * groups);
    if (r >= rows) return;
    const auto count = selected_count(first_position + r);
    const auto head0 = wave / groups % head_groups * kG, first = wave % groups * kSlots;
    if (first >= count) return;
    const auto kv_head = head0 / (kHeads / kKvHeads);
    float qv[kG][8];
#pragma unroll
    for (unsigned h = 0; h < kG; ++h) {
        const float* q = q_gate + static_cast<std::uint64_t>(r) * kQGateWidth + static_cast<std::uint64_t>(head0 + h) * kHeadDim * 2ULL;
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) qv[h][j] = q[lane + 32U * j];
    }
    const float scale = __fsqrt_rn(static_cast<float>(kHeadDim));
    const auto last = min(first + kSlots, count);
    const std::uint32_t* selected = all_selected + static_cast<std::uint64_t>(r) * kMaxSelected;
    const auto tokens = lane < last - first ? selected[first + lane] : 0U;
    float* const scores = all_scores + static_cast<std::uint64_t>(r) * kHeads * kMaxSelected;
    const std::uint16_t* kbase = kv + static_cast<std::uint64_t>(kv_head) * kHeadDim + lane;
    std::uint16_t raw[8];
    const auto load = [&](unsigned index) {
        const auto token = __shfl_sync(kFull, tokens, static_cast<int>(index));
        const auto* row = kbase + static_cast<std::uint64_t>(token) * (kKvWidth * 2ULL);
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) raw[j] = row[32U * j];
    };
    load(0U);
    constexpr unsigned P = 16;
    for (auto slot = first; slot < last; ++slot) {
        float kf[8];
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) kf[j] = bf16f(raw[j]);
        if (slot + 1U < last) load(slot + 1U - first);
        float partial[P];
#pragma unroll
        for (unsigned h = 0; h < P; ++h) {
            if (h >= kG) { partial[h] = 0.0F; continue; }
            float p[8];
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) p[j] = __fmul_rn(qv[h][j], kf[j]);
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) p[j] = __fadd_rn(p[j], p[j + 4U]);
            p[0] = __fadd_rn(p[0], p[2]); p[1] = __fadd_rn(p[1], p[3]);
            partial[h] = __fadd_rn(p[0], p[1]);
        }
        scatter<P>(partial, lane);
        const auto h = scatter_head<P>(lane);
        if (h < kG && (lane & 1U) == 0U)
            scores[static_cast<std::uint64_t>(head0 + h) * kMaxSelected + slot] = __fdiv_rn(partial[0], scale);
    }
}

// Same scores, half-warp form (default): the HIP kernel's shuffles bound it on sm_120 (60 SHFL per slot for 12 heads).
// A half-warp of 16 lanes serves one slot, lane l holding features l + 16j (j = 0..15) of 6 heads; the in-lane pairs
// (j, j+8), (j, j+4), (j, j+2), (0, 1) are strides 128, 64, 32, 16 and the xor reduce-scatter over lanes 8, 4, 2, 1
// (scatter over 16 lanes, 8 values) the strides 8..1: the identical 256-feature tree (strides 128..1), now with 15 SHFL
// per (2 slots x 6 heads). Block = 4 warps = (2 head halves) x (2 slot pairs) of one (row, KV head, 32-slot run); both
// head halves read the same K rows (L1).
constexpr unsigned kHalfHeads = 6, kRun = 32;
template <unsigned P, unsigned Step = 1U> __device__ __forceinline__ void scatter16(float (&v)[P], unsigned lane) {
    // steps D = 8, 4, 2, 1 of scatter (Step 1..4); P <= 8 values over the 16 lanes of a half-warp
    if constexpr (Step < 5U) {
        constexpr unsigned D = 16U >> Step, Count = (P >> (Step - 1U)) > 0U ? (P >> (Step - 1U)) : 1U;
        if constexpr (Count > 1U) {
            constexpr unsigned H = Count / 2U;
            const bool upper = (lane & D) != 0U;
#pragma unroll
            for (unsigned j = 0; j < H; ++j) {
                const float low = __shfl_xor_sync(kFull, v[j], D), high = __shfl_xor_sync(kFull, v[j + H], D);
                v[j] = upper ? __fadd_rn(v[j + H], high) : __fadd_rn(v[j], low);
            }
        } else {
            v[0] = __fadd_rn(v[0], __shfl_xor_sync(kFull, v[0], D));
        }
        scatter16<P, Step + 1U>(v, lane);
    }
}
// One slot of the half-warp form, shared by score_half and the fused kernel: the 6 heads' products and the 256-feature
// tree (strides 128..16 in-lane, 8..1 by scatter16). Every lane of the warp must call it (shuffles). Lane l returns the
// sum of head score6_head(l) of its half-warp's slot (meaningful on even l).
__device__ __forceinline__ float score6_slot(const float (&qv)[kHalfHeads][16], const float (&kf)[16], unsigned lane) {
    float partial[8];
#pragma unroll
    for (unsigned h = 0; h < 8U; ++h) {
        if (h >= kHalfHeads) { partial[h] = 0.0F; continue; }
        float p[16];
#pragma unroll
        for (unsigned j = 0; j < 16U; ++j) p[j] = __fmul_rn(qv[h][j], kf[j]);
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) p[j] = __fadd_rn(p[j], p[j + 8U]);
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) p[j] = __fadd_rn(p[j], p[j + 4U]);
        p[0] = __fadd_rn(p[0], p[2]); p[1] = __fadd_rn(p[1], p[3]);
        partial[h] = __fadd_rn(p[0], p[1]);
    }
    scatter16<8>(partial, lane);
    return partial[0];
}
__device__ __forceinline__ unsigned score6_head(unsigned l) {
    return ((l & 8U) ? 4U : 0U) + ((l & 4U) ? 2U : 0U) + ((l & 2U) ? 1U : 0U);
}
__device__ __forceinline__ void attention_score_half_kernel_body(const float* __restrict__ q_gate, std::uint64_t first_position,
    const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv, float* __restrict__ all_scores) {
    const unsigned lane = threadIdx.x & 31U, warp = threadIdx.x >> 5U, l = lane & 15U, half = lane >> 4U;
    const auto run = blockIdx.x, kvh = blockIdx.y, r = blockIdx.z;
    const auto count = selected_count(first_position + r);
    const auto first = run * kRun;
    if (first >= count) return;
    const unsigned hh = warp & 1U, sp = warp >> 1U;
    const auto head0 = kvh * kG + hh * kHalfHeads;
    float qv[kHalfHeads][16];
#pragma unroll
    for (unsigned h = 0; h < kHalfHeads; ++h) {
        const float* qp = q_gate + static_cast<std::uint64_t>(r) * kQGateWidth + static_cast<std::uint64_t>(head0 + h) * kHeadDim * 2ULL + l;
#pragma unroll
        for (unsigned j = 0; j < 16U; ++j) qv[h][j] = qp[16U * j];
    }
    const float scale = __fsqrt_rn(static_cast<float>(kHeadDim));
    const std::uint32_t* selected = all_selected + static_cast<std::uint64_t>(r) * kMaxSelected;
    float* const scores = all_scores + static_cast<std::uint64_t>(r) * kHeads * kMaxSelected;
    const std::uint16_t* kbase = kv + static_cast<std::uint64_t>(kvh) * kHeadDim + l;
    // slot of iteration i for this half-warp: first + 4i + 2sp + half (i = 0..7)
    const auto slot_of = [&](unsigned i) { return first + 4U * i + 2U * sp + half; };
    std::uint16_t raw[16];
    const auto load = [&](unsigned i) {
        const auto slot = slot_of(i);
        const auto token = slot < count ? selected[slot] : 0U;
        const auto* row = kbase + static_cast<std::uint64_t>(token) * (kKvWidth * 2ULL);
#pragma unroll
        for (unsigned j = 0; j < 16U; ++j) raw[j] = row[16U * j];
    };
    load(0U);
#pragma unroll 1
    for (unsigned i = 0; i < kRun / 4U; ++i) {
        if (first + 4U * i >= count) break;  // uniform per block
        float kf[16];
#pragma unroll
        for (unsigned j = 0; j < 16U; ++j) kf[j] = bf16f(raw[j]);
        if (i + 1U < kRun / 4U) load(i + 1U);
        const float partial = score6_slot(qv, kf, lane);
        const unsigned h = score6_head(l);
        const auto slot = slot_of(i);
        if (h < kHalfHeads && (l & 1U) == 0U && slot < count)
            scores[static_cast<std::uint64_t>(head0 + h) * kMaxSelected + slot] = __fdiv_rn(partial, scale);
    }
}
__global__ __launch_bounds__(128) void attention_score_half_kernel(const float* __restrict__ q_gate, std::uint64_t first_position,
    const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv, float* __restrict__ all_scores) {
    attention_score_half_kernel_body(q_gate, first_position, all_selected, kv, all_scores);
}

// attention_softmax_rows_kernel<64> -> attention_softmax_body (H:1680-1786; S:408329-408725). Grid (24 heads, rows),
// 256 threads. max: fmaxf over slots l, l+256, ... then LDS fmaxf tree (order-free); local_sum += expf(s - max)
// ascending (S:408533-408553: v_sub, OCML expf, v_add); LDS tree 128..1 (S:408571-408655);
// weight = round_bf16(expf(s - max) / den) (S:408678-408712: correctly rounded division).
__device__ __forceinline__ void softmax_kernel_body(float* __restrict__ all_scores, std::uint64_t first_position, am::Tables tables) {
    __shared__ float scratch[256];
    const auto head = blockIdx.x, r = blockIdx.y, lane = threadIdx.x;
    const auto count = selected_count(first_position + r);
    float* scores = all_scores + (static_cast<std::uint64_t>(r) * kHeads + head) * kMaxSelected;
    float local_max = (-kInf);
    for (std::uint32_t i = lane; i < count; i += 256U) local_max = fmaxf(local_max, scores[i]);
    scratch[lane] = local_max;
    __syncthreads();
    for (std::uint32_t stride = 128U; stride > 0; stride >>= 1U) {
        if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    const float maximum = scratch[0];
    __syncthreads();
    float local_sum = 0.0F;
    // HIP recomputes expf(s - max) in the weight pass; the value is the same, so it is kept in place of the score.
    for (std::uint32_t i = lane; i < count; i += 256U) {
        const float e = am::amd_expf(tables, __fsub_rn(scores[i], maximum));
        scores[i] = e;
        local_sum = __fadd_rn(local_sum, e);
    }
    scratch[lane] = local_sum;
    __syncthreads();
    for (std::uint32_t stride = 128U; stride > 0; stride >>= 1U) {
        if (lane < stride) scratch[lane] = __fadd_rn(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    const float denominator = scratch[0];
    for (std::uint32_t i = lane; i < count; i += 256U)
        scores[i] = round_bf16(__fdiv_rn(scores[i], denominator));
}
__global__ __launch_bounds__(256) void softmax_kernel(float* __restrict__ all_scores, std::uint64_t first_position, am::Tables tables) {
    softmax_kernel_body(all_scores, first_position, tables);
}

// attention_output_rows_group_prefetch_kernel<64,12> (H:4102-4163; S:414951). Per (row, head, feature):
// sum = fma(w[slot], v[slot], sum) over all selected slots ascending from 0 (S: v_fmac/v_fma chain, `__builtin_fmaf`);
// attn = round_bf16(sum); gate = round_bf16(1 / (1 + expf(-g))) (S:415646-415678: OCML expf with the negation folded
// into the constants = expf(-g) exactly, v_add 1.0, correctly rounded division); out = round_bf16(attn * gate).
// CUDA: block per (row, KV head), 128 threads x 2 features x 12 heads (24 chains per thread); the weights of a chunk of
// slots are staged in shared memory as [slot][12] (3 x 16-byte broadcast loads per slot), tokens too.
constexpr unsigned kOutThreads = 128, kOutChunk = 256;
// The gated output value of one (head, feature) from its finished chain `sum` and the gate input g (shared epilogue).
__device__ __forceinline__ float qsa_gated(float sum, float g, const am::Tables& tables) {
    const float attn = round_bf16(sum);
    const float gate = round_bf16(__fdiv_rn(1.0F, __fadd_rn(1.0F, am::amd_expf(tables, -g))));
    return round_bf16(__fmul_rn(attn, gate));
}
__device__ __forceinline__ void attention_output_kernel_body(const float* __restrict__ q_gate,
    std::uint64_t first_position, const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv,
    const float* __restrict__ all_scores, float* __restrict__ output, am::Tables tables) {
    __shared__ __align__(16) float ws[kOutChunk * kG];
    __shared__ std::uint32_t tok[kOutChunk];
    const auto kvh = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    const auto head0 = kvh * kG;
    const auto count = selected_count(first_position + r);
    const std::uint32_t* selected = all_selected + static_cast<std::uint64_t>(r) * kMaxSelected;
    const float* scores = all_scores + (static_cast<std::uint64_t>(r) * kHeads + head0) * kMaxSelected;
    const std::uint16_t* vbase = kv + kKvWidth + kvh * kHeadDim + 2U * t;
    float s0[kG], s1[kG];
#pragma unroll
    for (unsigned h = 0; h < kG; ++h) { s0[h] = 0.0F; s1[h] = 0.0F; }
    for (std::uint32_t c0 = 0; c0 < count; c0 += kOutChunk) {
        const auto n = min(kOutChunk, count - c0);
        __syncthreads();
        for (unsigned i = t; i < n * kG; i += kOutThreads) {
            const unsigned slot = i % n, h = i / n;
            ws[slot * kG + h] = scores[static_cast<std::uint64_t>(h) * kMaxSelected + c0 + slot];
        }
        for (unsigned i = t; i < n; i += kOutThreads) tok[i] = selected[c0 + i];
        __syncthreads();
        constexpr unsigned B = 8;
        unsigned i = 0;
        for (; i + B <= n; i += B) {
            std::uint32_t raw[B];
#pragma unroll
            for (unsigned b = 0; b < B; ++b)
                raw[b] = *reinterpret_cast<const std::uint32_t*>(vbase + static_cast<std::uint64_t>(tok[i + b]) * (kKvWidth * 2ULL));
#pragma unroll
            for (unsigned b = 0; b < B; ++b) {
                const float v0 = __uint_as_float(raw[b] << 16U), v1 = __uint_as_float(raw[b] & 0xffff0000U);
                const float4* wp = reinterpret_cast<const float4*>(ws + (i + b) * kG);
                const float4 a = wp[0], bq = wp[1], cq = wp[2];
                const float w[kG] = {a.x, a.y, a.z, a.w, bq.x, bq.y, bq.z, bq.w, cq.x, cq.y, cq.z, cq.w};
#pragma unroll
                for (unsigned h = 0; h < kG; ++h) { s0[h] = __fmaf_rn(w[h], v0, s0[h]); s1[h] = __fmaf_rn(w[h], v1, s1[h]); }
            }
        }
        for (; i < n; ++i) {
            const auto raw = *reinterpret_cast<const std::uint32_t*>(vbase + static_cast<std::uint64_t>(tok[i]) * (kKvWidth * 2ULL));
            const float v0 = __uint_as_float(raw << 16U), v1 = __uint_as_float(raw & 0xffff0000U);
#pragma unroll
            for (unsigned h = 0; h < kG; ++h) {
                const float w = ws[i * kG + h];
                s0[h] = __fmaf_rn(w, v0, s0[h]); s1[h] = __fmaf_rn(w, v1, s1[h]);
            }
        }
    }
    const float* qg = q_gate + static_cast<std::uint64_t>(r) * kQGateWidth;
    float* out = output + static_cast<std::uint64_t>(r) * kAttnWidth;
#pragma unroll
    for (unsigned h = 0; h < kG; ++h) {
        const auto head = head0 + h;
#pragma unroll
        for (unsigned e = 0; e < 2U; ++e) {
            const auto f = 2U * t + e;
            const float g = qg[static_cast<std::uint64_t>(head) * kHeadDim * 2ULL + kHeadDim + f];
            out[static_cast<std::uint64_t>(head) * kHeadDim + f] = qsa_gated(e ? s1[h] : s0[h], g, tables);
        }
    }
}
__global__ __launch_bounds__(kOutThreads) void attention_output_kernel(const float* __restrict__ q_gate,
    std::uint64_t first_position, const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv,
    const float* __restrict__ all_scores, float* __restrict__ output, am::Tables tables) {
    attention_output_kernel_body(q_gate, first_position, all_selected, kv, all_scores, output, tables);
}

// Decode shape (few rows): one block per (query head, row), so a row runs 24 blocks instead of 2. Each thread keeps the
// same two chains sum = fma(w[slot], v[slot], sum), slots ascending from 0, and the same epilogue.
// STRIX_QRTX_DECODE_KERNELS=0 keeps the per-KV-head kernel (reference seam).
bool decode_rows_enabled() {
    static const bool on = strix::envcfg::flag("STRIX_QRTX_DECODE_KERNELS", true);
    return on;
}
__device__ __forceinline__ void attention_output_head_kernel_body(const float* __restrict__ q_gate,
    std::uint64_t first_position, const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv,
    const float* __restrict__ all_scores, float* __restrict__ output, am::Tables tables) {
    __shared__ float ws[kOutChunk];
    __shared__ std::uint32_t tok[kOutChunk];
    const auto head = blockIdx.x, r = blockIdx.y, t = threadIdx.x, kvh = head / kG;
    const auto count = selected_count(first_position + r);
    const std::uint32_t* selected = all_selected + static_cast<std::uint64_t>(r) * kMaxSelected;
    const float* scores = all_scores + (static_cast<std::uint64_t>(r) * kHeads + head) * kMaxSelected;
    const std::uint16_t* vbase = kv + kKvWidth + kvh * kHeadDim + 2U * t;
    float s0 = 0.0F, s1 = 0.0F;
    for (std::uint32_t c0 = 0; c0 < count; c0 += kOutChunk) {
        const auto n = min(kOutChunk, count - c0);
        __syncthreads();
        for (unsigned i = t; i < n; i += kOutThreads) { ws[i] = scores[c0 + i]; tok[i] = selected[c0 + i]; }
        __syncthreads();
        constexpr unsigned B = 8;
        unsigned i = 0;
        for (; i + B <= n; i += B) {
            std::uint32_t raw[B];
#pragma unroll
            for (unsigned b = 0; b < B; ++b)
                raw[b] = *reinterpret_cast<const std::uint32_t*>(vbase + static_cast<std::uint64_t>(tok[i + b]) * (kKvWidth * 2ULL));
#pragma unroll
            for (unsigned b = 0; b < B; ++b) {
                const float w = ws[i + b];
                s0 = __fmaf_rn(w, __uint_as_float(raw[b] << 16U), s0);
                s1 = __fmaf_rn(w, __uint_as_float(raw[b] & 0xffff0000U), s1);
            }
        }
        for (; i < n; ++i) {
            const auto raw = *reinterpret_cast<const std::uint32_t*>(vbase + static_cast<std::uint64_t>(tok[i]) * (kKvWidth * 2ULL));
            s0 = __fmaf_rn(ws[i], __uint_as_float(raw << 16U), s0);
            s1 = __fmaf_rn(ws[i], __uint_as_float(raw & 0xffff0000U), s1);
        }
    }
    const float* qg = q_gate + static_cast<std::uint64_t>(r) * kQGateWidth;
    float* out = output + static_cast<std::uint64_t>(r) * kAttnWidth;
#pragma unroll
    for (unsigned e = 0; e < 2U; ++e) {
        const auto f = 2U * t + e;
        const float attn = round_bf16(e ? s1 : s0);
        const float g = qg[static_cast<std::uint64_t>(head) * kHeadDim * 2ULL + kHeadDim + f];
        const float gate = round_bf16(__fdiv_rn(1.0F, __fadd_rn(1.0F, am::amd_expf(tables, -g))));
        out[static_cast<std::uint64_t>(head) * kHeadDim + f] = round_bf16(__fmul_rn(attn, gate));
    }
}
__global__ __launch_bounds__(kOutThreads) void attention_output_head_kernel(const float* __restrict__ q_gate,
    std::uint64_t first_position, const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv,
    const float* __restrict__ all_scores, float* __restrict__ output, am::Tables tables) {
    attention_output_head_kernel_body(q_gate, first_position, all_selected, kv, all_scores, output, tables);
}

// STRIX_QRTX_QSA_HEAD (default 1): same per-thread chains (s0/s1 = fma(w[slot], v, s), slots ascending from 0) and epilogue as the
// body above, but the V rows (512 B per slot for this KV head) are streamed by cp.async into a ring of 7 stages x 8 slots
// (6 stages in flight, one __syncthreads per stage) and the scores/tokens of every slot are preloaded once, so the walk over the
// slots no longer pays one L2 round trip per 8 slots. Static smem 45 KB.
constexpr unsigned kPipeKeys = 8, kPipeStages = 7, kVRowBytes = kHeadDim * 2U;
__device__ __forceinline__ void qsa_cp16(void* dst, const void* src) {
    const auto d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src) : "memory");
}
__device__ __forceinline__ void qsa_cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void qsa_cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void attention_output_head_pipe_body(const float* __restrict__ q_gate,
    std::uint64_t first_position, const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv,
    const float* __restrict__ all_scores, float* __restrict__ output, am::Tables tables) {
    __shared__ __align__(16) unsigned char ring[kPipeStages * kPipeKeys * kVRowBytes];
    __shared__ float ws[kMaxSelected];
    __shared__ std::uint32_t tok[kMaxSelected];
    const auto head = blockIdx.x, r = blockIdx.y, t = threadIdx.x, kvh = head / kG;
    const auto count = selected_count(first_position + r);
    const std::uint32_t* selected = all_selected + static_cast<std::uint64_t>(r) * kMaxSelected;
    const float* scores = all_scores + (static_cast<std::uint64_t>(r) * kHeads + head) * kMaxSelected;
    const std::uint16_t* vsrc = kv + kKvWidth + kvh * kHeadDim;
    for (unsigned i = t; i < count; i += kOutThreads) { ws[i] = scores[i]; tok[i] = selected[i]; }
    __syncthreads();
    const unsigned nst = (count + kPipeKeys - 1U) / kPipeKeys;
    auto issue = [&](unsigned st) {
        if (st < nst) {
            unsigned char* slot = ring + (st % kPipeStages) * kPipeKeys * kVRowBytes;
            for (unsigned j = t; j < kPipeKeys * 32U; j += kOutThreads) {
                const unsigned b = j >> 5U, key = st * kPipeKeys + b;
                if (key < count) qsa_cp16(slot + b * kVRowBytes + (j & 31U) * 16U, vsrc + static_cast<std::uint64_t>(tok[key]) * (kKvWidth * 2ULL) + (j & 31U) * 8U);
            }
        }
        qsa_cp_commit();
    };
#pragma unroll
    for (unsigned s = 0; s + 1U < kPipeStages; ++s) issue(s);
    float s0 = 0.0F, s1 = 0.0F;
    for (unsigned st = 0; st < nst; ++st) {
        qsa_cp_wait<static_cast<int>(kPipeStages) - 2>();
        __syncthreads();
        issue(st + kPipeStages - 1U);
        const unsigned char* slot = ring + (st % kPipeStages) * kPipeKeys * kVRowBytes + 4U * t;
        const unsigned n = min(kPipeKeys, count - st * kPipeKeys);
        if (n == kPipeKeys) {
#pragma unroll
            for (unsigned b = 0; b < kPipeKeys; ++b) {
                const auto raw = *reinterpret_cast<const std::uint32_t*>(slot + b * kVRowBytes);
                const float w = ws[st * kPipeKeys + b];
                s0 = __fmaf_rn(w, __uint_as_float(raw << 16U), s0);
                s1 = __fmaf_rn(w, __uint_as_float(raw & 0xffff0000U), s1);
            }
        } else {
            for (unsigned b = 0; b < n; ++b) {
                const auto raw = *reinterpret_cast<const std::uint32_t*>(slot + b * kVRowBytes);
                const float w = ws[st * kPipeKeys + b];
                s0 = __fmaf_rn(w, __uint_as_float(raw << 16U), s0);
                s1 = __fmaf_rn(w, __uint_as_float(raw & 0xffff0000U), s1);
            }
        }
    }
    const float* qg = q_gate + static_cast<std::uint64_t>(r) * kQGateWidth;
    float* out = output + static_cast<std::uint64_t>(r) * kAttnWidth;
#pragma unroll
    for (unsigned e = 0; e < 2U; ++e) {
        const auto f = 2U * t + e;
        const float attn = round_bf16(e ? s1 : s0);
        const float g = qg[static_cast<std::uint64_t>(head) * kHeadDim * 2ULL + kHeadDim + f];
        const float gate = round_bf16(__fdiv_rn(1.0F, __fadd_rn(1.0F, am::amd_expf(tables, -g))));
        out[static_cast<std::uint64_t>(head) * kHeadDim + f] = round_bf16(__fmul_rn(attn, gate));
    }
}
__global__ __launch_bounds__(kOutThreads) void attention_output_head_pipe_kernel(const float* __restrict__ q_gate,
    std::uint64_t first_position, const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv,
    const float* __restrict__ all_scores, float* __restrict__ output, am::Tables tables) {
    attention_output_head_pipe_body(q_gate, first_position, all_selected, kv, all_scores, output, tables);
}
bool head_pipe_enabled() {
    static const bool on = strix::envcfg::flag("STRIX_QRTX_QSA_HEAD", true);
    return on;
}
bool idx_wide_enabled() {
    static const bool on = strix::envcfg::flag("STRIX_QRTX_QSA_IDX", true);
    return on;
}
// ------------------------------------------------------------------------------------------------------------------
// Exact fused form (STRIX_QRTX_QSA_FUSED): score_half + softmax + output of one (KV head, row) in one block, with the same
// bits. The scores of a (row, KV head) are count x 12 FP32 <= 98,448 B and stay in dynamic shared memory ([slot][12]), so
// the three-launch form's [rows][24][2051] round trip through DRAM (a 1,024-row chunk is 201 MB > L2) and its q reload
// per 32-slot run disappear. 12 warps:
//  1. scores: warp w = (6-head half w & 1, slot pair w >> 1); half-warp slot = 12 i + 2 (w >> 1) + half; q stays in
//     registers for the whole row; per slot the arithmetic is score6_slot (shared with score_half), then / sqrt(256);
//  2. softmax: warp w owns head w: max by fmaxf (order-free); e = amd_expf(s - max) in place; the denominator in the
//     reference lane order (qsa_fused_order.hpp); weight = round_bf16(__fdiv_rn(e, den)) in place;
//  3. output: thread = (4-head group t / 128, features 2 (t % 128), +1): chains fmaf(w, v, s) over slots ascending from 0
//     (each (head, feature) chain equals attention_output_kernel's), V rows from a two-buffer register prefetch of 8
//     slots, then the shared gate epilogue.
constexpr unsigned kFusedWarps = 12, kFusedThreads = kFusedWarps * 32U, kFusedGroup = 4, kFusedStep = kFusedWarps, kFusedVB = 8;
static_assert(kG == kFusedWarps && kG == 2U * kHalfHeads, "fused softmax: warp w owns head w of the KV head");
static_assert(kFusedThreads == (kG / kFusedGroup) * (kHeadDim / 2U), "fused output: one (4-head group, feature pair) per thread");
constexpr std::size_t fused_smem_bytes(std::uint32_t slots) { return std::size_t{slots} * kG * sizeof(float); }
static_assert(fused_smem_bytes(kMaxSelected) <= 99U * 1024U, "qsa fused: one (row, KV head) of scores in shared memory");
__device__ __forceinline__ void attention_fused_body(const float* __restrict__ q_gate, std::uint64_t first_position,
    const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv, float* __restrict__ output,
    const am::Tables& tables) {
    extern __shared__ __align__(16) float sc[];   // [slot][12]: score, then e, then weight
    const unsigned lane = threadIdx.x & 31U, warp = threadIdx.x >> 5U;
    const auto kvh = blockIdx.x, r = blockIdx.y;
    const auto count = selected_count(first_position + r);
    const std::uint32_t* selected = all_selected + static_cast<std::uint64_t>(r) * kMaxSelected;
    const float* qg = q_gate + static_cast<std::uint64_t>(r) * kQGateWidth;
    {   // 1. scores
        const unsigned l = lane & 15U, half = lane >> 4U, hh = warp & 1U, sp = warp >> 1U;
        const auto head0 = kvh * kG + hh * kHalfHeads;
        float qv[kHalfHeads][16];
#pragma unroll
        for (unsigned h = 0; h < kHalfHeads; ++h) {
            const float* qp = qg + static_cast<std::uint64_t>(head0 + h) * kHeadDim * 2ULL + l;
#pragma unroll
            for (unsigned j = 0; j < 16U; ++j) qv[h][j] = qp[16U * j];
        }
        const float scale = __fsqrt_rn(static_cast<float>(kHeadDim));
        const std::uint16_t* kbase = kv + static_cast<std::uint64_t>(kvh) * kHeadDim + l;
        const auto slot_of = [&](unsigned i) { return kFusedStep * i + 2U * sp + half; };
        std::uint16_t raw[16];
        const auto load = [&](unsigned i) {
            const auto slot = slot_of(i);
            const auto token = slot < count ? selected[slot] : 0U;
            const auto* row = kbase + static_cast<std::uint64_t>(token) * (kKvWidth * 2ULL);
#pragma unroll
            for (unsigned j = 0; j < 16U; ++j) raw[j] = row[16U * j];
        };
        const unsigned iters = (count + kFusedStep - 1U) / kFusedStep;   // block-uniform: every lane runs the shuffles
        load(0U);
#pragma unroll 1
        for (unsigned i = 0; i < iters; ++i) {
            float kf[16];
#pragma unroll
            for (unsigned j = 0; j < 16U; ++j) kf[j] = bf16f(raw[j]);
            if (i + 1U < iters) load(i + 1U);
            const float partial = score6_slot(qv, kf, lane);
            const unsigned h = score6_head(l);
            const auto slot = slot_of(i);
            if (h < kHalfHeads && (l & 1U) == 0U && slot < count) sc[slot * kG + hh * kHalfHeads + h] = __fdiv_rn(partial, scale);
        }
    }
    __syncthreads();
    {   // 2. softmax of head `warp`
        float* const col = sc + warp;
        float maximum = (-kInf);
        for (unsigned i = lane; i < count; i += 32U) maximum = fmaxf(maximum, col[i * kG]);
#pragma unroll
        for (unsigned d = 16U; d > 0U; d >>= 1U) maximum = fmaxf(maximum, __shfl_xor_sync(kFull, maximum, static_cast<int>(d)));
        const auto add = [](float a, float b) { return __fadd_rn(a, b); };
        const auto exp_store = [&](unsigned i) {
            const float e = am::amd_expf(tables, __fsub_rn(col[i * kG], maximum));
            col[i * kG] = e;
            return e;
        };
        float den = fused_order::lane_partial(lane, count, add, exp_store);
#pragma unroll
        for (unsigned d = 16U; d > 0U; d >>= 1U) den = __fadd_rn(den, __shfl_down_sync(kFull, den, d));
        den = __shfl_sync(kFull, den, 0);
        __syncwarp();
        for (unsigned i = lane; i < count; i += 32U) col[i * kG] = round_bf16(__fdiv_rn(col[i * kG], den));
    }
    __syncthreads();
    {   // 3. weighted values + gate
        const unsigned g = threadIdx.x >> 7U, tau = threadIdx.x & 127U, hq = g * kFusedGroup;
        const std::uint16_t* vbase = kv + kKvWidth + kvh * kHeadDim + 2U * tau;
        float s0[kFusedGroup], s1[kFusedGroup];
#pragma unroll
        for (unsigned h = 0; h < kFusedGroup; ++h) { s0[h] = 0.0F; s1[h] = 0.0F; }
        const auto vrow = [&](unsigned slot) {
            return *reinterpret_cast<const std::uint32_t*>(vbase + static_cast<std::uint64_t>(selected[slot]) * (kKvWidth * 2ULL));
        };
        const auto step = [&](std::uint32_t raw, unsigned slot) {
            const float v0 = __uint_as_float(raw << 16U), v1 = __uint_as_float(raw & 0xffff0000U);
            const float4 w = *reinterpret_cast<const float4*>(sc + slot * kG + hq);
            const float wv[kFusedGroup] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (unsigned h = 0; h < kFusedGroup; ++h) { s0[h] = __fmaf_rn(wv[h], v0, s0[h]); s1[h] = __fmaf_rn(wv[h], v1, s1[h]); }
        };
        const auto fill = [&](std::uint32_t (&buf)[kFusedVB], unsigned i0) {
#pragma unroll
            for (unsigned b = 0; b < kFusedVB; ++b) buf[b] = vrow(i0 + b);
        };
        const auto drain = [&](const std::uint32_t (&buf)[kFusedVB], unsigned i0) {
#pragma unroll
            for (unsigned b = 0; b < kFusedVB; ++b) step(buf[b], i0 + b);
        };
        const unsigned full = count / kFusedVB * kFusedVB;
        std::uint32_t ra[kFusedVB], rb[kFusedVB];
        if (full) fill(ra, 0U);
#pragma unroll 1
        for (unsigned i0 = 0; i0 < full; i0 += 2U * kFusedVB) {   // slots ascending: ra = [i0, i0+8), rb = [i0+8, i0+16)
            if (i0 + kFusedVB < full) fill(rb, i0 + kFusedVB);
            drain(ra, i0);
            if (i0 + kFusedVB >= full) break;
            if (i0 + 2U * kFusedVB < full) fill(ra, i0 + 2U * kFusedVB);
            drain(rb, i0 + kFusedVB);
        }
        for (unsigned i = full; i < count; ++i) step(vrow(i), i);
        float* out = output + static_cast<std::uint64_t>(r) * kAttnWidth;
#pragma unroll
        for (unsigned h = 0; h < kFusedGroup; ++h) {
            const auto head = kvh * kG + hq + h;
#pragma unroll
            for (unsigned e = 0; e < 2U; ++e) {
                const auto f = 2U * tau + e;
                const float gin = qg[static_cast<std::uint64_t>(head) * kHeadDim * 2ULL + kHeadDim + f];
                out[static_cast<std::uint64_t>(head) * kHeadDim + f] = qsa_gated(e ? s1[h] : s0[h], gin, tables);
            }
        }
    }
}
__global__ __launch_bounds__(kFusedThreads, 1) void attention_fused_kernel(const float* __restrict__ q_gate,
    std::uint64_t first_position, const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv,
    float* __restrict__ output, am::Tables tables) {
    attention_fused_body(q_gate, first_position, all_selected, kv, output, tables);
}
bool fused_enabled() {
    static const bool on = strix::envcfg::flag("STRIX_QRTX_QSA_FUSED", false);
    return on;
}

// ------------------------------------------------------------------------------------------------------------------
// FAST form (Config::forms.qsa_mma, identity qwen-cuda-qsa-bf16mma-v2): the whole attention of a prefill row (scores +
// softmax + weighted values + gate) on BF16 tensor cores through the shared row-group core (core/attention_rowgroup.cuh).
// Four neighbouring rows walk the union of their selections (selected[] is ascending: blocks, then the tail, see
// the shared picker), so each K/V row is gathered from L2 once per 4 rows instead of once per row; one warp per row with
// its 12 heads as the M rows. Numerics: BF16 Q (RNE), BF16 K/V (the cache), FP32 accumulation, base-2 online softmax with
// ex2.approx over the union chunks ascending, P in BF16, attn = round_bf16(O / l), gate via __expf. Not bit-comparable
// with the exact path (v1, the per-row 3-warp kernel, merged three warp partials and is retired).
struct QsaRowgroup {
    static constexpr unsigned D = kHeadDim, Heads = kG, Rows = 4, Cap = kMaxSelected;
    static constexpr float kScaleLog2 = 0.0625F * 1.44269504088896341F;  // 1/sqrt(256) * log2(e)
    const float* q_gate;
    std::uint64_t first_position;
    const std::uint32_t* selected;
    const std::uint16_t* kv;
    float* output;
    std::uint32_t rows;
    __device__ unsigned count(unsigned row) const { return selected_count(first_position + row); }
    __device__ std::uint32_t key(unsigned row, unsigned j) const { return selected[static_cast<std::uint64_t>(row) * kMaxSelected + j]; }
    __device__ const std::uint16_t* key_row(std::uint32_t token, unsigned kvh) const {
        return kv + static_cast<std::uint64_t>(token) * (kKvWidth * 2ULL) + static_cast<std::uint64_t>(kvh) * kHeadDim;
    }
    __device__ const std::uint16_t* value_row(std::uint32_t token, unsigned kvh) const { return key_row(token, kvh) + kKvWidth; }
    __device__ float query(unsigned row, unsigned kvh, unsigned h, unsigned d) const {
        return q_gate[static_cast<std::uint64_t>(row) * kQGateWidth + static_cast<std::uint64_t>(kvh * kG + h) * kHeadDim * 2ULL + d];
    }
    __device__ void store2(unsigned row, unsigned kvh, unsigned h, unsigned d, float a0, float a1) const {
        const auto head = kvh * kG + h;
        const float* g = q_gate + static_cast<std::uint64_t>(row) * kQGateWidth + static_cast<std::uint64_t>(head) * kHeadDim * 2ULL + kHeadDim + d;
        const float g0 = round_bf16(1.0F / (1.0F + __expf(-g[0]))), g1 = round_bf16(1.0F / (1.0F + __expf(-g[1])));
        *reinterpret_cast<float2*>(output + static_cast<std::uint64_t>(row) * kAttnWidth + static_cast<std::uint64_t>(head) * kHeadDim + d) =
            make_float2(round_bf16(round_bf16(a0) * g0), round_bf16(round_bf16(a1) * g1));
    }
};
namespace rg = strix::cuda::core::rowgroup;
using QsaRg = rg::Shape<QsaRowgroup>;
static_assert(QsaRg::FlashSmem <= 48U * 1024U && QsaRg::UnionSmem <= 99U * 1024U, "qsa rowgroup smem");
// attention() hands its scores workspace ([rows][24][2051] f32) to the FAST form: 4 rows need 41 KB of union, scores give 788 KB.
static_assert(rg::workspace_bytes<QsaRowgroup>(1) <= std::size_t{QsaRowgroup::Rows} * kHeads * kMaxSelected * sizeof(float), "union fits the scores workspace");

// ------------------------------------------------------------------------------------------------------------------
// Multi-segment launches (decode cohort windows, layer_core_multi): one launch per step for every session of a window.
// Every block (score kernel: every wave) belongs to one segment and runs the single-call body with that segment's
// pointers; its first_position argument is position[s] - first_row[s] (modulo 2^64), so first_position + r, with r the
// window row the block already uses for the row buffers, is the session's own position. Same code, same per-row and
// per-head arithmetic as one call per session.
__device__ __forceinline__ unsigned seg_of(const SegmentMap& m, unsigned row) {
    unsigned seg = 0;
    while (seg + 1U < m.n && row >= m.first_row[seg + 1U]) ++seg;
    return seg;
}
__device__ __forceinline__ std::uint64_t base_position(const SegmentMap& m, unsigned seg) { return m.position[seg] - m.first_row[seg]; }
// The map arrives by value (eager launches) or as a device descriptor (graph replays: the positions, pooled ranges and wave
// counts change every window while the captured launches stay fixed). Same bodies either way.
__device__ __forceinline__ const SegmentMap& map_of(const SegmentMap& m) { return m; }
__device__ __forceinline__ const SegmentMap& map_of(const SegmentMap* m) { return *m; }

template <class MapArg>
__global__ void norm_rope_multi_kernel(float* __restrict__ values, std::uint64_t row_stride, std::uint32_t head_dim,
    std::uint32_t head_stride, const float* __restrict__ norm, MapArg mapa, const float2* __restrict__ cs, am::Tables tables) {
    const SegmentMap& map = map_of(mapa);
    norm_rope_kernel_body(values, row_stride, head_dim, head_stride, norm, base_position(map, seg_of(map, blockIdx.y)), cs, tables);
}
template <class MapArg>
__global__ void store_multi_kernel(const float* __restrict__ ik, const float* __restrict__ k, const float* __restrict__ v, MapArg mapa) {
    const SegmentMap& map = map_of(mapa);
    const unsigned seg = seg_of(map, blockIdx.y);
    store_kernel_body(ik, k, v, base_position(map, seg), map.raw_index[seg], map.kv[seg]);
}
// grid-stride over the window's pooled blocks (eager grid = the block count: one pass per CTA)
template <class MapArg>
__global__ void block_pool_multi_kernel(MapArg mapa, const float* __restrict__ key_norm, const float2* __restrict__ cs,
    am::Tables tables) {
    const SegmentMap& map = map_of(mapa);
    const unsigned total = map.pool_begin[map.n];
    for (unsigned i = blockIdx.x; i < total; i += gridDim.x) {
        unsigned seg = 0;
        while (seg + 1U < map.n && i >= map.pool_begin[seg + 1U]) ++seg;
        block_pool_kernel_body(map.raw_index[seg], map.pool_first[seg] - map.pool_begin[seg] + i, key_norm, cs, tables, map.pooled[seg]);
        __syncthreads();  // the next block reuses the shared buffers
    }
}
template <class MapArg>
__global__ __launch_bounds__(128) void attention_score_half_multi_kernel(const float* __restrict__ q_gate, MapArg mapa,
    const std::uint32_t* __restrict__ all_selected, float* __restrict__ all_scores) {
    const SegmentMap& map = map_of(mapa);
    const unsigned seg = seg_of(map, blockIdx.z);
    attention_score_half_kernel_body(q_gate, base_position(map, seg), all_selected, map.kv[seg], all_scores);
}
template <class MapArg>
__global__ __launch_bounds__(256) void softmax_multi_kernel(float* __restrict__ all_scores, MapArg mapa, am::Tables tables) {
    const SegmentMap& map = map_of(mapa);
    softmax_kernel_body(all_scores, base_position(map, seg_of(map, blockIdx.y)), tables);
}
template <class MapArg>
__global__ __launch_bounds__(kOutThreads) void attention_output_multi_kernel(const float* __restrict__ q_gate, MapArg mapa,
    const std::uint32_t* __restrict__ all_selected, const float* __restrict__ all_scores, float* __restrict__ output, am::Tables tables) {
    const SegmentMap& map = map_of(mapa);
    const unsigned seg = seg_of(map, blockIdx.y);
    attention_output_kernel_body(q_gate, base_position(map, seg), all_selected, map.kv[seg], all_scores, output, tables);
}
template <class MapArg>
__global__ __launch_bounds__(kOutThreads) void attention_output_head_multi_kernel(const float* __restrict__ q_gate, MapArg mapa,
    const std::uint32_t* __restrict__ all_selected, const float* __restrict__ all_scores, float* __restrict__ output, am::Tables tables) {
    const SegmentMap& map = map_of(mapa);
    const unsigned seg = seg_of(map, blockIdx.y);
    attention_output_head_kernel_body(q_gate, base_position(map, seg), all_selected, map.kv[seg], all_scores, output, tables);
}
template <class MapArg>
__global__ __launch_bounds__(kOutThreads) void attention_output_head_multi_pipe_kernel(const float* __restrict__ q_gate, MapArg mapa,
    const std::uint32_t* __restrict__ all_selected, const float* __restrict__ all_scores, float* __restrict__ output, am::Tables tables) {
    const SegmentMap& map = map_of(mapa);
    const unsigned seg = seg_of(map, blockIdx.y);
    attention_output_head_pipe_body(q_gate, base_position(map, seg), all_selected, map.kv[seg], all_scores, output, tables);
}
template <class MapArg>
__global__ __launch_bounds__(kFusedThreads, 1) void attention_fused_multi_kernel(const float* __restrict__ q_gate, MapArg mapa,
    const std::uint32_t* __restrict__ all_selected, float* __restrict__ output, am::Tables tables) {
    const SegmentMap& map = map_of(mapa);
    const unsigned seg = seg_of(map, blockIdx.y);
    attention_fused_body(q_gate, base_position(map, seg), all_selected, map.kv[seg], output, tables);
}

}  // namespace

// ------------------------------------------------------------------------------------------------------------------
std::size_t workspace_bytes(std::uint32_t chunk_rows, std::uint64_t max_positions) {
    const auto max_blocks = (max_positions + kRatio - 1U) / kRatio;
    return max_blocks * kIndexDim * 4ULL + static_cast<std::uint64_t>(chunk_rows) * max_blocks * 4ULL +
        static_cast<std::uint64_t>(chunk_rows) * kMaxSelected * 4ULL + static_cast<std::uint64_t>(chunk_rows) * kHeads * kMaxSelected * 4ULL;
}

Workspace allocate_workspace(std::uint32_t chunk_rows, std::uint64_t max_positions) {
    Workspace w;
    w.chunk_rows = chunk_rows;
    w.max_blocks = static_cast<std::uint32_t>((max_positions + kRatio - 1U) / kRatio);
    strix::cuda::require_vram(workspace_bytes(chunk_rows, max_positions), "qsa workspace");
    check(cudaMalloc(&w.pooled_keys, std::max<std::size_t>(1, static_cast<std::size_t>(w.max_blocks) * kIndexDim * 4U)), "alloc pooled");
    check(cudaMalloc(&w.block_scores, std::max<std::size_t>(1, static_cast<std::size_t>(chunk_rows) * w.max_blocks * 4U)), "alloc block scores");
    check(cudaMalloc(&w.selected, static_cast<std::size_t>(chunk_rows) * kMaxSelected * 4U), "alloc selected");
    check(cudaMalloc(&w.scores, static_cast<std::size_t>(chunk_rows) * kHeads * kMaxSelected * 4U), "alloc scores");
    return w;
}
void free_workspace(Workspace& w) {
    cudaFree(w.pooled_keys); cudaFree(w.block_scores); cudaFree(w.selected); cudaFree(w.scores);
    w = Workspace{};
}

void indexer_projection(const std::uint16_t* hidden, std::uint32_t rows, const LayerWeights& w, float* index_query,
    float* index_key, cudaStream_t stream) {
    if (!rows) return;
    static const bool attr = [] {
        check(cudaFuncSetAttribute(indexer_kernel<kIdxSlice>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kIdxSmem)), "indexer smem attr");
        check(cudaFuncSetAttribute(indexer_kernel<kIdxWide>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kIdxSmem)), "indexer smem attr");
        return true;
    }();
    (void)attr;
    if (rows <= kIdxRows && idx_wide_enabled())
        indexer_kernel<kIdxWide><<<dim3(1, kIdxOutputs / kIdxWide), 256, kIdxSmem, stream>>>(hidden, rows, w.index_q_proj, w.index_k_proj, index_query, index_key);
    else
    indexer_kernel<kIdxSlice><<<dim3((rows + kIdxRows - 1U) / kIdxRows, (kIdxOutputs + kIdxSlice - 1U) / kIdxSlice), 256, kIdxSmem, stream>>>(hidden, rows, w.index_q_proj, w.index_k_proj, index_query, index_key);
    check(cudaGetLastError(), "indexer launch");
}

void norm_rope(float* values, std::uint32_t rows, std::uint64_t row_stride, std::uint32_t heads, std::uint32_t head_dim,
    std::uint32_t head_stride, const float* norm, std::uint64_t first_position, const RopeTable* cs,
    const amdmath::Tables& tables, cudaStream_t stream) {
    if (!rows) return;
    if (head_dim != 128U && head_dim != 256U) throw std::invalid_argument("qwen_qsa norm_rope: head_dim 128 or 256");
    norm_rope_kernel<<<dim3(heads, rows), head_dim, 2U * head_dim * sizeof(float), stream>>>(values, row_stride, head_dim,
        head_stride, norm, first_position, cs, tables);
    check(cudaGetLastError(), "norm_rope launch");
}

void store_rows(const float* index_key, const float* k, const float* v, std::uint32_t rows, std::uint64_t first_position,
    const LayerState& state, cudaStream_t stream) {
    if (!rows) return;
    store_kernel<<<dim3(kKvWidth / 256U, rows), 256, 0, stream>>>(index_key, k, v, first_position, state.raw_index, state.kv);
    check(cudaGetLastError(), "store launch");
}

void block_pool(const std::uint16_t* raw_index, std::uint32_t first_block, std::uint32_t count, const float* index_k_norm,
    const RopeTable* cs, const amdmath::Tables& tables, float* pooled_keys, cudaStream_t stream) {
    if (!count) return;
    block_pool_kernel<<<count, kIndexDim, 0, stream>>>(raw_index, first_block, index_k_norm, cs, tables, pooled_keys);
    check(cudaGetLastError(), "block pool launch");
}

void block_scores(const float* index_query, std::uint32_t rows, std::uint64_t first_position, const float* pooled_keys,
    float* scores, std::uint32_t max_blocks, cudaStream_t stream) {
    block_scores_form(score_tree_enabled() ? ScoreForm::tree : ScoreForm::shuffle, index_query, rows, first_position, pooled_keys,
        scores, max_blocks, stream);
}

void block_scores_form(ScoreForm form, const float* index_query, std::uint32_t rows, std::uint64_t first_position,
    const float* pooled_keys, float* scores, std::uint32_t max_blocks, cudaStream_t stream) {
    if (!rows) return;
    if (first_position > std::numeric_limits<std::uint64_t>::max() - rows)
        throw std::invalid_argument("qwen_qsa block_scores: visible length exceeds uint64_t");
    const auto groups = (first_position + rows) / kRatio;
    if (groups <= kTopK) return;
    const selection::ScoreSegmentInput segment{rows, first_position, pooled_keys};
    ScoreMap map;
    const selection::SegmentScoreShape shape{rows, rows, max_blocks};
    if (!selection::pack_score_segments(std::span{&segment, 1U}, shape, kBlockScoreSpec, map))
        throw std::invalid_argument("qwen_qsa block_scores: invalid segment");
    const selection::SegmentRows input{{index_query, nullptr, scores, nullptr}, {kIndexQWidth, max_blocks, 0}, map};
    enqueue_block_scores(input, {rows, rows, static_cast<unsigned>(groups)},
        form == ScoreForm::tree && tree_aligned(index_query) && tree_aligned(pooled_keys), stream);
}

// The largest selected count of rows [first_position, first_position + rows) (kMaxSelected at most).
static std::uint32_t max_selected_count(std::uint64_t first_position, std::uint32_t rows) {
    std::uint32_t slots = 0;
    for (std::uint64_t p = first_position; p < first_position + rows && slots < kMaxSelected; ++p) slots = std::max(slots, selected_count(p));
    return slots;
}

void select_rows(float* scores, std::uint32_t max_blocks, std::uint32_t rows, std::uint64_t first_position,
    std::uint32_t* selected, cudaStream_t stream) {
    if (!rows) return;
    if (first_position > std::numeric_limits<std::uint32_t>::max() - rows)
        throw std::invalid_argument("qwen_qsa select_rows: visible length exceeds uint32_t");
    const selection::AffineRows affine{{nullptr, nullptr, scores, selected}, {0, max_blocks, kMaxSelected},
        static_cast<std::uint32_t>(first_position)};
    check(selection::enqueue_pick(affine, rows, kBlockPickSpec, stream), "select launch");
}

void attention_scores(const float* q_gate, std::uint32_t rows, std::uint64_t first_position, const std::uint32_t* selected,
    const std::uint16_t* kv, float* scores, cudaStream_t stream) {
    if (!rows) return;
    const auto slots = max_selected_count(first_position, rows);
    static const bool wave = strix::envcfg::flag("STRIX_QSA_SCORE_WAVE", false);
    if (wave) {
        const auto waves = static_cast<std::uint64_t>(rows) * (kHeads / kG) * ((slots + kSlots - 1U) / kSlots);
        attention_score_kernel<<<static_cast<std::uint32_t>((waves + 7U) / 8U), 256, 0, stream>>>(q_gate, rows, first_position,
            selected, kv, scores, slots);
    } else {
        attention_score_half_kernel<<<dim3((slots + kRun - 1U) / kRun, kKvHeads, rows), 128, 0, stream>>>(q_gate, first_position,
            selected, kv, scores);
    }
    check(cudaGetLastError(), "score launch");
}
void attention_softmax(std::uint32_t rows, std::uint64_t first_position, float* scores, const amdmath::Tables& tables, cudaStream_t stream) {
    if (!rows) return;
    softmax_kernel<<<dim3(kHeads, rows), 256, 0, stream>>>(scores, first_position, tables);
    check(cudaGetLastError(), "softmax launch");
}
void attention_values(const float* q_gate, std::uint32_t rows, std::uint64_t first_position, const std::uint32_t* selected,
    const std::uint16_t* kv, const float* scores, float* attention_output, const amdmath::Tables& tables, cudaStream_t stream) {
    if (!rows) return;
    if (rows <= core::decode_rows_max() && decode_rows_enabled() && head_pipe_enabled())   // per-(head, row) blocks: any row count
        attention_output_head_pipe_kernel<<<dim3(kHeads, rows), kOutThreads, 0, stream>>>(q_gate, first_position, selected, kv, scores,
            attention_output, tables);
    else if (rows <= core::decode_rows_max() && decode_rows_enabled())
        attention_output_head_kernel<<<dim3(kHeads, rows), kOutThreads, 0, stream>>>(q_gate, first_position, selected, kv, scores,
            attention_output, tables);
    else
        attention_output_kernel<<<dim3(kKvHeads, rows), kOutThreads, 0, stream>>>(q_gate, first_position, selected, kv, scores,
            attention_output, tables);
    check(cudaGetLastError(), "output launch");
}
void attention(const float* q_gate, std::uint32_t rows, std::uint64_t first_position, const std::uint32_t* selected,
    const std::uint16_t* kv, float* scores, float* attention_output, const amdmath::Tables& tables, cudaStream_t stream, bool mma) {
    if (mma) { attention_mma(q_gate, rows, first_position, selected, kv, scores, attention_output, stream); return; }
    // the decode shape keeps the per-head output kernels (attention_values); the fused form serves the prefill shape
    if (fused_enabled() && !(rows <= core::decode_rows_max() && decode_rows_enabled())) {
        attention_fused(q_gate, rows, first_position, selected, kv, attention_output, tables, stream);
        return;
    }
    attention_scores(q_gate, rows, first_position, selected, kv, scores, stream);
    attention_softmax(rows, first_position, scores, tables, stream);
    attention_values(q_gate, rows, first_position, selected, kv, scores, attention_output, tables, stream);
}
namespace {
void fused_smem_attr() {
    static const bool attr = [] {
        check(cudaFuncSetAttribute(attention_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(fused_smem_bytes(kMaxSelected))), "fused smem attr");
        return true;
    }();
    (void)attr;
}
}  // namespace
void attention_fused(const float* q_gate, std::uint32_t rows, std::uint64_t first_position, const std::uint32_t* selected,
    const std::uint16_t* kv, float* attention_output, const amdmath::Tables& tables, cudaStream_t stream) {
    if (!rows) return;
    fused_smem_attr();
    attention_fused_kernel<<<dim3(kKvHeads, rows), kFusedThreads, fused_smem_bytes(max_selected_count(first_position, rows)), stream>>>(
        q_gate, first_position, selected, kv, attention_output, tables);
    check(cudaGetLastError(), "fused attention launch");
}
std::size_t attention_mma_workspace_bytes(std::uint32_t rows) {
    const auto tiles = rows ? 1U + (rows - 1U) / QsaRowgroup::Rows : 0U;
    return rg::workspace_bytes<QsaRowgroup>(tiles);
}
void prepare_attention_mma() {
    rg::prepare_rowgroup<QsaRowgroup>();
}
void enqueue_attention_mma(const float* q_gate, std::uint32_t rows, std::uint64_t first_position,
    const std::uint32_t* selected, const std::uint16_t* kv, void* workspace, float* attention_output,
    cudaStream_t stream) {
    const QsaRowgroup p{q_gate, first_position, selected, kv, attention_output, rows};
    rg::enqueue_rowgroup(p, kKvHeads, workspace, attention_mma_workspace_bytes(rows), stream);
}
namespace {
// Keep direct eager callers working. Streaming engine enqueues use the explicit
// prepare call at context setup. A direct captured call must be warmed first.
void prepare_attention_mma_if_eager(cudaStream_t stream) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (stream != nullptr && stream != cudaStreamLegacy)
        check(cudaStreamIsCapturing(stream, &capture), "rowgroup capture status");
    if (capture == cudaStreamCaptureStatusNone) prepare_attention_mma();
}
}  // namespace

void attention_mma(const float* q_gate, std::uint32_t rows, std::uint64_t first_position, const std::uint32_t* selected,
    const std::uint16_t* kv, void* workspace, float* attention_output, cudaStream_t stream) {
    if (!rows) return;
    prepare_attention_mma_if_eager(stream);
    enqueue_attention_mma(q_gate, rows, first_position, selected, kv, workspace, attention_output, stream);
}

void layer_core_prepared(const RowBuffers& b, std::uint32_t rows, std::uint64_t first_position, const LayerWeights& w,
    const LayerState& state, const RopeTable* cs, const amdmath::Tables& tables, Workspace& ws, cudaStream_t stream, bool mma_attention) {
    if (!rows) return;
    if (first_position + rows > static_cast<std::uint64_t>(ws.max_blocks) * kRatio + kRatio)
        throw std::invalid_argument("qwen_qsa layer_core: workspace too small for the positions");
    indexer_projection(b.hidden, rows, w, b.index_query, b.index_key, stream);
    // H:4588-4599: index_q, q, k norms (each row only touches its own values; launch order irrelevant).
    norm_rope(b.index_query, rows, kIndexQWidth, kIndexHeads, kIndexDim, kIndexDim, w.index_q_norm, first_position, cs, tables, stream);
    norm_rope(b.q_gate, rows, kQGateWidth, kHeads, kHeadDim, kHeadDim * 2U, w.q_norm, first_position, cs, tables, stream);
    norm_rope(b.k, rows, kKvWidth, kKvHeads, kHeadDim, kHeadDim, w.k_norm, first_position, cs, tables, stream);
    // Every row's state before any row attends: rows read only positions <= their own, so storing the whole range first
    // equals the HIP group-by-group order (H:5148-5159).
    store_rows(b.index_key, b.k, b.v, rows, first_position, state, stream);
    const auto last_complete = static_cast<std::uint32_t>((first_position + rows) / kRatio);
    float* pooled = state.pooled ? state.pooled : ws.pooled_keys;
    if (last_complete > kTopK) {
        if (last_complete > ws.max_blocks) throw std::invalid_argument("qwen_qsa layer_core: max_blocks");
        std::uint32_t first = 0;
        if (state.pooled) first = std::min(*state.pooled_valid, static_cast<std::uint32_t>(first_position / kRatio));
        if (last_complete > first) block_pool(state.raw_index, first, last_complete - first, w.index_k_norm, cs, tables, pooled, stream);
        if (state.pooled) *state.pooled_valid = last_complete;
    }
    for (std::uint32_t c = 0; c < rows; c += ws.chunk_rows) {
        const auto n = std::min(ws.chunk_rows, rows - c);
        const auto p0 = first_position + c;
        block_scores(b.index_query + static_cast<std::uint64_t>(c) * kIndexQWidth, n, p0, pooled, ws.block_scores, ws.max_blocks, stream);
        select_rows(ws.block_scores, ws.max_blocks, n, p0, ws.selected, stream);
        const auto* q_gate = b.q_gate + static_cast<std::uint64_t>(c) * kQGateWidth;
        auto* out = b.attention_output + static_cast<std::uint64_t>(c) * kAttnWidth;
        if (mma_attention) enqueue_attention_mma(q_gate, n, p0, ws.selected, state.kv, ws.scores, out, stream);
        else attention(q_gate, n, p0, ws.selected, state.kv, ws.scores, out, tables, stream);
    }
}

void layer_core(const RowBuffers& b, std::uint32_t rows, std::uint64_t first_position, const LayerWeights& w,
    const LayerState& state, const RopeTable* cs, const amdmath::Tables& tables, Workspace& ws, cudaStream_t stream, bool mma_attention) {
    if (rows && mma_attention) prepare_attention_mma_if_eager(stream);
    layer_core_prepared(b, rows, first_position, w, state, cs, tables, ws, stream, mma_attention);
}

bool prepare_multi(const SegmentState* segs, std::uint32_t n, const Workspace& ws, SegmentMap& map, MultiShape& shape,
    std::uint32_t* valid_after) {
    if (n == 0 || n > kMaxSegments) throw std::invalid_argument("qwen_qsa layer_core_multi: 1..8 segments");
    static const bool score_wave = strix::envcfg::flag("STRIX_QSA_SCORE_WAVE", false);
    // Every check and the whole segment map before the first launch; the host pooled-valid counts are written only after
    // the launches (a throw here leaves the sessions as they were).
    map = SegmentMap{};
    map.n = n;
    std::uint32_t rows = 0, max_rows = 0, min_rows = 0xffffffffU;
    for (std::uint32_t i = 0; i < n; ++i) {
        const auto& g = segs[i];
        if (g.rows == 0) throw std::invalid_argument("qwen_qsa layer_core_multi: empty segment");
        if (g.first_position > std::numeric_limits<std::uint32_t>::max() - g.rows ||
            g.first_position + g.rows > static_cast<std::uint64_t>(ws.max_blocks) * kRatio + kRatio)
            throw std::invalid_argument("qwen_qsa layer_core: workspace too small for the positions");
        if (!g.state.pooled && n > 1) throw std::invalid_argument("qwen_qsa layer_core_multi: sessions need their own pooled cache");
        if (static_cast<std::uint32_t>((g.first_position + g.rows) / kRatio) > ws.max_blocks && (g.first_position + g.rows) / kRatio > kTopK)
            throw std::invalid_argument("qwen_qsa layer_core: max_blocks");
        map.first_row[i] = rows;
        map.position[i] = g.first_position;
        map.kv[i] = g.state.kv;
        map.raw_index[i] = g.state.raw_index;
        rows += g.rows;
        max_rows = std::max(max_rows, g.rows);
        min_rows = std::min(min_rows, g.rows);
    }
    map.first_row[n] = rows;
    shape = MultiShape{};
    // the output kernel per-segment layer_core chooses for each segment (head kernel: <= 8 rows with the decode kernels on);
    // the multi launch needs one choice for all, else per segment
    const bool head_all = max_rows <= core::decode_rows_max() && decode_rows_enabled(), rows_all = !decode_rows_enabled() || min_rows > core::decode_rows_max();
    if (rows > ws.chunk_rows || score_wave || !(head_all || rows_all)) return false;  // outside the multi-launch shapes
    // pooled keys per session (host-side valid counts, as layer_core), block-score waves, attention slots
    std::uint32_t pool = 0, waves = 0, slots = 0, tree_x = 0, tree_y = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        const auto& g = segs[i];
        const auto last_complete = static_cast<std::uint32_t>((g.first_position + g.rows) / kRatio);
        map.pooled[i] = g.state.pooled ? g.state.pooled : ws.pooled_keys;
        map.pool_begin[i] = pool;
        map.pool_first[i] = 0;
        valid_after[i] = g.state.pooled ? *g.state.pooled_valid : 0U;
        if (last_complete > kTopK) {
            std::uint32_t first = 0;
            if (g.state.pooled) first = std::min(*g.state.pooled_valid, static_cast<std::uint32_t>(g.first_position / kRatio));
            map.pool_first[i] = first;
            if (last_complete > first) pool += last_complete - first;
            valid_after[i] = last_complete;
        }
        // block scores: as block_scores() for this segment's rows
        map.score_begin[i] = waves;
        const auto scored = last_complete;  // (first_position + rows) / kRatio: complete blocks of the segment's last row
        map.scored[i] = scored;
        if (scored > kTopK) {
            waves += static_cast<std::uint32_t>((g.rows + kScoreRowsPerWave - 1U) / kScoreRowsPerWave * ((scored + 31U) / 32U));
            tree_x = std::max(tree_x, (g.rows + kTreeRows - 1U) / kTreeRows);
            tree_y = std::max(tree_y, (scored + kTreeSeg - 1U) / kTreeSeg);
        }
        for (std::uint64_t p = g.first_position; p < g.first_position + g.rows && slots < kMaxSelected; ++p) slots = std::max(slots, selected_count(p));
    }
    map.pool_begin[n] = pool;
    map.score_begin[n] = waves;
    shape.rows = rows;
    shape.head_all = head_all;
    shape.pool_grid = pool;
    shape.score_grid = (waves + 7U) / 8U;
    shape.tree_grid = dim3(tree_x, tree_y, n);
    for (std::uint32_t i = 0; i < n; ++i) shape.tree_aligned = shape.tree_aligned && tree_aligned(map.pooled[i]);
    shape.slot_runs = (slots + kRun - 1U) / kRun;
    return true;
}

namespace {
selection::SegmentScoreShape score_shape(const MultiShape& shape, const Workspace& ws) {
    // The existing Qwen shape records rounded launch extents.
    const auto max_rows = std::min<std::uint64_t>(shape.rows, std::uint64_t{shape.tree_grid.x} * kTreeRows);
    const auto groups = std::min<std::uint64_t>(ws.max_blocks, std::uint64_t{shape.tree_grid.y} * kTreeSeg);
    return {shape.rows, static_cast<unsigned>(max_rows), static_cast<unsigned>(groups)};
}

ScoreMap pack_scores(const SegmentMap& map, const Workspace& ws, std::uint32_t rows) {
    selection::ScoreSegmentInput input[kMaxSegments]{};
    for (std::uint32_t i = 0; i < map.n; ++i)
        input[i] = {map.first_row[i + 1U] - map.first_row[i], map.position[i], map.pooled[i]};
    ScoreMap packed;
    // A dense prefix can have more groups than its unused score storage.
    if (!selection::pack_score_segments(std::span<const selection::ScoreSegmentInput>{input, map.n},
            {rows, rows, std::max(ws.max_blocks, kTopK)}, kBlockScoreSpec, packed))
        throw std::invalid_argument("qwen_qsa layer_core_multi: invalid score segments");
    return packed;
}

// The public launcher also accepts an older, state-only device descriptor.
// Project its fields into K5B's segment layout. Capture reads the fixed count
// and shuffle grid from MultiShape; only device code reads the replay payload.
struct StateMapScoreView {
    static constexpr auto capacity = kMaxSegments;
    struct Count {
        const SegmentMap* device;
        unsigned captured;
        __host__ __device__ operator unsigned() const {
#if defined(__CUDA_ARCH__)
            return device->n;
#else
            return captured;
#endif
        }
    } count;
    struct Segments {
        const SegmentMap* device;
        __host__ __device__ selection::detail::ScoreSegment operator[](unsigned i) const {
#if defined(__CUDA_ARCH__)
            return {device->position[i], device->pooled[i], device->first_row[i],
                device->first_row[i + 1U] - device->first_row[i], device->scored[i], device->score_begin[i + 1U]};
#else
            // A nonzero marker keeps the node in the graph even for a cold
            // descriptor. launch_multi supplies the fixed shuffle budget.
            (void)i;
            return {.wave_end = 1};
#endif
        }
    } segments;
};

template <class MapArg, class ScoreArg>
void launch_multi(const RowBuffers& b, MapArg map, ScoreArg score_map, const MultiShape& shape, const LayerWeights& w,
    const RopeTable* cs, const amdmath::Tables& tables, Workspace& ws, cudaStream_t stream) {
    const auto rows = shape.rows;
    indexer_projection(b.hidden, rows, w, b.index_query, b.index_key, stream);  // row-wise
    norm_rope_multi_kernel<<<dim3(kIndexHeads, rows), kIndexDim, 2U * kIndexDim * sizeof(float), stream>>>(b.index_query, kIndexQWidth,
        kIndexDim, kIndexDim, w.index_q_norm, map, cs, tables);
    norm_rope_multi_kernel<<<dim3(kHeads, rows), kHeadDim, 2U * kHeadDim * sizeof(float), stream>>>(b.q_gate, kQGateWidth, kHeadDim,
        kHeadDim * 2U, w.q_norm, map, cs, tables);
    norm_rope_multi_kernel<<<dim3(kKvHeads, rows), kHeadDim, 2U * kHeadDim * sizeof(float), stream>>>(b.k, kKvWidth, kHeadDim, kHeadDim,
        w.k_norm, map, cs, tables);
    check(cudaGetLastError(), "norm_rope multi launch");
    store_multi_kernel<<<dim3(kKvWidth / 256U, rows), 256, 0, stream>>>(b.index_key, b.k, b.v, map);
    check(cudaGetLastError(), "store multi launch");
    if (shape.pool_grid) {
        block_pool_multi_kernel<<<shape.pool_grid, kIndexDim, 0, stream>>>(map, w.index_k_norm, cs, tables);
        check(cudaGetLastError(), "block pool multi launch");
    }
    const selection::SegmentRows input{{b.index_query, nullptr, ws.block_scores, ws.selected},
        {kIndexQWidth, ws.max_blocks, kMaxSelected}, score_map};
    if (shape.score_grid)
        enqueue_block_scores(input, score_shape(shape, ws),
            score_tree_enabled() && shape.tree_aligned && tree_aligned(b.index_query), stream, shape.score_grid);
    check(selection::enqueue_pick(input, rows, kBlockPickSpec, stream), "select multi launch");
    if (!shape.head_all && fused_enabled()) {   // a prefill-shaped segment: the exact fused form (same bits as below)
        fused_smem_attr();
        const auto slots = std::min<std::uint32_t>(shape.slot_runs * kRun, kMaxSelected);
        attention_fused_multi_kernel<<<dim3(kKvHeads, rows), kFusedThreads, fused_smem_bytes(slots), stream>>>(b.q_gate, map,
            ws.selected, b.attention_output, tables);
        check(cudaGetLastError(), "fused attention multi launch");
        return;
    }
    attention_score_half_multi_kernel<<<dim3(shape.slot_runs, kKvHeads, rows), 128, 0, stream>>>(b.q_gate, map, ws.selected, ws.scores);
    check(cudaGetLastError(), "score multi launch");
    softmax_multi_kernel<<<dim3(kHeads, rows), 256, 0, stream>>>(ws.scores, map, tables);
    check(cudaGetLastError(), "softmax multi launch");
    // keyed on the segment rows (as per-segment layer_core), not on the window total
    if (shape.head_all && head_pipe_enabled())
        attention_output_head_multi_pipe_kernel<<<dim3(kHeads, rows), kOutThreads, 0, stream>>>(b.q_gate, map, ws.selected, ws.scores,
            b.attention_output, tables);
    else if (shape.head_all)
        attention_output_head_multi_kernel<<<dim3(kHeads, rows), kOutThreads, 0, stream>>>(b.q_gate, map, ws.selected, ws.scores,
            b.attention_output, tables);
    else
        attention_output_multi_kernel<<<dim3(kKvHeads, rows), kOutThreads, 0, stream>>>(b.q_gate, map, ws.selected, ws.scores,
            b.attention_output, tables);
    check(cudaGetLastError(), "output multi launch");
}
}  // namespace

void layer_core_multi(const RowBuffers& b, const SegmentState* segs, std::uint32_t n, const LayerWeights& w, const RopeTable* cs,
    const amdmath::Tables& tables, Workspace& ws, cudaStream_t stream) {
    SegmentMap map{};
    MultiShape shape{};
    std::uint32_t valid_after[kMaxSegments] = {};
    if (!prepare_multi(segs, n, ws, map, shape, valid_after)) {  // outside the multi-launch shapes: one call per segment
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::uint64_t r = map.first_row[i];
            const RowBuffers sb{b.hidden + r * kHidden, b.q_gate + r * kQGateWidth, b.k + r * kKvWidth, b.v + r * kKvWidth,
                                b.index_query + r * kIndexQWidth, b.index_key + r * kIndexDim, b.attention_output + r * kAttnWidth};
            layer_core(sb, segs[i].rows, segs[i].first_position, w, segs[i].state, cs, tables, ws, stream);
        }
        return;
    }
    launch_multi(b, map, pack_scores(map, ws, shape.rows), shape, w, cs, tables, ws, stream);
    for (std::uint32_t i = 0; i < n; ++i)
        if (segs[i].state.pooled) *segs[i].state.pooled_valid = valid_after[i];
}

bool graph_shape(const std::uint32_t* seg_rows, std::uint32_t n, const Workspace& ws, MultiShape& shape) {
    if (n == 0 || n > kMaxSegments) return false;
    static const bool score_wave = strix::envcfg::flag("STRIX_QSA_SCORE_WAVE", false);
    std::uint32_t rows = 0, max_rows = 0, min_rows = 0xffffffffU, waves = 0;
    const std::uint32_t chunks = (ws.max_blocks + 31U) / 32U;  // waves per row group at the workspace's last block
    for (std::uint32_t i = 0; i < n; ++i) {
        if (seg_rows[i] == 0) return false;
        rows += seg_rows[i];
        max_rows = std::max(max_rows, seg_rows[i]);
        min_rows = std::min(min_rows, seg_rows[i]);
        waves += (seg_rows[i] + kScoreRowsPerWave - 1U) / kScoreRowsPerWave * chunks;
    }
    const bool head_all = max_rows <= core::decode_rows_max() && decode_rows_enabled(), rows_all = !decode_rows_enabled() || min_rows > core::decode_rows_max();
    if (rows > ws.chunk_rows || score_wave || !(head_all || rows_all)) return false;
    static const std::uint32_t sms = [] {
        int dev = 0, count = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) return 64U;
        return static_cast<std::uint32_t>(std::max(count, 1));
    }();
    shape.rows = rows;
    shape.head_all = head_all;
    // Replay may start with a cold pool, regardless of the window's row count. Empty CTAs exit in the existing kernel.
    shape.pool_grid = 2U * sms;
    shape.score_grid = std::max(1U, std::min((waves + 7U) / 8U, 2U * sms));   // grid-stride beyond
    shape.tree_grid = dim3((max_rows + kTreeRows - 1U) / kTreeRows, (ws.max_blocks + kTreeSeg - 1U) / kTreeSeg, n);
    shape.slot_runs = (kMaxSelected + kRun - 1U) / kRun;                      // runs past a row's selected count return at once
    return true;
}

void layer_core_device(const RowBuffers& b, const ReplaySegmentMap* device_map, const MultiShape& shape, const LayerWeights& w,
    const RopeTable* cs, const amdmath::Tables& tables, Workspace& ws, cudaStream_t stream) {
    launch_multi(b, &device_map->state, &device_map->score, shape, w, cs, tables, ws, stream);
}

void layer_core_device(const RowBuffers& b, const SegmentMap* device_map, const MultiShape& shape, const LayerWeights& w,
    const RopeTable* cs, const amdmath::Tables& tables, Workspace& ws, cudaStream_t stream) {
    const StateMapScoreView scores{{device_map, shape.tree_grid.z}, {device_map}};
    launch_multi(b, device_map, scores, shape, w, cs, tables, ws, stream);
}

void prepare_replay(const SegmentState* segs, std::uint32_t n, const Workspace& ws, ReplaySegmentMap& map) {
    MultiShape shape{};
    std::uint32_t valid_after[kMaxSegments] = {};
    if (!prepare_multi(segs, n, ws, map.state, shape, valid_after))
        throw std::logic_error("qwen_qsa prepare_replay: window outside the captured multi-launch shape");
    // a captured graph keeps the scorer chosen at capture: refuse pooled keys the captured tree scorer cannot read
    if (score_tree_enabled() && !shape.tree_aligned)
        throw std::logic_error("qwen_qsa prepare_replay: pooled keys not 16-B aligned for the captured tree scorer");
    map.score = pack_scores(map.state, ws, shape.rows);
    for (std::uint32_t i = 0; i < n; ++i)
        if (segs[i].state.pooled) *segs[i].state.pooled_valid = valid_after[i];
}

void prepare_replay(const SegmentState* segs, std::uint32_t n, const Workspace& ws, SegmentMap& map) {
    ReplaySegmentMap replay;
    prepare_replay(segs, n, ws, replay);
    map = replay.state;
}

}  // namespace strix::cuda::qwen_qsa
