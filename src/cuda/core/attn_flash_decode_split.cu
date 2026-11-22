// Decode schedule of the prepared D128 GQA flash binding (FAST class): key split by strix/attention_split_plan.hpp and a
// ticket merge by the last CTA of each KV head. Contract: attention.hpp attention_decode_prepared_split_bf16.
//
// Why: the prefill flash used for one decode row runs one CTA per query head with a 128-row tile of which one row is real
// and walks every key tile serially (24 CTAs on 188 SMs for Inkling 2 Mini). Here one CTA = one KV head with its query
// heads packed (K/V read once per group) x one chunk of <= kMaxSplits chunks; one key per thread for the scores, one
// dimension pair per thread for P V, everything FP32. Separate translation unit: the prefill kernels are untouched.
#include "attn_decode.cuh"

#include "common.cuh"
#include "attn_core/masks.hpp"
#include "attn_core/relative_bias.hpp"
#include "strix/attention_split_plan.hpp"

#include <cmath>
#include <cstdint>

namespace strix::cuda_core {
namespace decode_split {

constexpr unsigned kThreads = attn_split::kUnit;  // one key per thread in the score pass
constexpr unsigned kDim = 128, kMaxGroup = 4, kWarps = kThreads / 32;
constexpr unsigned kPairs = kDim / 2, kQuarters = kThreads / kPairs;
constexpr unsigned kRecord = 2U + kDim;  // floats per (kv head, split, group row): m (log2 domain), l, o[kDim]
static_assert(kThreads % kPairs == 0 && kQuarters >= 1 && kRecord % 2 == 0);

std::size_t ticket_bytes(std::uint32_t kv_heads) { return (static_cast<std::size_t>(kv_heads) * 4U + 15U) / 16U * 16U; }

// Fixed-order block reduction of G values per thread: warp xor tree, then warps 0 .. 7 ascending. Every thread returns
// the same value. Ends with the block synchronised and red reusable.
template <unsigned G, bool Max>
__device__ __forceinline__ void block_reduce(float (&v)[G], float (&red)[G][kWarps]) {
    const unsigned lane = threadIdx.x & 31U, warp = threadIdx.x >> 5;
#pragma unroll
    for (unsigned r = 0; r < G; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float o = __shfl_xor_sync(0xffffffffu, v[r], off);
            v[r] = Max ? fmaxf(v[r], o) : v[r] + o;
        }
        if (lane == 0) red[r][warp] = v[r];
    }
    __syncthreads();
#pragma unroll
    for (unsigned r = 0; r < G; ++r) {
        float a = red[r][0];
#pragma unroll
        for (unsigned w = 1; w < kWarps; ++w) a = Max ? fmaxf(a, red[r][w]) : a + red[r][w];
        v[r] = a;
    }
    __syncthreads();
}

template <unsigned G>
__global__ void __launch_bounds__(kThreads)
    decode_split_kernel(AttentionDims d, PreparedBf16Flash prepared, const float* q, const __nv_bfloat16* kc,
                        const __nv_bfloat16* vc, const std::uint32_t* position, float* out, std::uint32_t* tickets,
                        float* partials, std::uint32_t capacity) {
    __shared__ __align__(16) float sq[G][kDim];
    __shared__ float sp[G][kThreads];
    __shared__ float red[G][kWarps];
    __shared__ __align__(16) float sacc[kQuarters - 1U][G][kDim];
    __shared__ bool last;
    constexpr float log2e = 1.4426950408889634f;
    constexpr float excluded = attn_core::mask::excluded_score;
    const unsigned kvh = blockIdx.x, split = blockIdx.y, t = threadIdx.x;
    const unsigned pos = *position;
    const auto plan = attn_split::decode_split_plan(pos, prepared.window);
    if (split >= plan.splits) return;  // uniform per CTA: never takes a ticket
    const unsigned k_begin = plan.first + split * plan.chunk;
    const unsigned k_end = min(k_begin + plan.chunk, pos + 1U);
    const unsigned head0 = kvh * G;
    for (unsigned i = t; i < G * kDim; i += kThreads) sq[i / kDim][i % kDim] = q[static_cast<std::size_t>(head0) * kDim + i];
    const float sl2 = d.scale * log2e;
    const attn_core::RelativeBiasRows bias{prepared.relative_bias, d.heads, prepared.relative_extent};
    const std::size_t row_stride = static_cast<std::size_t>(d.kv_heads) * kDim;
    const __nv_bfloat16* kh = kc + static_cast<std::size_t>(kvh) * kDim;
    const __nv_bfloat16* vh = vc + static_cast<std::size_t>(kvh) * kDim;
    const bool wrapped = prepared.wrapped;
    const unsigned slots = d.max_positions;
    float m[G], l[G], acc[G][2];
#pragma unroll
    for (unsigned r = 0; r < G; ++r) { m[r] = excluded; l[r] = 0.0f; acc[r][0] = acc[r][1] = 0.0f; }
    const unsigned dp = t % kPairs, quarter = t / kPairs;
    __syncthreads();

    for (unsigned base = k_begin; base < k_end; base += kThreads) {
        const unsigned key = base + t;
        const bool valid = key < k_end;
        float s[G];
        if (valid) {
            const unsigned slot = wrapped ? key % slots : key;
            const uint4* kr = reinterpret_cast<const uint4*>(kh + static_cast<std::size_t>(slot) * row_stride);
            float dot[G];
#pragma unroll
            for (unsigned r = 0; r < G; ++r) dot[r] = 0.0f;
#pragma unroll  // all 16 row loads in flight: one latency round
            for (unsigned c = 0; c < kDim / 8U; ++c) {
                const uint4 w = __ldg(kr + c);
                const std::uint32_t wv[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (unsigned e = 0; e < 4; ++e) {
                    const float lo = __uint_as_float(wv[e] << 16), hi = __uint_as_float(wv[e] & 0xffff0000u);
#pragma unroll
                    for (unsigned r = 0; r < G; ++r) {
                        dot[r] = fmaf(sq[r][8U * c + 2U * e], lo, dot[r]);
                        dot[r] = fmaf(sq[r][8U * c + 2U * e + 1U], hi, dot[r]);
                    }
                }
            }
#pragma unroll
            for (unsigned r = 0; r < G; ++r) s[r] = fmaf(dot[r], sl2, bias.score_bias(0, head0 + r, pos, key) * log2e);
        } else {
#pragma unroll
            for (unsigned r = 0; r < G; ++r) s[r] = excluded;
        }
        float red_v[G];
#pragma unroll
        for (unsigned r = 0; r < G; ++r) red_v[r] = s[r];
        block_reduce<G, true>(red_v, red);
        float alpha[G];
#pragma unroll
        for (unsigned r = 0; r < G; ++r) {
            const float nm = fmaxf(m[r], red_v[r]);
            const float ms = nm == excluded ? 0.0f : nm;
            alpha[r] = exp2f(m[r] - ms);  // m = -inf -> 0
            m[r] = nm;
            const float p = valid ? exp2f(s[r] - ms) : 0.0f;
            sp[r][t] = p;
            red_v[r] = p;
        }
        block_reduce<G, false>(red_v, red);  // its barriers also publish sp
#pragma unroll
        for (unsigned r = 0; r < G; ++r) {
            l[r] = fmaf(l[r], alpha[r], red_v[r]);
            acc[r][0] *= alpha[r];
            acc[r][1] *= alpha[r];
        }
        // P V: thread = (dimension pair dp, key quarter); quarter q takes the pass keys q, q + 4, q + 8, ... ascending,
        // so a short pass still uses every thread and the loads of a quarter issue 16 at a time.
        const unsigned n = min(kThreads, k_end - base);
        if (quarter < n) {
            unsigned slot = wrapped ? (base + quarter) % slots : base + quarter;
#pragma unroll 16
            for (unsigned j = quarter; j < n; j += kQuarters) {
                const __nv_bfloat162 v2 = *reinterpret_cast<const __nv_bfloat162*>(
                    vh + static_cast<std::size_t>(slot) * row_stride + 2U * dp);
                const float2 vf = __bfloat1622float2(v2);
#pragma unroll
                for (unsigned r = 0; r < G; ++r) {
                    acc[r][0] = fmaf(sp[r][j], vf.x, acc[r][0]);
                    acc[r][1] = fmaf(sp[r][j], vf.y, acc[r][1]);
                }
                slot += kQuarters;
                if (wrapped && slot >= slots) slot -= slots;
            }
        }
        __syncthreads();  // sp is rewritten by the next pass
    }

    // quarters added in ascending order, then this chunk's record
    if (quarter > 0) {
#pragma unroll
        for (unsigned r = 0; r < G; ++r) {
            sacc[quarter - 1U][r][2U * dp] = acc[r][0];
            sacc[quarter - 1U][r][2U * dp + 1U] = acc[r][1];
        }
    }
    __syncthreads();
    float* rec = partials + (static_cast<std::size_t>(kvh) * capacity + split) * G * kRecord;
    if (quarter == 0) {
#pragma unroll
        for (unsigned r = 0; r < G; ++r) {
            float o0 = acc[r][0], o1 = acc[r][1];
#pragma unroll
            for (unsigned k = 0; k + 1U < kQuarters; ++k) { o0 += sacc[k][r][2U * dp]; o1 += sacc[k][r][2U * dp + 1U]; }
            *reinterpret_cast<float2*>(rec + r * kRecord + 2U + 2U * dp) = make_float2(o0, o1);
            if (dp == 0) { rec[r * kRecord] = m[r]; rec[r * kRecord + 1U] = l[r]; }
        }
    }
    __threadfence();
    __syncthreads();
    if (t == 0) last = atomicAdd(tickets + kvh, 1U) == plan.splits - 1U;
    __syncthreads();
    if (!last) return;
    __threadfence();

    // merge: the sink term first, then the chunks ascending
    const float* recs = partials + static_cast<std::size_t>(kvh) * capacity * G * kRecord;
    const bool sink = prepared.sinks != nullptr;
    for (unsigned i = t; i < G * kDim; i += kThreads) {
        const unsigned r = i / kDim, dd = i % kDim, head = head0 + r;
        const float sink2 = sink ? attn_core::AttentionSinkRows{prepared.sinks, d.heads}.sink(0, head) * log2e : excluded;
        float mx = sink2;
        for (unsigned c = 0; c < plan.splits; ++c) mx = fmaxf(mx, __ldcg(recs + (c * G + r) * kRecord));
        float den = sink ? exp2f(sink2 - mx) : 0.0f, num = 0.0f;
        for (unsigned c = 0; c < plan.splits; ++c) {
            const float* rr = recs + (c * G + r) * kRecord;
            const float w = exp2f(__ldcg(rr) - mx);
            den = fmaf(w, __ldcg(rr + 1), den);
            num = fmaf(w, __ldcg(rr + 2U + dd), num);
        }
        out[static_cast<std::size_t>(head) * kDim + dd] = num / den;
    }
    if (t == 0) tickets[kvh] = 0U;
}

template <unsigned G>
void launch(dim3 grid, cudaStream_t stream, const AttentionDims& d, const PreparedBf16Flash& prepared, const float* q,
            const __nv_bfloat16* k, const __nv_bfloat16* v, const std::uint32_t* position, float* out,
            std::uint32_t* tickets, float* partials, std::uint32_t capacity) {
    decode_split_kernel<G><<<grid, kThreads, 0, stream>>>(d, prepared, q, k, v, position, out, tickets, partials, capacity);
}

}  // namespace decode_split

std::size_t decode_split_workspace_bytes(const AttentionDims& d, std::uint32_t window) {
    if (d.kv_heads == 0 || d.heads == 0 || d.heads % d.kv_heads != 0 || d.max_positions == 0) return 0;
    const std::size_t capacity = attn_split::decode_split_capacity(d.max_positions, window);
    const std::size_t group = d.heads / d.kv_heads;
    return decode_split::ticket_bytes(d.kv_heads) +
           static_cast<std::size_t>(d.kv_heads) * capacity * group * decode_split::kRecord * sizeof(float);
}

cudaError_t decode_split_workspace_init(void* workspace, const AttentionDims& d, std::uint32_t window, cudaStream_t stream) {
    if (!workspace || decode_split_workspace_bytes(d, window) == 0) return cudaErrorInvalidValue;
    return cudaMemsetAsync(workspace, 0, decode_split::ticket_bytes(d.kv_heads), stream);
}

bool can_launch_prepared_decode_split(const AttentionDims& d, const PreparedBf16Flash& prepared, const float* q,
                                      const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                                      const std::uint32_t* device_position, float* out, const void* workspace,
                                      std::size_t workspace_bytes) {
    const auto aligned = [](const void* ptr) { return ptr && reinterpret_cast<std::uintptr_t>(ptr) % 16U == 0; };
    if (d.head_dim != decode_split::kDim || d.heads == 0 || d.kv_heads == 0 || d.heads % d.kv_heads != 0 ||
        d.heads / d.kv_heads > decode_split::kMaxGroup || !std::isfinite(d.scale) || d.max_positions == 0 ||
        !aligned(q) || !aligned(k_cache) || !aligned(v_cache) || !aligned(out) || !aligned(workspace) ||
        !device_position || q == out || k_cache == v_cache || prepared.bidirectional ||
        (prepared.relative_extent != 0 && !prepared.relative_bias)) return false;
    if (prepared.wrapped && (prepared.window == 0 || d.max_positions < prepared.window)) return false;
    const std::size_t need = decode_split_workspace_bytes(d, prepared.window);
    return need != 0 && workspace_bytes >= need;
}

cudaError_t attention_decode_prepared_split_bf16(const AttentionDims& d, const PreparedBf16Flash& prepared,
                                                 const float* q, const __nv_bfloat16* k_cache,
                                                 const __nv_bfloat16* v_cache, const std::uint32_t* device_position,
                                                 float* out, void* workspace, std::size_t workspace_bytes,
                                                 cudaStream_t stream) {
    if (!can_launch_prepared_decode_split(d, prepared, q, k_cache, v_cache, device_position, out, workspace,
                                          workspace_bytes)) return cudaErrorInvalidValue;
    const std::uint32_t capacity = attn_split::decode_split_capacity(d.max_positions, prepared.window);
    auto* tickets = static_cast<std::uint32_t*>(workspace);
    auto* partials = reinterpret_cast<float*>(static_cast<unsigned char*>(workspace) + decode_split::ticket_bytes(d.kv_heads));
    const dim3 grid(d.kv_heads, capacity);
    switch (d.heads / d.kv_heads) {
    case 1: decode_split::launch<1>(grid, stream, d, prepared, q, k_cache, v_cache, device_position, out, tickets, partials, capacity); break;
    case 2: decode_split::launch<2>(grid, stream, d, prepared, q, k_cache, v_cache, device_position, out, tickets, partials, capacity); break;
    case 3: decode_split::launch<3>(grid, stream, d, prepared, q, k_cache, v_cache, device_position, out, tickets, partials, capacity); break;
    default: decode_split::launch<4>(grid, stream, d, prepared, q, k_cache, v_cache, device_position, out, tickets, partials, capacity); break;
    }
    return cudaGetLastError();
}

}  // namespace strix::cuda_core
