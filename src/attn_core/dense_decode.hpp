#pragma once
// Shared dense decode attention (rows 1..16 per window, GQA any group, BF16 KV cache, FP32 math).
// Two kernels; every reduction has a fixed order independent of the row count, so row r of a window
// has the bits it has alone:
//   prep    per-head RMSNorm ((x*rsqrt(mean+eps)) * gain) + NeoX partial RoPE of q and k, K/V -> BF16 cache
//   attend  per (row, q head) pair: 8 waves take positions w, w+8, ... with an online softmax, the waves merge in
//           wave order; optional sigmoid output gate; causal (row r sees keys 0..pos0+r). One block walks the K/V
//           of a kv head once for P pairs (strix/attn_row_share.hpp plan); per pair the arithmetic is the P = 1 form's.
// Optional learned sink (Sink = true, the main schedule only): a per-head F32 logit with a zero value; it seeds the wave
// merge as one more partial (m = sink, l = 1, acc = 0), the host's online softmax seeded with the sink. Sink = false
// is the unchanged incumbent code.
// Layouts: q_in [row][head][q_stride] (q at +0, gate at +Dim when q_stride == 2*Dim);
//          k_in/v_in [row][kv_head][Dim]; cache [position][kv_head][Dim] bf16.
// Home: shared (attn_core): no model names; Dim is a template parameter (256 instantiated by qwen35).
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <hip/hip_runtime.h>
#include "core/wave_uniform.hpp"
#include "strix/attn_row_share.hpp"
#include "strix/env_flags.hpp"
#include "strix/numeric/declared.hpp"

namespace strix::attn_core::dense_decode {

inline constexpr unsigned max_rows = 16U;
inline constexpr unsigned waves = 8U;

struct Params {
    unsigned rows, heads, kv_heads, q_stride, rot_dim, pos0;
    float eps, scale;
    double theta;
    bool gated;
};

__device__ __forceinline__ std::uint16_t to_bf16(float x) {
    const auto u = __builtin_bit_cast(std::uint32_t, x);
    if ((u & 0x7fffffffU) > 0x7f800000U)
        return std::uint16_t((u >> 16U) | 0x40U);
    return std::uint16_t((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U);
}
__device__ __forceinline__ float from_bf16(std::uint32_t h) { return __builtin_bit_cast(float, h << 16U); }

// NeoX RoPE (cos, sin) of position pos, frequency index i < rot_dim / 2: the one expression of both prep forms.
__device__ __forceinline__ void rope_cs(const Params& p, unsigned pos, unsigned i, float& c, float& s) {
    const double ang = double(pos) * pow(p.theta, -2.0 * double(i) / double(p.rot_dim));
    c = float(cos(ang));
    s = float(sin(ang));
}
// rope[2 * (row * half + i) + {0, 1}] = rope_cs(pos0 + row, i) for the window's rows: evaluated once per (row, i) instead of
// once per (row, q/k head, lane) inside prep (FP64 pow/cos/sin; 56x fewer for 24 + 4 heads of Dim 256, rot_dim 64).
__global__ void __launch_bounds__(256) rope_table_kernel(Params p, float* rope) {
    const unsigned half = p.rot_dim / 2U, idx = blockIdx.x * 256U + threadIdx.x;
    if (idx >= p.rows * half)
        return;
    float c, s;
    rope_cs(p, p.pos0 + idx / half, idx % half, c, s);
    rope[2U * idx] = c;
    rope[2U * idx + 1U] = s;
}

// grid (rows, heads + 2 * kv_heads): [0,heads) q, [heads, heads+kv) k, then v. block = Dim threads.
// Table = true: (cos, sin) from rope (rope_table_kernel of the same window), the same floats as the inline form.
template <unsigned Dim, bool Table = false>
__global__ void __launch_bounds__(Dim) prep_kernel(Params p, const float* q_in, const float* k_in, const float* v_in,
                                                   const float* q_gain, const float* k_gain, float* q_out,
                                                   std::uint16_t* k_cache, std::uint16_t* v_cache,
                                                   const float* rope = nullptr) {
    static_assert(Dim % 64U == 0U);
    __shared__ float red[Dim];
    __shared__ float vec[Dim];
    const unsigned row = blockIdx.x, slot = blockIdx.y, t = threadIdx.x;
    const unsigned pos = p.pos0 + row;
    if (slot >= p.heads + p.kv_heads) {
        const unsigned kh = slot - p.heads - p.kv_heads;
        v_cache[(std::size_t(pos) * p.kv_heads + kh) * Dim + t] = to_bf16(v_in[(std::size_t(row) * p.kv_heads + kh) * Dim + t]);
        return;
    }
    const bool is_q = slot < p.heads;
    const float x = is_q ? q_in[(std::size_t(row) * p.heads + slot) * p.q_stride + t]
                         : k_in[(std::size_t(row) * p.kv_heads + (slot - p.heads)) * Dim + t];
    red[t] = x * x;
    __syncthreads();
    for (unsigned s = Dim / 2U; s > 0U; s >>= 1U) {
        if (t < s)
            red[t] += red[t + s];
        __syncthreads();
    }
    const float r = 1.0F / sqrtf(red[0] / float(Dim) + p.eps);
    vec[t] = (x * r) * (is_q ? q_gain : k_gain)[t];
    __syncthreads();
    float y = vec[t];
    const unsigned half = p.rot_dim / 2U;
    if (t < p.rot_dim) {
        const unsigned i = t % half;
        float c, s;
        if constexpr (Table) {
            c = rope[2U * (row * half + i)];
            s = rope[2U * (row * half + i) + 1U];
        } else {
            rope_cs(p, pos, i, c, s);
        }
        y = t < half ? vec[t] * c - vec[t + half] * s : vec[t] * c + vec[t - half] * s;
    }
    if (is_q)
        q_out[(std::size_t(row) * p.heads + slot) * Dim + t] = y;
    else
        k_cache[(std::size_t(pos) * p.kv_heads + (slot - p.heads)) * Dim + t] = to_bf16(y);
}

// EXACT arithmetic of every attend form: main's source order with no contraction. The engine builds this header
// with -ffp-contract=off; `#pragma clang fp contract(off)` gives the same instructions under the HIP default
// (contraction that honours pragmas), so the bits do not depend on that flag. -ffp-contract=fast ignores pragmas: never build an EXACT
// includer with it. The only fused operation is the dot's explicit fmaf (in each kernel's walk).
// Per position, after the dot (one copy, used by both kernels; the loop over a lane's elements stays in the kernel,
// which keeps main's kernel instruction-identical to main): s = rnd(s * scale); the max and the exponent take that s;
// l = rnd(l * corr) + w; acc = rnd(acc * corr) + rnd(w * v). The wave merge is main's text in both kernels, each kernel
// body under the pragma (tests/dense_decode_exact_form_tests.py keeps the two copies byte-identical).
struct Step {
    float mn, corr, w;
};
// The exponential of the softmax (online weights, wave merge, sink term). ExpfMath is the library expf (every
// incumbent binding: the default of every kernel below); DeclaredExpMath is strix::numeric::declared_exp, the one
// definition host and device share, so a host reference that calls declared_exp reproduces the device bits.
struct ExpfMath {
    __device__ __forceinline__ static float exp(float x) { return expf(x); }
};
struct DeclaredExpMath {
    __device__ __forceinline__ static float exp(float x) {
#pragma clang fp contract(off)
        return numeric::declared_exp(x);
    }
};
template <class Math = ExpfMath>
__device__ __forceinline__ Step online_step(float s, float scale, float m, float& l, Math = Math()) {
#pragma clang fp contract(off)
    s *= scale;
    const float mn = fmaxf(m, s), corr = Math::exp(m - mn), w = Math::exp(s - mn);
    l = l * corr + w;
    return Step{mn, corr, w};
}
__device__ __forceinline__ float acc_step(float acc, float corr, float w, float v) {
#pragma clang fp contract(off)
    return acc * corr + w * v;
}

// The main schedule (main's kernel; per-position arithmetic in online_step / acc_step): grid (rows, heads), block =
// 32 * waves threads, one block walks the K/V prefix for one (row, q head). It is the off switch, the form for devices
// and heads shapes without a measured table and the table's (1, 1) choice. ctx [row][head][Dim].
template <unsigned Dim, bool Sink = false, class Math = ExpfMath>
__global__ void __launch_bounds__(32U * waves) attend_main_kernel(Params p, const float* q_n, const float* q_in,
                                                             const std::uint16_t* k_cache, const std::uint16_t* v_cache,
                                                             float* ctx, const float* sinks = nullptr) {
#pragma clang fp contract(off)  // the merge below is main's text; see online_step
    constexpr unsigned per_lane = Dim / 32U;
    static_assert(per_lane % 8U == 0U);
    __shared__ float m_s[waves], l_s[waves];
    __shared__ float acc_s[waves][Dim];
    const unsigned row = blockIdx.x, head = blockIdx.y, wave = threadIdx.x / 32U, lane = threadIdx.x % 32U;
    const unsigned group = p.heads / p.kv_heads, kh = head / group;
    const unsigned last = p.pos0 + row;
    float q[per_lane];
    for (unsigned e = 0; e < per_lane; ++e)
        q[e] = q_n[(std::size_t(row) * p.heads + head) * Dim + lane * per_lane + e];
    float m = -INFINITY, l = 0.0F, acc[per_lane] = {};
    for (unsigned pos = wave; pos <= last; pos += waves) {
        const std::size_t at = (std::size_t(pos) * p.kv_heads + kh) * Dim + lane * per_lane;
        float s = 0.0F;
        for (unsigned e = 0; e < per_lane; ++e)
            s = __builtin_fmaf(q[e], from_bf16(k_cache[at + e]), s);
#pragma unroll
        for (unsigned o = 16U; o > 0U; o >>= 1U)
            s += __shfl_xor(s, int(o), 32);
        const Step st = online_step(s, p.scale, m, l, Math{});
        for (unsigned e = 0; e < per_lane; ++e)
            acc[e] = acc_step(acc[e], st.corr, st.w, from_bf16(v_cache[at + e]));
        m = st.mn;
    }
    for (unsigned e = 0; e < per_lane; ++e)
        acc_s[wave][lane * per_lane + e] = acc[e];
    if (lane == 0) {
        m_s[wave] = m;
        l_s[wave] = l;
    }
    __syncthreads();
    for (unsigned d = threadIdx.x; d < Dim; d += 32U * waves) {
        float M = -INFINITY;
        for (unsigned w = 0; w < waves; ++w)
            M = fmaxf(M, m_s[w]);
        if constexpr (Sink)
            M = fmaxf(M, sinks[head]);
        float L = 0.0F, O = 0.0F;
        if constexpr (Sink)
            L = Math::exp(sinks[head] - M);
        for (unsigned w = 0; w < waves; ++w) {
            const float f = m_s[w] == -INFINITY ? 0.0F : Math::exp(m_s[w] - M);
            L += l_s[w] * f;
            O += acc_s[w][d] * f;
        }
        float o = O / L;
        if (p.gated) {
            const float g = q_in[(std::size_t(row) * p.heads + head) * p.q_stride + Dim + d];
            o *= 1.0F / (1.0F + Math::exp(-g));
        }
        ctx[(std::size_t(row) * p.heads + head) * Dim + d] = o;
    }
}

// grid (blocks per kv head, kv_heads), block = 32 * waves threads; block b of kv head kh takes pairs j = b*P .. b*P+P-1
// (attn_row_share pair order: row = j / group, q head = kh * group + j % group). Each K/V position is loaded once per
// wave and applied to every pair whose row sees it; per pair: the same dot / butterfly / online-softmax sequence and
// the same wave-order merge as P = 1, so the bits do not depend on P. ctx [row][head][Dim].
template <unsigned Dim, unsigned P = 1U, unsigned U = 1U, bool Sink = false, class Math = ExpfMath>
__global__ void __launch_bounds__(32U * waves) attend_kernel(Params p, const float* q_n, const float* q_in,
                                                             const std::uint16_t* k_cache, const std::uint16_t* v_cache,
                                                             float* ctx, const float* sinks = nullptr) {
#pragma clang fp contract(off)  // the merge below is main's text; see online_step
    constexpr unsigned per_lane = Dim / 32U;
    static_assert(per_lane % 8U == 0U && P >= 1U && U >= 1U);
    __shared__ float m_s[waves], l_s[waves];
    __shared__ float acc_s[waves][Dim];
    // wave in an SGPR: the walk's position, its bounds and the per-pair causal test are then scalar (SALU compare and
    // branch, no exec-mask save / restore), and the butterfly's lane addresses stay hoisted out of the loop.
    const unsigned wave = core::wave_index(threadIdx.x), lane = threadIdx.x % 32U;
    const unsigned group = p.heads / p.kv_heads, kh = blockIdx.y, j0 = blockIdx.x * P, total = p.rows * group;
    const unsigned n = total - j0 < P ? total - j0 : P;  // valid pairs of this block, >= 1
    const unsigned top = p.pos0 + attn_row_share::pair_row(j0 + n - 1U, group);
    float q[P][per_lane], m[P], l[P], acc[P][per_lane];
    unsigned last[P];
#pragma unroll
    for (unsigned i = 0; i < P; ++i) {
        const unsigned j = j0 + (i < n ? i : 0U), row = attn_row_share::pair_row(j, group);
        const unsigned head = attn_row_share::pair_head(j, kh, group);
        last[i] = p.pos0 + row;
        for (unsigned e = 0; e < per_lane; ++e)
            q[i][e] = q_n[(std::size_t(row) * p.heads + head) * Dim + lane * per_lane + e];
        m[i] = -INFINITY;
        l[i] = 0.0F;
        for (unsigned e = 0; e < per_lane; ++e)
            acc[i][e] = 0.0F;
    }
    // U positions per iteration (pos, pos + 8, ...): all their K/V loads are issued first, then the positions are
    // applied in ascending order, so a wave keeps U load pairs in flight (a 1-row walk is latency-bound) and every
    // pair still sees its positions in the one-at-a-time order.
    for (unsigned base = wave; base <= top; base += waves * U) {
        std::uint16_t kr[U][per_lane], vr[U][per_lane];
#pragma unroll
        for (unsigned u = 0; u < U; ++u) {
            if (base + u * waves > top)
                continue;
            const std::size_t at = (std::size_t(base + u * waves) * p.kv_heads + kh) * Dim + lane * per_lane;
            for (unsigned e = 0; e < per_lane; ++e) {
                kr[u][e] = k_cache[at + e];
                vr[u][e] = v_cache[at + e];
            }
        }
#pragma unroll
        for (unsigned u = 0; u < U; ++u) {
            const unsigned pos = base + u * waves;
            if (pos > top)
                continue;
            float kf[per_lane];
            for (unsigned e = 0; e < per_lane; ++e)
                kf[e] = from_bf16(kr[u][e]);
#pragma unroll
            for (unsigned i = 0; i < P; ++i) {
                if (i >= n || pos > last[i])  // wave-uniform: the causal tail of the earlier rows of the block
                    continue;
                float s = 0.0F;
                for (unsigned e = 0; e < per_lane; ++e)
                    s = __builtin_fmaf(q[i][e], kf[e], s);
#pragma unroll
                for (unsigned o = 16U; o > 0U; o >>= 1U)
                    s += __shfl_xor(s, int(o), 32);
                const Step st = online_step(s, p.scale, m[i], l[i], Math{});
                for (unsigned e = 0; e < per_lane; ++e)
                    acc[i][e] = acc_step(acc[i][e], st.corr, st.w, from_bf16(vr[u][e]));
                m[i] = st.mn;
            }
        }
    }
#pragma unroll
    for (unsigned i = 0; i < P; ++i) {
        if (i >= n)  // block-uniform; no break, so the loop stays fully unrolled (acc[i] in registers)
            continue;
        const unsigned j = j0 + i, row = attn_row_share::pair_row(j, group), head = attn_row_share::pair_head(j, kh, group);
        if (i > 0U)
            __syncthreads();  // the previous pair's merge has read acc_s / m_s / l_s
        for (unsigned e = 0; e < per_lane; ++e)
            acc_s[wave][lane * per_lane + e] = acc[i][e];
        if (lane == 0) {
            m_s[wave] = m[i];
            l_s[wave] = l[i];
        }
        __syncthreads();
        for (unsigned d = threadIdx.x; d < Dim; d += 32U * waves) {
            float M = -INFINITY;
            for (unsigned w = 0; w < waves; ++w)
                M = fmaxf(M, m_s[w]);
            if constexpr (Sink)
                M = fmaxf(M, sinks[head]);
            float L = 0.0F, O = 0.0F;
            if constexpr (Sink)
                L = Math::exp(sinks[head] - M);
            for (unsigned w = 0; w < waves; ++w) {
                const float f = m_s[w] == -INFINITY ? 0.0F : Math::exp(m_s[w] - M);
                L += l_s[w] * f;
                O += acc_s[w][d] * f;
            }
            float o = O / L;
            if (p.gated) {
                const float g = q_in[(std::size_t(row) * p.heads + head) * p.q_stride + Dim + d];
                o *= 1.0F / (1.0F + Math::exp(-g));
            }
            ctx[(std::size_t(row) * p.heads + head) * Dim + d] = o;
        }
    }
}

// The measured row-share table of the current device for this head shape (strix/attn_row_share.hpp
// dense_decode_tables, keyed by gcnArchName without features + multiProcessorCount, as include/strix/device/hip_enumerator.hpp
// reads it), or nullptr. Device properties are read once per device.
inline const attn_row_share::DeviceTable* device_table(unsigned group, unsigned kv_heads, unsigned dim) {
    struct Key {
        std::string arch;
        unsigned multiprocessors = 0U;
    };
    static std::mutex mu;
    static std::map<int, Key> keys;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess)
        return nullptr;
    std::lock_guard<std::mutex> lock(mu);
    const auto [it, fresh] = keys.try_emplace(dev);
    if (fresh) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) == hipSuccess) {
            it->second.arch = prop.gcnArchName;
            it->second.arch = it->second.arch.substr(0, it->second.arch.find(':'));
            it->second.multiprocessors = unsigned(prop.multiProcessorCount);
        }
    }
    return attn_row_share::find_table(it->second.arch, it->second.multiprocessors, group, kv_heads, dim);
}

// attend of a window whose prep has run: the device's measured plan (pairs per block and positions per walk step by
// (walk length band, row count); rows past the table read its last row); no table for the device or head shape, or
// STRIX_DECODE_ATTN_ROW_SHARE=0 launch the main schedule (attend_main_kernel). Same bits in every case. P = 6 (142 VGPR)
// is never the fastest for <= 16 rows and P = 12 spills (253 VGPR + 400 B scratch), so neither is instantiated.
template <unsigned Dim>
inline void attend(const Params& p, const float* q_scratch, const float* q_in, const std::uint16_t* k_cache,
                   const std::uint16_t* v_cache, float* ctx, hipStream_t stream) {
    static const bool share = envcfg::flag("STRIX_DECODE_ATTN_ROW_SHARE", true);
    namespace rs = attn_row_share;
    const unsigned group = p.kv_heads == 0U ? 0U : p.heads / p.kv_heads;
    const rs::DeviceTable* table = share ? device_table(group, p.kv_heads, Dim) : nullptr;
    // Walk length = positions the window's last row sees; the table's band for it (the best form flips with length).
    const auto plan = rs::plan_device(p.rows, group, p.pos0 + p.rows, table);
    const dim3 grid(plan.blocks, p.kv_heads), block(32U * waves);
    const unsigned form = plan.pairs * 10U + plan.walk;
    if (form == 12U)
        attend_kernel<Dim, 1U, 2U><<<grid, block, 0, stream>>>(p, q_scratch, q_in, k_cache, v_cache, ctx);
    else if (form == 22U)
        attend_kernel<Dim, 2U, 2U><<<grid, block, 0, stream>>>(p, q_scratch, q_in, k_cache, v_cache, ctx);
    else if (form == 21U)
        attend_kernel<Dim, 2U, 1U><<<grid, block, 0, stream>>>(p, q_scratch, q_in, k_cache, v_cache, ctx);
    else if (form == 31U)
        attend_kernel<Dim, 3U, 1U><<<grid, block, 0, stream>>>(p, q_scratch, q_in, k_cache, v_cache, ctx);
    else  // (1, 1): the main schedule
        attend_main_kernel<Dim><<<dim3(p.rows, p.heads), block, 0, stream>>>(p, q_scratch, q_in, k_cache, v_cache, ctx);
}

// attend with the learned sink (sinks [heads] F32 logits): the main schedule, one block per (row, head), any group / kv
// head split. A separate function, so the sink kernel is instantiated only where it is called.
template <unsigned Dim, class Math = ExpfMath>
inline void attend_sink(const Params& p, const float* q_scratch, const float* q_in, const std::uint16_t* k_cache,
                        const std::uint16_t* v_cache, float* ctx, const float* sinks, hipStream_t stream) {
    attend_main_kernel<Dim, true, Math><<<dim3(p.rows, p.heads), dim3(32U * waves), 0, stream>>>(p, q_scratch, q_in, k_cache,
                                                                                         v_cache, ctx, sinks);
}

template <unsigned Dim>
inline void launch(const Params& p, const float* q_in, const float* k_in, const float* v_in, const float* q_gain,
                   const float* k_gain, float* q_scratch, std::uint16_t* k_cache, std::uint16_t* v_cache, float* ctx,
                   hipStream_t stream) {
    prep_kernel<Dim><<<dim3(p.rows, p.heads + 2U * p.kv_heads), Dim, 0, stream>>>(p, q_in, k_in, v_in, q_gain, k_gain,
                                                                                  q_scratch, k_cache, v_cache);
    attend<Dim>(p, q_scratch, q_in, k_cache, v_cache, ctx, stream);
}

} // namespace strix::attn_core::dense_decode
