#pragma once

// Shared HIP hyper-connection (HC) core: the device bodies of include/strix/hc/hc_contract.hpp, one kernel family
// for every model with a multi-stream residual. The variant is the contract's Spec, a template parameter:
//   streams          number of residual copies (N);
//   site             sublayer (pre, post [, comb]) or head (pre only);
//   comb             sinkhorn (N x N comb rows, Sinkhorn on the lanes of one wave; N == 4) or identity (no stage);
//   fn               F16 or F32 mix weights;
//   store            FP32 or the bfloat16 rounding of every collapse / expand output;
//   post_magnitude, post_eps   compile-time constants of the post coefficient (post_eps 0: no add, as DS41).
// hidden, the Sinkhorn iterations, the pre/comb eps (hc_eps) and the RMS eps (norm_eps) are runtime arguments, so
// Spec::hidden / sinkhorn_iters / pre_eps / comb_eps / norm_eps are not read here.
//
// The arithmetic is the contract's declared order (chunked lane dots of 640, butterflies 16..1, scale after the dot,
// ascending sums from 0). Callers compile with -ffp-contract=off -fno-fast-math -fhip-fp32-correctly-rounded-divide-sqrt.
// Bindings: DeepSeek V4.1 (src/hip_ds41_layer.hip: F16 fn, sinkhorn, FP32 store; its fused decode site schedules call the
// bodies below), a Spec with F32 fn, identity comb and bf16 store (tests/hip_hc_core_tests.hip).
// Mix rows are written to a row type M with arrays pre[N] (+ post[N], + comb[N * N] when the Spec has them), e.g.
// MixRow<S> below or a layout-equal model type (::strix::ds41::HcMix).

#include "strix/ds41_layer.hpp"  // declared arithmetic (declared_sigmoid / rsqrt / exp, f16_to_float, round_to_bf16)
#include "strix/hc/hc_contract.hpp"

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#define STRIX_HC_HD __host__ __device__
#else
#define STRIX_HC_HD
#endif

#include <cstddef>
#include <cstdint>
#include <type_traits>
#if !defined(__HIPCC__) && !defined(__GNUC__)
#include <cmath>
#endif

namespace strix::core::hc {

using strix::hc::Comb;
using strix::hc::FnType;
using strix::hc::Spec;
using strix::hc::Store;

// Packed four-stream HC. A projected input is already in the model's R6
// arithmetic domain. A raw input has one trailing sum of squares per row.
enum class PackedInput { projected, raw_with_squares };
template <class Math, PackedInput Input, unsigned Iters>
struct PackedForm {
    using Arithmetic = Math;
    static constexpr auto input = Input;
    static constexpr unsigned iterations = Iters;
    static constexpr unsigned streams = 4;
    static constexpr unsigned mixes = 24;
};

// The declared scalar arithmetic of a packed form (basic IEEE operations only, no transcendental intrinsics):
// exp = declared_exp, sigmoid = 1 / (1 + declared_exp(-x)), rstd = 1 / correctly-rounded-sqrt(ss / n + eps).
// Every packed-form binding that declares this arithmetic names this struct instead of keeping a model-named copy.
struct DeclaredArithmetic {
    STRIX_HC_HD static float exp(float x) { return ::strix::ds41::declared_exp(x); }
    STRIX_HC_HD static float sigmoid(float x) { return 1.0F / (1.0F + ::strix::ds41::declared_exp(-x)); }
    STRIX_HC_HD static float rstd(float sum_squares, unsigned n, float eps) {
        return 1.0F / ::strix::ds41::declared_sqrt(sum_squares / static_cast<float>(n) + eps);
    }
};

STRIX_HC_HD inline float packed_sum(float a, float b, float c, float d) {
    return ((a + b) + c) + d;
}

STRIX_HC_HD inline float packed_dot(const float* p, const float* x, unsigned i, unsigned hidden, unsigned stride = 1) {
    return ((p[0] * x[i] + p[stride] * x[hidden + i]) + p[2 * stride] * x[2 * hidden + i]) +
           p[3 * stride] * x[3 * hidden + i];
}

STRIX_HC_HD inline float packed_expand(const float* m, const float* residual, float sub, unsigned k, unsigned i, unsigned hidden) {
    return m[4 + k] * sub + packed_dot(m + 8 + k, residual, i, hidden, 4);
}

template <bool Squares, class Decode>
STRIX_HC_HD inline float packed_lane_dot(const float* x, const std::uint16_t* fn, unsigned first, unsigned end,
                                       unsigned lane, Decode decode) {
    float acc = 0.0F;
    for (unsigned i = first + lane; i < end; i += 32U) {
        if constexpr (Squares) acc = acc + x[i] * x[i];
        else acc = acc + decode(fn[i]) * x[i];
    }
    return acc;
}

STRIX_HC_HD inline float packed_chunk_total(const float* chunks, unsigned count) {
    float total = chunks[0];
    for (unsigned i = 1; i < count; ++i) total = total + chunks[i];
    return total;
}

template <class Form>
STRIX_HC_HD inline void packed_softmax_exp(const float* row, float* e) {
    using Math = typename Form::Arithmetic;
    float mx = row[0];
    for (unsigned k = 1; k < 4; ++k) mx = row[k] > mx ? row[k] : mx;
#if defined(__HIPCC__)
#pragma unroll
#endif
    for (unsigned k = 0; k < 4; ++k) e[k] = Math::exp(row[k] - mx);
}

template <class Form>
STRIX_HC_HD inline void mix_final_packed(const float* in, const float* scale, const float* base, float eps, float* out,
                                        unsigned width = 0, float norm_eps = 0.0F) {
    using Math = typename Form::Arithmetic;
    constexpr unsigned h = Form::streams;
    float normalized[Form::mixes];
    if constexpr (Form::input == PackedInput::raw_with_squares) {
        const float rr = Math::rstd(in[Form::mixes], width, norm_eps);
        for (unsigned j = 0; j < Form::mixes; ++j) normalized[j] = in[j] * rr;
        in = normalized;
    }
    for (unsigned j = 0; j < h; ++j) out[j] = Math::sigmoid(in[j] * scale[0] + base[j]) + eps;
    for (unsigned j = h; j < 2 * h; ++j) out[j] = 2.0F * Math::sigmoid(in[j] * scale[1] + base[j]);
    float local_comb[h * h];
    float* c = out + 2 * h;
    if constexpr (Form::input == PackedInput::raw_with_squares) c = local_comb;
    for (unsigned j = 0; j < h * h; ++j) c[j] = in[2 * h + j] * scale[2] + base[2 * h + j];
    for (unsigned j = 0; j < h; ++j) {
        float* row = c + j * h;
        float e[h];
        packed_softmax_exp<Form>(row, e);
        const float sum = packed_sum(e[0], e[1], e[2], e[3]);
        for (unsigned k = 0; k < h; ++k) row[k] = e[k] / sum + eps;
    }
    for (unsigned it = 0; it < Form::iterations; ++it) {
        auto row_sum = [&](unsigned j) {
            const float* row = c + j * h;
            return packed_sum(row[0], row[1], row[2], row[3]) + eps;
        };
        auto col_sum = [&](unsigned k) { return packed_sum(c[k], c[h + k], c[2 * h + k], c[3 * h + k]) + eps; };
        if constexpr (Form::input == PackedInput::projected) {
            if (it != 0U) {
                float sr[h];
                for (unsigned j = 0; j < h; ++j) sr[j] = row_sum(j);
                for (unsigned j = 0; j < h; ++j)
                    for (unsigned k = 0; k < h; ++k) c[j * h + k] = c[j * h + k] / sr[j];
            }
            float sc[h];
            for (unsigned k = 0; k < h; ++k) sc[k] = col_sum(k);
            for (unsigned j = 0; j < h; ++j)
                for (unsigned k = 0; k < h; ++k) c[j * h + k] = c[j * h + k] / sc[k];
        } else {
            if (it != 0U)
                for (unsigned j = 0; j < h; ++j) {
                    const float sum = row_sum(j);
                    for (unsigned k = 0; k < h; ++k) c[j * h + k] = c[j * h + k] / sum;
                }
            for (unsigned k = 0; k < h; ++k) {
                const float sum = col_sum(k);
                for (unsigned j = 0; j < h; ++j) c[j * h + k] = c[j * h + k] / sum;
            }
        }
    }
    if constexpr (Form::input == PackedInput::raw_with_squares)
        for (unsigned i = 0; i < h * h; ++i) out[2 * h + i] = c[i];
}

// A folded peer buffer already contains the rank-ordered sum.
STRIX_HC_HD inline float consume_rank_sum(const float* local, const float* peer, unsigned rank, std::size_t i) {
    if (local == nullptr) return peer[i];
    const float a = local[i], b = peer[i];
    return rank == 0 ? a + b : b + a;
}

template <class Wait>
#if defined(__HIPCC__)
__device__ __forceinline__
#else
inline
#endif
const float* consume_peer(const float* peer, const void* payload, Wait wait) {
    if (peer != nullptr) return peer;
    wait();
    return static_cast<const float*>(payload);
}

// Normalized projection: a producer of the projected packed input (PackedInput::projected). Unlike the raw form, the
// stream row is normalized BEFORE the dot: f_i = x_i * rr (one rounding per element), then every mix is a blocked dot
// of f. Squares: Lanes lane-strided chains (lane l: i = l, l + Lanes, ... ascending, fmaf from +0), pairwise tree
// s_i = s_i + s_{i+h} for h = Lanes / 2 .. 1, rr = Math::rstd(ss, columns, eps). Dot: Blocks contiguous blocks of Block
// inputs, block j: fmaf ascending from +0, then the pairwise tree p_i = p_i + p_{i+h} for h = Blocks / 2 .. 1.
// Every helper is one stage of that order; schedules (block per j, last arrival, distributed tree) keep the bits.
STRIX_HC_HD inline float projection_fma(float a, float b, float c) {
#if defined(__HIPCC__) || defined(__GNUC__)
    return __builtin_fmaf(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

template <unsigned Lanes, unsigned Block, unsigned Blocks>
struct NormalizedProjection {
    static constexpr unsigned lanes = Lanes, block = Block, blocks = Blocks, columns = Block * Blocks;
    static_assert((Lanes & (Lanes - 1)) == 0 && (Blocks & (Blocks - 1)) == 0 && columns % Lanes == 0, "power-of-two trees");

    // Chain of lane `lane` of the sum of squares.
    STRIX_HC_HD static inline float lane_squares(const float* x, unsigned lane) {
        float a = 0.0F;
        for (unsigned i = lane; i < columns; i += Lanes) a = projection_fma(x[i], x[i], a);
        return a;
    }
    // The same chain with every load issued before the first fmaf.
    STRIX_HC_HD static inline float lane_squares_staged(const float* x, unsigned lane) {
        const float* xl = x + lane;
        float v[columns / Lanes];
#if defined(__HIPCC__)
#pragma unroll
#endif
        for (unsigned k = 0; k < columns / Lanes; ++k) v[k] = xl[k * Lanes];
        float a = 0.0F;
#if defined(__HIPCC__)
#pragma unroll
#endif
        for (unsigned k = 0; k < columns / Lanes; ++k) a = projection_fma(v[k], v[k], a);
        return a;
    }
    // Tree levels h = Count / 2 .. 1 for schedules that run one level at a time.
    template <unsigned Count, class Level>
    STRIX_HC_HD static inline void levels(Level level) {
        for (unsigned h = Count / 2; h > 0; h >>= 1U) level(h);
    }
    template <unsigned Count>
    STRIX_HC_HD static inline float tree(float* p) {
        for (unsigned h = Count / 2; h >= 1; h /= 2)
            for (unsigned i = 0; i < h; ++i) p[i] = p[i] + p[i + h];
        return p[0];
    }
    template <class Math>
    STRIX_HC_HD static inline float rstd(float ss, float eps) {
        return Math::rstd(ss, columns, eps);
    }
    STRIX_HC_HD static inline float normalized(float x, float rr) { return x * rr; }
    // Chain of one block: f and w point at the block's first input.
    STRIX_HC_HD static inline float block_dot(const float* f, const float* w) {
        float acc = 0.0F;
        for (unsigned i = 0; i < Block; ++i) acc = projection_fma(f[i], w[i], acc);
        return acc;
    }
    // Scalar order of one row: f[columns] (normalized row) and z[mixes] (projected input), fn row-major [mixes][columns].
    template <class Math>
    STRIX_HC_HD static inline void project_row(const float* x, const float* fn, unsigned mixes, float eps, float* f, float* z) {
        float s[Lanes];
        for (unsigned l = 0; l < Lanes; ++l) s[l] = lane_squares(x, l);
        const float rr = rstd<Math>(tree<Lanes>(s), eps);
        for (unsigned i = 0; i < columns; ++i) f[i] = normalized(x[i], rr);
        for (unsigned o = 0; o < mixes; ++o) {
            float p[Blocks];
            for (unsigned j = 0; j < Blocks; ++j) p[j] = block_dot(f + j * Block, fn + static_cast<std::size_t>(o) * columns + j * Block);
            z[o] = tree<Blocks>(p);
        }
    }
#if defined(__HIPCC__)
    // Squares tree over the Lanes threads of a block (s holds Lanes floats; one barrier per level); every lane returns ss.
    __device__ __forceinline__ static float lds_tree(float* s, float part) {
        const unsigned l = threadIdx.x;
        s[l] = part;
        __syncthreads();
        for (unsigned h = Lanes / 2; h > 0; h >>= 1U) {
            if (l < h) s[l] = s[l] + s[l + h];
            __syncthreads();
        }
        const float r = s[0];
        __syncthreads();
        return r;
    }
    // Block tree of one chain on 8 lanes: lane a owns partials a + 8 k; local levels are the tree's offsets 32, 16, 8
    // (same left operands), the shuffles its offsets 4, 2, 1. The total is valid on lane 0. Coherent: the partials were
    // stored by other work groups of the same launch, read at agent scope.
    template <bool Coherent>
    __device__ __forceinline__ static float lanes8_tree(const float* src, unsigned lane) {
        static_assert(Blocks == 64, "8 lanes x 8 partials");
        float p[8];
#pragma unroll
        for (unsigned k = 0; k < 8; ++k)
            p[k] = Coherent ? __hip_atomic_load(&src[lane + 8 * k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) : src[lane + 8 * k];
#pragma unroll
        for (unsigned h = 4; h > 0; h >>= 1U) {
#pragma unroll
            for (unsigned k = 0; k < h; ++k) p[k] = p[k] + p[k + h];
        }
        float acc = p[0];
        for (unsigned h = 4; h > 0; h >>= 1U) {
            const float peer = __shfl_down(acc, h, 8);
            if (lane < h) acc = acc + peer;
        }
        return acc;
    }
#endif
};

#if defined(__HIPCC__)
inline constexpr unsigned wave = 32;
inline constexpr unsigned chunk = static_cast<unsigned>(::strix::ds41::dot_chunk);  // elements per wave chunk of the long dots
inline constexpr unsigned chunk_k = chunk / wave;                          // values per lane in a chunk
inline constexpr unsigned mix_tile = 8;                                    // rows per mix-partial block

template <Spec S> inline constexpr unsigned mix_rows = S.mix_rows();
template <Spec S> inline constexpr unsigned mix_waves = S.mix_rows() + 1U;  // one wave per mix row + the squares wave
template <Spec S> using FnElem = std::conditional_t<S.fn == FnType::f16, std::uint16_t, float>;

// The mix row of a Spec (pre, then post, then comb, FP32), for binders without a model type of their own.
template <Spec S, bool Post = (S.post_rows() != 0U), bool HasComb = (S.comb_rows() != 0U)> struct MixRow;
template <Spec S> struct MixRow<S, false, false> {
    float pre[S.streams];
};
template <Spec S> struct MixRow<S, true, false> {
    float pre[S.streams];
    float post[S.streams];
};
template <Spec S> struct MixRow<S, true, true> {
    float pre[S.streams];
    float post[S.streams];
    float comb[S.streams * S.streams];
};

// The sublayer input an expansion reads: p[i], or, with rank partials, the canonical rank sum (p[i] + low[i]) + high[i]
// read in place (FP32 adds only; low and high both set or both null).
// A site dispatched with both terms always present can share the reduction without testing low on each load.
template <bool TermsPresent> struct SubSumView {
    const float* __restrict__ p;
    const float* __restrict__ low;
    const float* __restrict__ high;
    __device__ __forceinline__ float operator[](std::size_t i) const {
        const float s = p[i];
        if constexpr (TermsPresent) return (s + low[i]) + high[i];
        return low ? (s + low[i]) + high[i] : s;
    }
    __device__ __forceinline__ explicit operator bool() const { return p != nullptr; }
};

using SubSum = SubSumView<false>;
using CanonicalSubSum = SubSumView<true>;

__device__ __forceinline__ float butterfly(float v) {
    for (int d = 16; d; d >>= 1) v = v + __shfl_xor(v, d, wave);
    return v;
}
template <Spec S> __device__ __forceinline__ float fn_value(const FnElem<S>* __restrict__ fn, std::size_t i) {
    if constexpr (S.fn == FnType::f16) return ::strix::ds41::f16_to_float(fn[i]);
    else return fn[i];
}
template <Spec S> __device__ __forceinline__ float store_value(float v) {
    if constexpr (S.store == Store::bf16) return ::strix::ds41::round_to_bf16(v);
    else return v;
}
template <Spec S> __device__ __forceinline__ float post_value(float sigmoid) {
    const float v = S.post_magnitude * sigmoid;
    if constexpr (S.post_eps != 0.0F) return v + S.post_eps;
    else return v;
}

// Copy k of element i of row r after the expansion: store(post[k] * sub[i] + s), s = residual[k][i] (identity) or
// 0 + sum_j comb[j][k] * residual[j][i] in ascending j (sinkhorn).
template <Spec S, class M, class Sub>
__device__ __forceinline__ float copy_value(Sub sub, const float* __restrict__ residual, const M* __restrict__ em, unsigned r,
                                            unsigned k, unsigned i, unsigned hidden) {
    static_assert(S.post_rows() != 0U, "expansion needs a sublayer site");
    const float* res = residual + static_cast<std::size_t>(r) * S.streams * hidden;
    const M& m = em[r];
    if constexpr (S.comb == Comb::sinkhorn) {
        float s = 0.0F;
#pragma unroll
        for (unsigned j = 0; j < S.streams; ++j) s = s + m.comb[j * S.streams + k] * res[j * hidden + i];
        return store_value<S>(m.post[k] * sub[static_cast<std::size_t>(r) * hidden + i] + s);
    } else {
        const float s = res[k * hidden + i];
        return store_value<S>(m.post[k] * sub[static_cast<std::size_t>(r) * hidden + i] + s);
    }
}

// Mix partials of chunk c for the rows of a tile: wave m < mix_rows the chunk's lane dot of fn row m with the stream,
// wave mix_rows the chunk's sum of squares; the wave's fn values stay in registers over the tile.
// partial[(r * mix_waves + m) * chunks + c].
template <Spec S>
__device__ __forceinline__ void mix_partial_block(const float* __restrict__ stream, const FnElem<S>* __restrict__ fn,
                                                  float* __restrict__ partial, unsigned hidden, unsigned rows, unsigned chunks,
                                                  unsigned c, unsigned r0) {
    const unsigned m = threadIdx.x / wave, l = threadIdx.x % wave, n = S.streams * hidden;
    const unsigned c0 = c * chunk, len = c0 + chunk < n ? chunk : n - c0, k_count = len / wave;
    const bool squares = m == mix_rows<S>;
    float f[chunk_k];
#pragma unroll
    for (unsigned k = 0; k < chunk_k; ++k)
        f[k] = !squares && k < k_count ? fn_value<S>(fn, static_cast<std::size_t>(m) * n + c0 + k * wave + l) : 0.0F;
    for (unsigned r = r0; r < r0 + mix_tile && r < rows; ++r) {
        const float* x = stream + static_cast<std::size_t>(r) * n + c0 + l;
        float xv[chunk_k];
#pragma unroll
        for (unsigned k = 0; k < chunk_k; ++k) xv[k] = k < k_count ? x[k * wave] : 0.0F;
        float acc = 0.0F;
#pragma unroll
        for (unsigned k = 0; k < chunk_k; ++k)
            if (k < k_count) acc = acc + (squares ? xv[k] : f[k]) * xv[k];
        acc = butterfly(acc);
        if (l == 0) partial[(static_cast<std::size_t>(r) * mix_waves<S> + m) * chunks + c] = acc;
    }
}

// One wave per row: the ascending chunk sums (lane t <= mix_rows holds dot t), the RMS factor, the split and, for
// Comb::sinkhorn, the Sinkhorn on the lanes: lane t (and t + 16) holds comb entry (j, k) = (t / 4, t % 4); every row or
// column sum is gathered from its four lanes and added in ascending order from 0, as on the host.
template <Spec S, class M>
__device__ __forceinline__ void mix_final_wave(const float* __restrict__ partial, const float* __restrict__ base,
                                               const float* __restrict__ scale, unsigned n, unsigned chunks, unsigned iters,
                                               float hc_eps, float norm_eps, M* __restrict__ out, unsigned r) {
    static_assert(S.comb_rows() == 0U || S.streams == 4U, "the lane Sinkhorn holds a 4 x 4 comb in 16 lanes");
    static_assert(mix_waves<S> <= wave, "one lane per mix row");
    const unsigned t = threadIdx.x % wave;
    float d = 0.0F;
    if (t < mix_waves<S>) {
        const float* p = partial + (static_cast<std::size_t>(r) * mix_waves<S> + t) * chunks;
        for (unsigned c = 0; c < chunks; ++c) d = d + p[c];
    }
    constexpr unsigned h = S.streams;
    const float rr = ::strix::ds41::declared_rsqrt(__shfl(d, static_cast<int>(mix_rows<S>), wave) / static_cast<float>(n) + norm_eps);
    M& o = out[r];
    if constexpr (S.post_rows() == 0U) {
        if (t < h) o.pre[t] = ::strix::ds41::declared_sigmoid((d * rr) * scale[0] + base[t]) + hc_eps;
    } else {
        if (t < h) o.pre[t] = ::strix::ds41::declared_sigmoid((d * rr) * scale[0] + base[t]) + hc_eps;
        else if (t < 2 * h) o.post[t - h] = post_value<S>(::strix::ds41::declared_sigmoid((d * rr) * scale[1] + base[t]));
    }
    if constexpr (S.comb_rows() != 0U) {
        const unsigned e = t & 15U, j = e / h, k = e % h;
        float v = (__shfl(d, static_cast<int>(2 * h + e), wave) * rr) * scale[2] + base[2 * h + e];
        auto at = [&](float x, unsigned jj, unsigned kk) { return __shfl(x, static_cast<int>(jj * h + kk), wave); };
        float mx = at(v, j, 0);
        for (unsigned kk = 1; kk < h; ++kk) { const float y = at(v, j, kk); mx = y > mx ? y : mx; }
        v = ::strix::ds41::declared_exp(v - mx);
        float sum = 0.0F;
        for (unsigned kk = 0; kk < h; ++kk) sum = sum + at(v, j, kk);
        v = v / sum + hc_eps;
        for (unsigned it = 0; it < iters; ++it) {
            if (it) {
                float rs = 0.0F;
                for (unsigned kk = 0; kk < h; ++kk) rs = rs + at(v, j, kk);
                v = v / (rs + hc_eps);
            }
            float cs = 0.0F;
            for (unsigned jj = 0; jj < h; ++jj) cs = cs + at(v, jj, k);
            v = v / (cs + hc_eps);
        }
        if (t < 16) o.comb[t] = v;
    }
}

// ---- kernels -------------------------------------------------------------------------------------------------
template <Spec S>
__global__ __launch_bounds__(32 * mix_waves<S>) void mix_partial_kernel(const float* __restrict__ stream,
                                                                        const FnElem<S>* __restrict__ fn, unsigned hidden,
                                                                        unsigned rows, unsigned chunks, float* __restrict__ partial) {
    mix_partial_block<S>(stream, fn, partial, hidden, rows, chunks, blockIdx.x, blockIdx.y * mix_tile);
}
template <Spec S, class M>
__global__ __launch_bounds__(32) void mix_final_kernel(const float* __restrict__ partial, const float* __restrict__ base,
                                                       const float* __restrict__ scale, unsigned n, unsigned chunks, unsigned iters,
                                                       float hc_eps, float norm_eps, M* __restrict__ out) {
    mix_final_wave<S>(partial, base, scale, n, chunks, iters, hc_eps, norm_eps, out, blockIdx.x);
}
// out[r][i] = store(0 + pre[0] * x[0][i] + ... ascending); null mixes: pre = (1, 0, ..., 0) (a first layer's copy 0).
template <Spec S, class M>
__global__ __launch_bounds__(256) void collapse_kernel(const float* __restrict__ stream, const M* __restrict__ mixes,
                                                       unsigned hidden, unsigned long long total, float* __restrict__ out) {
    const auto index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const auto r = index / hidden, i = index % hidden;
    const float* x = stream + r * S.streams * hidden;
    float y = 0.0F;
    for (unsigned c = 0; c < S.streams; ++c) {
        const float p = mixes ? mixes[r].pre[c] : (c == 0 ? 1.0F : 0.0F);
        y = y + p * x[c * hidden + i];
    }
    out[index] = store_value<S>(y);
}
template <Spec S, class M, class Sub>
__global__ __launch_bounds__(256) void expand_kernel(Sub sub, const float* __restrict__ residual, const M* __restrict__ mixes,
                                                     unsigned hidden, unsigned long long total, float* __restrict__ out) {
    const auto index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;  // total = rows * streams * hidden
    const auto per_row = static_cast<unsigned long long>(S.streams) * hidden;
    const auto r = static_cast<unsigned>(index / per_row), rest = static_cast<unsigned>(index % per_row);
    out[index] = copy_value<S>(sub, residual, mixes, r, rest / hidden, rest % hidden, hidden);
}

template <class Form>
__device__ __forceinline__ void mix_final_wave(const float* z, const float* base, const float* scale, float eps, float* out) {
    static_assert(Form::input == PackedInput::projected);
    using Math = typename Form::Arithmetic;
    const unsigned l = threadIdx.x;
    if (l >= wave) return;
    const unsigned t = l < 16 ? l : 0, j = t / 4, k = t % 4;
    if (l >= 16 && l < 20) out[l - 16] = Math::sigmoid(z[l - 16] * scale[0] + base[l - 16]) + eps;
    if (l >= 20 && l < 24) out[l - 16] = 2.0F * Math::sigmoid(z[l - 16] * scale[1] + base[l - 16]);
    float c = z[8 + t] * scale[2] + base[8 + t];
    float row[4], e[4];
#pragma unroll
    for (unsigned q = 0; q < 4; ++q) row[q] = __shfl(c, static_cast<int>(j * 4 + q));
    packed_softmax_exp<Form>(row, e);
    const float ek = k == 0 ? e[0] : k == 1 ? e[1] : k == 2 ? e[2] : e[3];
    c = ek / packed_sum(e[0], e[1], e[2], e[3]) + eps;
    for (unsigned it = 0; it < Form::iterations; ++it) {
        if (it != 0U) {
            const float sr = packed_sum(__shfl(c, static_cast<int>(j * 4)), __shfl(c, static_cast<int>(j * 4 + 1)),
                                        __shfl(c, static_cast<int>(j * 4 + 2)), __shfl(c, static_cast<int>(j * 4 + 3))) + eps;
            c = c / sr;
        }
        const float sc = packed_sum(__shfl(c, static_cast<int>(k)), __shfl(c, static_cast<int>(4 + k)),
                                    __shfl(c, static_cast<int>(8 + k)), __shfl(c, static_cast<int>(12 + k))) + eps;
        c = c / sc;
    }
    if (l < 16) out[8 + l] = c;
}

// One block per (dot or squares, row). Each wave sums one whole chunk and
// publishes it to LDS; each block writes one ordered sum to the packed scratch.
template <class Form, unsigned Hidden, unsigned Chunk>
struct PackedBlock : Form {
    static constexpr unsigned hidden = Hidden;
    static constexpr unsigned columns = Form::streams * Hidden;
    static constexpr unsigned values_per_wave = Chunk;
    static_assert(columns % Chunk == 0 && Chunk % wave == 0 && columns / Chunk <= 32);
};

struct HalfDecode {
    __device__ __forceinline__ float operator()(std::uint16_t bits) const { return __half2float(__ushort_as_half(bits)); }
};

template <class Form>
__global__ __launch_bounds__(1024) void mix_partial_kernel(const float* stream, const std::uint16_t* fn, float* scratch) {
    __shared__ float partial[32];
    const unsigned lane = threadIdx.x % wave, w = threadIdx.x / wave, j = blockIdx.x, r = blockIdx.y;
    const float* x = stream + static_cast<std::size_t>(r) * Form::columns;
    const unsigned first = w * Form::values_per_wave, end = first + Form::values_per_wave;
    const float acc = j < Form::mixes
        ? packed_lane_dot<false>(x, fn + static_cast<std::size_t>(j) * Form::columns, first, end, lane, HalfDecode{})
        : packed_lane_dot<true>(x, nullptr, first, end, lane, HalfDecode{});
    const float part = butterfly(acc);
    if (lane == 0) partial[w] = part;
    __syncthreads();
    const float total = packed_chunk_total(partial, blockDim.x / wave);
    __syncthreads();
    if (threadIdx.x == 0) scratch[r * (Form::mixes + 1) + j] = total;
}

template <class Form>
__global__ void mix_final_kernel(const float* scratch, const float* scale, const float* base, unsigned rows, float* mixes,
                                 float eps, float norm_eps) {
    const unsigned r = threadIdx.x;
    if (r >= rows) return;
    mix_final_packed<Form>(scratch + r * (Form::mixes + 1), scale, base, eps, mixes + r * Form::mixes, Form::columns, norm_eps);
}

template <class Form>
__global__ __launch_bounds__(256) void collapse_kernel(const float* stream, const float* coefficients, unsigned stride, float* out) {
    const unsigned i = blockIdx.x * 256U + threadIdx.x, r = blockIdx.y;
    if (i >= Form::hidden) return;
    const float* x = stream + static_cast<std::size_t>(r) * Form::columns;
    const float* p = coefficients + static_cast<std::size_t>(r) * stride;
    out[static_cast<std::size_t>(r) * Form::hidden + i] = packed_dot(p, x, i, Form::hidden);
}

template <class Form>
__global__ __launch_bounds__(256) void expand_kernel(const float* x, const float* residual, const float* mixes, float* out) {
    const unsigned i = blockIdx.x * 256U + threadIdx.x, r = blockIdx.y;
    if (i >= Form::hidden) return;
    const float* m = mixes + static_cast<std::size_t>(r) * Form::mixes;
    const float* s = residual + static_cast<std::size_t>(r) * Form::columns;
    const float xi = x[static_cast<std::size_t>(r) * Form::hidden + i];
    for (unsigned k = 0; k < Form::streams; ++k)
        out[static_cast<std::size_t>(r) * Form::columns + k * Form::hidden + i] =
            packed_expand(m, s, xi, k, i, Form::hidden);
}

template <class Consume, bool Bracket>
__global__ __launch_bounds__(256) void expand_kernel(const float* input, const float* mixes, const float* local,
                                                      const float* peer, typename Consume::View view, unsigned rank, float* out,
                                                      unsigned* flag, unsigned flag_stride) {
    peer = consume_peer(peer, view.payload, [&] { Consume::wait(view); });
    const unsigned r = blockIdx.y, i = blockIdx.x * 256U + threadIdx.x;
    constexpr unsigned hidden = Consume::hidden;
    if (i >= hidden) return;
    const float* m = mixes + static_cast<std::size_t>(r) * 24;
    const auto row = static_cast<std::size_t>(r) * 4 * hidden;
    const float sub = consume_rank_sum(local, peer, rank, Consume::index(r, i));
    for (unsigned k = 0; k < 4; ++k) {
        const auto offset = row + k * hidden + i;
        float value;
        if constexpr (Bracket) value = m[4 + k] * sub + input[offset];
        else value = packed_expand(m, input + row, sub, k, i, hidden);
        Consume::flag(flag ? flag + r * flag_stride : nullptr, value);
        out[offset] = value;
    }
}

// ---- host launchers (no checks beyond geometry: the binder validates its views; returns hipGetLastError()) ----
template <Spec S> constexpr std::size_t mix_partial_floats(std::uint32_t rows, std::uint32_t hidden) {
    const std::size_t chunks = (std::size_t{hidden} * S.streams + chunk - 1) / chunk;
    return std::size_t{rows} * mix_waves<S> * chunks;
}
inline unsigned blocks_for(unsigned long long total, unsigned threads) {
    return static_cast<unsigned>((total + threads - 1) / threads);
}
// Mixes of `rows` rows of streams [rows][N * hidden]: partial needs mix_partial_floats<S>(rows, hidden) floats.
// iters is read only for Comb::sinkhorn. hidden * N must be a multiple of 32.
template <Spec S, class M>
hipError_t enqueue_mixes(const float* stream, const FnElem<S>* fn, const float* base, const float* scale, unsigned rows,
                         unsigned hidden, unsigned iters, float hc_eps, float norm_eps, M* out, float* partial, hipStream_t s) {
    const unsigned n = hidden * S.streams, chunks = (n + chunk - 1U) / chunk;
    hipLaunchKernelGGL((mix_partial_kernel<S>), dim3(chunks, (rows + mix_tile - 1U) / mix_tile), dim3(wave * mix_waves<S>), 0, s,
                       stream, fn, hidden, rows, chunks, partial);
    hipLaunchKernelGGL((mix_final_kernel<S, M>), dim3(rows), dim3(wave), 0, s, partial, base, scale, n, chunks, iters, hc_eps,
                       norm_eps, out);
    return hipGetLastError();
}
template <Spec S, class M>
hipError_t enqueue_collapse(const float* stream, const M* mixes, unsigned rows, unsigned hidden, float* out, hipStream_t s) {
    const auto total = static_cast<unsigned long long>(rows) * hidden;
    hipLaunchKernelGGL((collapse_kernel<S, M>), dim3(blocks_for(total, 256)), dim3(256), 0, s, stream, mixes, hidden, total, out);
    return hipGetLastError();
}
template <Spec S, class M, class Sub = SubSum>
hipError_t enqueue_expand(Sub sub, const float* residual, const M* mixes, unsigned rows, unsigned hidden, float* out,
                          hipStream_t s) {
    const auto total = static_cast<unsigned long long>(rows) * S.streams * hidden;
    hipLaunchKernelGGL((expand_kernel<S, M, Sub>), dim3(blocks_for(total, 256)), dim3(256), 0, s, sub, residual, mixes, hidden,
                       total, out);
    return hipGetLastError();
}

#endif  // __HIPCC__
#undef STRIX_HC_HD
} // namespace strix::core::hc
