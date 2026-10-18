#pragma once

// T-row Q8_0 projections for known-prefix windows (T = 1..8) and prompt
// windows (T = 16, 32, 64; see project_wide_rows_kernel).
//
// Exact path: every row reproduces the wave8 virtual256 single-row arithmetic.
// Lane l owns leaves 8l..8l+7 of each 256-wide chunk (eight ascending FMA
// chains), then the lane tree (16, 8, 4, 2, 1) and the leaf tree (4, 2, 1);
// every addition keeps its lower/upper operands. A wave decodes the Q8 blocks
// of Outs outputs once and applies them to all rows. The tree runs as a
// reduce-scatter: each exchange halves the values a lane carries, so all
// Outs x Rows trees cost about two full exchanges instead of five.
//
// Activations: for large T >= 4 projections whose rows fit kStageBytes they are
// staged in LDS once per (persistent) block; otherwise each lane loads its
// leaves directly (they stay cache-resident). Weights are prefetched two
// 256-column chunks ahead.
//
// Dense A8 path (numerical profile dense-a8-v1, different numerics): see
// launch_dense_a8; every row count shares one per-row contract.
//
// T may be any count up to 8 (3/5/6/7 pad the tree to a power of two), or a
// multiple of the wide batch (16, 32, 64) above 8.
// A Site supplies Input (uint16_t BF16 or float), rows, input[] (or
// row_input(r)), segments, width, outputs, the coefficient/fma/add arithmetic of
// its R1 kernel and store(out, row, value); it may override outs, sub, prefetch,
// dense_a8_outs and a8_family (the dense A8 scratch family). The including
// translation unit owns the arithmetic flags (the dense A8 kernels disable contraction
// themselves and use explicit FMAs and round-to-nearest divisions).

#include <hip/hip_fp16.h>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace strix {
// Launches that drop the AQL barrier bit for kernels proven independent of the one queued
// just before them (default on; STRIX_ANYORDER=0 keeps every launch barrier-ordered).
inline bool strix_any_order_enabled() {
    static const bool enabled = [] { const char* v = std::getenv("STRIX_ANYORDER"); return !v || v[0] != '0'; }();
    return enabled;
}
}  // namespace strix

namespace strix::q8_rows {

constexpr unsigned kWaves = 8U, kThreads = 32U * kWaves, kMaxSegments = 5U;
constexpr std::uint32_t kStageBytes = 48U * 1024U, kLdsBytes = 64U * 1024U;

// Output index -> weight matrix and its row; segments share one input width.
struct Segments {
    const unsigned char* weights[kMaxSegments];
    std::uint32_t first[kMaxSegments + 1U];
    unsigned count;
};

inline Segments segments(std::initializer_list<std::pair<const void*, std::uint32_t>> parts) {
    Segments result{};
    for (const auto& [weights, outputs] : parts) {
        result.weights[result.count] = static_cast<const unsigned char*>(weights);
        result.first[result.count + 1U] = result.first[result.count] + outputs;
        ++result.count;
    }
    return result;
}

// Measured defaults: two outputs per wave; direct T <= 2 takes four (FP32) or
// one output with a one-chunk prefetch (BF16).
template <class Site, bool Staged> constexpr bool narrow_bf16() {
    return !Staged && Site::rows <= 2U && sizeof(typename Site::Input) == 2U;
}
template <class Site, bool Staged> constexpr unsigned outs_of() {
    if constexpr (requires { Site::outs; }) return Site::outs;
    else if constexpr (narrow_bf16<Site, Staged>()) return 1U;
    else return !Staged && Site::rows <= 2U ? 4U : 2U;
}
#ifndef STRIX_Q8_ROWS_PREFETCH
#define STRIX_Q8_ROWS_PREFETCH 2U
#endif
template <class Site, bool Staged> constexpr unsigned prefetch_of() {
    if constexpr (requires { Site::prefetch; }) return Site::prefetch;
    else return narrow_bf16<Site, Staged>() ? 1U : STRIX_Q8_ROWS_PREFETCH;
}
constexpr unsigned log2u(unsigned v) { return v <= 1U ? 0U : 1U + log2u(v / 2U); }
// Trees reduce a power-of-two number of (output, row) pairs; spare pairs stay zero.
constexpr unsigned pow2u(unsigned v) { return v <= 1U ? 1U : 2U * pow2u((v + 1U) / 2U); }

__device__ __forceinline__ const unsigned char* segment_row(const Segments& s, std::uint32_t out, std::uint32_t width,
    unsigned& m, std::uint32_t& row) {
    m = 0;
    while (m + 1U < s.count && out >= s.first[m + 1U]) ++m;
    row = out - s.first[m];
    return s.weights[m] + static_cast<std::uint64_t>(row) * (width / 32U) * 34U;
}
template <unsigned Outs>
__device__ __forceinline__ void group_rows(const Segments& s, std::uint32_t g, std::uint32_t outputs, std::uint32_t width,
    const unsigned char* (&w)[Outs]) {
#pragma unroll
    for (unsigned o = 0; o < Outs; ++o) {
        unsigned m; std::uint32_t row;
        w[o] = segment_row(s, min(g * Outs + o, outputs - 1U), width, m, row);
    }
}

// Lane ^ D: the widest exchange runs on the LDS pipe, row-local ones as DPP moves.
template <unsigned D> __device__ __forceinline__ float lane_xor(float v) {
    if constexpr (D == 16U) return __shfl_xor(v, 16, 32);
    else return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x160 | static_cast<int>(D), 0xf, 0xf, false));
}

// v holds pairs x leaves values (index pair * L + leaf). Lane steps 16..1 split the
// highest remaining index bit (lower lane keeps the lower half); leaf stages then
// pair leaf i with i + L/2, ... either across the lanes that split that bit or in
// registers. Result: v[0] on lanes with lane % (32 >> log2(pairs)) == 0, for pair
// lane >> (5 - log2(pairs)).
template <class Tree, unsigned N, unsigned Step = 0U, unsigned Count = N>
__device__ __forceinline__ void scatter(float (&v)[N], unsigned lane) {
    if constexpr (Step < 5U) {
        constexpr unsigned D = 16U >> Step;
        if constexpr (Count > 1U) {
            constexpr unsigned H = Count / 2U;
            const bool upper = (lane & D) != 0U;
#pragma unroll
            for (unsigned j = 0; j < H; ++j) {
                const float low = lane_xor<D>(v[j]), high = lane_xor<D>(v[j + H]);
                v[j] = Tree::add(upper ? high : v[j], upper ? v[j + H] : low);
            }
        } else v[0] = Tree::add(v[0], lane_xor<D>(v[0]));
        scatter<Tree, N, Step + 1U, (Count > 1U ? Count / 2U : 1U)>(v, lane);
    }
}
template <class Tree, unsigned N, unsigned L, unsigned Bit = log2u(L)>
__device__ __forceinline__ void leaves(float (&v)[N]) {
    if constexpr (Bit > 0U) {
        constexpr unsigned b = Bit - 1U, pairs_bits = log2u(N / L), total = pairs_bits + log2u(L);
        constexpr unsigned kept_bits = total > 5U ? total - 5U : 0U, step = pairs_bits + (log2u(L) - 1U - b);
        if constexpr (b >= kept_bits) {
            constexpr unsigned D = 16U >> step;
#pragma unroll
            for (unsigned j = 0; j < (1U << kept_bits); ++j) v[j] = Tree::add(v[j], lane_xor<D>(v[j]));
        } else {
#pragma unroll
            for (unsigned j = 0; j < (1U << kept_bits); ++j)
                if (!(j & (1U << b))) v[j] = Tree::add(v[j], v[j + (1U << b)]);
        }
        leaves<Tree, N, L, b>(v);
    }
}
template <class Tree, unsigned N, unsigned L>
__device__ __forceinline__ void reduce(float (&v)[N], unsigned lane) {
    static_assert(N / L <= 32U && (N & (N - 1U)) == 0U);
    scatter<Tree, N>(v, lane);
    leaves<Tree, N, L>(v);
}
// Reduction of the wide (Rows > 8) kernels: the same reduce-scatter and leaf tree, every
// addition with the same operands in the same order. Only the data movement differs: the
// distance-16 exchange is one VALU v_permlanex16_b32 (lane L reads lane L ^ 16) instead of
// a ds_bpermute through LDS, and the two halves of a step are exec-masked blocks of adds
// instead of per-element selects of operands.
template <unsigned D> __device__ __forceinline__ float lane_xor_wide(float v) {
    if constexpr (D == 16U)
        return __uint_as_float(__builtin_amdgcn_permlanex16(__float_as_uint(v), __float_as_uint(v), 0x76543210U, 0xfedcba98U, false, false));
    else return lane_xor<D>(v);
}
template <class Tree, unsigned N, unsigned Step = 0U, unsigned Count = N>
__device__ __forceinline__ void scatter_wide(float (&v)[N], unsigned lane) {
    if constexpr (Step < 5U) {
        constexpr unsigned D = 16U >> Step;
        if constexpr (Count > 1U) {
            constexpr unsigned H = Count / 2U;
            const bool upper = (lane & D) != 0U;
            // Lower lanes: add(own v[j], partner's v[j]); upper lanes: add(partner's v[j + H],
            // own v[j + H]). Both exchanges, then one exec-masked add per half (no selects).
            constexpr unsigned B = H < 8U ? H : 8U;  // pairs per exec-masked group (register pressure)
#pragma unroll
            for (unsigned j0 = 0; j0 < H; j0 += B) {
                float low[B], high[B];
#pragma unroll
                for (unsigned j = 0; j < B; ++j) { low[j] = lane_xor_wide<D>(v[j0 + j]); high[j] = lane_xor_wide<D>(v[j0 + j + H]); }
                if (!upper) {
#pragma unroll
                    for (unsigned j = 0; j < B; ++j) v[j0 + j] = Tree::add(v[j0 + j], low[j]);
                } else {
#pragma unroll
                    for (unsigned j = 0; j < B; ++j) v[j0 + j] = Tree::add(high[j], v[j0 + j + H]);
                }
            }
        } else v[0] = Tree::add(v[0], lane_xor_wide<D>(v[0]));
        scatter_wide<Tree, N, Step + 1U, (Count > 1U ? Count / 2U : 1U)>(v, lane);
    }
}
template <class Tree, unsigned N, unsigned L, unsigned Bit = log2u(L)>
__device__ __forceinline__ void leaves_wide(float (&v)[N]) {
    if constexpr (Bit > 0U) {
        constexpr unsigned b = Bit - 1U, pairs_bits = log2u(N / L), total = pairs_bits + log2u(L);
        constexpr unsigned kept_bits = total > 5U ? total - 5U : 0U, step = pairs_bits + (log2u(L) - 1U - b);
        if constexpr (b >= kept_bits) {
            constexpr unsigned D = 16U >> step;
#pragma unroll
            for (unsigned j = 0; j < (1U << kept_bits); ++j) v[j] = Tree::add(v[j], lane_xor_wide<D>(v[j]));
        } else {
#pragma unroll
            for (unsigned j = 0; j < (1U << kept_bits); ++j)
                if (!(j & (1U << b))) v[j] = Tree::add(v[j], v[j + (1U << b)]);
        }
        leaves_wide<Tree, N, L, b>(v);
    }
}
template <class Tree, unsigned N, unsigned L>
__device__ __forceinline__ void reduce_wide(float (&v)[N], unsigned lane) {
    static_assert(N / L <= 32U && (N & (N - 1U)) == 0U);
    scatter_wide<Tree, N>(v, lane);
    leaves_wide<Tree, N, L>(v);
}
template <unsigned Pairs, class Store>
__device__ __forceinline__ void store_pairs(float value, unsigned lane, Store&& store) {
    constexpr unsigned bits = log2u(Pairs);
    if (!(lane & ((32U >> bits) - 1U))) store(lane >> (5U - bits), value);
}

template <class Input> __device__ __forceinline__ void load8(const Input* p, float (&x)[8]) {
    if constexpr (sizeof(Input) == 2U) {
        const auto q = *reinterpret_cast<const uint4*>(p);
        const unsigned w[4]{q.x, q.y, q.z, q.w};
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) { x[2U * i] = __uint_as_float(w[i] << 16U); x[2U * i + 1U] = __uint_as_float(w[i] & 0xffff0000U); }
    } else {
        const auto a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
        x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w; x[4] = b.x; x[5] = b.y; x[6] = b.z; x[7] = b.w;
    }
}
// load8 split into the 16/32-byte load and the unpack (same values).
template <class Input> struct Raw8 {
    uint4 q[sizeof(Input) / 2U];
    __device__ __forceinline__ void load(const Input* p) {
#pragma unroll
        for (unsigned i = 0; i < sizeof(Input) / 2U; ++i) q[i] = reinterpret_cast<const uint4*>(p)[i];
    }
    // Uniform base + 32-bit byte offset: lets the compiler use the SGPR-base form.
    __device__ __forceinline__ void load(const Input* p, std::uint32_t bytes) {
#pragma unroll
        for (unsigned i = 0; i < sizeof(Input) / 2U; ++i)
            q[i] = *reinterpret_cast<const uint4*>(reinterpret_cast<const unsigned char*>(p) + (bytes + 16U * i));
    }
    __device__ __forceinline__ void unpack(float (&x)[8]) const {
        if constexpr (sizeof(Input) == 2U) {
            const unsigned w[4]{q[0].x, q[0].y, q[0].z, q[0].w};
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) { x[2U * i] = __uint_as_float(w[i] << 16U); x[2U * i + 1U] = __uint_as_float(w[i] & 0xffff0000U); }
        } else {
            const unsigned w[8]{q[0].x, q[0].y, q[0].z, q[0].w, q[1].x, q[1].y, q[1].z, q[1].w};
#pragma unroll
            for (unsigned i = 0; i < 8U; ++i) x[i] = __uint_as_float(w[i]);
        }
    }
};
extern __shared__ __align__(16) unsigned char q8_rows_lds[];
inline unsigned compute_units() {
    static const unsigned units = [] {
        int device = 0, count = 0;
        if (hipGetDevice(&device) != hipSuccess ||
            hipDeviceGetAttribute(&count, hipDeviceAttributeMultiprocessorCount, device) != hipSuccess || count <= 0)
            return 40U;
        return static_cast<unsigned>(count);
    }();
    return units;
}
// Staging pays off for T >= 4 when the weight stream dwarfs every resident
// block's copy of the rows; otherwise direct loads win.
template <class Site> std::uint32_t staged_bytes(const Site& site) {
    const auto bytes = static_cast<std::uint64_t>(Site::rows) * site.width * sizeof(typename Site::Input);
    const auto weights = static_cast<std::uint64_t>(site.outputs) * (site.width / 32U) * 34U;
    return Site::rows >= 4U && bytes <= kStageBytes && weights >= 8U * bytes * compute_units() ? static_cast<std::uint32_t>(bytes) : 0U;
}

template <unsigned Outs> struct Q8Fetch {
    std::uint64_t codes[Outs];
    unsigned scale[Outs];
    __device__ __forceinline__ void load(const unsigned char* const (&w)[Outs], std::uint32_t base) {
#pragma unroll
        for (unsigned o = 0; o < Outs; ++o) {
            const unsigned char* block = w[o] + (base / 32U) * 34U;
            scale[o] = *reinterpret_cast<const std::uint16_t*>(block);
            std::memcpy(&codes[o], block + 2U + base % 32U, 8U);
        }
    }
};

// Staged: every block copies all rows to LDS once, then loops over output groups.
// Direct: one output group per wave.
template <class Site, unsigned Outs, bool Staged>
__global__ __launch_bounds__(kThreads) void project_rows_kernel(Site site) {
    using Input = typename Site::Input;
    constexpr unsigned Rows = Site::rows, Pairs = Outs * Rows, N = pow2u(Pairs) * 8U, Depth = prefetch_of<Site, Staged>();
    const unsigned lane = threadIdx.x & 31U;
    const std::uint32_t width = site.width, groups = (site.outputs + Outs - 1U) / Outs;
    if constexpr (Staged) {
        const std::uint32_t vectors = width * static_cast<std::uint32_t>(sizeof(Input)) / 16U;
        for (std::uint32_t i = threadIdx.x; i < Rows * vectors; i += kThreads)
            reinterpret_cast<uint4*>(q8_rows_lds)[i] = static_cast<const uint4*>(site.input[i / vectors])[i % vectors];
        __syncthreads();
    }
    for (std::uint32_t g = blockIdx.x * kWaves + threadIdx.x / 32U; g < groups; g += Staged ? gridDim.x * kWaves : groups) {
        const unsigned char* w[Outs];
        group_rows(site.segments, g, site.outputs, width, w);
        float v[N]{};
        Q8Fetch<Outs> next[Depth];
#pragma unroll
        for (unsigned d = 0; d < Depth; ++d) if (8U * lane + 256U * d < width) next[d].load(w, 8U * lane + 256U * d);
        for (std::uint32_t base = 8U * lane; base < width; base += 256U) {
            const auto current = next[0];
#pragma unroll
            for (unsigned d = 1; d < Depth; ++d) next[d - 1U] = next[d];
            if (base + 256U * Depth < width) next[Depth - 1U].load(w, base + 256U * Depth);
            float coefficient[Outs][8];
#pragma unroll
            for (unsigned o = 0; o < Outs; ++o)
#pragma unroll
                for (unsigned l = 0; l < 8U; ++l)
                    coefficient[o][l] = Site::coefficient(current.scale[o], static_cast<std::int8_t>(current.codes[o] >> (8U * l)));
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r) {
                float x[8];
                if constexpr (Staged) load8(reinterpret_cast<const Input*>(q8_rows_lds) + r * width + base, x);
                else load8(static_cast<const Input*>(site.input[r]) + base, x);
#pragma unroll
                for (unsigned o = 0; o < Outs; ++o)
#pragma unroll
                    for (unsigned l = 0; l < 8U; ++l) Site::fma(v[(o * Rows + r) * 8U + l], coefficient[o][l], x[l]);
            }
        }
        reduce<Site, N, 8U>(v, lane);
        store_pairs<pow2u(Pairs)>(v[0], lane, [&](unsigned pair, float value) {
            if (pair < Pairs && g * Outs + pair / Rows < site.outputs) site.store(g * Outs + pair / Rows, pair % Rows, value);
        });
    }
}

// Wide windows (Rows > 8, e.g. 16/32/64 prompt rows): the rows are split into
// Rows / Sub batches and each wave runs the exact chains of Outs outputs x Sub
// rows (Outs * Sub = 16 pairs: 128 accumulators, the R8 register footprint). The
// 8 waves of a block take 8 output groups of one batch and consecutive blocks
// take the batches of the same output groups, so they read the same weight bytes
// at about the same time: DRAM serves each weight byte about once per window and
// the other batches hit L2. Per (output, row) pair the arithmetic is the single-
// row kernel's: leaf l of lane L is the ascending-k chain Site::fma over features
// 8L + l + 256k, then the lane tree (16..1) and the leaf tree (4, 2, 1) of
// reduce(); the reduce-scatter order of a pair does not depend on how many pairs
// share the wave, so any (Outs, Sub) split gives the R1 bits.
//
// Kernels (see launch): sites with fused_fma and Rows % 16 == 0 default to the VOPD
// kernels below (shared coefficients, 4 outputs x 4 rows per wave, pinned-register
// VOPD FMA blocks); others, and FP32 inputs narrower than 1024, use this per-wave
// kernel with Sub = 8, Outs = 2 (Sub = 16, Outs = 1 also fits 128 accumulators but
// reads every activation once per output and is slower). A site may set
// `static constexpr unsigned sub`; the STRIX_Q8_ROWS_WIDE_* variables override the
// choice for tuning.
template <class Site> constexpr unsigned wide_sub_of() {
    if constexpr (requires { Site::sub; }) return Site::sub; else return 8U;
}
template <class Site> constexpr unsigned wide_prefetch_of() {
    if constexpr (requires { Site::prefetch; }) return Site::prefetch; else return STRIX_Q8_ROWS_PREFETCH;
}
// Row r's activations: a site may address rows itself (row_input(r), e.g. base +
// r * stride for a uniformly strided arena) instead of carrying one pointer per row.
template <class Site> __device__ __forceinline__ const typename Site::Input* row_input(const Site& site, unsigned r) {
    if constexpr (requires { site.row_input(r); }) return site.row_input(r);
    else return static_cast<const typename Site::Input*>(site.input[r]);
}
template <class Site, unsigned Outs, unsigned Sub>
__global__ __launch_bounds__(kThreads) void project_wide_rows_kernel(Site site) {
    using Input = typename Site::Input;
    constexpr unsigned Rows = Site::rows, Batches = Rows / Sub, Pairs = Outs * Sub, N = pow2u(Pairs) * 8U;
    constexpr unsigned Depth = wide_prefetch_of<Site>();
    static_assert(Rows > 8U && Rows % Sub == 0U && Pairs <= 32U);
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x / 32U;
    const std::uint32_t width = site.width, groups = (site.outputs + Outs - 1U) / Outs;
    const std::uint32_t g = blockIdx.x / Batches * kWaves + wave;
    const unsigned first = blockIdx.x % Batches * Sub;
    if (g >= groups) return;
    const unsigned char* w[Outs];
    group_rows(site.segments, g, site.outputs, width, w);
    const Input* x_row[Sub];
#pragma unroll
    for (unsigned r = 0; r < Sub; ++r) x_row[r] = row_input(site, first + r);
    float v[N]{};
    Q8Fetch<Outs> next[Depth];
#pragma unroll
    for (unsigned d = 0; d < Depth; ++d) if (8U * lane + 256U * d < width) next[d].load(w, 8U * lane + 256U * d);
    for (std::uint32_t base = 8U * lane; base < width; base += 256U) {
        // The chunk's activations are requested before the weight prefetch: loads
        // retire in order, so waiting for them must not wait for the prefetch.
        Raw8<Input> raw[Sub];
#pragma unroll
        for (unsigned r = 0; r < Sub; ++r) raw[r].load(x_row[r] + base);
        const auto current = next[0];
#pragma unroll
        for (unsigned d = 1; d < Depth; ++d) next[d - 1U] = next[d];
        if (base + 256U * Depth < width) next[Depth - 1U].load(w, base + 256U * Depth);
        float coefficient[Outs][8];
#pragma unroll
        for (unsigned o = 0; o < Outs; ++o)
#pragma unroll
            for (unsigned l = 0; l < 8U; ++l)
                coefficient[o][l] = Site::coefficient(current.scale[o], static_cast<std::int8_t>(current.codes[o] >> (8U * l)));
#pragma unroll
        for (unsigned r = 0; r < Sub; ++r) {
            float x[8];
            raw[r].unpack(x);
#pragma unroll
            for (unsigned o = 0; o < Outs; ++o)
#pragma unroll
                for (unsigned l = 0; l < 8U; ++l) Site::fma(v[(o * Sub + r) * 8U + l], coefficient[o][l], x[l]);
        }
    }
    reduce_wide<Site, N, 8U>(v, lane);
    store_pairs<pow2u(Pairs)>(v[0], lane, [&](unsigned pair, float value) {
        if (pair < Pairs && g * Outs + pair / Sub < site.outputs) site.store(g * Outs + pair / Sub, first + pair % Sub, value);
    });
}

// Chunk-staged wide kernel: the 8 waves of a block take 8 output groups of one
// row batch (consecutive blocks take the batches of the same output groups, so
// their weight reads meet in L2). Each 256-feature chunk of the batch's Sub rows
// is copied to LDS once per block, double-buffered with one barrier per chunk;
// the copy of chunk k + 1 is requested before the weight prefetch and lands in
// LDS after chunk k's FMAs. Waves past the last output group still copy and hit
// every barrier (their clamped weight rows are read but never stored).
// Barrier ordering LDS only: __syncthreads() also waits for every outstanding
// global load (the weight prefetch) and invalidates L0 each chunk.
__device__ __forceinline__ void lds_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
// Workgroup barrier ordering LDS only (fences on the local address space): the
// compiler knows it, so it waits for LDS traffic only and keeps it in program order.
__device__ __forceinline__ void lds_fence_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
template <class Input, unsigned Sub> constexpr unsigned stage_vectors() { return Sub * 256U * sizeof(Input) / 16U; }
template <class Site, unsigned Outs, unsigned Sub>
__global__ __launch_bounds__(kThreads) void project_wide_rows_staged_kernel(Site site) {
    using Input = typename Site::Input;
    constexpr unsigned Rows = Site::rows, Batches = Rows / Sub, Pairs = Outs * Sub, N = pow2u(Pairs) * 8U;
    constexpr unsigned Depth = wide_prefetch_of<Site>(), Vectors = stage_vectors<Input, Sub>(), RowVectors = Vectors / Sub;
    constexpr unsigned PerThread = (Vectors + kThreads - 1U) / kThreads;
    static_assert(Rows > 8U && Rows % Sub == 0U && Pairs <= 32U && kThreads % RowVectors == 0U);
    // Staged as FP32 (BF16 rows widened once per block, the exact bit shift of load8).
    __shared__ float4 stage[2][Sub * 64U];
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x / 32U;
    const std::uint32_t width = site.width, groups = (site.outputs + Outs - 1U) / Outs, chunks = (width + 255U) / 256U;
    const std::uint32_t g = blockIdx.x / Batches * kWaves + wave;
    const unsigned first = blockIdx.x % Batches * Sub;
    // Thread t copies vectors t + kThreads * j; their rows are wave-uniform.
    const Input* source[PerThread];
#pragma unroll
    for (unsigned j = 0; j < PerThread; ++j) {
        const unsigned i = threadIdx.x + kThreads * j;
        const unsigned row = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(i / RowVectors)));
        source[j] = row_input(site, first + (row < Sub ? row : 0U)) + (i % RowVectors) * (16U / sizeof(Input));
    }
    uint4 staged[PerThread]{};
    const auto fetch = [&](std::uint32_t k) {
#pragma unroll
        for (unsigned j = 0; j < PerThread; ++j) {
            const unsigned i = threadIdx.x + kThreads * j;
            const std::uint32_t feature = 256U * k + (i % RowVectors) * (16U / sizeof(Input));
            if ((Vectors % kThreads == 0U || i < Vectors) && feature < width)
                staged[j] = *reinterpret_cast<const uint4*>(source[j] + 256U * k);
        }
    };
    const auto commit = [&](std::uint32_t k) {
#pragma unroll
        for (unsigned j = 0; j < PerThread; ++j) {
            const unsigned i = threadIdx.x + kThreads * j;
            if (Vectors % kThreads == 0U || i < Vectors) {
                if constexpr (sizeof(Input) == 2U) {
                    const uint4 q = staged[j];
                    stage[k & 1U][2U * i] = make_float4(__uint_as_float(q.x << 16U), __uint_as_float(q.x & 0xffff0000U),
                        __uint_as_float(q.y << 16U), __uint_as_float(q.y & 0xffff0000U));
                    stage[k & 1U][2U * i + 1U] = make_float4(__uint_as_float(q.z << 16U), __uint_as_float(q.z & 0xffff0000U),
                        __uint_as_float(q.w << 16U), __uint_as_float(q.w & 0xffff0000U));
                } else stage[k & 1U][i] = make_float4(__uint_as_float(staged[j].x), __uint_as_float(staged[j].y),
                    __uint_as_float(staged[j].z), __uint_as_float(staged[j].w));
            }
        }
    };
    const unsigned char* w[Outs];
    group_rows(site.segments, g < groups ? g : groups - 1U, site.outputs, width, w);
    float v[N]{};
    Q8Fetch<Outs> next[Depth];
#pragma unroll
    for (unsigned d = 0; d < Depth; ++d) if (8U * lane + 256U * d < width) next[d].load(w, 8U * lane + 256U * d);
    fetch(0U);
    commit(0U);
    __syncthreads();
    // Full chunks run unconditionally (a lane-dependent branch around the FMAs makes the
    // compiler copy every accumulator each iteration); only a partial last chunk masks lanes.
    const auto step = [&](std::uint32_t k, auto tail) {
        const std::uint32_t base = 8U * lane + 256U * k;
        if (k + 1U < chunks) fetch(k + 1U);
        const auto current = next[0];
#pragma unroll
        for (unsigned d = 1; d < Depth; ++d) next[d - 1U] = next[d];
        if (base + 256U * Depth < width) next[Depth - 1U].load(w, base + 256U * Depth);
        if (!decltype(tail)::value || base < width) {
            float coefficient[Outs][8];
#pragma unroll
            for (unsigned o = 0; o < Outs; ++o)
#pragma unroll
                for (unsigned l = 0; l < 8U; ++l)
                    coefficient[o][l] = Site::coefficient(current.scale[o], static_cast<std::int8_t>(current.codes[o] >> (8U * l)));
            const float4* chunk = stage[k & 1U];
#pragma unroll
            for (unsigned r = 0; r < Sub; ++r) {
                const float4 lo = chunk[r * 64U + 2U * lane], hi = chunk[r * 64U + 2U * lane + 1U];
                const float x[8]{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
                for (unsigned o = 0; o < Outs; ++o)
#pragma unroll
                    for (unsigned l = 0; l < 8U; ++l) Site::fma(v[(o * Sub + r) * 8U + l], coefficient[o][l], x[l]);
            }
        }
        if (k + 1U < chunks) commit(k + 1U);
        lds_barrier();
    };
    const std::uint32_t full = width / 256U;
    for (std::uint32_t k = 0; k < full; ++k) step(k, std::false_type{});
    if (full < chunks) step(full, std::true_type{});
    if (g >= groups) return;
    reduce_wide<Site, N, 8U>(v, lane);
    store_pairs<pow2u(Pairs)>(v[0], lane, [&](unsigned pair, float value) {
        if (pair < Pairs && g * Outs + pair / Sub < site.outputs) site.store(g * Outs + pair / Sub, first + pair % Sub, value);
    });
}

// Shared-coefficient wide kernel. The waves of a block take row batches of ONE output
// group, so they need the same coefficients: each 256-feature chunk's Outs x 256
// coefficients are decoded once per block (every thread decodes Per of them with
// Site::coefficient) into LDS, double-buffered with one barrier per chunk, and every
// wave reads its lane's 8 per output with two conflict-free ds_load_b128. The FMA
// chains, lane tree and leaf tree are those of project_wide_rows_kernel (the
// coefficient bits are the same; only the thread that computes them differs). Weights
// are fetched once per block instead of once per wave and the decode costs
// Outs * 256 / threads per thread and chunk instead of Outs * 8.
//
// Vopd (sites with fused_fma, Outs = Sub = 4): the 128 FMAs of a chunk run as the
// generated pinned-register VOPD block of hip_q8_rows_vopd.inc (every FMA reads its
// three operands from three VGPR banks; the compiler's own allocation puts two of them
// in one bank for most FMAs, which costs an extra cycle each).
#include "hip_q8_rows_vopd.inc"
// Sites whose Site::fma compiles to the fused v_fmac_f32 (d = fma(c, x, d)) may take the
// hand-paired VOPD blocks; the projection test compares them bitwise with R1.
template <class Site> constexpr bool fused_fma_of() {
    if constexpr (requires { Site::fused_fma; }) return Site::fused_fma; else return false;
}
template <class Site, unsigned Sub, unsigned W = kWaves> constexpr unsigned coef_waves() {
    return Site::rows / Sub < W ? Site::rows / Sub : W;
}
__device__ __forceinline__ std::uint32_t umin(std::uint32_t a, std::uint32_t b) { return a < b ? a : b; }
__device__ __forceinline__ unsigned uniform(unsigned v) {
    return static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(v)));
}
// Producer side shared by the coefficient kernels: thread t decodes features
// pf..pf + Per - 1 of output po of every chunk (po is wave-uniform).
template <class Site, unsigned Outs, unsigned Per> struct CoefProducer {
    using Codes = std::conditional_t<Per == 2U, std::uint16_t, std::conditional_t<Per == 4U, std::uint32_t, std::uint64_t>>;
    struct Fetch { unsigned scale; Codes codes; };
    const unsigned char* pw;
    unsigned po, pf;
    std::uint32_t width;
    // Chunk k's block of this thread (features past the width reuse the last ones; never consumed).
    __device__ __forceinline__ Fetch fetch(std::uint32_t k) const {
        const std::uint32_t f = umin(256U * k + pf, width - Per);
        const unsigned char* block = pw + (f / 32U) * 34U;
        Fetch result;
        result.scale = *reinterpret_cast<const std::uint16_t*>(block);
        std::memcpy(&result.codes, block + 2U + f % 32U, sizeof(Codes));
        return result;
    }
    // Coefficient of feature f at quad (f / 4) % 2 * 32 + f / 8 of its output: lane L
    // reads leaves 0..3 at quad L and 4..7 at quad 32 + L.
    __device__ __forceinline__ void produce(const Fetch& in, float4* buffer) const {
        float c[Per];
#pragma unroll
        for (unsigned i = 0; i < Per; ++i) c[i] = Site::coefficient(in.scale, static_cast<std::int8_t>(in.codes >> (8U * i)));
        float* target = reinterpret_cast<float*>(buffer + po * 64U);
        const auto slot = [&](unsigned f) { return ((f / 4U) % 2U * 32U + f / 8U) * 4U + f % 4U; };
        if constexpr (Per == 2U) {
#pragma unroll
            for (unsigned i = 0; i < 2U; ++i) target[slot(pf + i)] = c[i];
        } else {
#pragma unroll
            for (unsigned i = 0; i < Per; i += 4U)
                *reinterpret_cast<float4*>(target + slot(pf + i)) = make_float4(c[i], c[i + 1U], c[i + 2U], c[i + 3U]);
        }
    }
};
// Producer of the 4-output VOPD kernels with Waves = 4, 2 or 1 waves per block: lane L
// of wave w decodes features 8L..8L+7 of outputs w * G .. w * G + G - 1 (G = 4 / Waves),
// so every weight row pointer stays wave-uniform; same slots as CoefProducer.
template <class Site, unsigned Waves> struct VopdProducer {
    static constexpr unsigned G = 4U / Waves;
    struct Fetch { unsigned scale[G]; std::uint64_t codes[G]; };
    const unsigned char* pw[G];
    unsigned po, pf;
    std::uint32_t width;
    __device__ __forceinline__ void init(const Site& site, std::uint32_t g, unsigned wave, unsigned lane) {
        po = wave * G; pf = 8U * lane; width = site.width;
#pragma unroll
        for (unsigned j = 0; j < G; ++j) {
            unsigned m; std::uint32_t row;
            pw[j] = segment_row(site.segments, min(g * 4U + po + j, site.outputs - 1U), width, m, row);
        }
    }
    __device__ __forceinline__ Fetch fetch(std::uint32_t k) const {
        const std::uint32_t f = umin(256U * k + pf, width - 8U);
        Fetch result;
#pragma unroll
        for (unsigned j = 0; j < G; ++j) {
            const unsigned char* block = pw[j] + (f / 32U) * 34U;
            result.scale[j] = *reinterpret_cast<const std::uint16_t*>(block);
            std::memcpy(&result.codes[j], block + 2U + f % 32U, 8U);
        }
        return result;
    }
    __device__ __forceinline__ void produce(const Fetch& in, float4* buffer) const {
#pragma unroll
        for (unsigned j = 0; j < G; ++j) {
            float c[8];
#pragma unroll
            for (unsigned i = 0; i < 8U; ++i) c[i] = Site::coefficient(in.scale[j], static_cast<std::int8_t>(in.codes[j] >> (8U * i)));
            float4* target = buffer + (po + j) * 64U;
            target[pf / 8U] = make_float4(c[0], c[1], c[2], c[3]);
            target[32U + pf / 8U] = make_float4(c[4], c[5], c[6], c[7]);
        }
    }
};
template <class Site, unsigned Outs, unsigned Sub, unsigned W, bool Vopd>
__global__ __launch_bounds__((32U * coef_waves<Site, Sub, W>())) void project_wide_rows_coef_kernel(Site site) {
    using Input = typename Site::Input;
    constexpr unsigned Rows = Site::rows, Batches = Rows / Sub, Waves = coef_waves<Site, Sub, W>(), Threads = 32U * Waves;
    constexpr unsigned Blocks = Batches / Waves, Pairs = Outs * Sub, N = pow2u(Pairs) * 8U, Per = Outs * 256U / Threads;
    static_assert((Rows > 8U || Vopd) && Rows % Sub == 0U && Batches % Waves == 0U && Pairs <= 32U);
    static_assert(Vopd || Per == 2U || Per == 4U || Per == 8U, "each thread decodes 2, 4 or 8 coefficients per chunk");
    static_assert(!Vopd || (Outs == 4U && Sub == 4U && fused_fma_of<Site>()));
    using Producer = std::conditional_t<Vopd, VopdProducer<Site, Waves>, CoefProducer<Site, Outs, (Per < 2U ? 2U : Per > 8U ? 8U : Per)>>;
    __shared__ float4 coef[2][Outs * 64U];
    const unsigned lane = threadIdx.x & 31U, wave = uniform(threadIdx.x / 32U);
    const std::uint32_t width = site.width, chunks = (width + 255U) / 256U, full = width / 256U;
    const std::uint32_t g = blockIdx.x / Blocks;
    const unsigned first = ((blockIdx.x % Blocks) * Waves + wave) * Sub;
    Producer producer;
    if constexpr (Vopd) producer.init(site, g, wave, lane);
    else {
        producer.po = uniform(threadIdx.x * Per / 256U);
        producer.pf = threadIdx.x * Per % 256U;
        producer.width = width;
        unsigned m; std::uint32_t row;
        producer.pw = segment_row(site.segments, min(g * Outs + producer.po, site.outputs - 1U), width, m, row);
    }
    const Input* x_row[Sub];
#pragma unroll
    for (unsigned r = 0; r < Sub; ++r) x_row[r] = row_input(site, first + r);
    float v[N]{};
    typename Producer::Fetch ring[2]{producer.fetch(umin(1U, chunks - 1U)), producer.fetch(umin(2U, chunks - 1U))};
    producer.produce(producer.fetch(0U), coef[0]);
    lds_fence_barrier();
    // Full chunks run unconditionally; only a partial last chunk masks lanes past the width
    // (they load a valid vector and skip the FMAs).
    const auto step = [&](std::uint32_t k, auto tail) {
        const std::uint32_t base = 8U * lane + 256U * k;
        const std::uint32_t at = (decltype(tail)::value ? umin(base, width - 8U) : base) * static_cast<std::uint32_t>(sizeof(Input));
        Raw8<Input> raw[Sub];
#pragma unroll
        for (unsigned r = 0; r < Sub; ++r) raw[r].load(x_row[r], at);
        const auto current = ring[0];
        ring[0] = ring[1];
        ring[1] = producer.fetch(umin(k + 3U, chunks - 1U));
        float4 cv[Outs][2];
        if constexpr (!Vopd) {
#pragma unroll
            for (unsigned o = 0; o < Outs; ++o) { cv[o][0] = coef[k & 1U][o * 64U + lane]; cv[o][1] = coef[k & 1U][o * 64U + 32U + lane]; }
        }
        if (k + 1U < chunks) producer.produce(current, coef[(k + 1U) & 1U]);
        if (!decltype(tail)::value || base < width) {
            if constexpr (Vopd) {
                constexpr unsigned Words = sizeof(Input) * 2U;
                unsigned w[4U * Words];
#pragma unroll
                for (unsigned r = 0; r < 4U; ++r)
#pragma unroll
                    for (unsigned i = 0; i < Words; ++i) {
                        const uint4 q = raw[r].q[i / 4U];
                        w[r * Words + i] = i % 4U == 0U ? q.x : i % 4U == 1U ? q.y : i % 4U == 2U ? q.z : q.w;
                    }
                // The block reads this lane's coefficient quads itself (LDS byte address).
                vopd_chunk(v, w, static_cast<unsigned>(reinterpret_cast<std::uintptr_t>(&coef[k & 1U][lane])));
            } else {
                float coefficient[Outs][8];
#pragma unroll
                for (unsigned o = 0; o < Outs; ++o) {
                    coefficient[o][0] = cv[o][0].x; coefficient[o][1] = cv[o][0].y; coefficient[o][2] = cv[o][0].z; coefficient[o][3] = cv[o][0].w;
                    coefficient[o][4] = cv[o][1].x; coefficient[o][5] = cv[o][1].y; coefficient[o][6] = cv[o][1].z; coefficient[o][7] = cv[o][1].w;
                }
#pragma unroll
                for (unsigned r = 0; r < Sub; ++r) {
                    float x[8];
                    raw[r].unpack(x);
#pragma unroll
                    for (unsigned o = 0; o < Outs; ++o)
#pragma unroll
                        for (unsigned l = 0; l < 8U; ++l) Site::fma(v[(o * Sub + r) * 8U + l], coefficient[o][l], x[l]);
                }
            }
        }
        lds_fence_barrier();
    };
    for (std::uint32_t k = 0; k < full; ++k) step(k, std::false_type{});
    if (full < chunks) step(full, std::true_type{});
    if constexpr (Rows > 8U) reduce_wide<Site, N, 8U>(v, lane); else reduce<Site, N, 8U>(v, lane);
    store_pairs<pow2u(Pairs)>(v[0], lane, [&](unsigned pair, float value) {
        if (pair < Pairs && g * Outs + pair / Sub < site.outputs) site.store(g * Outs + pair / Sub, first + pair % Sub, value);
    });
}

// BF16 VOPD kernel with the next chunk's activation rows prefetched (4 outputs x 4
// rows per wave, 4 waves per block). Loads retire in order, so a wave waiting for its
// activations also waits for every weight load issued before them: the next chunk's
// rows are requested at the top of each step, before that step's weight prefetch,
// which gives the DRAM weight loads two steps instead of one. The two raw buffers sit
// in v192..v207 and v208..v223 (vopd_chunk / vopd_chunk_b1). Same arithmetic as
// project_wide_rows_coef_kernel (and the same producer).
template <class Site, unsigned Waves = (Site::rows / 4U < 4U ? Site::rows / 4U : 4U)>
__global__ __launch_bounds__(32U * Waves) void project_wide_rows_vopd_bf16_kernel(Site site) {
    static_assert(sizeof(typename Site::Input) == 2U && Site::rows % (4U * Waves) == 0U && fused_fma_of<Site>());
    constexpr unsigned Outs = 4U, Sub = 4U, Blocks = Site::rows / Sub / Waves;
    __shared__ float4 coef[2][Outs * 64U];
    const unsigned lane = threadIdx.x & 31U, wave = uniform(threadIdx.x / 32U);
    const std::uint32_t width = site.width, chunks = (width + 255U) / 256U;
    const std::uint32_t g = blockIdx.x / Blocks;
    const unsigned first = ((blockIdx.x % Blocks) * Waves + wave) * Sub;
    VopdProducer<Site, Waves> producer;
    producer.init(site, g, wave, lane);
    const std::uint16_t* x_row[Sub];
#pragma unroll
    for (unsigned r = 0; r < Sub; ++r) x_row[r] = row_input(site, first + r);
    const auto load_rows = [&](unsigned (&w)[16], std::uint32_t k) {
        const std::uint32_t bytes = umin(8U * lane + 256U * k, width - 8U) * 2U;
#pragma unroll
        for (unsigned r = 0; r < Sub; ++r) {
            const uint4 q = *reinterpret_cast<const uint4*>(reinterpret_cast<const unsigned char*>(x_row[r]) + bytes);
            w[4U * r] = q.x; w[4U * r + 1U] = q.y; w[4U * r + 2U] = q.z; w[4U * r + 3U] = q.w;
        }
    };
    float v[128]{};
    unsigned wa[16], wb[16];
    load_rows(wa, 0U);
    typename VopdProducer<Site, Waves>::Fetch ring[2]{producer.fetch(umin(1U, chunks - 1U)), producer.fetch(umin(2U, chunks - 1U))};
    producer.produce(producer.fetch(0U), coef[0]);
    lds_fence_barrier();
    const auto step = [&](std::uint32_t k, unsigned (&now)[16], unsigned (&later)[16], auto odd) {
        if (k + 1U < chunks) load_rows(later, k + 1U);
        const auto current = ring[0];
        ring[0] = ring[1];
        ring[1] = producer.fetch(umin(k + 3U, chunks - 1U));
        if (k + 1U < chunks) producer.produce(current, coef[(k + 1U) & 1U]);
        if (8U * lane + 256U * k < width) {
            const auto lds = static_cast<unsigned>(reinterpret_cast<std::uintptr_t>(&coef[k & 1U][lane]));
            if constexpr (decltype(odd)::value) vopd_chunk_b1(v, now, lds); else vopd_chunk(v, now, lds);
        }
        lds_fence_barrier();
    };
    std::uint32_t k = 0;
    for (; k + 1U < chunks; k += 2U) { step(k, wa, wb, std::false_type{}); step(k + 1U, wb, wa, std::true_type{}); }
    if (k < chunks) step(k, wa, wb, std::false_type{});
    if constexpr (Site::rows > 8U) reduce_wide<Site, 128U, 8U>(v, lane); else reduce<Site, 128U, 8U>(v, lane);
    store_pairs<16U>(v[0], lane, [&](unsigned pair, float value) {
        if (g * Outs + pair / Sub < site.outputs) site.store(g * Outs + pair / Sub, first + pair % Sub, value);
    });
}

// Dense A8 (numerical profile dense-a8-v1): Q8_0 weights x per-32 int8 activations.
//
// Quantizer, row r, block b (x = the BF16/FP32 activations as FP32):
//   any x non-finite      -> delta = NaN (0x7fc00000), codes 0
//   amax = max |x|; tiny = 0 < amax < 2^-119
//   delta = amax == 0 || tiny ? 1 : amax / 127          (IEEE division, round to nearest even)
//   code  = tiny ? 0 : clamp(rint(x / delta), -127, 127)
// Projection, row r, output n, blocks in ascending order:
//   acc = +0
//   acc = fma(float(d_w[n][b]) * delta[r][b], float(sum_k q_w[n][b][k] * code[r][b][k]), acc)
//   y   = isnan(acc) ? 0x7fc00000 : acc, then the site's store
// The int32 block dot is exact in any order (|sum| < 2^20, so its float is exact); the FP32
// product, the FMA and the ascending block chain are the contract. Every form below runs
// that chain, so a row gets the same bits from R1 decode, a 2..8-row window (dot4 form)
// and a 16/32/64-row prompt window (iu8 WMMA form).
//
// One quantizer launch per projection launch writes the rows' codes (int8 [row][feature])
// and deltas (FP32 [block * pitch + row]) into a scratch slot of the site's family. Sites of at
// most kDenseA8Rows rows use the pitch-64 slots; prompt sites of up to kDenseA8WideRows rows (one
// launch per 256 prompt rows: each weight byte streams once per 256 rows, not per 64) use the
// pitch-256 slots. The site's coefficient/fma/add are not used; its store receives y.
constexpr std::uint32_t kDenseA8Rows = 64U, kDenseA8WideRows = 256U, kDenseA8MaxWidth = 8192U, kDenseA8Families = 4U;
constexpr std::uint64_t kDenseA8CodeBytes = std::uint64_t{kDenseA8Rows} * kDenseA8MaxWidth;
constexpr std::uint64_t kDenseA8SlotBytes = kDenseA8CodeBytes + std::uint64_t{kDenseA8MaxWidth / 32U} * kDenseA8Rows * 4U;
constexpr std::uint64_t kDenseA8WideCodeBytes = std::uint64_t{kDenseA8WideRows} * kDenseA8MaxWidth;
constexpr std::uint64_t kDenseA8WideSlotBytes = kDenseA8WideCodeBytes + std::uint64_t{kDenseA8MaxWidth / 32U} * kDenseA8WideRows * 4U;
template <unsigned Rows> constexpr std::uint32_t dense_a8_pitch() { return Rows <= kDenseA8Rows ? kDenseA8Rows : kDenseA8WideRows; }
constexpr std::uint32_t kDenseA8CanonicalNan = 0x7fc00000U;
// Widths the dense A8 forms accept: whole 64-feature block pairs (dword-aligned Q8 rows).
constexpr bool dense_a8_width(std::uint64_t width) { return width && width % 64U == 0U && width <= kDenseA8MaxWidth; }

struct DenseA8Slot { std::int8_t* codes; float* deltas; };
inline unsigned char* dense_a8_pool() {
    static unsigned char* const pool = [] {
        void* p = nullptr;
        if (hipMalloc(&p, kDenseA8SlotBytes * 2U * kDenseA8Families) != hipSuccess)
            throw std::runtime_error("q8_rows: dense A8 scratch allocation failed");
        return static_cast<unsigned char*>(p);
    }();
    return pool;
}
// Allocates the scratch before any launch (e.g. ahead of stream capture).
inline void dense_a8_reserve() { static_cast<void>(dense_a8_pool()); }
// Two slots per family, alternating per launch: projections are barrier-ordered, so a
// quantizer launched any-order overlaps at most the projection of the other slot.
inline DenseA8Slot dense_a8_slot(unsigned family) {
    static std::atomic<unsigned> turn[kDenseA8Families]{};
    family %= kDenseA8Families;
    const unsigned slot = family * 2U + (turn[family].fetch_add(1U, std::memory_order_relaxed) & 1U);
    auto* base = dense_a8_pool() + slot * kDenseA8SlotBytes;
    return {reinterpret_cast<std::int8_t*>(base), reinterpret_cast<float*>(base + kDenseA8CodeBytes)};
}
// Pitch-256 slots (prompt sites of more than 64 rows), allocated on first use.
inline DenseA8Slot dense_a8_wide_slot(unsigned family) {
    static unsigned char* const pool = [] {
        void* p = nullptr;
        if (hipMalloc(&p, kDenseA8WideSlotBytes * 2U * kDenseA8Families) != hipSuccess)
            throw std::runtime_error("q8_rows: dense A8 wide scratch allocation failed");
        return static_cast<unsigned char*>(p);
    }();
    static std::atomic<unsigned> turn[kDenseA8Families]{};
    family %= kDenseA8Families;
    const unsigned slot = family * 2U + (turn[family].fetch_add(1U, std::memory_order_relaxed) & 1U);
    auto* base = pool + slot * kDenseA8WideSlotBytes;
    return {reinterpret_cast<std::int8_t*>(base), reinterpret_cast<float*>(base + kDenseA8WideCodeBytes)};
}
template <class Site> constexpr unsigned a8_family_of() {
    if constexpr (requires { Site::a8_family; }) return Site::a8_family; else return 0U;
}

__device__ __forceinline__ float dense_a8_canonical(float v) { return v != v ? __uint_as_float(kDenseA8CanonicalNan) : v; }
template <unsigned D> __device__ __forceinline__ int lane_xor_int(int v) {
    return __builtin_amdgcn_mov_dpp(v, 0x160 | static_cast<int>(D), 0xf, 0xf, false);
}

// One block's quarter (8 features; the block's four lanes form one DPP quad): codes packed
// little-endian into two words, and the block's delta.
__device__ __forceinline__ void dense_a8_quantize8(const float (&x)[8], unsigned (&word)[2], float& delta) {
#pragma clang fp contract(off)
    int bad = 0;
    float amax = 0.0F;
#pragma unroll
    for (unsigned l = 0; l < 8U; ++l) { bad |= !isfinite(x[l]); amax = fmaxf(amax, fabsf(x[l])); }
    bad |= lane_xor_int<1U>(bad); bad |= lane_xor_int<2U>(bad);
    amax = fmaxf(amax, lane_xor<1U>(amax)); amax = fmaxf(amax, lane_xor<2U>(amax));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    delta = bad ? __uint_as_float(kDenseA8CanonicalNan) : amax == 0.0F || tiny ? 1.0F : __fdiv_rn(amax, 127.0F);
    word[0] = word[1] = 0U;
#pragma unroll
    for (unsigned l = 0; l < 8U; ++l) {
        const float rounded = bad || tiny ? 0.0F : rintf(__fdiv_rn(x[l], delta));
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        word[l / 4U] |= (static_cast<unsigned>(code) & 0xffU) << (8U * (l % 4U));
    }
}
// Four threads per block, 8 features each; rows * width / 8 is a multiple of four, so the
// grid tail drops whole quads.
template <class Site>
__global__ __launch_bounds__(kThreads) void dense_a8_quantize_kernel(Site site, std::int8_t* codes, float* deltas) {
    const std::uint32_t width = site.width, groups = width / 8U;
    const std::uint32_t item = blockIdx.x * kThreads + threadIdx.x;
    if (item >= Site::rows * groups) return;
    const unsigned r = item / groups;
    const std::uint32_t g = item % groups;
    float x[8];
    load8(row_input(site, r) + 8U * g, x);
    unsigned word[2];
    float delta;
    dense_a8_quantize8(x, word, delta);
    *reinterpret_cast<uint2*>(codes + static_cast<std::uint64_t>(r) * width + 8U * g) = make_uint2(word[0], word[1]);
    if (!(g & 3U)) deltas[(g / 4U) * dense_a8_pitch<Site::rows>() + r] = delta;
}

// Producer-side quantizer: the 32 lanes of a full wave hold one block's activations (lane =
// feature). Same values as dense_a8_quantize8; `code` and `delta` point at the slot entries of
// this lane's feature and of the block (only lane 0 writes the delta).
__device__ __forceinline__ void dense_a8_quantize_lane(float x, std::int8_t* code, float* delta) {
#pragma clang fp contract(off)
    const bool bad = __ballot(!isfinite(x)) != 0;
    float amax = fabsf(x);
#pragma unroll
    for (int d = 16; d != 0; d /= 2) amax = fmaxf(amax, __shfl_xor(amax, d, 32));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float scale = bad ? __uint_as_float(kDenseA8CanonicalNan) : amax == 0.0F || tiny ? 1.0F : __fdiv_rn(amax, 127.0F);
    const float rounded = bad || tiny ? 0.0F : rintf(__fdiv_rn(x, scale));
    *code = static_cast<std::int8_t>(static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded))));
    if ((threadIdx.x & 31U) == 0U) *delta = scale;
}

struct __attribute__((aligned(4))) DenseA8Dword4 { unsigned x, y, z, w; };
// Nine dwords from the block's dword-aligned start: an even block starts at its scale and
// its codes sit two bytes in; an odd block starts two bytes before its scale (the previous
// block's last codes) and its codes are dwords 1..8. Q8 rows hold an even block count, so
// neither reads outside the row.
template <unsigned Outs> struct DenseA8Weights {
    unsigned d[Outs][9];
    __device__ __forceinline__ void load(const unsigned char* const (&w)[Outs], std::uint32_t b) {
#pragma unroll
        for (unsigned o = 0; o < Outs; ++o) {
            const unsigned char* p = w[o] + (34U * b & ~3U);
            const auto lo = reinterpret_cast<const DenseA8Dword4*>(p)[0], hi = reinterpret_cast<const DenseA8Dword4*>(p)[1];
            d[o][0] = lo.x; d[o][1] = lo.y; d[o][2] = lo.z; d[o][3] = lo.w;
            d[o][4] = hi.x; d[o][5] = hi.y; d[o][6] = hi.z; d[o][7] = hi.w;
            d[o][8] = reinterpret_cast<const unsigned*>(p)[8];
        }
    }
};
// Program order of one wave's LDS accesses (the chain slots are wave-private).
__device__ __forceinline__ void wave_lds_order() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// 1..8 rows (R1 decode included). A wave owns Outs outputs at a time; lane l takes block
// 32s + l of every 32-block superchunk s (whole blocks: eight dot4 per row, no lane
// reduction) and writes (d_w * delta, float(dot)) of each (output, row) pair to its wave's
// LDS slots; lane p < Pairs then runs pair p's chain over the superchunk's blocks in
// ascending order. Weights stream two superchunks ahead through the wave's whole sequence
// of (output group, superchunk) steps.
//
// The grid is persistent (no more blocks than are resident, waves loop over output groups),
// so a kernel queued after it without a barrier can start alongside it. Staged (small rows x
// width): each block quantizes the rows into LDS once (codes [row][feature], deltas
// [block * Rows + row]); otherwise the rows come pre-quantized from the scratch slot.
template <class Site> constexpr unsigned dense_a8_outs_of() {
    if constexpr (requires { Site::dense_a8_outs; }) return Site::dense_a8_outs;
    else return Site::rows <= 2U ? 1U : 2U;
}
template <unsigned Rows> constexpr std::uint64_t dense_a8_stage_bytes(std::uint64_t width) { return Rows * width + Rows * (width / 32U) * 4U; }
template <class Site, unsigned Outs, bool Staged>
__global__ __launch_bounds__(kThreads) void project_rows_dense_a8_kernel(Site site, const std::int8_t* codes, const float* deltas) {
#pragma clang fp contract(off)
    constexpr unsigned Rows = Site::rows, Pairs = Outs * Rows, Depth = 2U;
    static_assert(Rows <= 8U && Pairs <= 16U);
    __shared__ float2 chain[kWaves][Pairs][33];
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x / 32U;
    const std::uint32_t width = site.width, blocks = width / 32U, supers = (blocks + 31U) / 32U;
    const std::uint32_t groups = (site.outputs + Outs - 1U) / Outs, first = blockIdx.x * kWaves + wave;
    const std::uint32_t step = gridDim.x * kWaves;
    const std::uint32_t steps = first < groups ? (groups - first + step - 1U) / step * supers : 0U;
    // Block 32s + l has the parity of l: byte selectors and scale half of the nine dwords.
    const unsigned odd = lane & 1U, select = odd ? 0x07060504U : 0x05040302U, shift = odd ? 16U : 0U;
    // Lanes past the last block repeat it; the chain never reads their slots.
    const auto fetch = [&](std::uint32_t t, DenseA8Weights<Outs>& into) {
        const unsigned char* w[Outs];
        group_rows(site.segments, first + t / supers * step, site.outputs, width, w);
        into.load(w, std::min<std::uint32_t>(32U * (t % supers) + lane, blocks - 1U));
    };
    DenseA8Weights<Outs> next[Depth];
#pragma unroll
    for (unsigned d = 0; d < Depth; ++d) if (d < steps) fetch(d, next[d]);
    auto* stage = reinterpret_cast<std::int8_t*>(q8_rows_lds);
    auto* stage_delta = reinterpret_cast<float*>(q8_rows_lds + Rows * width);
    if constexpr (Staged) {
        for (std::uint32_t item = threadIdx.x; item < Rows * (width / 8U); item += kThreads) {
            const unsigned r = item / (width / 8U);
            const std::uint32_t g = item % (width / 8U);
            float x[8];
            load8(row_input(site, r) + 8U * g, x);
            unsigned word[2];
            float delta;
            dense_a8_quantize8(x, word, delta);
            *reinterpret_cast<uint2*>(stage + r * width + 8U * g) = make_uint2(word[0], word[1]);
            if (!(g & 3U)) stage_delta[(g / 4U) * Rows + r] = delta;
        }
        __syncthreads();
    }
    float acc = 0.0F;
    std::uint32_t s = 0, g = first;
    for (std::uint32_t t = 0; t < steps; ++t) {
        const std::uint32_t b = min(32U * s + lane, blocks - 1U);
        int4 x[Rows][2];
        float delta[Rows];
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            const auto* a = reinterpret_cast<const int4*>((Staged ? stage : codes) + static_cast<std::uint64_t>(r) * width + 32U * b);
            x[r][0] = a[0]; x[r][1] = a[1];
            delta[r] = Staged ? stage_delta[b * Rows + r] : deltas[b * kDenseA8Rows + r];
        }
        const auto current = next[0];
#pragma unroll
        for (unsigned d = 1; d < Depth; ++d) next[d - 1U] = next[d];
        if (t + Depth < steps) fetch(t + Depth, next[Depth - 1U]);
        wave_lds_order();
#pragma unroll
        for (unsigned o = 0; o < Outs; ++o) {
            int q[8];
#pragma unroll
            for (unsigned k = 0; k < 8U; ++k) q[k] = static_cast<int>(__builtin_amdgcn_perm(current.d[o][k + 1U], current.d[o][k], select));
            const float dw = __half2float(__ushort_as_half(static_cast<unsigned short>((current.d[o][0] >> shift) & 0xffffU)));
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r) {
                const int a[8]{x[r][0].x, x[r][0].y, x[r][0].z, x[r][0].w, x[r][1].x, x[r][1].y, x[r][1].z, x[r][1].w};
                int dot = 0;
#pragma unroll
                for (unsigned k = 0; k < 8U; ++k) dot = __builtin_amdgcn_sudot4(true, a[k], true, q[k], dot, false);
                chain[wave][o * Rows + r][lane] = make_float2(dw * delta[r], static_cast<float>(dot));
            }
        }
        wave_lds_order();
        if (lane < Pairs) {
            const std::uint32_t count = min(32U, blocks - 32U * s);
#pragma unroll
            for (unsigned j = 0; j < 32U; ++j)
                if (j < count) { const float2 c = chain[wave][lane][j]; acc = __builtin_fmaf(c.x, c.y, acc); }
        }
        if (++s == supers) {
            if (lane < Pairs) {
                const std::uint32_t out = g * Outs + lane / Rows;
                if (out < site.outputs) site.store(out, lane % Rows, dense_a8_canonical(acc));
            }
            acc = 0.0F; s = 0; g += step;
        }
    }
}

// 16/32/64 rows. A wave owns TM x TN 16x16 tiles (rows x outputs); the 4 waves of a block
// share the row tiles and consecutive blocks share the output tiles. Per Q8_0 block pair
// (68 bytes, dword aligned) each lane loads 17 dwords of its output row; the even block's
// codes are realigned by two bytes, the odd block's are aligned. Per block: two
// v_wmma_i32_16x16x16_iu8 (K = 16 + 16) on a zero accumulator, then the chain step per D
// element. Layouts (wave32): A lane i%16 = row i, B lane i%16 = output i, D element e of
// lane l = (row 2e + l/16, output l%16); the upper half wave repeats A/B.
typedef int dense_a8_v4i __attribute__((ext_vector_type(4)));
typedef int dense_a8_v8i __attribute__((ext_vector_type(8)));
template <class Site, unsigned TM, unsigned TN>
__global__ __launch_bounds__(128) void project_rows_dense_a8_wmma_kernel(Site site, const std::int8_t* codes, const float* deltas) {
#pragma clang fp contract(off)
    constexpr unsigned Rows = Site::rows;
    static_assert(Rows % (16U * TM) == 0U && Rows <= kDenseA8WideRows);
    constexpr std::uint32_t pitch = dense_a8_pitch<Rows>();
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x / 32U, i = lane & 15U, h = lane >> 4U;
    constexpr std::uint32_t row_tiles = Rows / (16U * TM);
    const std::uint32_t width = site.width, outputs = site.outputs, pairs = width / 64U;
    const std::uint32_t m0 = blockIdx.x % row_tiles * 16U * TM, n0 = (blockIdx.x / row_tiles * 4U + wave) * 16U * TN;
    if (n0 >= outputs) return;
    const unsigned char* w[TN];
#pragma unroll
    for (unsigned t = 0; t < TN; ++t) {
        unsigned m; std::uint32_t row;
        w[t] = segment_row(site.segments, min(n0 + 16U * t + i, outputs - 1U), width, m, row);
    }
    // A lane i carries row 8 (i % 2) + i / 2 of its tile, so D element e of lane l (A row 2e + l/16)
    // is row 8 h + e: a lane's eight epilogue rows (and their deltas) are consecutive.
    const std::int8_t* a[TM];
#pragma unroll
    for (unsigned t = 0; t < TM; ++t) a[t] = codes + static_cast<std::uint64_t>(m0 + 16U * t + 8U * (i & 1U) + (i >> 1U)) * width;
    const std::uint32_t erow = m0 + 8U * h;
    float acc[TM][TN][8];
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
        for (unsigned u = 0; u < TN; ++u)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) acc[t][u][e] = 0.0F;
    // The block dot accumulates onto 0x4B400000 (1.5 * 2^23): |dot| <= 32 * 127 * 127 < 2^22, so the
    // sum's bits are the float 1.5 * 2^23 + dot and subtracting 1.5 * 2^23 gives float(dot) exactly
    // (one v_sub_f32, which dual-issues; v_cvt_f32_i32 does not).
    constexpr int kMagic = 0x4B400000;
    constexpr float kMagicF = 12582912.0F;
    const dense_a8_v8i magic{kMagic, kMagic, kMagic, kMagic, kMagic, kMagic, kMagic, kMagic};
    // Weights stream one block pair ahead (the next pair's 17 dwords per output tile are in flight
    // during this pair's WMMAs and epilogue); the last pair reloads itself.
    unsigned next[TN][17];
    const auto fetch = [&](std::uint32_t pair) {
#pragma unroll
        for (unsigned u = 0; u < TN; ++u) {
            const auto* q4 = reinterpret_cast<const DenseA8Dword4*>(w[u] + 68U * pair);
            const DenseA8Dword4 c0 = q4[0], c1 = q4[1], c2 = q4[2], c3 = q4[3];
            const unsigned last = reinterpret_cast<const unsigned*>(w[u] + 68U * pair)[16];
            const unsigned q[17]{c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w, c2.x, c2.y, c2.z, c2.w, c3.x, c3.y, c3.z, c3.w, last};
#pragma unroll
            for (unsigned k = 0; k < 17U; ++k) next[u][k] = q[k];
        }
    };
    // 4 x 2 tiles have no VGPRs left for a second set (256 with spills): they load in place.
    constexpr bool ahead = TM * TN <= 4U;
    if constexpr (ahead) fetch(0U);
    for (std::uint32_t p = 0; p < pairs; ++p) {
        dense_a8_v4i wb[2][TN][2];
        float dw[2][TN];
        unsigned cur[TN][17];
        if constexpr (!ahead) fetch(p);
#pragma unroll
        for (unsigned u = 0; u < TN; ++u)
#pragma unroll
            for (unsigned k = 0; k < 17U; ++k) cur[u][k] = next[u][k];
        if constexpr (ahead) fetch(p + 1U < pairs ? p + 1U : p);
#pragma unroll
        for (unsigned u = 0; u < TN; ++u) {
            const unsigned (&q)[17] = cur[u];
            dw[0][u] = static_cast<float>(__ushort_as_half(static_cast<unsigned short>(q[0] & 0xffffU)));
            dw[1][u] = static_cast<float>(__ushort_as_half(static_cast<unsigned short>(q[8] >> 16U)));
            // Opaque FP32 scales: the products stay v_mul_f32 (dual-issuable) instead of folding
            // the f16 conversion into v_fma_mix_f32 (same value: the conversion is exact).
            asm volatile("" : "+v"(dw[0][u]), "+v"(dw[1][u]));
#pragma unroll
            for (unsigned k = 0; k < 4U; ++k) {
                wb[0][u][0][k] = static_cast<int>(__builtin_amdgcn_alignbyte(q[k + 1U], q[k], 2U));
                wb[0][u][1][k] = static_cast<int>(__builtin_amdgcn_alignbyte(q[k + 5U], q[k + 4U], 2U));
                wb[1][u][0][k] = static_cast<int>(q[9U + k]);
                wb[1][u][1][k] = static_cast<int>(q[13U + k]);
            }
        }
        dense_a8_v4i ab[2][TM][2];
#pragma unroll
        for (unsigned t = 0; t < TM; ++t) {
            const auto* x = reinterpret_cast<const dense_a8_v4i*>(a[t] + 64U * p);
            ab[0][t][0] = x[0]; ab[0][t][1] = x[1]; ab[1][t][0] = x[2]; ab[1][t][1] = x[3];
        }
#pragma unroll
        for (unsigned s = 0; s < 2U; ++s) {
            const float* d = deltas + ((2U * p + s) * pitch + erow);
            float da[TM][8];
#pragma unroll
            for (unsigned t = 0; t < TM; ++t) {
                const float4 lo = reinterpret_cast<const float4*>(d + 16U * t)[0], hi = reinterpret_cast<const float4*>(d + 16U * t)[1];
                da[t][0] = lo.x; da[t][1] = lo.y; da[t][2] = lo.z; da[t][3] = lo.w;
                da[t][4] = hi.x; da[t][5] = hi.y; da[t][6] = hi.z; da[t][7] = hi.w;
            }
            // All tiles' first K halves, then all second halves, then the epilogues: no WMMA waits
            // on the one just issued, and no epilogue on a WMMA still in flight.
            dense_a8_v8i c[TM][TN];
#pragma unroll
            for (unsigned t = 0; t < TM; ++t)
#pragma unroll
                for (unsigned u = 0; u < TN; ++u)
                    c[t][u] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ab[s][t][0], true, wb[s][u][0], magic, false);
#pragma unroll
            for (unsigned t = 0; t < TM; ++t)
#pragma unroll
                for (unsigned u = 0; u < TN; ++u)
                    c[t][u] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ab[s][t][1], true, wb[s][u][1], c[t][u], false);
#pragma unroll
            for (unsigned t = 0; t < TM; ++t)
#pragma unroll
                for (unsigned u = 0; u < TN; ++u)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e)
                        acc[t][u][e] = __builtin_fmaf(dw[s][u] * da[t][e], __int_as_float(c[t][u][e]) - kMagicF, acc[t][u][e]);
        }
    }
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
        for (unsigned u = 0; u < TN; ++u) {
            const std::uint32_t n = n0 + 16U * u + i;
            if (n < outputs)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) site.store(n, m0 + 16U * t + 8U * h + e, dense_a8_canonical(acc[t][u][e]));
        }
}

// Staged rows for the 4 x 2 WMMA form below: a block's 64 rows (codes of one block pair, plane-major
// [K quarter q][A slot][16 bytes], A slot 16 t + i = the row lane i of tile t carries, so a tile's
// operand is one conflict-free ds_load_b128 per K half) and their deltas, double-buffered.
struct DenseA8Stage {
    dense_a8_v4i codes[2][4][64];
    float deltas[2][2][64];
};
// 64 rows x 32 outputs per wave (4 x 2 tiles), rows staged in LDS. The Waves waves of a block share the
// row tiles: each block pair's codes (64 rows x 64 bytes) and deltas (2 x 64) are loaded from memory
// once per block, not once per wave (with per-wave loads the operand reads kept the memory and LDS
// paths as busy as the WMMAs), and each row tile's codes and deltas are read from LDS once for two
// output tiles. The row tiles run one after another (scheduling barriers keep the compiler from
// hoisting every tile's operands, which spills): per row tile, 2 x 2 WMMAs, then its 2 x 8 chain steps.
// Weights stream one block pair ahead per wave. Same WMMAs, epilogue expressions and block order as
// project_rows_dense_a8_wmma_kernel; one barrier per block pair.
template <class Site, unsigned Waves>
__global__ __launch_bounds__(32U * Waves) void project_rows_dense_a8_wmma_lds2_kernel(Site site, const std::int8_t* codes,
    const float* deltas) {
#pragma clang fp contract(off)
    constexpr unsigned Rows = Site::rows, TM = 4U, TN = 2U, Threads = 32U * Waves;
    static_assert(Rows % 64U == 0U && Rows <= kDenseA8WideRows && Threads >= 128U);
    constexpr std::uint32_t pitch = dense_a8_pitch<Rows>();
    __shared__ DenseA8Stage stage;
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x / 32U, i = lane & 15U, h = lane >> 4U;
    constexpr std::uint32_t row_tiles = Rows / 64U;
    const std::uint32_t width = site.width, outputs = site.outputs, pairs = width / 64U;
    const std::uint32_t m0 = blockIdx.x % row_tiles * 64U, n0 = (blockIdx.x / row_tiles * Waves + wave) * 16U * TN;
    const bool active = n0 < outputs;
    const unsigned char* w[TN];
#pragma unroll
    for (unsigned t = 0; t < TN; ++t) {
        unsigned m; std::uint32_t row;
        w[t] = segment_row(site.segments, min(min(n0, outputs - 1U) + 16U * t + i, outputs - 1U), width, m, row);
    }
    const unsigned slot = (threadIdx.x / 2U) % 64U, half = threadIdx.x % 2U;
    const std::uint32_t slot_row = m0 + 16U * (slot / 16U) + 8U * (slot % 2U) + (slot % 16U) / 2U;
    const auto* code_src = codes + static_cast<std::uint64_t>(slot_row) * width + 32U * half;
    const unsigned dq = threadIdx.x % 32U;
    const float* delta_src = deltas + (dq / 16U) * pitch + m0 + 4U * (dq % 16U);
    const bool stage_codes = threadIdx.x < 128U, stage_deltas = Waves == 4U ? threadIdx.x < 32U : threadIdx.x >= 128U && threadIdx.x < 160U;
    dense_a8_v4i sc[2];
    float4 sd;
    const auto load = [&](std::uint32_t p) {
        if (stage_codes) {
            const auto* x = reinterpret_cast<const dense_a8_v4i*>(code_src + 64U * p);
            sc[0] = x[0]; sc[1] = x[1];
        }
        if (stage_deltas) sd = *reinterpret_cast<const float4*>(delta_src + 2U * p * pitch);
    };
    const auto store = [&](unsigned buffer) {
        if (stage_codes) { stage.codes[buffer][2U * half][slot] = sc[0]; stage.codes[buffer][2U * half + 1U][slot] = sc[1]; }
        if (stage_deltas) *reinterpret_cast<float4*>(&stage.deltas[buffer][dq / 16U][4U * (dq % 16U)]) = sd;
    };
    float acc[TM][TN][8];
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
        for (unsigned u = 0; u < TN; ++u)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) acc[t][u][e] = 0.0F;
    constexpr int kMagic = 0x4B400000;
    constexpr float kMagicF = 12582912.0F;
    const dense_a8_v8i magic{kMagic, kMagic, kMagic, kMagic, kMagic, kMagic, kMagic, kMagic};
    unsigned next[TN][17];
    const auto fetch = [&](std::uint32_t pair) {
#pragma unroll
        for (unsigned u = 0; u < TN; ++u) {
            const auto* q4 = reinterpret_cast<const DenseA8Dword4*>(w[u] + 68U * pair);
            const DenseA8Dword4 c0 = q4[0], c1 = q4[1], c2 = q4[2], c3 = q4[3];
            const unsigned last = reinterpret_cast<const unsigned*>(w[u] + 68U * pair)[16];
            const unsigned q[17]{c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w, c2.x, c2.y, c2.z, c2.w, c3.x, c3.y, c3.z, c3.w, last};
#pragma unroll
            for (unsigned k = 0; k < 17U; ++k) next[u][k] = q[k];
        }
    };
    load(0U);
    store(0U);
    if (active) fetch(0U);
    if (pairs > 1U) load(1U);
    __syncthreads();
    for (std::uint32_t p = 0; p < pairs; ++p) {
        const unsigned buffer = p & 1U;
        if (active) {
            dense_a8_v4i wb[2][TN][2];
            float dw[2][TN];
#pragma unroll
            for (unsigned u = 0; u < TN; ++u) {
                const unsigned (&q)[17] = next[u];
                dw[0][u] = static_cast<float>(__ushort_as_half(static_cast<unsigned short>(q[0] & 0xffffU)));
                dw[1][u] = static_cast<float>(__ushort_as_half(static_cast<unsigned short>(q[8] >> 16U)));
                asm volatile("" : "+v"(dw[0][u]), "+v"(dw[1][u]));
#pragma unroll
                for (unsigned k = 0; k < 4U; ++k) {
                    wb[0][u][0][k] = static_cast<int>(__builtin_amdgcn_alignbyte(q[k + 1U], q[k], 2U));
                    wb[0][u][1][k] = static_cast<int>(__builtin_amdgcn_alignbyte(q[k + 5U], q[k + 4U], 2U));
                    wb[1][u][0][k] = static_cast<int>(q[9U + k]);
                    wb[1][u][1][k] = static_cast<int>(q[13U + k]);
                }
            }
            fetch(p + 1U < pairs ? p + 1U : p);
#pragma unroll
            for (unsigned s = 0; s < 2U; ++s)
#pragma unroll
                for (unsigned t = 0; t < TM; ++t) {
                    const dense_a8_v4i a0 = stage.codes[buffer][2U * s][16U * t + i], a1 = stage.codes[buffer][2U * s + 1U][16U * t + i];
                    const float4 lo = *reinterpret_cast<const float4*>(&stage.deltas[buffer][s][16U * t + 8U * h]);
                    const float4 hi = *reinterpret_cast<const float4*>(&stage.deltas[buffer][s][16U * t + 8U * h + 4U]);
                    const float da[8]{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                    dense_a8_v8i c[TN];
#pragma unroll
                    for (unsigned u = 0; u < TN; ++u) c[u] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, wb[s][u][0], magic, false);
#pragma unroll
                    for (unsigned u = 0; u < TN; ++u) c[u] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, wb[s][u][1], c[u], false);
#pragma unroll
                    for (unsigned u = 0; u < TN; ++u)
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e)
                            acc[t][u][e] = __builtin_fmaf(dw[s][u] * da[e], __int_as_float(c[u][e]) - kMagicF, acc[t][u][e]);
                    __builtin_amdgcn_sched_barrier(0);
                }
        }
        if (p + 1U < pairs) {
            store(buffer ^ 1U);
            if (p + 2U < pairs) load(p + 2U);
        }
        __syncthreads();
    }
    if (!active) return;
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
        for (unsigned u = 0; u < TN; ++u) {
            const std::uint32_t n = n0 + 16U * u + i;
            if (n < outputs)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) site.store(n, m0 + 16U * t + 8U * h + e, dense_a8_canonical(acc[t][u][e]));
        }
}

// Persistent grids never exceed the blocks that are resident at once.
template <auto Kernel, unsigned Threads> unsigned resident_blocks(std::uint32_t lds_bytes) {
    static std::atomic<std::uint64_t> cached{0};
    const auto hit = cached.load(std::memory_order_relaxed);
    if (hit && hit >> 32U == lds_bytes) return static_cast<unsigned>(hit);
    int per_unit = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_unit, Kernel, static_cast<int>(Threads), lds_bytes) != hipSuccess ||
        per_unit <= 0)
        per_unit = 1;
    const auto blocks = static_cast<unsigned>(per_unit) * compute_units();
    cached.store(static_cast<std::uint64_t>(lds_bytes) << 32U | blocks, std::memory_order_relaxed);
    return blocks;
}

// Rows > 8 tuning overrides: STRIX_Q8_ROWS_WIDE_KERNEL = 0 (one output group per wave,
// project_wide_rows_kernel), 1 (shared coefficients, C++ FMA chains) or 2 (default:
// the VOPD kernels where the site allows them); STRIX_Q8_ROWS_WIDE_SUB = 2/4/8/16 rows
// per wave for kernels 0 and 1; STRIX_Q8_ROWS_WIDE_STAGE=1 the LDS-staged kernel.
inline unsigned env_unsigned(const char* name, unsigned fallback) {
    const char* v = std::getenv(name);
    return v ? static_cast<unsigned>(std::atoi(v)) : fallback;
}
inline unsigned wide_sub() { static const unsigned sub = env_unsigned("STRIX_Q8_ROWS_WIDE_SUB", 0U); return sub; }
inline unsigned wide_kernel() { static const unsigned kernel = env_unsigned("STRIX_Q8_ROWS_WIDE_KERNEL", 2U); return kernel; }
inline bool wide_stage() { static const bool stage = env_unsigned("STRIX_Q8_ROWS_WIDE_STAGE", 0U) == 1U; return stage; }
inline unsigned narrow_vopd() { static const unsigned vopd = env_unsigned("STRIX_Q8_ROWS_VOPD", 1U); return vopd; }
// STRIX_Q8_ROWS_WIDE_FP32_MIN: narrowest FP32 input the wide (T >= 16) VOPD coefficient kernel takes.
// 512 includes the shared down projection (640 wide): 130 -> 100 us per T=64 call after reduce_wide.
inline unsigned wide_fp32_min_width() { static const unsigned w = env_unsigned("STRIX_Q8_ROWS_WIDE_FP32_MIN", 512U); return w; }
inline unsigned narrow_fp32_min_width() { static const unsigned w = env_unsigned("STRIX_Q8_ROWS_VOPD_FP32_MIN", 1024U); return w; }
// Rows <= 8 whose quantized activations and chain slots fit one block's LDS quantize them in
// the projection kernel (no quantizer launch; measured faster in 8-row MTP windows even at one
// block per CU). STRIX_DENSE_A8_STAGE=0 never stages, =1 stages only up to kDenseA8StageBytes.
inline unsigned dense_a8_stage_mode() {
    static const unsigned mode = [] { const char* v = std::getenv("STRIX_DENSE_A8_STAGE"); return v ? static_cast<unsigned>(std::atoi(v)) : 2U; }();
    return mode;
}
constexpr std::uint64_t kDenseA8StageBytes = 24U * 1024U;
// The WMMA projection of 16..256 rows from a quantized slot (barrier-ordered).
template <class Site> void launch_dense_a8_projection(const Site& site, const DenseA8Slot& slot, hipStream_t stream) {
    static_assert(Site::rows % 16U == 0U && Site::rows <= kDenseA8WideRows);
    // Tiles measured in prefill on gfx1151 (T = 1032, 64- and 256-row launches): 4 x 1 is the
    // fastest for every site (GDN/QSA in and out, shared gate/up and down).
    const auto tiled = [&]<unsigned TM, unsigned TN>() {
        constexpr unsigned row_tiles = Site::rows / (16U * TM);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(project_rows_dense_a8_wmma_kernel<Site, TM, TN>), dim3(row_tiles * ((site.outputs + 64U * TN - 1U) / (64U * TN))),
            dim3(128U), 0, stream, site, slot.codes, slot.deltas);
    };
    constexpr unsigned TM = Site::rows >= 64U ? 4U : Site::rows / 16U;
    // STRIX_DENSE_A8_TILE=11/12/21/22/41/42 (TM TN) forces one tile shape for every site (tuning).
    static const unsigned forced = env_unsigned("STRIX_DENSE_A8_TILE", 0U);
    // Launches of 64k rows take the LDS-staged 4 x 2 form when it still has at least 40 blocks of 8
    // waves (one per CU): GDN/QSA input and the 256-row output launches; narrower launches (shared
    // expert gate/up, 64-row QSA output) keep the 4 x 1 form.
    // STRIX_DENSE_A8_LDS=0 never stages, =2 stages every 64k-row launch; STRIX_DENSE_A8_LDS_WAVES=4|8.
    static const unsigned lds = env_unsigned("STRIX_DENSE_A8_LDS", 1U);
    if constexpr (Site::rows % 64U == 0U) {
        const auto lds_blocks = Site::rows / 64U * ((site.outputs + 255U) / 256U);
        if (!forced && (lds == 2U || (lds == 1U && lds_blocks >= 40U))) {
            static const unsigned lds_waves = env_unsigned("STRIX_DENSE_A8_LDS_WAVES", 8U);
            const auto staged = [&]<unsigned Waves>() {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(project_rows_dense_a8_wmma_lds2_kernel<Site, Waves>),
                    dim3(Site::rows / 64U * ((site.outputs + 32U * Waves - 1U) / (32U * Waves))), dim3(32U * Waves), 0, stream,
                    site, slot.codes, slot.deltas);
            };
            if (lds_waves == 4U) staged.template operator()<4U>(); else staged.template operator()<8U>();
            return;
        }
    }
    const auto pick = [&]<unsigned M, unsigned N>() {
        if constexpr (Site::rows % (16U * M) == 0U) { tiled.template operator()<M, N>(); return true; } else return false;
    };
    if (forced == 11U && pick.template operator()<1U, 1U>()) return;
    if (forced == 12U && pick.template operator()<1U, 2U>()) return;
    if (forced == 21U && pick.template operator()<2U, 1U>()) return;
    if (forced == 22U && pick.template operator()<2U, 2U>()) return;
    if (forced == 41U && pick.template operator()<4U, 1U>()) return;
    if (forced == 42U && pick.template operator()<4U, 2U>()) return;
    tiled.template operator()<TM, 1U>();
}
// Dense A8. Rows <= 8 whose activations fit LDS: one persistent launch that quantizes them
// per block (the 6144-wide 3..8-row inputs do not fit; windows quantize those in their
// producers, launch_dense_a8_quantized). Otherwise one quantizer launch into the family's
// scratch slot (any-order when the caller allows it; it writes only the slot no queued
// projection reads) and one barrier-ordered projection launch (dot4 rows <= 8, WMMA
// 16/32/64).
template <class Site> void launch_dense_a8(const Site& site, hipStream_t stream, bool any_order) {
    static_assert(Site::rows <= 8U || (Site::rows % 16U == 0U && Site::rows <= kDenseA8WideRows));
    if (!dense_a8_width(site.width))
        throw std::invalid_argument("q8_rows: dense A8 requires a width that is a positive multiple of 64, at most 8192");
    for (unsigned m = 0; m < site.segments.count; ++m)
        if (reinterpret_cast<std::uintptr_t>(site.segments.weights[m]) % 4U)
            throw std::invalid_argument("q8_rows: dense A8 requires dword-aligned Q8 weights");
    const auto run = [&](auto kernel, unsigned blocks, unsigned threads, std::uint32_t bytes, bool unordered, const std::int8_t* codes,
                         const float* deltas) {
        if (unordered)
            hipExtLaunchKernelGGL(kernel, dim3(blocks), dim3(threads), bytes, stream, nullptr, nullptr, hipExtAnyOrderLaunch, site,
                codes, deltas);
        else hipLaunchKernelGGL(kernel, dim3(blocks), dim3(threads), bytes, stream, site, codes, deltas);
    };
    if constexpr (Site::rows <= 8U) {
        constexpr unsigned Outs = dense_a8_outs_of<Site>();
        const unsigned groups = (site.outputs + Outs - 1U) / Outs;
        const auto stage = dense_a8_stage_bytes<Site::rows>(site.width);
        const auto chain = static_cast<std::uint64_t>(kWaves) * Outs * Site::rows * 33U * 8U;
        const unsigned mode = dense_a8_stage_mode();
        if (mode && stage + chain <= (mode == 2U ? kLdsBytes : kDenseA8StageBytes)) {
            constexpr auto kernel = project_rows_dense_a8_kernel<Site, Outs, true>;
            const auto bytes = static_cast<std::uint32_t>(stage);
            run(kernel, std::min((groups + kWaves - 1U) / kWaves, resident_blocks<kernel, kThreads>(bytes)), kThreads, bytes, any_order,
                nullptr, nullptr);
            return;
        }
    }
    const auto slot = Site::rows > kDenseA8Rows ? dense_a8_wide_slot(a8_family_of<Site>()) : dense_a8_slot(a8_family_of<Site>());
    const auto items = Site::rows * (site.width / 8U);
    const auto quantizer_blocks = dim3((items + kThreads - 1U) / kThreads);
    if (any_order)
        hipExtLaunchKernelGGL(dense_a8_quantize_kernel<Site>, quantizer_blocks, dim3(kThreads), 0, stream, nullptr, nullptr,
            hipExtAnyOrderLaunch, site, slot.codes, slot.deltas);
    else hipLaunchKernelGGL(dense_a8_quantize_kernel<Site>, quantizer_blocks, dim3(kThreads), 0, stream, site, slot.codes, slot.deltas);
    if constexpr (Site::rows <= 8U) {
        constexpr unsigned Outs = dense_a8_outs_of<Site>();
        const unsigned groups = (site.outputs + Outs - 1U) / Outs;
        constexpr auto kernel = project_rows_dense_a8_kernel<Site, Outs, false>;
        run(kernel, std::min((groups + kWaves - 1U) / kWaves, resident_blocks<kernel, kThreads>(0U)), kThreads, 0U, false, slot.codes,
            slot.deltas);
    } else launch_dense_a8_projection(site, slot, stream);
}

// Projection only, from a slot its producer filled (dense_a8_quantize_lane): codes
// [row][width], deltas [block * kDenseA8Rows + row] for rows 0..Site::rows-1.
template <class Site> void launch_dense_a8_quantized(const Site& site, const DenseA8Slot& slot, hipStream_t stream) {
    static_assert(Site::rows <= 8U);
    if (!dense_a8_width(site.width))
        throw std::invalid_argument("q8_rows: dense A8 requires a width that is a positive multiple of 64, at most 8192");
    for (unsigned m = 0; m < site.segments.count; ++m)
        if (reinterpret_cast<std::uintptr_t>(site.segments.weights[m]) % 4U)
            throw std::invalid_argument("q8_rows: dense A8 requires dword-aligned Q8 weights");
    constexpr unsigned Outs = dense_a8_outs_of<Site>();
    constexpr auto kernel = project_rows_dense_a8_kernel<Site, Outs, false>;
    const unsigned groups = (site.outputs + Outs - 1U) / Outs;
    hipLaunchKernelGGL(kernel, dim3(std::min((groups + kWaves - 1U) / kWaves, resident_blocks<kernel, kThreads>(0U))), dim3(kThreads), 0,
        stream, site, slot.codes, slot.deltas);
}

// The caller validated width % 32 == 0 and 16-byte aligned rows. any_order drops the
// packet's barrier bit: the caller guarantees the projection neither reads nor writes what
// the previously queued kernel touches, so it may start while that kernel still runs.
// a8 selects the dense A8 profile arithmetic (launch_dense_a8).
template <class Site> void launch(const Site& site, bool a8, hipStream_t stream, bool any_order = false) {
    if (a8) { launch_dense_a8(site, stream, any_order); return; }
    const auto start = [&](auto kernel, unsigned blocks, unsigned threads, std::uint32_t bytes) {
        if (any_order)
            hipExtLaunchKernelGGL(kernel, dim3(blocks), dim3(threads), bytes, stream, nullptr, nullptr, hipExtAnyOrderLaunch, site);
        else hipLaunchKernelGGL(kernel, dim3(blocks), dim3(threads), bytes, stream, site);
    };
    const auto groups_for = [&](unsigned outs, unsigned waves) { return ((site.outputs + outs - 1U) / outs + waves - 1U) / waves; };
    // Verify windows of 4 and 8 rows: the 4-output x 4-row VOPD kernels with 1 or 2 waves per
    // block. Default: FP32 inputs of width >= 1024 (GDN and QSA output projections), which are
    // compute-bound at T = 4/8. BF16 inputs are at DRAM bandwidth there and the staged
    // project_rows_kernel overlaps better with the window's concurrent kernels in decode, so
    // they take VOPD only with STRIX_Q8_ROWS_VOPD=2. STRIX_Q8_ROWS_VOPD=0 disables both; the
    // A8 path is unaffected.
    if constexpr (fused_fma_of<Site>() && (Site::rows == 4U || Site::rows == 8U)) {
        if (!a8 && narrow_vopd() != 0U) {
            constexpr unsigned waves = Site::rows / 4U;
            const unsigned blocks = (site.outputs + 3U) / 4U;
            if constexpr (sizeof(typename Site::Input) == 2U) {
                if (narrow_vopd() == 2U) {
                    constexpr auto kernel = project_wide_rows_vopd_bf16_kernel<Site>;
                    start(kernel, blocks, 32U * waves, 0U);
                    return;
                }
            } else if (site.width >= narrow_fp32_min_width()) {
                constexpr auto kernel = project_wide_rows_coef_kernel<Site, 4U, 4U, 4U, true>;
                start(kernel, blocks, 32U * waves, 0U);
                return;
            }
        }
    }
    if constexpr (Site::rows > 8U) {
        // Default: the VOPD kernels (4 outputs x 4 rows per wave, 4-wave blocks), FP32 inputs from
        // wide_fp32_min_width() up; narrower ones keep the per-wave kernel.
        if constexpr (fused_fma_of<Site>() && Site::rows % 16U == 0U) {
            if (wide_kernel() == 2U && !wide_stage()) {
                if constexpr (sizeof(typename Site::Input) == 2U) {
                    constexpr auto kernel = project_wide_rows_vopd_bf16_kernel<Site>;
                    start(kernel, (site.outputs + 3U) / 4U * (Site::rows / 16U), 128U, 0U);
                    return;
                } else if (site.width >= wide_fp32_min_width()) {
                    constexpr auto kernel = project_wide_rows_coef_kernel<Site, 4U, 4U, 4U, true>;
                    start(kernel, (site.outputs + 3U) / 4U * (Site::rows / 16U), 128U, 0U);
                    return;
                }
            }
        }
        const auto wide = [&]<unsigned Sub>() {
            constexpr unsigned Outs = 16U / Sub;
            if (wide_stage()) {
                constexpr auto kernel = project_wide_rows_staged_kernel<Site, Outs, Sub>;
                start(kernel, groups_for(Outs, kWaves) * (Site::rows / Sub), kThreads, 0U);
            } else if (wide_kernel() == 1U) {
                constexpr unsigned waves = coef_waves<Site, Sub>();
                constexpr auto kernel = project_wide_rows_coef_kernel<Site, Outs, Sub, kWaves, false>;
                start(kernel, (site.outputs + Outs - 1U) / Outs * (Site::rows / Sub / waves), 32U * waves, 0U);
            } else {
                constexpr auto kernel = project_wide_rows_kernel<Site, Outs, Sub>;
                start(kernel, groups_for(Outs, kWaves) * (Site::rows / Sub), kThreads, 0U);
            }
        };
        switch (wide_sub()) {
        case 2U: wide.template operator()<2U>(); break;
        case 4U: wide.template operator()<4U>(); break;
        case 8U: wide.template operator()<8U>(); break;
        case 16U: wide.template operator()<16U>(); break;
        default: wide.template operator()<wide_sub_of<Site>()>(); break;
        }
    } else if (const auto bytes = staged_bytes(site)) {
        constexpr unsigned Outs = outs_of<Site, true>();
        constexpr auto kernel = project_rows_kernel<Site, Outs, true>;
        const auto blocks = std::min(groups_for(Outs, kWaves), resident_blocks<kernel, kThreads>(bytes));
        start(kernel, blocks, kThreads, bytes);
    } else {
        constexpr unsigned Outs = outs_of<Site, false>();
        constexpr auto kernel = project_rows_kernel<Site, Outs, false>;
        start(kernel, groups_for(Outs, kWaves), kThreads, 0U);
    }
}

} // namespace strix::q8_rows
