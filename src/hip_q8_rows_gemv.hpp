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
// A8 path (opt-in, different numerics): activations are quantised per 32-element
// block to int8 with a float scale in LDS; each lane owns whole Q8 blocks (one
// dot4 chain per row) and outputs/rows are reduced with a fixed tree.
//
// T may be any count up to 8 (3/5/6/7 pad the tree to a power of two), or a
// multiple of the wide batch (16, 32, 64) above 8; the A8 path stops at 8.
// A Site supplies Input (uint16_t BF16 or float), rows, input[] (or
// row_input(r)), segments, width, outputs, the coefficient/fma/add arithmetic of
// its R1 kernel and store(out, row, value); it may override outs, a8_outs, sub
// and prefetch. The including translation unit owns the arithmetic flags.

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

constexpr unsigned kWaves = 8U, kThreads = 32U * kWaves, kA8Waves = 16U, kA8Threads = 32U * kA8Waves, kMaxSegments = 5U;
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
template <class Site> constexpr unsigned a8_outs_of() {
    if constexpr (requires { Site::a8_outs; }) return Site::a8_outs; else return Site::rows > 1U ? 4U : 2U;
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
template <unsigned Rows> constexpr std::uint64_t a8_bytes(std::uint64_t width) { return Rows * width + Rows * (width / 32U) * 4U; }

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
    reduce<Site, N, 8U>(v, lane);
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
    reduce<Site, N, 8U>(v, lane);
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
template <class Site, unsigned Outs, unsigned Sub, unsigned W, bool Vopd>
__global__ __launch_bounds__((32U * coef_waves<Site, Sub, W>())) void project_wide_rows_coef_kernel(Site site) {
    using Input = typename Site::Input;
    constexpr unsigned Rows = Site::rows, Batches = Rows / Sub, Waves = coef_waves<Site, Sub, W>(), Threads = 32U * Waves;
    constexpr unsigned Blocks = Batches / Waves, Pairs = Outs * Sub, N = pow2u(Pairs) * 8U, Per = Outs * 256U / Threads;
    static_assert(Rows > 8U && Rows % Sub == 0U && Batches % Waves == 0U && Pairs <= 32U);
    static_assert(Per == 2U || Per == 4U || Per == 8U, "each thread decodes 2, 4 or 8 coefficients per chunk");
    static_assert(!Vopd || (Outs == 4U && Sub == 4U && fused_fma_of<Site>()));
    __shared__ float4 coef[2][Outs * 64U];
    const unsigned lane = threadIdx.x & 31U, wave = uniform(threadIdx.x / 32U);
    const std::uint32_t width = site.width, chunks = (width + 255U) / 256U, full = width / 256U;
    const std::uint32_t g = blockIdx.x / Blocks;
    const unsigned first = ((blockIdx.x % Blocks) * Waves + wave) * Sub;
    CoefProducer<Site, Outs, Per> producer;
    producer.po = uniform(threadIdx.x * Per / 256U);
    producer.pf = threadIdx.x * Per % 256U;
    producer.width = width;
    {
        unsigned m; std::uint32_t row;
        producer.pw = segment_row(site.segments, min(g * Outs + producer.po, site.outputs - 1U), width, m, row);
    }
    const Input* x_row[Sub];
#pragma unroll
    for (unsigned r = 0; r < Sub; ++r) x_row[r] = row_input(site, first + r);
    float v[N]{};
    typename CoefProducer<Site, Outs, Per>::Fetch ring[2]{producer.fetch(umin(1U, chunks - 1U)), producer.fetch(umin(2U, chunks - 1U))};
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
    reduce<Site, N, 8U>(v, lane);
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
template <class Site>
__global__ __launch_bounds__(128) void project_wide_rows_vopd_bf16_kernel(Site site) {
    static_assert(sizeof(typename Site::Input) == 2U && Site::rows % 16U == 0U && fused_fma_of<Site>());
    constexpr unsigned Outs = 4U, Sub = 4U, Waves = 4U, Blocks = Site::rows / Sub / Waves, Per = 8U;
    __shared__ float4 coef[2][Outs * 64U];
    const unsigned lane = threadIdx.x & 31U, wave = uniform(threadIdx.x / 32U);
    const std::uint32_t width = site.width, chunks = (width + 255U) / 256U;
    const std::uint32_t g = blockIdx.x / Blocks;
    const unsigned first = ((blockIdx.x % Blocks) * Waves + wave) * Sub;
    // Producer: wave w decodes output w's 256 coefficients of a chunk, 8 per lane.
    CoefProducer<Site, Outs, Per> producer;
    producer.po = wave;
    producer.pf = lane * Per;
    producer.width = width;
    {
        unsigned m; std::uint32_t row;
        producer.pw = segment_row(site.segments, min(g * Outs + wave, site.outputs - 1U), width, m, row);
    }
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
    typename CoefProducer<Site, Outs, Per>::Fetch ring[2]{producer.fetch(umin(1U, chunks - 1U)), producer.fetch(umin(2U, chunks - 1U))};
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
    reduce<Site, 128U, 8U>(v, lane);
    store_pairs<16U>(v[0], lane, [&](unsigned pair, float value) {
        if (g * Outs + pair / Sub < site.outputs) site.store(g * Outs + pair / Sub, first + pair % Sub, value);
    });
}

struct PlainTree { __device__ static float add(float lower, float upper) { return lower + upper; } };

// Opt-in A8: per-row, per-32 block int8 activations with scale amax/127 (round to
// nearest even, clamped to +-127; non-finite values quantise to zero). One
// thread quantises one block into LDS; each lane then owns whole Q8 blocks.
template <unsigned Outs> struct A8Fetch {
    int codes[Outs][8];
    float scale[Outs];
    __device__ __forceinline__ void load(const unsigned char* const (&w)[Outs], std::uint32_t b) {
#pragma unroll
        for (unsigned o = 0; o < Outs; ++o) {
            const unsigned char* block = w[o] + b * 34U;
            scale[o] = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(block)));
            std::memcpy(codes[o], block + 2U, 32U);
        }
    }
};
template <class Site, unsigned Outs>
__global__ __launch_bounds__(kA8Threads) void project_rows_a8_kernel(Site site) {
    using Input = typename Site::Input;
    constexpr unsigned Rows = Site::rows, Pairs = Outs * Rows, N = pow2u(Pairs);
    const unsigned lane = threadIdx.x & 31U;
    const std::uint32_t width = site.width, blocks = width / 32U, groups = (site.outputs + Outs - 1U) / Outs;
    auto* codes = reinterpret_cast<std::int8_t*>(q8_rows_lds);
    auto* scales = reinterpret_cast<float*>(q8_rows_lds + Rows * width);
    for (std::uint32_t q = threadIdx.x; q < Rows * blocks; q += kA8Threads) {
        const Input* source = static_cast<const Input*>(site.input[q / blocks]) + (q % blocks) * 32U;
        float x[4][8];
        float amax = 0.0F;
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) {
            load8(source + 8U * i, x[i]);
#pragma unroll
            for (unsigned l = 0; l < 8U; ++l) {
                if (!isfinite(x[i][l])) x[i][l] = 0.0F;
                amax = fmaxf(amax, fabsf(x[i][l]));
            }
        }
        const float inverse = amax > 0.0F ? 127.0F / amax : 0.0F;
        int packed[8];
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k) {
            unsigned word = 0;
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) {
                const auto code = static_cast<int>(rintf(fminf(127.0F, fmaxf(-127.0F, x[k / 2U][(k % 2U) * 4U + j] * inverse))));
                word |= (static_cast<unsigned>(code) & 0xffU) << (8U * j);
            }
            packed[k] = static_cast<int>(word);
        }
        auto* target = reinterpret_cast<int4*>(codes + q * 32U);
        target[0] = int4{packed[0], packed[1], packed[2], packed[3]};
        target[1] = int4{packed[4], packed[5], packed[6], packed[7]};
        scales[q] = amax / 127.0F;
    }
    __syncthreads();
    for (std::uint32_t g = blockIdx.x * kA8Waves + threadIdx.x / 32U; g < groups; g += gridDim.x * kA8Waves) {
        const unsigned char* w[Outs];
        group_rows(site.segments, g, site.outputs, width, w);
        float v[N]{};
        A8Fetch<Outs> next;
        if (lane < blocks) next.load(w, lane);
        for (std::uint32_t b = lane; b < blocks; b += 32U) {
            const A8Fetch<Outs> current = next;
            if (b + 32U < blocks) next.load(w, b + 32U);
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r) {
                const auto* a = reinterpret_cast<const int4*>(codes + (r * blocks + b) * 32U);
                const int4 a0 = a[0], a1 = a[1];
                const int av[8]{a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                const float s = scales[r * blocks + b];
#pragma unroll
                for (unsigned o = 0; o < Outs; ++o) {
                    int dot = 0;
#pragma unroll
                    for (unsigned k = 0; k < 8U; ++k) dot = __builtin_amdgcn_sudot4(true, current.codes[o][k], true, av[k], dot, false);
                    v[o * Rows + r] = fmaf(current.scale[o] * s, static_cast<float>(dot), v[o * Rows + r]);
                }
            }
        }
        reduce<PlainTree, N, 1U>(v, lane);
        store_pairs<N>(v[0], lane, [&](unsigned pair, float value) {
            if (pair < Pairs && g * Outs + pair / Rows < site.outputs) site.store(g * Outs + pair / Rows, pair % Rows, value);
        });
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
// The caller validated width % 32 == 0, 16-byte aligned rows and, for A8,
// a8_bytes<rows>(width) <= kLdsBytes. any_order drops the packet's barrier bit: the
// caller guarantees the projection neither reads nor writes what the previously
// queued kernel touches, so it may start while that kernel still runs.
template <class Site> void launch(const Site& site, bool a8, hipStream_t stream, bool any_order = false) {
    const auto start = [&](auto kernel, unsigned blocks, unsigned threads, std::uint32_t bytes) {
        if (any_order)
            hipExtLaunchKernelGGL(kernel, dim3(blocks), dim3(threads), bytes, stream, nullptr, nullptr, hipExtAnyOrderLaunch, site);
        else hipLaunchKernelGGL(kernel, dim3(blocks), dim3(threads), bytes, stream, site);
    };
    const auto groups_for = [&](unsigned outs, unsigned waves) { return ((site.outputs + outs - 1U) / outs + waves - 1U) / waves; };
    if constexpr (Site::rows > 8U) {
        if (a8) throw std::invalid_argument("q8_rows: the A8 path supports at most 8 rows");
        // Default: the VOPD kernels (4 outputs x 4 rows per wave, 4-wave blocks). FP32 inputs
        // narrower than 4 chunks (shared down, 640) keep the per-wave kernel, which is faster there.
        if constexpr (fused_fma_of<Site>() && Site::rows % 16U == 0U) {
            if (wide_kernel() == 2U && !wide_stage()) {
                if constexpr (sizeof(typename Site::Input) == 2U) {
                    constexpr auto kernel = project_wide_rows_vopd_bf16_kernel<Site>;
                    start(kernel, (site.outputs + 3U) / 4U * (Site::rows / 16U), 128U, 0U);
                    return;
                } else if (site.width >= 1024U) {
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
        return;
    } else if (a8) {
        constexpr unsigned Outs = a8_outs_of<Site>();
        constexpr auto kernel = project_rows_a8_kernel<Site, Outs>;
        const auto bytes = static_cast<std::uint32_t>(a8_bytes<Site::rows>(site.width));
        const auto blocks = std::min(groups_for(Outs, kA8Waves), resident_blocks<kernel, kA8Threads>(bytes));
        start(kernel, blocks, kA8Threads, bytes);
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
