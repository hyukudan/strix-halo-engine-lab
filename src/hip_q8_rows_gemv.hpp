#pragma once

// T-row Q8_0 projections for known-prefix windows (T = 1, 2, 4, 8).
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
// T may be any count up to 8 (3/5/6/7 pad the tree to a power of two).
// A Site supplies Input (uint16_t BF16 or float), rows, input[], segments,
// width, outputs, the coefficient/fma/add arithmetic of its R1 kernel and
// store(out, row, value); it may override outs, a8_outs and prefetch. The
// including translation unit owns the arithmetic flags.

#include <hip/hip_fp16.h>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
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
template <class Site, bool Staged> constexpr unsigned prefetch_of() {
    if constexpr (requires { Site::prefetch; }) return Site::prefetch; else return narrow_bf16<Site, Staged>() ? 1U : 2U;
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
    if (a8) {
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
