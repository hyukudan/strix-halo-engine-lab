#pragma once
// Named policy dense-iu4a16-f16-v1 of the shared FP16 dense core (f16_dense_tall.hpp): IU4 g64 weights (core/iu4.hpp
// tile layout, the strix::gguf::type_strix_iu4_g64 pack: packed tiles then FP16 scales [N][K/64]) x FP16 activations in
// ONE FP16 WMMA chain per output, y[r][o] = sum_k fp16(x[r][k]) * w16[o][k] (FP32 accumulation, k ascending, from +0).
//
// Class: FAST (not the bits of the exact W4A16 small-M core of dense_linear_iu4.hpp, which scales each K64 group sum).
// Arithmetic contract:
//   w16[o][k] = fp16_rne(float(code) * float(scale))   code = signed nibble in [-8, 7], scale = FP16 of (o, k / 64)
//                                                       (the FP32 product is exact: 4 x 11 significant bits)
//   x16[r][k] = fp16_rne(x[r][k])                       (the core's register conversion, lane = row)
// Weights are decoded by the CTA's stagers once per (CTA, stage of 64 k = one scale group) into LDS: an IU4 byte is read
// once per 16 * Waves rows (0.53 bytes per weight, a quarter of an FP16 mirror). No FP16 copy of the weights exists.
// The decode is the packed FP16 form of f16_dense_q8a8.hpp: 0x6400 | (code + 8) is the FP16 integer 1032 + code, the
// packed subtraction of 1032 is exact and one packed FP16 multiply rounds the exact product once (RNE), the single
// rounding of the FP32 path (subnormal, infinite and NaN scales included; host_w16 below is the reference form).
// Any engine with IU4 g64 dense weights binds it by filling View; nothing here names a model.
// Home: shared (backend core, HIP; arch: gfx11 WMMA through f16_dense_tall). Bound: qwen35 HIP (rows > 16).
// Limits (callers check with supported()): in % 128 == 0, n % 16 == 0, count <= n, rows * in * 4 < 2^32.
#include "dense_pair_map.hpp"
#include "f16_dense_q8a8.hpp"

namespace strix::core::f16_dense_tall::iu4a16 {
inline constexpr char kPolicyName[] = "dense-iu4a16-f16-v1";

struct View {
    const std::uint8_t* weight;  // packed tiles [in/16][n/16][16][8] bytes
    const std::uint16_t* scale;  // FP16 bits [n][in/64] (the pack stores them right after the tiles)
    const float* x;              // FP32 rows [rows][in]
    float* out;                  // out[r * out_stride + o]
    std::uint32_t in, n, count, rows, out_stride;
    std::uint32_t tile0;         // first column tile of the launch (set by enqueue)
    std::uint32_t accumulate;    // out += sum
};

namespace detail {
using GlobalU2 = const std::uint32_t __attribute__((ext_vector_type(2), address_space(1), aligned(8)));
using Half2 = _Float16 __attribute__((ext_vector_type(2)));
// The two w16 of byte `byte` of `biased` (= word ^ 0x88888888: code + 8 per nibble): low half = low nibble (even k).
__device__ __forceinline__ std::uint32_t w16_pair(std::uint32_t biased, unsigned byte, Half2 scale2) {
    const std::uint32_t t = biased >> (8U * byte);
    const std::uint32_t bits = (t & 0xFU) | ((t & 0xF0U) << 12U) | 0x64006400U;
    const _Float16 bias = static_cast<_Float16>(1032.0F);
    return __builtin_bit_cast(std::uint32_t, (__builtin_bit_cast(Half2, bits) - Half2{bias, bias}) * scale2);
}
}  // namespace detail

// Reference form of the contract's weight value (FP16 bits), for tests.
inline _Float16 host_w16(int code, std::uint16_t scale_bits) {
    return static_cast<_Float16>(static_cast<float>(code) * static_cast<float>(__builtin_bit_cast(_Float16, scale_bits)));
}

template <unsigned Waves> struct Cols {
    static constexpr unsigned kWaves = Waves;
    View v;
    __device__ __forceinline__ ColumnTile tile(unsigned i) const {
        const unsigned o = 16U * (v.tile0 + i);
        return {nullptr, v.out + o, v.out_stride, o < v.count};
    }
    // 64 weights of the stager's column = four 8-byte rows of four K16 tiles (one K64 scale group); columns past count
    // read the last one (never stored). unit % 4 == 0 (the core stages four k units at a time from 0).
    // Split for the core's schedule (f16_dense_tall.hpp ColsStageSplit): stage_load issues the loads, stage_decode does
    // the arithmetic; stage() is their composition (same bits wherever the decode runs).
    struct Raw {
        std::uint32_t w[8];
        std::uint32_t s;
    };
    // Loads of output column o of the pack (weight, scale) with n columns and `in` inputs (shared by the pair policy).
    __device__ __forceinline__ static Raw load(const std::uint8_t* weight, const std::uint16_t* scale, unsigned n, unsigned in,
                                               unsigned o, unsigned unit) {
        const std::size_t col_bytes = (std::size_t(o / 16U) * 16U + o % 16U) * 8U, kt_bytes = std::size_t(n) * 8U;
        Raw r;
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            const auto t = *(detail::GlobalU2*)(weight + std::size_t(unit + j) * kt_bytes + col_bytes);
            r.w[2U * j] = t[0], r.w[2U * j + 1U] = t[1];
        }
        r.s = scale[std::size_t(o) * (in / 64U) + unit / 4U];
        return r;
    }
    __device__ __forceinline__ Raw stage_load(unsigned tile, unsigned col, unsigned unit) const {
        const unsigned o0 = 16U * (v.tile0 + tile) + col, o = o0 < v.count ? o0 : v.count - 1U;
        return load(v.weight, v.scale, v.n, v.in, o, unit);
    }
    __device__ __forceinline__ void stage_decode(const Raw& raw, U4* nb) const {
        // The empty asm keeps every decode instruction here (measured: the compiler otherwise hoists the XOR and the
        // scale pack next to the loads, and the stager waits for DRAM before the stage's WMMAs).
        Raw r = raw;
#pragma unroll
        for (unsigned q = 0; q < 8U; ++q) asm volatile("" : "+v"(r.w[q]));
        asm volatile("" : "+v"(r.s));
        const detail::Half2 scale2 = __builtin_bit_cast(detail::Half2, r.s | (r.s << 16U));
#pragma unroll
        for (unsigned q = 0; q < 8U; ++q) {  // nb[q] = k 8q .. 8q + 7 of the group = word q (8 nibbles)
            const std::uint32_t b = r.w[q] ^ 0x88888888U;
            U4 h;
#pragma unroll
            for (unsigned p = 0; p < 4U; ++p) h[p] = detail::w16_pair(b, p, scale2);
            nb[q] = h;
        }
    }
    __device__ __forceinline__ void stage(unsigned tile, unsigned col, unsigned unit, U4* nb) const {
        stage_decode(stage_load(tile, col, unit), nb);
    }
};

using Store = q8a8::Store;

template <unsigned TC, unsigned Waves> __global__ __launch_bounds__(32 * Waves) void kernel(View v) {
    __shared__ Storage<TC> lds;
    core<TC>(v.x, v.in, v.rows, Cols<Waves>{v}, lds, Store{v.count, v.tile0, v.accumulate});
}
// x16 form: A fragments from the FP16 x tiles of f16_dense_tall (enqueue_x16_tiles: the core's own RNE cast done once per
// input), CTAs column-fast (X16Cols): same bits as kernel (the core's x16 contract). Pays when the FP32 x re-read per
// column block misses the caches (see x16_pays).
template <unsigned TC, unsigned Waves> __global__ __launch_bounds__(32 * Waves) void kernel_x16(View v, const _Float16* xt) {
    __shared__ Storage<TC> lds;
    core<TC>(v.x, v.in, v.rows, X16Cols<Cols<Waves>>{{v}, xt, v.rows}, lds, Store{v.count, v.tile0, v.accumulate});
}

// GLU pair sweep: h = F(first x, second x) for two IU4 projections of the same input and width (e.g. gate and up), in
// ONE launch of the x16 form. Each 16-tile CTA computes 8 output tiles twice (slots 0-7 first member, 8-15 second:
// core/dense_pair_map.hpp); the core's pair store gets both FP32 sums of an output in registers and writes F of them,
// rounded to FP16 (RNE), straight into the x16 tiles of h (the A operand of the next x16 GEMM). Bits: each sum is the
// WMMA chain of kernel_x16 for that column (same A fragments, same decoded B column, same k order), and the store is
// enqueue_x16_tiles_map2<F, TileRne>'s expression (F::apply, FP32 value pinned, one RNE cast): the tiles of h are
// bit-identical to kernel_x16(first) + kernel_x16(second) + map2, without the two FP32 outputs and the map2 pass.
// F: static float apply(float first, float second) (the engine's activation; nothing here names one).
struct GluView {
    View first;                          // first member's pack, x / in / rows; n == count (all outputs); out unused
    const std::uint8_t* second_weight;   // second member's pack: same n, in and layout
    const std::uint16_t* second_scale;
    _Float16* h;                         // x16 tiles of h: (o / 64 * rows + r) * 64 + o % 64, o < count
};
inline bool glu_pair_supported(const GluView& g) {
    const View& v = g.first;
    return v.weight && v.scale && g.second_weight && g.second_scale && g.h && v.rows && v.in && v.in % 128U == 0U &&
           v.count == v.n && v.count % 128U == 0U && std::uint64_t(v.rows) * v.in * 4U < (1ULL << 32U);
}
template <unsigned Waves> struct PairCols : Cols<Waves> {
    const std::uint8_t* second_weight;
    const std::uint16_t* second_scale;
    __device__ __forceinline__ ColumnTile tile(unsigned i) const {
        const unsigned o = 16U * dense_pair_map::out_tile<16U>(i);
        return {nullptr, nullptr, 0U, o < this->v.count};
    }
    __device__ __forceinline__ typename Cols<Waves>::Raw stage_load(unsigned tile, unsigned col, unsigned unit) const {
        const bool second = dense_pair_map::second<16U>(tile);
        const unsigned o = 16U * dense_pair_map::out_tile<16U>(tile) + col;  // count % 128 == 0: every column is live
        return Cols<Waves>::load(second ? second_weight : this->v.weight, second ? second_scale : this->v.scale, this->v.n, this->v.in,
                                 o, unit);
    }
    __device__ __forceinline__ void stage(unsigned tile, unsigned col, unsigned unit, U4* nb) const {
        this->stage_decode(stage_load(tile, col, unit), nb);
    }
};
template <class F> struct GluX16Store {
    _Float16* h;
    std::uint32_t rows;
    struct Tile {
        unsigned o;
    };
    __device__ __forceinline__ Tile open(unsigned i, const ColumnTile&, unsigned c) const { return {16U * dense_pair_map::out_tile<16U>(i) + c}; }
    __device__ __forceinline__ void put_pair(const Tile& t, unsigned r, float first, float second) const {
        float y = F::apply(first, second);
        asm volatile("" : "+v"(y));  // as x16_tiles_map2_kernel: the FP32 value is fixed before the FP16 cast
        h[dense_pair_map::x16_index(rows, r, t.o)] = static_cast<_Float16>(y);
    }
    __device__ __forceinline__ void close() const {}
    template <class Fn> __device__ __forceinline__ void run(Fn&& f) { f(*this); }
};
template <class F, unsigned Waves> __global__ __launch_bounds__(32 * Waves) void kernel_glu_x16(GluView g, const _Float16* xt) {
    __shared__ Storage<16> lds;
    core<16>(g.first.x, g.first.in, g.first.rows, X16Cols<PairCols<Waves>>{{{g.first}, g.second_weight, g.second_scale}, xt, g.first.rows},
             lds, GluX16Store<F>{g.h, g.first.rows});
}

inline bool supported(const View& v) {
    return v.weight && v.scale && v.x && v.out && v.rows && v.count && v.in && v.in % 128U == 0U && v.n % 16U == 0U &&
           v.count <= v.n && v.out_stride >= v.count && std::uint64_t(v.rows) * v.in * 4U < (1ULL << 32U);
}

// Rows per weight read = 16 * waves (the same bits either way): STRIX_F16_IU4_WAVES (8 or 16) when set, else the rule
// measured for the sibling q8a8 policy on gfx1151 (q8a8::waves_for: 256-row CTAs for in >= 8192 or 512-1023 rows).
inline unsigned waves_for(unsigned rows, unsigned in, unsigned waves = 0U) {
    static const unsigned env = static_cast<unsigned>(envcfg::integer("STRIX_F16_IU4_WAVES", 0));
    if (!waves) waves = env;
    return waves == 8U || waves == 16U ? waves : rows >= 512U && (in >= 8192U || rows < 1024U) ? 16U : 8U;
}

// The plan's launches (full 16-tile CTAs, then the remainder width): k(width constant, grid, view with tile0 set).
// ColumnFast: grid (column blocks, row blocks), the order of the x16 form.
template <unsigned Waves, bool ColumnFast = false, class K> inline void launch_plan(View v, K&& k) {
    const unsigned tiles = (v.count + 15U) / 16U, gx = (v.rows + 16U * Waves - 1U) / (16U * Waves);
    const Plan pl = plan(tiles, v.rows, true);
    const auto grid = [&](unsigned cols) { return ColumnFast ? dim3(cols, gx) : dim3(gx, cols); };
    v.tile0 = 0U;
    if (pl.full) k(std::integral_constant<unsigned, 16U>{}, grid(pl.full), v);
    v.tile0 = 16U * pl.full;
    if (pl.tc == 1U) k(std::integral_constant<unsigned, 1U>{}, grid(pl.gy), v);
    if (pl.tc == 2U) k(std::integral_constant<unsigned, 2U>{}, grid(pl.gy), v);
    if (pl.tc == 4U) k(std::integral_constant<unsigned, 4U>{}, grid(pl.gy), v);
}
template <unsigned Waves> inline void launch(View v, hipStream_t stream) {
    launch_plan<Waves>(v, [&](auto tc, dim3 grid, const View& w) {
        hipLaunchKernelGGL((kernel<decltype(tc)::value, Waves>), grid, dim3(32U * Waves), 0, stream, w);
    });
}

// Caller checks supported(v) and the launch error (hipGetLastError).
inline void enqueue(const View& v, hipStream_t stream, unsigned waves = 0U) {
    if (waves_for(v.rows, v.in, waves) == 16U) launch<16>(v, stream);
    else launch<8>(v, stream);
}
// STRIX_F16_IU4_X16_256 (default on; 0 = the previous rules): the x16 form from 256 rows, 16-wave x16 CTAs (below).
inline bool x16_from_256() {
    static const bool on = envcfg::flag("STRIX_F16_IU4_X16_256", true);
    return on;
}
// CTA waves of the x16 form (the same bits for 8 and 16): STRIX_F16_IU4_WAVES or the caller's choice when set, else 16
// from 256 rows (measured on gfx1151, x16 form, 7 Qwen shapes in, n in [4096, 17408], two rounds: 16 waves faster at
// 384 / 512 rows on every shape, at 4096 rows by 4-8 % where the direct rule picked 8 (in < 8192, e.g. 4096 -> 12288:
// 10.3-11.1 vs 11.1-11.4 ms) and equal on 5120 -> 17408), else the direct form's rule (waves_for).
inline unsigned x16_waves_for(unsigned rows, unsigned in, unsigned waves = 0U) {
    static const unsigned env = static_cast<unsigned>(envcfg::integer("STRIX_F16_IU4_WAVES", 0));
    if (waves == 8U || waves == 16U || env == 8U || env == 16U) return waves_for(rows, in, waves);
    return x16_from_256() && rows >= 256U ? 16U : waves_for(rows, in);
}
// x16 form of enqueue: xt = enqueue_x16_tiles(v.x, xt, v.rows, v.in) of the same x (x16_bytes(rows, in)); same bits.
inline void enqueue_x16(const View& v, const _Float16* xt, hipStream_t stream, unsigned waves = 0U) {
    const auto go = [&](auto wv) {
        constexpr unsigned W = decltype(wv)::value;
        launch_plan<W, true>(v, [&](auto tc, dim3 grid, const View& w) {
            hipLaunchKernelGGL((kernel_x16<decltype(tc)::value, W>), grid, dim3(32U * W), 0, stream, w, xt);
        });
    };
    if (x16_waves_for(v.rows, v.in, waves) == 16U) go(std::integral_constant<unsigned, 16U>{});
    else go(std::integral_constant<unsigned, 8U>{});
}
// GLU pair sweep (kernel_glu_x16): xt = the x16 tiles of g.first.x; writes the x16 tiles of h into g.h (x16_bytes(rows,
// count), must not alias xt). Caller checks glu_pair_supported(g) and the launch error. Column-fast grid, as enqueue_x16.
template <class F> inline void enqueue_glu_x16(const GluView& g, const _Float16* xt, hipStream_t stream, unsigned waves = 0U) {
    const auto go = [&](auto wv) {
        constexpr unsigned W = decltype(wv)::value;
        const dim3 grid(dense_pair_map::column_blocks<16U>(g.first.count / 16U), (g.first.rows + 16U * W - 1U) / (16U * W));
        hipLaunchKernelGGL((kernel_glu_x16<F, W>), grid, dim3(32U * W), 0, stream, g, xt);
    };
    if (waves_for(g.first.rows, g.first.in, waves) == 16U) go(std::integral_constant<unsigned, 16U>{});
    else go(std::integral_constant<unsigned, 8U>{});
}
// Where the x16 form was measured faster on gfx1151 (tile pass included): from 512 rows, FP32 x above 16 MiB
// (17408 -> 5120: 3163 -> 2266 us; x = 35.6 MB is re-read from DRAM by each of the 20 column blocks, the FP16 tiles fit
// the 32 MB MALL) or n >= 10240 (5120 -> 17408: 2202 -> 2121 us, 5120 -> 10240: 1305 -> 1246 us); from 256 to 511 rows
// (x16_waves_for's 16-wave CTAs) on every measured shape with in, n in [4096, 17408] (tile pass included, vs the direct
// form with 8 waves: 256 rows -1..-9 %, 384 rows -1..-27 %, e.g. 17408 -> 5120 at 384 rows: 2712 -> 1797 + 32 us).
// Other shapes keep the direct form until measured.
inline bool x16_pays(unsigned rows, unsigned in, unsigned n) {
    return (rows >= 512U && (n >= 10240U || std::uint64_t(rows) * in * 4U > (std::uint64_t(16) << 20U))) ||
           (x16_from_256() && rows >= 256U && rows < 512U && in >= 4096U && n >= 4096U);
}
}  // namespace strix::core::f16_dense_tall::iu4a16
