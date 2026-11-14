#pragma once
// Named policy dense-q8a8-f16-v1 of the shared FP16 dense core (f16_dense_tall.hpp): Q8_0 weights x A8 activations in
// ONE FP16 WMMA chain per output, y[r][o] = sum_k a16[r][k] * w16[o][k] (FP32 accumulation, k ascending, from +0).
//
// Class: FAST (not the bits of the integer Q8_0 x A8 products: those scale an exact integer sum per 32-k group).
// Arithmetic contract:
//   w16[o][k] = fp16_rne(float(code_w) * float(d))   d = the FP16 scale of the weight's Q8_0 block (product exact in FP32)
//   a16[r][k] = fp16_rne(float(code_a) * delta)       delta = the FP32 step of the row's 32-k group
// Weights are decoded by the CTA's stagers once per (CTA, stage of 64 k) into LDS: a Q8_0 byte is read once per
// 16 * Waves rows (34 bytes per 32 weights, about half the bytes of an FP16 mirror) and no FP16 copy of the weights
// exists. Activations are converted in registers (lane = row). Any engine with Q8_0 rows and A8 codes binds it by
// filling View; nothing here names a model.
// Limits (callers check with supported()): in % 128 == 0, x_stride % 32 == 0, in <= x_stride.
#include "f16_dense_tall.hpp"

namespace strix::core::f16_dense_tall::q8a8 {
inline constexpr char kPolicyName[] = "dense-q8a8-f16-v1";

struct View {
    const std::uint8_t* weight;  // Q8_0 rows: per 32 weights {FP16 scale, 32 x int8}; output o at weight + o * row_bytes
    const std::int8_t* codes;    // A8 codes: row r at codes + r * x_stride
    const float* deltas;         // deltas[r * (x_stride / 32) + k / 32]
    float* out;                  // out[r * out_stride + o]
    std::uint32_t row_bytes;     // bytes between consecutive weight rows
    std::uint32_t k_bytes;       // byte offset of the first used block inside a weight row (a K part)
    std::uint32_t in, count, rows, x_stride, out_stride;
    std::uint32_t tile0;         // first column tile of the launch (set by enqueue)
    std::uint32_t accumulate;    // out += sum
};

namespace detail {
using GlobalBytes16 = const U4 __attribute__((address_space(1), aligned(1)));
using GlobalBytes4 = const std::uint32_t __attribute__((address_space(1), aligned(1)));
using Half8 = _Float16 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ float code(std::uint32_t word, unsigned byte) {  // signed byte `byte` of a little-endian word
    return static_cast<float>(static_cast<std::int32_t>(word << (24U - 8U * byte)) >> 24);
}
}  // namespace detail

template <unsigned Waves> struct Cols {
    static constexpr unsigned kWaves = Waves;
    View v;
    __device__ __forceinline__ ColumnTile tile(unsigned i) const {
        const unsigned o = 16U * (v.tile0 + i);
        return {nullptr, v.out + o, v.out_stride, o < v.count};
    }
    // 64 weights = two Q8_0 blocks (68 bytes) of the stager's column; columns past count read the last one (never stored)
    __device__ __forceinline__ void stage(unsigned tile, unsigned col, unsigned unit, U4* nb) const {
        const unsigned o0 = 16U * (v.tile0 + tile) + col, o = o0 < v.count ? o0 : v.count - 1U;
        const auto* p = v.weight + static_cast<std::size_t>(o) * v.row_bytes + v.k_bytes + (unit / 2U) * 34U;
        std::uint32_t d[17];
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) {
            const U4 t = *(detail::GlobalBytes16*)(p + 16U * i);
            d[4U * i] = t[0], d[4U * i + 1U] = t[1], d[4U * i + 2U] = t[2], d[4U * i + 3U] = t[3];
        }
        d[16] = *(detail::GlobalBytes4*)(p + 64U);
#pragma unroll
        for (unsigned b = 0; b < 2U; ++b) {
            const unsigned at = 34U * b;  // block: scale at bytes at, at + 1; codes from at + 2
            const float scale = static_cast<float>(
                __builtin_bit_cast(_Float16, static_cast<std::uint16_t>(d[at / 4U] >> (8U * (at % 4U)))));
#pragma unroll
            for (unsigned q = 0; q < 4U; ++q) {
                detail::Half8 h;
#pragma unroll
                for (unsigned i = 0; i < 8U; ++i) {
                    const unsigned byte = at + 2U + 8U * q + i;
                    h[i] = static_cast<_Float16>(detail::code(d[byte / 4U], byte % 4U) * scale);
                }
                nb[4U * b + q] = __builtin_bit_cast(U4, h);
            }
        }
    }
    __device__ __forceinline__ Half16 fragment(unsigned row, unsigned unit) const {
        const auto* p = reinterpret_cast<const std::uint8_t*>(v.codes) + static_cast<std::size_t>(row) * v.x_stride + 16U * unit;
        const U4 t = *(detail::GlobalBytes16*)p;
        const float delta = v.deltas[static_cast<std::size_t>(row) * (v.x_stride / 32U) + unit / 2U];
        Half16 h;
#pragma unroll
        for (unsigned i = 0; i < 16U; ++i) h[i] = static_cast<_Float16>(detail::code(t[i / 4U], i % 4U) * delta);
        return h;
    }
};

// Stores only outputs below count (the last column tile may be partial); accumulate adds to what is there.
struct Store {
    std::uint32_t count, tile0, accumulate;
    struct Tile {
        float* y;
        unsigned stride, c;
        bool live;
    };
    __device__ __forceinline__ Tile open(unsigned i, const ColumnTile& ct, unsigned c) const {
        return {f16_dense_tall::detail::uniform(ct.y), f16_dense_tall::detail::uni(ct.stride), c, 16U * (tile0 + i) + c < count};
    }
    __device__ __forceinline__ void put(const Tile& t, unsigned r, float value) const {
        if (!t.live) return;
        float* dst = t.y + static_cast<std::size_t>(r) * t.stride + t.c;
        *dst = accumulate ? *dst + value : value;
    }
    __device__ __forceinline__ void close() const {}
    template <class F> __device__ __forceinline__ void run(F&& f) { f(*this); }
};

template <unsigned TC, unsigned Waves> __global__ __launch_bounds__(32 * Waves) void kernel(View v) {
    __shared__ Storage<TC> lds;
    core<TC>(nullptr, v.in, v.rows, Cols<Waves>{v}, lds, Store{v.count, v.tile0, v.accumulate});
}

inline bool supported(const View& v) {
    return v.weight && v.codes && v.deltas && v.out && v.rows && v.count && v.in && v.in % 128U == 0U && v.x_stride % 32U == 0U &&
           v.in <= v.x_stride && v.out_stride >= v.count;
}

// Rows per weight read = 16 * waves (the same bits either way). waves 0 = the rule: STRIX_F16_Q8A8_WAVES (8 or 16) when
// set, else the measured one (gfx1151, ms per call 128-row / 256-row CTAs): in 1024: 512 rows 0.186 / 0.165, 1024 rows
// equal, 2048-4096 rows 1.48 / 1.72 and 3.48 / 3.91; in 8192: 2048 rows 1.70 / 1.30. So 256-row CTAs for wide inputs
// (in >= 8192) and for 512-1023 rows, 128-row CTAs elsewhere.
inline unsigned waves_for(unsigned rows, unsigned in, unsigned waves = 0U) {
    static const unsigned env = static_cast<unsigned>(envcfg::integer("STRIX_F16_Q8A8_WAVES", 0));
    if (!waves) waves = env;
    return waves == 8U || waves == 16U ? waves : rows >= 512U && (in >= 8192U || rows < 1024U) ? 16U : 8U;
}

template <unsigned Waves> inline void launch(View v, hipStream_t stream) {
    const unsigned tiles = (v.count + 15U) / 16U, gx = (v.rows + 16U * Waves - 1U) / (16U * Waves);
    const Plan pl = plan(tiles, v.rows, true);
    v.tile0 = 0U;
    if (pl.full) hipLaunchKernelGGL((kernel<16, Waves>), dim3(gx, pl.full), dim3(32U * Waves), 0, stream, v);
    v.tile0 = 16U * pl.full;
    if (pl.tc == 1U) hipLaunchKernelGGL((kernel<1, Waves>), dim3(gx, pl.gy), dim3(32U * Waves), 0, stream, v);
    if (pl.tc == 2U) hipLaunchKernelGGL((kernel<2, Waves>), dim3(gx, pl.gy), dim3(32U * Waves), 0, stream, v);
    if (pl.tc == 4U) hipLaunchKernelGGL((kernel<4, Waves>), dim3(gx, pl.gy), dim3(32U * Waves), 0, stream, v);
}

// Caller checks supported(v) and the launch error (hipGetLastError).
inline void enqueue(const View& v, hipStream_t stream, unsigned waves = 0U) {
    if (waves_for(v.rows, v.in, waves) == 16U) launch<16>(v, stream);
    else launch<8>(v, stream);
}
}  // namespace strix::core::f16_dense_tall::q8a8
