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

// Forms (bits of the kernel template; every form gives the contract's bits):
//   kWPacked : the stagers decode weights two at a time in packed FP16 (w16_pair) instead of one FP32 product each.
inline constexpr unsigned kWPacked = 1U;

namespace detail {
using GlobalBytes16 = const U4 __attribute__((address_space(1), aligned(1)));
using GlobalBytes4 = const std::uint32_t __attribute__((address_space(1), aligned(1)));
using Half8 = _Float16 __attribute__((ext_vector_type(8)));
using Half2 = _Float16 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float code(std::uint32_t word, unsigned byte) {  // signed byte `byte` of a little-endian word
    return static_cast<float>(static_cast<std::int32_t>(word << (24U - 8U * byte)) >> 24);
}
// w16 of the contract for byte `byte` of a word of codes: the FP32 product is exact (8 x 11 significant bits).
__device__ __forceinline__ _Float16 w16_scalar(std::uint32_t word, unsigned byte, float scale) {
    return static_cast<_Float16>(code(word, byte) * scale);
}
// The same bits for the two codes of one 16-bit half of a word (hi: bytes 2 and 3). `biased` = word ^ 0x80808080 holds
// code + 128 per byte; v_perm_b32 puts 0x64 above each byte, i.e. the FP16 integer 1024 + code + 128; the packed
// subtraction of 1152 is exact (|code| <= 128, so code is an FP16 integer) and one packed FP16 multiply rounds the exact
// product code * scale once (RNE), the single rounding of the FP32 path: equal for subnormal, infinite and NaN scales
// too (the device test compares all 65,536 scales x 256 codes bit for bit).
__device__ __forceinline__ Half2 w16_pair(std::uint32_t biased, unsigned hi, Half2 scale2) {
    const std::uint32_t bits = __builtin_amdgcn_perm(0x64646464U, biased, hi ? 0x04030402U : 0x04010400U);
    const _Float16 bias = static_cast<_Float16>(1152.0F);
    return (__builtin_bit_cast(Half2, bits) - Half2{bias, bias}) * scale2;
}
// a16 of the contract: fp16_rne(float(code) * delta), delta FP32 (never narrowed)
__device__ __forceinline__ _Float16 a16_value(std::uint32_t word, unsigned byte, float delta) {
    return static_cast<_Float16>(code(word, byte) * delta);
}
}  // namespace detail

template <unsigned Waves, unsigned Form = 0U> struct Cols {
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
        if constexpr ((Form & kWPacked) != 0U) {
            std::uint32_t x[17];
#pragma unroll
            for (unsigned i = 0; i < 17U; ++i) x[i] = d[i] ^ 0x80808080U;
#pragma unroll
            for (unsigned b = 0; b < 2U; ++b) {
                const unsigned at = 34U * b;  // the scale's half of word at / 4 (at % 4 is 0 or 2)
                const std::uint32_t s = (d[at / 4U] >> (8U * (at % 4U))) & 0xffffU;
                const detail::Half2 scale2 = __builtin_bit_cast(detail::Half2, s | (s << 16U));
#pragma unroll
                for (unsigned q = 0; q < 4U; ++q) {
                    U4 h;
#pragma unroll
                    for (unsigned j = 0; j < 4U; ++j) {  // codes at bytes at + 2 + 8 q + 2 j, + 1: one half of a word
                        const unsigned byte = at + 2U + 8U * q + 2U * j;
                        h[j] = __builtin_bit_cast(std::uint32_t, detail::w16_pair(x[byte / 4U], (byte % 4U) / 2U, scale2));
                    }
                    nb[4U * b + q] = h;
                }
            }
            return;
        }
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
                    h[i] = detail::w16_scalar(d[byte / 4U], byte % 4U, scale);
                }
                nb[4U * b + q] = __builtin_bit_cast(U4, h);
            }
        }
    }
    __device__ __forceinline__ Half16 fragment(unsigned row, unsigned unit) const {
        const std::size_t at = static_cast<std::size_t>(row) * v.x_stride + 16U * unit;
        const float delta = v.deltas[static_cast<std::size_t>(row) * (v.x_stride / 32U) + unit / 2U];
        const U4 t = *(detail::GlobalBytes16*)(reinterpret_cast<const std::uint8_t*>(v.codes) + at);
        Half16 h;
#pragma unroll
        for (unsigned i = 0; i < 16U; ++i) h[i] = detail::a16_value(t[i / 4U], i % 4U, delta);
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

template <unsigned TC, unsigned Waves, unsigned Form> __global__ __launch_bounds__(32 * Waves) void kernel(View v) {
    __shared__ Storage<TC> lds;
    core<TC>(nullptr, v.in, v.rows, Cols<Waves, Form>{v}, lds, Store{v.count, v.tile0, v.accumulate});
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

// Form of a call (kWPacked or 0; the same bits either way). kFormRule = STRIX_F16_Q8A8_WPACK (default on; 0 = the scalar
// FP32 decode of every weight, the previous form).
inline constexpr unsigned kFormRule = ~0U;
inline unsigned form_for(unsigned form = kFormRule) {
    static const unsigned rule = envcfg::flag("STRIX_F16_Q8A8_WPACK", true) ? kWPacked : 0U;
    return (form == kFormRule ? rule : form) & kWPacked;
}

template <unsigned Waves, unsigned Form> inline void launch(View v, hipStream_t stream) {
    const unsigned tiles = (v.count + 15U) / 16U, gx = (v.rows + 16U * Waves - 1U) / (16U * Waves);
    const Plan pl = plan(tiles, v.rows, true);
    v.tile0 = 0U;
    if (pl.full) hipLaunchKernelGGL((kernel<16, Waves, Form>), dim3(gx, pl.full), dim3(32U * Waves), 0, stream, v);
    v.tile0 = 16U * pl.full;
    if (pl.tc == 1U) hipLaunchKernelGGL((kernel<1, Waves, Form>), dim3(gx, pl.gy), dim3(32U * Waves), 0, stream, v);
    if (pl.tc == 2U) hipLaunchKernelGGL((kernel<2, Waves, Form>), dim3(gx, pl.gy), dim3(32U * Waves), 0, stream, v);
    if (pl.tc == 4U) hipLaunchKernelGGL((kernel<4, Waves, Form>), dim3(gx, pl.gy), dim3(32U * Waves), 0, stream, v);
}
template <unsigned Waves> inline void launch_form(const View& v, unsigned form, hipStream_t stream) {
    if (form == kWPacked) launch<Waves, kWPacked>(v, stream);
    else launch<Waves, 0U>(v, stream);
}

// Caller checks supported(v) and the launch error (hipGetLastError).
inline void enqueue(const View& v, hipStream_t stream, unsigned waves = 0U, unsigned form = kFormRule) {
    const unsigned f = form_for(form);
    if (waves_for(v.rows, v.in, waves) == 16U) launch_form<16>(v, f, stream);
    else launch_form<8>(v, f, stream);
}
// Independent FP32 chains per K part, added in unfused order by Store.
// Caller validates each equal-width part; the stream orders the output additions.
inline void enqueue_parts(View v, unsigned parts, hipStream_t stream, unsigned form = kFormRule) {
    for (unsigned part = 0; part < parts; ++part) {
        enqueue(v, stream, 0U, form);
        if (part + 1U == parts) break;
        v.k_bytes += v.in / 32U * 34U;
        v.codes += v.in;
        v.deltas += v.in / 32U;
        v.accumulate = 1U;
    }
}
}  // namespace strix::core::f16_dense_tall::q8a8
