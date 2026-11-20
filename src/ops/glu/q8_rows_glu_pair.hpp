#pragma once
// GLU projection group on the q8_rows direct form (src/hip_q8_rows_gemv.hpp paired store,
// strix/decode_projection_group.hpp pair_unit): the one-row Q8_0 Linears that read the same input x in ONE launch,
//   units [0, pair_first)                   an optional plain member, stored as its Linear output R(dot) at out[u];
//   units [pair_first, pair_first + 2 half) gate / up walked interleaved; the gate unit's store receives both dots and
//                                           writes act[j] = swiglu<Exp, R>(R(gate j), R(up j), limit).
// R(dot) is glu::epilogue<R> without bias (the Linear's stored output), swiglu the glu.hpp device function: the same
// functions on the same FP32 values as the separate stores and the glu::swiglu_kernel launch they replace, and every dot
// is bitwise the one-member launch's (the paired-store contract), so the group changes no bits.
// Binders: Hy4-flash decode FFN (latent down + shared expert gate / up; dense layer gate / up).
// Home: shared (backend core, ops/glu on the q8_rows family). The including TU owns the arithmetic flags
// (-ffp-contract=off, as for glu_device.hpp).
#include "../../hip_q8_rows_gemv.hpp"
#include "glu.hpp"

#include <cstdint>

namespace strix::ops::glu {

template <class Exp, Round R>
struct Q8RowsGluPair {
    using Input = float;
    static constexpr unsigned rows = 1U;
    const void* input[1];
    strix::q8_rows::Segments segments;
    std::uint32_t width, outputs, pair_first, pair_half;
    float* out;   // plain member output [pair_first]
    float* act;   // GLU output [pair_half]
    float limit;  // swiglu clamp (no_clamp: none)
    __device__ static float coefficient(unsigned scale, std::int8_t code) {
        return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(scale))) * static_cast<float>(code);
    }
    __device__ static void fma(float& acc, float c, float x) { acc = __builtin_fmaf(c, x, acc); }
    __device__ static float add(float lower, float upper) { return lower + upper; }
    __device__ void store_pair(std::uint32_t u, float value, float next) const {
        const float v = epilogue<R>(value, nullptr, 0U);
        if (u < pair_first) {
            out[u] = v;
            return;
        }
        const std::uint32_t k = u - pair_first;
        if (k & 1U) return;  // the up unit: consumed by its gate's store
        act[k >> 1U] = swiglu<Exp, R>(v, epilogue<R>(next, nullptr, 0U), limit);
    }
};

// Host helper: the group over `x` (width columns): plain (may be null, plain_outputs 0), gate, up (ff rows each).
template <class Exp, Round R>
inline Q8RowsGluPair<Exp, R> q8_rows_glu_pair(const void* x, std::uint32_t width, const void* plain, std::uint32_t plain_outputs,
                                              float* plain_out, const void* gate, const void* up, std::uint32_t ff, float* act,
                                              float limit) {
    Q8RowsGluPair<Exp, R> site{};
    site.input[0] = x;
    site.segments = plain ? strix::q8_rows::segments({{plain, plain_outputs}, {gate, ff}, {up, ff}})
                          : strix::q8_rows::segments({{gate, ff}, {up, ff}});
    site.width = width;
    site.pair_first = plain ? plain_outputs : 0U;
    site.pair_half = ff;
    site.outputs = site.pair_first + 2U * ff;
    site.out = plain_out;
    site.act = act;
    site.limit = limit;
    return site;
}

}  // namespace strix::ops::glu
