#pragma once

#include "norm.hpp"
#include "norm_numeric_traits.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace strix::ops::norm {

// Host emulation of ReductionOrder::wave_chunk_serial (norm_numeric_traits.hpp) on one row of n > 0 values: 32-lane
// chains per chunk, the xor butterfly d = 16 .. 1 (lane 0's value), chunk sums added serially from chunk 0.
template <class Traits>
float wave_chunk_square_sum(const float* xr, unsigned n) {
    constexpr unsigned chunk = wave_chunk_of<Traits>();
    constexpr SquareForm form = square_form_of<Traits>();
    const unsigned span = chunk == 0U ? n : chunk, chunks = wave_chunk_count(n, chunk);
    float total = 0.0F;
    for (unsigned c = 0; c < chunks; ++c) {
        const unsigned first = c * span, end = n - first < span ? n : first + span;
        std::array<float, kWaveLanes> lane{};
        for (unsigned l = 0; l < kWaveLanes; ++l) {
            float a = 0.0F;
            for (unsigned i = first + l; i < end; i += kWaveLanes) a = square_step<form>(a, xr[i]);
            lane[l] = a;
        }
        for (unsigned d = kWaveLanes / 2; d > 0; d >>= 1U) {
            std::array<float, kWaveLanes> next{};
            for (unsigned l = 0; l < kWaveLanes; ++l) next[l] = lane[l] + lane[l ^ d];
            lane = next;
        }
        total = c == 0 ? lane[0] : total + lane[0];
    }
    return total;
}

// Host emulation of ReductionOrder::lane_strided_fma_lds_halving on one row of n > 0 values with `lanes` (a power of
// two) virtual lanes: lane t chains the squares of x[t], x[t + lanes], ... in form F, then the halving tree
// partial[l] = partial[l] + partial[l + h] for h = lanes / 2 .. 1; the row sum is partial[0]. `partial` has `lanes`
// floats of scratch. Rows of at most 32 J values on the wave-rows kernel have the same bits as any lanes >= 32 J.
template <SquareForm F>
float lane_strided_square_sum(const float* xr, unsigned n, unsigned lanes, float* partial) {
    for (unsigned lane = 0; lane < lanes; ++lane) {
        float a = 0.0F;
        for (unsigned i = lane; i < n; i += lanes) a = square_step<F>(a, xr[i]);
        partial[lane] = a;
    }
    for (unsigned h = lanes / 2; h > 0; h >>= 1U)
        for (unsigned lane = 0; lane < h; ++lane) partial[lane] = partial[lane] + partial[lane + h];
    return partial[0];
}

// Host-only reference for the supported reduction contracts. Host traits supply
// load() and inv_sqrt(); a host result does not qualify device code generation.
// Compile with contraction/fast math disabled, as for the engine's device path.
template <class Traits>
void reference(const typename Traits::Input* x, const typename Traits::Scale* w,
               typename Traits::ScaleFormat wt, unsigned n, float eps,
               typename Traits::Output* y, unsigned rows) {
    constexpr bool chunked = Traits::reduction_order == ReductionOrder::wave_chunk_serial;
    static_assert(chunked || Traits::reduction_order == ReductionOrder::lane_strided_fma_lds_halving);
    static_assert(std::is_same_v<typename Traits::Input, float>);
    static_assert(std::is_same_v<typename Traits::Output, float>);
    const unsigned lanes = Norm<Traits>::lanes(n);
    if (n == 0 || lanes == 0 || (lanes & (lanes - 1U)) != 0)
        throw std::invalid_argument("RMSNorm requires nonzero width and power-of-two lanes");
    std::vector<float> partial(lanes);
    for (unsigned row = 0; row < rows; ++row) {
        const float* xr = x + static_cast<std::size_t>(row) * n;
        float* yr = y + static_cast<std::size_t>(row) * n;
        float ss = 0.0F;
        if constexpr (chunked) {
            ss = wave_chunk_square_sum<Traits>(xr, n);
        } else {
            ss = lane_strided_square_sum<square_form_of<Traits>()>(xr, n, lanes, partial.data());
        }
        const float r = Traits::inv_sqrt(ss / static_cast<float>(n) + eps);
        constexpr OutputForm form = output_form_of<Traits>();
        for (unsigned i = 0; i < n; ++i) {
            if constexpr (form == OutputForm::plain)
                yr[i] = w ? (xr[i] * r) * Traits::load(w, wt, i) : xr[i] * r;
            else
                yr[i] = finish<form>(xr[i] * r, w ? Traits::load(w, wt, i) : 1.0F, w != nullptr);
        }
    }
}

}  // namespace strix::ops::norm
