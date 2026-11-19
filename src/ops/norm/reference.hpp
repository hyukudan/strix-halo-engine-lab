#pragma once

#include "norm.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace strix::ops::norm {

// Host-only reference for the supported reduction contract. Host traits supply
// load() and inv_sqrt(); a host result does not qualify device code generation.
// Compile with contraction/fast math disabled, as for the engine's device path.
template <class Traits>
void reference(const typename Traits::Input* x, const typename Traits::Scale* w,
               typename Traits::ScaleFormat wt, unsigned n, float eps,
               typename Traits::Output* y, unsigned rows) {
    static_assert(Traits::reduction_order == ReductionOrder::lane_strided_fma_lds_halving);
    static_assert(std::is_same_v<typename Traits::Input, float>);
    static_assert(std::is_same_v<typename Traits::Output, float>);
    const unsigned lanes = Norm<Traits>::lanes(n);
    if (n == 0 || lanes == 0 || (lanes & (lanes - 1U)) != 0)
        throw std::invalid_argument("RMSNorm requires nonzero width and power-of-two lanes");
    std::vector<float> partial(lanes);
    for (unsigned row = 0; row < rows; ++row) {
        const float* xr = x + static_cast<std::size_t>(row) * n;
        float* yr = y + static_cast<std::size_t>(row) * n;
        for (unsigned lane = 0; lane < lanes; ++lane) {
            float a = 0.0F;
            for (unsigned i = lane; i < n; i += lanes) a = std::fma(xr[i], xr[i], a);
            partial[lane] = a;
        }
        for (unsigned h = lanes / 2; h > 0; h >>= 1U)
            for (unsigned lane = 0; lane < h; ++lane)
                partial[lane] = partial[lane] + partial[lane + h];
        const float r = Traits::inv_sqrt(partial[0] / static_cast<float>(n) + eps);
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
