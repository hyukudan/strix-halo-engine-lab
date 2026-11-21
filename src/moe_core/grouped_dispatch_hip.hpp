#pragma once

#include "moe_core/grouped_dispatch.hpp"
#include "moe_core/grouped_microscaled.hpp"

namespace strix::moe_core::grouped_dispatch {

struct HipBackend {
    static constexpr unsigned rows_per_tile = grouped_microscaled::gemm::kBM;
    static constexpr auto fits = grouped_microscaled::fits;
    template<weight::microscaled::MatrixFormat Format>
    static constexpr auto fits_matrix_tile = grouped_microscaled::fits_tile<Format>;

#if defined(__HIPCC__)
    using Status = hipError_t;
    using Stream = hipStream_t;
    static constexpr auto success = hipSuccess;
    static constexpr auto invalid_value = hipErrorInvalidValue;
    static constexpr auto gather = grouped_rows::gather;
    template<weight::microscaled::MatrixFormat Format>
    static constexpr auto launch_projection = grouped_microscaled::launch<Format>;
    template<unsigned Experts, unsigned TopK>
    static constexpr auto materialize = device_grouped::enqueue<Experts, TopK>;
#endif
};

}  // namespace strix::moe_core::grouped_dispatch
