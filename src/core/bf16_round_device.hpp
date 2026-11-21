#pragma once

#include "strix/weight/formats/scalar_float.hpp"

namespace strix::core::prepared_rows_device {

__device__ inline float round_bf16(float value) {
    namespace scalar = weight::scalar_float;
    return scalar::u2f(std::uint32_t{scalar::bf16_encode(value)} << 16U);
}

}  // namespace strix::core::prepared_rows_device
