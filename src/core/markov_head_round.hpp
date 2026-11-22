#pragma once

#include "strix/weight/formats/scalar_float.hpp"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_MARKOV_HD __host__ __device__
#else
#define STRIX_MARKOV_HD
#endif

namespace strix::core {

// Direct RNE from FP64, correcting a possible FP32 midpoint first.
STRIX_MARKOV_HD inline std::uint16_t markov_fold_bf16(double value) {
    using namespace weight::scalar_float;
    auto nearest = static_cast<float>(value);
    const auto bits = f2u(nearest);
    if ((bits & 0xffffU) == 0x8000U && static_cast<double>(nearest) != value) {
        const bool negative = (bits & 0x80000000U) != 0U;
        const bool increases = value > static_cast<double>(nearest);
        nearest = u2f(increases == negative ? bits - 1U : bits + 1U);
    }
    return bf16_encode(nearest);
}

}  // namespace strix::core

#undef STRIX_MARKOV_HD
