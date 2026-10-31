#pragma once
#include "gemma4/contracts/g4_types.hpp"
#include <cstdlib>
#include <string_view>

namespace strix::gemma4::gpu {
// Cache policy changes scheduling only. Small quantized passes retain normal
// reads; dense BF16 and shared passes with at least four rows stream weights.
inline int streaming_override() {
    static const int policy = [] {
        const auto* p = std::getenv("STRIX_G4_STREAMING_WEIGHTS");
        return p ? (std::string_view(p) == "1" ? 1 : 0) : -1;
    }();
    return policy;
}
inline bool stream_weights(WeightFormat format, unsigned rows) {
    const int policy = streaming_override();
    return policy >= 0 ? policy == 1 : format == WeightFormat::bf16 || rows >= 4U;
}
} // namespace strix::gemma4::gpu
