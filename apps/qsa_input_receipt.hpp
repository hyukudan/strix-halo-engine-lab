#pragma once
#include "strix/qsa_input_prefetch.hpp"
#include <string>

namespace strix::decode_detail {
// Query at result publication, after execution. Disabled mode adds no field.
inline std::string qsa_input_dispatch_receipt(
    const qsa_input_prefetch::RuntimeDispatchStats* stats) {
    if (!stats) return {};
    return "  \"prefetch_q8_qsa_input_dispatched_projections\": " +
        std::to_string(stats->dispatched_projections.load(std::memory_order_relaxed)) + ",\n";
}
} // namespace strix::decode_detail
