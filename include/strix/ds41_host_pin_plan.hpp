#pragma once
// CPU-only host placement for an ascending, single-pass expert schedule.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix::ds41 {

enum class HostPinOrder { layer, deadline, early };

inline HostPinOrder parse_host_pin_order(std::string_view value) {
    if (value == "layer") return HostPinOrder::layer;
    if (value == "deadline") return HostPinOrder::deadline;
    if (value == "early") return HostPinOrder::early;
    throw std::invalid_argument("--host-pin-order must be layer, deadline or early");
}
inline const char* host_pin_order_name(HostPinOrder order) {
    return order == HostPinOrder::deadline ? "deadline" : order == HostPinOrder::early ? "early" : "layer";
}

// Zero bytes means fully device resident. The first S nonzero layers already occupy
// the idle-preload slots, regardless of tier. Deadline placement keeps those on FILE
// and pins the longest affordable suffix of the remaining streamed schedule. In
// particular, small decoder tails are considered before full encoder layers.
// Stop at the first non-fitting layer: filling the remainder with earlier HOST
// layers would put a host-copy burst before a later FILE deadline. The loader has
// no independent disk lookahead past that burst (its four staging pages are bounded).
// This is a suffix policy, not a general heterogeneous knapsack optimizer. It avoids
// internal HOST/FILE transitions and keeps the disk prefix continuous at short rows.
inline std::vector<std::uint8_t> deadline_host_layers(std::span<const std::uint64_t> bytes,
                                                    std::uint32_t slots, double budget_bytes) {
    std::vector<std::uint8_t> host(bytes.size(), 0);
    std::vector<std::size_t> streamed;
    for (std::size_t l = 0; l < bytes.size(); ++l)
        if (bytes[l]) streamed.push_back(l);
    const auto preloaded = std::min<std::size_t>(std::max(1U, slots), streamed.size());
    for (std::size_t j = streamed.size(); j > preloaded;) {
        const auto l = streamed[--j];
        if (budget_bytes < static_cast<double>(bytes[l])) break;
        budget_bytes -= static_cast<double>(bytes[l]);
        host[l] = 1;
    }
    return host;
}

// Early placement: the legacy ascending fill (every streamed layer that still fits, in
// layer order), minus the first S streamed layers that the idle preload already holds.
// Pinned layers then sit at the start of the schedule, where little compute hides a
// FILE read, while later FILE reads overlap the compute of the pinned ones (measured: the
// suffix order was no faster than the legacy one at 30K for that reason).
inline std::vector<std::uint8_t> early_host_layers(std::span<const std::uint64_t> bytes,
                                                 std::uint32_t slots, double budget_bytes) {
    std::vector<std::uint8_t> host(bytes.size(), 0);
    std::size_t seen = 0;
    const std::size_t preloaded = std::max(1U, slots);
    for (std::size_t l = 0; l < bytes.size(); ++l) {
        if (!bytes[l]) continue;
        if (seen++ < preloaded) continue;
        if (budget_bytes < static_cast<double>(bytes[l])) continue;
        budget_bytes -= static_cast<double>(bytes[l]);
        host[l] = 1;
    }
    return host;
}

}  // namespace strix::ds41
