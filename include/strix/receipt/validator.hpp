#pragma once

#include "strix/receipt/receipt.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string_view>

namespace strix::receipt {

namespace detail {
inline void require(bool condition, std::string_view error) {
    if (!condition) throw std::invalid_argument(std::string(error));
}

inline bool is_hex_id(std::string_view id) {
    return id.size() == 16 && std::ranges::all_of(id, [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

inline void nonnegative(std::optional<double> value, std::string_view field) {
    require(!value || (std::isfinite(*value) && *value >= 0.0), field);
}

template<class Rows, class Key>
void unique(const Rows& rows, Key key, std::string_view field) {
    std::set<std::string_view> seen;
    for (const auto& row : rows) {
        const std::string_view name = key(row);
        require(!name.empty() && seen.insert(name).second, field);
    }
}
}  // namespace detail

inline void validate(const Receipt& receipt) {
    using detail::nonnegative;
    using detail::require;
    const auto& id = receipt.identity;
    require(id.route_fingerprint.empty() || detail::is_hex_id(id.route_fingerprint), "identity.route_fingerprint");
    detail::unique(id.programs, [](const auto& value) -> const auto& { return value; }, "identity.programs");
    detail::unique(id.digests, [](const auto& pair) -> const auto& { return pair.first; }, "identity.digests");
    if (receipt.resolution) {
        detail::unique(receipt.resolution->levers, [](const Lever& row) -> const auto& { return row.name; },
                       "resolution.levers");
    }
    if (receipt.step) for (const auto& row : *receipt.step) {
        nonnegative(row.predicted_ms, "step.predicted_ms");
        nonnegative(row.wall_ms, "step.wall_ms");
        require(row.guard_failures <= row.guards, "step.guard_failures exceeds guards");
    }
    if (receipt.speculation) {
        require(receipt.speculation->accepted <= receipt.speculation->proposed, "speculation.accepted exceeds proposed");
        nonnegative(receipt.speculation->drafter_ms, "speculation.drafter_ms");
    }
    if (receipt.serving) for (const auto& row : *receipt.serving) nonnegative(row.queue_ms, "serving.queue_ms");
    if (receipt.devices) for (const auto& row : *receipt.devices) {
        nonnegative(row.bytes_per_second, "devices.bytes_per_second");
        nonnegative(row.clock_mhz, "devices.clock_mhz");
        nonnegative(row.memory_clock_mhz, "devices.memory_clock_mhz");
        for (const auto& link : row.links) {
            nonnegative(link.latency_ms, "devices.links.latency_ms");
            nonnegative(link.bytes_per_second, "devices.links.bytes_per_second");
        }
        for (const auto& sample : row.telemetry) {
            nonnegative(sample.elapsed_ms, "devices.telemetry.elapsed_ms");
            nonnegative(sample.clock_mhz, "devices.telemetry.clock_mhz");
            nonnegative(sample.memory_clock_mhz, "devices.telemetry.memory_clock_mhz");
            nonnegative(sample.power_w, "devices.telemetry.power_w");
        }
    }
}

}  // namespace strix::receipt
