#pragma once

#include "strix/q51_grouped_down_contract.hpp"

#include <cstdint>
#include <string_view>

namespace strix::q51_slot_parallel {
inline constexpr std::string_view kernel_suffix = ".q51-slot-parallel-v1";
inline constexpr std::string_view numerical_identity =
    "q51-slot-parallel-down-v1:original-f32-activation:virtual256-tree:"
    "rawdot-f32-scratch:ascending-slot-weight-src0-fma:positive-zero-seed";
using Shape = q51_grouped_down::Shape;
using DeviceBytes = q51_grouped_down::DeviceBytes;
struct Request {
    bool enabled = false;
    std::string_view approved_numerical_identity;
    Shape shape;
};
struct Bindings {
    DeviceBytes intermediates, weights, routes, route_weights, status, accumulation, rawdots;
};
[[nodiscard]] std::uint64_t required_rawdots_bytes(Shape shape);
// Exact grouped views plus an exact, 256-byte-aligned raw-dot destination.
// Raw dots are writable and must be disjoint from all six grouped views.
// This preflight does not clear, inspect or publish device memory.
void validate(const Request&, const Bindings&);
} // namespace strix::q51_slot_parallel
