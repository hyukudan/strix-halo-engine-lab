#pragma once
// Backend-neutral exchange of one router call with the FAST gate v2 route tap (route_file.hpp): copy the selected ids
// and weights of a device router call to host, hand them to RouteTap::on_call, and copy them back only when the tap
// changed them (replay or one-ulp null). A backend supplies the copies; the sequence, the inactive fast path and the
// "write back only on change" rule live here once (host tests: tests/moe_route_file_tests.cpp).
//
// Copier concept (all calls on the router's stream, in order):
//   void to_host(void* host, const void* device, std::size_t bytes);
//   void to_device(void* device, const void* host, std::size_t bytes);
//   void wait();                         // the host buffers are valid / the device copies have landed
// Rows are contiguous: ids[rows][top_k] (u32) and weights[rows][top_k] (f32).

#include "strix/moe/route_file.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strix::moe {

// Returns true when the device routes were overwritten. An inactive tap returns before touching the copier.
template <class Copier>
bool route_tap_exchange(RouteTap& tap, Copier& copier, std::uint32_t layer, std::uint32_t rows, std::uint32_t top_k,
                        std::uint32_t experts, std::uint32_t* device_ids, float* device_weights) {
    if (!tap.active() || rows == 0U) return false;
    if (top_k == 0 || top_k > kRouteMaxTopK || top_k > experts)
        throw std::invalid_argument("route tap: unsupported top-k for this expert table");
    const std::size_t n = std::size_t{rows} * top_k;
    std::vector<std::uint32_t> ids(n);
    std::vector<float> weights(n);
    copier.to_host(ids.data(), device_ids, n * sizeof(std::uint32_t));
    copier.to_host(weights.data(), device_weights, n * sizeof(float));
    copier.wait();
    if (!tap.on_call(layer, rows, top_k, ids.data(), weights.data())) return false;
    for (std::size_t i = 0; i < n; ++i) {
        if (ids[i] >= experts || !std::isfinite(weights[i]))
            throw std::runtime_error("route tap: invalid expert id or weight for this router");
    }
    copier.to_device(device_ids, ids.data(), n * sizeof(std::uint32_t));
    copier.to_device(device_weights, weights.data(), n * sizeof(float));
    copier.wait();  // the host vectors die here; the experts read the device copies next on the same stream
    return true;
}

}  // namespace strix::moe
