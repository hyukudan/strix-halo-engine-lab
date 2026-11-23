#pragma once
// HIP binding of the shared FAST gate v2 route tap (include/strix/moe/route_file.hpp, route_tap_exchange.hpp), the
// twin of include/strix/cuda/core/moe_route_tap.hpp. A HIP MoE engine calls route_tap() once per router call, right
// after its router wrote the selected ids and weights and before the experts read them. Inactive (no
// STRIX_MOE_ROUTES_OUT / _IN / STRIX_MOE_ROUTE_NULL / _STATS): one cached test, no sync, no copy. Active: one stream
// sync and two
// contiguous copies each way per call; a stream being captured into a graph is refused (replay needs eager execution).
#include "strix/moe/route_tap_exchange.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strix::moe_core {

inline bool route_tap_active() {
    static const bool on = strix::moe::RouteTap::instance().active();
    return on;
}

struct HipRouteCopier {
    hipStream_t stream;
    static void check(hipError_t e, const char* what) {
        if (e != hipSuccess) throw std::runtime_error(std::string("MoE route tap: ") + what + ": " + hipGetErrorString(e));
    }
    void to_host(void* host, const void* device, std::size_t bytes) {
        check(hipMemcpyAsync(host, device, bytes, hipMemcpyDeviceToHost, stream), "copy to host");
    }
    void to_device(void* device, const void* host, std::size_t bytes) {
        check(hipMemcpyAsync(device, host, bytes, hipMemcpyHostToDevice, stream), "copy to device");
    }
    void wait() { check(hipStreamSynchronize(stream), "sync"); }
};

// ids / weights: device, contiguous [rows][top_k]. layer: the model layer of this router call.
inline void route_tap(strix::moe::RouteTap& tap, std::uint32_t layer, std::uint32_t rows, std::uint32_t top_k,
                      std::uint32_t experts, std::uint32_t* ids, float* weights, hipStream_t stream) {
    if (!tap.active()) return;
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    HipRouteCopier::check(hipStreamIsCapturing(stream, &capture), "capture query");
    if (capture != hipStreamCaptureStatusNone)
        throw std::runtime_error("MoE route tap needs eager execution: the router stream is being captured into a graph");
    HipRouteCopier copier{stream};
    (void)strix::moe::route_tap_exchange(tap, copier, layer, rows, top_k, experts, ids, weights);
}

inline void route_tap(std::uint32_t layer, std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts,
                      std::uint32_t* ids, float* weights, hipStream_t stream) {
    if (!route_tap_active()) return;
    route_tap(strix::moe::RouteTap::instance(), layer, rows, top_k, experts, ids, weights, stream);
}

inline void route_tap(std::uint32_t layer, std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts,
                      std::uint32_t* ids, float* weights, hipStream_t stream, strix::moe::RouteTap* tap) {
    if (tap) route_tap(*tap, layer, rows, top_k, experts, ids, weights, stream);
    else route_tap(layer, rows, top_k, experts, ids, weights, stream);
}

}  // namespace strix::moe_core
