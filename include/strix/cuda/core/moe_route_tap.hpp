#pragma once
// CUDA binding of the shared FAST gate v2 route tap (include/strix/moe/route_file.hpp, route_tap_exchange.hpp).
// A CUDA MoE engine calls route_tap() once per router call, right after its router wrote the selected ids and
// weights and before the experts read them. Inactive (no STRIX_MOE_ROUTES_OUT / _IN / STRIX_MOE_ROUTE_NULL / _STATS):
// one cached test, no sync, no copy. Active: one stream sync and two contiguous copies each way per call; a stream being
// captured into a CUDA graph is refused (replay needs eager execution, the engine turns its graphs off when
// route_tap_active()).

#include "strix/moe/route_tap_exchange.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strix::cuda::core {

inline bool route_tap_active() {
    static const bool on = strix::moe::RouteTap::instance().active();
    return on;
}

struct CudaRouteCopier {
    cudaStream_t stream;
    static void check(cudaError_t e, const char* what) {
        if (e != cudaSuccess) throw std::runtime_error(std::string("MoE route tap: ") + what + ": " + cudaGetErrorString(e));
    }
    void to_host(void* host, const void* device, std::size_t bytes) {
        check(cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, stream), "copy to host");
    }
    void to_device(void* device, const void* host, std::size_t bytes) {
        check(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream), "copy to device");
    }
    void wait() { check(cudaStreamSynchronize(stream), "sync"); }
};

// ids / weights: device, contiguous [rows][top_k]. layer: the model layer of this router call.
inline void route_tap(strix::moe::RouteTap& tap, std::uint32_t layer, std::uint32_t rows, std::uint32_t top_k,
                      std::uint32_t experts, std::uint32_t* ids, float* weights, cudaStream_t stream) {
    if (!tap.active()) return;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CudaRouteCopier::check(cudaStreamIsCapturing(stream, &capture), "capture query");
    if (capture != cudaStreamCaptureStatusNone)
        throw std::runtime_error("MoE route tap needs eager execution: the router stream is being captured into a graph");
    CudaRouteCopier copier{stream};
    (void)strix::moe::route_tap_exchange(tap, copier, layer, rows, top_k, experts, ids, weights);
}

inline void route_tap(std::uint32_t layer, std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts,
                      std::uint32_t* ids, float* weights, cudaStream_t stream) {
    if (!route_tap_active()) return;
    route_tap(strix::moe::RouteTap::instance(), layer, rows, top_k, experts, ids, weights, stream);
}

inline void route_tap(std::uint32_t layer, std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts,
                      std::uint32_t* ids, float* weights, cudaStream_t stream, strix::moe::RouteTap* tap) {
    if (tap) route_tap(*tap, layer, rows, top_k, experts, ids, weights, stream);
    else route_tap(layer, rows, top_k, experts, ids, weights, stream);
}

}  // namespace strix::cuda::core
