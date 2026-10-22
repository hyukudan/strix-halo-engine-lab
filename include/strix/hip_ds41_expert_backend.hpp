#pragma once

// HIP backend of the deepseek41 expert server (ds41_expert_server.hpp): keeps the routed experts [first, end) of the
// listed layers resident in device memory (no dense weights) and computes the requested (row, slot) entries with the
// decode kernels of hip_ds41_formats (fused IQ2_XXS gate/up + SwiGLU + A8, Q2_K down: the raw slot vectors, bit for
// bit the host reference), then the answer items (FOLD of leading remote slots, RAW vectors) on the device.

#include "strix/ds41_expert_server.hpp"

#include <memory>
#include <string>

namespace strix::ds41::xs {

// layer_mask: layers to load (those absent from the file are skipped); end 0 = all experts. max_rows bounds the rows
// of one request (buffers are sized for max_rows * top_k entries). spin: the CPU spins on device completion
// (hipDeviceScheduleSpin) instead of sleeping in the driver: lower and steadier latency after idle gaps. keep_warm_us:
// after that long without a request, a trivial kernel every keep_warm_us keeps the GPU out of its idle state (0: off).
[[nodiscard]] std::unique_ptr<xrpc::ExpertBackend> make_hip_backend(const std::string& gguf, std::uint64_t layer_mask, std::uint32_t first,
                                                                    std::uint32_t end, xrpc::Geometry* geometry_out, std::uint32_t max_rows,
                                                                    bool spin = false, std::uint32_t keep_warm_us = 0);

}  // namespace strix::ds41::xs
