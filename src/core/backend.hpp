#pragma once
#include <cstddef>
#include <cstdint>
#include "linear_plan.hpp"
namespace strix::core {
struct DeviceWeights {
    const uint32_t* trellis;
    const uint16_t* suh;
    const uint16_t* svh;
    const float* bias;
    int input;
    int output;
    int bits;
    int codebook;
    bool fused_output_capable = false;
    bool fused_both_capable = false;
};
// Complete operator, original-basis row-major FP32 tensors. No allocations,
// transfers or synchronization. Caller-owned 256-byte-aligned workspace.
// Integer bits 1..8 initially. bias is optional device FP32 [output].
size_t linear_workspace_bytes(const DeviceWeights& weights, int rows);
// K partitions of launch_linear; fixes the numerics of the partial sums.
int linear_split_count(const DeviceWeights& weights, int rows);
LinearExecution resolve_linear(const DeviceWeights& weights, const float* input,
    float* output, int rows, bool prefer_wmma, bool completion = false);
LinearExecution resolve_linear_with_policy(const DeviceWeights& weights, const float* input,
    float* output, int rows, bool prefer_wmma, bool completion, PhysicalPolicy policy);
// Completion counter words a prepared plan needs to run this call as one launch (0: not applicable).
// The counters must be zero before the first launch; every launch leaves them zero.
size_t linear_completion_counters(const DeviceWeights& weights, int rows, bool prefer_wmma);
}
