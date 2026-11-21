#pragma once
#include "strix/token_sampling_timing.hpp"
#include "strix/execution_memory.hpp"
#include <cstdint>
#include <vector>
namespace strix {
// Copies one FP32 vocabulary row from device memory into `out` on `stream`
// and waits for it; the producer of the row must already be ordered before.
void read_hip_logits_row(const ExecutionBufferAddress& row,std::uint32_t vocabulary,void* stream,std::vector<float>& out);
// STRIX_SAMPLE_TIMING=1: host wall time of sampled decisions (logits readback and the decision itself) per
// kind and window rows, printed to stderr at exit. Inert otherwise.
// kernels / draft-kernels: device sampler kernel time of target rows / draft steps (events around its kernels;
// "decide" column), rows = rows per call.
} // namespace strix
