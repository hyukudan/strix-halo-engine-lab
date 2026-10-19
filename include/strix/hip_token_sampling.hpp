#pragma once
#include "strix/execution_memory.hpp"
#include <cstdint>
#include <vector>
namespace strix {
// Copies one FP32 vocabulary row from device memory into `out` on `stream`
// and waits for it; the producer of the row must already be ordered before.
void read_hip_logits_row(const ExecutionBufferAddress& row,std::uint32_t vocabulary,void* stream,std::vector<float>& out);
// STRIX_SAMPLE_TIMING=1: host wall time of sampled decisions (logits readback and the decision itself) per
// kind and window rows, printed to stderr at exit. Inert otherwise.
// kernels: device sampler kernel time (events around its kernels; "decide" column), rows = rows per call.
enum class SampleTimingKind : unsigned {window = 0, serial = 1, draft = 2, kernels = 3};
[[nodiscard]] bool sample_timing_enabled() noexcept;
void record_sample_timing(SampleTimingKind kind,std::uint32_t rows,double readback_seconds,double decide_seconds,
    std::uint32_t decided_rows);
} // namespace strix
