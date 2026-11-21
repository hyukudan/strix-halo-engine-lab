#pragma once

#include <cstdint>

namespace strix {

// STRIX_SAMPLE_TIMING=1 accumulates timings for the process exit report.
enum class SampleTimingKind : unsigned {window = 0, serial = 1, draft = 2, kernels = 3, draft_kernels = 4};
[[nodiscard]] bool sample_timing_enabled() noexcept;
void record_sample_timing(SampleTimingKind kind,std::uint32_t rows,double readback_seconds,double decide_seconds,
    std::uint32_t decided_rows);

} // namespace strix
