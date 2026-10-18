#pragma once
#include "strix/execution_memory.hpp"
#include <cstdint>
#include <vector>
namespace strix {
// Copies one FP32 vocabulary row from device memory into `out` on `stream`
// and waits for it; the producer of the row must already be ordered before.
void read_hip_logits_row(const ExecutionBufferAddress& row,std::uint32_t vocabulary,void* stream,std::vector<float>& out);
} // namespace strix
