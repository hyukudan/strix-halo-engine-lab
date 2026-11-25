#pragma once

#include "core/lookup_plan.hpp"

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace strix::cuda::core::lookup {

#if defined(__CUDACC__)
// Buffers are non-overlapping device addresses, ordered by the caller's stream.
// X3 uses resolved StepRow::token and validity in program order. Masked rows
// leave outputs and status untouched. Out-of-range active IDs leave the row
// untouched except in the row-zero form, which preserves MTP's fallback.
cudaError_t enqueue(View view, TokenIds ids, cudaStream_t stream);
cudaError_t enqueue(View view, StepTokens ids, cudaStream_t stream);
#endif

}  // namespace strix::cuda::core::lookup
