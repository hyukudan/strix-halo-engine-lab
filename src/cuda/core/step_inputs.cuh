#pragma once

#include "core/step_inputs_plan.hpp"

#if defined(__CUDACC__)
#include <cuda_runtime.h>

namespace strix::cuda::core::step_inputs {

// The caller orders descriptor/sidecar producers and consumers on the stream.
cudaError_t enqueue(const exec::StepBlock* step, Linear v, cudaStream_t stream);
cudaError_t enqueue(const exec::StepBlock* step, Tree v, cudaStream_t stream);
cudaError_t enqueue(const exec::StepBlock* step, FedRow v, cudaStream_t stream);
cudaError_t enqueue(const exec::StepBlock* step, FedRows v, cudaStream_t stream);
cudaError_t enqueue(const exec::StepBlock* step, Slots v, cudaStream_t stream);

}  // namespace strix::cuda::core::step_inputs
#endif
