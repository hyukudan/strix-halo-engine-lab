#include "step_inputs.cuh"

namespace strix::cuda::core::step_inputs {
namespace {

template <class View>
__global__ void scatter(const exec::StepBlock* step, View v) {
    detail::scatter_row(*step, v, threadIdx.x);
}

template <class View>
cudaError_t launch(const exec::StepBlock* step, View v, cudaStream_t stream) {
    scatter<<<1, kRows, 0, stream>>>(step, v);
    return cudaGetLastError();
}

}  // namespace

cudaError_t enqueue(const exec::StepBlock* step, Linear v, cudaStream_t stream) { return launch(step, v, stream); }
cudaError_t enqueue(const exec::StepBlock* step, Tree v, cudaStream_t stream) { return launch(step, v, stream); }
cudaError_t enqueue(const exec::StepBlock* step, FedRow v, cudaStream_t stream) { return launch(step, v, stream); }
cudaError_t enqueue(const exec::StepBlock* step, FedRows v, cudaStream_t stream) { return launch(step, v, stream); }
cudaError_t enqueue(const exec::StepBlock* step, Slots v, cudaStream_t stream) { return launch(step, v, stream); }

}  // namespace strix::cuda::core::step_inputs
