#include "moe_core/microscaled_r1.hpp"
#include "strix/device/runtime.hpp"

namespace strix::moe_core::microscaled_r1 {
namespace {

template <unsigned Rows, bool GateUp>
cudaError_t launch_width(const StepArgs& args, packed::MatrixFormat format, cudaStream_t stream) {
    switch (format) {
    case packed::MatrixFormat::nvfp4: return enqueue_step<Rows, packed::MatrixFormat::nvfp4, kStepTopK, GateUp>(args, stream);
    case packed::MatrixFormat::mxfp8: return enqueue_step<Rows, packed::MatrixFormat::mxfp8, kStepTopK, GateUp>(args, stream);
    }
    return cudaErrorInvalidValue;
}

template <bool GateUp>
cudaError_t launch_width(const StepArgs& args, packed::MatrixFormat format, unsigned rows, cudaStream_t stream) {
    switch (rows) {
    case 1U: return launch_width<1, GateUp>(args, format, stream);
    case 2U: return launch_width<2, GateUp>(args, format, stream);
    case 4U: return launch_width<4, GateUp>(args, format, stream);
    case 8U: return launch_width<8, GateUp>(args, format, stream);
    case 16U: return launch_width<16, GateUp>(args, format, stream);
    default: return cudaErrorInvalidValue;
    }
}

}  // namespace

void enqueue_cuda_step(const StepArgs& args, packed::MatrixFormat format, unsigned rows, bool gate_up, void* stream) {
    const auto queue = static_cast<cudaStream_t>(stream);
    const auto status = gate_up ? launch_width<true>(args, format, rows, queue) : launch_width<false>(args, format, rows, queue);
    device::native::check(status, "enqueue WR2 packed leaf");
}

}  // namespace strix::moe_core::microscaled_r1
