#include "cuda_decode_wr2.hpp"

#include <cstring>

namespace strix::exec::cuda_decode {
namespace {

namespace packed = moe_core::microscaled_r1;
using Format = weight::microscaled::MatrixFormat;

struct PackedCall {
    packed::StepArgs args;
    Format format;
    unsigned rows;
    bool gate_up;
};
static_assert(std::is_trivially_copyable_v<PackedCall>);

void enqueue_packed(std::span<const std::byte> bytes, const Grid&, void* stream) {
    if (bytes.size() != sizeof(PackedCall)) throw std::invalid_argument("invalid WR2 packed arguments");
    PackedCall call{};
    std::memcpy(&call, bytes.data(), sizeof(call));
    packed::enqueue_cuda_step(call.args, call.format, call.rows, call.gate_up, stream);
}

BoundStepLeaf bind_packed(PackedCall call) {
    if (!supports_wr2_width(call.rows)) throw std::invalid_argument("unsupported WR2 packed width");
    if (call.format != Format::nvfp4 && call.format != Format::mxfp8) {
        throw std::invalid_argument("unsupported WR2 packed format");
    }
    const auto& b = call.args.batch;
    if (!b.hidden || !b.intermediate) throw std::invalid_argument("invalid WR2 packed shape");
    KernelBinding binding;
    binding.fn = enqueue_packed;
    binding.identity = "cuda-sm120-packed-";
    binding.identity += call.format == Format::nvfp4 ? "nvfp4-" : "mxfp8-";
    binding.identity += call.gate_up ? "gate-up-k8-x3-r" : "down-combine-k8-x3-r";
    // v2: the rows launch of the decode body (one CTA per assignment / row); per-row arithmetic is unchanged.
    binding.identity += std::to_string(call.rows) + "-v2";
    const auto grid = call.gate_up
        ? packed::gate_up_grid(b.intermediate, call.rows, packed::kStepTopK, packed::kGeometry.gate_columns)
        : packed::down_combine_grid(b.hidden, call.rows, packed::kGeometry.down_columns);
    binding.grid = {{grid.x, grid.y, 1U}, {packed::kWavesPerBlock * 32U, 1, 1}, 0};
    binding.predicted.launches = 1U;
    const auto bytes = std::as_bytes(std::span(&call, 1U));
    binding.args.assign(bytes.begin(), bytes.end());
    return {std::move(binding), call.rows};
}

}  // namespace

BoundStepLeaf bind_step(const StepPackedGateUp& a, unsigned rows) {
    return bind_packed({a.args, a.format, rows, true});
}

BoundStepLeaf bind_step(const StepPackedDownCombine& a, unsigned rows) {
    return bind_packed({a.args, a.format, rows, false});
}

}  // namespace strix::exec::cuda_decode
