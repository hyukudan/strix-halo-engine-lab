#pragma once

#include "core/dense_row_plan.hpp"
#include "core/prepared_rows_device.hpp"

namespace strix::core {

struct Bf16HeadShape {
    unsigned input_width;
    unsigned stored_outputs;
    unsigned live_outputs;
    float softcap;
};

// Project only the live vocabulary. Keep the BF16 boundary before softcap
// and the result in FP32 for scoring or the backend's shared greedy picker.
template <class Project>
prepared_rows_device::Status enqueue_bf16_softcap_head(unsigned rows, const Bf16RowProjection& projection,
                                                       float cap, prepared_rows_device::Stream stream,
                                                       const Project& project) {
    const auto status = project(std::span<const Bf16RowProjection>{&projection, 1U});
#if defined(__HIPCC__)
    if (status != hipSuccess) return status;
#else
    if (status != cudaSuccess) return status;
#endif
    return prepared_rows_device::elements(std::size_t{rows} * projection.outputs,
        prepared_rows_device::Bf16Softcap{projection.output, cap}, stream);
}

template <class Project>
prepared_rows_device::Status enqueue_bf16_softcap_head(unsigned rows, const Bf16HeadShape& shape,
                                                       std::span<const std::uint16_t> weights, std::span<float> logits,
                                                       prepared_rows_device::Stream stream, const Project& project) {
    if (shape.input_width == 0 || shape.live_outputs == 0 || shape.live_outputs > shape.stored_outputs ||
        weights.size() / shape.input_width != shape.stored_outputs || weights.size() % shape.input_width != 0 ||
        rows > logits.size() / shape.live_outputs) return prepared_rows_device::invalid;
    return enqueue_bf16_softcap_head(rows, {weights.data(), shape.live_outputs, logits.data()}, shape.softcap,
                                     stream, project);
}

}  // namespace strix::core
