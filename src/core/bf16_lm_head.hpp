#pragma once

#include "core/dense_row_plan.hpp"
#include "core/row_transforms.hpp"

namespace strix::core {

struct Bf16HeadShape {
    unsigned input_width;
    unsigned stored_outputs;
    unsigned live_outputs;
    float softcap;
};

inline row_transforms::Status enqueue_bf16_head_softcap(
    unsigned rows, unsigned outputs, float* logits, float cap, row_transforms::Stream stream) {
    return row_transforms::elements(std::size_t{rows} * outputs,
        row_transforms::Bf16Softcap{logits, cap}, stream);
}

// Project only the live vocabulary. Keep the BF16 boundary before softcap
// and the result in FP32 for scoring or the backend's shared greedy picker.
template <class Project>
row_transforms::Status enqueue_bf16_softcap_head(unsigned rows, const Bf16RowProjection& projection,
                                                       float cap, row_transforms::Stream stream,
                                                       const Project& project) {
    const auto status = project(std::span<const Bf16RowProjection>{&projection, 1U});
#if defined(__HIPCC__)
    if (status != hipSuccess) return status;
#else
    if (status != cudaSuccess) return status;
#endif
    return enqueue_bf16_head_softcap(rows, projection.outputs, projection.output, cap, stream);
}

template <class Project>
row_transforms::Status enqueue_bf16_softcap_head(unsigned rows, const Bf16HeadShape& shape,
                                                       std::span<const std::uint16_t> weights, std::span<float> logits,
                                                       row_transforms::Stream stream, const Project& project) {
    if (shape.input_width == 0 || shape.live_outputs == 0 || shape.live_outputs > shape.stored_outputs ||
        weights.size() / shape.input_width != shape.stored_outputs || weights.size() % shape.input_width != 0 ||
        rows > logits.size() / shape.live_outputs) return row_transforms::invalid;
    return enqueue_bf16_softcap_head(rows, {weights.data(), shape.live_outputs, logits.data()}, shape.softcap,
                                     stream, project);
}

}  // namespace strix::core
