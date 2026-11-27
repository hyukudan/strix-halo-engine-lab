#pragma once

#include "core/dense_row_plan.hpp"
#include "core/row_transforms.hpp"
#include "strix/core/logit_epilogue.hpp"
#include "strix/logit_rows.hpp"

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

// Linear / quadratic hinges on FP32 logit rows, in place (row_transforms::LogitHinges): the head epilogue of a
// LinearQuadraticHinges vocabulary activation. A template, so its kernel is instantiated only where it is called.
template <class Hinges = LinearQuadraticHinges>
row_transforms::Status enqueue_logit_hinges(unsigned rows, unsigned outputs, float* logits, const Hinges& hinges,
                                            bool preserve_mask, row_transforms::Stream stream) {
    const auto& t = hinges.thresholds;
    const auto& g = hinges.gains;
    return row_transforms::elements(std::size_t{rows} * outputs,
        row_transforms::LogitHinges{logits, t[0], t[1], t[2], t[3], g[0], g[1], g[2], g[3], preserve_mask}, stream);
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

// Softcapped score rows of one head at a row pitch (verify windows, cohort lanes): row r lands at
// head::logit_row_offset(r, pitch), so every consumer keeps one pitch for any row count. Each launch projects
// `pitch` outputs per row (the live outputs plus pad head rows nobody reads; strix/logit_rows.hpp) for at most
// `max_rows` rows, the cap up to which the caller's row family keeps every row on its one-row arithmetic (HIP BF16
// tight rows: gemv_rows::max_rows). The weights are read once per launch instead of once per row.
// project(first, rows, projections) issues the dense rows for input rows first .. first + rows - 1.
template <class ProjectRows>
row_transforms::Status enqueue_bf16_softcap_head_rows(unsigned rows, unsigned max_rows, const Bf16HeadShape& shape,
                                                      unsigned pitch, std::span<const std::uint16_t> weights,
                                                      std::span<float> logits, row_transforms::Stream stream,
                                                      const ProjectRows& project) {
    if (shape.input_width == 0 || max_rows == 0 || weights.size() % shape.input_width != 0 ||
        weights.size() / shape.input_width != shape.stored_outputs ||
        !head::pitched_rows_fit(rows, shape.live_outputs, pitch, shape.stored_outputs, logits.size()))
        return row_transforms::invalid;
    for (unsigned first = 0; first < rows;) {
        const auto launch = static_cast<unsigned>(head::logit_rows_launch(rows, first, max_rows).rows);
        const Bf16RowProjection projection{weights.data(), pitch, logits.data() + head::logit_row_offset(first, pitch)};
        const auto status = enqueue_bf16_softcap_head(
            launch, projection, shape.softcap, stream,
            [&](std::span<const Bf16RowProjection> projections) { return project(first, launch, projections); });
#if defined(__HIPCC__)
        if (status != hipSuccess) return status;
#else
        if (status != cudaSuccess) return status;
#endif
        first += launch;
    }
#if defined(__HIPCC__)
    return hipSuccess;
#else
    return cudaSuccess;
#endif
}

}  // namespace strix::core
