#pragma once

#include "strix/hip_lane_chain_hyper_projection_tests.hpp"

namespace strix::hc_repacked_projection_diagnostic {

// Diagnostic only: no catalog, application flag, or production dispatch selects
// these schedules. The caller retains every owner until the stream is drained.
enum class Schedule : std::uint32_t {
    s1_lds_k8_d8 = 1,
    s2_lds_k11_d8 = 2,
    s3_scalar_k8_d8 = 3,
};

void enqueue(
    hc_projection_diagnostic::Shape shape,
    hc_projection_diagnostic::Span normalized,
    hc_projection_diagnostic::Span codes,
    hc_projection_diagnostic::Span scales,
    std::uint32_t rank_pad,
    hc_projection_diagnostic::Span inject,
    hc_projection_diagnostic::Span low,
    hc_projection_diagnostic::Span injection,
    Schedule schedule,
    hipStream_t stream);

}  // namespace strix::hc_repacked_projection_diagnostic
