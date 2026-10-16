#pragma once

#include "strix/model_contract.hpp"
#include "strix/mtp_forward_semantics.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct MtpNormalizedInputReference {
    std::vector<double> embedding;
    std::vector<double> wide_hidden;
};
struct MtpInputFusionPlan {
    std::uint64_t hidden_size = 0;
    std::uint64_t hc_count = 0;
    std::uint64_t wide_hidden_size = 0;
    std::uint64_t fc_embedding_elements = 0;
    std::uint64_t fc_hidden_elements = 0;
    std::uint64_t eh_proj_input_size = 0;
    std::uint64_t eh_proj_elements = 0;
};

[[nodiscard]] MtpInputFusionPlan
build_mtp_input_fusion_plan(
    const ModelContract& model);

// Mathematical FP64 oracle, not an upstream dtype/rounding qualification.
// Explicit encoding prevents applying Gemma's +1 twice to converted GGUF gains.
[[nodiscard]] MtpNormalizedInputReference mtp_input_normalization_reference(
    const MtpInputFusionPlan&, std::span<const double> embedding,
    std::span<const double> wide_hidden, std::span<const double> embedding_weights,
    std::span<const double> hidden_weights, double epsilon, MtpNormWeightEncoding);

// Diagnostic FP32 arithmetic profiles over the SAME packed [We,Wh] storage.
// Split uses two separately rounded dot products followed by a rounded add;
// packed uses one continuous chain. Neither claims native BF16 equivalence.
[[nodiscard]] std::vector<float> mtp_input_fusion_fp32_reference(
    const MtpInputFusionPlan&, std::span<const float> normalized_embedding,
    std::span<const float> normalized_wide_hidden, std::span<const float> packed_weights,
    MtpInputFusionArithmetic);

[[nodiscard]] std::vector<double>
mtp_eh_projection_reference(
    const MtpInputFusionPlan& plan,
    std::span<const double> normalized_embedding,
    std::span<const double> normalized_wide_hidden,
    std::span<const double> eh_proj_row_major);

[[nodiscard]] std::vector<double>
mtp_input_fusion_reference(
    const MtpInputFusionPlan& plan,
    std::span<const double> normalized_embedding,
    std::span<const double> normalized_wide_hidden,
    std::span<const double> fc_embedding_row_major,
    std::span<const double> fc_hidden_row_major);

}  // namespace strix
