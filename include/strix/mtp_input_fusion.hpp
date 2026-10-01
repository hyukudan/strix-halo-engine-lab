#pragma once

#include "strix/model_contract.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct MtpInputFusionPlan {
    std::uint64_t hidden_size = 0;
    std::uint64_t hc_count = 0;
    std::uint64_t wide_hidden_size = 0;
    std::uint64_t fc_embedding_elements = 0;
    std::uint64_t fc_hidden_elements = 0;
};

[[nodiscard]] MtpInputFusionPlan
build_mtp_input_fusion_plan(
    const ModelContract& model);

[[nodiscard]] std::vector<double>
mtp_input_fusion_reference(
    const MtpInputFusionPlan& plan,
    std::span<const double> normalized_embedding,
    std::span<const double> normalized_wide_hidden,
    std::span<const double> fc_embedding_row_major,
    std::span<const double> fc_hidden_row_major);

}  // namespace strix
