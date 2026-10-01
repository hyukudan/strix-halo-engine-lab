#pragma once

#include "strix/model_contract.hpp"

#include <cstdint>

namespace strix {

enum class MtpDraftStateSource {
    post_block_wide_residual_pre_head_mix,
};

struct MtpDraftStateContract {
    std::uint64_t hidden_size = 0;
    std::uint64_t hc_count = 0;
    std::uint64_t state_width = 0;
    MtpDraftStateSource source =
        MtpDraftStateSource::
            post_block_wide_residual_pre_head_mix;
    bool one_state_row_per_request = true;
};

[[nodiscard]] MtpDraftStateContract
build_mtp_draft_state_contract(
    const ModelContract& model);

void validate_mtp_draft_state_contract(
    const ModelContract& model,
    const MtpDraftStateContract& contract);

}  // namespace strix
