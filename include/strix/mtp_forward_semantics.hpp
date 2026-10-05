#pragma once

#include "strix/model_contract.hpp"

#include <cstdint>

namespace strix {

enum class MtpInputConcatOrder : std::uint8_t {
    embedding_then_hidden,
};

enum class MtpNextStatePublicationPoint : std::uint8_t {
    post_mlp_combine_pre_head_mix,
};

struct MtpForwardSemanticContract {
    std::uint32_t semantic_revision = 1;

    std::uint64_t hidden_size = 0;
    std::uint64_t hc_count = 0;
    std::uint64_t wide_hidden_size = 0;
    std::uint64_t eh_proj_input_size = 0;
    std::uint64_t eh_proj_output_size = 0;

    bool hidden_rms_per_hc_stream = false;
    bool token_rms_repeated_across_hc = false;
    MtpInputConcatOrder input_concat_order =
        MtpInputConcatOrder::embedding_then_hidden;

    bool attention_precedes_mlp = false;
    bool attention_combined_before_mlp_mix = false;
    bool mlp_combined_before_state_publish = false;

    MtpNextStatePublicationPoint next_state_publication =
        MtpNextStatePublicationPoint::
            post_mlp_combine_pre_head_mix;

    bool head_mix_after_state_publish = false;
    bool head_mix_has_injection = true;
    bool lm_head_after_head_mix = false;

    std::uint64_t contract_identity = 0;
};

[[nodiscard]] MtpForwardSemanticContract
build_mtp_forward_semantic_contract(
    const ModelContract& model);

void validate_mtp_forward_semantic_contract(
    const ModelContract& model,
    const MtpForwardSemanticContract& contract);

}  // namespace strix
