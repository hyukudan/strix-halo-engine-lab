#pragma once

#include "strix/model_contract.hpp"

#include <cstdint>

namespace strix {

// Arithmetic profiles are distinct even when weights share packed storage.
enum class MtpInputFusionArithmetic : std::uint8_t {
    split_projections_then_add,
    packed_concat_chain,
};
enum class MtpNormWeightEncoding : std::uint8_t {
    gemma_offset, // Source parameter: effective gain is 1 + weight.
    effective_scale, // Converted checkpoint gain: NEVER add one again.
};
// The predictor consumes hidden[i] and token[i+1] at rotary/cache position i.
enum class MtpPredictorPosition : std::uint8_t { hidden_source };

enum class MtpHiddenNormScope : std::uint8_t { wide_residual };

enum class MtpInputConcatOrder : std::uint8_t {
    embedding_then_hidden,
};

enum class MtpNextStatePublicationPoint : std::uint8_t {
    post_mlp_combine_pre_head_mix,
};

struct MtpForwardSemanticContract {
    std::uint32_t semantic_revision = 3;

    MtpPredictorPosition predictor_position = MtpPredictorPosition::hidden_source;

    std::uint64_t hidden_size = 0;
    std::uint64_t hc_count = 0;
    std::uint64_t wide_hidden_size = 0;
    // Packed storage geometry, not a claim that one continuous reduction
    // equals the canonical separate projections and addition.
    std::uint64_t eh_proj_input_size = 0;
    std::uint64_t eh_proj_output_size = 0;

    // HC>1 normalizes the WHOLE HC*H residual before reshaping into streams.
    std::uint64_t hidden_norm_elements = 0;
    float rms_norm_epsilon = 0.0F;
    bool embedding_projected_once_then_broadcast = false;
    MtpHiddenNormScope hidden_norm_scope = MtpHiddenNormScope::wide_residual;
    MtpNormWeightEncoding source_norm_weight_encoding = MtpNormWeightEncoding::gemma_offset;
    MtpInputFusionArithmetic input_arithmetic = MtpInputFusionArithmetic::split_projections_then_add;
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
    const ModelContract& model,
    MtpInputFusionArithmetic arithmetic = MtpInputFusionArithmetic::split_projections_then_add);

void validate_mtp_forward_semantic_contract(
    const ModelContract& model,
    const MtpForwardSemanticContract& contract);

}  // namespace strix
