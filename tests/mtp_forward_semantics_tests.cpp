#include "strix/mtp_forward_semantics.hpp"

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::
            qwen38_flash_next();
    const auto contract =
        strix::build_mtp_forward_semantic_contract(
            model);

    expect(
        contract.semantic_revision == 3 &&
            contract.hidden_size == 2560 &&
            contract.hc_count == 4 &&
            contract.wide_hidden_size == 10240 &&
            contract.eh_proj_input_size == 5120 &&
            contract.eh_proj_output_size == 2560,
        "MTP semantic contract pins official NextN input/HC geometry");

    expect(
        !contract.hidden_rms_per_hc_stream &&
            contract.hidden_norm_elements == 10240 &&
            contract.rms_norm_epsilon == model.rms_norm_epsilon &&
            contract.embedding_projected_once_then_broadcast &&
            contract.hidden_norm_scope == strix::MtpHiddenNormScope::wide_residual &&
            contract.source_norm_weight_encoding == strix::MtpNormWeightEncoding::gemma_offset &&
            contract.input_arithmetic == strix::MtpInputFusionArithmetic::split_projections_then_add &&
            contract.token_rms_repeated_across_hc &&
            contract.input_concat_order ==
                strix::MtpInputConcatOrder::
                    embedding_then_hidden,
        "MTP revision two pins wide hidden normalization and separate projection arithmetic");

    expect(
        contract.attention_precedes_mlp &&
            contract.attention_combined_before_mlp_mix &&
            contract.mlp_combined_before_state_publish &&
            contract.next_state_publication ==
                strix::MtpNextStatePublicationPoint::
                    post_mlp_combine_pre_head_mix,
        "MTP semantic contract publishes next state only after complete block");

    expect(
        contract.head_mix_after_state_publish &&
            !contract.head_mix_has_injection &&
            contract.lm_head_after_head_mix &&
            contract.contract_identity != 0,
        "MTP semantic contract pins no-injection final head mix and LM-head order");

    strix::validate_mtp_forward_semantic_contract(
        model,
        contract);

    auto wrong =
        contract;
    wrong.head_mix_has_injection = true;
    bool rejected = false;
    try {
        strix::validate_mtp_forward_semantic_contract(
            model,
            wrong);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(
        rejected,
        "MTP semantic contract rejects stale/invented head injection");

    const auto packed = strix::build_mtp_forward_semantic_contract(model,
        strix::MtpInputFusionArithmetic::packed_concat_chain);
    expect(packed.contract_identity != contract.contract_identity,
        "split and packed arithmetic require different semantic identities");
    strix::validate_mtp_forward_semantic_contract(model,packed);
    wrong=contract;wrong.semantic_revision=1;wrong.hidden_rms_per_hc_stream=true;
    rejected=false;try {strix::validate_mtp_forward_semantic_contract(model,wrong);}
    catch(const std::invalid_argument&) {rejected=true;}
    expect(rejected,"stale per-stream RMS revision must not qualify as wide normalization");
    wrong=contract;wrong.input_arithmetic=strix::MtpInputFusionArithmetic::packed_concat_chain;
    rejected=false;try {strix::validate_mtp_forward_semantic_contract(model,wrong);}
    catch(const std::invalid_argument&) {rejected=true;}
    expect(rejected,"arithmetic mutation cannot retain a former profile identity");
    wrong=contract;wrong.source_norm_weight_encoding=strix::MtpNormWeightEncoding::effective_scale;
    rejected=false;try {strix::validate_mtp_forward_semantic_contract(model,wrong);}
    catch(const std::invalid_argument&) {rejected=true;}
    expect(rejected,"source Gemma offset and converted runtime gain must not be conflated");

    wrong=contract;wrong.rms_norm_epsilon=1e-5F;
    rejected=false;try {strix::validate_mtp_forward_semantic_contract(model,wrong);}
    catch(const std::invalid_argument&) {rejected=true;}
    expect(rejected,"wide normalization epsilon is part of the semantic identity");

    if (failures == 0) {
        std::cout
            << "MTP forward semantic contract tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
