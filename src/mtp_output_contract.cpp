#include "strix/mtp_output_contract.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {

std::uint64_t checked_mul(
    std::uint64_t left,
    std::uint64_t right) {
    if (left != 0 &&
        right >
            std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(
            "MTP output geometry overflow");
    }
    return left * right;
}

}  // namespace

MtpOutputStructuralContract
build_mtp_output_structural_contract(
    const ModelContract& model) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }

    MtpOutputStructuralContract contract;
    contract.wide_input_size =
        checked_mul(
            model.hc_count,
            model.hidden_size);
    contract.mixed_hidden_size =
        model.hidden_size;
    contract.vocab_size =
        model.vocab_size;
    contract.uses_gated_residual_head_mix =
        true;
    contract.head_mix_has_injection =
        false;
    contract.lm_head_required =
        true;
    contract.mixer_semantics_resolved =
        true;
    contract.head_binding_resolved =
        false;
    contract.output_semantics_resolved =
        false;
    return contract;
}

void validate_mtp_output_structural_contract(
    const ModelContract& model,
    const MtpOutputStructuralContract& contract) {
    const auto expected =
        build_mtp_output_structural_contract(
            model);

    if (contract.wide_input_size !=
            expected.wide_input_size ||
        contract.mixed_hidden_size !=
            expected.mixed_hidden_size ||
        contract.vocab_size !=
            expected.vocab_size ||
        !contract.uses_gated_residual_head_mix ||
        contract.head_mix_has_injection ||
        !contract.lm_head_required ||
        !contract.mixer_semantics_resolved ||
        contract.head_binding_resolved ||
        contract.output_semantics_resolved) {
        throw std::invalid_argument(
            "MTP output structural contract is stale, incompatible or overclaims head binding");
    }
}

MtpOutputStructuralContract
bind_mtp_output_head(
    const ModelContract& model,
    const ModelManifest& manifest,
    const MtpDraftTensorAudit& audit) {
    auto contract =
        build_mtp_output_structural_contract(
            model);

    if (audit.manifest_layout_identity !=
            manifest.layout_identity) {
        throw std::invalid_argument(
            "MTP output head audit/manifest identity mismatch");
    }

    for (const auto& family :
         audit.families) {
        if (family.family !=
            MtpTensorFamily::output_head) {
            continue;
        }
        if (family.tensor_indices.size() > 1) {
            throw std::invalid_argument(
                "MTP sidecar contains ambiguous output-head tensors");
        }
        if (family.tensor_indices.size() == 1) {
            const auto index =
                family.tensor_indices.front();
            if (index >=
                manifest.tensors.size()) {
                throw std::invalid_argument(
                    "MTP output-head tensor index exceeds manifest");
            }
            contract.lm_head_source =
                MtpLmHeadSource::
                    mtp_sidecar_shared_head;
            contract.lm_head_tensor_index =
                index;
            contract.head_binding_resolved =
                true;
            contract.output_semantics_resolved =
                contract.mixer_semantics_resolved;
        }
    }

    return contract;
}

}  // namespace strix
