#include "strix/mtp_output_contract.hpp"

#include <limits>
#include <stdexcept>
#include <string_view>

namespace strix {
namespace {

bool contains(
    std::string_view value,
    std::string_view needle) noexcept {
    return value.find(needle) !=
        std::string_view::npos;
}

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
        contract.head_mix_binding_resolved ||
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

    const auto wide =
        checked_mul(
            model.hidden_size,
            model.hc_count);

    std::optional<std::uint32_t>
        head_norm;
    std::optional<std::uint32_t>
        head_down;
    std::optional<std::uint32_t>
        head_up;

    for (std::uint32_t index = 0;
         index < manifest.tensors.size();
         ++index) {
        const auto& tensor =
            manifest.tensors[index];
        if (tensor.role !=
            TensorRole::mtp) {
            continue;
        }

        auto bind_unique =
            [&](std::optional<std::uint32_t>& slot,
                std::string_view needle) {
                if (!contains(
                        tensor.name,
                        needle)) {
                    return;
                }
                if (slot.has_value()) {
                    throw std::invalid_argument(
                        "MTP output head-mix tensor binding is ambiguous");
                }
                slot = index;
            };

        bind_unique(
            head_norm,
            "hc_head_norm");
        bind_unique(
            head_down,
            "hc_head_down");
        bind_unique(
            head_up,
            "hc_head_up");
    }

    if (head_norm.has_value() ||
        head_down.has_value() ||
        head_up.has_value()) {
        if (!head_norm.has_value() ||
            !head_down.has_value() ||
            !head_up.has_value()) {
            throw std::invalid_argument(
                "MTP output head mix requires norm/down/up tensor triplet");
        }

        const auto& norm =
            manifest.tensors[
                *head_norm];
        const auto& down =
            manifest.tensors[
                *head_down];
        const auto& up =
            manifest.tensors[
                *head_up];

        const bool norm_geometry =
            (norm.dimensions.size() == 2 &&
             norm.dimensions[0] ==
                 model.hidden_size &&
             norm.dimensions[1] ==
                 model.hc_count) ||
            (norm.dimensions.size() == 1 &&
             norm.dimensions[0] ==
                 wide);
        const bool down_geometry =
            down.dimensions.size() == 2 &&
            down.dimensions[0] ==
                wide &&
            down.dimensions[1] ==
                model.hc_lowrank;
        const bool up_geometry =
            up.dimensions.size() == 2 &&
            up.dimensions[0] ==
                model.hc_lowrank &&
            up.dimensions[1] ==
                wide;

        if (!norm_geometry ||
            !down_geometry ||
            !up_geometry) {
            throw std::invalid_argument(
                "MTP output head-mix tensor geometry does not match model contract");
        }

        contract.head_mix_binding_resolved =
            true;
        contract.head_norm_tensor_index =
            *head_norm;
        contract.head_down_tensor_index =
            *head_down;
        contract.head_up_tensor_index =
            *head_up;
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
                contract.mixer_semantics_resolved &&
                contract.head_mix_binding_resolved;
        }
    }

    return contract;
}

}  // namespace strix
