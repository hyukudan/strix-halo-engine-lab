#include "strix/mtp_draft_tensor_audit.hpp"

#include <algorithm>
#include <array>
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

MtpTensorFamilyAudit* family(
    MtpDraftTensorAudit& audit,
    MtpTensorFamily wanted) noexcept {
    for (auto& item : audit.families) {
        if (item.family == wanted) {
            return &item;
        }
    }
    return nullptr;
}

bool classify(
    std::string_view name,
    MtpTensorFamily& result) noexcept {
    if (contains(name, "fc_embedding") ||
        contains(name, "fc_hidden") ||
        contains(name, "eh_proj")) {
        result =
            MtpTensorFamily::input_combiner;
        return true;
    }
    if (contains(name, "enorm")) {
        result =
            MtpTensorFamily::embedding_norm;
        return true;
    }
    if (contains(name, "hnorm")) {
        result =
            MtpTensorFamily::hidden_norm;
        return true;
    }
    if (contains(name, "hyper_connection") ||
        contains(name, "hc_") ||
        contains(name, "nextn.hc_")) {
        result =
            MtpTensorFamily::hyper_connection_mixer;
        return true;
    }
    if (contains(name, "indexer") ||
        contains(name, "index_qk")) {
        result =
            MtpTensorFamily::indexer;
        return true;
    }
    if (contains(name, "attn") ||
        contains(name, "q_proj") ||
        contains(name, "k_proj") ||
        contains(name, "v_proj") ||
        contains(name, "o_proj")) {
        result =
            MtpTensorFamily::attention;
        return true;
    }
    if (contains(name, "shared_head") ||
        contains(name, "lm_head")) {
        result =
            MtpTensorFamily::output_head;
        return true;
    }
    if (contains(name, "expert") ||
        contains(name, "router") ||
        contains(name, "ffn") ||
        contains(name, "mlp") ||
        contains(name, "gate_proj") ||
        contains(name, "up_proj") ||
        contains(name, "down_proj")) {
        result =
            MtpTensorFamily::moe;
        return true;
    }
    return false;
}

}  // namespace

MtpDraftTensorAudit
audit_mtp_draft_tensor_families(
    const ModelManifest& manifest) {
    if (manifest.layout_identity == 0 ||
        manifest.layout_identity !=
            model_manifest_layout_identity(
                manifest)) {
        throw std::invalid_argument(
            "MTP tensor audit requires valid manifest identity");
    }

    MtpDraftTensorAudit audit;
    audit.manifest_layout_identity =
        manifest.layout_identity;
    audit.families = {
        {MtpTensorFamily::input_combiner, true, {}},
        {MtpTensorFamily::embedding_norm, true, {}},
        {MtpTensorFamily::hidden_norm, true, {}},
        {MtpTensorFamily::hyper_connection_mixer, true, {}},
        {MtpTensorFamily::attention, true, {}},
        {MtpTensorFamily::indexer, true, {}},
        {MtpTensorFamily::moe, true, {}},
        {MtpTensorFamily::output_head, false, {}},
    };

    for (std::uint32_t index = 0;
         index < manifest.tensors.size();
         ++index) {
        const auto& tensor =
            manifest.tensors[index];
        if (tensor.role !=
            TensorRole::mtp) {
            continue;
        }

        MtpTensorFamily kind;
        if (!classify(
                tensor.name,
                kind)) {
            audit.unclassified_mtp_tensors.push_back(
                index);
            continue;
        }

        auto* target =
            family(
                audit,
                kind);
        if (target == nullptr) {
            throw std::logic_error(
                "MTP tensor audit family table is incomplete");
        }
        target->tensor_indices.push_back(
            index);
    }

    for (const auto& item :
         audit.families) {
        if (item.required &&
            item.tensor_indices.empty()) {
            audit.missing_required_families.push_back(
                item.family);
        }
    }

    // The checkpoint may carry the combiner either as the original
    // fc_embedding + fc_hidden pair or as one losslessly fused eh_proj tensor.
    const auto* combiner =
        family(
            audit,
            MtpTensorFamily::input_combiner);
    if (combiner != nullptr &&
        !combiner->tensor_indices.empty()) {
        bool fused = false;
        bool embedding = false;
        bool hidden = false;
        for (const auto index :
             combiner->tensor_indices) {
            const auto& name =
                manifest.tensors[index].name;
            fused =
                fused ||
                contains(name, "eh_proj");
            embedding =
                embedding ||
                contains(name, "fc_embedding");
            hidden =
                hidden ||
                contains(name, "fc_hidden");
        }
        if (!fused &&
            !(embedding && hidden)) {
            if (std::find(
                    audit.missing_required_families.begin(),
                    audit.missing_required_families.end(),
                    MtpTensorFamily::input_combiner) ==
                audit.missing_required_families.end()) {
                audit.missing_required_families.push_back(
                    MtpTensorFamily::input_combiner);
            }
        }
    }

    return audit;
}

MtpDraftTensorAudit
audit_mtp_draft_tensor_families(
    const ModelManifest& manifest,
    const MtpDraftRuntimeBinding& binding) {
    if (binding.manifest_layout_identity !=
            manifest.layout_identity ||
        binding.compatibility_identity == 0 ||
        binding.tensor_indices.empty()) {
        throw std::invalid_argument(
            "MTP tensor audit requires matching initialized binding");
    }

    const auto audit =
        audit_mtp_draft_tensor_families(
            manifest);

    std::vector<std::uint32_t> expected;
    for (std::uint32_t index = 0;
         index < manifest.tensors.size();
         ++index) {
        if (manifest.tensors[index].role ==
            TensorRole::mtp) {
            expected.push_back(index);
        }
    }
    if (binding.tensor_indices != expected) {
        throw std::invalid_argument(
            "MTP tensor audit binding does not cover manifest MTP tensors exactly");
    }

    auto bound = audit;
    bound.proposal_identity =
        binding.compatibility_identity;
    return bound;
}

const char*
to_string(MtpTensorFamily family) noexcept {
    switch (family) {
    case MtpTensorFamily::input_combiner:
        return "input_combiner";
    case MtpTensorFamily::embedding_norm:
        return "embedding_norm";
    case MtpTensorFamily::hidden_norm:
        return "hidden_norm";
    case MtpTensorFamily::hyper_connection_mixer:
        return "hyper_connection_mixer";
    case MtpTensorFamily::attention:
        return "attention";
    case MtpTensorFamily::indexer:
        return "indexer";
    case MtpTensorFamily::moe:
        return "moe";
    case MtpTensorFamily::output_head:
        return "output_head";
    }
    return "unknown";
}

}  // namespace strix
