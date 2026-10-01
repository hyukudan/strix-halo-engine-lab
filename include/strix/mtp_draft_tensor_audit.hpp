#pragma once

#include "strix/model_manifest.hpp"
#include "strix/mtp_draft_binding.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strix {

enum class MtpTensorFamily {
    input_combiner,
    embedding_norm,
    hidden_norm,
    hyper_connection_mixer,
    attention,
    moe,
};

struct MtpTensorFamilyAudit {
    MtpTensorFamily family =
        MtpTensorFamily::embedding_combiner;
    bool required = true;
    std::vector<std::uint32_t> tensor_indices;
};

struct MtpDraftTensorAudit {
    std::uint64_t manifest_layout_identity = 0;
    std::uint64_t proposal_identity = 0;
    std::vector<MtpTensorFamilyAudit> families;
    std::vector<std::uint32_t> unclassified_mtp_tensors;
    std::vector<MtpTensorFamily> missing_required_families;

    [[nodiscard]] bool structurally_complete() const noexcept {
        return missing_required_families.empty();
    }
};

[[nodiscard]] MtpDraftTensorAudit
audit_mtp_draft_tensor_families(
    const ModelManifest& manifest);

[[nodiscard]] MtpDraftTensorAudit
audit_mtp_draft_tensor_families(
    const ModelManifest& manifest,
    const MtpDraftRuntimeBinding& binding);

[[nodiscard]] const char*
to_string(MtpTensorFamily family) noexcept;

}  // namespace strix
