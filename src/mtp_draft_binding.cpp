#include "strix/mtp_draft_binding.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

void hash_content(
    std::uint64_t& hash,
    const ModelContentIdentity& identity) noexcept {
    for (const auto byte : identity.bytes) {
        hash ^= static_cast<std::uint64_t>(byte);
        hash *= fnv_prime;
    }
}

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(
            "MTP draft tensor byte count overflow");
    }
    return left + right;
}

std::uint64_t compatibility_identity(
    const MtpDraftRuntimeBinding& draft) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, draft.manifest_layout_identity);
    hash_content(hash, draft.model_content_identity);
    hash_u64(hash, draft.tensor_indices.size());
    hash_u64(hash, draft.total_tensor_bytes);
    for (const auto index : draft.tensor_indices) {
        hash_u64(hash, index);
    }
    return hash;
}

std::uint64_t physical_identity(
    const MtpDraftRuntimeBinding& draft,
    const ModelRuntimeBinding& runtime) {
    std::uint64_t hash =
        compatibility_identity(draft);
    for (const auto index : draft.tensor_indices) {
        const auto* binding =
            find_runtime_tensor_binding(
                runtime,
                index);
        if (binding == nullptr) {
            throw std::invalid_argument(
                "MTP draft tensor lacks runtime binding");
        }
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                binding->storage));
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                binding->address));
        hash_u64(hash, binding->bytes);
        hash_u64(
            hash,
            binding->allocation_generation);
        hash_u64(hash, binding->owner_node);
    }
    return hash;
}

}  // namespace

MtpDraftRuntimeBinding
build_mtp_draft_runtime_binding(
    const ModelManifest& manifest,
    const ModelRuntimeBinding& runtime) {
    validate_model_runtime_binding(
        manifest,
        runtime);

    MtpDraftRuntimeBinding draft;
    draft.manifest_layout_identity =
        manifest.layout_identity;
    draft.model_content_identity =
        runtime.model_content_identity;

    for (std::uint32_t index = 0;
         index < manifest.tensors.size();
         ++index) {
        const auto& tensor =
            manifest.tensors[index];
        if (tensor.role !=
            TensorRole::mtp) {
            continue;
        }

        const auto* binding =
            find_runtime_tensor_binding(
                runtime,
                index);
        if (binding == nullptr ||
            binding->bytes <
                tensor.byte_size) {
            throw std::invalid_argument(
                "MTP draft tensor runtime binding is missing or undersized");
        }

        draft.tensor_indices.push_back(index);
        draft.total_tensor_bytes =
            checked_add(
                draft.total_tensor_bytes,
                tensor.byte_size);
    }

    if (draft.tensor_indices.empty()) {
        throw std::invalid_argument(
            "model manifest contains no MTP tensors");
    }

    draft.compatibility_identity =
        compatibility_identity(draft);
    draft.physical_binding_identity =
        physical_identity(
            draft,
            runtime);
    if (draft.compatibility_identity == 0 ||
        draft.physical_binding_identity == 0) {
        throw std::logic_error(
            "MTP draft identities must be non-zero");
    }

    return draft;
}

void validate_mtp_draft_runtime_binding(
    const ModelManifest& manifest,
    const ModelRuntimeBinding& runtime,
    const MtpDraftRuntimeBinding& draft) {
    const auto expected =
        build_mtp_draft_runtime_binding(
            manifest,
            runtime);

    if (draft.manifest_layout_identity !=
            expected.manifest_layout_identity ||
        draft.model_content_identity !=
            expected.model_content_identity ||
        draft.tensor_indices !=
            expected.tensor_indices ||
        draft.total_tensor_bytes !=
            expected.total_tensor_bytes ||
        draft.compatibility_identity !=
            expected.compatibility_identity ||
        draft.physical_binding_identity !=
            expected.physical_binding_identity) {
        throw std::invalid_argument(
            "MTP draft runtime binding is stale or incompatible");
    }
}

DraftProposalStateContext
make_mtp_draft_proposal_state(
    const MtpDraftRuntimeBinding& draft,
    const SerialSessionContext& target_session) {
    if (draft.compatibility_identity == 0) {
        throw std::invalid_argument(
            "MTP draft binding requires non-zero compatibility identity");
    }
    return make_draft_proposal_state(
        draft.compatibility_identity,
        target_session);
}

void validate_mtp_draft_proposal_state(
    const MtpDraftRuntimeBinding& draft,
    const DraftProposalStateContext& state,
    const SerialSessionContext& target_session) {
    if (draft.compatibility_identity == 0 ||
        state.proposal_identity !=
            draft.compatibility_identity) {
        throw std::invalid_argument(
            "MTP draft proposal state identity mismatch");
    }
    validate_draft_proposal_state(
        state,
        target_session);
}

}  // namespace strix
