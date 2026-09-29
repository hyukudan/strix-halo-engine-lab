#include "strix/model_manifest.hpp"

#include <charconv>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {

bool contains(std::string_view value, std::string_view needle) noexcept {
    return value.find(needle) != std::string_view::npos;
}

bool starts_with(std::string_view value, std::string_view prefix) noexcept {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

std::optional<std::uint32_t> parse_layer_after(std::string_view name,
                                               std::string_view marker) noexcept {
    const auto marker_pos = name.find(marker);
    if (marker_pos == std::string_view::npos) {
        return std::nullopt;
    }
    const auto begin_pos = marker_pos + marker.size();
    if (begin_pos >= name.size()) {
        return std::nullopt;
    }
    const auto end_pos = name.find('.', begin_pos);
    const auto text = name.substr(begin_pos, end_pos == std::string_view::npos
                                                ? name.size() - begin_pos
                                                : end_pos - begin_pos);
    if (text.empty()) {
        return std::nullopt;
    }
    std::uint32_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return std::nullopt;
    }
    return value;
}

std::optional<std::uint32_t> layer_of(std::string_view name) noexcept {
    if (const auto layer = parse_layer_after(name, "blk."); layer.has_value()) {
        return layer;
    }
    return parse_layer_after(name, "layers.");
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error("model manifest byte count overflow");
    }
    return left + right;
}

}  // namespace

TensorRole classify_tensor_role(std::string_view name) noexcept {
    if (contains(name, "mtp")) {
        return TensorRole::mtp;
    }
    if (contains(name, "ngram") || contains(name, "ple.")) {
        return TensorRole::ngram_table;
    }
    if (contains(name, "indexer") || contains(name, "sparse_index") ||
        contains(name, "attn_index")) {
        return TensorRole::sparse_index;
    }
    if (contains(name, "deltanet") || contains(name, "gdn") ||
        contains(name, "recurrent") || contains(name, ".ssm_")) {
        return TensorRole::recurrent_state;
    }
    if (contains(name, "experts.") || contains(name, ".expert.") ||
        contains(name, "expert_")) {
        return TensorRole::moe_expert;
    }
    if (contains(name, "router") || contains(name, "gate_inp")) {
        return TensorRole::moe_router;
    }
    if (starts_with(name, "token_embd.") || starts_with(name, "tok_embeddings.") ||
        contains(name, "embed_tokens")) {
        return TensorRole::embedding;
    }
    if (starts_with(name, "output.") || starts_with(name, "lm_head.")) {
        return TensorRole::output_head;
    }
    if (contains(name, "norm")) {
        return TensorRole::normalization;
    }
    if (contains(name, "attn_q") || contains(name, "attn_k") ||
        contains(name, "attn_v") || contains(name, "attn_output") ||
        contains(name, "q_proj") || contains(name, "k_proj") ||
        contains(name, "v_proj") || contains(name, "o_proj")) {
        return TensorRole::attention_projection;
    }
    if (contains(name, "ffn_") || contains(name, "mlp.") ||
        contains(name, "gate_proj") || contains(name, "up_proj") ||
        contains(name, "down_proj")) {
        return TensorRole::dense_projection;
    }
    return TensorRole::unknown;
}

ResidencyPolicy default_residency(TensorRole role) noexcept {
    switch (role) {
    case TensorRole::ngram_table:
        return ResidencyPolicy::file_backed_candidate;
    default:
        return ResidencyPolicy::resident;
    }
}

std::string_view to_string(TensorRole role) noexcept {
    switch (role) {
    case TensorRole::unknown: return "unknown";
    case TensorRole::embedding: return "embedding";
    case TensorRole::output_head: return "output_head";
    case TensorRole::normalization: return "normalization";
    case TensorRole::attention_projection: return "attention_projection";
    case TensorRole::dense_projection: return "dense_projection";
    case TensorRole::moe_router: return "moe_router";
    case TensorRole::moe_expert: return "moe_expert";
    case TensorRole::recurrent_state: return "recurrent_state";
    case TensorRole::sparse_index: return "sparse_index";
    case TensorRole::ngram_table: return "ngram_table";
    case TensorRole::mtp: return "mtp";
    }
    return "unknown";
}

std::string_view to_string(ResidencyPolicy policy) noexcept {
    switch (policy) {
    case ResidencyPolicy::resident: return "resident";
    case ResidencyPolicy::file_backed_candidate: return "file_backed_candidate";
    case ResidencyPolicy::bounded_cache_candidate: return "bounded_cache_candidate";
    }
    return "resident";
}

std::string_view to_string(PlacementKind kind) noexcept {
    switch (kind) {
    case PlacementKind::local: return "local";
    case PlacementKind::remote: return "remote";
    }
    return "local";
}

ModelManifest build_model_manifest(const gguf::FileInfo& info) {
    ModelManifest manifest;
    if (const auto* architecture = info.find_metadata("general.architecture");
        architecture != nullptr && architecture->string_value.has_value()) {
        manifest.architecture = *architecture->string_value;
    }

    manifest.tensors.reserve(info.tensors.size());
    for (const auto& tensor : info.tensors) {
        ManifestTensor entry;
        entry.name = tensor.name;
        entry.dimensions = tensor.dimensions;
        entry.ggml_type = tensor.ggml_type;
        entry.byte_size = tensor.byte_size;
        entry.absolute_offset = tensor.absolute_offset;
        entry.role = classify_tensor_role(tensor.name);
        entry.residency = default_residency(entry.role);
        entry.layer = layer_of(tensor.name);

        manifest.total_tensor_bytes = checked_add(manifest.total_tensor_bytes, entry.byte_size);
        if (entry.residency == ResidencyPolicy::file_backed_candidate) {
            manifest.file_backed_candidate_bytes =
                checked_add(manifest.file_backed_candidate_bytes, entry.byte_size);
        } else if (entry.residency == ResidencyPolicy::bounded_cache_candidate) {
            manifest.bounded_cache_candidate_bytes =
                checked_add(manifest.bounded_cache_candidate_bytes, entry.byte_size);
        }
        if (entry.role == TensorRole::unknown) {
            ++manifest.unknown_tensor_count;
        }
        manifest.tensors.push_back(std::move(entry));
    }
    return manifest;
}

}  // namespace strix
