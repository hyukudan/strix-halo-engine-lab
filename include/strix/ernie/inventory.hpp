#pragma once

#include "strix/model/bind.hpp"
#include "strix/model_data/ernie_5_5_flash_preview.hpp"
#include "strix/weight/admission.hpp"

#include <filesystem>
#include <iterator>
#include <map>
#include <optional>
#include <string_view>
#include <vector>

namespace strix::ernie {

namespace data = model_data::ernie_5_5_flash_preview;
inline constexpr auto kLayerCount = model::narrow<unsigned>(data::dims::layers);
inline constexpr auto kMainShardCount = model::narrow<unsigned>(data::weights::checkpoint::shards);
inline constexpr auto kDraftLayerCount = model::narrow<unsigned>(data::drafter::layers);
inline constexpr auto kExpertCount = model::narrow<unsigned>(data::moe::n_routed_experts);
inline constexpr auto& kMlaLayers = data::attention::mla::layers;
inline constexpr auto kFlashIndexerShard = data::attention::indexer::sidecar;
static_assert(std::size(data::attention::compress_ratios) == kLayerCount);

enum class Module {
    unclassified, embedding, layer_norm, hyper_connection, hca, compressor,
    mla, attention_gate, value_head_mix, sparse_indexer, flash_indexer,
    dense_mlp, router, latent_projection, shared_expert, routed_expert,
    final_norm, hc_contract, lm_head, draft_input, draft_markov, draft_confidence
};

[[nodiscard]] std::string_view module_name(Module module) noexcept;
[[nodiscard]] std::string main_shard_name(unsigned one_based_index);
[[nodiscard]] std::string target_layer_prefix(unsigned layer);
[[nodiscard]] std::string inverse_scale_name(std::string_view weight_name);

[[nodiscard]] constexpr bool layer_is_mla(unsigned layer) {
    return layer < kLayerCount && data::attention::compress_ratios[layer] < 0;
}
[[nodiscard]] constexpr bool layer_has_moe(unsigned layer) {
    return layer < kLayerCount && layer >= data::moe::first_k_dense_replace &&
           layer % data::moe::moe_layer_freq == 0;
}

struct Tensor {
    Module module;
    bool drafter;
    std::optional<unsigned> layer;
    std::string wire_dtype;
    std::vector<std::uint64_t> shape;    // Physical safetensors shape, row-major.
    weight::Slice payload;             // Position and owner for a future loader.
};

struct Inventory {
    std::map<std::string, Tensor, std::less<>> tensors;
    std::vector<std::string> missing_shards;
    std::size_t missing_known_tensors{};
    bool has_manifest{};
};

// All existing shards are read through the shared header reader. Missing files and
// known tensors are counted for partial downloads. Unknown names remain visible
// for schema discovery. The result never admits an inference session.
[[nodiscard]] weight::Status inspect_weights(const std::filesystem::path& root, Inventory& out);

// Accumulate one already indexed shard. On failure, tensors is unchanged.
[[nodiscard]] weight::Status add_shard(const weight::TensorIndex& shard, Inventory& inventory);
[[nodiscard]] std::size_t count_missing_known_tensors(const Inventory& inventory);

} // namespace strix::ernie
