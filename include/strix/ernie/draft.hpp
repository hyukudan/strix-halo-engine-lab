#pragma once

#include "strix/model/bind.hpp"
#include "strix/model_data/ernie_5_5_flash_preview.hpp"
#include "strix/spec/block_parallel.hpp"
#include "strix/spec/draft_decl.hpp"

#include <array>

namespace strix::ernie {

namespace draft_data = model_data::ernie_5_5_flash_preview;

inline constexpr std::array<std::int32_t, 3> dspark_taps{
    model::narrow<std::int32_t>(draft_data::drafter::taps[0]), model::narrow<std::int32_t>(draft_data::drafter::taps[1]),
    model::narrow<std::int32_t>(draft_data::drafter::taps[2])};
inline constexpr std::array<std::string_view, 2> dspark_target_weights{
    "model.embed_tokens.weight", "lm_head.weight"};

inline constexpr spec::DraftDecl dspark_declaration{
    .id = "ernie55-dspark",
    .kind = spec::DraftKind::block,
    .tensors = "model.mtp.",
    .feed = "post_mhc_stream_mean",
    .max_k = draft_data::drafter::block,
    .block = draft_data::drafter::block,
    .layers = draft_data::drafter::layers,
    .taps = dspark_taps,
    .head = {.rows = spec::HeadRows::full, .size = draft_data::dims::vocab},
    .shares = dspark_target_weights,
    .noise_token = draft_data::extras::dspark_noise_token_id,
    .attention = spec::DraftAttention::block_noncausal,
    .confidence = "model.mtp.confidence_head.projection.weight",
    .chain = "markov",
    .anchor_proposes = true,
};

inline constexpr spec::BlockPolicy dspark_block_policy{
    .block = dspark_declaration.block,
    .fill_token = dspark_declaration.noise_token,
    .kv = spec::KvSource::own_ring,
    .ring = draft_data::attention::sliding_window,
    .mask = spec::BlockMask::bidirectional,
    .anchor_proposes = dspark_declaration.anchor_proposes,
    .draft_layers = dspark_declaration.layers,
    .target_layer_count = static_cast<std::uint32_t>(dspark_taps.size()),
    .target_layers = {static_cast<std::uint16_t>(dspark_taps[0]), static_cast<std::uint16_t>(dspark_taps[1]),
                      static_cast<std::uint16_t>(dspark_taps[2])},
};

inline constexpr unsigned echo_mtp_layers = draft_data::extras::num_nextn_predict_layers;

} // namespace strix::ernie
