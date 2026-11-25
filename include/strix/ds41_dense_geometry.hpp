#pragma once

// Dense site geometry of the DeepSeek V4.1 engine, taken from the generated constants of models/deepseek-v4.1-flash.toml.
// Every shape-keyed decision of the HIP drivers calls strix::core::dense_sites (include/strix/core/dense_site_policy.hpp)
// with this geometry; no engine source carries a dimension literal for them.

#include "strix/core/dense_site_policy.hpp"
#include "strix/model/bind.hpp"
#include "strix/model_data/deepseek_v4_1_flash.hpp"

#include <cstdint>

namespace strix::ds41::dense_geometry {

namespace m = strix::model_data::deepseek_v4_1_flash;
using strix::model::narrow;

// Ranks of the tensor-parallel dense route: the pair contract of the engine (heads, o_b K, FFN and vocabulary are halved).
inline constexpr std::uint32_t kTpWorld = 2U;

inline constexpr core::dense_sites::Geometry kGeometry{
    narrow<std::uint32_t>(m::dims::hidden),
    narrow<std::uint32_t>(m::attention::q_lora_rank),
    narrow<std::uint32_t>(m::dims::n_kv_heads * m::dims::head_dim),
    narrow<std::uint32_t>(m::dims::n_heads),
    narrow<std::uint32_t>(m::dims::head_dim),
    narrow<std::uint32_t>(m::attention::o_groups),
    narrow<std::uint32_t>(m::attention::o_lora_rank),
    narrow<std::uint32_t>(m::dims::intermediate),
    narrow<std::uint32_t>(m::dims::vocab),
    kTpWorld,
    narrow<std::uint32_t>((m::extras::engram_max_ngram_size - 1) * m::extras::engram_n_heads * m::extras::engram_head_dim),
    narrow<std::uint32_t>(m::dims::hidden * (m::extras::hc_mult + 1))};
static_assert(core::dense_sites::valid(kGeometry));

inline constexpr core::dense_sites::DensePolicy kPolicy{};

}  // namespace strix::ds41::dense_geometry
