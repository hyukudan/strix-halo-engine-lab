#pragma once
// Model dimensions shared by device arithmetic and host-only state layout.
// The values come from models/glm-5.3-flash.toml (generated header, strix/model/bind.hpp); the static_asserts below
// keep the former literals.
#include <cstdint>

#include "strix/model/bind.hpp"
#include "strix/model_data/glm_5_3_flash.hpp"
namespace strix::glm {
namespace dims_bind {
namespace toml = strix::model_data::glm_5_3_flash;
constexpr unsigned u(std::int64_t v) { return strix::model::narrow<unsigned>(v); }
}  // namespace dims_bind
inline constexpr unsigned kHidden = dims_bind::u(dims_bind::toml::dims::hidden);
inline constexpr unsigned kHc = dims_bind::u(dims_bind::toml::extras::hyper_connections::rank), kHcMix = (2 + kHc) * kHc;
inline constexpr unsigned kHcIters = dims_bind::u(dims_bind::toml::extras::hyper_connections::iterations);
inline constexpr unsigned kLayers = dims_bind::u(dims_bind::toml::dims::layers), kMtpLayer = dims_bind::u(dims_bind::toml::extras::mtp::layer);
inline constexpr unsigned kFirstMoe = dims_bind::u(dims_bind::toml::moe::first_layer), kVocab = dims_bind::u(dims_bind::toml::dims::vocab);
inline constexpr unsigned kKdaHeads = dims_bind::u(dims_bind::toml::attention::kda::heads), kKdaDim = dims_bind::u(dims_bind::toml::attention::kda::head_dim);
inline constexpr unsigned kKdaQkv = kKdaHeads * kKdaDim, kConvK = dims_bind::u(dims_bind::toml::attention::kda::conv_kernel);
inline constexpr unsigned kQLora = dims_bind::u(dims_bind::toml::attention::mla::q_lora), kKvLora = dims_bind::u(dims_bind::toml::attention::mla::kv_lora);
inline constexpr unsigned kMlaHeads = dims_bind::u(dims_bind::toml::attention::mla::heads), kMlaDim = dims_bind::u(dims_bind::toml::attention::mla::head_dim);
inline constexpr unsigned kIdxHeads = dims_bind::u(dims_bind::toml::attention::select::heads), kIdxDim = dims_bind::u(dims_bind::toml::attention::select::head_dim);
inline constexpr unsigned kKpool = dims_bind::u(dims_bind::toml::attention::select::pool), kIdxTopk = dims_bind::u(dims_bind::toml::attention::select::top_k);
inline constexpr unsigned kIdxPools = kIdxTopk / kKpool;
inline constexpr unsigned kExperts = dims_bind::u(dims_bind::toml::moe::experts), kTopE = dims_bind::u(dims_bind::toml::moe::top_k);
inline constexpr unsigned kExpertI = dims_bind::u(dims_bind::toml::moe::intermediate), kDenseI = dims_bind::u(dims_bind::toml::dims::intermediate);
static_assert(kHidden == 4096 && kHc == 4 && kHcMix == 24 && kHcIters == 20);
static_assert(kLayers == 45 && kMtpLayer == 45 && kFirstMoe == 3 && kVocab == 154880);
static_assert(kKdaHeads == 64 && kKdaDim == 128 && kKdaQkv == 8192 && kConvK == 4);
static_assert(kQLora == 1536 && kKvLora == 512 && kMlaHeads == 64 && kMlaDim == 256);
static_assert(kIdxHeads == 32 && kIdxDim == 128 && kKpool == 4 && kIdxTopk == 2048 && kIdxPools == 512);
static_assert(kExperts == 288 && kTopE == 8 && kExpertI == 2048 && kDenseI == 12288);
static_assert(dims_bind::toml::attention::select::pools == kIdxPools && dims_bind::toml::attention::kda::qkv == kKdaQkv,
              "the generated derived values agree with the products here");
}  // namespace strix::glm
