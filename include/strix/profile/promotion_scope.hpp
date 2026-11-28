#pragma once
// The run scope of a promotion key, read once per process by every engine adapter (profile::ladder_promotion_execution).
#include "strix/engine_profile.hpp"
#include "strix/model_content_hash.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>

namespace strix::profile {

// Owned storage of a LadderScope and the device product it was read with. Empty (known() false) when the run does not
// spell the promotion key: no promotion row decides, and no pack identity was read.
struct PromotionScope {
    std::string product, pack_sha256, route;
    std::uint32_t ranks = 0;
    [[nodiscard]] bool known() const noexcept { return !pack_sha256.empty(); }
    [[nodiscard]] LadderScope scope() const noexcept { return {pack_sha256, route, ranks}; }
};

// The scope of this run when `wanted` (promotion_key_wanted()): the device product (device/{hip,cuda}_cost_identity.hpp),
// the content identity of the pack at `pack` (pack_content_identity(); a metadata read once the digest cache is warm),
// the route ("cli", "serve", ...) and the rank count. Throws what pack_content_identity() throws.
[[nodiscard]] inline PromotionScope promotion_scope(bool wanted, std::string product, const std::filesystem::path& pack,
                                                    std::string route, std::uint32_t ranks) {
    if (!wanted) return {};
    return {std::move(product), model_content_identity_hex(pack_content_identity(pack)), std::move(route), ranks};
}

}  // namespace strix::profile
