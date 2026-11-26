#pragma once

#include "strix/engine_profile.hpp"

namespace strix::weight {

inline constexpr profile::Lever kLowBitFoldedScaleLever{
    .name = "lowbit.folded-scale", .env = "STRIX_LOWBIT_FOLDED_SCALE", .option = "",
    .value = {"", "f16_folded_scale", "", "f16_folded_scale"},
    .builtin = "f16_group_scale", .kind = profile::Kind::bit_changing,
    .ready = {false, true, false, true}, .runnable = {false, true, false, true},
    .target = "f16_folded_scale", .off = "f16_group_scale",
    .source = "awaits gate v2 and regime timing receipts for HIP low-bit rows",
    .omit_builtin_fingerprint = true,
    .identity = {{{"f16_group_scale", "hip.lowbit.f16-group-scale-chain@1"},
                   {"f16_folded_scale", "hip.lowbit.f16-folded-scale-tile@2"}}},
};

}  // namespace strix::weight
