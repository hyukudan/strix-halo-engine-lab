#pragma once

#include "strix/engine_profile.hpp"

namespace strix::weight {

inline constexpr profile::Lever kLowBitRowsSelectLever{
    .name = "lowbit.rows-select", .env = "STRIX_LOWBIT_ROWS_SELECT", .option = "",
    .value = {"", "f16_rows_select", "", "f16_rows_select"},
    .builtin = "f16_group_scale", .kind = profile::Kind::bit_changing,
    .ready = {false, true, false, true}, .runnable = {false, true, false, true},
    .target = "f16_rows_select", .off = "f16_group_scale",
    .source = "awaits gate v2 and regime timing receipts for HIP low-bit rows",
    .omit_builtin_fingerprint = true,
    .identity = {{{"f16_group_scale", "hip.lowbit.f16-group-scale-chain@1"},
                   {"f16_rows_select", "lowbit.rows-select.fp32-sweep-cells@2"}}},
};

}  // namespace strix::weight
