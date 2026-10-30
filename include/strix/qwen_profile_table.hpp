#pragma once
// Included by qwen_profile.hpp inside strix::qwen::profile. This is the single
// product table. Columns are default, fast, quality; empty means leave alone.
// No proposed fast implementation is present in this build. Promotion requires
// compiler, native execution, model fidelity and complete-workload gates.
inline constexpr Lever table[] = {
    {.name = "qsa_budget", .env = "STRIX_QPF2K_QSA_BUDGET", .option = "--qsa-budget",
     .value = {"", "1536", ""}, .builtin = "default",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, false, false},
     .target = "1536", .off = "0", .source = "pending attention-budget holdout; implementation absent"},
    {.name = "moe_route_bf16", .env = "STRIX_QPF2K_MOE_ROUTE_BF16", .option = "--moe-route-bf16",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, false, false},
     .target = "bf16 expert routing", .off = "0", .source = "pending expert-routing implementation and qualification"},
    {.name = "gdn_fp16_chunk", .env = "", .option = "--gdn-fp16-chunk",
     .value = {"", "qwen-gdn-fp16-chunk", ""}, .builtin = "default",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, false, false},
     .target = "qwen-gdn-fp16-chunk", .off = "off", .source = "pending FP16 recurrent-chunk implementation and qualification"},
    {.name = "output_head", .env = "", .option = "--output-head-profile",
     .value = {"", "fp16-output-head", ""}, .builtin = "default",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, false, false},
     .target = "fp16 output-head projection", .off = "off", .source = "pending output-head implementation and qualification"},
};
