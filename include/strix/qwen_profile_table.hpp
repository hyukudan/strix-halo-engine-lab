#pragma once
// Included by qwen_profile.hpp inside strix::qwen::profile. This is the single
// product table. Columns are default, fast, quality; empty means leave alone.
// Ready means qualified; runnable means implemented. Promotion requires
// compiler, native execution, model fidelity and complete-workload gates.
// Provisional selects runnable code ahead of qualification without marking it ready.
inline constexpr Lever table[] = {
    {.name = "qsa_budget", .env = "STRIX_QPF2K_QSA_BUDGET", .option = "--qsa-budget",
     .value = {"", "1536", ""}, .builtin = "default",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, false, false},
     .target = "1536", .off = "0", .source = "pending attention-budget qualification; implementation absent"},
    {.name = "moe_route_bf16", .env = "STRIX_QPF2K_MOE_ROUTE_BF16", .option = "--prefill-moe-route-bf16",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, true, false},
     .target = "BF16 MoE route storage", .off = "0",
     .source = "selected for fast ahead of its qualification receipt; qualification in progress; changes logits",
     .native_option = true, .provisional = {false, true, false}},
    {.name = "gdn_fp16_chunk", .env = "STRIX_QPF2K_GDN_FP16_CHUNK", .option = "--gdn-fp16-chunk",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, true, false},
     .target = "qwen-gdn-fp16-chunk rev 2", .off = "0",
     .source = "chunked GDN prefill on FP16 WMMA (FP64 fixture screen relL2 ~4e-4); pending fidelity qualification; changes logits"},
    {.name = "output_head", .env = "", .option = "--output-head-profile",
     .value = {"", "fp16-output-head", ""}, .builtin = "default",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, false, false},
     .target = "fp16 output-head projection", .off = "off", .source = "pending output-head implementation and qualification"},
};
