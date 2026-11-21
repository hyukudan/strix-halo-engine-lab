#pragma once
// Included by profiles/qwen.hpp inside strix::qwen::profile. This is the single
// product table. Columns are default, fast, quality; empty means leave alone.
// Ready means qualified; runnable means implemented. Promotion requires
// compiler, native execution, model fidelity and complete-workload gates.
// Provisional selects runnable code ahead of qualification without marking it ready.
inline constexpr Lever table[] = {
    {.name = "qsa_budget", .env = "STRIX_QPF2K_QSA_BUDGET", .option = "--qsa-budget",
     .value = {"", "1536", ""}, .builtin = "default",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, true, false},
     .target = "1536", .off = "0", .source = "prefill ranked-block budget; qualification in progress; changes logits",
     .provisional = {false, true, false}},
    {.name = "moe_route_bf16", .env = "STRIX_QPF2K_MOE_ROUTE_BF16", .option = "--prefill-moe-route-bf16",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, true, false},
     .target = "BF16 MoE route storage", .off = "0",
     .source = "selected for fast ahead of its qualification receipt; qualification in progress; changes logits",
     .native_option = true, .provisional = {false, true, false}},
    {.name = "moe_route_prune", .env = "STRIX_QPF2K_MOE_ROUTE_PRUNE", .option = "--prefill-moe-route-prune",
     .value = {"", "", "", "rel0.02"}, .builtin = "off",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {false, true, false, true},
     .target = "rel<f>[n] | topp<f>[n] | cap<k>[n]", .off = "off",
     .source = "RoutePrune: the lowest-weight route assignments leave the prefill MoE plan; optional toggle; the turbo cell (rel0.02) is a PENDING default whose value waits for the calibration sweep, applied only with STRIX_QWEN_PROFILE_ALLOW_PENDING=1; changes logits"},
    {.name = "gdn_fp16_chunk", .env = "STRIX_QPF2K_GDN_FP16_CHUNK", .option = "--gdn-fp16-chunk",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, true, false},
     .target = "qwen-gdn-fp16-chunk rev 2 (=1), rev 3 (=3)", .off = "0",
     .source = "chunked GDN prefill on FP16 WMMA (FP64 fixture screen relL2 ~4e-4); selected for fast ahead of its qualification receipt; qualification in progress; changes logits",
     .provisional = {false, true, false}},
    // TURBO-only prefill registry rows (include/strix/prefill_identity.hpp): FAST arithmetic that is a candidate there, selected
    // by turbo ahead of its qualification receipt (provisional; stamped "qualification in progress"). default/fast/quality: unset.
    {.name = "index_proj_wmma", .env = "STRIX_PREFILL_INDEX_PROJ_WMMA", .option = "",
     .value = {"", "", "", "1"}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {false, false, false, true},
     .target = "idxproj16", .off = "0", .source = "turbo ahead of qualification; changes logits",
     .provisional = {false, false, false, true}},
    {.name = "index_score_wmma", .env = "STRIX_PREFILL_INDEX_SCORE_WMMA", .option = "",
     .value = {"", "", "", "1"}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {false, false, false, true},
     .target = "idxscore16", .off = "0", .source = "turbo ahead of qualification; changes logits",
     .provisional = {false, false, false, true}},
    {.name = "ple_rows_f16", .env = "STRIX_PREFILL_PLE_ROWS_F16", .option = "",
     .value = {"", "", "", "1"}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {false, false, false, true},
     .target = "plef16", .off = "0", .source = "turbo ahead of qualification; changes logits",
     .provisional = {false, false, false, true}},
    {.name = "gdn_gates_f16", .env = "STRIX_PREFILL_GDN_GATES_F16", .option = "",
     .value = {"", "", "", "1"}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {false, false, false, true},
     .target = "gatef16 (needs gdn_fp16_chunk)", .off = "0", .source = "turbo ahead of qualification; changes logits",
     .provisional = {false, false, false, true}},
    {.name = "output_head", .env = "", .option = "--output-head-profile",
     .value = {"", "fp16-output-head", ""}, .builtin = "default",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, false, false},
     .target = "fp16 output-head projection", .off = "off", .source = "pending output-head implementation and qualification"},
};
// Regimes where the HIP fast cells above are qualified (strix::profile::RegimeGate ledger; the HIP engine passes these rows,
// the CUDA rows in profiles/qwen_cuda.hpp carry none). Gate: KL <= 0.02, top-1 >= 0.93, dNLL <= 0.01 pooled AND per
// window vs the exact path, teacher-forced 256 rows. Above 8,192 prompt tokens the cells stay provisional: the 32K gate
// failed per window (chat-en-ho7-01 KL 0.0298 top-1 0.891; chat-en-ho7-15 KL 0.0372) at a ruler whose floor on those
// windows is the limit itself (each lever alone, and the qualified default hc_output_wmma alone, move them as much).
inline constexpr ::strix::profile::RegimeGate fast_regimes[] = {
    {"qsa_budget", "1536", 1, 8192, "8K gate per window PASS: 8 ho7 cases KL <= 0.0088, top-1 >= 0.957 (FAST bundle); 8K 1,895 -> 2,035 tok/s"},
    {"moe_route_bf16", "1", 1, 8192, "gated in the same FAST bundle (route value 1 captured in every FAST run)"},
    {"gdn_fp16_chunk", "1", 1, 8192, "gated in the same FAST bundle (identity fp16-v1+hcout16+gdnchunk16+qsab1536)"},
};
