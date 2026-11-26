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
     .provisional = {false, true, false}, .identity = {{{"default", "v1"}, {"1536", "v1"}, {"0", "v1"}}}},
    {.name = "moe_route_bf16", .env = "STRIX_QPF2K_MOE_ROUTE_BF16", .option = "--prefill-moe-route-bf16",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, true, false},
     .target = "BF16 MoE route storage", .off = "0",
     .source = "selected for fast ahead of its qualification receipt; qualification in progress; changes logits",
     .native_option = true, .provisional = {false, true, false}, .identity = {{{"0", "v1"}, {"1", "v1"}}}},
    {.name = "moe_route_prune", .env = "STRIX_QPF2K_MOE_ROUTE_PRUNE", .option = "--prefill-moe-route-prune",
     .value = {"", "", "", "cap9"}, .builtin = "off",
     .kind = Kind::bit_changing, .ready = {true, false, true, true}, .runnable = {false, true, false, true},
     .target = "rel<f>[n] | topp<f>[n] | cap<k>[n]", .off = "off",
     .source = "RoutePrune: cap9 keeps the 9 heaviest routes of each row; changes logits. "
               "TURBO applies it only where a workload speed receipt over FAST covers the regime (ladder_rows: gfx1151 prefill bands); quality: composite at 8K KL 0.00528, top-1 0.9707, dNLL +0.0050 vs EXACT", .identity = {{{"off", "v1"}, {"rel0.02", "route-prune-v1"}, {"cap9", "route-prune-v1"}}}},
    {.name = "gdn_fp16_chunk", .env = "STRIX_QPF2K_GDN_FP16_CHUNK", .option = "--gdn-fp16-chunk",
     .value = {"", "1", ""}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, true, false},
     .target = "qwen-gdn-fp16-chunk rev 2 (=1), rev 3 (=3)", .off = "0",
     .source = "chunked GDN prefill on FP16 WMMA (FP64 fixture screen relL2 ~4e-4); selected for fast ahead of its qualification receipt; qualification in progress; changes logits",
     .provisional = {false, true, false}, .identity = {{{"0", "v1"}, {"1", "v1"}, {"3", "qwen-gdn-fp16-chunk-v3"}}}},
    // TURBO prefill candidates (include/strix/prefill_identity.hpp). Explicit switches remain available while timing is pending.
    {.name = "index_proj_wmma", .env = "STRIX_PREFILL_INDEX_PROJ_WMMA", .option = "",
     .value = {"", "", "", "1"}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {false, false, false, true},
     .target = "idxproj16", .off = "0", .source = "turbo ahead of qualification; changes logits",
     .provisional = {false, false, false, true}, .identity = {{{"0", "v1"}, {"1", "v1"}}}},
    {.name = "index_score_wmma", .env = "STRIX_PREFILL_INDEX_SCORE_WMMA", .option = "",
     .value = {"", "", "", "1"}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {false, false, false, true},
     .target = "idxscore16", .off = "0", .source = "turbo ahead of qualification; changes logits",
     .provisional = {false, false, false, true}, .identity = {{{"0", "v1"}, {"1", "v1"}}}},
    {.name = "ple_rows_f16", .env = "STRIX_PREFILL_PLE_ROWS_F16", .option = "",
     .value = {"", "", "", "1"}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {false, false, false, true},
     .target = "plef16", .off = "0", .source = "turbo ahead of qualification; changes logits",
     .provisional = {false, false, false, true}, .identity = {{{"0", "v1"}, {"1", "v1"}}}},
    {.name = "gdn_gates_f16", .env = "STRIX_PREFILL_GDN_GATES_F16", .option = "",
     .value = {"", "", "", "1"}, .builtin = "0",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {false, false, false, true},
     .target = "gatef16 (needs gdn_fp16_chunk)", .off = "0", .source = "turbo ahead of qualification; changes logits",
     .provisional = {false, false, false, true}, .identity = {{{"0", "v1"}, {"1", "v1"}}}},
    {.name = "output_head", .env = "", .option = "--output-head-profile",
     .value = {"", "fp16-output-head", ""}, .builtin = "default",
     .kind = Kind::bit_changing, .ready = {true, false, true}, .runnable = {false, false, false},
     .target = "fp16 output-head projection", .off = "off", .source = "pending output-head implementation and qualification", .identity = {{{"default", "v1"}, {"fp16-output-head", "v1"}, {"off", "v1"}}}},
    {.name = "decode_numerical_profile", .env = "", .option = "--numerical-profile",
     .value = {"", "", "", ""}, .builtin = "exact-v5",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, false, true, true},
     .target = "A8 decode recipe", .off = "exact-v5",
     .source = "explicit decode recipe (--numerical-profile); it needs the explicit low-level organization it names, so no profile cell selects it",
     .native_option = true, .identity = {{{"exact-v5", "exact-v5"}}}},
    {.name = "gsq_expert_pack", .env = "STRIX_GSQ_Q20_A8", .option = "",
     .value = {"", "", "", ""}, .builtin = "0", .kind = Kind::pack,
     .ready = {true, false, true, false}, .runnable = {true, false, true, true},
     .target = "admit GSQ-RCO expert storage", .off = "0", .source = "requires a supplied GSQ pack; loader admission and model identity are independent of the tier", .identity = {{{"0", "v1"}, {"1", "gsq-rco-v1"}}}},
    {.name = "hc_output_wmma", .env = "STRIX_PREFILL_HC_OUTPUT_WMMA", .option = "",
     .value = {"", "", "", ""}, .builtin = "1", .kind = Kind::bit_changing,
     .ready = {true, false, true, false}, .runnable = {true, false, true, true},
     .target = "hcout16", .off = "0", .source = "native prefill identity registry and its route/master admission", .identity = {{{"0", "v1"}, {"1", "hcout16-v1"}}}},
    {.name = "moe_small", .env = "STRIX_QPF2K_MOE_SMALL", .option = "",
     .value = {"", "", "", ""}, .builtin = "0", .kind = Kind::exact,
     .ready = {true, false, true, false}, .runnable = {true, false, true, true},
     .target = "gate/up count-class prefetch (1..128)", .off = "0", .source = "bitwise schedule of the native K-quant prefill tile; requires tune 80/82 without DA16", .identity = {{{"0", "v1"}, {"1", "v1"}}}},
    {.name = "moe_tile64", .env = "STRIX_QPF2K_MOE_TILE64", .option = "",
     .value = {"", "", "", ""}, .builtin = "0", .kind = Kind::exact,
     .ready = {true, false, true, false}, .runnable = {true, false, true, true},
     .target = "64x64 DA16 register tile", .off = "0", .source = "bitwise schedule; declines with the narrow tier split", .identity = {{{"0", "v1"}, {"1", "v1"}}}},
    {.name = "moe_direct_b", .env = "STRIX_QPF2K_MOE_DIRECT_B", .option = "",
     .value = {"", "", "", ""}, .builtin = "0", .kind = Kind::exact,
     .ready = {true, false, true, false}, .runnable = {true, false, true, true},
     .target = "Q4_K prepared direct-B planes", .off = "0", .source = "bitwise operand schedule; requires native route and staging budget", .identity = {{{"0", "v1"}, {"1", "v1"}}}},
    {.name = "moe_tiers", .env = "STRIX_QPF2K_MOE_TIERS", .option = "",
     .value = {"", "", ""}, .builtin = "0", .kind = Kind::exact,
     .ready = {true, false, true, false}, .runnable = {true, false, true, true},
     .target = "padded or compact XOR narrow tier", .off = "0", .source = "bitwise CSR batch schedule; see the native MoE policy", .identity = {{{"0", "v1"}, {"1", "v1"}}}},
    {.name = "moe_down_lds_epi", .env = "STRIX_QPF2K_MOE_DOWN_LDS_EPI", .option = "",
     .value = {"", "", ""}, .builtin = "0", .kind = Kind::exact,
     .ready = {true, false, true, false}, .runnable = {true, false, true, true},
     .target = "vector epilogue from LDS", .off = "0", .source = "bitwise down store for matching raw-dot storage", .identity = {{{"0", "v1"}, {"1", "v1"}}}},
    {.name = "moe_route_ctrl", .env = "STRIX_QPF2K_MOE_ROUTE_CTRL", .option = "",
     .value = {"", "", "", "1"}, .builtin = "0", .kind = Kind::bit_changing,
     .ready = {true, false, true, false}, .runnable = {true, false, true, true},
     .target = "r22 raw-dot storage", .off = "0", .source = "experimental route-control rounding; quality and timing against FAST pending",
     .provisional = {false, false, false, true}, .identity = {{{"0", "v1"}, {"1", "route-control-r22-v1"}}}},
    {.name = "gsq_prefill_f16", .env = "STRIX_GSQ_PREFILL_F16", .option = "",
     .value = {"", "", ""}, .builtin = "1", .kind = Kind::bit_changing,
     .ready = {true, false, true, false}, .runnable = {true, false, true, true},
     .target = "GSQ FP16 versus integer prefill form", .off = "0", .source = "requires a supplied GSQ pack and admitted fast prefill form", .identity = {{{"0", "v1"}, {"1", "v1"}}}},
    {.name = "dense_q6p_pack", .env = "", .option = "--dense-q6p-pack",
     .value = {"", "", ""}, .builtin = "native", .kind = Kind::pack,
     .ready = {true, false, true, false}, .runnable = {true, false, true, true},
     .target = "packed dense Q6P32", .off = "", .source = "explicit dense asset and its existing loader admission", .native_option = true, .identity = {{{"native", "v1"}}}},
    {.name = "dense_trellis_pack", .env = "", .option = "--dense-trellis-pack",
     .value = {"", "", ""}, .builtin = "native", .kind = Kind::pack,
     .ready = {true, false, true, false}, .runnable = {true, false, true, true},
     .target = "pinned dense trellis asset", .off = "", .source = "explicit dense asset paired with the named numerical recipe", .native_option = true, .identity = {{{"native", "v1"}}}},
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

// Receipts of the cap9 TURBO rows (eng_land Q-CAP, binary 88ce20962408081f, p8192, same ticket, FAST vs FAST+cap9):
// 4K 2107.1 -> 2047.1 ms; 8K 4020.9 (3 valid) -> 3919.3 ms (4/4 valid), +2.6 %; 32K 16140.4 -> 15846.1 ms, +1.9 %.
// Confirmed on main ca2de6ae8 (bin 12ea00ce): FAST+cap9 3937.4 ms (2/2 valid), record FAST 4041.2 ms. Quality of the
// composite vs EXACT at 8K: KL 0.00528, top-1 0.9707, dNLL +0.0050. The 4096-8192 band takes the 8K ratio (the 4K
// ratio, 2107.1 / 2047.1 = 1.029, is above it). Lower edges timed on the integration head: 4,096 +4.2 %,
// 8,193 +3.9 %, so both rows keep their nominal lower edges.
inline constexpr double kCap9Speedup8k = 4020.9 / 3919.3;
inline constexpr double kCap9Speedup32k = 16140.4 / 15846.1;
inline constexpr double kCap9Noise = 0.01;
inline constexpr std::string_view kCap9Receipt8k =
    "eng_land Q-CAP 11-24 bin 88ce20962408081f p8192 FAST 4020.9 ms -> cap9 3919.3 ms (+2.6 %, 4/4 valid); "
    "main ca2de6ae8 re-measure cap9 3937.4 ms; lower edge (integration head 2019eacec5, bin fb814e2f1dba372a, "
    "one gfx1151 node, ABBA x2) p4096 FAST 2396.2 ms -> cap9 2300.3 ms (medians, 3 / 4 valid, +4.2 %); "
    "quality vs EXACT 8K KL 0.00528 top-1 0.9707 dNLL +0.0050";
inline constexpr std::string_view kCap9Receipt32k =
    "eng_land Q-CAP 11-24 bin 88ce20962408081f 32K FAST 16140.4 ms -> cap9 15846.1 ms (+1.9 %); "
    "lower edge (integration head 2019eacec5, bin fb814e2f1dba372a, one gfx1151 node, ABBA x2) p8193 FAST 3995.0 ms "
    "(1 valid, 3 evicted all slower) -> cap9 3843.7 ms (median of 3 valid, +3.9 %); "
    "quality measured at 8K (KL 0.00528 top-1 0.9707 dNLL +0.0050)";

// Ladder rows of this table (resolve() always reads them): workload speed receipts for automatic profile cells, and the
// incumbent grants of the cells the resolver ran before the ladder rule (inside fast_regimes they are qualified, outside
// provisional, as before). Quality gates above also apply.
inline constexpr ::strix::profile::LadderRow ladder_rows[] = {
    ::strix::profile::incumbent_cell("qsa_budget", "1536", Name::fast,
        "FAST bundle: 8K gate per window PASS (fast_regimes, prompts <= 8,192); provisional above", "v1"),
    ::strix::profile::incumbent_cell("moe_route_bf16", "1", Name::fast,
        "FAST bundle: 8K gate per window PASS (fast_regimes, prompts <= 8,192); provisional above", "v1"),
    ::strix::profile::incumbent_cell("gdn_fp16_chunk", "1", Name::fast,
        "FAST bundle: 8K gate per window PASS (fast_regimes, prompts <= 8,192); provisional above", "v1"),
    ::strix::profile::incumbent_cell("qsa_budget", "1536", Name::turbo, ::strix::profile::kProvisionalBeforeLadder, "v1"),
    ::strix::profile::incumbent_cell("moe_route_bf16", "1", Name::turbo, ::strix::profile::kProvisionalBeforeLadder, "v1"),
    ::strix::profile::incumbent_cell("gdn_fp16_chunk", "1", Name::turbo, ::strix::profile::kProvisionalBeforeLadder, "v1"),
    ::strix::profile::incumbent_cell("gdn_gates_f16", "1", Name::turbo, ::strix::profile::kProvisionalBeforeLadder, "v1"),
    ::strix::profile::incumbent_cell("index_proj_wmma", "1", Name::turbo, ::strix::profile::kProvisionalBeforeLadder, "v1"),
    ::strix::profile::incumbent_cell("index_score_wmma", "1", Name::turbo, ::strix::profile::kProvisionalBeforeLadder, "v1"),
    ::strix::profile::incumbent_cell("ple_rows_f16", "1", Name::turbo, ::strix::profile::kProvisionalBeforeLadder, "v1"),
    // TURBO cap9 timing receipts (band receipts: bound per run to its own FAST reference by bind_band_receipts, so the
    // row decides only on a known execution, the declared lever chain, gfx1151 prefill and the band). Update the
    // numbers here (ms FAST over cap9 gives speedup = fast / cap9); the receipt text names the evidence.
    {.lever = "moe_route_prune", .value = "cap9", .tier = Name::turbo, .phase = ::strix::profile::Phase::prefill,
     .device = "gfx1151", .min_context = 4096, .max_context = 8192, .min_lanes = 1, .max_lanes = 1,
     .speedup = kCap9Speedup8k, .noise = kCap9Noise, .receipt = kCap9Receipt8k, .band_bound = true,
     .band_evidence = {{4096, 8192}, {8192}, "Composite vs EXACT at 8K, KL 0.00528, top-1 0.9707, dNLL +0.0050.", false}},
    {.lever = "moe_route_prune", .value = "cap9", .tier = Name::turbo, .phase = ::strix::profile::Phase::prefill,
     .device = "gfx1151", .min_context = 8193, .max_context = 32768, .min_lanes = 1, .max_lanes = 1,
     .speedup = kCap9Speedup32k, .noise = kCap9Noise, .receipt = kCap9Receipt32k, .band_bound = true,
     .band_evidence = {{8193, 32768}, {8192}, "Composite gate at 8K; no 32K gate length cited.", false}},
};
