// src/minimax/engine/engine.hip:2056
{"decode-layout", 'a', "model+slot+position+prefix", "msa_fused_tests.hip:run_case fused vs current; rows == alone hardware gate for 1..8 rows", ""},
// src/minimax/engine/engine.hip:1456
{"msa-dense", 'a', "model+position+prefix; 128-key blocks", "msa_fused_tests.hip:run_case dense v3/v4 and n4-mixed/n8-mixed; prefill_batch_cases", ""},
// src/minimax/attn/index_proj.hip:341
{"index-proj", 'a', "index projection FP32 chain", "index_proj_tests.hip; split/ring/LDS bitwise gates", ""},
// src/minimax/engine/engine.hip:1372
{"index-fallback", 'a', "index projection FP32 chain", "mm_kernels.hpp dense_t16 contract; engine index fallback contract; index_proj_tests.hip", ""},
// src/minimax/moe/mmx_moe.hip:1128
{"decode-wide", 'a', "EXL3 vector shape+bits+S", "mm_defaults.cpp moe_wide; rows == alone hardware gate; mmx_moe.hip wide-column contract", ""},
// src/minimax/moe/mmx_moe.hip:1195
{"decode-moe-core", 'a', "EXL3 vector shape+bits+S; router row", "hip-mmx-moe-bench exact-form gates; mmx_moe.hpp arithmetic contract; hip_minimax_g4.hip bitwise gate", ""},
// src/minimax/moe/mmx_moe.hip:1071
{"decode-moe-depth", 'a', "fixed dot2/split history", "mmx_moe.hip launch_experts/launch_experts3/launch_experts4; bench bitwise form gates", ""},
// src/minimax/moe/mmx_moe.hip:2404
{"decode-linears", 'a', "EXL3 vector shape+bits+S", "mmx_moe.hip linear schedule; mm_defaults.cpp linear_v11 bitwise gate; ids==greedy is secondary", ""},
// src/minimax/moe/mmx_moe.hip:2206
{"linear-split", 'a', "input+output+vector policy version", "mmx_trellis_device.hpp:vector_splits and declared finish order", ""},
// src/minimax/moe/mmx_moe.hip:2188
{"linear-resources", 'a', "job descriptors+fixed arithmetic family", "plan_linears; enqueue_linears_impl validation; strict source receipt", ""},
// src/minimax/moe/mmx_prefill.hip:785
{"prefill-tile", 'a', "matrix+fixed splits+WMMA chain", "mmx_prefill_core.hpp header contract; mmx_prefill.hip TILE comment", ""},
// src/minimax/moe/mmx_prefill.hip:639
{"prefill-mt", 'a', "matrix+fixed splits+per-row WMMA chain", "mm_defaults.cpp pf_mtfit bitwise gate; core contract", ""},
// src/minimax/moe/mmx_prefill.hip:573
{"prefill-core", 'a', "matrix+fixed splits+per-row WMMA chain", "mm_defaults.cpp pf_core_v4; every split partial bitwise gate", ""},
// src/minimax/moe/mmx_prefill.hip:710
{"prefill-pipe", 'a', "matrix+fixed splits+WMMA chain", "mm_defaults.cpp pf_moe_pipe bitwise gate; source core8 comments; nondefault combinations need same-partial gate", ""},
// src/minimax/moe/mmx_prefill.hip:647
{"prefill-fuse", 'a', "single-split WMMA+epilogue contract", "prefill gufuse intermediate/producer debug gate in hip-mmx-moe-bench; default OFF; combinations outside gate remain unqualified", ""},
// src/minimax/moe/mmx_prefill.hip:416
{"prefill-table", 'a', "identical item table bytes", "prefill table ring event ownership; default OFF; A/B pftab gate in rank app", ""},
// src/minimax/attn/msa_fused.hip:1714
{"msa-attend", 'a', "MSA block128/chunk/FP32 combine contract", "msa_fused_tests.hip:run_case versions/head counts/graph replay", ""},
// src/minimax/attn/msa_fused.hip:1693
{"msa-grid", 'a', "MSA query+block identity", "msa_fused_tests.hip fused grid3; index tests; same_out score bytes", ""},
// src/minimax/attn/msa_fused.hip:1934
{"msa-scratch", 'a', "fixed unit rows+absolute positions+MSA contract", "msa_fused_tests.hip:prefill_batch_cases and every-batch producer/selection/output bytes", ""},
// src/minimax/engine/engine.hip:1727
{"prefill-schedule", 'a', "immutable ordered chunk list (max_rows fixed)", "serve --resume-check with layer budgets 1/7/60 bitwise gate; engine state+final rows contract; no max_rows equivalence claim", ""},
// src/minimax/engine/engine.hip:1087
{"post-path", 'a', "rank order0,1+post arithmetic", "mm_defaults.cpp xbound bitwise gate; rank xb_check N/verify; mm_kernels post contract", ""},
// src/minimax/engine/engine.hip:1066
{"router-batch", 'a', "router row FP32 chain+expert tie order", "mm_defaults.cpp pf_router_batch bitwise gate; enqueue_router_rows byte contract", ""},
// src/minimax/engine/engine.hip:1670
{"head-path", 'a', "vocab partition+GEMV+lower-id tie", "xfer selftest/argmax_exchange; rows == alone hardware gate; hip-minimax-g4", ""},
// src/minimax/engine/engine.hip:214
{"workspace", 'a', "descriptor+required capacity", "engine Lib::mm; DraftLinears::lib_mm; enqueue_prefill_moe workspace guard", ""},
// src/minimax/engine/engine.hip:579
{"staging", 'a', "same per-row operations and rank order", "hip_minimax_g4.hip bitwise gate; shared-first source sorted expert loop", ""},
// src/minimax/xfer/minimax_xfer.hip:161
{"transport", 'a', "wire tags+rank order+payload bytes", "xfer selftest (xfer-report); mm_defaults.cpp transport hp exact; pfx CRC/tag validation", ""},
// src/minimax/rtxpf/fp32_identity.hpp:73
{"rtx-tile", 'a', "F32 identity version+matrix+input order", "fp32_identity_device.inc f32_core_body; hip-mm-fp32id-tests; mm_fp32_identity_tests.cu", ""},
// src/minimax/rtxpf/cuda/mm_rtx_prefill_main.cu:635
{"io-cache", 'a', "model tensor content+dtype+rank slices", "weight loader sizes/format validation; actual dependency notices preserved", ""},
// src/minimax/dspark/hip_minimax_dspark.hip:755
{"draft-native-linears", 'a', "drafter model dtype+matrix+reduction contract", "hip_minimax_dspark_tests.hip linears/attention bitwise gates; mm_defaults.cpp draft linears contract", ""},
// src/minimax/dspark/markov_cand.hip:472
{"markov-topc", 'a', "vocabulary+C+candidate score contract", "hip_markov_cand_tests.hip payload v1/v2 vs CPU; mm_defaults.cpp markov_topc_v2", ""},
// src/minimax/engine/engine.hip:436
{"operation-plan", 'b', "model+operation kind+arithmetic profile+ordered units", "apps/minimax_rank_main.cpp teacher-force comparison explicitly rounding not bitwise; not a gate between identities", ""},
// src/minimax/attn/msa_fused.hip:1780
{"context-plan", 'b', "model+absolute position+prefix tokens+MSA block/chunk contract", "MSA host/reference bitwise tests at boundary and long context", ""},
// src/minimax/dspark/hip_minimax_dspark.hip:990
{"draft-context", 'b', "draft model+absolute position+context window+chunk64", "mm_defaults.cpp dspark_attn bitwise gate at positions 300/1000/4095; no claim all profiles match", ""},
// src/minimax/engine/draft_lin.hip:245
{"draft-model", 'b', "target model hash+draft asset hashes+dtype+head index", "DSpark reference and target acceptance verification; explicit --dspark-exl3/--dspark-head", ""},
// apps/minimax_rank_main.cpp:488
{"draft-markov", 'c', "BLOCKED draft-history identity until request fixes method+C+H+vocabulary+seed", "UNCOVERED candidate/full arithmetic-history equality; target greedy verification is a separate gate", "Markov candidate pruning before rank-dot correction/argmax (full vocabulary vs C/H subset)"},
// apps/minimax_serve_main.cpp:614
{"draft-policy", 'a', "target prefix+per-row vector arithmetic", "speculative tokens == greedy tested chats; rows == alone hardware gate for 1..8 rows; not universal workload qualification", ""},
// apps/minimax_rank_main.cpp:292
{"import-plan", 'b', "target model+arithmetic profile+original chunk list+position/cut+payload digest", "rtxpf_state_compare.py / rank import comparisons; UNCOVERED cross-profile handoff, explicit blocked gate", ""},
// src/minimax/engine/engine.hip:569
{"model-layout", 'b', "model tensor hashes+TP2 partition+rank order", "G3/G4 model gates; pack validation; F32 model fidelity remains draft pending all gates", ""},
// src/minimax/moe/mmx_prefill.hip:476
{"prefill-routed-split", 'c', "BLOCKED until matrix+arithmetic-version fixed S (no expert occupancy)", "UNCOVERED: known failed identity; existing core/mtfit receipts hold S fixed", "L3 routed gate/up WMMA k-tile24 reset vs continuation (rows1->9,S16->8)"},
// src/minimax/moe/mmx_prefill.hip:477
{"prefill-shared-split", 'c', "BLOCKED until shared matrix+arithmetic-version fixed S", "UNCOVERED: known failed max_rows identity", "L3 shared gate/up WMMA k-tile24 reset vs continuation (rows1->9,S16->8)"},
// src/minimax/engine/engine.hip:227
{"library-prefill", 'c', "BLOCKED until library exposes frozen full matrix arithmetic tuple", "UNCOVERED across row counts; bitwise gate compares fixed row count only", "L0 q_proj strix_gemm core (first internal divergent FP op unavailable without dependency receipt)"},
// src/minimax/engine/engine.hip:1734
{"prefill-unit-cuts", 'c', "BLOCKED until original request prompt+absolute chunk list+profile immutable", "UNCOVERED across max_rows/cache-cut boundaries; server deliberately has no prefix cache", "L0 q_proj GEMM after first changed unit cut (before L3 MoE)"},
// src/minimax/engine/engine.hip:658
{"arithmetic-profile", 'c', "BLOCKED until request stores target+draft arithmetic profile and rejects mismatch", "UNCOVERED profile equality; FP32 path is an opt-in distinct draft identity", "L0 q_proj input transform RN16 materialization vs FP32 transform; draft unchunked PV vs chunk partials"},
// src/minimax/dspark/hip_minimax_dspark.hip:949
{"draft-append", 'c', "BLOCKED until draft append absolute-position plan fixed or pin vector per row", "UNCOVERED append32 vs8 intermediate/cache byte identity", "draft append FC GEMM32 vs GEMV8, then K/V projected context (first library core op opaque)"},
// src/minimax/moe/mmx_prefill.hip:881
{"diagnostic-cores", 'c', "BLOCKED in production; diagnostic-only plan", "UNCOVERED for production by design; expected-wrong-output timing APIs", "first skipped A/weight load or attention compute in diagnostic producer"},
// src/minimax/rtxpf/cuda/mm_rtx_kernels.cu:601
{"rtx-attention", 'c', "BLOCKED handoff until MSA cache/output bytes match exact producer plan", "UNCOVERED full native/CUDA MSA model fidelity for all positions/cuts", "RTX MSA index score or attention QK/PV legacy vs shared (first op needs CUDA byte gate)"},
