#pragma once

#include "strix/engine_profile.hpp"

#include <array>
#include <string>
#include <string_view>

namespace strix::glm::rank_defaults {

inline constexpr char kSampledMtpCoupledIdentity[] = "glm-sampled-mtp-coupled-v1";
inline constexpr char kSampledMtpRejectionIdentity[] = "glm-sampled-mtp-rejection-v1";

// The complete WMMA bundle (X3, MLA, KDA, sparse attention) passed FAST gate v2 at 8K with MoE route replay. FAST and
// TURBO run it only where a ladder row below covers the regime; elsewhere the exact prefill cores run.
// Individual arithmetic switches remain explicit; their standalone proposals need their own receipts.
inline constexpr std::array<strix::profile::Lever, 11> product_profile_table{{
    {"pf_wmma", "STRIX_GLM_PF_WMMA", "", {"", "1", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, true, true, true}, {true, true, true, true}, "glm-pf-wmma-x", "0",
     "docs/optimizations.md#glm-pf-wmma-x: FAST gate v2 PASS at 8K with route replay (5 windows, 1,275 rows, KL 0.00018); "
     "FAST and TURBO only in the measured pair regimes (ladder rows); OFF equals exact anchors", false, {}, false, {}, {{{"0", "v1"}, {"1", "v1"}}}},
    {"stored_i4", "STRIX_GLM_MODEL_FORMAT", "", {"", "", "", "fast_i4"}, "exl3", strix::profile::Kind::pack,
     {true, false, true, false}, {true, false, true, true}, "", "exl3",
     "glm-fast-i4: stored routed gate/up on MoE ordinals 25-41, W4A4 prefill and W4A16 decode; load-scoped, ladder evidence pending",
     false, {}, true, {}, {{{"exl3", "exl3-v1"}, {"fast_i4", "glm-fast-i4-v1"}}}},
    {"mla_wmma", "STRIX_GLM_PF_MLA_WMMA", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, false, true, true}, "", "0", "glm-pf-mla-wmma-v1: isolated gate and ladder evidence pending", false, {}, true, {}, {{{"0", "v1"}, {"1", "glm-pf-mla-wmma-v1"}}}},
    {"mla_ob16", "STRIX_GLM_PF_MLA_OB16", "", {}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, false, true, true}, "", "0", "FP16 o-bar staging on MLA WMMA; prior 8K timing neutral", false, {}, true, {}, {{{"0", "v1"}, {"1", "fp16-o-bar-v1"}}}},
    {"kda_wmma", "STRIX_GLM_PF_KDA_WMMA", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, false, true, true}, "", "0", "glm-pf-kda-wmma-v1: isolated gate and ladder evidence pending", false, {}, true, {}, {{{"0", "v1"}, {"1", "glm-pf-kda-wmma-v1"}}}},
    {"attn_wmma", "STRIX_GLM_PF_ATTN_WMMA", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, false, true, true}, "", "0", "glm-pf-attn-wmma: isolated gate and ladder evidence pending", false, {}, true, {}, {{{"0", "v1"}, {"1", "glm-pf-attn-wmma-v1"}}}},
    {"idxproj_wmma", "STRIX_GLM_PF_IDXPROJ_WMMA", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, false, true, true}, "", "0", "glm-pf-idxproj-wmma-v1: isolated gate and ladder evidence pending", false, {}, true, {}, {{{"0", "v1"}, {"1", "glm-pf-idxproj-wmma-v1"}}}},
    {"router_wmma", "STRIX_GLM_PF_ROUTER_WMMA", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, false, true, true}, "", "0", "glm-pf-router-wmma: isolated gate and ladder evidence pending", false, {}, true, {}, {{{"0", "v1"}, {"1", "glm-pf-router-wmma-v1"}}}},
    {"idx_scores", "STRIX_IDX_WMMA", "", {"", "", "", "f16"}, "exact", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, false, true, true}, "", "0", "glm-idx-wmma: gate and ladder evidence on sparse prompts pending", false, {}, true, {}, {{{"exact", "glm-idx-wmma-exact-v1"}, {"0", "glm-idx-valu-v1"}, {"f16", "glm-idx-wmma-f16-v1"},
       {"bf16", "glm-idx-wmma-bf16-v1"}, {"1", "glm-idx-wmma-bf16-v1"}, {"bf16x2", "glm-idx-wmma-bf16x2-v1"}}}},
    {"kda_state16", "STRIX_GLM_PF_KDA_STATE16", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {false, false, false, false}, "", "0",
     "pending: KDA recurrent state in 16-bit (not covered by STRIX_GLM_PF_WMMA)", false, {}, false, {}, {{{"0", "v1"}, {"1", "v1"}}}},
    // Server: sampled requests decode through MTP only by explicit opt-in. Coupled keeps the tokens of plain
    // sampling. Rejection changes per-seed tokens and is admitted by fast / turbo only. CLI and serve share
    // rank_defaults::resolve_sampled_mtp. The builtin keeps the legacy fingerprint.
    {"serve_sampled_mtp", "STRIX_GLM_SERVE_SAMPLED_MTP", "", {}, "0", strix::profile::Kind::exact,
     {false, false, false, false}, {true, true, true, true}, "coupled", "0",
     "pending: pair gates of sampled MTP (coupled: seeded tokens and committed rows == plain sampling, forced rejection, "
     "batch handover, prefix restore, stops; rejection: acceptance/state oracle + distribution gate)",
     false, {}, true, {}, {{{"0", "glm-plain-sampling-v1"},
                           {"coupled", kSampledMtpCoupledIdentity}, {"rejection", kSampledMtpRejectionIdentity}}}},
}};
inline constexpr std::string_view kTurboPrefillDevice = "gfx1151/Radeon-8060S";
// Pair link layouts (strix::dist::LinkLayout::key()) the FAST band receipt was timed on. The first entry is the layout
// of every literal row below (USB4 layout 1: data0 + data1 host streams on both 40 Gb/s cables). A layout outside the
// list declines in the startup pair vote and runs the exact prefill cores. USB4 layout 2bq through the host route
// (data0 host, data1 gpu_direct on both cables), lower band edge 2,049 ABAB: exact cores 3.821 / 3.824 s, bundle
// 2.777 / 2.775 s (x1.377; layout 1 x1.37); the same bundle and exchange bits, one host stream per cable. The GPU-direct
// route (+direct) is not a measured layout.
inline constexpr std::string_view kMeasuredPrefillLayouts[] = {"usb4-host-2x2x40", "usb4-host-2x1x40"};
// The execution settings of a measured prefill: the link layout is part of the key, so a literal row decides only the
// layout it was timed on. prefill_settings(kMeasuredPrefillLayouts[0], routed) spells the two constants below.
inline std::string prefill_settings(std::string_view link, bool routed_scratch) {
    return "tp2-" + std::string(link) + "-r1b-gemv-exl3-pipeline2-" + (routed_scratch ? "routed" : "full") +
           "-qwin0-mtp-wmma-x3-mla-kda-attn16";
}
inline constexpr std::string_view kTurboPrefillSettings =
    "tp2-usb4-host-2x2x40-r1b-gemv-exl3-pipeline2-routed-qwin0-mtp-wmma-x3-mla-kda-attn16";
// The same execution on the original full scratch arena (storage only; STRIX_GLM_PF_SCRATCH auto picks it when it fits).
inline constexpr std::string_view kFullScratchPrefillSettings =
    "tp2-usb4-host-2x2x40-r1b-gemv-exl3-pipeline2-full-qwin0-mtp-wmma-x3-mla-kda-attn16";
inline constexpr std::uint64_t kTurboPrefillCapacity = 36864;
inline constexpr unsigned kTurboPrefillRows = 8192, kTurboPrefillGenerated = 64;

#define STRIX_GLM_TURBO_ARM(prompt) \
    "ladder-arm/v4;exec/v1 backend=hip device=gfx1151/Radeon-8060S prompt=" prompt \
    " context=36864 lanes=1 chunk=8192 prefill_rows=8192 decode_rows=1 verify_rows=0 generated=64" \
    " settings=tp2-usb4-host-2x2x40-r1b-gemv-exl3-pipeline2-routed-qwin0-mtp-wmma-x3-mla-kda-attn16;" \
    "pf_wmma=0{v1},stored_i4=exl3{exl3-v1},mla_wmma=0{v1},mla_ob16=0{v1},kda_wmma=0{v1},attn_wmma=0{v1}," \
    "idxproj_wmma=0{v1},router_wmma=0{v1},idx_scores=exact{glm-idx-wmma-exact-v1}," \
    "kda_state16=0{v1},serve_sampled_mtp=0{glm-plain-sampling-v1};candidate=pf_wmma=1{v1};tier=turbo"

// FAST gate ledger (strix::profile::FastGate): the qualified fast cell and its receipt.
inline constexpr std::string_view kFastGatePfWmma =
    "FAST gate v2 PASS, pair, 8K with MoE route replay (glm_fast gate-fast2.json): 5 windows of 8,192 tokens, 1,275 rows, "
    "pooled KL 0.00018 (null 0.00007), top-1 0.9969, dNLL -0.0013 (se 0.0009), free route-flip ratio 1.05; product-route "
    "head dump bitwise equal to the gated arm. "
    "Gate re-score pending: the current gate v2 re-run with the free-running arms passes with route replay, fails "
    "the absolute bars free-running (KL 0.363, top-1 0.902; the exact arm's own free one-ulp null gives KL 0.367) "
    "and passes with --allow-null-floor. Which rule applies is a pending gate policy decision; the FAST default "
    "stays until the owner decides.";
inline constexpr strix::profile::FastGate fast_gates[] = {{"pf_wmma", "1", kFastGatePfWmma}};

// A prompt inside the FAST band: the startup pair vote asks whether this execution can run the bundle at all; each
// prompt is then decided by the rows below (select_prefill_wmma).
inline constexpr std::uint64_t kPrefillProbePrompt = 8192;

// TURBO: one fresh request at a time on the USB4 pair, ratio = the lower of the two observed TURBO/FAST throughput
// ratios with a conservative 10% margin. Inside the FAST band the FAST cell runs the bundle and TURBO nests it (its
// former 8,108-token receipt, measured against the exact cores, is superseded there by the band row).
inline constexpr strix::profile::LadderRow product_ladder_rows[] = {
    {"pf_wmma", "1", strix::profile::Name::turbo, strix::profile::Phase::prefill, kTurboPrefillDevice,
     31258, 31258, 1, 1, 1.54, 0.10, "GLM pair 32K TURBO vs FAST, docs/models/glm-5.3/README.md",
     false, {}, {}, STRIX_GLM_TURBO_ARM("31258"), {}, false,
     {{31258}, {}, "The TURBO quality citation is 8K, 512 scored rows (exact prompt length unspecified); no 32K gate cited."}},
    // FAST band receipt (LadderRow::band_bound: bound per run to its own reference by profile::bind_band_receipts, so it
    // decides only a known pair execution). Band: the gate windows are 8,192 tokens; the 8K band below them follows the
    // shared band convention (2,049..8,192). Prompts of 2,048 or fewer tokens and above 8,192 keep the exact prefill
    // cores until a replay gate covers them (32K stays TURBO). Ratio: the lower observed 8K EXACT -> bundle ratio
    // (11-19 G1 cold 12.319 -> 8.227 s x1.497; warm 655 -> 1,000 tok/s x1.53; glm_fast SE2/SF2 cold 604.1 -> 991.8
    // tok/s x1.64), 10 % margin. Lower edge timed on the pair CLI (ABBA, cold prefill): 2,049 x1.37, 4,096 x1.47
    // (both above 1 + noise) -> the edge stays at 2,049.
    {.lever = "pf_wmma", .value = "1", .tier = strix::profile::Name::fast, .phase = strix::profile::Phase::prefill,
     .device = kTurboPrefillDevice, .min_context = 2049, .max_context = 8192, .min_lanes = 1, .max_lanes = 1,
     .speedup = 1.37, .noise = 0.10,  // the lower measured ratio of the band (2,049; 8,192 gave x1.50)
     .receipt = "GLM pair 8K FAST bundle vs EXACT cores (gate v2 8K route replay PASS), docs/optimizations.md#glm-pf-wmma-x; "
                "lower edge 2,049: EXACT 3.83 s, FAST 2.79 s (x1.37); 4,096: x1.47",
     .identity = "v1", .band_bound = true, .band_evidence = {{2049, 4096, 8192}, {8192}, kFastGatePfWmma, false}}};
#undef STRIX_GLM_TURBO_ARM
inline const strix::profile::Engine product_profile_engine{"glm-pair", "", product_profile_table, false, "glm-turbo-v1",
                                                           {}, {}, product_ladder_rows};

}  // namespace strix::glm::rank_defaults
