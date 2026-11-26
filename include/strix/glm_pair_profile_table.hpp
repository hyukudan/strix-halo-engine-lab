#pragma once

#include "strix/engine_profile.hpp"

#include <array>

namespace strix::glm::rank_defaults {

inline constexpr char kSampledMtpCoupledIdentity[] = "glm-sampled-mtp-coupled-v1";
inline constexpr char kSampledMtpRejectionIdentity[] = "glm-sampled-mtp-rejection-v1";

// The WMMA bundle failed the FAST gate. TURBO admits the measured complete bundle through the ladder below.
// Individual arithmetic switches remain explicit; their standalone proposals need their own receipts.
inline constexpr std::array<strix::profile::Lever, 11> product_profile_table{{
    {"pf_wmma", "STRIX_GLM_PF_WMMA", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, true}, {true, true, true, true}, "glm-pf-wmma-x", "0",
     "docs/optimizations.md#glm-pf-wmma-x: UNQUALIFIED for fast: the FAST prefill set fails the 8K quality gate (KL 0.0587 and "
     "0.492 on two windows, no single member carries it); TURBO only in the measured pair regime; OFF equals exact anchors", false, {}, false, {}, {{{"0", "v1"}, {"1", "v1"}}}},
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
inline constexpr std::string_view kTurboPrefillSettings =
    "tp2-usb4-host-2x2x40-r1b-gemv-exl3-pipeline2-routed-qwin0-mtp-wmma-x3-mla-kda-attn16";
inline constexpr std::uint64_t kTurboPrefillCapacity = 36864;
inline constexpr unsigned kTurboPrefillRows = 8192, kTurboPrefillGenerated = 64;

#define STRIX_GLM_TURBO_ARM(prompt) \
    "ladder-arm/v4;exec/v1 backend=hip device=gfx1151/Radeon-8060S prompt=" prompt \
    " context=36864 lanes=1 chunk=8192 prefill_rows=8192 decode_rows=1 verify_rows=0 generated=64" \
    " settings=tp2-usb4-host-2x2x40-r1b-gemv-exl3-pipeline2-routed-qwin0-mtp-wmma-x3-mla-kda-attn16;" \
    "pf_wmma=0{v1},stored_i4=exl3{exl3-v1},mla_wmma=0{v1},mla_ob16=0{v1},kda_wmma=0{v1},attn_wmma=0{v1}," \
    "idxproj_wmma=0{v1},router_wmma=0{v1},idx_scores=exact{glm-idx-wmma-exact-v1}," \
    "kda_state16=0{v1},serve_sampled_mtp=0{glm-plain-sampling-v1};candidate=pf_wmma=1{v1};tier=turbo"

// Two measured prompt lengths, one fresh request at a time on the USB4 pair. Ratios are the lower of
// the two observed TURBO/FAST throughput ratios at each length, with a conservative 10% margin.
inline constexpr strix::profile::LadderRow product_ladder_rows[] = {
    {"pf_wmma", "1", strix::profile::Name::turbo, strix::profile::Phase::prefill, kTurboPrefillDevice,
     8108, 8108, 1, 1, 1.51, 0.10, "GLM pair 8K TURBO vs FAST, docs/models/glm-5.3/README.md",
     false, {}, {}, STRIX_GLM_TURBO_ARM("8108")},
    {"pf_wmma", "1", strix::profile::Name::turbo, strix::profile::Phase::prefill, kTurboPrefillDevice,
     31258, 31258, 1, 1, 1.54, 0.10, "GLM pair 32K TURBO vs FAST, docs/models/glm-5.3/README.md",
     false, {}, {}, STRIX_GLM_TURBO_ARM("31258")}};
#undef STRIX_GLM_TURBO_ARM
inline const strix::profile::Engine product_profile_engine{"glm-pair", "", product_profile_table, false, "glm-turbo-v1",
                                                           {}, {}, product_ladder_rows};

}  // namespace strix::glm::rank_defaults
