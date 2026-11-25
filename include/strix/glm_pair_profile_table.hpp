#pragma once

#include "strix/engine_profile.hpp"

#include <array>

namespace strix::glm::rank_defaults {

// Default/quality export nothing. FAST selects no lever: the FAST prefill set (pf_wmma) fails the 8K quality gate, so it
// is UNQUALIFIED for fast and runs only with an explicit STRIX_GLM_PF_WMMA=1. TURBO keeps pf_wmma (incumbent grant
// below). The other bit-changing prefill switches are rows so that an explicit setting is named in the receipt; their
// TURBO cells are candidates that stay at the builtin until a timing row shows them faster in the regime.
inline constexpr std::array<strix::profile::Lever, 10> product_profile_table{{
    {"pf_wmma", "STRIX_GLM_PF_WMMA", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, true}, {true, true, true, true}, "glm-pf-wmma-x", "0",
     "docs/optimizations.md#glm-pf-wmma-x: UNQUALIFIED for fast: the FAST prefill set fails the 8K quality gate (KL 0.0587 and "
     "0.492 on two windows, no single member carries it); explicit STRIX_GLM_PF_WMMA=1 only until fixed; OFF equals exact anchors", false, {}, false, {}, {{{"0", "v1"}, {"1", "v1"}}}},
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
}};
// Ladder rows (strix::profile::LadderRow): incumbent grant for the turbo cell the resolver ran before the ladder rule.
inline constexpr strix::profile::LadderRow product_ladder_rows[] = {
    strix::profile::incumbent_cell("pf_wmma", "1", strix::profile::Name::turbo,
                                   "glm-pf-wmma-x turbo cell ready before the ladder rule (docs/optimizations.md#glm-pf-wmma-x)", "v1")};
inline const strix::profile::Engine product_profile_engine{"glm-pair", "", product_profile_table, false, "glm-turbo-v1",
                                                           {}, {}, product_ladder_rows};

}  // namespace strix::glm::rank_defaults
