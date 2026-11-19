#pragma once

#include "strix/engine_profile.hpp"

#include <array>

namespace strix::glm::rank_defaults {

// Default/quality export nothing. The FAST column selects no lever today: the FAST prefill set (pf_wmma) fails the
// 8K quality gate, so it is UNQUALIFIED for fast and runs only with an explicit STRIX_GLM_PF_WMMA=1 (the existing
// WMMA startup vote checks both ranks). The turbo cell keeps its previous value and status (turbo has its own gate).
inline constexpr std::array<strix::profile::Lever, 3> product_profile_table{{
    {"pf_wmma", "STRIX_GLM_PF_WMMA", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, true}, {true, true, true, true}, "glm-pf-wmma-x", "0",
     "docs/optimizations.md#glm-pf-wmma-x: UNQUALIFIED for fast: the FAST prefill set fails the 8K quality gate (KL 0.0587 and "
     "0.492 on two windows, no single member carries it); explicit STRIX_GLM_PF_WMMA=1 only until fixed; OFF equals exact anchors"},
    // PENDING turbo cells: not covered by pf_wmma (no code yet; the written analysis precedes any kernel).
    {"router_indexer_wmma", "STRIX_GLM_PF_ROUTER_INDEXER_WMMA", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {false, false, false, false}, "", "0",
     "pending: router and DSA indexer projections on WMMA (not covered by STRIX_GLM_PF_WMMA)"},
    {"kda_state16", "STRIX_GLM_PF_KDA_STATE16", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {false, false, false, false}, "", "0",
     "pending: KDA recurrent state in 16-bit (not covered by STRIX_GLM_PF_WMMA)"},
}};
inline const strix::profile::Engine product_profile_engine{"glm-pair", "", product_profile_table, false, "glm-turbo-v1"};

}  // namespace strix::glm::rank_defaults
