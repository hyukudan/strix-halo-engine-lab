#pragma once
// FAST identity glm-pf-wmma-x, host side (no HIP): calls that actually ran on the WMMA linears, so a run's identity
// line reports executed arithmetic, not only the requested switch (STRIX_GLM_PF_WMMA=1 still leaves the dense calls
// below the X3 threshold and the MoE calls below V11 on the exact cores).
#include "strix/env_flags.hpp"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strix::glm::pf::wmma_x {
// STRIX_GLM_PF_WMMA=1 selects this identity for the exact prefill X3 calls (dense and MoE); unset or 0 = exact.
// Part of the TP2 startup agreement (rank_defaults::kPfWmmaConfig): both ranks must resolve the same value.
inline bool enabled() {
    const char* p = std::getenv("STRIX_GLM_PF_WMMA");
    return p && std::strcmp(p, "1") == 0;
}
inline std::atomic<std::uint64_t> dense_calls{0}, moe_calls{0}, iu4_moe_calls{0};
}  // namespace strix::glm::pf::wmma_x

namespace strix::glm::pf::kda_wmma {
// FAST identity glm-pf-kda-wmma-v1 (glm_pf_kda_wmma.hpp): on by default when the FAST prefill route is selected
// (STRIX_GLM_PF_WMMA=1); STRIX_GLM_PF_KDA_WMMA=0 opts out, =1 forces it. The exact route (FAST not selected,
// switch unset) never takes it. Part of the TP2 startup agreement (rank_defaults::kPfKdaWmmaConfig).
inline bool enabled() {
    const char* p = std::getenv("STRIX_GLM_PF_KDA_WMMA");
    if (p == nullptr || *p == '\0') return wmma_x::enabled();
    if (std::strcmp(p, "0") == 0) return false;
    if (std::strcmp(p, "1") == 0) return true;
    throw std::invalid_argument("STRIX_GLM_PF_KDA_WMMA must be 0 or 1");
}
inline std::atomic<std::uint64_t> calls{0};  // launches that ran on the WMMA core (identity reporting)
// x16 tile form of the WMMA groups of x (KDA f_a | g_a | b and the indexer group): same bits as the direct form.
// STRIX_GLM_PF_X16=0 restores the direct form (A/B and rollback).
inline bool x16_enabled() { return strix::envcfg::flag("STRIX_GLM_PF_X16", true, strix::EnvFlagMatch::exact); }  // recorded
inline std::atomic<std::uint64_t> x16_calls{0};  // of those, the x16 tile form (core f16_dense_tall X16Cols; same bits)
}  // namespace strix::glm::pf::kda_wmma

namespace strix::glm::pf::idxproj_wmma {
// FAST identity glm-pf-idxproj-wmma-v1: the DSA indexer projections of x (wk | gate | weights_proj, one group) on the
// shared dense FP16 WMMA core (x16 form from core::f16_dense_tall::kX16Rows rows) instead of the exact dense v3
// chains: changes the indexer scores, so the selected tokens may differ on near ties. kWithFast = false: opt-in
// (STRIX_GLM_PF_IDXPROJ_WMMA=1) until its gate passes; true: on by default with the FAST route (STRIX_GLM_PF_WMMA=1).
// STRIX_GLM_PF_IDXPROJ_WMMA=0 opts out, =1 forces it. Part of the TP2 startup agreement (rank_defaults::kPfIdxProjWmmaConfig).
inline constexpr bool kWithFast = false;
inline bool enabled() {
    const char* v = strix::envcfg::text("STRIX_GLM_PF_IDXPROJ_WMMA");  // recorded for the effective config / receipt
    if (v == nullptr || *v == '\0') return kWithFast && wmma_x::enabled();
    if (std::strcmp(v, "0") == 0 || std::strcmp(v, "1") == 0) return v[0] == '1';
    throw std::invalid_argument("STRIX_GLM_PF_IDXPROJ_WMMA must be 0 or 1");
}
inline std::atomic<std::uint64_t> calls{0};  // indexer groups that ran on the WMMA core (identity reporting)
}  // namespace strix::glm::pf::idxproj_wmma
