#pragma once
// FAST identity glm-pf-wmma-x, host side (no HIP): calls that actually ran on the WMMA linears, so a run's identity
// line reports executed arithmetic, not only the requested switch (STRIX_GLM_PF_WMMA=1 still leaves the dense calls
// below the X3 threshold and the MoE calls below V11 on the exact cores).
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
}  // namespace strix::glm::pf::kda_wmma

#include "strix/env_flags.hpp"

namespace strix::glm::pf::mla_wmma {
// FAST identity glm-pf-mla-wmma-v1 (glm_pf_mla_wmma.hpp): the MLA absorb / out GEMMs on the shared short-K FP16 WMMA
// tile GEMM (core/f16_tile_gemm.hpp). FAST gate PASS (vs FAST without it: KL 0.0053, top-1 98.2 %, dNLL -0.0002, 1,536
// rows prose/code/chat): on by default when the FAST prefill route is selected (STRIX_GLM_PF_WMMA=1);
// STRIX_GLM_PF_MLA_WMMA=0 opts out (exact FP32 GEMM), =1 forces it. The exact route (FAST not selected, switch unset)
// never takes it. Part of the TP2 startup agreement (rank_defaults::kPfMlaWmmaConfig).
inline bool enabled() {
    static const bool value = envcfg::flag("STRIX_GLM_PF_MLA_WMMA", wmma_x::enabled());
    return value;
}
// With it: the fused flash core stores o-bar as the FP16 image the out GEMM's staging would make (same bits, half the
// o-bar bytes written and read). Default on; STRIX_GLM_PF_MLA_OB16=0 restores the FP32 o-bar.
inline bool ob16() {
    static const bool value = envcfg::flag("STRIX_GLM_PF_MLA_OB16", true);
    return value;
}
inline std::atomic<std::uint64_t> calls{0};  // launches that ran on the WMMA core (identity reporting)
}  // namespace strix::glm::pf::mla_wmma

namespace strix::glm::pf::mla_gemm {
// Exact MLA absorb / out GEMM placement (glm_prefill.hip mla_gemm_rows_v3_kernel): heads-mid CTA order, the same bits.
// Default on; STRIX_GLM_PF_MLA_GEMM_HEADS=0 restores the rows-then-head order.
inline bool heads_mid() {
    static const bool value = envcfg::flag("STRIX_GLM_PF_MLA_GEMM_HEADS", true);
    return value;
}

// Explicit-version gates default to exact arithmetic, independent of the environment.
enum class Arithmetic { exact, wmma };
struct Dispatch {
    Arithmetic arithmetic = Arithmetic::exact;
    bool ob16 = true;
    bool heads_mid = true;
    constexpr bool use_wmma(bool fits) const { return arithmetic == Arithmetic::wmma && fits; }
};
// First use must follow application defaults. All readers retain these process-lifetime values;
// changing an environment control requires a restart.
inline Dispatch automatic_dispatch() {
    return {mla_wmma::enabled() ? Arithmetic::wmma : Arithmetic::exact, mla_wmma::ob16(), heads_mid()};
}
}  // namespace strix::glm::pf::mla_gemm
