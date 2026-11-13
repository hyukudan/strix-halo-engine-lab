#pragma once
// GLM levers of the shared route receipt (strix/route_receipt.hpp). The rank CLI and the server record the same keys
// through this one function, so their requested and effective values can be compared entry point against entry point.

#include <cstdint>
#include <string>
#include <string_view>

#include "strix/env_flags.hpp"
#include "strix/route_receipt.hpp"
#include "glm/prefill/glm_pf_exact.hpp"
#include "glm/prefill/glm_pf_wmma.hpp"

namespace strix::glm {

// Records the levers that decide the prefill arithmetic and returns the numerical-route fingerprint.
// entry: "cli" or "http" (informative, outside the fingerprint, like the chunk rows).
inline std::uint64_t record_route_levers(std::string_view entry, unsigned rank, std::string_view prefill_requested,
                                         std::string_view prefill_effective, std::string_view identity, unsigned max_rows) {
    namespace rr = strix::route;
    rr::set_engine("glm");
    rr::set_rank(static_cast<int>(rank));
    const auto asked = [](const char* name) {
        const char* v = strix::envcfg::text(name);
        return std::string(v != nullptr && *v != 0 ? v : "default");
    };
    const std::string wmma = asked("STRIX_GLM_PF_WMMA"), x = asked("STRIX_GLM_PF_EXACT_X"), kda = asked("STRIX_GLM_PF_KDA_WMMA");
    const bool wmma_on = pf::wmma_x::enabled();
    // Resolve at both ends of the supported threshold range. Only auto has two routes;
    // forced routes must never read the unused auto threshold.
    const auto low = pf::exact::route(0), high = pf::exact::route(65536);
    const std::string x_effective = low == high ? std::to_string(static_cast<unsigned>(low)) :
        "auto:min_rows=" + std::to_string(pf::exact::x_auto_rows());
    rr::lever("glm.entry", entry, entry, rr::Reason::selected, false);
    rr::lever("glm.prefill", prefill_requested, prefill_effective);
    rr::lever("glm.identity", identity, identity);
    rr::lever("glm.pf_wmma", wmma, wmma_on ? "f16_wmma" : "exact", wmma_on ? rr::Reason::selected : rr::Reason::disabled);
    rr::lever("glm.pf_kda_wmma", kda, pf::kda_wmma::enabled() ? "1" : "0");
    rr::lever("glm.pf_exact_x", x + ":min_rows=" + asked("STRIX_GLM_PF_EXACT_X_MIN_ROWS"), x_effective, rr::Reason::selected, true,
              "auto takes the X3 route (and the WMMA identity when on) only for calls of at least min_rows rows");
    rr::lever("glm.max_rows", std::to_string(max_rows), std::to_string(max_rows), rr::Reason::selected, false);
    return rr::fingerprint();
}

}  // namespace strix::glm
