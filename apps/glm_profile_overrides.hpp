#pragma once
// GLM prefill switches that change arithmetic but are not rows of the profile table: an explicit =1 forces the
// bit-changing path whatever the tier resolved. The rank CLI hands them to the shared profile resolution so its tier
// notice never claims the exact column for a run that takes one of them.

#include <string>
#include <vector>

#include "strix/engine_profile.hpp"

namespace strix::glm {

// getenv-shaped lookup; returns "name=1" for each standalone switch forced on (first character '1', the envcfg::flag rule).
template <class Getenv>
[[nodiscard]] inline std::vector<std::string> explicit_prefill_overrides(Getenv&& get) {
    static constexpr const char* kSwitches[] = {"STRIX_GLM_PF_MLA_WMMA", "STRIX_GLM_PF_KDA_WMMA", "STRIX_GLM_PF_ATTN_WMMA",
                                                "STRIX_GLM_PF_IDXPROJ_WMMA", "STRIX_GLM_PF_ROUTER_WMMA"};
    std::vector<std::string> out;
    for (const char* key : kSwitches) {
        const char* v = get(key);
        if (v != nullptr && v[0] == '1') out.push_back(std::string(key) + "=1");
    }
    return out;
}

}  // namespace strix::glm
