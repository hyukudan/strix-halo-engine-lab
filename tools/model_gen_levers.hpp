#pragma once
// The profile lever names each declared model accepts in its TOML profile tables (strix-model-gen and the drift
// test validate the same names).
#include "minimax/engine/mm_profile_table.hpp"
#include "strix/deepseek41_pair_profile_table.hpp"
#include "strix/glm_pair_profile_table.hpp"
#include "strix/profiles/ds41_rtx.hpp"
#include "strix/profiles/hy4.hpp"
#include "strix/profiles/inkling2_mini.hpp"
#include "strix/profiles/qwen_cuda.hpp"
#include "strix/profiles/qwen_dense_cuda.hpp"
#include "strix/profiles/qwen_dense_hip.hpp"

#include <span>
#include <string_view>
#include <vector>

namespace strix::model {

inline std::vector<std::string_view> profile_lever_names(std::string_view model) {
    using namespace ::strix;
    std::vector<std::string_view> names;
    const auto append = [&names](std::span<const profile::Lever> rows) {
        for (const auto& row : rows) names.push_back(row.name);
    };

    // A declaration may carry defaults for any of its model's backends.
    if (model == "qwen3.8-flash-next") {
        append(qwen::profile::table);
        append(qwen::cuda_profile::table);
    } else if (model == "qwen3.8-27b") {
        append(qwen35::profile::table);
        append(qwen35::hip_profile::table);
    } else if (model == "deepseek-v4.1-flash") {
        append(deepseek41::pair_defaults::profile_table);
        append(ds41::rtx::profile::shared_rows());
    } else if (model == "glm-5.3-flash") {
        append(glm::rank_defaults::product_profile_table);
    } else if (model == "minimax-m3.1-flash") {
        append(minimax::defaults::profile_table);
    } else if (model == "hy4-flash") {
        append(hy4::profile::table);
    } else if (model == inkling2::kMiniProfile.tag) {
        append(inkling2::kMiniProfile.table);
    }
    return names;
}

} // namespace strix::model
