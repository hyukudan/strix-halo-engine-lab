#pragma once

#include "strix/model_description.hpp"
#include "strix/model_graph.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace strix::glm::model {

namespace contract {
inline constexpr double rms_eps = 1e-5, hc_eps = 1e-6, l2_eps = 1e-6, ln_eps = 1e-6;
inline constexpr double gate_lower_bound = -5.0, clamp = 10.0, route_scale = 2.5, route_den_eps = 1e-20;
inline constexpr double inv_sqrt128 = 0.088388347648318, inv_sqrt32 = 0.17677669529663687, mla_scale = 0.0625;
}  // namespace contract

// Validation only. Compiled GLM tensor mappings, state effects, schedules and
// printed numerical identities remain authoritative; description text is never
// used to select arithmetic or construct output labels.
[[nodiscard]] std::vector<std::string> description_mismatches(
    const model_description::ModelDescription& description);
// complete=false: only the keys present are checked (admission of a legacy deployment).
[[nodiscard]] std::vector<std::string> config_mismatches(std::string_view json, bool complete = true);
void validate_description(const model_description::ModelDescription& description);
void validate_config(std::string_view json);

// Admission (engine load): a deployed glm-5.3-flash.toml is validated; config.json is checked only when present, and
// only for the keys it carries, so a model directory that loaded before still loads. STRIX_GLM_MODEL_DESCRIPTION_CHECK=0
// skips the whole check (before any file is opened). No current-directory or build-machine path is used.
void validate_model_directory(const std::string& directory);
// Tools: config.json is required and complete.
void validate_model_directory_complete(const std::string& directory);
[[nodiscard]] bool description_check_enabled();

// Opt-in description tooling only. build_model_graph always validates its input;
// inspection requires the deployed TOML and index, even if admission checks are off.
[[nodiscard]] model_graph::ModelGraph build_model_graph(const model_description::ModelDescription& description);
[[nodiscard]] std::string print_model_graph(const std::string& directory);

}  // namespace strix::glm::model
