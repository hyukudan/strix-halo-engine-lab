#pragma once

#include "strix/deepseek41_model.hpp"
#include "strix/model_description.hpp"
#include "strix/model_graph.hpp"

#include <string>

namespace strix::deepseek41 {

// Host inspection only. The existing Config parser and explicit tensor/layer adapters
// remain authoritative. Description fields never supply runtime identity text.
[[nodiscard]] std::string describe_model_toml(const Config& config);

// Require every field (including types) to match the supplied, parsed weight config.
// Reports all contract mismatches together, then reuses parse_config_json's semantic
// checks. This is deliberately not connected to any inference entry point.
[[nodiscard]] Config config_from_model_description(
    const model_description::ModelDescription& description, const Config& expected);

[[nodiscard]] model_graph::ModelGraph graph_from_model_description(
    const model_description::ModelDescription& description, const Config& expected);

// Lossless Config serialization in the existing embedded-config JSON shape.
[[nodiscard]] std::string model_description_config_json(const Config& config);

}  // namespace strix::deepseek41
