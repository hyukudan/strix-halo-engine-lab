#pragma once

#include "strix/gguf_reader.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_description.hpp"

#include <vector>

namespace strix::qwen_description {

// Descriptive template only. The adapter owns tensor mapping, activations,
// routing and state transitions; existing compiled schedules remain authoritative.
enum class AttentionKind { gdn, qsa };
enum class StateEffect { gdn_recurrence, gdn_convolution, qsa_kv, qsa_index_pool, ple_convolution };
struct LayerTemplate {
    std::uint64_t index;
    AttentionKind attention; // gdn mutates recurrent/conv state; qsa mutates KV/index pools.
    bool ple;               // PLE also mutates its convolution history.
    std::vector<StateEffect> state_effects;
};
struct GraphTemplate {
    ModelContract contract;
    std::vector<LayerTemplate> layers;
};

[[nodiscard]] ModelContract contract_from_description(const model_description::ModelDescription& file);
[[nodiscard]] ModelContract contract_from_file(const std::string& path);
[[nodiscard]] GraphTemplate graph_template(const model_description::ModelDescription& file);
// Same validated cadence for a resolved host contract; no schedule is executed.
[[nodiscard]] GraphTemplate graph_template(const ModelContract& contract);
// Every mismatch is collected in one std::invalid_argument, with field names,
// expected and actual values (including missing/invalid types).
void assert_matches(const ModelContract& baked, const model_description::ModelDescription& file);
// Checked description graph used by both host inspection commands.
[[nodiscard]] inline GraphTemplate inspect(const model_description::ModelDescription& file) {
    assert_matches(ModelContract::qwen38_flash_next(), file);
    return graph_template(file);
}
// Checks recognized keys only when present; no missing metadata is invented.
// Returns the checked metadata keys, so tooling can disclose partial coverage.
[[nodiscard]] std::vector<std::string> assert_gguf_matches(const ModelContract& baked, const gguf::FileInfo& file);
inline void validate_file(const ModelContract& baked, const std::string& description_path, const gguf::FileInfo& weights) {
    assert_matches(baked, model_description::load(description_path));
    (void)assert_gguf_matches(baked, weights);
}

} // namespace strix::qwen_description
