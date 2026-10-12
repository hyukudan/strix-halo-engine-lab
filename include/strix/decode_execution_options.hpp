#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace strix {
enum class DecodeExecutionMode { eager, graph };
enum class DecodeIntegerHead { none, a8_iu8, a4_iu8, a4_iu4 };
struct DecodeExecutionOptions {
    DecodeExecutionMode mode = DecodeExecutionMode::eager;
    DecodeIntegerHead integer_head = DecodeIntegerHead::none;
    std::uint64_t graph_workspace_bytes = 0;
    std::uint64_t graph_executable_bytes = 0;
    std::uint64_t graph_capture_peak_bytes = 0;
    std::uint64_t graph_budget_bytes = 0;
    std::uint64_t integer_internal_bytes = 0;
    std::uint64_t integer_retained_bytes = 0;
    void validate(bool resident_model, bool prefix_roundtrip) const {
        (void)prefix_roundtrip; // Both native routes restore only their own exact plan.
        constexpr auto maximum = 1ULL << 40U;
        const auto bounded = [](std::uint64_t value) { return value && value <= maximum; };
        if (mode != DecodeExecutionMode::eager && mode != DecodeExecutionMode::graph)
            throw std::invalid_argument("unknown decode execution mode");
        if (integer_head != DecodeIntegerHead::none && integer_head != DecodeIntegerHead::a8_iu8 &&
            integer_head != DecodeIntegerHead::a4_iu8 && integer_head != DecodeIntegerHead::a4_iu4)
            throw std::invalid_argument("unknown integer LM-head mode");
        if (mode == DecodeExecutionMode::graph) {
            if (integer_head != DecodeIntegerHead::none)
                throw std::invalid_argument("graph mode cannot use blocking integer LM head");
            if (!resident_model || !bounded(graph_workspace_bytes) || !bounded(graph_executable_bytes) ||
                !bounded(graph_capture_peak_bytes) || !bounded(graph_budget_bytes))
                throw std::invalid_argument("graph mode requires resident model and four explicit bounded memory ceilings");
            if (graph_workspace_bytes > graph_budget_bytes || graph_executable_bytes > graph_budget_bytes - graph_workspace_bytes ||
                graph_capture_peak_bytes > graph_budget_bytes - graph_workspace_bytes - graph_executable_bytes)
                throw std::invalid_argument("graph declared complete peak exceeds memory budget");
        } else if (graph_workspace_bytes || graph_executable_bytes || graph_capture_peak_bytes || graph_budget_bytes) {
            throw std::invalid_argument("graph memory ceilings require graph execution mode");
        }
        if (integer_head != DecodeIntegerHead::none) {
            if (!resident_model || !bounded(integer_internal_bytes) || !bounded(integer_retained_bytes))
                throw std::invalid_argument("integer LM head requires resident model and explicit internal/retained ceilings");
        } else if (integer_internal_bytes || integer_retained_bytes) {
            throw std::invalid_argument("integer memory ceilings require an integer LM head");
        }
    }
};
inline DecodeExecutionMode parse_decode_execution_mode(std::string_view value) {
    if (value == "eager") return DecodeExecutionMode::eager;
    if (value == "graph") return DecodeExecutionMode::graph;
    throw std::invalid_argument("--execution-mode requires eager or graph");
}
inline DecodeIntegerHead parse_decode_integer_head(std::string_view value) {
    if (value == "a8-iu8") return DecodeIntegerHead::a8_iu8;
    if (value == "a4-iu8") return DecodeIntegerHead::a4_iu8;
    if (value == "a4-iu4") return DecodeIntegerHead::a4_iu4;
    throw std::invalid_argument("--integer-lm-head requires a8-iu8, a4-iu8 or a4-iu4");
}
inline void validate_decode_prefix_roundtrip(std::uint64_t prefix_tokens,
    std::uint64_t context_tokens, std::uint64_t generated_tokens) {
    if (!prefix_tokens || generated_tokens < 2 || prefix_tokens >= context_tokens)
        throw std::invalid_argument("prefix-cache roundtrip requires at least two generated tokens and remaining context");
}
} // namespace strix
