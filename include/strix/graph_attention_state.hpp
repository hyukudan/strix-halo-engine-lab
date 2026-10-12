#pragma once

#include "strix/graph_inputs.hpp"
#include "strix/session_state_memory.hpp"

#include <cstdint>

namespace strix {

enum class AttentionPositionMode { scalar_control, device_input };

// Experimental C1 state stage after Q/K/V projection, before attention scores.
// It is deliberately not a whole attention or whole-model graph declaration.
struct GraphAttentionStateRequest {
    std::uint64_t execution_identity = 0;
    std::uint32_t query_heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t query_head_stride = 0;
    std::uint32_t rotary_dim = 0;
    float epsilon = 0;
    float theta = 0;
    std::uint64_t context_tokens = 0;
    std::uint32_t norm_ggml_type = 0;
    AttentionPositionMode position_mode = AttentionPositionMode::device_input;
    GraphDynamicArguments arguments;
    ExecutionBufferAddress query;
    ExecutionBufferAddress key;
    ExecutionBufferAddress value;
    RuntimeTensorBinding query_norm;
    RuntimeTensorBinding key_norm;
    SessionStateAddress kv_state;
    ExecutionMemoryBinding status;
};

struct GraphAttentionStatePlan {
    GraphAttentionStateRequest request;
    std::uint64_t identity = 0;
    std::uint64_t query_elements = 0;
    std::uint64_t kv_elements = 0;
    std::uint64_t required_state_bytes = 0;
    std::uint64_t retained_view_bytes = 0;
};

inline constexpr std::uint64_t graph_attention_state_arithmetic_v1 = 0x4154545354413031ULL;

[[nodiscard]] GraphAttentionStatePlan prepare_graph_attention_state(
    const GraphAttentionStateRequest& request);
void validate_graph_attention_state_plan(const GraphAttentionStatePlan& plan);

} // namespace strix
