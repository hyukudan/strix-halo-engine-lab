#include "strix/graph_attention_state.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strix {
namespace {
std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("attention state byte overflow");
    }
    return a * b;
}
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("attention state accounting overflow");
    }
    return a + b;
}
struct Range { std::uintptr_t address; std::uint64_t bytes; };
void range(Range r, std::uint64_t required, std::uint64_t alignment) {
    if (r.address == 0 || r.address % alignment != 0 || r.bytes < required ||
        r.bytes > std::numeric_limits<std::uintptr_t>::max() - r.address) {
        throw std::invalid_argument("attention state range is undersized, unaligned or overflowing");
    }
}
bool overlap(Range a, Range b) {
    return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
void word(std::uint64_t& hash, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 255ULL;
        hash *= 1099511628211ULL;
    }
}
void buffer_hash(std::uint64_t& hash, const ExecutionBufferAddress& b) {
    for (auto v : {static_cast<std::uint64_t>(b.use.buffer.buffer_class),
        static_cast<std::uint64_t>(b.use.buffer.slot), b.use.generation,
        static_cast<std::uint64_t>(b.address), b.bytes, b.allocation_generation}) word(hash, v);
}
void norm_hash(std::uint64_t& hash, const RuntimeTensorBinding& b) {
    for (auto v : {static_cast<std::uint64_t>(b.tensor_index), static_cast<std::uint64_t>(b.storage),
        static_cast<std::uint64_t>(b.address), b.bytes, b.allocation_generation,
        static_cast<std::uint64_t>(b.owner_node)}) word(hash, v);
}
} // namespace

GraphAttentionStatePlan prepare_graph_attention_state(const GraphAttentionStateRequest& r) {
    if (r.execution_identity == 0 || r.query_heads == 0 || r.query_heads > 4096 ||
        r.kv_heads == 0 || r.kv_heads > r.query_heads || r.query_heads % r.kv_heads != 0 ||
        r.head_dim < 2 || r.head_dim > 1024 || !std::has_single_bit(r.head_dim) ||
        (r.query_head_stride != r.head_dim && r.query_head_stride != 2U * r.head_dim) ||
        r.rotary_dim == 0 || r.rotary_dim > r.head_dim ||
        r.rotary_dim % 2 != 0 || !std::isfinite(r.epsilon) || r.epsilon <= 0 ||
        !std::isfinite(r.theta) || r.theta <= 0 || r.context_tokens == 0 ||
        (r.position_mode != AttentionPositionMode::scalar_control && r.position_mode != AttentionPositionMode::device_input)) {
        throw std::invalid_argument("attention state geometry, position mode or scalar arithmetic is invalid");
    }
    if (r.arguments.memory.execution_identity != r.execution_identity ||
        r.arguments != bind_graph_inputs(graph_input_layout(1), r.arguments.memory) ||
        r.status.execution_identity != r.execution_identity || r.status.allocation_generation == 0 || r.status.arena_bytes != 4) {
        throw std::invalid_argument("attention state requires exact C1 argument and status binding");
    }
    std::uint64_t norm_element_bytes = 0;
    switch (r.norm_ggml_type) {
    case 0: norm_element_bytes = 4; break;
    case 1: case 30: norm_element_bytes = 2; break;
    default: throw std::invalid_argument("attention state normalization encoding unsupported");
    }
    for (const auto* b : {&r.query, &r.key, &r.value}) {
        if (b->allocation_generation == 0 || b->use.generation == 0) {
            throw std::invalid_argument("attention state requires exact workspace generations");
        }
    }
    for (const auto* b : {&r.query_norm, &r.key_norm}) {
        if (b->allocation_generation == 0 || b->storage != RuntimeTensorStorage::resident) {
            throw std::invalid_argument("attention state requires resident normalization weights");
        }
    }
    if (r.kv_state.component != SessionStateComponent::attention_kv ||
        r.kv_state.state_flag != PersistentState::attention_kv || r.kv_state.storage != SessionStateStorage::device ||
        r.kv_state.provisional || r.kv_state.allocation_generation == 0) {
        throw std::invalid_argument("attention state requires an owned serial device KV layer");
    }
    GraphAttentionStatePlan p;
    p.request = r;
    p.query_elements = add(mul(r.query_heads - 1ULL, r.query_head_stride), r.head_dim);
    p.kv_elements = mul(r.kv_heads, r.head_dim);
    p.required_state_bytes = mul(mul(r.context_tokens, p.kv_elements), 4);
    const std::array<Range, 8> ranges{{{r.query.address, r.query.bytes}, {r.key.address, r.key.bytes},
        {r.value.address, r.value.bytes}, {r.query_norm.address, r.query_norm.bytes},
        {r.key_norm.address, r.key_norm.bytes}, {r.kv_state.address, r.kv_state.bytes},
        {r.arguments.memory.arena_base, r.arguments.memory.arena_bytes}, {r.status.arena_base, r.status.arena_bytes}}};
    range(ranges[0], mul(p.query_elements, 4), 4);
    range(ranges[1], mul(p.kv_elements, 4), 4); range(ranges[2], mul(p.kv_elements, 4), 4);
    range(ranges[3], mul(r.head_dim, norm_element_bytes), norm_element_bytes);
    range(ranges[4], mul(r.head_dim, norm_element_bytes), norm_element_bytes);
    range(ranges[5], p.required_state_bytes, 2);
    range(ranges[6], 16, 8); range(ranges[7], 4, 4);
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        const bool shared_norm = i == 4 && ranges[3].address == ranges[4].address &&
            ranges[3].bytes == ranges[4].bytes && r.query_norm.allocation_generation == r.key_norm.allocation_generation;
        if (!shared_norm) p.retained_view_bytes = add(p.retained_view_bytes, ranges[i].bytes);
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            // The two read-only norm views may share exactly one tensor.
            if (i == 3 && j == 4 && ranges[i].address == ranges[j].address &&
                ranges[i].bytes == ranges[j].bytes && r.query_norm.allocation_generation == r.key_norm.allocation_generation) continue;
            if (overlap(ranges[i], ranges[j])) throw std::invalid_argument("attention state resource views must not alias");
        }
    }
    auto hash = std::uint64_t{14695981039346656037ULL};
    for (auto v : {graph_attention_state_arithmetic_v1, r.execution_identity,
        static_cast<std::uint64_t>(r.query_heads), static_cast<std::uint64_t>(r.kv_heads),
        static_cast<std::uint64_t>(r.head_dim), static_cast<std::uint64_t>(r.query_head_stride),
        static_cast<std::uint64_t>(r.rotary_dim), static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(r.epsilon)),
        static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(r.theta)), r.context_tokens,
        static_cast<std::uint64_t>(r.norm_ggml_type), static_cast<std::uint64_t>(r.position_mode),
        r.arguments.memory.execution_identity, static_cast<std::uint64_t>(r.arguments.memory.arena_base),
        r.arguments.memory.arena_bytes, r.arguments.memory.allocation_generation,
        r.arguments.token_offset, r.arguments.position_offset, r.arguments.device_argument_abi_identity,
        static_cast<std::uint64_t>(r.kv_state.address), r.kv_state.bytes, r.kv_state.allocation_generation,
        static_cast<std::uint64_t>(r.kv_state.component), static_cast<std::uint64_t>(r.kv_state.state_flag),
        static_cast<std::uint64_t>(r.kv_state.storage), static_cast<std::uint64_t>(r.kv_state.provisional),
        r.status.execution_identity, static_cast<std::uint64_t>(r.status.arena_base), r.status.arena_bytes,
        r.status.allocation_generation, p.query_elements, p.kv_elements, p.required_state_bytes, p.retained_view_bytes}) word(hash, v);
    buffer_hash(hash, r.query); buffer_hash(hash, r.key); buffer_hash(hash, r.value);
    norm_hash(hash, r.query_norm); norm_hash(hash, r.key_norm);
    if (hash == 0) throw std::logic_error("attention state identity is zero");
    p.identity = hash;
    return p;
}
void validate_graph_attention_state_plan(const GraphAttentionStatePlan& p) {
    const auto expected = prepare_graph_attention_state(p.request);
    if (p.identity != expected.identity || p.query_elements != expected.query_elements ||
        p.kv_elements != expected.kv_elements || p.required_state_bytes != expected.required_state_bytes ||
        p.retained_view_bytes != expected.retained_view_bytes) {
        throw std::invalid_argument("attention state plan metadata or resource bindings changed");
    }
}
} // namespace strix
