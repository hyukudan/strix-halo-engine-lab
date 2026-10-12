#pragma once

#include "strix/graph_eligibility.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace strix {

inline constexpr std::uint64_t graph_input_abi_v1 = 0x535447494e503031ULL;

// Native little-endian U32 tokens followed by aligned U64 absolute positions.
// Padding is initialized, and all rows are validated before allocating/uploading.
struct GraphInputLayout {
    std::uint32_t rows = 0;
    std::uint64_t position_offset = 0;
    std::uint64_t bytes = 0;
    friend bool operator==(const GraphInputLayout&, const GraphInputLayout&) = default;
};

[[nodiscard]] GraphInputLayout graph_input_layout(std::uint32_t rows);
[[nodiscard]] GraphDynamicArguments bind_graph_inputs(
    const GraphInputLayout& layout, const ExecutionMemoryBinding& memory);
[[nodiscard]] std::vector<std::byte> pack_graph_inputs(
    const GraphInputLayout& layout, std::span<const std::uint32_t> tokens,
    std::span<const std::uint64_t> positions, std::uint64_t vocabulary_size,
    std::uint64_t context_tokens);

// C1 embedding's graph-safe ABI preflight, shared with the native consumer.
void validate_graph_embedding_bindings(
    std::uint32_t width, std::uint32_t ggml_type, std::uint32_t vocabulary_size,
    std::uint64_t context_tokens, std::uint32_t output_element_bytes,
    const RuntimeTensorBinding& weights, const GraphDynamicArguments& arguments,
    const ExecutionBufferAddress& output, const ExecutionMemoryBinding& status);

} // namespace strix
