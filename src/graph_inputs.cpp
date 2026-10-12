#include "strix/graph_inputs.hpp"

#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strix {
namespace {
std::uint64_t product(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("graph embedding size overflow");
    }
    return a * b;
}
struct Range { std::uintptr_t address; std::uint64_t bytes; };
void check_range(Range r, std::uint64_t required, std::uint64_t alignment) {
    if (r.address == 0 || r.address % alignment != 0 || r.bytes < required ||
        r.bytes > std::numeric_limits<std::uintptr_t>::max() - r.address) {
        throw std::invalid_argument("graph embedding range is invalid");
    }
}
} // namespace
GraphInputLayout graph_input_layout(std::uint32_t rows) {
    if (rows == 0 || rows > 65536) {
        throw std::invalid_argument("graph input rows exceed bounded ABI");
    }
    if constexpr (std::endian::native != std::endian::little) {
        throw std::invalid_argument("graph input ABI requires little-endian host");
    }
    const auto token_bytes = static_cast<std::uint64_t>(rows) * sizeof(std::uint32_t);
    const auto offset = (token_bytes + 7ULL) & ~7ULL;
    return {rows, offset, offset + static_cast<std::uint64_t>(rows) * sizeof(std::uint64_t)};
}

GraphDynamicArguments bind_graph_inputs(
    const GraphInputLayout& layout, const ExecutionMemoryBinding& memory) {
    if (layout != graph_input_layout(layout.rows) || memory.execution_identity == 0 ||
        memory.allocation_generation == 0 || memory.arena_base == 0 ||
        memory.arena_base % alignof(std::uint64_t) != 0 || memory.arena_bytes != layout.bytes ||
        memory.arena_bytes > std::numeric_limits<std::uintptr_t>::max() - memory.arena_base) {
        throw std::invalid_argument("graph inputs require exact aligned bounded allocation");
    }
    GraphDynamicArguments result;
    result.mode = GraphArgumentMode::device_data;
    result.memory = memory;
    result.position_offset = layout.position_offset;
    result.device_argument_abi_identity = graph_input_abi_v1;
    return result;
}

std::vector<std::byte> pack_graph_inputs(
    const GraphInputLayout& layout, std::span<const std::uint32_t> tokens,
    std::span<const std::uint64_t> positions, std::uint64_t vocabulary_size,
    std::uint64_t context_tokens) {
    if (layout != graph_input_layout(layout.rows) || tokens.size() != layout.rows ||
        positions.size() != layout.rows || vocabulary_size == 0 || context_tokens == 0) {
        throw std::invalid_argument("graph input shape or model limits mismatch");
    }
    for (std::size_t row = 0; row < tokens.size(); ++row) {
        if (tokens[row] >= vocabulary_size || positions[row] >= context_tokens) {
            throw std::out_of_range("graph input token or position exceeds model limits");
        }
        if (row > 0 && positions[row] != positions[row - 1] + 1) {
            throw std::invalid_argument("graph rows require consecutive absolute positions");
        }
    }
    std::vector<std::byte> result(static_cast<std::size_t>(layout.bytes), std::byte{0});
    std::memcpy(result.data(), tokens.data(), tokens.size_bytes());
    std::memcpy(result.data() + layout.position_offset, positions.data(), positions.size_bytes());
    return result;
}

void validate_graph_embedding_bindings(
    std::uint32_t width, std::uint32_t ggml_type, std::uint32_t vocabulary_size,
    std::uint64_t context_tokens, std::uint32_t output_element_bytes,
    const RuntimeTensorBinding& weights, const GraphDynamicArguments& arguments,
    const ExecutionBufferAddress& output, const ExecutionMemoryBinding& status) {
    if (width == 0 || vocabulary_size == 0 || context_tokens == 0 ||
        (output_element_bytes != 2 && output_element_bytes != 4)) {
        throw std::invalid_argument("graph embedding limits/storage are invalid");
    }
    std::uint64_t block_elements = 1, block_bytes = 0, alignment = 2;
    switch (ggml_type) {
    case 0: block_bytes = 4; alignment = 4; break;
    case 1: case 30: block_bytes = 2; break;
    case 8: block_elements = 32; block_bytes = 34; break;
    case 23: block_elements = 256; block_bytes = 136; break;
    default: throw std::invalid_argument("graph embedding weight encoding is unsupported");
    }
    if (width % block_elements != 0 || weights.storage != RuntimeTensorStorage::resident ||
        weights.allocation_generation == 0 || output.allocation_generation == 0 || output.use.generation == 0 ||
        status.execution_identity != arguments.memory.execution_identity || status.allocation_generation == 0) {
        throw std::invalid_argument("graph embedding requires resident exact-generation buffers");
    }
    const auto expected_arguments = bind_graph_inputs(graph_input_layout(1), arguments.memory);
    if (arguments != expected_arguments) throw std::invalid_argument("graph embedding device-input ABI mismatch");
    const Range ranges[] = {{weights.address, weights.bytes},
        {arguments.memory.arena_base, arguments.memory.arena_bytes}, {output.address, output.bytes},
        {status.arena_base, status.arena_bytes}};
    check_range(ranges[0], product(product(width / block_elements, vocabulary_size), block_bytes), alignment);
    check_range(ranges[1], graph_input_layout(1).bytes, 8);
    check_range(ranges[2], product(width, output_element_bytes), output_element_bytes);
    check_range(ranges[3], sizeof(std::uint32_t), alignof(std::uint32_t));
    if (status.arena_bytes != sizeof(std::uint32_t)) throw std::invalid_argument("graph embedding status must be one U32");
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = i + 1; j < 4; ++j) {
            if (ranges[i].address < ranges[j].address + ranges[j].bytes &&
                ranges[j].address < ranges[i].address + ranges[i].bytes) {
                throw std::invalid_argument("graph embedding inputs/output/status must not alias");
            }
        }
    }
}
} // namespace strix
