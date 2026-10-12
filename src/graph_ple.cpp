#include "strix/graph_ple.hpp"
#include <array>
#include <limits>
#include <stdexcept>

namespace strix {
namespace {
struct Range { std::uintptr_t base; std::uint64_t bytes; };
void check(Range r, std::uint64_t bytes, std::uint64_t alignment) {
    if (!r.base || r.base % alignment || r.bytes < bytes ||
        r.bytes > std::numeric_limits<std::uintptr_t>::max() - r.base)
        throw std::invalid_argument("graph PLE range is invalid");
}
}
void validate_graph_ple_bindings(const PleHashContract& h, std::uint32_t vocabulary,
    std::uint64_t context, const RuntimeTensorBinding& table,
    const GraphDynamicArguments& args, const ExecutionMemoryBinding& status,
    const ExecutionBufferAddress& gathered, const ExecutionMemoryBinding& previous,
    const ExecutionMemoryBinding& next) {
    if (h.ngram_size < 2 || h.ngram_size > PleHashContract::maximum_ngram ||
        !h.heads_per_ngram || h.head_count != (h.ngram_size - 1) * h.heads_per_ngram ||
        h.head_count > PleHashContract::maximum_heads || !h.head_dim || h.head_dim > 256 ||
        h.row_bytes != static_cast<std::uint64_t>(h.head_dim) * 2 || !vocabulary ||
        h.eos_token_id >= vocabulary || !context || !h.minimum_table_rows ||
        h.minimum_table_rows > std::numeric_limits<std::uint64_t>::max() / h.row_bytes)
        throw std::invalid_argument("graph PLE hash geometry is invalid");
    for (std::uint32_t i = 0; i < h.head_count; ++i)
        if (!h.head_vocab_sizes[i] || h.head_offsets[i] >= h.minimum_table_rows ||
            h.head_vocab_sizes[i] > h.minimum_table_rows - h.head_offsets[i])
            throw std::invalid_argument("graph PLE head table range is invalid");
    const auto execution = args.memory.execution_identity;
    if (args != bind_graph_inputs(graph_input_layout(1), args.memory) ||
        (table.storage != RuntimeTensorStorage::resident && table.storage != RuntimeTensorStorage::repacked_cache) ||
        !table.allocation_generation || !gathered.allocation_generation || !gathered.use.generation ||
        !execution || status.execution_identity != execution || previous.execution_identity != execution ||
        next.execution_identity != execution || !status.allocation_generation ||
        !previous.allocation_generation || !next.allocation_generation)
        throw std::invalid_argument("graph PLE resource identity is invalid");
    const std::array<Range, 6> ranges{{{table.address, table.bytes},
        {args.memory.arena_base, args.memory.arena_bytes}, {status.arena_base, status.arena_bytes},
        {gathered.address, gathered.bytes}, {previous.arena_base, previous.arena_bytes},
        {next.arena_base, next.arena_bytes}}};
    check(ranges[0], h.minimum_table_rows * h.row_bytes, 2);
    check(ranges[1], graph_input_layout(1).bytes, 8);
    check(ranges[2], 4, 4);
    check(ranges[3], h.head_count * h.row_bytes, 2);
    check(ranges[4], (h.ngram_size - 1ULL) * 4, 4);
    check(ranges[5], (h.ngram_size - 1ULL) * 4, 4);
    for (std::size_t i = 0; i < ranges.size(); ++i)
        for (std::size_t j = i + 1; j < ranges.size(); ++j)
            if (ranges[i].base < ranges[j].base + ranges[j].bytes &&
                ranges[j].base < ranges[i].base + ranges[i].bytes)
                throw std::invalid_argument("graph PLE views must be disjoint");
}
}
