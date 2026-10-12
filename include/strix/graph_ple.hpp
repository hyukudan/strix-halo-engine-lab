#pragma once
#include "strix/graph_inputs.hpp"
#include "strix/ple_lookup.hpp"

namespace strix {
void validate_graph_ple_bindings(const PleHashContract&, std::uint32_t vocabulary_size,
    std::uint64_t context_tokens, const RuntimeTensorBinding& table,
    const GraphDynamicArguments&, const ExecutionMemoryBinding& status,
    const ExecutionBufferAddress& gathered, const ExecutionMemoryBinding& history_input,
    const ExecutionMemoryBinding& history_output);
}
