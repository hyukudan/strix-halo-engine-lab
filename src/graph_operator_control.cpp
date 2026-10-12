#include "strix/graph_operator_control.hpp"

#include <bit>
#include <limits>
#include <stdexcept>

namespace strix {
namespace {
void range(const GraphOperatorResourceView& r) {
    if (!r.address || !r.bytes || !r.allocation_generation || !std::has_single_bit(r.alignment) ||
        r.address % r.alignment || r.bytes > std::numeric_limits<std::uintptr_t>::max() - r.address) {
        throw std::invalid_argument("graph operator resource has invalid extent/alignment/generation");
    }
}
bool overlaps(const GraphOperatorResourceView& a, const GraphOperatorResourceView& b) {
    return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
void word(std::uint64_t& hash, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 255ULL;
        hash *= 1099511628211ULL;
    }
}
} // namespace
GraphOperatorControlPlan prepare_graph_operator_control(const GraphOperatorControlRequest& r) {
    if (!r.execution_identity || !r.context_tokens || r.resources.empty() || r.resources.size() > 64 ||
        r.arguments.memory.execution_identity != r.execution_identity ||
        r.arguments != bind_graph_inputs(graph_input_layout(1), r.arguments.memory) ||
        r.status.execution_identity != r.execution_identity || r.status.arena_bytes != 4) {
        throw std::invalid_argument("graph operator requires exact C1 execution/arguments/status binding");
    }
    const GraphOperatorResourceView arguments{r.arguments.memory.arena_base, r.arguments.memory.arena_bytes,
        r.arguments.memory.allocation_generation, 8};
    const GraphOperatorResourceView status{r.status.arena_base, r.status.arena_bytes,
        r.status.allocation_generation, 4};
    range(arguments); range(status);
    if (overlaps(arguments, status)) throw std::invalid_argument("graph operator arguments and status alias");
    auto hash = std::uint64_t{14695981039346656037ULL};
    for (const auto v : {std::uint64_t{0x47524f5043543031ULL}, r.execution_identity, r.context_tokens,
        static_cast<std::uint64_t>(arguments.address), arguments.bytes, arguments.allocation_generation,
        static_cast<std::uint64_t>(status.address), status.bytes, status.allocation_generation,
        r.arguments.token_offset, r.arguments.position_offset, r.arguments.device_argument_abi_identity,
        static_cast<std::uint64_t>(r.resources.size())}) word(hash, v);
    for (const auto& resource : r.resources) {
        range(resource);
        if (overlaps(resource, arguments) || overlaps(resource, status)) {
            throw std::invalid_argument("graph controls alias an operator resource");
        }
        for (const auto v : {static_cast<std::uint64_t>(resource.address), resource.bytes,
            resource.allocation_generation, static_cast<std::uint64_t>(resource.alignment)}) word(hash, v);
    }
    if (!hash) throw std::logic_error("graph operator control identity is zero");
    return {r, hash};
}
void validate_graph_operator_control(const GraphOperatorControlPlan& plan) {
    if (prepare_graph_operator_control(plan.request).identity != plan.identity) {
        throw std::invalid_argument("graph operator control binding changed");
    }
}
} // namespace strix
