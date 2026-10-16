#include "strix/prepared_arena_views.hpp"
#include "strix/compiled_execution.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace strix {
namespace {
bool use_less(const ExecutionBufferUse& a, const ExecutionBufferUse& b) noexcept {
    return std::tie(a.buffer.buffer_class, a.buffer.slot, a.generation) <
           std::tie(b.buffer.buffer_class, b.buffer.slot, b.generation);
}
} // namespace

struct PreparedArenaViews::Impl {
    const CompiledSerialExecution& compiled;
    const std::uint64_t execution_identity;
    std::vector<std::uint32_t> row_start;
    std::vector<std::uint32_t> allocation_index;

    explicit Impl(const CompiledSerialExecution& source)
        : compiled(source), execution_identity(source.execution_identity) {
        constexpr auto limit = std::numeric_limits<std::uint32_t>::max();
        const auto& steps = compiled.execution.steps;
        const auto& io = compiled.io.steps;
        const auto& arena = compiled.arena;
        if (!execution_identity || !arena.total_bytes || steps.size() != io.size() ||
            steps.size() >= limit || arena.allocations.size() > limit)
            throw std::invalid_argument("prepared arena index has invalid topology bounds");
        std::uint64_t count = 0;
        for (std::size_t step = 0; step < steps.size(); ++step) {
            if (steps[step].id != step || io[step].step_id != step || io[step].op != steps[step].op)
                throw std::invalid_argument("prepared arena index requires dense matching IO steps");
            for (const auto values : std::array{std::span{io[step].reads},
                     std::span{io[step].writes}, std::span{io[step].temporaries}}) {
                const auto size = static_cast<std::uint64_t>(values.size());
                if (size > limit - count)
                    throw std::invalid_argument("prepared arena index exceeds uint32 IO capacity");
                count += size;
            }
        }
        row_start.reserve(steps.size() + 1U);
        allocation_index.reserve(static_cast<std::size_t>(count));
        std::vector<std::uint32_t> sorted(arena.allocations.size());
        std::iota(sorted.begin(), sorted.end(), std::uint32_t{0});
        std::sort(sorted.begin(), sorted.end(), [&](auto a, auto b) {
            return use_less(arena.allocations[a].use, arena.allocations[b].use);
        });
        for (std::size_t i = 0; i < sorted.size(); ++i) {
            const auto& allocation = arena.allocations[sorted[i]];
            if (i && arena.allocations[sorted[i - 1U]].use == allocation.use)
                throw std::invalid_argument("prepared arena index rejects duplicate allocation uses");
            if (allocation.offset > arena.total_bytes ||
                allocation.bytes > arena.total_bytes - allocation.offset)
                throw std::invalid_argument("prepared arena allocation exceeds compiled arena");
        }
        row_start.push_back(0);
        for (const auto& step : io) {
            const auto row_begin = allocation_index.size();
            for (const auto values : std::array{std::span{step.reads},
                     std::span{step.writes}, std::span{step.temporaries}}) {
                for (const auto use : values) {
                    const auto found = std::lower_bound(sorted.begin(), sorted.end(), use,
                        [&](auto index, const auto& requested) {
                            return use_less(arena.allocations[index].use, requested);
                        });
                    if (found == sorted.end() || arena.allocations[*found].use != use)
                        throw std::invalid_argument("prepared arena index is missing an IO generation");
                    allocation_index.push_back(*found);
                }
            }
            if (allocation_index.size() - row_begin !=
                step.reads.size() + step.writes.size() + step.temporaries.size())
                throw std::logic_error("prepared arena index row width differs");
            row_start.push_back(static_cast<std::uint32_t>(allocation_index.size()));
        }
        if (allocation_index.size() != count || row_start.size() != steps.size() + 1U)
            throw std::logic_error("prepared arena index construction is incomplete");
    }

    ExecutionBufferAddress resolve(const ExecutionDispatchContext& context,
        PreparedBufferKind kind, std::size_t index, const ExecutionMemoryBinding& memory) const {
        // Preserve binding-fault order and check the current allocation on every
        // call. Neither physical addresses nor binding generations are cached.
        validate_execution_memory_binding(execution_identity, compiled.arena, memory);
        if (!context.execution_identity || context.execution_identity != execution_identity)
            throw std::invalid_argument("prepared arena execution identity differs");
        if (context.arena != &compiled.arena)
            throw std::invalid_argument("prepared arena handle differs from its immutable owner");
        if (context.step_id >= compiled.execution.steps.size())
            throw std::out_of_range("prepared arena step is out of range");
        const auto& step = compiled.execution.steps[context.step_id];
        const auto& io = compiled.io.steps[context.step_id];
        if (context.op != step.op || context.layer != step.layer)
            throw std::invalid_argument("prepared arena dispatch step differs");
        if (context.reads.size() != io.reads.size() || context.writes.size() != io.writes.size() ||
            context.temporaries.size() != io.temporaries.size())
            throw std::invalid_argument("prepared arena dispatch IO sizes differ");
        std::span<const ExecutionBufferUse> requested;
        std::size_t base = 0;
        switch (kind) {
        case PreparedBufferKind::read: requested = context.reads; break;
        case PreparedBufferKind::write:
            requested = context.writes; base = io.reads.size(); break;
        case PreparedBufferKind::temporary:
            requested = context.temporaries; base = io.reads.size() + io.writes.size(); break;
        default: throw std::invalid_argument("prepared arena buffer kind is invalid");
        }
        if (index >= requested.size())
            throw std::out_of_range("prepared arena buffer index is out of range");
        const auto position = static_cast<std::size_t>(row_start[context.step_id]) + base + index;
        if (position >= row_start[static_cast<std::size_t>(context.step_id) + 1U] ||
            position >= allocation_index.size() || allocation_index[position] >= compiled.arena.allocations.size())
            throw std::logic_error("prepared arena allocation index is invalid");
        const auto& allocation = compiled.arena.allocations[allocation_index[position]];
        const auto use = requested[index];
        // Unique full uses make this the exact first match of the legacy scan.
        // Copied spans are allowed; address equality of the span is unnecessary.
        if (allocation.use != use)
            throw std::invalid_argument("prepared arena buffer generation differs");
        if (allocation.offset > static_cast<std::uint64_t>(
                std::numeric_limits<std::uintptr_t>::max() - memory.arena_base))
            throw std::overflow_error("execution buffer address overflows uintptr_t");
        if (allocation.offset > memory.arena_bytes ||
            allocation.bytes > memory.arena_bytes - allocation.offset)
            throw std::invalid_argument("execution buffer exceeds bound arena allocation");
        return {use, memory.arena_base + static_cast<std::uintptr_t>(allocation.offset),
                allocation.bytes, memory.allocation_generation};
    }
};

PreparedArenaViews::PreparedArenaViews(const CompiledSerialExecution& immutable_owned)
    : impl_(std::make_unique<const Impl>(immutable_owned)) {}
PreparedArenaViews::~PreparedArenaViews() = default;
ExecutionBufferAddress PreparedArenaViews::resolve(const ExecutionDispatchContext& context,
    PreparedBufferKind kind, std::size_t index, const ExecutionMemoryBinding& memory) const {
    return impl_->resolve(context, kind, index, memory);
}
} // namespace strix
