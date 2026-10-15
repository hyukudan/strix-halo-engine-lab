#include "strix/moe_fork_join_arena.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template <typename Function> bool rejects(Function&& function) {
    try { function(); } catch (const std::exception&) { return true; }
    return false;
}
// Uses the real serial DAG/IO/buffer/arena builders, with a synthetic execution
// identity. No catalog, GPU launcher, model arithmetic or qualification implied.
strix::CompiledSerialExecution source_plan(bool router = false, bool a8 = false) {
    const auto model = strix::ModelContract::qwen38_flash_next();
    strix::CompiledSerialExecution source;
    source.execution = strix::build_serial_decode_execution_plan(model);
    source.io = strix::build_serial_execution_io_plan(source.execution, router);
    strix::ExecutionBufferRequest buffers;
    buffers.parallel_moe_router = router; buffers.q4k_a8g32_gate_up = a8;
    buffers.grouped_exact_q51_down = a8;
    source.arena = strix::build_serial_execution_arena_plan(model, buffers, source.execution, source.io);
    source.execution_identity = 0x5330344153594e31ULL;
    return source;
}
bool separated(const strix::ExecutionArenaAllocation& a, const strix::ExecutionArenaAllocation& b) {
    return a.offset + a.bytes <= b.offset || b.offset + b.bytes <= a.offset;
}
void layout_contract() {
    for (const bool extra_workspace : {false, true}) {
        const auto source = source_plan(extra_workspace, extra_workspace);
        const auto plan = strix::prepare_moe_fork_join_arena(source, std::numeric_limits<std::uint64_t>::max());
        const auto exact_budget = strix::prepare_moe_fork_join_arena(source, plan.arena().total_bytes);
        require(plan.layout_identity() == exact_budget.layout_identity(), "admission budget contaminated layout identity");
        require(plan.layout_identity() != source.execution_identity && plan.source_execution_identity() == source.execution_identity,
                "fork/join arena did not get its own source-bound identity");
        require(plan.layers().size() == 48U && plan.serial_arena_bytes() == source.arena.total_bytes &&
                plan.shared_region_offset() == source.arena.total_bytes && plan.additional_bytes() > 0U &&
                plan.arena().total_bytes == source.arena.total_bytes + plan.shared_region_bytes(),
                "fork/join arena does not expose complete incremental memory charge");
        for (const auto& layer : plan.layers()) {
            const auto* output = strix::find_arena_allocation(plan.arena(), layer.shared_output);
            const auto* temporary = strix::find_arena_allocation(plan.arena(), layer.shared_temporary);
            require(output && temporary && output->offset >= source.arena.total_bytes &&
                    temporary->offset >= source.arena.total_bytes && separated(*output, *temporary) &&
                    output->first_step == layer.route_step && output->last_step == layer.combine_step &&
                    temporary->first_step == layer.route_step && temporary->last_step == layer.combine_step,
                    "shared allocations are not separately live through the join");
        }
        for (const auto& old : source.arena.allocations) {
            const auto* current = strix::find_arena_allocation(plan.arena(), old.use);
            require(current && current->bytes == old.bytes, "fork/join lost a buffer extent");
            const auto relocated = std::any_of(plan.layers().begin(), plan.layers().end(), [&](const auto& layer) {
                return old.use == layer.shared_output || old.use == layer.shared_temporary;
            });
            if (!relocated) require(current->offset == old.offset && current->first_step == old.first_step &&
                current->last_step == old.last_step, "fork/join changed a serial base allocation");
        }
        require(rejects([&] { static_cast<void>(strix::prepare_moe_fork_join_arena(source, plan.arena().total_bytes - 1U)); }) &&
                rejects([&] { static_cast<void>(strix::prepare_moe_fork_join_arena(source, 0U)); }),
                "fork/join did not reject the complete memory charge before admission");
    }
}
struct HostArena {
    std::vector<std::byte> bytes;
    strix::ExecutionMemoryBinding memory;
    explicit HostArena(const strix::MoeForkJoinArenaPlan& plan) {
        constexpr std::size_t guard_bytes = 8;
        const auto total = static_cast<std::size_t>(plan.arena().total_bytes);
        const auto alignment = static_cast<std::size_t>(plan.arena().alignment);
        bytes.assign(total + alignment + 2U * guard_bytes, std::byte{0xD3});
        void* pointer = bytes.data() + guard_bytes;
        auto remaining = bytes.size() - 2U * guard_bytes;
        require(std::align(alignment, total, pointer, remaining) != nullptr, "host arena cannot align retained allocation");
        memory = {plan.layout_identity(), reinterpret_cast<std::uintptr_t>(pointer), total, 7U};
        plan.validate_binding(memory);
    }
    void guards() const {
        const auto* pointer = reinterpret_cast<const std::byte*>(memory.arena_base);
        for (std::size_t index = 0; index < 8U; ++index)
            require(pointer[-static_cast<std::ptrdiff_t>(index) - 1] == std::byte{0xD3} &&
                    pointer[memory.arena_bytes + index] == std::byte{0xD3}, "fork/join test crossed actual allocation guards");
    }
};
void poison_interleavings() {
    const auto source = source_plan(true, true);
    const auto plan = strix::prepare_moe_fork_join_arena(source, std::numeric_limits<std::uint64_t>::max());
    HostArena owner(plan);
    const auto paint = [&](strix::ExecutionBufferUse use, std::byte pattern) {
        const auto address = plan.resolve(owner.memory, use);
        std::fill_n(reinterpret_cast<std::byte*>(address.address), static_cast<std::size_t>(address.bytes), pattern);
    };
    const auto check = [&](strix::ExecutionBufferUse use, std::byte pattern) {
        const auto address = plan.resolve(owner.memory, use);
        const auto* first = reinterpret_cast<const std::byte*>(address.address);
        require(std::all_of(first, first + static_cast<std::size_t>(address.bytes),
                            [&](auto value) { return value == pattern; }), "legal branch interleaving corrupted a live buffer");
    };
    // Atomic host stages model shared begin/end and routed route/experts. Each
    // linear extension obeys shared-begin < shared-end and route < experts.
    // These are memory-contract simulations, not GPU overlap/completion proof.
    std::array<unsigned, 4> order{0U, 1U, 2U, 3U};
    unsigned legal = 0;
    do {
        const auto at = [&](unsigned stage) { return std::find(order.begin(), order.end(), stage); };
        if (at(0U) > at(1U) || at(2U) > at(3U)) continue;
        ++legal;
        for (const auto& layer : plan.layers()) {
            paint(layer.input, std::byte{0x11});
            for (const auto stage : order) {
                if (stage == 0U) {
                    check(layer.input, std::byte{0x11});
                    paint(layer.shared_temporary, std::byte{0xA1}); paint(layer.shared_output, std::byte{0xA2});
                } else if (stage == 1U) {
                    check(layer.input, std::byte{0x11}); check(layer.shared_temporary, std::byte{0xA1});
                    check(layer.shared_output, std::byte{0xA2}); paint(layer.shared_output, std::byte{0x41});
                } else if (stage == 2U) {
                    check(layer.input, std::byte{0x11});
                    for (const auto use : source.io.steps[layer.route_step].writes) paint(use, std::byte{0x21});
                    for (const auto use : source.io.steps[layer.route_step].temporaries) paint(use, std::byte{0x21});
                } else {
                    check(layer.input, std::byte{0x11});
                    for (const auto use : source.io.steps[layer.route_step].writes) check(use, std::byte{0x21});
                    paint(layer.routed_temporary, std::byte{0x31}); paint(layer.routed_output, std::byte{0x32});
                }
            }
            // Join barrier precedes combine and cross-layer reuse. Both branch
            // outputs remain available, including a deliberately late shared.
            check(layer.shared_output, std::byte{0x41}); check(layer.routed_output, std::byte{0x32});
            for (const auto use : source.io.steps[layer.combine_step].writes) paint(use, std::byte{0x51});
            owner.guards();
        }
    } while (std::next_permutation(order.begin(), order.end()));
    require(legal == 6U, "memory fixture did not cover every legal four-stage interleaving");
}
void rejected_sources_and_bindings() {
    const auto source = source_plan(true, true);
    const auto plan = strix::prepare_moe_fork_join_arena(source, std::numeric_limits<std::uint64_t>::max());
    const auto layer = plan.layers().front();
    const auto rejected = [&](const auto& invalid) {
        require(rejects([&] { static_cast<void>(strix::prepare_moe_fork_join_arena(
            invalid, std::numeric_limits<std::uint64_t>::max())); }), "unsafe fork/join source was admitted");
    };
    auto invalid = source; invalid.execution_identity = 0U; rejected(invalid);
    invalid = source; invalid.device_graph_inputs = true; rejected(invalid);
    invalid = source; invalid.logical_rows = 2U; rejected(invalid);
    invalid = source; invalid.execution.steps[layer.shared_step].dependencies.push_back(layer.route_step); rejected(invalid);
    invalid = source; invalid.execution.steps[layer.combine_step].dependencies.pop_back(); rejected(invalid);
    invalid = source; invalid.execution.steps[layer.shared_step].writes_state = strix::PersistentState::attention_kv; rejected(invalid);
    invalid = source; ++*invalid.execution.steps[layer.shared_step].layer; rejected(invalid);
    invalid = source; invalid.io.steps[layer.combine_step + 1U].reads.push_back(layer.shared_output); rejected(invalid);
    invalid = source; invalid.arena.allocations.front().last_step = std::numeric_limits<std::uint32_t>::max(); rejected(invalid);
    invalid = source; invalid.arena.allocations.front().offset = invalid.arena.total_bytes; rejected(invalid);
    invalid = source; invalid.arena.allocations.push_back(invalid.arena.allocations.front()); rejected(invalid);
    invalid = source;
    auto temporary = std::find_if(invalid.arena.allocations.begin(), invalid.arena.allocations.end(),
        [&](const auto& allocation) { return allocation.use == layer.shared_temporary; });
    temporary->offset = strix::find_arena_allocation(source.arena, layer.input)->offset; rejected(invalid);
    invalid = source; invalid.arena.alignment = 0U; rejected(invalid);
    invalid = source; invalid.io.steps[layer.shared_step].reads.front().buffer.slot = 8U; rejected(invalid);
    invalid = source;
    invalid.arena.total_bytes = std::numeric_limits<std::uint64_t>::max() -
        std::numeric_limits<std::uint64_t>::max() % invalid.arena.alignment;
    rejected(invalid);
    HostArena owner(plan);
    const auto bad_binding = [&](const auto& binding) {
        require(rejects([&] { plan.validate_binding(binding); }) &&
                rejects([&] { static_cast<void>(plan.resolve(binding, layer.shared_output)); }),
                "fork/join admitted a stale, short or unaligned binding");
    };
    auto binding = owner.memory; binding.execution_identity = source.execution_identity; bad_binding(binding);
    binding = owner.memory; ++binding.arena_base; bad_binding(binding);
    binding = owner.memory; binding.arena_bytes = plan.arena().total_bytes - 1U; bad_binding(binding);
    binding = owner.memory; binding.allocation_generation = 0U; bad_binding(binding);
    binding = owner.memory; binding.arena_bytes = std::numeric_limits<std::uintptr_t>::max(); bad_binding(binding);
    binding = owner.memory; binding.arena_base = std::numeric_limits<std::uintptr_t>::max() -
        std::numeric_limits<std::uintptr_t>::max() % plan.arena().alignment; bad_binding(binding);
    auto missing = layer.shared_output; missing.generation = std::numeric_limits<std::uint64_t>::max();
    require(rejects([&] { static_cast<void>(plan.resolve(owner.memory, missing)); }), "fork/join resolved an absent generation");
}
void source_snapshot_and_rebind() {
    const auto source = source_plan(true, true);
    const auto plan = strix::prepare_moe_fork_join_arena(source, std::numeric_limits<std::uint64_t>::max());
    auto rebound = plan.rebind_execution(source);
    plan.validate_execution(rebound);
    require(rebound.execution_identity == plan.layout_identity() &&
            rebound.kernel_plan_identity == source.kernel_plan_identity && !rebound.whole_plan_graph_safe &&
            !rebound.device_graph_inputs && rebound.arena.total_bytes == plan.arena().total_bytes &&
            source.arena.total_bytes == plan.serial_arena_bytes(),
            "fork/join rebind did not preserve numerical identity and replace physical/calendar identity");
    require(rejects([&] { plan.validate_execution(source); }), "fork/join accepted its old serial execution unchanged");
    const auto changed_source = [&](const auto& changed) {
        require(changed.execution_identity == source.execution_identity &&
                rejects([&] { static_cast<void>(plan.rebind_execution(changed)); }),
                "fork/join trusted a reused execution ID over changed source structure");
    };
    auto changed = source; ++changed.vocab_size; changed_source(changed);
    changed = source; ++changed.kernel_plan_identity; changed_source(changed);
    changed = source; changed.execution.steps.front().graph_safe_candidate = !changed.execution.steps.front().graph_safe_candidate;
    changed_source(changed);
    changed = source; changed.io.steps.back().reads.clear(); changed_source(changed);
    changed = source; ++changed.bindings.bound_steps; changed_source(changed);
    changed = source; changed.bindings.bindings.push_back({0U, strix::SerialOp::token_embedding,
        strix::kernels::OperatorFamily::embedding_lookup, "changed.fixture", false}); changed_source(changed);
    changed = source; ++changed.arena.allocations.front().bytes; changed_source(changed);
    changed = source; ++changed.arena.naive_nonaliased_bytes; changed_source(changed);
    ++rebound.arena.allocations.front().bytes;
    require(rejects([&] { plan.validate_execution(rebound); }), "fork/join admitted a mutated rebound arena");
    changed = source; ++changed.kernel_plan_identity;
    const auto changed_plan = strix::prepare_moe_fork_join_arena(changed, std::numeric_limits<std::uint64_t>::max());
    require(changed_plan.layout_identity() != plan.layout_identity(), "source snapshot fingerprint omitted numerical plan identity");
}
} // namespace
int main() try {
    layout_contract(); poison_interleavings(); rejected_sources_and_bindings(); source_snapshot_and_rebind();
    std::cout << "MoE fork/join arena host contracts passed; no native execution\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
