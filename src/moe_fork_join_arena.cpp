#include "strix/moe_fork_join_arena.hpp"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace strix {
namespace {
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::overflow_error("fork/join arena size overflow");
    return a + b;
}
std::uint64_t aligned(std::uint64_t bytes, std::uint64_t alignment) {
    if (!alignment) throw std::invalid_argument("fork/join arena alignment is zero");
    const auto remainder = bytes % alignment;
    return remainder ? add(bytes, alignment - remainder) : bytes;
}
bool overlaps(const ExecutionArenaAllocation& a, const ExecutionArenaAllocation& b) {
    return a.offset < add(b.offset, b.bytes) && b.offset < add(a.offset, a.bytes);
}
bool live_together(const ExecutionArenaAllocation& a, const ExecutionArenaAllocation& b) {
    return a.first_step <= b.last_step && b.first_step <= a.last_step;
}
bool same_set(std::span<const std::uint32_t> actual, std::initializer_list<std::uint32_t> expected) {
    return actual.size() == expected.size() &&
        std::all_of(expected.begin(), expected.end(), [&](auto value) {
            return std::count(actual.begin(), actual.end(), value) == 1;
        });
}
bool same_uses(std::span<const ExecutionBufferUse> actual, std::initializer_list<ExecutionBufferUse> expected) {
    return actual.size() == expected.size() &&
        std::all_of(expected.begin(), expected.end(), [&](auto value) {
            return std::count(actual.begin(), actual.end(), value) == 1;
        });
}
bool buffer_is(ExecutionBufferUse use, ExecutionBufferClass buffer_class, std::uint32_t slot) {
    return use.buffer.buffer_class == buffer_class && use.buffer.slot == slot;
}
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
void hash_word(std::uint64_t& hash, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 255ULL;
        hash *= 1099511628211ULL;
    }
}
void hash_use(std::uint64_t& hash, ExecutionBufferUse use) {
    hash_word(hash, static_cast<std::uint64_t>(use.buffer.buffer_class));
    hash_word(hash, use.buffer.slot); hash_word(hash, use.generation);
}
bool same_compiled(const CompiledSerialExecution& a, const CompiledSerialExecution& b) {
    const auto scalars = [](const auto& value) {
        return std::tie(value.execution_identity, value.kernel_plan_identity, value.dynamic_steps,
            value.non_graph_safe_steps, value.whole_plan_graph_safe, value.device_graph_inputs,
            value.logical_rows, value.request_concurrency, value.vocab_size, value.max_position_embeddings,
            value.execution.linear_attention_steps, value.execution.indexed_attention_steps,
            value.execution.ple_steps, value.execution.moe_route_steps, value.execution.moe_expert_steps,
            value.bindings.bound_steps, value.bindings.missing_steps, value.bindings.host_or_runtime_steps,
            value.io.expanded_hidden_slots, value.io.hidden_slots, value.io.injection_slots, value.io.route_index_slots,
            value.io.route_weight_slots, value.io.logits_slots, value.io.mlp_intermediate_slots,
            value.io.gdn_workspace_slots, value.io.ple_workspace_slots, value.io.indexed_attention_workspace_slots,
            value.io.hyper_workspace_slots, value.io.router_workspace_slots,
            value.arena.alignment, value.arena.total_bytes, value.arena.naive_nonaliased_bytes);
    };
    return scalars(a) == scalars(b) &&
        std::equal(a.execution.steps.begin(), a.execution.steps.end(), b.execution.steps.begin(), b.execution.steps.end(),
            [](const auto& x, const auto& y) { return std::tie(x.id, x.op, x.layer, x.dependencies, x.reads_state,
                x.writes_state, x.graph_safe_candidate, x.dynamic_shape) == std::tie(y.id, y.op, y.layer, y.dependencies,
                y.reads_state, y.writes_state, y.graph_safe_candidate, y.dynamic_shape); }) &&
        std::equal(a.bindings.bindings.begin(), a.bindings.bindings.end(), b.bindings.bindings.begin(), b.bindings.bindings.end(),
            [](const auto& x, const auto& y) { return std::tie(x.step_id, x.op, x.family, x.kernel_id, x.host_or_runtime_step) ==
                std::tie(y.step_id, y.op, y.family, y.kernel_id, y.host_or_runtime_step); }) &&
        std::equal(a.io.steps.begin(), a.io.steps.end(), b.io.steps.begin(), b.io.steps.end(),
            [](const auto& x, const auto& y) { return std::tie(x.step_id, x.op, x.reads, x.writes, x.temporaries) ==
                std::tie(y.step_id, y.op, y.reads, y.writes, y.temporaries); }) &&
        std::equal(a.arena.allocations.begin(), a.arena.allocations.end(), b.arena.allocations.begin(), b.arena.allocations.end(),
            [](const auto& x, const auto& y) { return std::tie(x.use, x.first_step, x.last_step, x.offset, x.bytes) ==
                std::tie(y.use, y.first_step, y.last_step, y.offset, y.bytes); });
}

void hash_source(std::uint64_t& hash, const CompiledSerialExecution& source) {
    const auto word = [&](auto value) { hash_word(hash, static_cast<std::uint64_t>(value)); };
    word(source.execution_identity); word(source.kernel_plan_identity); word(source.dynamic_steps);
    word(source.non_graph_safe_steps); word(source.whole_plan_graph_safe); word(source.device_graph_inputs);
    word(source.logical_rows); word(source.request_concurrency); word(source.vocab_size); word(source.max_position_embeddings);
    word(source.execution.linear_attention_steps); word(source.execution.indexed_attention_steps);
    word(source.execution.ple_steps); word(source.execution.moe_route_steps); word(source.execution.moe_expert_steps);
    word(source.execution.steps.size());
    for (const auto& step : source.execution.steps) {
        word(step.id); word(step.op); word(step.layer.has_value()); if (step.layer) word(*step.layer);
        word(step.dependencies.size()); for (const auto dependency : step.dependencies) word(dependency);
        word(step.reads_state); word(step.writes_state); word(step.graph_safe_candidate); word(step.dynamic_shape);
    }
    word(source.bindings.bound_steps); word(source.bindings.missing_steps); word(source.bindings.host_or_runtime_steps);
    word(source.bindings.bindings.size());
    for (const auto& binding : source.bindings.bindings) {
        word(binding.step_id); word(binding.op); word(binding.family.has_value()); if (binding.family) word(*binding.family);
        word(binding.kernel_id.has_value());
        if (binding.kernel_id) { word(binding.kernel_id->size()); for (const unsigned char byte : *binding.kernel_id) word(byte); }
        word(binding.host_or_runtime_step);
    }
    word(source.io.expanded_hidden_slots); word(source.io.hidden_slots); word(source.io.injection_slots);
    word(source.io.route_index_slots); word(source.io.route_weight_slots); word(source.io.logits_slots);
    word(source.io.mlp_intermediate_slots); word(source.io.gdn_workspace_slots); word(source.io.ple_workspace_slots);
    word(source.io.indexed_attention_workspace_slots); word(source.io.hyper_workspace_slots); word(source.io.router_workspace_slots);
    word(source.io.steps.size());
    for (const auto& io : source.io.steps) {
        word(io.step_id); word(io.op);
        for (const auto* uses : {&io.reads, &io.writes, &io.temporaries}) {
            word(uses->size()); for (const auto use : *uses) hash_use(hash, use);
        }
    }
    word(source.arena.alignment); word(source.arena.total_bytes); word(source.arena.naive_nonaliased_bytes);
    word(source.arena.allocations.size());
    for (const auto& allocation : source.arena.allocations) {
        hash_use(hash, allocation.use); word(allocation.first_step); word(allocation.last_step);
        word(allocation.offset); word(allocation.bytes);
    }
}

// Reconstruct live intervals from the IO instead of trusting first/last_step.
// Reject invalid enums/slots before invoking the serial IO validator's flat map.
void validate_source(const CompiledSerialExecution& source) {
    const auto count = source.execution.steps.size();
    require(source.execution_identity && count && count <= std::numeric_limits<std::uint32_t>::max(),
            "fork/join source requires nonempty identified execution");
    require(source.logical_rows == 1U && source.request_concurrency == 1U,
            "fork/join arena currently admits C1/R1 only");
    require(!source.device_graph_inputs, "fork/join arena does not admit device-graph input topology");
    require(source.io.steps.size() == count, "fork/join source IO size mismatch");
    for (std::size_t index = 0; index < count; ++index) {
        const auto& step = source.execution.steps[index];
        require(step.id == index, "fork/join source step IDs are not dense");
        for (const auto dependency : step.dependencies)
            require(dependency < step.id, "fork/join source dependency is not backward");
        const auto check_uses = [](const auto& uses) {
            for (const auto use : uses)
                require(static_cast<unsigned>(use.buffer.buffer_class) <=
                            static_cast<unsigned>(ExecutionBufferClass::router_workspace) && use.buffer.slot < 8U,
                        "fork/join source buffer is outside serial IO bounds");
        };
        check_uses(source.io.steps[index].reads);
        check_uses(source.io.steps[index].writes);
        check_uses(source.io.steps[index].temporaries);
    }
    validate_serial_execution_io_plan(source.execution, source.io);
    const auto& arena = source.arena;
    require(arena.alignment && arena.total_bytes && arena.total_bytes % arena.alignment == 0U,
            "fork/join source arena is unaligned or empty");
    std::vector<ExecutionArenaAllocation> expected;
    for (std::size_t index = 0; index < count; ++index) {
        const auto& io = source.io.steps[index];
        const auto produce = [&](const auto& uses) {
            for (const auto use : uses) {
                require(std::none_of(expected.begin(), expected.end(), [&](const auto& item) { return item.use == use; }),
                        "fork/join source generation has multiple producers");
                const auto* allocation = find_arena_allocation(arena, use);
                require(allocation != nullptr, "fork/join source allocation is missing");
                auto item = *allocation;
                item.first_step = static_cast<std::uint32_t>(index); item.last_step = item.first_step;
                expected.push_back(item);
            }
        };
        produce(io.writes); produce(io.temporaries);
        for (const auto use : io.reads) {
            const auto found = std::find_if(expected.begin(), expected.end(), [&](const auto& item) { return item.use == use; });
            require(found != expected.end(), "fork/join source read has no allocation producer");
            found->last_step = static_cast<std::uint32_t>(index);
        }
    }
    require(expected.size() == arena.allocations.size(), "fork/join source has duplicate or extraneous allocations");
    for (const auto& item : expected) {
        const auto* allocation = find_arena_allocation(arena, item.use);
        require(allocation->bytes && allocation->offset % arena.alignment == 0U &&
                add(allocation->offset, allocation->bytes) <= arena.total_bytes &&
                allocation->first_step == item.first_step && allocation->last_step == item.last_step,
                "fork/join source lifetime, range or alignment is invalid");
    }
    for (std::size_t a = 0; a < arena.allocations.size(); ++a)
        for (std::size_t b = a + 1U; b < arena.allocations.size(); ++b)
            require(!(live_together(arena.allocations[a], arena.allocations[b]) && overlaps(arena.allocations[a], arena.allocations[b])),
                    "fork/join source aliases simultaneously live serial buffers");
}

MoeForkJoinArenaLayer inspect_layer(const CompiledSerialExecution& source, std::size_t index) {
    require(source.execution.steps.size() - index >= 6U, "fork/join source has an incomplete MoE group");
    const auto& mix = source.execution.steps[index];
    const auto& route = source.execution.steps[index + 1U];
    const auto& shared = source.execution.steps[index + 2U];
    const auto& experts = source.execution.steps[index + 3U];
    const auto& combine = source.execution.steps[index + 4U];
    const auto& inject = source.execution.steps[index + 5U];
    require(mix.layer.has_value() && route.op == SerialOp::moe_route && shared.op == SerialOp::shared_expert &&
            experts.op == SerialOp::moe_experts && combine.op == SerialOp::moe_combine && inject.op == SerialOp::mlp_inject,
            "fork/join requires the supported contiguous MoE topology");
    for (const auto* step : {&route, &shared, &experts, &combine, &inject})
        require(step->layer == mix.layer && step->reads_state == PersistentState::none && step->writes_state == PersistentState::none,
                "fork/join layer or persistent-state effects are unsupported");
    require(same_set(route.dependencies, {mix.id}) && same_set(shared.dependencies, {mix.id}) &&
            same_set(experts.dependencies, {mix.id, route.id}) && same_set(combine.dependencies, {experts.id, shared.id}) &&
            same_set(inject.dependencies, {mix.id, combine.id}), "fork/join dependencies differ from the supported schedule");
    const auto& m = source.io.steps[index]; const auto& r = source.io.steps[index + 1U];
    const auto& s = source.io.steps[index + 2U]; const auto& e = source.io.steps[index + 3U];
    const auto& c = source.io.steps[index + 4U];
    require(m.writes.size() == 2U && r.reads.size() == 1U && r.writes.size() == 2U &&
            s.reads.size() == 1U && s.writes.size() == 1U && s.temporaries.size() == 1U &&
            e.reads.size() == 3U && e.writes.size() == 1U && e.temporaries.size() == 1U &&
            c.reads.size() == 2U && c.writes.size() == 1U && c.temporaries.empty(),
            "fork/join MoE IO cardinality is unsupported");
    const auto input = r.reads.front();
    require(buffer_is(input, ExecutionBufferClass::hidden, 0U) &&
            std::count(m.writes.begin(), m.writes.end(), input) == 1 && s.reads.front() == input &&
            buffer_is(r.writes[0], ExecutionBufferClass::route_indices, 0U) &&
            buffer_is(r.writes[1], ExecutionBufferClass::route_weights, 0U) &&
            buffer_is(s.writes.front(), ExecutionBufferClass::hidden, 1U) &&
            buffer_is(e.writes.front(), ExecutionBufferClass::hidden, 2U) &&
            buffer_is(s.temporaries.front(), ExecutionBufferClass::mlp_intermediate, 0U) &&
            buffer_is(e.temporaries.front(), ExecutionBufferClass::mlp_intermediate, 0U) &&
            same_uses(e.reads, {input, r.writes[0], r.writes[1]}) &&
            same_uses(c.reads, {s.writes.front(), e.writes.front()}) &&
            buffer_is(c.writes.front(), ExecutionBufferClass::hidden, 0U),
            "fork/join MoE IO does not preserve independent read-only inputs and branch outputs");
    // No unrepresented consumer may extend shared output/temporary lifetime.
    for (const auto& io : source.io.steps) {
        for (const auto use : io.reads) {
            require(use != s.temporaries.front(), "fork/join temporary has an external consumer");
            require(use != s.writes.front() || io.step_id == combine.id,
                    "fork/join shared output has a consumer outside its join");
        }
    }
    return {*mix.layer, mix.id, route.id, shared.id, experts.id, combine.id, input,
            s.writes.front(), s.temporaries.front(), e.writes.front(), e.temporaries.front()};
}
} // namespace

MoeForkJoinArenaPlan prepare_moe_fork_join_arena(const CompiledSerialExecution& source, std::uint64_t budget_bytes) {
    validate_source(source);
    MoeForkJoinArenaPlan result;
    result.source_identity_ = source.execution_identity;
    result.source_ = source;
    result.serial_bytes_ = source.arena.total_bytes;
    result.arena_ = source.arena;
    std::uint64_t output_bytes = 0, temporary_bytes = 0;
    for (std::size_t index = 0; index < source.execution.steps.size(); ++index) {
        const auto op = source.execution.steps[index].op;
        if (op == SerialOp::mlp_hyper_mix) {
            auto layer = inspect_layer(source, index);
            require(result.layers_.empty() || result.layers_.back().layer < layer.layer,
                    "fork/join layer order is not strictly increasing");
            output_bytes = std::max(output_bytes, find_arena_allocation(source.arena, layer.shared_output)->bytes);
            temporary_bytes = std::max(temporary_bytes, find_arena_allocation(source.arena, layer.shared_temporary)->bytes);
            result.layers_.push_back(layer); index += 5U;
        } else {
            require(op != SerialOp::moe_route && op != SerialOp::shared_expert && op != SerialOp::moe_experts &&
                    op != SerialOp::moe_combine && op != SerialOp::mlp_inject,
                    "fork/join source contains an ungrouped MoE operation");
        }
    }
    require(!result.layers_.empty(), "fork/join source has no supported MoE layers");
    const auto temporary_offset = add(result.serial_bytes_, aligned(output_bytes, source.arena.alignment));
    result.arena_.total_bytes = add(temporary_offset, aligned(temporary_bytes, source.arena.alignment));
    if (!budget_bytes || result.arena_.total_bytes > budget_bytes)
        throw std::length_error("fork/join complete arena exceeds the admitted memory budget");
    for (const auto& layer : result.layers_) {
        for (auto& allocation : result.arena_.allocations) {
            if (allocation.use == layer.shared_output || allocation.use == layer.shared_temporary) {
                allocation.offset = allocation.use == layer.shared_output ? result.serial_bytes_ : temporary_offset;
                // Conservative annotation covers both branch starts through join.
                // Safety still requires the documented schedule, not these IDs.
                allocation.first_step = layer.route_step; allocation.last_step = layer.combine_step;
            }
        }
    }
    for (std::size_t a = 0; a < result.arena_.allocations.size(); ++a)
        for (std::size_t b = a + 1U; b < result.arena_.allocations.size(); ++b)
            require(!(live_together(result.arena_.allocations[a], result.arena_.allocations[b]) &&
                      overlaps(result.arena_.allocations[a], result.arena_.allocations[b])),
                    "fork/join arena aliases live branch buffers");
    std::uint64_t hash = 14695981039346656037ULL;
    hash_word(hash, 0x4d4f45464a415231ULL); // MoE fork/join arena v1, single layer in flight
    hash_source(hash, source); hash_word(hash, result.arena_.alignment);
    hash_word(hash, result.serial_bytes_); hash_word(hash, result.arena_.total_bytes);
    for (const auto& allocation : result.arena_.allocations) {
        hash_use(hash, allocation.use); hash_word(hash, allocation.first_step); hash_word(hash, allocation.last_step);
        hash_word(hash, allocation.offset); hash_word(hash, allocation.bytes);
    }
    for (const auto& layer : result.layers_) {
        hash_word(hash, layer.layer); hash_word(hash, layer.mix_step); hash_word(hash, layer.route_step);
        hash_word(hash, layer.shared_step); hash_word(hash, layer.experts_step); hash_word(hash, layer.combine_step);
    }
    require(hash && hash != result.source_identity_, "fork/join layout identity is invalid");
    result.layout_identity_ = hash;
    return result;
}

CompiledSerialExecution MoeForkJoinArenaPlan::rebind_execution(const CompiledSerialExecution& source) const {
    require(same_compiled(source, source_), "fork/join rebind source differs from the prepared snapshot");
    auto rebound = source_;
    rebound.arena = arena_;
    rebound.execution_identity = layout_identity_;
    rebound.whole_plan_graph_safe = false;
    return rebound;
}
void MoeForkJoinArenaPlan::validate_execution(const CompiledSerialExecution& rebound) const {
    require(same_compiled(rebound, rebind_execution(source_)),
            "fork/join rebound execution differs from its complete prepared snapshot");
}

void MoeForkJoinArenaPlan::validate_binding(const ExecutionMemoryBinding& memory) const {
    validate_execution_memory_binding(layout_identity_, arena_, memory);
    require(memory.arena_base % arena_.alignment == 0U, "fork/join arena base violates alignment");
    require(memory.arena_bytes <= std::numeric_limits<std::uintptr_t>::max() - memory.arena_base,
            "fork/join bound physical range overflows uintptr_t");
}
ExecutionBufferAddress MoeForkJoinArenaPlan::resolve(const ExecutionMemoryBinding& memory, ExecutionBufferUse use) const {
    validate_binding(memory);
    return resolve_execution_buffer(layout_identity_, arena_, memory, use);
}
} // namespace strix
