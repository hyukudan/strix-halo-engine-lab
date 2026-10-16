#include "strix/execution_arena.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strix {
namespace {

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(
            "execution arena addition overflow");
    }
    return left + right;
}


std::uint64_t align_up(
    std::uint64_t value,
    std::uint64_t alignment) {
    if (alignment == 0) {
        throw std::invalid_argument(
            "execution arena alignment must be non-zero");
    }
    const auto remainder = value % alignment;
    return remainder == 0
        ? value
        : checked_add(
              value,
              alignment - remainder);
}

bool same_use(
    ExecutionBufferUse left,
    ExecutionBufferUse right) noexcept {
    return left == right;
}

std::uint64_t bytes_for(
    const ExecutionBufferPlan& sizes,
    ExecutionBufferClass buffer_class) {
    switch (buffer_class) {
    case ExecutionBufferClass::expanded_hidden:
        return sizes.expanded_hidden_bytes;
    case ExecutionBufferClass::hidden:
        return sizes.hidden_bytes;
    case ExecutionBufferClass::injection_weights:
        return sizes.injection_weight_bytes;
    case ExecutionBufferClass::route_indices:
        return sizes.route_index_bytes;
    case ExecutionBufferClass::route_weights:
        return sizes.route_weight_bytes;
    case ExecutionBufferClass::logits:
        return sizes.logits_bytes;
    case ExecutionBufferClass::mlp_intermediate:
        return sizes.mlp_intermediate_bytes;
    case ExecutionBufferClass::gdn_workspace:
        return sizes.gdn_workspace_bytes;
    case ExecutionBufferClass::ple_workspace:
        return sizes.ple_workspace_bytes;
    case ExecutionBufferClass::indexed_attention_workspace:
        return sizes.indexed_attention_workspace_bytes;
    case ExecutionBufferClass::hyper_workspace:
        return sizes.hyper_workspace_bytes;
    case ExecutionBufferClass::router_workspace:
        return sizes.router_workspace_bytes;
    }
    throw std::logic_error(
        "unknown execution buffer class");
}


}  // namespace

ExecutionArenaPlan build_serial_execution_arena_plan(
    const ModelContract& model,
    const ExecutionBufferRequest& buffer_request,
    const SerialExecutionPlan& execution_plan,
    const SerialExecutionIoPlan& io_plan,
    std::uint64_t alignment) {
    if (alignment == 0) {
        throw std::invalid_argument(
            "execution arena alignment must be non-zero");
    }

    validate_serial_execution_io_plan(
        execution_plan, io_plan);

    if ((io_plan.router_workspace_slots != 0U) != buffer_request.parallel_moe_router)
        throw std::invalid_argument("router workspace admission differs from buffer request");
    if (buffer_request.parallel_moe_router && alignment % 16U != 0)
        throw std::invalid_argument("parallel router arena requires at least 16-byte alignment");

    if (buffer_request.q4k_a8g32_gate_up && alignment % 256U != 0)
        throw std::invalid_argument("Q4K A8 runtime arena requires 256-byte alignment");

    const auto sizes =
        build_serial_execution_buffer_plan(
            model, buffer_request);

    ExecutionArenaPlan plan;
    plan.alignment = alignment;

    for (std::size_t step_index = 0;
         step_index < io_plan.steps.size();
         ++step_index) {
        const auto& io = io_plan.steps[step_index];

        for (const auto use : io.writes) {
            const auto found =
                std::find_if(
                    plan.allocations.begin(),
                    plan.allocations.end(),
                    [&](const ExecutionArenaAllocation& allocation) {
                        return same_use(
                            allocation.use, use);
                    });
            if (found != plan.allocations.end()) {
                throw std::invalid_argument(
                    "execution buffer generation is written more than once");
            }

            const auto bytes =
                bytes_for(
                    sizes,
                    use.buffer.buffer_class);
            if (bytes == 0) {
                throw std::invalid_argument(
                    "execution arena buffer size must be non-zero");
            }

            plan.allocations.push_back(
                ExecutionArenaAllocation{
                    use,
                    static_cast<std::uint32_t>(step_index),
                    static_cast<std::uint32_t>(step_index),
                    0,
                    bytes,
                });
            plan.naive_nonaliased_bytes =
                checked_add(
                    plan.naive_nonaliased_bytes,
                    align_up(bytes, alignment));
        }

        for (const auto use : io.reads) {
            const auto found =
                std::find_if(
                    plan.allocations.begin(),
                    plan.allocations.end(),
                    [&](const ExecutionArenaAllocation& allocation) {
                        return same_use(
                            allocation.use, use);
                    });
            if (found == plan.allocations.end()) {
                throw std::invalid_argument(
                    "execution arena read has no producing generation");
            }
            found->last_step =
                std::max(
                    found->last_step,
                    static_cast<std::uint32_t>(
                        step_index));
        }

        for (const auto use : io.temporaries) {
            const auto found =
                std::find_if(
                    plan.allocations.begin(),
                    plan.allocations.end(),
                    [&](const ExecutionArenaAllocation& allocation) {
                        return same_use(
                            allocation.use, use);
                    });
            if (found != plan.allocations.end()) {
                throw std::invalid_argument(
                    "execution temporary generation is allocated more than once");
            }

            const auto bytes =
                bytes_for(
                    sizes,
                    use.buffer.buffer_class);
            if (bytes == 0) {
                throw std::invalid_argument(
                    "execution temporary size must be non-zero");
            }

            plan.allocations.push_back(
                ExecutionArenaAllocation{
                    use,
                    static_cast<std::uint32_t>(step_index),
                    static_cast<std::uint32_t>(step_index),
                    0,
                    bytes,
                });
            plan.naive_nonaliased_bytes =
                checked_add(
                    plan.naive_nonaliased_bytes,
                    align_up(bytes, alignment));
        }
    }

    std::stable_sort(
        plan.allocations.begin(),
        plan.allocations.end(),
        [](const auto& left, const auto& right) {
            if (left.first_step != right.first_step) {
                return left.first_step <
                       right.first_step;
            }
            if (left.bytes != right.bytes) {
                return left.bytes > right.bytes;
            }
            if (left.use.buffer.buffer_class !=
                right.use.buffer.buffer_class) {
                return left.use.buffer.buffer_class <
                       right.use.buffer.buffer_class;
            }
            if (left.use.buffer.slot !=
                right.use.buffer.slot) {
                return left.use.buffer.slot <
                       right.use.buffer.slot;
            }
            return left.use.generation <
                   right.use.generation;
        });

    std::vector<const ExecutionArenaAllocation*> active;

    for (auto& allocation : plan.allocations) {
        active.erase(
            std::remove_if(
                active.begin(),
                active.end(),
                [&](const ExecutionArenaAllocation* other) {
                    return other->last_step <
                           allocation.first_step;
                }),
            active.end());

        std::sort(
            active.begin(),
            active.end(),
            [](const auto* left, const auto* right) {
                return left->offset < right->offset;
            });

        std::uint64_t candidate = 0;
        for (const auto* other : active) {
            candidate =
                align_up(candidate, alignment);
            const auto candidate_end =
                checked_add(
                    candidate,
                    allocation.bytes);
            const auto other_end =
                checked_add(
                    other->offset,
                    other->bytes);

            if (candidate_end <= other->offset) {
                break;
            }
            if (candidate >= other_end) {
                continue;
            }

            candidate = other_end;
        }

        allocation.offset =
            align_up(candidate, alignment);

        const auto end =
            checked_add(
                allocation.offset,
                allocation.bytes);
        plan.total_bytes =
            std::max(
                plan.total_bytes,
                end);
        active.push_back(&allocation);
    }

    plan.total_bytes =
        align_up(
            plan.total_bytes,
            alignment);
    return plan;
}

namespace {
KnownWindowArena build_known_rows_arena(
    const ExecutionArenaPlan& serial, std::uint64_t alignment, std::uint32_t logical_rows) {
    if (!serial.alignment || !alignment || alignment % serial.alignment != 0 ||
        !serial.total_bytes || serial.allocations.empty()) {
        throw std::invalid_argument("known window requires a complete aligned serial arena");
    }
    for (std::size_t i = 0; i < serial.allocations.size(); ++i) {
        const auto& current = serial.allocations[i];
        if (!current.bytes || current.first_step > current.last_step ||
            current.offset % serial.alignment != 0 ||
            checked_add(current.offset, current.bytes) > serial.total_bytes) {
            throw std::invalid_argument("known window serial allocation is invalid");
        }
        for (std::size_t j = 0; j < i; ++j) {
            const auto& previous = serial.allocations[j];
            if (current.use == previous.use) {
                throw std::invalid_argument("known window serial use is duplicated");
            }
            const bool live = current.first_step <= previous.last_step &&
                              previous.first_step <= current.last_step;
            if (live && current.offset < checked_add(previous.offset, previous.bytes) &&
                previous.offset < checked_add(current.offset, current.bytes)) {
                throw std::invalid_argument("known window source aliases live serial allocations");
            }
        }
    }
    KnownWindowArena result;
    result.physical.alignment = alignment;
    result.physical.allocations = serial.allocations;
    for (auto& allocation : result.physical.allocations) {
        const auto stride = align_up(allocation.bytes, alignment);
        allocation.bytes = 0;
        for(std::uint32_t row=0;row<logical_rows;++row) allocation.bytes=checked_add(allocation.bytes,stride);
        allocation.offset = 0;
        result.physical.naive_nonaliased_bytes = checked_add(
            result.physical.naive_nonaliased_bytes, allocation.bytes);
    }
    std::stable_sort(result.physical.allocations.begin(), result.physical.allocations.end(),
        [](const auto& left, const auto& right) {
            if (left.first_step != right.first_step) return left.first_step < right.first_step;
            if (left.bytes != right.bytes) return left.bytes > right.bytes;
            if (left.use.buffer.buffer_class != right.use.buffer.buffer_class)
                return left.use.buffer.buffer_class < right.use.buffer.buffer_class;
            if (left.use.buffer.slot != right.use.buffer.slot)
                return left.use.buffer.slot < right.use.buffer.slot;
            return left.use.generation < right.use.generation;
        });
    std::vector<const ExecutionArenaAllocation*> active;
    for (auto& allocation : result.physical.allocations) {
        active.erase(std::remove_if(active.begin(), active.end(), [&](const auto* previous) {
            return previous->last_step < allocation.first_step;
        }), active.end());
        std::sort(active.begin(), active.end(), [](const auto* left, const auto* right) {
            return left->offset < right->offset;
        });
        std::uint64_t candidate = 0;
        for (const auto* previous : active) {
            candidate = align_up(candidate, alignment);
            if (checked_add(candidate, allocation.bytes) <= previous->offset) break;
            const auto previous_end = checked_add(previous->offset, previous->bytes);
            if (candidate < previous_end) candidate = previous_end;
        }
        allocation.offset = align_up(candidate, alignment);
        result.physical.total_bytes = std::max(result.physical.total_bytes,
            checked_add(allocation.offset, allocation.bytes));
        active.push_back(&allocation);
    }
    result.physical.total_bytes = align_up(result.physical.total_bytes, alignment);
    result.row_view.assign(logical_rows,result.physical);
    for (std::size_t i = 0; i < result.physical.allocations.size(); ++i) {
        const auto& physical = result.physical.allocations[i];
        const auto* original = find_arena_allocation(serial, physical.use);
        if (!original) throw std::logic_error("known window lost a serial allocation");
        const auto stride = physical.bytes / logical_rows;
        result.row_stride.push_back(stride);
        for (std::size_t row = 0; row < result.row_view.size(); ++row) {
            auto& allocation = result.row_view[row].allocations[i];
            allocation.offset = checked_add(physical.offset, static_cast<std::uint64_t>(row)*stride);
            allocation.bytes = original->bytes;
        }
    }
    return result;
}
} // namespace

KnownWindowArena build_known_window_arena(
    const ExecutionArenaPlan& serial,std::uint64_t alignment,std::uint32_t logical_rows) {
    if (logical_rows!=2U && logical_rows!=4U && logical_rows!=8U)
        throw std::invalid_argument("known window rows must be two, four or eight");
    return build_known_rows_arena(serial,alignment,logical_rows);
}
KnownWindowArena build_known_prefill_arena(
    const ExecutionArenaPlan& serial,std::uint64_t alignment,std::uint32_t logical_rows) {
    if(logical_rows<16U||logical_rows>2048U)
        throw std::invalid_argument("known prefill rows must be within 16 and 2048");
    return build_known_rows_arena(serial,alignment,logical_rows);
}

const ExecutionArenaAllocation* find_arena_allocation(
    const ExecutionArenaPlan& plan,
    ExecutionBufferUse use) noexcept {
    const auto found =
        std::find_if(
            plan.allocations.begin(),
            plan.allocations.end(),
            [&](const ExecutionArenaAllocation& allocation) {
                return allocation.use == use;
            });
    return found == plan.allocations.end()
        ? nullptr
        : &*found;
}

}  // namespace strix
