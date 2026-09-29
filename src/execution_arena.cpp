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
