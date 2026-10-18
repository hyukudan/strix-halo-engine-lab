#include "strix/execution_arena.hpp"

#include "strix/model_contract.hpp"

#include <algorithm>
#include <limits>

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto execution =
        strix::build_serial_decode_execution_plan(model);
    const auto io =
        strix::build_serial_execution_io_plan(execution);

    strix::ExecutionBufferRequest request;
    request.logical_rows = 1;

    const auto sizes =
        strix::build_serial_execution_buffer_plan(
            model, request);

    const auto exact =
        strix::build_serial_execution_arena_plan(
            model,
            request,
            execution,
            io,
            1);

    expect(exact.total_bytes ==
               sizes.peak_with_materialized_logits_bytes,
           "liveness allocator reproduces the analytical serial peak");
    expect(exact.total_bytes <
               exact.naive_nonaliased_bytes,
           "liveness allocator aliases dead buffer generations");

    const auto aligned =
        strix::build_serial_execution_arena_plan(
            model,
            request,
            execution,
            io,
            256);
    expect(aligned.total_bytes >= exact.total_bytes &&
               aligned.total_bytes % 256ULL == 0,
           "aligned arena rounds the physical allocation safely");
    expect(aligned.total_bytes <
               aligned.naive_nonaliased_bytes,
           "aligned arena still reuses dead lifetimes");




    const auto indexed_it =
        std::find_if(
            io.steps.begin(),
            io.steps.end(),
            [](const auto& step) {
                return step.op ==
                    strix::SerialOp::indexed_attention;
            });
    expect(
        indexed_it != io.steps.end() &&
            indexed_it->temporaries.size() == 1,
        "arena fixture exposes indexed-attention temporary");
    if (indexed_it != io.steps.end() &&
        !indexed_it->temporaries.empty()) {
        const auto* temporary =
            strix::find_arena_allocation(
                aligned,
                indexed_it->temporaries[0]);
        expect(
            temporary != nullptr &&
                temporary->first_step ==
                    temporary->last_step,
            "indexed-attention temporary has one-step lifetime");
        expect(
            temporary != nullptr &&
                temporary->bytes ==
                    sizes.indexed_attention_workspace_bytes,
            "indexed-attention temporary uses analytical workspace size");
    }

    const auto ple_it =
        std::find_if(
            io.steps.begin(),
            io.steps.end(),
            [](const auto& step) {
                return step.op ==
                    strix::SerialOp::ple;
            });
    expect(
        ple_it != io.steps.end() &&
            ple_it->temporaries.size() == 1,
        "arena fixture exposes PLE temporary");
    if (ple_it != io.steps.end() &&
        !ple_it->temporaries.empty()) {
        const auto* temporary =
            strix::find_arena_allocation(
                aligned,
                ple_it->temporaries[0]);
        expect(
            temporary != nullptr &&
                temporary->first_step ==
                    temporary->last_step,
            "PLE temporary has one-step lifetime");
        expect(
            temporary != nullptr &&
                temporary->bytes ==
                    sizes.ple_workspace_bytes,
            "PLE temporary uses canonical workspace size");
    }

    const auto shared_it =
        std::find_if(
            io.steps.begin(),
            io.steps.end(),
            [](const auto& step) {
                return step.op ==
                    strix::SerialOp::shared_expert;
            });
    expect(
        shared_it != io.steps.end() &&
            shared_it->temporaries.size() == 1,
        "arena fixture exposes shared-expert temporary");
    if (shared_it != io.steps.end() &&
        !shared_it->temporaries.empty()) {
        const auto* temporary =
            strix::find_arena_allocation(
                aligned,
                shared_it->temporaries[0]);
        expect(
            temporary != nullptr &&
                temporary->first_step ==
                    temporary->last_step,
            "temporary workspace has one-step lifetime");
        expect(
            temporary != nullptr &&
                temporary->bytes ==
                    sizes.mlp_intermediate_bytes,
            "temporary workspace uses analytical MLP size");
    }

    const auto& lm_io = io.steps.back();
    expect(lm_io.op == strix::SerialOp::lm_head &&
               lm_io.reads.size() == 1 &&
               lm_io.writes.size() == 1,
           "LM head IO fixture is simple");

    const auto* final_hidden =
        strix::find_arena_allocation(
            aligned,
            lm_io.reads[0]);
    const auto* logits =
        strix::find_arena_allocation(
            aligned,
            lm_io.writes[0]);
    expect(final_hidden != nullptr &&
               logits != nullptr,
           "arena resolves versioned LM-head buffers");
    if (final_hidden != nullptr &&
        logits != nullptr) {
        const auto hidden_end =
            final_hidden->offset +
            final_hidden->bytes;
        const auto logits_end =
            logits->offset +
            logits->bytes;
        expect(hidden_end <= logits->offset ||
                   logits_end <= final_hidden->offset,
               "LM-head input and output do not alias while simultaneously live");
    }

    request.logical_rows = 8;
    const auto rows_eight =
        strix::build_serial_execution_arena_plan(
            model,
            request,
            execution,
            io,
            256);
    expect(rows_eight.total_bytes >
               aligned.total_bytes,
           "arena scales with logical row class");

    bool invalid_alignment = false;
    try {
        static_cast<void>(
            strix::build_serial_execution_arena_plan(
                model,
                request,
                execution,
                io,
                0));
    } catch (const std::invalid_argument&) {
        invalid_alignment = true;
    }
    expect(invalid_alignment,
           "zero arena alignment is rejected");


    const auto window = strix::build_known_window_arena(aligned);
    expect(window.row_stride.size() == window.physical.allocations.size(),
        "known window retains a stride for every actual serial allocation");
    for (std::size_t index = 0; index < window.physical.allocations.size(); ++index) {
        const auto& physical = window.physical.allocations[index];
        const auto& first = window.row_view[0].allocations[index];
        const auto& second = window.row_view[1].allocations[index];
        const auto* original = strix::find_arena_allocation(aligned, physical.use);
        expect(original && first.bytes == original->bytes && second.bytes == original->bytes &&
            second.offset == first.offset + window.row_stride[index] &&
            first.offset + first.bytes <= second.offset &&
            second.offset + second.bytes <= physical.offset + physical.bytes,
            "known window preserves exact R1 payloads in disjoint aligned row slices");
        expect(window.row_view[0].total_bytes == window.physical.total_bytes &&
            window.row_view[1].total_bytes == window.physical.total_bytes,
            "row views bind the complete physical allocation extent");
        for (std::size_t previous = 0; previous < index; ++previous) {
            const auto& other = window.physical.allocations[previous];
            if (physical.first_step <= other.last_step && other.first_step <= physical.last_step)
                expect(physical.offset + physical.bytes <= other.offset ||
                    other.offset + other.bytes <= physical.offset,
                    "known window never aliases live two-row allocation regions");
        }
    }
    auto duplicate = aligned; duplicate.allocations.push_back(duplicate.allocations.front());
    bool rejected_window = false;
    try { (void)strix::build_known_window_arena(duplicate); }
    catch (const std::invalid_argument&) { rejected_window = true; }
    expect(rejected_window, "known window rejects duplicated serial uses");
    auto malformed = aligned; malformed.allocations.front().first_step = malformed.allocations.front().last_step + 1U;
    rejected_window = false;
    try { (void)strix::build_known_window_arena(malformed); }
    catch (const std::invalid_argument&) { rejected_window = true; }
    expect(rejected_window, "known window rejects malformed source lifetimes");
    strix::ExecutionArenaPlan huge;
    huge.alignment = 1; huge.total_bytes = std::numeric_limits<std::uint64_t>::max();
    auto huge_allocation = aligned.allocations.front();
    huge_allocation.offset = 0; huge_allocation.bytes = std::numeric_limits<std::uint64_t>::max() / 2U + 1U;
    huge.allocations.push_back(huge_allocation);
    rejected_window = false;
    try { (void)strix::build_known_window_arena(huge,1); }
    catch (const std::overflow_error&) { rejected_window = true; }
    expect(rejected_window, "known window rejects doubled row extent overflow");

    {
        // The scoped index answers every lookup as the linear search does: first allocation of a
        // duplicated use, nullptr for an absent one, and a plan that grows inside the scope.
        auto lookup = aligned; lookup.allocations.push_back(lookup.allocations.front());
        auto absent = lookup.allocations.front().use; absent.generation = std::numeric_limits<std::uint64_t>::max();
        std::vector<const strix::ExecutionArenaAllocation*> linear;
        for (const auto& allocation : lookup.allocations) linear.push_back(strix::find_arena_allocation(lookup, allocation.use));
        bool same = strix::find_arena_allocation(lookup, absent) == nullptr && linear.back() == &lookup.allocations.front();
        {
            const strix::ArenaLookupScope scope;
            for (std::size_t i = 0; i < lookup.allocations.size(); ++i)
                same = same && strix::find_arena_allocation(lookup, lookup.allocations[i].use) == linear[i];
            same = same && strix::find_arena_allocation(lookup, absent) == nullptr;
            auto grown = lookup.allocations.back(); grown.use = absent;
            lookup.allocations.push_back(grown);
            same = same && strix::find_arena_allocation(lookup, absent) == &lookup.allocations.back();
        }
        expect(same, "scoped arena lookup index matches the linear search");
    }

    if (failures == 0) {
        std::cout << "execution arena tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
