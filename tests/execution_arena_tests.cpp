#include "strix/execution_arena.hpp"

#include "strix/model_contract.hpp"

#include <algorithm>

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

    if (failures == 0) {
        std::cout << "execution arena tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
