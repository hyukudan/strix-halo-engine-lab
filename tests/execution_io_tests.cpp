#include "strix/execution_io.hpp"

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

bool has_buffer(
    const std::vector<strix::ExecutionBufferUse>& values,
    strix::ExecutionBufferClass buffer_class,
    std::uint32_t slot) {
    for (const auto& value : values) {
        if (value.buffer.buffer_class == buffer_class &&
            value.buffer.slot == slot) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto execution =
        strix::build_serial_decode_execution_plan(model);
    const auto io =
        strix::build_serial_execution_io_plan(execution);

    expect(io.steps.size() == execution.steps.size(),
           "IO plan covers every execution step");
    expect(io.expanded_hidden_slots == 2 &&
               io.hidden_slots == 3,
           "IO plan uses two hyper and three hidden slots");
    expect(io.injection_slots == 1 &&
               io.route_index_slots == 1 &&
               io.route_weight_slots == 1 &&
               io.logits_slots == 1 &&
               io.mlp_intermediate_slots == 1 &&
               io.gdn_workspace_slots == 1,
           "IO plan reuses one side-value slot per serial lifetime class");

    std::uint32_t ple_count = 0;
    std::uint32_t combine_count = 0;
    std::uint32_t lm_head_count = 0;
    std::uint32_t shared_temp_count = 0;
    std::uint32_t routed_temp_count = 0;
    std::uint32_t gdn_temp_count = 0;

    for (const auto& step : io.steps) {
        if (step.op == strix::SerialOp::ple) {
            ++ple_count;
            expect(step.reads.size() == 1 &&
                       step.writes.size() == 1 &&
                       step.reads[0].buffer.buffer_class ==
                           strix::ExecutionBufferClass::expanded_hidden &&
                       step.writes[0].buffer.buffer_class ==
                           strix::ExecutionBufferClass::expanded_hidden &&
                       step.reads[0].buffer.slot != step.writes[0].buffer.slot,
                   "PLE ping-pongs expanded hidden state");
            expect(
                step.temporaries.size() == 1 &&
                step.temporaries[0].buffer.buffer_class ==
                    strix::ExecutionBufferClass::ple_workspace,
                "PLE owns one step-local workspace");
        }



        if (step.op == strix::SerialOp::gated_delta_net) {
            ++gdn_temp_count;
            expect(
                step.temporaries.size() == 1 &&
                step.temporaries[0].buffer.buffer_class ==
                    strix::ExecutionBufferClass::gdn_workspace,
                "GDN owns one step-local workspace");
        }

        if (step.op == strix::SerialOp::shared_expert) {
            ++shared_temp_count;
            expect(
                step.temporaries.size() == 1 &&
                step.temporaries[0].buffer.buffer_class ==
                    strix::ExecutionBufferClass::mlp_intermediate,
                "shared expert owns one step-local MLP temporary");
        }

        if (step.op == strix::SerialOp::moe_experts) {
            ++routed_temp_count;
            expect(
                step.temporaries.size() == 1 &&
                step.temporaries[0].buffer.buffer_class ==
                    strix::ExecutionBufferClass::mlp_intermediate,
                "routed experts own one step-local MLP temporary");
        }


        if (step.op == strix::SerialOp::indexed_attention) {
            expect(
                step.temporaries.size() == 1 &&
                step.temporaries[0].buffer.buffer_class ==
                    strix::ExecutionBufferClass::indexed_attention_workspace,
                "indexed attention owns one step-local workspace");
        }

        if (step.op == strix::SerialOp::moe_combine) {
            ++combine_count;
            expect(has_buffer(
                       step.reads,
                       strix::ExecutionBufferClass::hidden,
                       1) &&
                       has_buffer(
                           step.reads,
                           strix::ExecutionBufferClass::hidden,
                           2) &&
                       has_buffer(
                           step.writes,
                           strix::ExecutionBufferClass::hidden,
                           0),
                   "MoE combine restores into reusable hidden slot zero");
        }

        if (step.op == strix::SerialOp::lm_head) {
            ++lm_head_count;
            expect(has_buffer(
                       step.reads,
                       strix::ExecutionBufferClass::hidden,
                       0) &&
                       has_buffer(
                           step.writes,
                           strix::ExecutionBufferClass::logits,
                           0),
                   "LM head consumes final hidden and materializes logits");
        }
    }

    expect(ple_count == 1,
           "IO plan includes exactly one PLE ping-pong");
    expect(combine_count == 48,
           "IO plan includes one MoE combine per layer");
    expect(lm_head_count == 1,
           "IO plan includes one final LM head");
    expect(shared_temp_count == 48 &&
               routed_temp_count == 48,
           "every MoE branch step receives one temporary workspace");
    expect(gdn_temp_count == 36,
           "every recurrent layer receives one GDN workspace");

    auto invalid = io;
    invalid.steps.back().reads[0].buffer =
        strix::ExecutionBufferRef{
            strix::ExecutionBufferClass::hidden,
            7};
    bool slot_rejected = false;
    try {
        strix::validate_serial_execution_io_plan(
            execution, invalid);
    } catch (const std::invalid_argument&) {
        slot_rejected = true;
    }
    expect(slot_rejected,
           "invalid buffer slot is rejected");

    invalid = io;
    invalid.steps[2].reads[0].buffer =
        strix::ExecutionBufferRef{
            strix::ExecutionBufferClass::route_indices,
            0};
    bool early_read_rejected = false;
    try {
        strix::validate_serial_execution_io_plan(
            execution, invalid);
    } catch (const std::invalid_argument&) {
        early_read_rejected = true;
    }
    expect(early_read_rejected,
           "read-before-produce is rejected");

    invalid = io;
    invalid.steps[2].writes[0].buffer =
        invalid.steps[2].reads[0].buffer;
    bool alias_rejected = false;
    try {
        strix::validate_serial_execution_io_plan(
            execution, invalid);
    } catch (const std::invalid_argument&) {
        alias_rejected = true;
    }
    expect(alias_rejected,
           "same-step read/write alias is rejected");


    invalid = io;
    invalid.steps.back().reads[0].generation -= 1;
    bool stale_generation_rejected = false;
    try {
        strix::validate_serial_execution_io_plan(
            execution, invalid);
    } catch (const std::invalid_argument&) {
        stale_generation_rejected = true;
    }
    expect(stale_generation_rejected,
           "stale buffer generation is rejected");

    invalid = io;
    invalid.steps.back().writes[0].generation += 1;
    bool skipped_generation_rejected = false;
    try {
        strix::validate_serial_execution_io_plan(
            execution, invalid);
    } catch (const std::invalid_argument&) {
        skipped_generation_rejected = true;
    }
    expect(skipped_generation_rejected,
           "skipped write generation is rejected");


    auto stale_temporary = io;
    const auto temp_step =
        std::find_if(
            stale_temporary.steps.begin(),
            stale_temporary.steps.end(),
            [](const auto& step) {
                return !step.temporaries.empty();
            });
    expect(
        temp_step != stale_temporary.steps.end(),
        "IO fixture contains at least one temporary");
    if (temp_step != stale_temporary.steps.end()) {
        temp_step->temporaries[0].generation = 0;
        bool rejected = false;
        try {
            strix::validate_serial_execution_io_plan(
                execution,
                stale_temporary);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(
            rejected,
            "stale temporary generation is rejected");
    }

    if (failures == 0) {
        std::cout << "execution IO tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
