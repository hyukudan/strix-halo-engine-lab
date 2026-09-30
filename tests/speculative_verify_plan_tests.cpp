#include "strix/speculative_verify_plan.hpp"

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

    const auto verify =
        strix::build_speculative_verify_plan(
            model,
            4);

    expect(verify.candidate_depth == 4 &&
               verify.candidate_input_rows == 4 &&
               verify.produced_logit_rows == 4 &&
               verify.requires_base_target_token,
           "verify plan models D candidate inputs plus external base target token");

    std::uint32_t gdn = 0;
    std::uint32_t indexed = 0;
    std::uint32_t ple = 0;
    std::uint32_t moe_experts = 0;
    std::uint32_t lm_head = 0;
    std::uint32_t position = 0;

    for (const auto& step :
         verify.steps) {
        if (step.op ==
            strix::SerialOp::gated_delta_net) {
            ++gdn;
            expect(step.mode ==
                       strix::SpeculativeVerifyMode::sequential_window,
                   "GDN verify uses sequential retained-state window");
        }
        if (step.op ==
            strix::SerialOp::indexed_attention) {
            ++indexed;
            expect(step.mode ==
                       strix::SpeculativeVerifyMode::causal_window,
                   "indexed attention verify uses causal candidate window");
        }
        if (step.op ==
            strix::SerialOp::ple) {
            ++ple;
            expect(step.mode ==
                       strix::SpeculativeVerifyMode::sequential_window,
                   "PLE verify preserves sequential n-gram/conv state");
        }
        if (step.op ==
            strix::SerialOp::moe_experts) {
            ++moe_experts;
            expect(step.mode ==
                       strix::SpeculativeVerifyMode::dynamic_row_parallel,
                   "routed experts remain dynamic across verify rows");
        }
        if (step.op ==
            strix::SerialOp::lm_head) {
            ++lm_head;
            expect(step.mode ==
                       strix::SpeculativeVerifyMode::row_parallel,
                   "LM head produces one logit row per processed candidate");
        }
        if (step.op ==
            strix::SerialOp::position_prepare) {
            ++position;
            expect(step.mode ==
                       strix::SpeculativeVerifyMode::runtime_position_window,
                   "position preparation materializes candidate position window");
        }
    }

    expect(gdn == 36 &&
               indexed == 12 &&
               ple == 1 &&
               moe_experts == 48 &&
               lm_head == 1 &&
               position == 1,
           "verify plan preserves pinned model layer topology");

    expect(verify.sequential_window_steps == 37 &&
               verify.causal_window_steps == 12,
           "verify plan counts stateful sequential and causal steps");
    expect(verify.dynamic_steps >= 49,
           "verify plan retains dynamic PLE/indexed/MoE requirements");

    bool zero_rejected = false;
    try {
        static_cast<void>(
            strix::build_speculative_verify_plan(
                model,
                0));
    } catch (const std::invalid_argument&) {
        zero_rejected = true;
    }
    expect(zero_rejected,
           "verify plan rejects depth zero because serial baseline is separate");

    if (failures == 0) {
        std::cout << "speculative verify plan tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
