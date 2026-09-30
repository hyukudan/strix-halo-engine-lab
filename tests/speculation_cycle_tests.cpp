#include "strix/speculation_cycle.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::PersistentState all_state() {
    using strix::PersistentState;
    return PersistentState::attention_kv |
           PersistentState::recurrent_matrix |
           PersistentState::convolution_history |
           PersistentState::sparse_index |
           PersistentState::ngram_history |
           PersistentState::position_history;
}

strix::GreedySpeculationCycleRequest base_request(
    const std::vector<std::uint32_t>& candidates,
    const std::vector<std::uint32_t>& target) {
    strix::GreedySpeculationCycleRequest request;
    request.compiled_execution_identity =
        0x1234ULL;
    request.runtime_compatibility_identity =
        0x5678ULL;
    request.base_position = 100;
    request.committed_tokens = 100;
    request.state.valid = all_state();
    request.state.generations.fill(9);
    request.candidate_tokens = candidates;
    request.target_tokens = target;
    request.token_budget = 16;
    return request;
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    strix::SerialSessionStateLayoutRequest layout_request;
    layout_request.context_tokens = 1024;
    const auto layout =
        strix::build_serial_session_state_layout(
            model,
            layout_request);

    {
        const std::vector<std::uint32_t> candidates{
            10, 20, 30, 40};
        const std::vector<std::uint32_t> target{
            10, 20, 99, 77, 88};
        auto request =
            base_request(
                candidates,
                target);

        const auto plan =
            strix::build_greedy_speculation_cycle_plan(
                model,
                layout,
                request);

        expect(plan.snapshot_required &&
                   plan.restore_required,
               "partial match requires pre-window snapshot and restore");
        expect(plan.verify_state_advance == 4 &&
                   plan.committed_state_advance == 2,
               "verify mutates through all candidates but commits only matches");
        expect(plan.replay_tokens ==
                   std::vector<std::uint32_t>({10, 20}),
               "partial match replays only accepted candidate prefix");
        expect(plan.emitted_tokens ==
                   std::vector<std::uint32_t>({10, 20, 99}),
               "cycle emits accepted candidates plus target bonus");
        expect(plan.post_cycle_state_position == 102 &&
                   plan.post_cycle_output_position == 103,
               "bonus output is not yet represented in target state");
        expect(plan.post_cycle_committed_tokens == 103,
               "output token count advances by emitted continuation size");
        expect(plan.pre_window_snapshot.next_position == 100 &&
                   plan.pre_window_snapshot.compiled_execution_identity ==
                       request.compiled_execution_identity,
               "snapshot captures exact pre-window serial state identity");
    }

    {
        const std::vector<std::uint32_t> candidates{
            1, 2, 3};
        const std::vector<std::uint32_t> target{
            1, 2, 3, 4};
        auto request =
            base_request(
                candidates,
                target);

        const auto plan =
            strix::build_greedy_speculation_cycle_plan(
                model,
                layout,
                request);

        expect(plan.snapshot_required &&
                   !plan.restore_required &&
                   plan.replay_tokens.empty(),
               "all-match cycle keeps verified target state without restore");
        expect(plan.post_cycle_state_position == 103 &&
                   plan.post_cycle_output_position == 104,
               "all accepted candidates advance state; bonus remains unprocessed");
    }

    {
        const std::vector<std::uint32_t> candidates;
        const std::vector<std::uint32_t> target{42};
        auto request =
            base_request(
                candidates,
                target);
        request.token_budget = 1;

        const auto plan =
            strix::build_greedy_speculation_cycle_plan(
                model,
                layout,
                request);

        expect(!plan.speculation_used &&
                   !plan.snapshot_required &&
                   !plan.restore_required &&
                   plan.post_cycle_state_position == 100 &&
                   plan.post_cycle_output_position == 101,
               "depth zero reduces to one serial target bonus without snapshot");
    }

    {
        const std::vector<std::uint32_t> candidates{
            5, 6, 7};
        const std::vector<std::uint32_t> target{
            5, 6, 7, 8};
        const std::vector<std::uint32_t> eos{6};
        auto request =
            base_request(
                candidates,
                target);
        request.eos_tokens = eos;

        const auto plan =
            strix::build_greedy_speculation_cycle_plan(
                model,
                layout,
                request);

        expect(plan.verification.accepted_matches == 1 &&
                   plan.restore_required &&
                   plan.replay_tokens ==
                       std::vector<std::uint32_t>({5}) &&
                   plan.emitted_tokens ==
                       std::vector<std::uint32_t>({5, 6}),
               "EOS crop restores and replays only state for retained match");
    }

    {
        const std::vector<std::uint32_t> candidates{
            1, 2, 3, 4};
        const std::vector<std::uint32_t> target{
            1, 2, 3, 4, 5};
        auto request =
            base_request(
                candidates,
                target);
        request.base_position = 1022;

        bool rejected = false;
        try {
            static_cast<void>(
                strix::build_greedy_speculation_cycle_plan(
                    model,
                    layout,
                    request));
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        expect(rejected,
               "verify window cannot exceed allocated session capacity");
    }

    if (failures == 0) {
        std::cout << "speculation cycle tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
