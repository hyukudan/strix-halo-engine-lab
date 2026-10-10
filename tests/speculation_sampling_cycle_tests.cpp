#include "strix/speculation_sampling_cycle.hpp"

#include <iostream>
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

strix::SampledSpeculationCycleRequest base_request(
    const std::vector<std::uint32_t>& candidates) {
    strix::SampledSpeculationCycleRequest request;
    request.compiled_execution_identity =
        0x1111ULL;
    request.runtime_compatibility_identity =
        0x2222ULL;
    request.base_position = 100;
    request.committed_tokens = 100;
    request.state.valid = all_state();
    request.state.generations.fill(7);
    request.candidate_tokens = candidates;
    request.token_budget = 8;
    return request;
}

strix::SpeculationRngIdentity rng() {
    return {1234, 0x3333ULL, 0x4444ULL};
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
        const std::vector<std::uint32_t> candidates{0, 1};
        auto request =
            base_request(candidates);
        const auto preparation =
            strix::prepare_sampled_speculation_cycle(
                model,
                layout,
                request);

        const std::vector<std::vector<double>> target{
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
        };
        const std::vector<std::vector<double>> draft{
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
        };
        strix::SampledSpeculationFinalizeRequest finalize;
        finalize.target_probabilities = target;
        finalize.draft_probabilities = draft;
        finalize.rng = rng();
        finalize.token_budget = 3;

        const auto plan =
            strix::finalize_sampled_speculation_cycle(
                model,
                layout,
                preparation,
                finalize);
        expect(
            plan.accepted_candidates == 2 &&
                plan.committed_state_advance == 2 &&
                !plan.restore_required &&
                plan.replay_tokens.empty() &&
                plan.emitted_tokens ==
                    std::vector<std::uint32_t>({0, 1, 2}),
            "all accepted sampled candidates keep verified state and emit target bonus");
    }

    {
        const std::vector<std::uint32_t> candidates{0, 0};
        auto request =
            base_request(candidates);
        const auto preparation =
            strix::prepare_sampled_speculation_cycle(
                model,
                layout,
                request);

        const std::vector<std::vector<double>> target{
            {1.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
        };
        const std::vector<std::vector<double>> draft{
            {1.0, 0.0},
            {1.0, 0.0},
        };
        strix::SampledSpeculationFinalizeRequest finalize;
        finalize.target_probabilities = target;
        finalize.draft_probabilities = draft;
        finalize.rng = rng();
        finalize.token_budget = 3;

        const auto plan =
            strix::finalize_sampled_speculation_cycle(
                model,
                layout,
                preparation,
                finalize);
        expect(
            plan.accepted_candidates == 1 &&
                plan.committed_state_advance == 1 &&
                plan.rejected &&
                plan.used_residual &&
                plan.restore_required &&
                plan.replay_tokens ==
                    std::vector<std::uint32_t>({0}) &&
                plan.emitted_tokens ==
                    std::vector<std::uint32_t>({0, 1}),
            "first sampled rejection restores and replays only accepted prefix");
    }

    {
        const std::vector<std::uint32_t> candidates{0, 1, 2};
        const std::vector<std::uint32_t> eos{1};
        auto request =
            base_request(candidates);
        request.eos_tokens = eos;
        const auto preparation =
            strix::prepare_sampled_speculation_cycle(
                model,
                layout,
                request);

        const std::vector<std::vector<double>> target{
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
            {1.0, 0.0, 0.0},
        };
        const std::vector<std::vector<double>> draft{
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
        };
        strix::SampledSpeculationFinalizeRequest finalize;
        finalize.target_probabilities = target;
        finalize.draft_probabilities = draft;
        finalize.rng = rng();
        finalize.token_budget = 4;
        finalize.eos_tokens = eos;

        const auto plan =
            strix::finalize_sampled_speculation_cycle(
                model,
                layout,
                preparation,
                finalize);
        expect(
            plan.accepted_candidates == 2 &&
                plan.committed_state_advance == 1 &&
                plan.restore_required &&
                plan.replay_tokens ==
                    std::vector<std::uint32_t>({0}) &&
                plan.emitted_tokens ==
                    std::vector<std::uint32_t>({0, 1}),
            "accepted EOS is emitted but not retained in committed target state");
    }

    {
        const std::vector<std::uint32_t> candidates;
        auto request =
            base_request(candidates);
        request.token_budget = 1;
        const auto preparation =
            strix::prepare_sampled_speculation_cycle(
                model,
                layout,
                request);

        const std::vector<std::vector<double>> target{
            {0.0, 1.0},
        };
        const std::vector<std::vector<double>> draft;
        strix::SampledSpeculationFinalizeRequest finalize;
        finalize.target_probabilities = target;
        finalize.draft_probabilities = draft;
        finalize.rng = rng();
        finalize.token_budget = 1;

        const auto plan =
            strix::finalize_sampled_speculation_cycle(
                model,
                layout,
                preparation,
                finalize);
        expect(
            !plan.speculation_used &&
                !plan.snapshot_required &&
                plan.committed_state_advance == 0 &&
                plan.emitted_tokens ==
                    std::vector<std::uint32_t>({1}),
            "depth-zero sampled cycle reduces to one serial target sample");
    }

    {
        const std::vector<std::uint32_t> candidates{0, 1};
        auto request =
            base_request(candidates);
        request.token_budget = 2;
        bool rejected = false;
        try {
            static_cast<void>(
                strix::prepare_sampled_speculation_cycle(
                    model,
                    layout,
                    request));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(
            rejected,
            "sampled cycle candidate depth must reserve one output slot");
    }

    if (failures == 0) {
        std::cout
            << "sampled speculation cycle tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
