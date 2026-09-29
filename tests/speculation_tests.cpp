#include "strix/speculation.hpp"

#include <cstdint>
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

}  // namespace

int main() {
    {
        const std::vector<std::uint32_t> candidates{10, 20, 30, 40};
        const std::vector<std::uint32_t> target{10, 20, 99, 77, 88};

        strix::GreedyVerificationRequest request;
        request.candidate_tokens = candidates;
        request.target_tokens = target;
        request.token_budget = 5;

        const auto plan =
            strix::verify_greedy_candidates(request);

        expect(plan.raw_matches == 2 &&
                   plan.accepted_matches == 2,
               "greedy verification accepts the contiguous matching prefix");
        expect(plan.valid_tokens ==
                   std::vector<std::uint32_t>({10, 20, 99}),
               "first target mismatch becomes the bonus token");
        expect(plan.candidate_states_to_crop == 2,
               "unmatched candidate state is cropped");
        expect(!plan.all_candidates_matched,
               "partial match is not marked all-matched");
    }

    {
        const std::vector<std::uint32_t> candidates{1, 2, 3};
        const std::vector<std::uint32_t> target{1, 2, 3, 4};

        strix::GreedyVerificationRequest request;
        request.candidate_tokens = candidates;
        request.target_tokens = target;
        request.token_budget = 4;

        const auto plan =
            strix::verify_greedy_candidates(request);

        expect(plan.all_candidates_matched &&
                   plan.raw_matches == 3 &&
                   plan.accepted_matches == 3,
               "all candidate matches are retained");
        expect(plan.valid_tokens ==
                   std::vector<std::uint32_t>({1, 2, 3, 4}),
               "all matches receive one target bonus token");
        expect(plan.candidate_states_to_crop == 0,
               "fully accepted candidate state needs no tail crop");
    }

    {
        const std::vector<std::uint32_t> candidates{5, 6, 7};
        const std::vector<std::uint32_t> target{5, 6, 7, 8};
        const std::vector<std::uint32_t> eos{6};

        strix::GreedyVerificationRequest request;
        request.candidate_tokens = candidates;
        request.target_tokens = target;
        request.token_budget = 4;
        request.eos_tokens = eos;

        const auto plan =
            strix::verify_greedy_candidates(request);

        expect(plan.stopped_on_eos &&
                   plan.valid_tokens ==
                       std::vector<std::uint32_t>({5, 6}),
               "EOS truncates the verified continuation inclusively");
        expect(plan.accepted_matches == 1 &&
                   plan.candidate_states_to_crop == 2,
               "EOS cropping recomputes accepted matches from valid length");
    }

    {
        const std::vector<std::uint32_t> candidates{1, 2, 3, 4};
        const std::vector<std::uint32_t> target{1, 2, 3, 4, 5};

        strix::GreedyVerificationRequest request;
        request.candidate_tokens = candidates;
        request.target_tokens = target;
        request.token_budget = 2;

        const auto plan =
            strix::verify_greedy_candidates(request);

        expect(plan.stopped_on_length &&
                   plan.valid_tokens ==
                       std::vector<std::uint32_t>({1, 2}),
               "length budget truncates the verified continuation");
        expect(plan.accepted_matches == 1 &&
                   plan.candidate_states_to_crop == 3,
               "length cropping recomputes candidate matches");
    }

    {
        const std::vector<std::uint32_t> candidates{10, 20};
        const std::vector<std::uint32_t> target{99, 88, 77};

        strix::GreedyVerificationRequest request;
        request.candidate_tokens = candidates;
        request.target_tokens = target;
        request.token_budget = 3;

        const auto plan =
            strix::verify_greedy_candidates(request);

        expect(plan.raw_matches == 0 &&
                   plan.accepted_matches == 0 &&
                   plan.valid_tokens ==
                       std::vector<std::uint32_t>({99}),
               "zero candidate matches reduces to one target token");
        expect(plan.candidate_states_to_crop == 2,
               "zero-match cycle discards the entire candidate tail");
    }

    bool bad_target = false;
    try {
        const std::vector<std::uint32_t> candidates{1, 2};
        const std::vector<std::uint32_t> target{1, 2};
        strix::GreedyVerificationRequest request;
        request.candidate_tokens = candidates;
        request.target_tokens = target;
        request.token_budget = 3;
        static_cast<void>(
            strix::verify_greedy_candidates(request));
    } catch (const std::invalid_argument&) {
        bad_target = true;
    }
    expect(bad_target,
           "verification requires candidate_length plus one target tokens");

    if (failures == 0) {
        std::cout << "speculation tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
