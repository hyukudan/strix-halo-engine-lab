#include "strix/serial_request.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool good, const char* message) {
    if (!good) throw std::runtime_error(message);
}
struct Logits final : strix::SerialGreedyStepBackend {
    unsigned calls = 0;
    std::vector<std::uint32_t> consumed;
    std::vector<float> execute(std::uint32_t input, std::uint64_t position) override {
        ++calls;
        consumed.push_back(input);
        std::vector<float> values(8, -7.0F);
        values[static_cast<std::size_t>((position + 2U) % 8U)] = 3.0F;
        return values;
    }
};
struct Tokens final : strix::SerialGreedyStepBackend {
    unsigned calls = 0;
    std::vector<std::uint32_t> consumed;
    bool invalid = false;
    std::vector<float> execute(std::uint32_t, std::uint64_t) override {
        throw std::logic_error("token-only controller requested full logits");
    }
    void execute_state(std::uint32_t input, std::uint64_t) override {
        ++calls;
        consumed.push_back(input);
    }
    std::uint32_t execute_greedy(std::uint32_t input, std::uint64_t position, std::uint64_t vocab) override {
        ++calls;
        consumed.push_back(input);
        return invalid ? static_cast<std::uint32_t>(vocab)
                       : static_cast<std::uint32_t>((position + 2U) % vocab);
    }
};
void same(const strix::SerialGreedyRequestResult& left, const strix::SerialGreedyRequestResult& right) {
    require(left.generated_tokens == right.generated_tokens && left.final_position == right.final_position &&
            left.stop_reason == right.stop_reason && left.steps_executed == right.steps_executed,
            "scalar/full controllers disagree on tokens, consumed position or stop boundary");
}
}
int main() try {
    const std::array<std::uint32_t, 2> prefix{1, 2};
    for (std::uint64_t capacity : {2ULL, 3ULL, 8ULL, 16ULL}) {
        for (std::uint64_t maximum : {0ULL, 1ULL, 2ULL, 7ULL, 20ULL}) {
            for (std::uint32_t eos : {0U, 3U, 7U}) {
                strix::SerialGreedyRequest request{prefix, 8, capacity, maximum, eos};
                Logits full, fallback;
                Tokens token;
                const auto reference = strix::execute_serial_greedy_request(request, full);
                same(reference, strix::execute_serial_greedy_token_request(request, token));
                same(reference, strix::execute_serial_greedy_token_request(request, fallback));
                require(full.calls == token.calls && full.calls == fallback.calls,
                        "scalar controller neither skips nor duplicates state-consuming steps");
                require(full.consumed == token.consumed && full.consumed == fallback.consumed,
                        "scalar feedback consumes the same prefix and generated input tokens");
            }
        }
    }
    strix::SerialGreedyRequest request{prefix, 8, 8, 2, 7};
    Tokens invalid;
    invalid.invalid = true;
    bool rejected = false;
    try { static_cast<void>(strix::execute_serial_greedy_token_request(request, invalid)); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && invalid.calls == 1, "invalid scalar result never reaches a subsequent token step");
    Tokens unused;
    request.context_capacity = 1;
    rejected = false;
    try { static_cast<void>(strix::execute_serial_greedy_token_request(request, unused)); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && unused.calls == 0, "invalid request is rejected before any backend call");
    std::cout << "greedy token request source fixtures PASS\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
