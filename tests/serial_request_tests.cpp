#include "strix/serial_request.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class ScriptedBackend final
    : public strix::SerialGreedyStepBackend {
public:
    std::vector<std::vector<float>> outputs;
    std::vector<std::pair<std::uint32_t, std::uint64_t>> calls;

    std::vector<float> execute(
        std::uint32_t token,
        std::uint64_t position) override {
        calls.emplace_back(
            token,
            position);
        const auto index =
            calls.size() - 1U;
        if (index >= outputs.size()) {
            throw std::runtime_error(
                "scripted backend exhausted");
        }
        return outputs[index];
    }
};

std::vector<float> logits(
    std::size_t vocab,
    std::uint32_t best,
    float best_value = 10.0F) {
    std::vector<float> values(
        vocab,
        -10.0F);
    values[best] =
        best_value;
    return values;
}

}  // namespace

int main() {
    {
        const std::vector<std::uint32_t> prefix{
            1, 2,
        };
        ScriptedBackend backend;
        backend.outputs = {
            logits(8, 5),
            logits(8, 6),
            logits(8, 7),
        };

        strix::SerialGreedyRequest request;
        request.prefix_tokens =
            prefix;
        request.vocab_size = 8;
        request.context_capacity = 8;
        request.max_new_tokens = 2;
        request.eos_token_id = 7;

        const auto result =
            strix::execute_serial_greedy_request(
                request,
                backend);

        expect(
            result.generated_tokens ==
                std::vector<std::uint32_t>({6, 7}),
            "greedy request samples from last prefix logits then generated logits");
        expect(
            backend.calls ==
                std::vector<std::pair<std::uint32_t, std::uint64_t>>({
                    {1, 0},
                    {2, 1},
                    {6, 2},
                }),
            "greedy request prefills prefix then advances generated token position");
        expect(
            result.steps_executed == 3 &&
                result.final_position == 3 &&
                result.stop_reason ==
                    strix::SerialRequestStopReason::eos,
            "EOS stops without executing EOS as a new input token");
    }

    {
        const std::vector<std::uint32_t> prefix{3};
        ScriptedBackend backend;
        auto tie = logits(6, 4);
        tie[2] = tie[4];
        backend.outputs = {tie};

        strix::SerialGreedyRequest request;
        request.prefix_tokens = prefix;
        request.vocab_size = 6;
        request.context_capacity = 4;
        request.max_new_tokens = 1;
        request.eos_token_id = 5;

        const auto result =
            strix::execute_serial_greedy_request(
                request,
                backend);
        expect(
            result.generated_tokens ==
                std::vector<std::uint32_t>({2}),
            "greedy ties resolve to the lower token index");
        expect(
            result.stop_reason ==
                strix::SerialRequestStopReason::
                    max_new_tokens,
            "token budget stops after requested generation count");
    }

    {
        const std::vector<std::uint32_t> prefix{
            1, 2,
        };
        ScriptedBackend backend;
        backend.outputs = {
            logits(5, 3),
            logits(5, 3),
        };

        strix::SerialGreedyRequest request;
        request.prefix_tokens = prefix;
        request.vocab_size = 5;
        request.context_capacity = 2;
        request.max_new_tokens = 3;
        request.eos_token_id = 4;

        const auto result =
            strix::execute_serial_greedy_request(
                request,
                backend);
        expect(
            result.generated_tokens ==
                std::vector<std::uint32_t>({3}) &&
                result.stop_reason ==
                    strix::SerialRequestStopReason::
                        context_capacity &&
                result.steps_executed == 2,
            "full context can expose one next token but cannot execute another step");
    }

    {
        const std::vector<std::uint32_t> prefix{1};
        ScriptedBackend backend;
        auto invalid =
            logits(4, 2);
        invalid[3] =
            std::numeric_limits<float>::quiet_NaN();
        backend.outputs = {invalid};

        strix::SerialGreedyRequest request;
        request.prefix_tokens = prefix;
        request.vocab_size = 4;
        request.context_capacity = 4;
        request.max_new_tokens = 1;
        request.eos_token_id = 3;

        bool rejected = false;
        try {
            static_cast<void>(
                strix::execute_serial_greedy_request(
                    request,
                    backend));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(
            rejected,
            "greedy request rejects non-finite logits before sampling");
    }

    {
        ScriptedBackend backend;
        backend.outputs = {
            logits(7, 6),
        };

        const auto initial =
            logits(7, 4);
        strix::SerialGreedyContinuationRequest continuation;
        continuation.initial_logits =
            initial;
        continuation.start_position = 3;
        continuation.vocab_size = 7;
        continuation.context_capacity = 8;
        continuation.max_new_tokens = 2;
        continuation.eos_token_id = 6;

        const auto result =
            strix::execute_serial_greedy_continuation(
                continuation,
                backend);
        expect(
            result.generated_tokens ==
                std::vector<std::uint32_t>({4, 6}),
            "continuation samples directly from cached prefix logits");
        expect(
            backend.calls ==
                std::vector<std::pair<std::uint32_t, std::uint64_t>>({
                    {4, 3},
                }),
            "continuation skips prefix prefill and starts at restored position");
        expect(
            result.steps_executed == 1 &&
                result.final_position == 4 &&
                result.stop_reason ==
                    strix::SerialRequestStopReason::eos,
            "continuation tracks only post-prefix execution steps");
    }

    if (failures == 0) {
        std::cout
            << "serial request tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
