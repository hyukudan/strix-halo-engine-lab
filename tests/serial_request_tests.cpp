#include "strix/serial_request.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
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

struct PromptCall {
    char kind;
    std::uint32_t first;
    std::uint32_t second;
    std::uint64_t position;
    bool operator==(const PromptCall&) const = default;
};

// Position-derived full rows make a wrong tail/frontier observable without
// sharing the production scheduler's pair-selection logic.
struct PromptScript {
    std::vector<PromptCall> calls;
    std::size_t window_calls = 0;
    std::size_t throw_window = 0;
    bool return_eos = false;

    std::vector<float> row(std::uint64_t position) const {
        return logits(8, return_eos ? 7U :
            static_cast<std::uint32_t>((position + 2U) % 7U));
    }
};

class PromptWindows final : public strix::KnownPromptWindowBackend {
public:
    explicit PromptWindows(PromptScript& script) : script_(script) {}
    std::vector<float> execute_window(std::array<std::uint32_t, 2> tokens,
        std::uint64_t position) override {
        script_.calls.push_back({'W', tokens[0], tokens[1], position});
        ++script_.window_calls;
        if (script_.throw_window && script_.window_calls == script_.throw_window)
            throw std::runtime_error("window failure witness");
        return script_.row(position + 1U);
    }
private:
    PromptScript& script_;
};

class PromptSerial final : public strix::SerialGreedyStepBackend {
public:
    explicit PromptSerial(PromptScript& script) : script_(script) {}
    std::vector<float> execute(std::uint32_t token, std::uint64_t position) override {
        script_.calls.push_back({'S', token, 0, position});
        return script_.row(position);
    }
    void execute_state(std::uint32_t token, std::uint64_t position) override {
        script_.calls.push_back({'X', token, 0, position});
        throw std::logic_error("known prompt must retain full heads");
    }
    std::uint32_t execute_greedy(std::uint32_t token, std::uint64_t position,
        std::uint64_t) override {
        script_.calls.push_back({'G', token, 0, position});
        throw std::logic_error("known prompt continuation must retain full logits");
    }
private:
    PromptScript& script_;
};

strix::KnownPromptGreedyRequest prompt_request(std::span<const std::uint32_t> tokens) {
    strix::KnownPromptGreedyRequest request;
    request.prompt_tokens = tokens;
    request.vocab_size = 8;
    request.context_capacity = 32;
    request.max_new_tokens = 2;
    request.eos_token_id = 7;
    return request;
}

void known_prompt_tests() {
    const std::vector<std::vector<std::uint32_t>> prompts{
        {1}, {1, 2}, {1, 2, 3, 4}, {1, 2, 7, 3, 4}};
    const std::vector<std::vector<PromptCall>> expected{
        {{'S', 1, 0, 0}, {'S', 2, 0, 1}},
        {{'W', 1, 2, 0}, {'S', 3, 0, 2}},
        {{'W', 1, 2, 0}, {'W', 3, 4, 2}, {'S', 5, 0, 4}},
        {{'W', 1, 2, 0}, {'W', 7, 3, 2}, {'S', 4, 0, 4}, {'S', 6, 0, 5}}};
    for (std::size_t index = 0; index < prompts.size(); ++index) {
        PromptScript script;
        PromptWindows windows(script);
        PromptSerial serial(script);
        auto request = prompt_request(prompts[index]);
        std::vector<std::uint64_t> boundaries;
        request.stop_between_windows = [&](std::uint64_t position) {
            boundaries.push_back(position);
            return false;
        };
        const auto result = strix::execute_known_prompt_greedy_request(request, windows, serial);
        const auto count = static_cast<std::uint64_t>(prompts[index].size());
        expect(script.calls == expected[index], "known prompt exact pair/tail/full-head continuation call order");
        expect(result.consumed_prompt_tokens == count && result.windows_executed == count / 2U &&
            result.serial_prompt_steps == count % 2U, "known prompt reports only actually consumed prompt work");
        expect(result.generation && result.generation->steps_executed == 1U &&
            result.generation->final_position == count + 1U &&
            result.generation->stop_reason == strix::SerialRequestStopReason::max_new_tokens &&
            result.generation->generated_tokens == std::vector<std::uint32_t>{
                static_cast<std::uint32_t>((count + 1U) % 7U),
                static_cast<std::uint32_t>((count + 2U) % 7U)},
            "generation begins from terminal prompt logits and leaves last output unconsumed");
        expect(boundaries == (count >= 4U ? std::vector<std::uint64_t>{2} :
            std::vector<std::uint64_t>{}), "predicate runs only before a subsequent complete pair");
    }
    {
        PromptScript script;
        PromptWindows windows(script);
        PromptSerial serial(script);
        auto request = prompt_request(prompts[2]);
        request.start_position = 2;
        const auto result = strix::execute_known_prompt_greedy_request(request, windows, serial);
        expect(script.calls == std::vector<PromptCall>{{'W', 1, 2, 2}, {'W', 3, 4, 4}, {'S', 0, 0, 6}} &&
            result.generation && result.generation->final_position == 7U,
            "nonzero start offsets both known windows and continuation without replaying prefix");
    }
    {
        PromptScript script;
        script.return_eos = true;
        PromptWindows windows(script);
        PromptSerial serial(script);
        const auto result = strix::execute_known_prompt_greedy_request(prompt_request(prompts[3]), windows, serial);
        expect(script.calls == std::vector<PromptCall>{{'W', 1, 2, 0}, {'W', 7, 3, 2}, {'S', 4, 0, 4}} &&
            result.consumed_prompt_tokens == 5U && result.generation &&
            result.generation->generated_tokens == std::vector<std::uint32_t>{7} &&
            result.generation->stop_reason == strix::SerialRequestStopReason::eos &&
            result.generation->steps_executed == 0U && result.generation->final_position == 5U,
            "interior prompt EOS is data; first generated EOS is emitted and never consumed");
    }
    {
        PromptScript script;
        PromptWindows windows(script);
        PromptSerial serial(script);
        auto request = prompt_request(prompts[3]);
        request.max_new_tokens = 0;
        const auto result = strix::execute_known_prompt_greedy_request(request, windows, serial);
        expect(script.calls == std::vector<PromptCall>{{'W', 1, 2, 0}, {'W', 7, 3, 2}, {'S', 4, 0, 4}} &&
            result.consumed_prompt_tokens == 5U && result.generation &&
            result.generation->generated_tokens.empty() && result.generation->steps_executed == 0U &&
            result.generation->final_position == 5U,
            "zero generation budget still consumes the entire known prompt and returns continuation result");
    }
    {
        PromptScript script;
        PromptWindows windows(script);
        PromptSerial serial(script);
        auto request = prompt_request(prompts[1]);
        request.start_position = 2;
        request.context_capacity = 4;
        request.max_new_tokens = 5;
        const auto result = strix::execute_known_prompt_greedy_request(request, windows, serial);
        expect(script.calls == std::vector<PromptCall>{{'W', 1, 2, 2}} && result.generation &&
            result.generation->generated_tokens == std::vector<std::uint32_t>{5} &&
            result.generation->stop_reason == strix::SerialRequestStopReason::context_capacity &&
            result.generation->steps_executed == 0U && result.generation->final_position == 4U,
            "exact capacity exposes one terminal prediction without executing it");
    }
    {
        const std::vector<std::uint32_t> three{1, 2, 3};
        PromptScript script;
        PromptWindows windows(script);
        PromptSerial serial(script);
        auto request = prompt_request(three);
        request.stop_between_windows = [](std::uint64_t) -> bool {
            throw std::logic_error("no boundary before the tail or continuation");
        };
        const auto result = strix::execute_known_prompt_greedy_request(request, windows, serial);
        expect(result.generation && script.calls == std::vector<PromptCall>{
            {'W', 1, 2, 0}, {'S', 3, 0, 2}, {'S', 4, 0, 3}},
            "N3 never queries cancellation before tail or generation");
    }
    {
        PromptScript uninterrupted;
        PromptWindows uw(uninterrupted);
        PromptSerial us(uninterrupted);
        auto request = prompt_request(prompts[3]);
        request.max_new_tokens = 4;
        const auto reference = strix::execute_known_prompt_greedy_request(request, uw, us);
        PromptScript resumed;
        PromptWindows rw(resumed);
        PromptSerial rs(resumed);
        std::vector<std::uint64_t> stops;
        request.stop_between_windows = [&](std::uint64_t position) { stops.push_back(position); return true; };
        const auto cut = strix::execute_known_prompt_greedy_request(request, rw, rs);
        expect(!cut.generation && cut.consumed_prompt_tokens == 2U && cut.windows_executed == 1U &&
            cut.serial_prompt_steps == 0U && stops == std::vector<std::uint64_t>{2} &&
            resumed.calls == std::vector<PromptCall>{{'W', 1, 2, 0}},
            "cancellation observes first completed window and returns before second without generation");
        request.prompt_tokens = std::span<const std::uint32_t>(prompts[3]).subspan(2);
        request.start_position = 2;
        const auto suffix = strix::execute_known_prompt_greedy_request(request, rw, rs);
        expect(resumed.calls == uninterrupted.calls && stops == std::vector<std::uint64_t>{2} &&
            suffix.consumed_prompt_tokens + cut.consumed_prompt_tokens == 5U &&
            suffix.generation && reference.generation &&
            suffix.generation->generated_tokens == reference.generation->generated_tokens &&
            suffix.generation->steps_executed == reference.generation->steps_executed &&
            suffix.generation->final_position == reference.generation->final_position &&
            suffix.generation->stop_reason == reference.generation->stop_reason,
            "stop/resume keeps exact combined backend calls, terminal logits and controller frontier");
    }
    {
        auto valid = prompt_request(prompts[2]);
        auto invalid_requests = std::vector<strix::KnownPromptGreedyRequest>(10, valid);
        invalid_requests[0].prompt_tokens = {};
        invalid_requests[1].context_capacity = 3;
        invalid_requests[2].vocab_size = 0;
        invalid_requests[3].context_capacity = 0;
        invalid_requests[4].eos_token_id = 8;
        invalid_requests[5].start_position = std::numeric_limits<std::uint64_t>::max() - 1U;
        invalid_requests[5].context_capacity = std::numeric_limits<std::uint64_t>::max();
        const std::vector<std::uint32_t> early{8, 1, 2, 3}, late{1, 2, 3, 8};
        invalid_requests[6].prompt_tokens = early;
        invalid_requests[7].prompt_tokens = late;
        invalid_requests[8].start_position = 33;
        invalid_requests[9].vocab_size = static_cast<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max()) + 2ULL;
        for (auto request : invalid_requests) {
            PromptScript script;
            PromptWindows windows(script);
            PromptSerial serial(script);
            std::size_t predicates = 0;
            request.stop_between_windows = [&](std::uint64_t) { ++predicates; return false; };
            bool rejected = false;
            try { static_cast<void>(strix::execute_known_prompt_greedy_request(request, windows, serial)); }
            catch (const std::invalid_argument&) { rejected = true; }
            catch (const std::overflow_error&) { rejected = true; }
            expect(rejected && script.calls.empty() && predicates == 0U,
                "complete suffix/geometry/EOS/overflow admission precedes all callbacks including late invalid token");
        }
    }
    {
        PromptScript script;
        script.throw_window = 2;
        PromptWindows windows(script);
        PromptSerial serial(script);
        std::size_t predicates = 0;
        auto request = prompt_request(prompts[3]);
        request.stop_between_windows = [&](std::uint64_t) { ++predicates; return false; };
        bool failed = false;
        try { static_cast<void>(strix::execute_known_prompt_greedy_request(request, windows, serial)); }
        catch (const std::runtime_error&) { failed = true; }
        expect(failed && predicates == 1U && script.calls == std::vector<PromptCall>{
            {'W', 1, 2, 0}, {'W', 7, 3, 2}},
            "second window exception propagates without retry, tail or continuation callbacks");
    }
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

    known_prompt_tests();

    if (failures == 0) {
        std::cout
            << "serial request tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
