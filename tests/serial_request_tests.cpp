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

class PrefixBackend final : public strix::SerialGreedyStepBackend {
public:
    std::optional<strix::GreedyPrefixRestore> restored;
    std::vector<std::pair<std::uint32_t,std::uint64_t>> calls;
    std::vector<std::uint32_t> captured;
    std::uint64_t position=0;
    std::uint32_t pending=0, restores=0, captures=0;
    bool fail_restore=false,fail_capture=false;
    std::vector<float> execute(std::uint32_t token,std::uint64_t p) override {
        return logits(8,execute_greedy(token,p,8));
    }
    std::uint32_t execute_greedy(std::uint32_t token,std::uint64_t p,std::uint64_t) override {
        expect(p==position,"restored cursor never replays an already consumed input");
        calls.emplace_back(token,p);++position;return(token+1U)%8U;
    }
    void execute_state(std::uint32_t token,std::uint64_t p) override { (void)execute_greedy(token,p,8); }
    std::optional<strix::GreedyPrefixRestore> try_restore_greedy_prefix(const strix::SerialGreedyRequest&) override {
        ++restores;if(fail_restore)throw std::runtime_error("contained restore failure");
        if(restored)position=restored->consumed_prefix;
        return restored;
    }
    void capture_greedy_prefix(std::span<const std::uint32_t> prefix,std::uint32_t next,std::uint64_t vocab) override {
        ++captures;if(fail_capture)throw std::runtime_error("contained capture failure");
        expect(position==prefix.size() && next<vocab,"prefix capture occurs at the consumed prompt/pending token frontier");
        captured.assign(prefix.begin(),prefix.end());pending=next;
    }
};

void prefix_cursor_tests() {
    const std::vector<std::uint32_t> prefix{1,2,3};
    strix::SerialGreedyRequest request{prefix,8,16,3,7};
    PrefixBackend baseline;
    const auto expected=strix::execute_serial_greedy_token_request(request,baseline);
    expect(expected.generated_tokens==std::vector<std::uint32_t>{4,5,6} && expected.steps_executed==5U &&
        baseline.restores==1U && baseline.captures==1U && baseline.captured==prefix && baseline.pending==4U,
        "cache miss preserves greedy calls and captures the unconsumed first output exactly once");
    for(const std::uint64_t consumed:{1ULL,2ULL,3ULL}) {
        PrefixBackend backend;
        backend.restored=strix::GreedyPrefixRestore{consumed,consumed==prefix.size()?std::optional<std::uint32_t>{4}:std::nullopt};
        strix::SerialGreedyTokenCursor cursor(request);
        while(!cursor.done())cursor.advance(backend,1);
        const auto result=cursor.take_result();
        expect(result.generated_tokens==expected.generated_tokens && result.final_position==expected.final_position &&
            result.stop_reason==expected.stop_reason && result.restored_prefix_tokens==consumed &&
            result.steps_executed+consumed==expected.steps_executed && backend.restores==1U,
            "proper/exact restored prefixes retain greedy output, frontier and actual execution counts");
        expect(backend.calls.size()==result.steps_executed && backend.calls.front().second==consumed &&
            backend.calls.back()==std::pair<std::uint32_t,std::uint64_t>{5,4},
            "cached pending token is consumed once only when another output is requested");
        expect(backend.captures==(consumed==prefix.size()?0U:1U),"exact hits do not capture another identical snapshot");
    }
    for(const unsigned boundary:{0U,1U,2U,3U}) {
        auto bounded=request;
        PrefixBackend backend;backend.restored=strix::GreedyPrefixRestore{3,4};
        if(boundary==0U)bounded.max_new_tokens=0;
        if(boundary==1U)bounded.max_new_tokens=1;
        if(boundary==2U)bounded.context_capacity=prefix.size();
        if(boundary==3U)backend.restored->pending_token=7;
        const auto result=strix::execute_serial_greedy_token_request(bounded,backend);
        expect(result.steps_executed==0U && result.restored_prefix_tokens==3U && result.final_position==3U &&
            backend.calls.empty() && backend.captures==0U && result.generated_tokens.size()==(boundary==0U?0U:1U),
            "zero budget/EOS/output budget/capacity leave the cached final output unconsumed");
    }
    {
        PrefixBackend backend;backend.restored=strix::GreedyPrefixRestore{3,4};
        strix::SerialGreedyTokenCursor cursor(request);cursor.advance(backend);cursor.cancel();
        expect(cursor.cancelled() && cursor.result().generated_tokens==std::vector<std::uint32_t>{4} &&
            backend.calls.empty() && backend.position==3U,"cancellation after a hit retains its unconsumed bonus");
        PrefixBackend before;strix::SerialGreedyTokenCursor cancelled(request);cancelled.cancel();cancelled.advance(before);
        expect(before.restores==0U && before.calls.empty(),"pre-start cancellation cannot restore target state");
    }
    for(const auto bad:std::array<strix::GreedyPrefixRestore,5>{{{0,std::nullopt},{4,std::nullopt},{3,std::nullopt},{1,4},{3,8}}}) {
        PrefixBackend backend;backend.restored=bad;strix::SerialGreedyTokenCursor cursor(request);
        bool rejected=false;try{cursor.advance(backend);}catch(const std::invalid_argument&){rejected=true;}
        bool sticky=false;try{cursor.advance(backend);}catch(const std::logic_error&){sticky=true;}
        expect(rejected && sticky && cursor.failed() && backend.calls.empty() && backend.restores==1U,
            "malformed restore frontiers fail without target calls or a second restore attempt");
    }
    for(const bool capture:{false,true}) {
        PrefixBackend backend;backend.fail_restore=!capture;backend.fail_capture=capture;
        strix::SerialGreedyTokenCursor cursor(request);
        bool failed=false;try{cursor.advance(backend,8);}catch(const std::runtime_error&){failed=true;}
        expect(failed && cursor.failed() && cursor.result().generated_tokens.empty() &&
            backend.calls.size()==(capture?3U:0U),"contained cache transfer errors stop before any output publication");
    }
    {
        const std::array<std::uint32_t,1> short_prefix{1};
        PrefixBackend backend;
        strix::SerialGreedyTokenCursor cursor({short_prefix,8,UINT64_MAX,UINT64_MAX,7});
        bool rejected=false;try{cursor.advance(backend);}catch(const std::length_error&){rejected=true;}
        expect(rejected && backend.restores==0U && backend.calls.empty(),"response capacity is reserved before any cache restoration");
    }
}

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

void prepared_target_cursor_tests() {
    const std::vector<std::uint32_t> prefix{1,2,3};
    for(const std::uint64_t restored:{0ULL,1ULL,2ULL,3ULL})
    for(const std::uint64_t budget:{0ULL,1ULL,3ULL})
    for(const std::uint64_t capacity:{3ULL,16ULL})
    for(const std::uint32_t eos:{4U,7U}) {
        const strix::SerialGreedyRequest request{prefix,8,capacity,budget,eos};
        PrefixBackend ordinary,split;
        if(restored) ordinary.restored=split.restored=strix::GreedyPrefixRestore{restored,
            restored==prefix.size()?std::optional<std::uint32_t>{4U}:std::nullopt};
        const auto expected=strix::execute_serial_greedy_token_request(request,ordinary);
        strix::SerialGreedyTokenCursor cursor(request);
        while(!cursor.done()) {
            const auto calls=split.calls;
            const auto input=cursor.prepare_target(split);
            expect(split.calls==calls,"preparing a cohort input never executes its target");
            if(!input) continue; // Exact cached frontier needs no target input.
            expect(input->position==cursor.result().final_position && input->vocab_size==8U && input->ticket,
                "prepared input carries its original position/vocabulary and nonzero lease");
            bool twice=false;try{(void)cursor.prepare_target(split);}catch(const std::logic_error&){twice=true;}
            expect(twice && split.calls==calls,"second preparation cannot duplicate a leased target input");
            std::optional<std::uint32_t> selected;
            if(input->state_only) split.execute_state(input->token,input->position);
            else selected=split.execute_greedy(input->token,input->position,input->vocab_size);
            cursor.accept_target(split,*input,selected);
        }
        const auto result=cursor.take_result();
        expect(result.generated_tokens==expected.generated_tokens && result.steps_executed==expected.steps_executed &&
            result.final_position==expected.final_position && result.stop_reason==expected.stop_reason &&
            result.restored_prefix_tokens==expected.restored_prefix_tokens && split.calls==ordinary.calls &&
            split.captures==ordinary.captures && split.restores==ordinary.restores,
            "split target completion preserves cache/prefill/greedy/EOS/capacity/budget and call ordering");
    }
    for(const unsigned mutation:{0U,1U,2U,3U,4U}) {
        PrefixBackend backend;strix::SerialGreedyTokenCursor cursor({prefix,8,16,3,7});
        const auto lease=cursor.prepare_target(backend);auto forged=*lease;
        if(mutation==0U) ++forged.token;
        if(mutation==1U) ++forged.position;
        if(mutation==2U) ++forged.ticket;
        if(mutation==3U) forged.state_only=false;
        if(mutation==4U) ++forged.vocab_size;
        bool rejected=false;try{cursor.accept_target(backend,forged,std::nullopt);}catch(const std::invalid_argument&){rejected=true;}
        expect(rejected && cursor.failed() && backend.calls.empty() && !cursor.result().steps_executed,
            "forged target lease fails before cursor publication and cannot resume");
    }
    {
        PrefixBackend a,b;strix::SerialGreedyTokenCursor ca({prefix,8,16,3,7}),cb({prefix,8,16,3,7});
        const auto la=ca.prepare_target(a),lb=cb.prepare_target(b);
        expect(la->ticket==lb->ticket && la->token==lb->token,"foreign lease fixture has identical public value fields");
        bool rejected=false;try{ca.accept_target(a,*lb,std::nullopt);}catch(const std::invalid_argument&){rejected=true;}
        expect(rejected && ca.failed() && a.calls.empty(),"private cursor ownership prevents borrowing another request's equal lease");
    }
    {
        PrefixBackend original,foreign;strix::SerialGreedyTokenCursor cursor({prefix,8,16,3,7});
        const auto lease=cursor.prepare_target(original);
        bool rejected=false;try{cursor.accept_target(foreign,*lease,std::nullopt);}catch(const std::invalid_argument&){rejected=true;}
        expect(rejected && cursor.failed() && original.calls.empty() && foreign.calls.empty(),
            "target lease completion cannot switch its borrowed backend/capture owner");
    }
    {
        PrefixBackend backend;strix::SerialGreedyTokenCursor cursor({prefix,8,16,3,7});
        const auto lease=cursor.prepare_target(backend);
        cursor.cancel_prepared_target(*lease);
        const auto result=cursor.take_result();
        expect(cursor.cancelled() && !cursor.failed() && backend.calls.empty() && result.generated_tokens.empty() &&
            result.steps_executed==0U && result.final_position==0U,
            "pre-enqueue cancellation abandons exactly one lease without fake completed work");
    }
    for(const bool state_only:{false,true}) {
        const std::vector<std::uint32_t> prompt=state_only?prefix:std::vector<std::uint32_t>{1U};
        PrefixBackend backend;strix::SerialGreedyTokenCursor cursor({prompt,8,16,3,7});
        const auto lease=cursor.prepare_target(backend);std::optional<std::uint32_t> selected;
        if(lease->state_only)backend.execute_state(lease->token,lease->position);
        else selected=backend.execute_greedy(lease->token,lease->position,lease->vocab_size);
        cursor.cancel();cursor.accept_target(backend,*lease,selected);
        const auto result=cursor.take_result();
        expect(result.steps_executed==1U && result.final_position==1U && result.generated_tokens.empty() &&
            backend.position==1U && backend.captures==0U && cursor.cancelled() && !cursor.failed(),
            "cancellation after submission accounts completed state but emits/captures no pending result");
    }
    {
        PrefixBackend backend;strix::SerialGreedyTokenCursor cursor({prefix,8,16,3,7});
        const auto lease=cursor.prepare_target(backend);backend.execute_state(lease->token,lease->position);
        cursor.accept_target(backend,*lease,std::nullopt);
        bool rejected=false;try{cursor.accept_target(backend,*lease,std::nullopt);}catch(const std::invalid_argument&){rejected=true;}
        expect(rejected && cursor.failed() && cursor.result().steps_executed==1U && backend.calls.size()==1U,
            "completed target lease cannot publish or consume twice");
    }
    for(const bool state_only:{false,true}) {
        const std::vector<std::uint32_t> prompt=state_only?prefix:std::vector<std::uint32_t>{1U};
        PrefixBackend backend;strix::SerialGreedyTokenCursor cursor({prompt,8,16,3,7});const auto lease=cursor.prepare_target(backend);
        bool rejected=false;try{cursor.accept_target(backend,*lease,state_only?std::optional<std::uint32_t>{2U}:std::nullopt);}
        catch(const std::invalid_argument&){rejected=true;}
        expect(rejected && cursor.failed() && backend.calls.empty(),"target completion requires exactly the selection intent leased");
    }
}

}  // namespace

int main() {
    prefix_cursor_tests();
    prepared_target_cursor_tests();
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
