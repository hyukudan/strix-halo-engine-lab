#include "strix/known_prompt_chunks.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
int failures=0;
void expect(bool condition,const char* text) {
    if(!condition) {std::cerr<<"FAIL: "<<text<<'\n';++failures;}
}
class Rows final : public strix::KnownPromptWindowBackend, public strix::SerialGreedyStepBackend {
public:
    std::uint32_t vocabulary=16;
    unsigned windows=0,serials=0;
    bool fail_partial_window=false,bad_extent=false,nonfinite=false,poisoned=false;
    std::vector<std::pair<std::uint64_t,std::uint32_t>> inputs;
    std::vector<float> values(std::uint32_t input) const {
        std::vector<float> result(vocabulary,-1.0F);result[(input+1U)%vocabulary]=1;
        if(bad_extent) result.pop_back();
        if(nonfinite) result[0]=std::numeric_limits<float>::infinity();
        return result;
    }
    std::vector<float> execute(std::uint32_t token,std::uint64_t position) override {
        ++serials;inputs.emplace_back(position,token);return values(token);
    }
    std::vector<float> execute_window(std::array<std::uint32_t,2> tokens,std::uint64_t position) override {
        ++windows;inputs.emplace_back(position,tokens[0]);
        if(fail_partial_window) {poisoned=true;throw std::runtime_error("partial target window failure");}
        inputs.emplace_back(position+1U,tokens[1]);return values(tokens[1]);
    }
};
} // namespace
int main() {
    // Same canonical pair/tail and existing generation policy, with no prompt
    // recomputation when the new cursor yields between chunks.
    for(const std::uint64_t start:{0U,2U}) for(const std::size_t count:{1U,2U,3U,4U,5U}) {
        const std::vector<std::uint32_t> all{1,2,3,4,5};
        const std::span<const std::uint32_t> prompt(all.data(),count);
        Rows ordinary,chunked;
        const auto expected=strix::execute_known_prompt_greedy_request({prompt,start,16,32,3,15,{}},ordinary,ordinary);
        strix::KnownPromptChunkCursor cursor({prompt,start,16,32,3,15},chunked,chunked);
        std::size_t completed=0;
        while(cursor.advance()) {
            ++completed;
            const auto trace=chunked.inputs;
            const auto frontier=cursor.frontier();
            const std::vector<float> retained(frontier.logits.begin(),frontier.logits.end());
            expect(frontier.valid && frontier.state_position==start+chunked.inputs.size() && !frontier.selection_emitted &&
                cursor.progress().emitted_tokens==0 && chunked.inputs==trace &&
                std::vector<float>(cursor.frontier().logits.begin(),cursor.frontier().logits.end())==retained,
                "yield retains completed logits and does not emit or recompute known inputs");
        }
        expect(completed==(count+1U)/2U && cursor.progress().windows_executed==count/2U &&
            cursor.progress().serial_tail_steps==count%2U,"each advance is one real pair or final odd row");
        const auto& actual=cursor.finish_greedy();
        expect(expected.generation && actual.generated_tokens==expected.generation->generated_tokens &&
            actual.final_position==expected.generation->final_position && actual.stop_reason==expected.generation->stop_reason &&
            chunked.inputs==ordinary.inputs,"chunked route delegates identical serial generation and token/input ordering");
        expect(cursor.progress().consumed_prompt_tokens==count && cursor.progress().consumed_generation_tokens==actual.steps_executed &&
            cursor.progress().emitted_tokens==actual.generated_tokens.size() &&
            cursor.progress().bonus_pending==actual.generated_tokens.back() &&
            cursor.frontier().unconsumed_selection==actual.generated_tokens.back(),
            "prompt inputs, generation inputs, emissions and unconsumed bonus remain distinct");
        expect(!actual.steps_executed || cursor.frontier().logits.empty(),"old prompt logits never masquerade as new generation frontier");
        bool rejected=false;
        try {(void)cursor.finish_greedy();} catch(const std::logic_error&) {rejected=true;}
        expect(rejected,"finished generation cannot execute or emit twice");
    }
    {
        Rows rows;const std::vector<std::uint32_t> prompt{1,2};
        strix::KnownPromptChunkCursor cursor({prompt,2,16,8,1,15},rows,rows);
        cursor.cancel();
        expect(!cursor.advance() && rows.inputs.empty() && cursor.progress().next_position==2 &&
            cursor.progress().cancelled && !cursor.frontier().valid,"cancel before first chunk neither consumes nor fabricates a frontier");
    }
    {
        Rows rows;const std::vector<std::uint32_t> prompt{1,2,3};
        strix::KnownPromptChunkCursor cursor({prompt,0,16,8,0,15},rows,rows);
        expect(cursor.advance() && cursor.progress().consumed_prompt_tokens==2,"first pair completes atomically");
        const auto retained=cursor.frontier().logits.data();cursor.cancel();
        expect(!cursor.advance() && rows.inputs.size()==2 && cursor.progress().cancelled &&
            cursor.frontier().logits.data()==retained,"cancel before odd tail is sticky and retains quiescent frontier storage");
        bool rejected=false;try {(void)cursor.finish_greedy();} catch(const std::logic_error&) {rejected=true;}
        expect(rejected && rows.inputs.size()==2,"cancelled prefill cannot start generation");
    }
    {
        Rows rows;const std::vector<std::uint32_t> prompt{2};
        strix::KnownPromptChunkCursor cursor({prompt,0,16,4,4,3},rows,rows);
        (void)cursor.advance();const auto& result=cursor.finish_greedy();
        expect(result.generated_tokens==std::vector<std::uint32_t>{3} && rows.inputs.size()==1 &&
            cursor.progress().bonus_pending==3U && cursor.progress().consumed_generation_tokens==0 &&
            result.stop_reason==strix::SerialRequestStopReason::eos,"first generated EOS remains emitted but unconsumed");
    }
    {
        Rows rows;const std::vector<std::uint32_t> prompt{1,2};
        strix::KnownPromptChunkCursor cursor({prompt,2,16,4,4,15},rows,rows);
        (void)cursor.advance();const auto& result=cursor.finish_greedy();
        expect(result.generated_tokens==std::vector<std::uint32_t>{3} && result.final_position==4 && rows.inputs.size()==2 &&
            result.stop_reason==strix::SerialRequestStopReason::context_capacity,"exact-capacity prompt emits pending selection without another input");
    }
    {
        Rows rows;const std::vector<std::uint32_t> prompt{1,15,2};
        strix::KnownPromptChunkCursor cursor({prompt,0,16,4,0,15},rows,rows);
        while(cursor.advance()) {}
        const auto& result=cursor.finish_greedy();
        expect(rows.inputs.size()==3 && result.generated_tokens.empty() && !cursor.progress().bonus_pending &&
            cursor.progress().next_position==3 && !cursor.frontier().logits.empty(),
            "interior known EOS and maxnew0 still consume complete prompt without publishing a bonus");
    }
    {
        const std::vector<std::uint32_t> good{1,2},bad{1,16};
        const std::vector<strix::KnownPromptChunkRequest> invalid{{{},0,16,8,1,15},{bad,0,16,8,1,15},
            {good,9,16,8,1,15},{good,7,16,8,1,15},{good,0,0,8,1,0},{good,0,16,8,1,16},
            {good,std::numeric_limits<std::uint64_t>::max()-1U,16,std::numeric_limits<std::uint64_t>::max(),1,15}};
        for(const auto& request:invalid) {
            Rows rows;bool rejected=false;
            try {strix::KnownPromptChunkCursor cursor(request,rows,rows);} catch(const std::invalid_argument&) {rejected=true;}
            expect(rejected && rows.inputs.empty(),"entire malformed suffix rejects before any target input");
        }
    }
    for(const unsigned failure:{0U,1U,2U}) {
        Rows rows;const std::vector<std::uint32_t> prompt{1,2};
        rows.fail_partial_window=failure==0;rows.bad_extent=failure==1;rows.nonfinite=failure==2;
        strix::KnownPromptChunkCursor cursor({prompt,0,16,8,1,15},rows,rows);
        bool threw=false;try {(void)cursor.advance();} catch(const std::exception&) {threw=true;}
        const auto calls=rows.windows;bool blocked=false;
        try {(void)cursor.advance();} catch(const std::logic_error&) {blocked=true;}
        expect(threw && blocked && cursor.progress().failed && rows.windows==calls && cursor.frontier().logits.empty() && !cursor.frontier().valid,
            "partial/native/output failure forbids resume and invalidates retained frontier without fake rollback");
        if(failure==0) expect(rows.poisoned && rows.inputs.size()==1 && cursor.progress().consumed_prompt_tokens==0,
            "failed-call counters do not pretend partially consumed opaque target state was restored");
    }
    if(!failures) std::cout<<"known prompt chunk tests passed\n";
    return failures?1:0;
}
