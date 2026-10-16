#include "strix/generation_output.hpp"

#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
int failures=0;
void expect(bool condition,const char* text) {
    if(!condition) {std::cerr<<"FAIL: "<<text<<'\n';++failures;}
}
class Backend final : public strix::SerialGreedyStepBackend {
public:
    std::vector<float> values{-0.0F,+0.0F,-1.0F};
    unsigned full_calls=0,scalar_calls=0,state_calls=0;
    std::uint32_t scalar=0,last_input=0;
    std::uint64_t last_position=0;
    std::vector<float> execute(std::uint32_t input,std::uint64_t position) override {
        ++full_calls;last_input=input;last_position=position;return values;
    }
    std::uint32_t execute_greedy(std::uint32_t input,std::uint64_t position,std::uint64_t) override {
        ++scalar_calls;last_input=input;last_position=position;return scalar;
    }
    void execute_state(std::uint32_t input,std::uint64_t position) override {
        ++state_calls;last_input=input;last_position=position;
    }
    unsigned calls() const {return full_calls+scalar_calls+state_calls;}
};
using Need=strix::GenerationOutputNeed;
} // namespace
int main() {
    {
        Backend backend;
        const auto result=strix::execute_generation_output({Need::token_only|Need::verify_decisions,2,7,3,{}},backend);
        expect(backend.scalar_calls==1 && backend.full_calls==0 && backend.state_calls==0 && result.token==0U &&
            result.verification_selection==0U && result.logits.empty() && result.position==7 && backend.last_input==2,
            "combined token/verification uses one scalar target call without full delivery");
    }
    {
        Backend backend;
        const auto result=strix::execute_generation_output({Need::state_only,1,9,3,{}},backend);
        expect(backend.state_calls==1 && backend.full_calls==0 && backend.scalar_calls==0 && !result.token &&
            !result.verification_selection && result.logits.empty() && backend.last_position==9,
            "state-only explicitly consumes state without requesting a head");
    }
    {
        Backend backend;
        strix::GenerationOutputRequest request;request.vocab_size=3;
        const auto result=strix::execute_generation_output(request,backend);
        expect(backend.full_calls==1 && result.logits==backend.values && !result.token,
            "default output remains complete untouched F32 logits");
    }
    {
        Backend backend;
        const auto result=strix::execute_generation_output({Need::token_only|Need::full_logits,0,0,3,{}},backend);
        expect(result.token==0U && std::bit_cast<std::uint32_t>(result.logits[0])==0x80000000U &&
            std::bit_cast<std::uint32_t>(result.logits[1])==0U && backend.full_calls==1 && backend.scalar_calls==0,
            "full-output combination retains signed-zero bits and first-max lowest ID");
    }
    {
        Backend backend;backend.values={0,0};
        const std::vector<std::uint32_t> ids{1,0,1};
        const auto result=strix::execute_generation_output({Need::token_only|Need::verify_decisions|
            Need::requested_logprobs|Need::full_logits,0,4,2,ids},backend);
        expect(backend.full_calls==1 && backend.scalar_calls==0 && result.token==0U && result.verification_selection==0U &&
            result.requested_logprobs.size()==3 && result.requested_logprobs[0].token_id==1 &&
            result.requested_logprobs[1].token_id==0 && result.requested_logprobs[2].token_id==1 &&
            result.requested_logprobs[0].log_probability==-std::log(2.0) && result.logits==backend.values,
            "compatible needs share one full call, preserve requested order/duplicates and use the full normalizer");
    }
    {
        Backend backend;backend.values={std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),
            -std::numeric_limits<float>::max()};
        const std::vector<std::uint32_t> ids{0,2};
        const auto result=strix::execute_generation_output({Need::requested_logprobs,0,0,3,ids},backend);
        expect(result.logits.empty() && !result.token && std::isfinite(result.requested_logprobs[1].log_probability) &&
            result.requested_logprobs[0].log_probability==-std::log(2.0),
            "centered FP64 logsoftmax preserves equal huge maxima and extreme finite tails");
    }
    {
        Backend backend;backend.values={0,1,2};
        const std::vector<std::uint32_t> ids{0};
        const auto result=strix::execute_generation_output({Need::requested_logprobs,0,0,3,ids},backend);
        const auto expected=-2.0-std::log(std::exp(-2.0)+std::exp(-1.0)+1.0);
        expect(result.requested_logprobs[0].log_probability==expected && result.requested_logprobs[0].log_probability<0,
            "requested subset never becomes the distribution support");
    }
    for(const auto invalid:{std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()}) {
        Backend backend;backend.values={0,invalid};
        const std::vector<std::uint32_t> ids{0};
        bool rejected=false;
        try {(void)strix::execute_generation_output({Need::requested_logprobs,0,0,2,ids},backend);}
        catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected && backend.full_calls==1,"nonfinite unrequested vocabulary item rejects subset output");
    }
    {
        const std::vector<std::uint32_t> bad_ids{0,3},good_ids{0};
        std::vector<strix::GenerationOutputRequest> invalid{
            {static_cast<Need>(0),0,0,3,{}}, {static_cast<Need>(1U<<31U),0,0,3,{}},
            {Need::mtp_hidden,0,0,3,{}}, {Need::state_only|Need::token_only,0,0,3,{}},
            {Need::requested_logprobs,0,0,3,{}}, {Need::requested_logprobs,0,0,3,bad_ids},
            {Need::full_logits,0,0,3,good_ids}, {Need::token_only,3,0,3,{}}, {Need::full_logits,0,0,0,{}}};
        for(const auto& request:invalid) {
            Backend backend;bool rejected=false;
            try {(void)strix::execute_generation_output(request,backend);} catch(const std::invalid_argument&) {rejected=true;}
            expect(rejected && backend.calls()==0,"invalid needs/IDs/inputs reject before any backend call");
        }
    }
    {
        Backend backend;backend.values={0,1};bool rejected=false;
        try {(void)strix::execute_generation_output({Need::full_logits,0,0,3,{}},backend);}
        catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected && backend.calls()==1,"incorrect full vocabulary extent cannot be published");
    }
    {
        Backend backend;backend.scalar=3;bool rejected=false;
        try {(void)strix::execute_generation_output({Need::verify_decisions,0,0,3,{}},backend);}
        catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected,"out-of-vocabulary scalar cannot become a valid verification selection");
    }
    if(!failures) std::cout<<"generation output tests passed\n";
    return failures?1:0;
}
