#include "strix/mtp_target_window_policy.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
int failures=0;
void expect(bool pass,const char* message) {
    if(!pass) { std::cerr<<"FAIL: "<<message<<'\n'; ++failures; }
}
template<class Work> void invalid(Work&& work,const char* message) {
    bool rejected=false;
    try { work(); } catch(const std::invalid_argument&) { rejected=true; }
    expect(rejected,message);
}
} // namespace
int main() {
    using namespace strix;
    constexpr auto none=std::numeric_limits<std::uint32_t>::max();
    const std::array<std::uint32_t,2> eos={62,63};
    for(const std::uint32_t rows:{2U,4U,8U}) {
        std::vector<std::uint32_t> inputs(rows),predictions(rows);
        for(std::uint32_t row=0;row<rows;++row) { inputs[row]=10+row; predictions[row]=11+row; }
        const auto assess=[&](std::uint32_t budget) {
            return assess_mtp_target_window(inputs,predictions,10,budget,eos,64);
        };
        auto result=assess(rows);
        expect(result.accepts_complete_chunk() && result.mismatch_input_row==none,"exact causal chunk accepted");
        expect(assess(rows+1).accepts_complete_chunk(),"budget above R accepted");
        result=assess(rows-1);
        expect(result.decision==MtpTargetWindowDecision::insufficient_output_budget &&
               result.mismatch_input_row==none && !result.accepts_complete_chunk(),"R-1 budget rejects whole chunk");
        for(std::uint32_t row=1;row<rows;++row) {
            const auto old=inputs[row]; inputs[row]=40;
            result=assess(rows);
            expect(result.decision==MtpTargetWindowDecision::target_input_mismatch &&
                   result.mismatch_input_row==row,"each nonzero mismatch row identified");
            inputs[row]=old;
        }
        inputs[1]=40; inputs[rows-1]=41;
        expect(assess(rows).mismatch_input_row==1,"earliest mismatch wins");
        inputs[1]=11; inputs[rows-1]=10+rows-1;
        inputs[0]=9;
        invalid([&] { (void)assess(rows); },"row zero mismatch is invalid metadata");
        inputs[0]=10;
        predictions.back()=eos[0];
        expect(assess(rows).accepts_complete_chunk(),"last output EOS is allowed because it is not consumed");
        predictions.back()=11+rows-1;
        inputs.back()=eos[1]; predictions[rows-2]=eos[1];
        result=assess(rows);
        expect(result.decision==MtpTargetWindowDecision::eos_would_be_consumed && result.mismatch_input_row==none,
               "last input EOS is rejected even with matching predictions");
        inputs.back()=10+rows-1; predictions[rows-2]=11+rows-2;
        const std::array<std::uint32_t,1> first_eos={10};
        expect(assess_mtp_target_window(inputs,predictions,10,rows,first_eos,64).decision==
               MtpTargetWindowDecision::eos_would_be_consumed,"pending input EOS cannot be consumed");
        expect(assess_mtp_target_window(inputs,predictions,10,rows,{},64).accepts_complete_chunk(),"empty EOS set is valid");
        predictions.back()=64;
        invalid([&] { (void)assess(0); },"out of vocabulary final prediction rejected before budget decision");
        predictions.back()=11+rows-1;
        inputs.back()=64;
        invalid([&] { (void)assess(rows); },"out of vocabulary input rejected");
        inputs.back()=10+rows-1;
        invalid([&] { (void)assess_mtp_target_window(inputs,predictions,64,rows,eos,64); },"invalid pending rejected");
        invalid([&] { (void)assess_mtp_target_window(inputs,predictions,10,rows,eos,0); },"zero vocabulary rejected");
        const std::array<std::uint32_t,1> bad_eos={64};
        invalid([&] { (void)assess_mtp_target_window(inputs,predictions,10,rows,bad_eos,64); },"invalid EOS metadata rejected");
        invalid([&] { (void)assess_mtp_target_window(inputs,std::span<const std::uint32_t>(predictions).first(rows-1),10,rows,eos,64); },
                "wrong prediction row count rejected");
    }
    for(const std::size_t rows:{0U,1U,3U,5U,16U}) {
        const std::vector<std::uint32_t> tokens(rows,1);
        invalid([&] { (void)assess_mtp_target_window(tokens,tokens,1,32,{},64); },"unsupported window size rejected");
    }
    const std::array<std::uint32_t,2> edge_inputs={0,std::numeric_limits<std::uint32_t>::max()};
    const std::array<std::uint32_t,2> edge_predictions={std::numeric_limits<std::uint32_t>::max(),0};
    expect(assess_mtp_target_window(edge_inputs,edge_predictions,0,2,{},1ULL<<32).accepts_complete_chunk(),
           "uint64 vocabulary admits the complete uint32 token domain without narrowing");
    return failures?1:0;
}
