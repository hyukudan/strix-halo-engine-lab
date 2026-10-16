// Host controller/scheduler fixtures only; mock cohort is not a GPU batching claim.
#include "strix/serving_cohort.hpp"
#include "strix/greedy_serving_scheduler.hpp"
#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
namespace {
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
struct Gate {
    std::mutex mutex; std::condition_variable cv; bool released=false; std::atomic<bool> first{true};
    void pause() { if(!first.exchange(false)) return; std::unique_lock lock(mutex); cv.wait(lock,[&]{return released;}); }
    void release() { std::lock_guard lock(mutex); released=true; cv.notify_all(); }
};
struct Backend final : strix::SerialGreedyStepBackend {
    std::shared_ptr<Gate> gate;
    std::vector<std::uint32_t> consumed;
    explicit Backend(std::shared_ptr<Gate> value={}) :gate(std::move(value)) {}
    void consume(std::uint32_t token,std::uint64_t position) {
        if(gate) gate->pause();
        require(position==consumed.size(),"cohort changed an independent request position");
        consumed.push_back(token);
    }
    std::vector<float> execute(std::uint32_t t,std::uint64_t p) override {
        consume(t,p); std::vector<float> result(16,-1); result[(t+1U)%16U]=1; return result;
    }
    std::uint32_t execute_greedy(std::uint32_t t,std::uint64_t p,std::uint64_t v) override {
        require(v==16,"mock vocabulary changed"); consume(t,p); return (t+1U)%16U;
    }
    void execute_state(std::uint32_t t,std::uint64_t p) override { consume(t,p); }
};
struct Cohort final : strix::ServingCohortExecutor {
    unsigned calls=0; std::function<void()> after_submit;
    bool compatible(std::span<strix::SerialGreedyStepBackend* const> values) const noexcept override {
        return values.size()==2U || values.size()==4U;
    }
    void execute(std::span<strix::ServingCohortTarget> values) override {
        strix::validate_serving_cohort_targets(values); ++calls;
        for(auto& value:values) strix::execute_serving_target_fallback(value);
        if(after_submit) { auto once=std::move(after_submit); once(); }
    }
};
strix::ServingCompatibility key() { strix::ServingCompatibility result;result.model_content.bytes[0]=1;result.numerical_identity=1;return result; }
void scheduler_case(unsigned width,bool cancel_one) {
    auto owner=std::make_shared<int>(1); auto gate=std::make_shared<Gate>(); auto cohort=std::make_shared<Cohort>();
    strix::GreedyServingPolicy policy;
    policy.concurrency=width;policy.tensor_cohort_rows=width;policy.max_retained_requests=4;
    policy.reservation_budget_bytes=65536;policy.prefill_chunk_rows=1;
    strix::GreedyServingScheduler scheduler(policy,key(),owner,cohort);
    struct Release {std::shared_ptr<Gate> gate;~Release(){gate->release();}} release{gate};
    std::array<std::array<std::uint32_t,3>,4> prompts{{{1,2,3},{4,5,6},{7,8,9},{10,11,12}}};
    std::array<strix::ServingRequestHandle,4> handles;
    for(unsigned i=0;i<width;++i) {
        const auto submission=scheduler.submit({prompts[i],16,16,3,15},key(),i+1U,1,4096,owner,
            [gate]{return std::make_unique<Backend>(gate);});
        require(submission.admission==strix::ServingAdmission::accepted,"cohort request was not admitted");
        handles[i]=submission.request;
    }
    if(cancel_one) cohort->after_submit=[handle=handles[1]] { require(handle.cancel(),"in-flight individual cancellation failed"); };
    gate->release(); scheduler.run_until_idle();
    require(cohort->calls>0,"ready C2/C4 requests never used the actual cohort executor seam");
    for(unsigned i=0;i<width;++i) {
        if(cancel_one && i==1U) {
            require(handles[i].state()==strix::ServingRequestState::cancelled && handles[i].result().steps_executed>0,
                "cancel after completed target lost actual consumption accounting");
            continue;
        }
        Backend baseline; const auto reference=strix::execute_serial_greedy_request({prompts[i],16,16,3,15},baseline);
        const auto& actual=handles[i].result();
        require(handles[i].state()==strix::ServingRequestState::completed && actual.generated_tokens==reference.generated_tokens &&
            actual.steps_executed==reference.steps_executed && actual.final_position==reference.final_position &&
            actual.stop_reason==reference.stop_reason,"cohort scheduling changed greedy/frontier policy or cancelled another request");
    }
}
} // namespace
int main() try {
    for(const auto width:{2U,4U}) { scheduler_case(width,false); scheduler_case(width,true); }
    // An isolated ready request completes via C1 immediately; no companion/fill
    // signal is provided. This would deadlock if the new scheduler waited to fill.
    auto owner=std::make_shared<int>(1);auto cohort=std::make_shared<Cohort>();
    strix::GreedyServingPolicy policy;policy.concurrency=4;policy.tensor_cohort_rows=4;
    policy.reservation_budget_bytes=8192;
    strix::GreedyServingScheduler scheduler(policy,key(),owner,cohort);
    const std::array<std::uint32_t,1> prompt{1};
    const auto one=scheduler.submit({prompt,16,16,1,15},key(),1,1,4096,owner,[]{return std::make_unique<Backend>();});
    scheduler.run_until_idle();
    require(one.request.state()==strix::ServingRequestState::completed && cohort->calls==0,
        "singleton was called a tensor cohort or waited for nonexistent requests");
    // Public target descriptors cannot repeat a backend to fake independent rows.
    Backend backend;strix::SerialGreedyTokenCursor cursor({prompt,16,16,1,15});
    const auto target=cursor.prepare_target(backend);
    require(target.has_value(),"target preparation unexpectedly completed");
    std::array<strix::ServingCohortTarget,2> duplicate{{{&backend,*target,{}},{&backend,*target,{}}}};
    bool rejected=false;try {strix::validate_serving_cohort_targets(duplicate);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected && backend.consumed.empty(),"duplicate cohort member consumed an input");
    cursor.cancel_prepared_target(*target);
    std::cout<<"serving cohort source fixtures passed\n";return 0;
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
