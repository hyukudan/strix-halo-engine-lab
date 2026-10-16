#include "strix/hip_event_queue_trace.hpp"
#include "strix/hip_moe_overlap_dispatcher.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
void check(hipError_t result) { if(result!=hipSuccess) throw std::runtime_error(hipGetErrorString(result)); }
void expect(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
__global__ void trace_fixture_write(float* out,float value) {if(!threadIdx.x) *out=value;}
__global__ void trace_fixture_combine(float* values) {if(!threadIdx.x) values[3]=values[0]+values[1]+values[2];}
struct Owner {
    std::shared_ptr<const strix::HipSetupQuiescenceGuard> guard;
    hipStream_t main=nullptr;
    float* values=nullptr;
    std::uint64_t* status=nullptr;
    explicit Owner(std::shared_ptr<const strix::HipSetupQuiescenceGuard> original):guard(std::move(original)) {}
    void initialize() {
        check(hipStreamCreateWithFlags(&main,hipStreamNonBlocking));
        check(hipMalloc(&values,4U*sizeof(float)));check(hipMalloc(&status,8));
        check(hipMemset(values,0,4U*sizeof(float)));check(hipMemset(status,0,8));
    }
    ~Owner() {
        guard->recover_or_fail_stop();
        if(status) (void)hipFree(status);if(values) (void)hipFree(values);
        if(main) (void)hipStreamDestroy(main);
    }
};
struct Launch {
    float* values=nullptr;std::uint32_t index=0;float value=0;bool combine=false,fail_after_submit=false;
};
void launch(const strix::ExecutionDispatchContext&,void* opaque) {
    auto& value=*static_cast<Launch*>(opaque);
    if(value.combine) hipLaunchKernelGGL(trace_fixture_combine,dim3(1),dim3(32),0,strix::hip_execution_stream(),value.values);
    else hipLaunchKernelGGL(trace_fixture_write,dim3(1),dim3(32),0,strix::hip_execution_stream(),value.values+value.index,value.value);
    check(hipGetLastError());
    if(value.fail_after_submit) throw std::runtime_error("injected launcher exception after actual enqueue");
}
strix::HipLaunchContext scope_for(const Owner& owner) {
    strix::HipLaunchContext scope;scope.stream=owner.main;
    scope.deferred_moe_status={17,reinterpret_cast<std::uintptr_t>(owner.status),8,1};return scope;
}
std::size_t occurrences(const std::string& text,const std::string& needle) {
    std::size_t count=0,offset=0;
    while((offset=text.find(needle,offset))!=std::string::npos) {++count;offset+=needle.size();}
    return count;
}
}
int main() {
    try {
        auto guard=std::make_shared<const strix::HipSetupQuiescenceGuard>();
        auto owner=std::make_shared<Owner>(guard);owner->initialize();
        strix::HostDispatchCounters host;
        strix::HipEventQueueTrace trace(host,owner,guard,8);
        strix::ExecutionTelemetryBinding observed(&trace,{.request_id=11,.session_epoch=7});
        strix::HipMoeSiblingStreams streams(owner->main,owner,guard);
        Launch calls[4]{{owner->values,0,3,false,false},{owner->values,1,5,false,false},
            {owner->values,2,7,false,false},{owner->values,0,0,true,false}};
        const strix::SerialOp ops[4]{strix::SerialOp::token_embedding,strix::SerialOp::shared_expert,
            strix::SerialOp::moe_experts,strix::SerialOp::moe_combine};
        strix::CompiledLaunchPlan plan;plan.execution_identity=17;plan.plan_identity=19;
        for(std::uint32_t i=0;i<4;++i) plan.steps.push_back({i,ops[i],false,100U+i,&launch,&calls[i],nullptr});
        strix::DirectLaunchDispatcher dispatcher(plan,&trace);
        auto context_scope=scope_for(*owner);
        const auto dispatch=[&](std::uint32_t index) {
            strix::ExecutionDispatchContext context;context.execution_identity=17;context.step_id=index;
            context.op=ops[index];context.position=0;context.layer=2;context.logical_rows=1;
            dispatcher.dispatch_kernel(context);
        };
        {
            strix::HipLaunchScope active(context_scope);
            streams.begin_token(0,0,1);dispatch(0);streams.fork_shared(2);
            {strix::HipSharedExpertSideScope side(owner->main,streams.side_stream());dispatch(1);}
            streams.record_shared_done(2);streams.begin_experts(2);dispatch(2);
            streams.join_before_combine(2);dispatch(3);streams.verify_before_commit();
            trace.require_healthy_before_commit();streams.mark_committed(1,1);
        }
        float output=0;check(hipMemcpy(&output,owner->values+3,sizeof(output),hipMemcpyDeviceToHost));
        expect(output==15,"instrumented actual fork/join preserves producer and combine result");
        std::ostringstream json;trace.write_after_completion(json);
        const auto text=json.str();
        expect(occurrences(text,"\"launcher_returned\":true")==4,"four actual launchers delimit four queue spans");
        expect(occurrences(text,"\"timing_known\":true")==4,"both actual queues share completed event-origin timing");
        expect(text.find("\"all_recorded_events_completed\":true")!=std::string::npos,"completion comes from querying every recorded event");
        expect(occurrences(text,"\"host_api_accepted\":true")==4,"real fork/join contributes four accepted dependency operations");
        expect(text.find("\"source_record\":1")!=std::string::npos&&text.find("\"source_record\":3")!=std::string::npos,
            "wait dependencies reference their actual preceding record versions");
        expect(text.find("\"kernel_busy_ns\":null")!=std::string::npos,"queue spans never manufacture kernel busy time");
        expect(host.totals().returned==4,"device observer forwards ordinary host observations");
        // A closed pool never re-records an event still owned by an old span.
        {
            strix::HipLaunchScope active(context_scope);
            strix::ExecutionDispatchContext context;
            expect(trace.begin_queue_span(context)==0,"sealed trace rejects later observations without enqueue");
        }
        {
            strix::HipEventQueueTrace capture_trace(host,owner,guard,2);
            hipGraph_t graph=nullptr;
            {
                strix::HipLaunchScope active(context_scope);
                bool rejected=false;
                try {strix::HipEventQueueTrace nested(host,owner,guard,1);}
                catch(const std::invalid_argument&) {rejected=true;}
                expect(rejected,"event owner setup rejects active foreign launch scope before allocation");
                check(hipStreamBeginCapture(owner->main,hipStreamCaptureModeThreadLocal));
                strix::ExecutionDispatchContext context;
                expect(capture_trace.begin_queue_span(context)==0,"underlying stream capture rejected before recording any timing event");
                check(hipStreamEndCapture(owner->main,&graph));
            }
            std::size_t nodes=0;check(hipGraphGetNodes(graph,nullptr,&nodes));check(hipGraphDestroy(graph));
            expect(nodes==0,"trace capture rejection adds no graph nodes");
            std::ostringstream skipped;capture_trace.write_after_completion(skipped);
            expect(skipped.str().find("\"capture_skipped\":1")!=std::string::npos,"capture skip is explicit observation metadata");
        }
        {
            strix::HipEventQueueTrace bounded(host,owner,guard,1);
            strix::DirectLaunchDispatcher limited(plan,&bounded);
            strix::ExecutionDispatchContext context;context.execution_identity=17;context.step_id=0;context.op=ops[0];
            {
                strix::HipLaunchScope active(context_scope);
                limited.dispatch_kernel(context);
                calls[0].fail_after_submit=true;
                bool failed=false;
                try {limited.dispatch_kernel(context);} catch(const std::runtime_error&) {failed=true;}
                calls[0].fail_after_submit=false;
                expect(failed,"full trace pool preserves an actual launcher exception");
                check(hipStreamSynchronize(owner->main));
            }
            std::ostringstream limited_json;bounded.write_after_completion(limited_json);
            expect(limited_json.str().find("\"dropped\":1")!=std::string::npos,"fixed event capacity reports the unrecorded launcher");
            expect(occurrences(limited_json.str(),"\"ticket\":")==1,"capacity overflow does not reuse recorded event handles");
        }
        {
            strix::HipEventQueueTrace failed_trace(host,owner,guard,1);
            strix::DirectLaunchDispatcher failing(plan,&failed_trace);
            strix::ExecutionDispatchContext context;context.execution_identity=17;context.step_id=0;context.op=ops[0];
            {
                strix::HipLaunchScope active(context_scope);calls[0].fail_after_submit=true;
                bool failed=false;try {failing.dispatch_kernel(context);} catch(const std::runtime_error&) {failed=true;}
                calls[0].fail_after_submit=false;
                expect(failed,"recorded launcher failure propagates unchanged");check(hipStreamSynchronize(owner->main));
            }
            std::ostringstream failure_json;failed_trace.write_after_completion(failure_json);
            expect(failure_json.str().find("\"launcher_returned\":false")!=std::string::npos&&
                failure_json.str().find("\"gpu_event_completed\":true")!=std::string::npos,
                "event completion after drain is independent of launcher host outcome");
        }
        // Teardown with an unfinished trace retains events/allocations until
        // original-device recovery; this is an owner-lifetime fixture only.
        {
            strix::HipEventQueueTrace queued(host,owner,guard,1);
            strix::DirectLaunchDispatcher queued_dispatcher(plan,&queued);
            strix::ExecutionDispatchContext context;context.execution_identity=17;context.op=ops[0];context.step_id=0;
            {strix::HipLaunchScope active(context_scope);queued_dispatcher.dispatch_kernel(context);}
        }
        std::cout<<"HIP event queue trace source fixtures completed\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
