#include "strix/hip_event_queue_trace.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/execution_runtime.hpp"
#include <hip/hip_runtime.h>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix {
namespace {
std::atomic<std::uint64_t> next_trace_domain{1};
void increment(std::uint64_t& value) noexcept { if(value!=UINT64_MAX) ++value; }
}
struct HipEventQueueTrace::Impl {
    struct Span {
        hipEvent_t begin=nullptr,end=nullptr;
        hipStream_t stream=nullptr;
        ExecutionCorrelation correlation;
        DispatchOutcome outcome=DispatchOutcome::failed;
        bool began=false,ended=false;
    };
    struct Dependency {
        QueueDependencyKind kind=QueueDependencyKind::event_record;
        std::uintptr_t stream=0,event=0;
        ExecutionCorrelation correlation;
        std::uint64_t source_record=0;
    };
    struct LastRecord { std::uintptr_t event=0;std::uint64_t sequence=0; };
    ExecutionTelemetry& host;
    std::shared_ptr<const void> owner;
    std::shared_ptr<const HipSetupQuiescenceGuard> guard;
    std::vector<Span> spans;
    std::vector<Dependency> dependencies;
    std::array<hipStream_t,32> queues{};
    std::array<LastRecord,64> last_records{};
    hipEvent_t anchor=nullptr;
    std::size_t used=0,dependency_used=0,queue_count=0;
    std::uint64_t domain=0,dropped=0,capture_skipped=0,api_errors=0,dependency_dropped=0;
    hipError_t first_error=hipSuccess;
    bool queued=false,complete=false,broken=false,sealed=false;

    Impl(ExecutionTelemetry& forward,std::shared_ptr<const void> retained,
        std::shared_ptr<const HipSetupQuiescenceGuard> original,std::size_t capacity)
        :host(forward),owner(std::move(retained)),guard(std::move(original)) {
        if(!owner||!guard||!capacity||capacity>32768U||current_hip_launch_context())
            throw std::invalid_argument("device queue trace requires retained owners, no active scope and capacity within one and 32768");
        guard->require_original();
        // No event handle or clock domain is reused by another trace owner.
        auto value=next_trace_domain.load(std::memory_order_relaxed);
        for(;;) {
            if(!value||value==UINT64_MAX) throw std::overflow_error("device trace clock domain exhausted");
            if(next_trace_domain.compare_exchange_weak(value,value+1U,std::memory_order_relaxed)) {domain=value;break;}
        }
        spans.resize(capacity);dependencies.resize(capacity*2U);
    }
    bool accepted(hipError_t error) noexcept {
        if(error==hipSuccess) return true;
        increment(api_errors);if(first_error==hipSuccess) first_error=error;
        // Observation APIs may surface asynchronous target failures. Never
        // clear the HIP error slot or attribute a failed API solely to tracing.
        return false;
    }
    void initialize() {
        const auto create=[&](hipEvent_t& event) {
            if((hipEventCreateWithFlags)(&event,hipEventDefault)!=hipSuccess)
                throw std::runtime_error("device queue trace timing-event allocation failed");
        };
        create(anchor);
        for(auto& span:spans) {create(span.begin);create(span.end);}
    }
    ~Impl() {
        // Exceptional teardown never frees an event while its outcome is
        // unknown. No per-operator synchronization is introduced.
        if(queued&&!complete) guard->recover_or_fail_stop();
        else {try {guard->require_original();} catch(...) {guard->recover_or_fail_stop();}}
        for(auto& span:spans) {
            if(span.end) (void)(hipEventDestroy)(span.end);
            if(span.begin) (void)(hipEventDestroy)(span.begin);
        }
        if(anchor) (void)(hipEventDestroy)(anchor);
    }
    bool eager_stream(hipStream_t stream) noexcept {
        const auto* context=current_hip_launch_context();
        if(!context||context->graph_capture||!stream||stream==hipStreamLegacy||stream==hipStreamPerThread) {
            increment(capture_skipped);return false;
        }
        int device=-1;
        if(!accepted(hipGetDevice(&device))||device!=guard->device_index()) {broken=true;return false;}
        hipStreamCaptureStatus status=hipStreamCaptureStatusNone;
        if(!accepted(hipStreamIsCapturing(stream,&status))) {broken=true;return false;}
        if(status!=hipStreamCaptureStatusNone) {increment(capture_skipped);return false;}
        return true;
    }
    bool register_queue(hipStream_t stream) noexcept {
        for(std::size_t i=0;i<queue_count;++i) if(queues[i]==stream) return true;
        if(queue_count==queues.size()) {increment(dropped);return false;}
        queued=true;
        if(!queue_count) {
            if(!accepted((hipEventRecord)(anchor,stream))) {broken=true;return false;}
        } else {
            // One common, same-device event origin. This opt-in ordering edge
            // is explicit profiling overhead, not a measured model dependency.
            if(!accepted((hipStreamWaitEvent)(stream,anchor,0))) {broken=true;return false;}
        }
        queues[queue_count++]=stream;
        return true;
    }
};
HipEventQueueTrace::HipEventQueueTrace(ExecutionTelemetry& host,std::shared_ptr<const void> owner,
    std::shared_ptr<const HipSetupQuiescenceGuard> guard,std::size_t capacity)
    :impl_(std::make_unique<Impl>(host,std::move(owner),std::move(guard),capacity)) {impl_->initialize();}
HipEventQueueTrace::~HipEventQueueTrace()=default;
void HipEventQueueTrace::observe(const HostDispatchObservation& observation) noexcept {impl_->host.observe(observation);}
void HipEventQueueTrace::observe_activity(const ExecutionActivity& activity) noexcept {impl_->host.observe_activity(activity);}
std::uint64_t HipEventQueueTrace::begin_queue_span(const ExecutionDispatchContext& context) noexcept {
    auto& s=*impl_;
    if(s.sealed||s.broken||s.used==s.spans.size()) {increment(s.dropped);return 0;}
    const auto pending_error=(hipPeekAtLastError)();
    if(pending_error!=hipSuccess) {(void)s.accepted(pending_error);s.broken=true;increment(s.dropped);return 0;}
    const auto stream=hip_execution_stream();
    if(!s.eager_stream(stream)||!s.register_queue(stream)) return 0;
    const auto index=s.used++;
    auto& span=s.spans[index];span.stream=stream;span.correlation=current_execution_correlation();
    span.correlation.execution_identity=context.execution_identity;span.correlation.position=context.position;
    span.correlation.step_id=context.step_id;span.correlation.op=context.op;span.correlation.layer=context.layer.value_or(UINT32_MAX);
    span.correlation.rows=context.logical_rows;span.correlation.phase=ExecutionPhase::operator_step;
    s.queued=true;
    if(!s.accepted((hipEventRecord)(span.begin,stream))) {s.broken=true;return 0;}
    span.began=true;
    return index+1U;
}
void HipEventQueueTrace::end_queue_span(std::uint64_t ticket,DispatchOutcome outcome) noexcept {
    if(!ticket) return;
    auto& s=*impl_;
    if(ticket>s.used) {increment(s.dropped);return;}
    auto& span=s.spans[static_cast<std::size_t>(ticket-1U)];span.outcome=outcome;
    const auto pending_error=(hipPeekAtLastError)();
    if(pending_error!=hipSuccess) {(void)s.accepted(pending_error);s.broken=true;increment(s.dropped);return;}
    if(span.ended||!span.began||hip_execution_stream()!=span.stream||!s.eager_stream(span.stream)) {
        increment(s.dropped);return;
    }
    s.queued=true;
    if(!s.accepted((hipEventRecord)(span.end,span.stream))) {s.broken=true;return;}
    span.ended=true;
}
void HipEventQueueTrace::queue_dependency(QueueDependencyKind kind,std::uintptr_t stream,std::uintptr_t event) noexcept {
    auto& s=*impl_;
    if(s.sealed||s.dependency_used==s.dependencies.size()) {increment(s.dependency_dropped);return;}
    auto& d=s.dependencies[s.dependency_used];d={kind,stream,event,current_execution_correlation(),0};
    Impl::LastRecord* slot=nullptr;
    for(auto& record:s.last_records) {
        if(record.event==event) {slot=&record;break;}
        if(!record.event&&!slot) slot=&record;
    }
    if(kind==QueueDependencyKind::event_record) {
        if(slot) *slot={event,s.dependency_used+1U};
        else increment(s.dependency_dropped);
    } else if(slot&&slot->event==event) d.source_record=slot->sequence;
    ++s.dependency_used;
}
void HipEventQueueTrace::require_healthy_before_commit() const {
    if(impl_->broken||impl_->api_errors) throw std::runtime_error("device queue trace API or original-device observation failed");
}
void HipEventQueueTrace::write_after_completion(std::ostream& out) {
    auto& s=*impl_;s.sealed=true;s.guard->require_original();
    const auto anchor_query=s.queued?hipEventQuery(s.anchor):hipSuccess;
    if(anchor_query!=hipSuccess&&anchor_query!=hipErrorNotReady) (void)s.accepted(anchor_query);
    bool all_complete=anchor_query==hipSuccess;
    const auto precision=out.precision(17);
    out<<"{\"device_queue_trace\":{\"profiling_only\":true,\"scope\":\"launcher-queue-spans-including-gaps-and-waits\","
        "\"kernel_busy_ns\":null,\"overlap_of_kernels_ns\":null,\"timing_resolution_approx_ns\":1000,"
        "\"clock_domain\":"<<s.domain<<",\"device_binding\":"<<s.guard->observed_binding_identity()<<
        ",\"common_origin\":\"same-device-HIP-timing-event\",\"instrumentation_queue_anchor_waits\":"<<(s.queue_count?s.queue_count-1U:0U)<<
        ",\"capacity\":"<<s.spans.size()<<",\"dropped\":"<<s.dropped<<",\"capture_skipped\":"<<s.capture_skipped<<
        ",\"spans\":[";
    for(std::size_t i=0;i<s.used;++i) {
        const auto& span=s.spans[i];
        float begin=0,end=0;
        const auto qb=span.began?hipEventQuery(span.begin):hipErrorNotReady;
        const auto qe=span.ended?hipEventQuery(span.end):hipErrorNotReady;
        if(qb!=hipSuccess&&qb!=hipErrorNotReady) (void)s.accepted(qb);
        if(qe!=hipSuccess&&qe!=hipErrorNotReady) (void)s.accepted(qe);
        const bool done=span.began&&span.ended&&qb==hipSuccess&&qe==hipSuccess;
        all_complete=all_complete&&done;
        bool timing=false;
        if(done) {
            const auto eb=hipEventElapsedTime(&begin,s.anchor,span.begin);
            const auto ee=hipEventElapsedTime(&end,s.anchor,span.end);
            const bool begin_ok=s.accepted(eb),end_ok=s.accepted(ee);
            timing=begin_ok&&end_ok&&std::isfinite(begin)&&std::isfinite(end)&&begin>=0&&end>=begin;
        }
        if(i) out<<',';
        out<<"{\"ticket\":"<<i+1U<<",\"queue\":"<<reinterpret_cast<std::uintptr_t>(span.stream)<<
            ",\"execution_identity\":"<<span.correlation.execution_identity<<",\"position\":"<<span.correlation.position<<
            ",\"step_id\":"<<span.correlation.step_id<<",\"layer\":"<<span.correlation.layer<<
            ",\"request_id\":"<<span.correlation.request_id<<",\"session_epoch\":"<<span.correlation.session_epoch<<
            ",\"op\":\""<<to_string(span.correlation.op)<<"\",\"logical_rows\":"<<span.correlation.rows<<
            ",\"launcher_returned\":"<<(span.outcome==DispatchOutcome::returned?"true":"false")<<
            ",\"gpu_event_completed\":"<<(done?"true":"false")<<",\"timing_known\":"<<(timing?"true":"false")<<",\"begin_ns\":";
        if(timing) out<<static_cast<double>(begin)*1000000.0;else out<<"null";
        out<<",\"end_ns\":";if(timing) out<<static_cast<double>(end)*1000000.0;else out<<"null";
        out<<'}';
    }
    s.complete=all_complete;
    out<<"],\"dependencies\":[";
    for(std::size_t i=0;i<s.dependency_used;++i) {
        const auto& d=s.dependencies[i];if(i) out<<',';
        out<<"{\"sequence\":"<<i+1U<<",\"kind\":\""<<(d.kind==QueueDependencyKind::event_record?"event-record":"event-wait")<<
            "\",\"queue\":"<<d.stream<<",\"event\":"<<d.event<<",\"source_record\":"<<d.source_record<<
            ",\"host_api_accepted\":true,\"gpu_dependency_completion\":null,\"position\":"<<d.correlation.position<<
            ",\"step_id\":"<<d.correlation.step_id<<",\"layer\":"<<d.correlation.layer<<'}';
    }
    out<<"],\"dependency_dropped\":"<<s.dependency_dropped<<",\"observer_failed\":"<<((s.broken||s.api_errors)?"true":"false")<<",\"api_errors\":"<<s.api_errors<<
        ",\"first_api_error\":"<<static_cast<int>(s.first_error)<<",\"all_recorded_events_completed\":"<<(s.complete?"true":"false")<<"}}";
    out.precision(precision);
}
} // namespace strix
