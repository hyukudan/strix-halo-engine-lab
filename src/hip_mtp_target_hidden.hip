#include "strix/hip_mtp_target_hidden.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/hip_mtp_native_validation.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>
namespace strix {
namespace {
void check(hipError_t e,const char* op) {
    if(e!=hipSuccess) throw std::runtime_error(std::string(op)+": "+hipGetErrorString(e));
}
}
struct HipMtpTargetHiddenTicket::Lease {
    std::shared_ptr<HipMtpTargetHidden> owner;
    MtpDeviceView snapshot;
    std::uint64_t allocation_lease=0,position=0;
    std::uint32_t token=0,predicted=0;
    SerialSessionContext committed;
    ModelContentIdentity content;
};
struct HipMtpTargetHidden::Impl {
    struct Slot {
        MtpDeviceView view;
        std::uint64_t lease=0;
        std::weak_ptr<const HipMtpTargetHiddenTicket::Lease> reader;
    };
    ModelContract model;
    CompiledSerialExecution compiled;
    std::shared_ptr<const RuntimeExecutionMemoryOwner> memory;
    std::shared_ptr<const HipModelResidency> weights;
    std::shared_ptr<HipRuntimeAllocator> allocator;
    std::shared_ptr<HipSetupQuiescenceGuard> guard;
    hipStream_t stream=nullptr;
    hipEvent_t copied=nullptr;
    std::vector<Slot> slots;
    std::uint32_t tap_step=0;
    ExecutionBufferUse tap_use;
    ExecutionBufferAddress logits;
    std::unique_ptr<HipGreedyOutput> selector;
    std::uint64_t epoch=0;
    bool busy=false,failed=false;
    ~Impl() {
        if(guard) guard->recover_or_fail_stop();
        if(copied) (void)hipEventDestroy(copied);
        if(stream) (void)hipStreamDestroy(stream);
        if(allocator) for(auto& s:slots)
            if(s.view.address) allocator->free_device(s.view.address,s.view.bytes);
    }
};
HipMtpTargetHidden::HipMtpTargetHidden():impl_(std::make_unique<Impl>()){}
HipMtpTargetHidden::~HipMtpTargetHidden()=default;
std::uint32_t HipMtpTargetHidden::retained_rows() const noexcept {return static_cast<std::uint32_t>(impl_->slots.size());}
bool HipMtpTargetHiddenTicket::valid() const noexcept {return !!lease_;}
bool HipMtpTargetHiddenTicket::same_producer(const HipMtpTargetHiddenTicket& b) const noexcept {
    return lease_ && b.lease_ && lease_->owner==b.lease_->owner;
}
MtpDeviceView HipMtpTargetHiddenTicket::view() const {
    if(!lease_ || !lease_->owner) throw std::invalid_argument("MTP target hidden ticket is empty");
    const auto& i=*lease_->owner->impl_;
    i.guard->require_original();
    i.memory->validate_inventory_ownership();
    const auto v=lease_->snapshot;
    if(i.allocator->allocation_lease_identity(v.address,v.bytes,false)!=lease_->allocation_lease)
        throw std::invalid_argument("MTP target hidden allocation lease changed");
    return v;
}
std::uint64_t HipMtpTargetHiddenTicket::source_position() const {(void)view();return lease_->position;}
std::uint32_t HipMtpTargetHiddenTicket::input_token() const {(void)view();return lease_->token;}
std::uint32_t HipMtpTargetHiddenTicket::predicted_token() const {(void)view();return lease_->predicted;}
const SerialSessionContext& HipMtpTargetHiddenTicket::committed_session() const {(void)view();return lease_->committed;}
ModelContentIdentity HipMtpTargetHiddenTicket::target_content_identity() const {(void)view();return lease_->content;}
std::shared_ptr<HipMtpTargetHidden> HipMtpTargetHidden::create(
    const ModelContract& m,const CompiledSerialExecution& c,
    std::shared_ptr<const RuntimeExecutionMemoryOwner> memory,
    std::shared_ptr<const HipModelResidency> weights,
    std::shared_ptr<HipRuntimeAllocator> allocator,std::uint32_t rows,std::uint64_t budget) {
    if(!m.validate().empty() || !memory || !weights || !allocator || !rows ||
       !c.execution_identity || m.vocab_size>std::numeric_limits<std::uint32_t>::max() || c.logical_rows!=1 || c.request_concurrency!=1 ||
       !weights->binding().model_content_identity.valid())
        throw std::invalid_argument("MTP target tap requires actual single-row retained target resources");
    if(weights.owner_before(weights->weak_from_this().lock()) ||
       weights->weak_from_this().lock().owner_before(weights))
        throw std::invalid_argument("MTP target capture requires actual shared residency control block");
    (void)weights->owner_inventory(weights);
    memory->validate_inventory_ownership();
    const auto bytes=mtp_native::product(mtp_native::product(m.hidden_size,m.hc_count),2);
    const auto selector_bytes=HipGreedyOutput::required_scratch_bytes(static_cast<std::uint32_t>(m.vocab_size));
    if(selector_bytes>budget || mtp_native::product(rows,mtp_native::sum(bytes,255))>budget-selector_bytes)
        throw std::invalid_argument("MTP target capture exceeds upfront retained budget");
    auto result=std::shared_ptr<HipMtpTargetHidden>(new HipMtpTargetHidden);
    auto& i=*result->impl_;
    i.model=m;i.compiled=c;i.memory=std::move(memory);i.weights=std::move(weights);i.allocator=std::move(allocator);
    i.guard=std::make_shared<HipSetupQuiescenceGuard>();
    auto head=std::find_if(c.execution.steps.begin(),c.execution.steps.end(),[](const auto& s){return s.op==SerialOp::final_hyper_mix;});
    if(head==c.execution.steps.end() || head==c.execution.steps.begin())
        throw std::invalid_argument("MTP target tap requires final wide hidden producer");
    const auto& producer=*std::prev(head);
    if(producer.op!=SerialOp::mlp_inject || producer.layer!=m.hidden_layers-1)
        throw std::invalid_argument("MTP target hidden must be last MLP injection before final mix");
    auto io=std::find_if(c.io.steps.begin(),c.io.steps.end(),[&](const auto& s){return s.step_id==producer.id;});
    auto headio=std::find_if(c.io.steps.begin(),c.io.steps.end(),[&](const auto& s){return s.step_id==head->id;});
    if(io==c.io.steps.end() || headio==c.io.steps.end() || io->writes.size()!=1 ||
       io->writes[0].buffer.buffer_class!=ExecutionBufferClass::expanded_hidden ||
       std::find(headio->reads.begin(),headio->reads.end(),io->writes[0])==headio->reads.end())
        throw std::invalid_argument("MTP target tap wide producer/consumer IO differs");
    const auto* allocation=find_arena_allocation(c.arena,io->writes[0]);
    if(!allocation || allocation->bytes!=bytes)
        throw std::invalid_argument("MTP target capture requires exact BF16 wide activation layout");
    i.tap_step=producer.id;i.tap_use=io->writes[0];
    if(c.execution.steps.back().op!=SerialOp::lm_head || c.io.steps.empty() ||
       c.io.steps.back().op!=SerialOp::lm_head || c.io.steps.back().writes.size()!=1)
        throw std::invalid_argument("MTP target producer requires actual terminal target logits");
    i.logits=resolve_execution_buffer(c.execution_identity,c.arena,i.memory->scratch(),c.io.steps.back().writes[0]);
    if(i.logits.bytes!=mtp_native::product(m.vocab_size,4))
        throw std::invalid_argument("MTP target producer requires full FP32 target vocabulary");
    i.selector=std::make_unique<HipGreedyOutput>(static_cast<std::uint32_t>(m.vocab_size),
        i.memory->scratch(),i.memory,i.guard);
    check(hipStreamCreateWithFlags(&i.stream,hipStreamNonBlocking),"MTP target capture stream");
    check(hipEventCreateWithFlags(&i.copied,hipEventDisableTiming),"MTP target capture event");
    i.slots.resize(rows);
    for(auto& s:i.slots) {
        s.view.address=i.allocator->allocate_device(bytes,256);s.view.bytes=bytes;
        s.lease=i.allocator->allocation_lease_identity(s.view.address,bytes,false);
        if(!s.lease) throw std::runtime_error("MTP target capture lacks allocation lease");
        s.view.allocation_generation=s.lease;
    }
    return result;
}
HipMtpTargetHiddenTicket HipMtpTargetHidden::execute_serial(
    SerialSessionContext& session,const ExecutionRuntimeResources& resources,
    DirectLaunchDispatcher& dispatcher,std::uint32_t token) {
    auto& i=*impl_;
    if(i.busy || i.failed || session.health!=SerialSessionHealth::ready ||
       session.execution_identity!=i.compiled.execution_identity ||
       resources.scratch!=&i.memory->scratch() || resources.session_state_memory!=&i.memory->state() ||
       resources.model_weights!=&i.weights->binding() || !resources.model_manifest ||
       resources.model_manifest->layout_identity!=i.weights->binding().manifest_layout_identity ||
       token>=i.model.vocab_size || current_hip_launch_context())
        throw std::invalid_argument("MTP target capture requires idle exact retained eager target binding");
    i.guard->require_original(); i.memory->validate_inventory_ownership();
    if(i.epoch==std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("MTP target output epoch exhausted");
    auto slot=std::find_if(i.slots.begin(),i.slots.end(),[](auto& s){return s.reader.expired();});
    if(slot==i.slots.end()) throw std::runtime_error("MTP target capture slots retained by active readers");
    // Allocate host publication metadata before starting native work.
    auto lease=std::make_shared<HipMtpTargetHiddenTicket::Lease>();
    lease->owner=shared_from_this();lease->snapshot=slot->view;lease->allocation_lease=slot->lease;
    lease->position=session.next_position;lease->token=token;
    lease->content=i.weights->binding().model_content_identity;
    struct Tap final:EagerExecutionDispatcher {
        Impl& owner;DirectLaunchDispatcher& next;MtpDeviceView destination;
        std::uint64_t position;bool captured=false;
        Tap(Impl& o,DirectLaunchDispatcher& n,MtpDeviceView d,std::uint64_t p):owner(o),next(n),destination(d),position(p){}
        void after(const ExecutionDispatchContext& c) {
            if(c.step_id!=owner.tap_step) return;
            if(captured || c.execution_identity!=owner.compiled.execution_identity ||
               c.position!=position || c.logical_rows!=1 || c.op!=SerialOp::mlp_inject ||
               c.writes.size()!=1 || c.writes[0]!=owner.tap_use ||
               c.memory!=&owner.memory->scratch())
                throw std::invalid_argument("MTP target capture dispatch producer differs");
            auto source=resolve_dispatch_write(c,0);
            if(source.bytes!=destination.bytes || !source.allocation_generation)
                throw std::invalid_argument("MTP target capture source is undersized");
            check(hipMemcpyAsync(reinterpret_cast<void*>(destination.address),
                reinterpret_cast<const void*>(source.address),destination.bytes,hipMemcpyDeviceToDevice,owner.stream),"MTP target wide capture");
            check(hipEventRecord(owner.copied,owner.stream),"MTP target capture record");
            captured=true;
        }
        void dispatch_kernel(const ExecutionDispatchContext& c) override {next.dispatch_kernel(c);after(c);}
        void dispatch_runtime_step(const ExecutionDispatchContext& c) override {next.dispatch_runtime_step(c);after(c);}
    } tap(i,dispatcher,slot->view,session.next_position);
    i.busy=true;
    try {
        HipLaunchContext context;context.stream=i.stream;
        HipLaunchScope scope(context);
        (void)execute_serial_token(i.compiled,session,resources,token,tap,[&] {
            if(!tap.captured) throw std::runtime_error("MTP target hidden producer was not executed");
            check(hipEventSynchronize(i.copied),"MTP target capture completion");
            check(hipStreamSynchronize(i.stream),"MTP target execution completion");
            i.guard->require_original();
            lease->predicted=i.selector->select(i.logits,static_cast<std::uint32_t>(i.model.vocab_size),
                lease->position,++i.epoch);
        });
        lease->committed=session;
        if(session.next_position!=lease->position+1 || session.health!=SerialSessionHealth::ready)
            throw std::runtime_error("MTP target capture lacks committed target row");
        slot->reader=lease; i.busy=false;
        HipMtpTargetHiddenTicket ticket;ticket.lease_=std::move(lease);return ticket;
    } catch(...) {
        i.failed=true;i.guard->recover_or_fail_stop();i.busy=false;
        session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true;
        throw;
    }
}
} // namespace strix
