#include "strix/hip_mtp_target_hidden.hpp"
#include "strix/hip_mtp_target_window.hpp"
#include "strix/mtp_target_window_policy.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/serial_host_state_shadow.hpp"
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
namespace {
bool same_target_state(const SessionStateMemoryBinding& a,const SessionStateMemoryBinding& b) {
    return a.layout_identity==b.layout_identity && a.device_base==b.device_base && a.device_bytes==b.device_bytes &&
        a.device_allocation_generation==b.device_allocation_generation && a.host_base==b.host_base &&
        a.host_bytes==b.host_bytes && a.host_allocation_generation==b.host_allocation_generation &&
        !a.stage_mapping && !b.stage_mapping;
}
bool same_target_weights(const ModelRuntimeBinding& a,const ModelRuntimeBinding& b) {
    if(a.manifest_layout_identity!=b.manifest_layout_identity || a.model_content_identity!=b.model_content_identity ||
       a.tensors.size()!=b.tensors.size())return false;
    for(std::size_t index=0;index<a.tensors.size();++index) {
        const auto& x=a.tensors[index];const auto& y=b.tensors[index];
        if(x.tensor_index!=y.tensor_index || x.storage!=y.storage || x.address!=y.address || x.bytes!=y.bytes ||
           x.allocation_generation!=y.allocation_generation || x.owner_node!=y.owner_node)return false;
    }
    return true;
}
bool same_target_session(const SerialSessionContext& a,const SerialSessionContext& b) {
    return a.execution_identity==b.execution_identity && a.state_layout_identity==b.state_layout_identity &&
        a.runtime_compatibility_identity==b.runtime_compatibility_identity && a.next_position==b.next_position &&
        a.committed_tokens==b.committed_tokens && a.health==b.health && a.state.valid==b.state.valid &&
        a.state.generations==b.state.generations && a.arithmetic_lineage.history_identity==b.arithmetic_lineage.history_identity &&
        a.arithmetic_lineage.tainted==b.arithmetic_lineage.tainted && a.arithmetic_lineage.intervals==b.arithmetic_lineage.intervals;
}
}
struct HipMtpTargetHiddenTicket::Lease {
    std::shared_ptr<HipMtpTargetHidden> owner;
    MtpDeviceView snapshot;
    std::uint64_t allocation_lease=0,position=0;
    std::uint32_t token=0,predicted=0;
    SerialSessionContext committed;
    std::shared_ptr<const HipMtpTargetWindowBatch> batch;
    std::uint32_t batch_row=0;
    ModelContentIdentity content;
};
struct HipMtpTargetHidden::Impl {
    struct Slot {
        MtpDeviceView view;
        std::uint64_t lease=0;
        std::weak_ptr<const HipMtpTargetHiddenTicket::Lease> reader;
    };
    std::unique_ptr<const ModelContract> model;
    std::unique_ptr<const CompiledSerialExecution> compiled;
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
    std::unique_ptr<SerialHostStateShadow> shadow;
    ExecutionMemoryBinding deferred_status;
    std::uint64_t status_lease=0;
    bool busy=false,failed=false;
    ~Impl() {
        if(guard) guard->recover_or_fail_stop();
        if(copied) (void)hipEventDestroy(copied);
        if(stream) (void)hipStreamDestroy(stream);
        if(allocator && deferred_status.arena_base)
            allocator->free_device(deferred_status.arena_base,deferred_status.arena_bytes);
        if(allocator) for(auto& s:slots)
            if(s.view.address) allocator->free_device(s.view.address,s.view.bytes);
    }
};
HipMtpTargetHidden::HipMtpTargetHidden():impl_(std::make_unique<Impl>()){}
HipMtpTargetHidden::~HipMtpTargetHidden()=default;
std::uint32_t HipMtpTargetHidden::retained_rows() const noexcept {return static_cast<std::uint32_t>(impl_->slots.size());}
const ModelContract& HipMtpTargetHidden::model() const noexcept {return *impl_->model;}
const CompiledSerialExecution& HipMtpTargetHidden::compiled() const noexcept {return *impl_->compiled;}
std::shared_ptr<const RuntimeExecutionMemoryOwner> HipMtpTargetHidden::memory_owner() const noexcept {return impl_->memory;}
std::shared_ptr<const HipModelResidency> HipMtpTargetHidden::weights_owner() const noexcept {return impl_->weights;}
std::shared_ptr<HipRuntimeAllocator> HipMtpTargetHidden::allocator() const noexcept {return impl_->allocator;}
void* HipMtpTargetHidden::stream() const noexcept {return impl_->stream;}
std::shared_ptr<const HipSetupQuiescenceGuard> HipMtpTargetHidden::guard() const noexcept {return impl_->guard;}

bool HipMtpTargetHiddenTicket::valid() const noexcept {return !!lease_;}
bool HipMtpTargetHiddenTicket::same_producer(const HipMtpTargetHiddenTicket& b) const noexcept {
    return lease_ && b.lease_ && lease_->owner==b.lease_->owner;
}
bool HipMtpTargetHiddenTicket::same_commit(const HipMtpTargetHiddenTicket& b) const noexcept {
    return lease_ && b.lease_ && (lease_->batch ? lease_->batch==b.lease_->batch : lease_==b.lease_);
}
bool HipMtpTargetHiddenTicket::is_window() const noexcept {return lease_ && !!lease_->batch;}
std::uint64_t HipMtpTargetHiddenTicket::chunk_begin() const {
    if(!lease_)throw std::invalid_argument("MTP target hidden ticket is empty");
    return lease_->batch?lease_->batch->chunk_begin():lease_->position;
}
std::uint64_t HipMtpTargetHiddenTicket::chunk_end() const {
    if(!lease_)throw std::invalid_argument("MTP target hidden ticket is empty");
    return lease_->batch?lease_->batch->chunk_end():lease_->position+1;
}
MtpDeviceView HipMtpTargetHiddenTicket::view() const {
    if(!lease_ || !lease_->owner) throw std::invalid_argument("MTP target hidden ticket is empty");
    const auto& i=*lease_->owner->impl_;
    i.guard->require_original();
    i.memory->validate_inventory_ownership();
    if(lease_->batch) return lease_->batch->view(lease_->batch_row);
    const auto v=lease_->snapshot;
    if(i.allocator->allocation_lease_identity(v.address,v.bytes,false)!=lease_->allocation_lease)
        throw std::invalid_argument("MTP target hidden allocation lease changed");
    return v;
}
std::uint64_t HipMtpTargetHiddenTicket::source_position() const {if(!lease_)throw std::invalid_argument("MTP target hidden ticket is empty");return lease_->position;}
std::uint32_t HipMtpTargetHiddenTicket::input_token() const {if(!lease_)throw std::invalid_argument("MTP target hidden ticket is empty");return lease_->token;}
std::uint32_t HipMtpTargetHiddenTicket::predicted_token() const {if(!lease_)throw std::invalid_argument("MTP target hidden ticket is empty");return lease_->predicted;}
const SerialSessionContext& HipMtpTargetHiddenTicket::committed_session() const {if(!lease_)throw std::invalid_argument("MTP target hidden ticket is empty");return lease_->batch?lease_->batch->committed_session():lease_->committed;}
ModelContentIdentity HipMtpTargetHiddenTicket::target_content_identity() const {if(!lease_)throw std::invalid_argument("MTP target hidden ticket is empty");return lease_->content;}
std::shared_ptr<HipMtpTargetHidden> HipMtpTargetHidden::create(
    const ModelContract& m,const CompiledSerialExecution& c,
    std::shared_ptr<const RuntimeExecutionMemoryOwner> memory,
    std::shared_ptr<const HipModelResidency> weights,
    std::shared_ptr<HipRuntimeAllocator> allocator,std::uint32_t rows,std::uint64_t budget,HipMtpTargetDeferredConfig deferred) {
    if(!m.validate().empty() || !memory || !weights || !allocator || !rows ||
       !c.execution_identity || m.vocab_size>std::numeric_limits<std::uint32_t>::max() || c.logical_rows!=1 || c.request_concurrency!=1 ||
       !weights->binding().model_content_identity.valid())
        throw std::invalid_argument("MTP target tap requires actual single-row retained target resources");
    if(weights.owner_before(weights->weak_from_this().lock()) ||
       weights->weak_from_this().lock().owner_before(weights))
        throw std::invalid_argument("MTP target capture requires actual shared residency control block");
    (void)weights->owner_inventory(weights);
    memory->validate_inventory_ownership();
    if(!!deferred.state_layout!=!!deferred.generations)
        throw std::invalid_argument("MTP target D1 requires actual state layout and generation issuer together");
    std::uint64_t extra=0;
    if(deferred.state_layout) {
        validate_session_state_memory_binding(*deferred.state_layout,memory->state());
        if(!deferred.state_layout->host_total_bytes || !deferred.state_layout->host_alignment)
            throw std::invalid_argument("MTP target D1 requires bounded transactional host history");
        extra=mtp_native::sum(263,mtp_native::sum(deferred.state_layout->host_total_bytes,deferred.state_layout->host_alignment-1));
    }
    const auto bytes=mtp_native::product(mtp_native::product(m.hidden_size,m.hc_count),2);
    const auto selector_bytes=HipGreedyOutput::required_scratch_bytes(static_cast<std::uint32_t>(m.vocab_size));
    const auto required=mtp_native::sum(extra,mtp_native::sum(selector_bytes,
        mtp_native::product(rows,mtp_native::sum(bytes,255))));
    if(required>budget)
        throw std::invalid_argument("MTP target capture exceeds upfront retained budget");
    auto result=std::shared_ptr<HipMtpTargetHidden>(new HipMtpTargetHidden);
    auto& i=*result->impl_;
    i.model=std::make_unique<const ModelContract>(m);i.compiled=std::make_unique<const CompiledSerialExecution>(c);i.memory=std::move(memory);i.weights=std::move(weights);i.allocator=std::move(allocator);
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
    if(deferred.state_layout) {
        i.shadow=std::make_unique<SerialHostStateShadow>(*deferred.state_layout,i.memory->state(),deferred.generations->next());
        i.deferred_status={c.execution_identity,i.allocator->allocate_device(8,256),8,deferred.generations->next()};
        i.status_lease=i.allocator->allocation_lease_identity(i.deferred_status.arena_base,8,false);
        if(!i.status_lease)throw std::runtime_error("MTP target D1 lacks actual allocator lease");
        check(hipMemsetAsync(reinterpret_cast<void*>(i.deferred_status.arena_base),0,8,i.stream),"MTP target D1 initialization");
        check(hipStreamSynchronize(i.stream),"MTP target D1 setup completion");
        i.guard->require_original();
    }
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
       session.execution_identity!=i.compiled->execution_identity ||
       resources.scratch!=&i.memory->scratch() || resources.session_state_memory!=&i.memory->state() ||
       resources.model_weights!=&i.weights->binding() || !resources.model_manifest ||
       resources.model_manifest->layout_identity!=i.weights->binding().manifest_layout_identity ||
       token>=i.model->vocab_size || current_hip_launch_context())
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
            if(captured || c.execution_identity!=owner.compiled->execution_identity ||
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
    if(i.deferred_status.arena_base && i.allocator->allocation_lease_identity(i.deferred_status.arena_base,8,false)!=i.status_lease)
        throw std::invalid_argument("MTP target D1 backing lease changed");
    i.busy=true;
    try {
        auto launch_resources=resources;
        if(i.shadow){i.shadow->begin();launch_resources.session_state_memory=&i.shadow->binding();}
        HipLaunchContext context;context.stream=i.stream;context.deferred_moe_status=i.deferred_status;
        HipLaunchScope scope(context);
        (void)execute_serial_token(*i.compiled,session,launch_resources,token,tap,[&] {
            if(!tap.captured) throw std::runtime_error("MTP target hidden producer was not executed");
            DeferredMoeFirstError status{};
            if(i.deferred_status.arena_base)
                check(hipMemcpyAsync(&status,reinterpret_cast<const void*>(i.deferred_status.arena_base),sizeof(status),
                    hipMemcpyDeviceToHost,i.stream),"MTP target D1 readback");
            check(hipEventSynchronize(i.copied),"MTP target capture completion");
            check(hipStreamSynchronize(i.stream),"MTP target execution completion");
            i.guard->require_original();
            if(i.deferred_status.arena_base)check_deferred_moe_status(status);
            lease->predicted=i.selector->select(i.logits,static_cast<std::uint32_t>(i.model->vocab_size),
                lease->position,++i.epoch);
            if(i.shadow)i.shadow->publish();
        });
        lease->committed=session;
        if(session.next_position!=lease->position+1 || session.health!=SerialSessionHealth::ready)
            throw std::runtime_error("MTP target capture lacks committed target row");
        slot->reader=lease; i.busy=false;
        HipMtpTargetHiddenTicket ticket;ticket.lease_=std::move(lease);return ticket;
    } catch(...) {
        i.failed=true;i.guard->recover_or_fail_stop();if(i.shadow)i.shadow->cancel();i.busy=false;
        session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true;
        throw;
    }
}
void HipMtpTargetHidden::validate_window_capture(const HipMtpTargetWindowCapture& capture) const {
    const auto& i=*impl_;i.guard->require_original();i.memory->validate_inventory_ownership();
    if(i.busy || i.failed || capture.source_serial_execution_identity()!=i.compiled->execution_identity ||
       !same_target_state(capture.source_state_memory_binding(),i.memory->state()) ||
       !same_target_weights(capture.source_weights_binding(),i.weights->binding()))
        throw std::invalid_argument("MTP window capture does not bind the actual retained C1 target");
}
std::vector<HipMtpTargetHiddenTicket> HipMtpTargetHidden::accept_verified_window(
    std::shared_ptr<const HipMtpTargetWindowBatch> batch,const HipMtpTargetHiddenTicket& base,
    const SerialSessionContext& target,std::span<const std::uint32_t> eos,std::uint32_t budget) {
    auto& i=*impl_;i.guard->require_original();i.memory->validate_inventory_ownership();
    if(!batch || !batch->valid() || !base.lease_ || base.lease_->owner.get()!=this || i.busy || i.failed ||
       !same_target_state(batch->source_state_memory_binding(),i.memory->state()) ||
       !same_target_weights(batch->source_weights_binding(),i.weights->binding()) ||
       batch->target_content_identity()!=i.weights->binding().model_content_identity ||
       !same_target_session(batch->base_session(),base.committed_session()) ||
       !same_target_session(batch->committed_session(),target) || target.health!=SerialSessionHealth::ready ||
       batch->chunk_begin()!=base.source_position()+1 || batch->chunk_end()!=target.next_position ||
       batch->chunk_end()<=batch->chunk_begin() ||
       batch->chunk_end()-batch->chunk_begin()!=batch->logical_rows() ||
       !arithmetic_lineage_valid(target.arithmetic_lineage,target.next_position))
        throw std::invalid_argument("MTP window promotion lacks exact actual target chunk binding");
    const auto rows=batch->logical_rows();
    const auto& previous=base.committed_session();
    if(target.committed_tokens<previous.committed_tokens || target.committed_tokens-previous.committed_tokens!=rows ||
       target.arithmetic_lineage.intervals.empty())
        throw std::invalid_argument("MTP window promotion target counters are not the actual chunk advance");
    const auto& interval=target.arithmetic_lineage.intervals.back();
    if(interval.producer!=ArithmeticProducer::known_window_chunk || interval.begin!=batch->chunk_begin() ||
       interval.end!=batch->chunk_end() || interval.numerical_plan_identity!=batch->window_execution_identity())
        throw std::invalid_argument("MTP window promotion lacks its actual producer recipe");
    const auto expected=prepare_known_window_chunk_lineage(previous.arithmetic_lineage,interval.begin,interval.end,
        interval.numerical_plan_identity,interval.reduction_plan_identity,interval.state_update_identity);
    if(!arithmetic_lineage_matches(expected,target.arithmetic_lineage,target.next_position))
        throw std::invalid_argument("MTP window promotion changed preceding target producer history");
    std::vector<std::uint32_t> inputs,predictions;
    inputs.reserve(rows);predictions.reserve(rows);
    for(std::uint32_t row=0;row<rows;++row) {
        if(batch->source_position(row)!=batch->chunk_begin()+row)
            throw std::invalid_argument("MTP window source positions are not the actual contiguous chunk");
        const auto v=batch->view(row);
        if(v.bytes!=mtp_native::product(mtp_native::product(i.model->hidden_size,i.model->hc_count),2))
            throw std::invalid_argument("MTP window wide snapshot is not the actual target geometry");
        inputs.push_back(batch->input_token(row));predictions.push_back(batch->predicted_token(row));
    }
    if(!assess_mtp_target_window(inputs,predictions,base.predicted_token(),budget,eos,i.model->vocab_size).accepts_complete_chunk())
        throw std::invalid_argument("MTP provisional window is not a complete accepted greedy target chunk");
    std::vector<HipMtpTargetHiddenTicket> result;result.reserve(rows);
    for(std::uint32_t row=0;row<rows;++row) {
        auto lease=std::make_shared<HipMtpTargetHiddenTicket::Lease>();
        lease->owner=shared_from_this();lease->batch=batch;lease->batch_row=row;
        lease->position=batch->source_position(row);lease->token=inputs[row];lease->predicted=predictions[row];
        lease->content=batch->target_content_identity();
        HipMtpTargetHiddenTicket ticket;ticket.lease_=std::move(lease);result.push_back(std::move(ticket));
    }
    return result;
}
} // namespace strix
