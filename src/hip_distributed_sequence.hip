#include "strix/hip_distributed_sequence.hpp"
#include "strix/distributed_remote_control.hpp"
#include "strix/execution_memory.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {
void require_sequence(bool value,const char* reason) {if(!value) throw std::invalid_argument(reason);}
bool same_flow(const DistributedBoundaryFlow& a,const DistributedBoundaryFlow& b) {
    return a.flow_id==b.flow_id && a.direction==b.direction && a.priority==b.priority;
}
class SequenceMemory final : public DirectGpuBoundaryMemoryOwner {
public:
    std::shared_ptr<const void> owner;
    std::shared_ptr<HipDistributedTokenParticipant> participant;
    CompiledSerialExecution serial;
    ExecutionRuntimeResources resources;
    DistributedExecutionBoundary boundary;
    std::uint32_t node=0;
    std::vector<ExecutionBufferAddress> current_buffers() const override {
        if(!node) return participant->completed_boundary_views(boundary);
        // Destination metadata only: no fictitious completed producer. The
        // actual native receive proof gates every later suffix command.
        std::vector<ExecutionBufferAddress> result;
        for(const auto use:boundary.payloads)
            result.push_back(resolve_execution_buffer(serial.execution_identity,serial.arena,*resources.scratch,use));
        return result;
    }
};
struct SuffixControl final : DistributedNativeControlPlane {
    const DistributedPlacementPlan& plan;
    DistributedControlReceiver& receiver;
    DistributedRequestState& receive_state;
    SuffixControl(const DistributedPlacementPlan& p,DistributedControlReceiver& r,DistributedRequestState& s)
        :plan(p),receiver(r),receive_state(s) {}
    void execute_remote_step(const DistributedStepPlacement&) override {
        throw std::logic_error("suffix sequence cannot command a remote target");
    }
    void publish_boundary_transfer(const DistributedExecutionBoundary&,const DistributedBoundaryTransfer&) override {
        throw std::logic_error("suffix sequence cannot publish an outbound boundary");
    }
    DistributedBoundaryTransfer await_boundary_transfer(const DistributedExecutionBoundary& b) override {
        auto transfer=receiver.take_boundary(b.boundary_id);
        require_sequence(same_flow(transfer.flow,classify_distributed_boundary_flow(plan,b,{},0)),
            "sequence wire boundary does not match the canonical sender flow");
        // Validate sender metadata BEFORE converting its local direction into
        // the receiver's local flow. Descriptor/canonical plan identity stay
        // unchanged. No transport or target operation has started here.
        transfer.flow=classify_distributed_boundary_flow(plan,b,{},1);
        validate_distributed_boundary_transfer_receive(receive_state,plan,b,transfer,1);
        return transfer;
    }
};
} // namespace

struct HipDistributedKnownSequence::Impl {
    HipDistributedSequenceSetup setup;
    HipSetupQuiescenceGuard guard;
    DistributedExecutionBoundary boundary;
    std::uint32_t cut=0,last=0;
    bool used=false,started=false,failed=false,transport_live=false;
    std::uint64_t completed=0,initial_generation=0,expected_binding_identity=0;
    bool terminal_row_published=false;
    struct Local final : DistributedNativeLocalExecutor {
        Impl& owner;
        std::uint32_t next;
        bool inbound_complete=false;
        std::optional<DistributedTokenMessage> ready;
        std::shared_ptr<HipDistributedKnownInputResult> result;
        Local(Impl& o,std::shared_ptr<HipDistributedKnownInputResult> r)
            :owner(o),next(o.setup.local_node?o.cut+1U:0U),result(std::move(r)) {}
        void execute_local_step(const DistributedStepPlacement& step) override {
            const auto node=owner.setup.local_node;
            require_sequence(step.node==node && step.step_id==next && (!node || inbound_complete),
                "sequence step precedes real boundary completion or is outside ordered local stage");
            owner.setup.participant->execute_local_step(step);
            ++next;
            if(step.step_id==(node?owner.last:owner.cut)) {
                if(node) owner.read_terminal(*result);
                ready=owner.setup.participant->finish_local();
            }
        }
    };
    struct Token {
        std::shared_ptr<SequenceMemory> memory;
        std::shared_ptr<DirectGpuBoundaryStaging> staging;
        std::unique_ptr<FramedDistributedControlWire> framed;
        std::unique_ptr<SequencedDistributedNativeControlPlane> coordinator;
        std::unique_ptr<DistributedControlReceiver> receiver;
        std::unique_ptr<SuffixControl> suffix;
        std::unique_ptr<DistributedControlServer> server;
        std::unique_ptr<NativeDistributedBackend> backend;
        std::array<transport::DirectTransportNativeLinkSession*,1> links{};
        DistributedRuntimeState runtime;
        // Excludes backend itself: the backend ledger pin must not form a
        // shared ownership cycle with its own container.
        struct Pins {
            HipDistributedSequenceTransport transport;
            std::shared_ptr<const void> model;
            std::shared_ptr<SequenceMemory> memory;
            std::shared_ptr<DirectGpuBoundaryStaging> staging;
            std::shared_ptr<Local> local;
        };
        std::shared_ptr<Pins> pins;
    };
    std::unique_ptr<Token> active;
    std::optional<NativeDistributedTokenRetirement> retirement;
    // Allocated once at streaming setup. The previous proof/adapters must be
    // consumed/released BEFORE resetting this executor or writing this row.
    std::shared_ptr<Local> streaming_local;

    explicit Impl(HipDistributedSequenceSetup value):setup(std::move(value)) {
        const auto& p=setup.canonical_plan;
        require_sequence(setup.participant && setup.actual_model_owner && setup.local_node<=1U &&
            (setup.streaming || (setup.maximum_inputs>=2U && setup.maximum_inputs<=16U)) && setup.maximum_logit_payload_bytes &&
            setup.resources.scratch && setup.resources.model_weights && setup.resources.model_manifest && setup.resources.tensor_access &&
            setup.resources.session_state_layout && setup.resources.session_state_memory,
            "sequence requires bounded actual model, participant and resource owners");
        require_sequence(setup.model.validate().empty() && setup.model.vocab_size==setup.serial.vocab_size &&
            setup.model.max_position_embeddings==setup.serial.max_position_embeddings,
            "sequence model and compiled input/output bounds differ");
        require_sequence(p.local_node==0U && p.executable && p.boundaries.size()==1U &&
            p.compiled_execution_identity==setup.serial.execution_identity &&
            p.steps.size()==setup.serial.execution.steps.size() && !p.steps.empty(),
            "sequence requires a canonical two-peer plan with exactly one boundary");
        boundary=p.boundaries.front();cut=boundary.source_step;
        require_sequence(p.steps.size()<=UINT32_MAX && cut+1ULL<p.steps.size() &&
            boundary.source_node==0U && boundary.destination_node==1U && boundary.destination_step==cut+1U,
            "sequence boundary must join the last prefix step to the first suffix step");
        last=static_cast<std::uint32_t>(p.steps.size()-1U);
        for(std::uint32_t i=0;i<p.steps.size();++i)
            require_sequence(p.steps[i].step_id==i && p.steps[i].op==setup.serial.execution.steps[i].op &&
                p.steps[i].node==(i<=cut?0U:1U) && p.steps[i].remote==(i>cut) && !p.steps[i].placement_ambiguous,
                "sequence requires a complete contiguous prefix/suffix placement");
        std::vector<std::uint32_t> nodes;
        for(const auto& step:p.steps) nodes.push_back(step.node);
        const auto canonical=build_distributed_placement_plan(setup.serial,*setup.resources.model_manifest,
            *setup.resources.tensor_access,nodes,0);
        require_sequence(canonical.plan_identity==p.plan_identity && canonical.remote_nodes==p.remote_nodes &&
            canonical.has_remote_work==p.has_remote_work &&
            canonical.boundary_payload_bytes_per_token==p.boundary_payload_bytes_per_token && canonical.boundaries.size()==1U &&
            canonical.boundaries.front().boundary_id==boundary.boundary_id &&
            canonical.boundaries.front().payloads==boundary.payloads &&
            canonical.boundaries.front().payload_bytes==boundary.payload_bytes,
            "sequence canonical placement metadata or identity changed");
        const auto& binding=setup.participant->current_binding();
        require_sequence(setup.participant->phase()==DistributedTokenPhase::preparing &&
            setup.participant->local_node()==setup.local_node &&
            binding.serial_execution_identity==setup.serial.execution_identity &&
            binding.kernel_plan_identity==setup.serial.kernel_plan_identity && binding.placement_plan_identity==p.plan_identity &&
            binding.model_content_identity==setup.resources.model_weights->model_content_identity &&
            binding.peers[setup.local_node].first_step==(setup.local_node?cut+1U:0U) &&
            binding.peers[setup.local_node].last_step==(setup.local_node?last:cut),
            "sequence participant and model do not describe the same stage");
        require_sequence(setup.serial.logical_rows==1U && setup.serial.execution.steps.back().op==SerialOp::lm_head &&
            setup.serial.io.steps.size()==p.steps.size() && setup.serial.io.steps.back().writes.size()==1U &&
            setup.model.vocab_size && setup.model.vocab_size<=UINT32_MAX,
            "sequence requires one actual complete terminal FP32 head");
        const auto terminal=resolve_execution_buffer(setup.serial.execution_identity,setup.serial.arena,
            *setup.resources.scratch,setup.serial.io.steps.back().writes.front());
        require_sequence(setup.model.vocab_size<=UINT64_MAX/sizeof(float) && terminal.bytes==setup.model.vocab_size*sizeof(float),
            "sequence terminal allocation does not contain exactly one full FP32 row");
        const auto& t=setup.transport;
        require_sequence(t.actual_owner && t.link && t.link->active() && t.gpu && t.receive && t.scheduler &&
            t.step_control && t.token_control && t.step_control.get()!=t.token_control.get() &&
            t.link->lease().link_index==0U && t.scheduler->links().size()==1U,
            "sequence requires real one-link transport and distinct bounded kind5/step channels");
        t.link->require_current_retirement_lease(t.link->lease());
        const auto& profile=t.scheduler->links().front();
        require_sequence(profile.outbound.health==transport::LinkHealth::qualified &&
            profile.inbound.health==transport::LinkHealth::qualified &&
            profile.outbound.id==t.link->lease().physical_link_id && profile.inbound.id==profile.outbound.id,
            "sequence scheduler lacks the current qualified physical link");
        require_sequence(setup.scope.stream && setup.scope.stream!=hipStreamLegacy && setup.scope.stream!=hipStreamPerThread &&
            !setup.scope.graph_capture && !current_hip_launch_context(),"sequence requires its own eager HIP scope");
        initial_generation=binding.generation;
        expected_binding_identity=setup.participant->binding_identity();
        require_sequence(setup.streaming?setup.max_target_inputs>0U:setup.max_target_inputs==0U,
            "sequence streaming requires an explicit positive target limit and legacy mode requires zero");
        if(setup.streaming) {
            admit_count(binding,setup.max_target_inputs);
            require_sequence(setup.maximum_logit_payload_bytes>=setup.model.vocab_size*sizeof(float),
                "sequence streaming full FP32 row exceeds its explicit payload budget");
            auto row=std::make_shared<HipDistributedKnownInputResult>();
            if(setup.local_node) row->terminal_logits.resize(static_cast<std::size_t>(setup.model.vocab_size));
            streaming_local=std::make_shared<Local>(*this,std::move(row));
        }
    }
    void admit_count(const DistributedTokenBinding& binding,std::uint64_t count) const {
        require_sequence(count && count-1U<=UINT64_MAX-binding.generation && count<=UINT64_MAX-binding.position,
            "sequence input/frontier or generation bound is invalid");
        for(const auto& peer:binding.peers)
            require_sequence(binding.position<=peer.context_capacity && count<=peer.context_capacity-binding.position,
                "sequence exceeds a peer's context capacity");
        require_sequence(binding.position<=setup.serial.max_position_embeddings &&
            count<=setup.serial.max_position_embeddings-binding.position && count<=UINT64_MAX-binding.committed_tokens,
            "sequence exceeds compiled context or committed-token counter capacity");
    }
    void close_uncertain() noexcept {
        failed=true;
        setup.participant->communication_failed();
        guard.recover_or_fail_stop(); // Local work only; not an external DMA proof.
        if(transport_live) {
            try {setup.transport.link->close();transport_live=false;}
            catch(...) {std::terminate();} // No release proof: do not unwind DMA owners.
        }
    }
    ~Impl() {if(started && failed) close_uncertain();}
    void read_terminal(HipDistributedKnownInputResult& result) {
        guard.require_original();
        const auto view=resolve_execution_buffer(setup.serial.execution_identity,setup.serial.arena,
            *setup.resources.scratch,setup.serial.io.steps.back().writes.front());
        require_sequence(view.bytes==setup.model.vocab_size*sizeof(float) && result.terminal_logits.size()==setup.model.vocab_size,
            "sequence terminal output is not a full FP32 vocabulary row");
        const auto status=hipMemcpyAsync(result.terminal_logits.data(),reinterpret_cast<const void*>(view.address),
            static_cast<std::size_t>(view.bytes),hipMemcpyDeviceToHost,setup.scope.stream);
        if(status!=hipSuccess || hipStreamSynchronize(setup.scope.stream)!=hipSuccess)
            throw std::runtime_error("sequence terminal FP32 completion failed");
        guard.require_original();
        result.terminal_firstmax=0;
        for(std::uint32_t i=0;i<result.terminal_logits.size();++i) {
            require_sequence(std::isfinite(result.terminal_logits[i]),"sequence terminal FP32 row contains a nonfinite value");
            if(result.terminal_logits[i]>result.terminal_logits[result.terminal_firstmax]) result.terminal_firstmax=i;
        }
    }
    void exchange(FramedDistributedTokenWire& wire,const DistributedTokenMessage& own) {
        if(!setup.local_node) {setup.participant->send(wire,own);setup.participant->receive(wire);}
        else {setup.participant->receive(wire);setup.participant->send(wire,own);}
    }
    void adapters(const std::shared_ptr<Local>& local) {
        auto token=std::make_unique<Token>();
        const auto& b=setup.participant->current_binding();
        const DistributedRuntimeRequest request{b.connection_epoch,b.request_id,b.generation,b.boundary_dtype_id};
        token->runtime=make_distributed_runtime_state(setup.canonical_plan,request);
        token->memory=std::make_shared<SequenceMemory>();
        auto& memory=*token->memory;memory.owner=setup.actual_model_owner;memory.participant=setup.participant;
        memory.serial=setup.serial;memory.resources=setup.resources;memory.boundary=boundary;memory.node=setup.local_node;
        token->staging=std::make_shared<DirectGpuBoundaryStaging>(setup.canonical_plan,request,setup.transport.link->lease(),
            token->memory,setup.transport.gpu,setup.transport.receive,setup.local_node);
        DistributedNativeControlPlane* control=nullptr;
        if(!setup.local_node) {
            token->framed=std::make_unique<FramedDistributedControlWire>(*setup.transport.step_control);
            token->coordinator=std::make_unique<SequencedDistributedNativeControlPlane>(setup.canonical_plan,request,*token->framed);
            control=token->coordinator.get();
        } else {
            token->receiver=std::make_unique<DistributedControlReceiver>(setup.canonical_plan,request,1U,*local);
            token->suffix=std::make_unique<SuffixControl>(setup.canonical_plan,*token->receiver,token->runtime.receive);
            token->server=std::make_unique<DistributedControlServer>(*setup.transport.step_control,*token->receiver);
            control=token->suffix.get();
        }
        token->links[0]=setup.transport.link.get();
        token->backend=std::make_unique<NativeDistributedBackend>(token->links,*control,*local,*token->staging);
        token->pins=std::make_shared<Token::Pins>(Token::Pins{setup.transport,setup.actual_model_owner,token->memory,token->staging,local});
        token->backend->begin_token_retirement(setup.canonical_plan,request,setup.local_node,
            setup.participant->binding_identity(),token->pins);
        active=std::move(token);
        transport_live=true;
    }
    // Shared real one-input protocol. The legacy wrapper preallocates every
    // requested row; streaming supplies the same setup-allocated row each time.
    // Caller serializes access, admits count/input and owns the error boundary.
    void execute_one(const std::shared_ptr<Local>& local,std::uint32_t input) {
        require_sequence(setup.participant->binding_identity()==expected_binding_identity &&
            setup.participant->phase()==(completed?DistributedTokenPhase::committed:DistributedTokenPhase::preparing),
            "sequence participant changed outside its retained transaction owner");
        terminal_row_published=false;
        if(completed) {
            require_sequence(retirement.has_value(),"sequence has no actual previous-token retirement");
            setup.participant->begin_next(input,initial_generation+completed,std::move(*retirement));
            retirement.reset();active.reset();
        } else require_sequence(input==setup.participant->current_binding().input_token,
            "sequence first input differs from its negotiated binding");
        // No previous backend/proof can retain mutable executor state now.
        local->next=setup.local_node?cut+1U:0U;
        local->inbound_complete=false;local->ready.reset();
        auto& row=*local->result;row.input_token=input;row.commit={};row.terminal_firstmax=0;
        started=true;
        FramedDistributedTokenWire wire(*setup.transport.token_control);
        auto prepared=setup.participant->prepare();exchange(wire,prepared);
        require_sequence(setup.participant->phase()==DistributedTokenPhase::executing,"sequence peers are not both prepared");
        if(!setup.local_node) {
            for(std::uint32_t step=0;step<=cut;++step) local->execute_local_step(setup.canonical_plan.steps[step]);
            // Creating producer staging is intentionally AFTER finish_local.
            adapters(local);
            const auto& b=setup.participant->current_binding();
            auto transfer=plan_distributed_boundary_transfer(setup.canonical_plan,boundary,*setup.transport.scheduler,
                b.connection_epoch,b.request_id,b.generation,1U,b.boundary_dtype_id);
            require_sequence(transfer.link_plan.kind==transport::PlanKind::single_link && transfer.link_plan.slices.size()==1U,
                "sequence has no qualified single-link boundary plan");
            active->backend->send_boundary(boundary,transfer);
            for(std::uint32_t step=cut+1U;step<=last;++step)
                active->backend->execute_remote_step(setup.canonical_plan.steps[step]);
        } else {
            adapters(local);
            require_sequence(active->server->serve_one()==DistributedControlServerEvent::boundary_transfer,
                "sequence suffix requires boundary metadata before any remote step");
            (void)active->backend->receive_boundary(boundary);
            local->inbound_complete=true;
            for(std::uint32_t step=cut+1U;step<=last;++step)
                require_sequence(active->server->serve_one()==DistributedControlServerEvent::remote_step,
                    "sequence suffix received a duplicate boundary or unexpected frame");
        }
        require_sequence(local->ready.has_value(),"sequence local stage has no completed readiness receipt");
        exchange(wire,*local->ready);
        if(!setup.local_node) {const auto decision=setup.participant->decide_commit();setup.participant->send(wire,decision);}
        else setup.participant->receive(wire);
        const auto ack=setup.participant->acknowledge_commit();exchange(wire,ack);
        const auto committed=setup.participant->committed_frontier();
        require_sequence(committed.has_value() && setup.participant->frontier().has_value(),
            "sequence cannot publish without both committed ACKs and typed frontier");
        retirement.emplace(active->backend->retire_token());
        row.commit=*committed; // Only after real transport retirement.
        expected_binding_identity=setup.participant->binding_identity();
        ++completed;
        terminal_row_published=setup.streaming && setup.local_node==1U;
    }
};

HipDistributedKnownSequence::HipDistributedKnownSequence(HipDistributedSequenceSetup setup)
    :impl_(std::make_unique<Impl>(std::move(setup))) {}
HipDistributedKnownSequence::~HipDistributedKnownSequence()=default;
HipDistributedSequenceResult HipDistributedKnownSequence::execute(std::span<const std::uint32_t> inputs) {
    auto& s=*impl_;
    require_sequence(!s.setup.streaming && !s.used && !s.failed && inputs.size()>=2U && inputs.size()<=s.setup.maximum_inputs,
        "sequence is single-use and requires bounded two-or-more known inputs");
    const auto binding=s.setup.participant->current_binding();
    require_sequence(inputs.front()==binding.input_token && inputs.size()-1U<=UINT64_MAX-binding.generation &&
        inputs.size()<=UINT64_MAX-binding.position,"sequence input/frontier or generation bound is invalid");
    for(const auto token:inputs) require_sequence(token<s.setup.model.vocab_size,"sequence input is outside vocabulary");
    for(const auto& peer:binding.peers)
        require_sequence(binding.position<=peer.context_capacity && inputs.size()<=peer.context_capacity-binding.position,
            "sequence exceeds a peer's context capacity");
    require_sequence(binding.position<=s.setup.serial.max_position_embeddings &&
        inputs.size()<=s.setup.serial.max_position_embeddings-binding.position &&
        inputs.size()<=UINT64_MAX-binding.committed_tokens,
        "sequence exceeds compiled context or committed-token counter capacity");
    require_sequence(s.setup.model.vocab_size<=UINT64_MAX/sizeof(float) &&
        inputs.size()<=s.setup.maximum_logit_payload_bytes/(s.setup.model.vocab_size*sizeof(float)),
        "sequence full FP32 result exceeds its explicit payload budget");
    HipDistributedSequenceResult result;result.local_node=s.setup.local_node;
    result.inputs.resize(inputs.size());
    std::vector<std::shared_ptr<Impl::Local>> locals;
    locals.reserve(inputs.size());
    for(std::size_t i=0;i<inputs.size();++i) {
        auto row=std::make_shared<HipDistributedKnownInputResult>();
        row->input_token=inputs[i];
        if(s.setup.local_node) row->terminal_logits.resize(static_cast<std::size_t>(s.setup.model.vocab_size));
        locals.push_back(std::make_shared<Impl::Local>(s,std::move(row)));
    }
    // All result payloads allocated before snapshot/PLE/target mutation.
    s.used=true;
    try {
        for(std::size_t token=0;token<inputs.size();++token) {
            s.execute_one(locals[token],inputs[token]);
            result.inputs[token]=std::move(*locals[token]->result);
        }
        result.coordinator_confirmation=s.setup.local_node==0U;
        return result;
    } catch(...) {s.close_uncertain();throw;}
}
HipDistributedInputCompletion HipDistributedKnownSequence::execute_input(std::uint32_t input) {
    auto& s=*impl_;
    require_sequence(s.setup.streaming && !s.failed && s.completed<s.setup.max_target_inputs &&
        input<s.setup.model.vocab_size,"sequence streaming target is outside its admitted mode, count or vocabulary");
    if(!s.completed) require_sequence(input==s.setup.participant->current_binding().input_token,
        "sequence first input differs from its negotiated binding");
    try {
        s.execute_one(s.streaming_local,input);
        const auto& row=*s.streaming_local->result;
        HipDistributedInputCompletion result;
        result.commit=row.commit;result.input_token=input;result.terminal_peer=s.setup.local_node==1U;
        if(result.terminal_peer) {
            result.terminal_firstmax=row.terminal_firstmax;
            result.terminal_firstmax_bits=std::bit_cast<std::uint32_t>(row.terminal_logits[row.terminal_firstmax]);
        }
        return result;
    } catch(...) {s.close_uncertain();throw;}
}
std::span<const float> HipDistributedKnownSequence::last_terminal_logits() const noexcept {
    const auto& s=*impl_;
    if(s.failed || !s.terminal_row_published || !s.streaming_local) return {};
    return s.streaming_local->result->terminal_logits;
}
std::uint64_t HipDistributedKnownSequence::completed_inputs() const noexcept {return impl_->completed;}
} // namespace strix
