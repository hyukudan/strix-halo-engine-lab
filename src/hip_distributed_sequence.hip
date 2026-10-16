#include "strix/hip_distributed_sequence.hpp"
#include "strix/distributed_remote_control.hpp"
#include "strix/execution_memory.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <tuple>
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
bool same_boundary(const DistributedExecutionBoundary& a,const DistributedExecutionBoundary& b) {
    return a.boundary_id==b.boundary_id && a.source_step==b.source_step && a.destination_step==b.destination_step &&
        a.source_node==b.source_node && a.destination_node==b.destination_node && a.payload_bytes==b.payload_bytes && a.payloads==b.payloads;
}
void validate_target_limit(const DistributedTokenBinding& binding,std::uint64_t capacity,std::uint64_t count) {
    require_sequence(count && count-1U<=UINT64_MAX-binding.generation && count<=UINT64_MAX-binding.position,
        "sequence input/frontier or generation bound is invalid");
    for(const auto& peer:binding.peers)
        require_sequence(binding.position<=peer.context_capacity && count<=peer.context_capacity-binding.position,
            "sequence exceeds a peer's context capacity");
    require_sequence(binding.position<=capacity && count<=capacity-binding.position && count<=UINT64_MAX-binding.committed_tokens,
        "sequence exceeds compiled context or committed-token counter capacity");
}
bool same_resources(const ExecutionRuntimeResources& a,const ExecutionRuntimeResources& b) {
    return a.scratch==b.scratch && a.model_manifest==b.model_manifest && a.model_weights==b.model_weights &&
        a.tensor_access==b.tensor_access && a.session_state_layout==b.session_state_layout &&
        a.session_state_memory==b.session_state_memory && a.ple_lookup==b.ple_lookup &&
        a.ple_hash_contract==b.ple_hash_contract && a.ple_prefetch==b.ple_prefetch && a.distributed_stage==b.distributed_stage;
}
bool same_memory(const ExecutionMemoryBinding& a,const ExecutionMemoryBinding& b) {
    return a.execution_identity==b.execution_identity && a.arena_base==b.arena_base && a.arena_bytes==b.arena_bytes &&
        a.allocation_generation==b.allocation_generation;
}
bool same_scope(const HipLaunchContext& a,const HipLaunchContext& b) {
    return a.stream==b.stream && a.graph_capture==b.graph_capture && a.arguments==b.arguments &&
        same_memory(a.status,b.status) && a.context_tokens==b.context_tokens && a.vocabulary_size==b.vocabulary_size &&
        same_memory(a.ngram_input,b.ngram_input) && same_memory(a.ngram_output,b.ngram_output) &&
        same_memory(a.deferred_moe_status,b.deferred_moe_status);
}
bool same_model(const ModelContract& a,const ModelContract& b) {
    return a.vocab_size==b.vocab_size && a.hidden_size==b.hidden_size && a.max_position_embeddings==b.max_position_embeddings &&
        a.hidden_layers==b.hidden_layers && a.full_attention_interval==b.full_attention_interval && a.ple_layer_1based==b.ple_layer_1based &&
        a.ple_embed_dim==b.ple_embed_dim && a.ple_conv_kernel_size==b.ple_conv_kernel_size && a.ngram_size==b.ngram_size &&
        a.heads_per_ngram==b.heads_per_ngram && a.ngram_vocab_size_base==b.ngram_vocab_size_base && a.ngram_vocab_divisor==b.ngram_vocab_divisor &&
        a.ngram_hash_seed==b.ngram_hash_seed && a.ple_eos_token_id==b.ple_eos_token_id && a.hc_count==b.hc_count &&
        a.hc_lowrank==b.hc_lowrank && a.full_attention_layers==b.full_attention_layers && a.gdn_layers==b.gdn_layers &&
        a.attention_heads==b.attention_heads && a.kv_heads==b.kv_heads && a.head_dim==b.head_dim &&
        a.mtp_full_attention_layers==b.mtp_full_attention_layers && a.mtp_use_dedicated_embeddings==b.mtp_use_dedicated_embeddings && a.mtp_hidden_state_source_layer_explicit==b.mtp_hidden_state_source_layer_explicit &&
        a.gdn_key_heads==b.gdn_key_heads && a.gdn_key_head_dim==b.gdn_key_head_dim && a.gdn_value_heads==b.gdn_value_heads &&
        a.gdn_value_head_dim==b.gdn_value_head_dim && a.gdn_conv_kernel==b.gdn_conv_kernel && a.indexer_n_heads==b.indexer_n_heads &&
        a.indexer_kv_heads==b.indexer_kv_heads && a.indexer_head_dim==b.indexer_head_dim && a.indexer_budget==b.indexer_budget &&
        a.indexer_compress_ratio==b.indexer_compress_ratio && std::bit_cast<std::uint32_t>(a.partial_rotary_factor)==std::bit_cast<std::uint32_t>(b.partial_rotary_factor) && std::bit_cast<std::uint32_t>(a.rope_theta)==std::bit_cast<std::uint32_t>(b.rope_theta) &&
        a.moe_intermediate_size==b.moe_intermediate_size && a.shared_expert_intermediate_size==b.shared_expert_intermediate_size && a.num_experts==b.num_experts &&
        a.num_experts_per_token==b.num_experts_per_token && std::bit_cast<std::uint32_t>(a.rms_norm_epsilon)==std::bit_cast<std::uint32_t>(b.rms_norm_epsilon) && a.norm_topk_probability==b.norm_topk_probability &&
        a.gdn_output_gate_sigmoid==b.gdn_output_gate_sigmoid && a.gdn_qk_l2norm==b.gdn_qk_l2norm && std::bit_cast<std::uint32_t>(a.gdn_qk_l2_epsilon)==std::bit_cast<std::uint32_t>(b.gdn_qk_l2_epsilon) &&
        a.gdn_conv_silu==b.gdn_conv_silu && a.bytes_bf16==b.bytes_bf16 && a.bytes_fp32==b.bytes_fp32;
}
bool same_compiled(const CompiledSerialExecution& a,const CompiledSerialExecution& b) {
    if(a.execution_identity!=b.execution_identity || a.kernel_plan_identity!=b.kernel_plan_identity ||
        a.dynamic_steps!=b.dynamic_steps || a.non_graph_safe_steps!=b.non_graph_safe_steps ||
        a.whole_plan_graph_safe!=b.whole_plan_graph_safe || a.device_graph_inputs!=b.device_graph_inputs ||
        a.logical_rows!=b.logical_rows || a.request_concurrency!=b.request_concurrency ||
        a.vocab_size!=b.vocab_size || a.max_position_embeddings!=b.max_position_embeddings ||
        a.execution.steps.size()!=b.execution.steps.size() || a.io.steps.size()!=b.io.steps.size() ||
        a.bindings.bindings.size()!=b.bindings.bindings.size() || a.arena.allocations.size()!=b.arena.allocations.size()) return false;
    if(std::tie(a.execution.linear_attention_steps,a.execution.indexed_attention_steps,a.execution.ple_steps,
        a.execution.moe_route_steps,a.execution.moe_expert_steps)!=
        std::tie(b.execution.linear_attention_steps,b.execution.indexed_attention_steps,b.execution.ple_steps,
        b.execution.moe_route_steps,b.execution.moe_expert_steps) ||
        std::tie(a.bindings.bound_steps,a.bindings.missing_steps,a.bindings.host_or_runtime_steps)!=
        std::tie(b.bindings.bound_steps,b.bindings.missing_steps,b.bindings.host_or_runtime_steps) ||
        std::tie(a.arena.alignment,a.arena.total_bytes,a.arena.naive_nonaliased_bytes)!=
        std::tie(b.arena.alignment,b.arena.total_bytes,b.arena.naive_nonaliased_bytes) ||
        std::tie(a.io.expanded_hidden_slots,a.io.hidden_slots,a.io.injection_slots,a.io.route_index_slots,a.io.route_weight_slots,
        a.io.logits_slots,a.io.mlp_intermediate_slots,a.io.gdn_workspace_slots,a.io.ple_workspace_slots,
        a.io.indexed_attention_workspace_slots,a.io.hyper_workspace_slots,a.io.router_workspace_slots)!=
        std::tie(b.io.expanded_hidden_slots,b.io.hidden_slots,b.io.injection_slots,b.io.route_index_slots,b.io.route_weight_slots,
        b.io.logits_slots,b.io.mlp_intermediate_slots,b.io.gdn_workspace_slots,b.io.ple_workspace_slots,
        b.io.indexed_attention_workspace_slots,b.io.hyper_workspace_slots,b.io.router_workspace_slots)) return false;
    for(std::size_t i=0;i<a.execution.steps.size();++i) {
        const auto& x=a.execution.steps[i];const auto& y=b.execution.steps[i];
        if(std::tie(x.id,x.op,x.layer,x.dependencies,x.reads_state,x.writes_state,x.graph_safe_candidate,x.dynamic_shape)!=
            std::tie(y.id,y.op,y.layer,y.dependencies,y.reads_state,y.writes_state,y.graph_safe_candidate,y.dynamic_shape)) return false;
    }
    for(std::size_t i=0;i<a.io.steps.size();++i) {
        const auto& x=a.io.steps[i];const auto& y=b.io.steps[i];
        if(std::tie(x.step_id,x.op,x.reads,x.writes,x.temporaries)!=std::tie(y.step_id,y.op,y.reads,y.writes,y.temporaries)) return false;
    }
    for(std::size_t i=0;i<a.bindings.bindings.size();++i) {
        const auto& x=a.bindings.bindings[i];const auto& y=b.bindings.bindings[i];
        if(std::tie(x.step_id,x.op,x.family,x.kernel_id,x.host_or_runtime_step,x.tuning_selection_reason)!=
            std::tie(y.step_id,y.op,y.family,y.kernel_id,y.host_or_runtime_step,y.tuning_selection_reason)) return false;
    }
    for(std::size_t i=0;i<a.arena.allocations.size();++i) {
        const auto& x=a.arena.allocations[i];const auto& y=b.arena.allocations[i];
        if(std::tie(x.use,x.first_step,x.last_step,x.offset,x.bytes)!=std::tie(y.use,y.first_step,y.last_step,y.offset,y.bytes)) return false;
    }
    return true;
}
void validate_stage_setup(const HipDistributedSequenceSetup& setup) {
    if(!setup.stage_owner) {
        require_sequence(!setup.resources.distributed_stage &&
            (!setup.resources.session_state_memory || !setup.resources.session_state_memory->stage_mapping) &&
            (!setup.participant || !setup.participant->stage_resources()),
            "sequence scoped resources require their explicit actual stage owner");
        return;
    }
    const auto& owner=*setup.stage_owner;owner.validate();
    const std::shared_ptr<const void> actual=setup.stage_owner->shared_from_this();
    require_sequence(!actual.owner_before(setup.stage_owner) && !setup.stage_owner.owner_before(actual) &&
        setup.actual_model_owner.get()==actual.get() && !actual.owner_before(setup.actual_model_owner) &&
        !setup.actual_model_owner.owner_before(actual) && setup.participant && setup.participant->stage_resources()==&owner,
        "sequence stage participant and retained ownership do not match the actual owner");
    require_sequence(setup.local_node==owner.plan().local_node() &&
        setup.canonical_plan.plan_identity==owner.plan().placement_identity() &&
        same_resources(setup.resources,owner.resources()) && same_scope(setup.scope,owner.scope()) &&
        same_compiled(setup.serial,owner.serial()) && same_model(setup.model,owner.model()),
        "sequence stage metadata/resources/scope differ from the immutable actual owner");
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
    std::size_t position_dependencies=0;
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
        validate_stage_setup(setup);
        const auto& p=setup.canonical_plan;
        require_sequence(setup.participant && setup.actual_model_owner && setup.local_node<=1U &&
            (setup.streaming || (setup.maximum_inputs>=2U && setup.maximum_inputs<=16U)) && setup.maximum_logit_payload_bytes &&
            setup.resources.scratch && setup.resources.model_weights && setup.resources.model_manifest && setup.resources.tensor_access &&
            setup.resources.session_state_layout && setup.resources.session_state_memory,
            "sequence requires bounded actual model, participant and resource owners");
        require_sequence(setup.model.validate().empty() && setup.model.vocab_size==setup.serial.vocab_size &&
            setup.model.max_position_embeddings==setup.serial.max_position_embeddings,
            "sequence model and compiled input/output bounds differ");
        require_sequence(p.local_node==0U && p.executable &&
            (setup.layer_boundary_position_dependencies?(!p.boundaries.empty() &&
                p.boundaries.size()<=NativeDistributedPositionCompletion::maximum_dependencies+1U):p.boundaries.size()==1U) &&
            p.compiled_execution_identity==setup.serial.execution_identity &&
            p.steps.size()==setup.serial.execution.steps.size() && !p.steps.empty(),
            setup.layer_boundary_position_dependencies?
                "sequence requires one data boundary and at most64 logical dependencies in a canonical two-peer plan":
                "sequence requires a canonical two-peer plan with exactly one boundary");
        if(setup.layer_boundary_position_dependencies) {
            std::size_t data=0;
            for(const auto& candidate:p.boundaries) if(candidate.payload_bytes && !candidate.payloads.empty()) {
                boundary=candidate;++data;
            }
            require_sequence(data==1U,"sequence layer boundary requires exactly one data boundary");
        } else boundary=p.boundaries.front();
        cut=boundary.source_step;
        require_sequence(p.steps.size()<=UINT32_MAX && cut+1ULL<p.steps.size() &&
            boundary.source_node==0U && boundary.destination_node==1U && boundary.destination_step==cut+1U,
            "sequence boundary must join the last prefix step to the first suffix step");
        last=static_cast<std::uint32_t>(p.steps.size()-1U);
        for(std::uint32_t i=0;i<p.steps.size();++i)
            require_sequence(p.steps[i].step_id==i && p.steps[i].op==setup.serial.execution.steps[i].op &&
                p.steps[i].node==(i<=cut?0U:1U) && p.steps[i].remote==(i>cut) && !p.steps[i].placement_ambiguous,
                "sequence requires a complete contiguous prefix/suffix placement");
        if(setup.layer_boundary_position_dependencies) {
            const auto& source=setup.serial.execution.steps[cut];
            const auto& destination=setup.serial.execution.steps[cut+1U];
            require_sequence(source.op==SerialOp::mlp_inject && source.layer.has_value() &&
                destination.op==SerialOp::attn_hyper_mix && destination.layer.has_value() &&
                *source.layer<UINT32_MAX && *destination.layer==*source.layer+1U,
                "sequence layer cut must join complete adjacent layers without a PLE or interior-layer crossing");
            for(std::size_t i=0;i<p.boundaries.size();++i) {
                require_sequence(p.boundaries[i].boundary_id!=0U,"sequence canonical boundary identity is empty or duplicated");
                for(std::size_t j=0;j<i;++j)
                    require_sequence(p.boundaries[i].boundary_id!=p.boundaries[j].boundary_id,
                        "sequence canonical boundary identity is empty or duplicated");
            }
            for(const auto& candidate:p.boundaries) if(candidate.boundary_id!=boundary.boundary_id) {
                require_sequence(is_materialized_input_position_dependency(p,candidate) && candidate.source_step<=cut &&
                    candidate.destination_step>cut,"sequence layer cut has an unsupported crossing dependency");
                ++position_dependencies;
            }
            require_sequence(position_dependencies<=NativeDistributedPositionCompletion::maximum_dependencies,
                "sequence layer cut has too many logical position dependencies");
        }
        std::vector<std::uint32_t> nodes;
        for(const auto& step:p.steps) nodes.push_back(step.node);
        const auto canonical=build_distributed_placement_plan(setup.serial,*setup.resources.model_manifest,
            *setup.resources.tensor_access,nodes,0);
        require_sequence(canonical.plan_identity==p.plan_identity && canonical.remote_nodes==p.remote_nodes &&
            canonical.manifest_layout_identity==p.manifest_layout_identity && canonical.tensor_access_identity==p.tensor_access_identity &&
            canonical.executable==p.executable &&
            canonical.has_remote_work==p.has_remote_work &&
            canonical.boundary_payload_bytes_per_token==p.boundary_payload_bytes_per_token &&
            canonical.boundaries.size()==p.boundaries.size() &&
            std::equal(canonical.boundaries.begin(),canonical.boundaries.end(),p.boundaries.begin(),same_boundary),
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
        validate_target_limit(binding,setup.serial.max_position_embeddings,count);
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
        if(setup.layer_boundary_position_dependencies)
            token->staging=std::make_shared<DirectGpuBoundaryStaging>(setup.canonical_plan,request,setup.transport.link->lease(),
                token->memory,setup.transport.gpu,setup.transport.receive,setup.local_node,boundary.boundary_id,
                NativeDistributedPositionDependencyPolicy::materialized_input_position_v1);
        else token->staging=std::make_shared<DirectGpuBoundaryStaging>(setup.canonical_plan,request,setup.transport.link->lease(),
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
            setup.participant->binding_identity(),token->pins,setup.layer_boundary_position_dependencies?
                NativeDistributedPositionDependencyPolicy::materialized_input_position_v1:NativeDistributedPositionDependencyPolicy::reject);
        active=std::move(token);
        transport_live=true;
    }
    // Shared real one-input protocol. The legacy wrapper preallocates every
    // requested row; streaming supplies the same setup-allocated row each time.
    // Caller serializes access, admits count/input and owns the error boundary.
    void execute_one(const std::shared_ptr<Local>& local,std::uint32_t input) {
        if(setup.stage_owner) setup.stage_owner->validate();
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
        if(position_dependencies)
            active->backend->complete_position_dependencies(setup.participant->mint_position_dependency_completion());
        retirement.emplace(active->backend->retire_token());
        row.commit=*committed; // Only after real transport retirement.
        expected_binding_identity=setup.participant->binding_identity();
        ++completed;
        terminal_row_published=setup.streaming && setup.local_node==1U;
    }
};

HipDistributedSequenceSetup make_hip_distributed_stage_sequence_setup(
    std::shared_ptr<HipDistributedStageResources> owner,DistributedTokenBinding binding,
    HipDistributedSequenceTransport transport,HipDistributedStageSequenceOptions options) {
    require_sequence(owner && options.maximum_logit_payload_bytes &&
        (options.streaming?options.max_target_inputs>0U:
            (options.max_target_inputs==0U && options.maximum_inputs>=2U && options.maximum_inputs<=16U)),
        "stage sequence factory requires an actual owner and explicit bounded input/row policy");
    owner->validate();
    const auto actual=owner->shared_from_this();
    require_sequence(!owner.owner_before(actual) && !actual.owner_before(owner),
        "stage sequence factory requires the actual retained ownership group");
    const auto vocab=owner->model().vocab_size;
    require_sequence(vocab && vocab<=UINT64_MAX/sizeof(float) &&
        options.maximum_logit_payload_bytes/(vocab*sizeof(float))>=(options.streaming?1U:2U),
        "stage sequence factory full FP32 row exceeds its explicit payload budget");
    if(options.streaming) validate_target_limit(binding,owner->serial().max_position_embeddings,options.max_target_inputs);
    HipDistributedSequenceSetup setup;
    setup.stage_owner=owner;setup.serial=owner->serial();setup.canonical_plan=owner->plan().canonical_placement();
    setup.model=owner->model();setup.resources=owner->resources();setup.scope=owner->scope();setup.actual_model_owner=owner;
    setup.transport=std::move(transport);setup.local_node=owner->plan().local_node();
    setup.maximum_inputs=options.maximum_inputs;setup.maximum_logit_payload_bytes=options.maximum_logit_payload_bytes;
    setup.streaming=options.streaming;setup.max_target_inputs=options.max_target_inputs;
    setup.layer_boundary_position_dependencies=options.layer_boundary_position_dependencies;
    setup.participant=std::make_shared<HipDistributedTokenParticipant>(std::move(owner),std::move(binding));
    return setup;
}

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
