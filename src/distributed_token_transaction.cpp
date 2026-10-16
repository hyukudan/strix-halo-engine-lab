#include "strix/distributed_token_transaction.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix {
namespace {
constexpr std::uint64_t offset=14695981039346656037ULL,prime=1099511628211ULL;
void hash(std::uint64_t& h,std::uint64_t n) noexcept {
    for(unsigned s=0;s<64;s+=8) {h^=(n>>s)&255U;h*=prime;}
}
void state_hash(std::uint64_t& h,const PersistentStateGenerations& s) noexcept {
    hash(h,static_cast<std::uint32_t>(s.valid));for(const auto g:s.generations) hash(h,g);
}
bool same_state(const PersistentStateGenerations& a,const PersistentStateGenerations& b) noexcept {
    return a.valid==b.valid && a.generations==b.generations;
}
bool zero_receipt(const DistributedTokenReceipt& r) noexcept {
    return !r.snapshot_plan_identity && !r.snapshot_generation && !r.completed_steps &&
        r.local_state.valid==PersistentState::none && r.local_state.generations==std::array<std::uint64_t,6>{};
}
constexpr auto all_state=PersistentState::attention_kv|PersistentState::recurrent_matrix|
    PersistentState::convolution_history|PersistentState::sparse_index|PersistentState::ngram_history|PersistentState::position_history;
}
std::uint64_t distributed_token_binding_identity(const DistributedTokenBinding& b) noexcept {
    auto h=offset;hash(h,0x4454584e544f4b31ULL);hash(h,b.contract_version);
    hash(h,b.serial_execution_identity);hash(h,b.kernel_plan_identity);hash(h,b.placement_plan_identity);
    for(const auto byte:b.model_content_identity.bytes) hash(h,byte);
    hash(h,b.request_id);hash(h,b.connection_epoch);hash(h,b.generation);hash(h,b.position);
    hash(h,b.committed_tokens);hash(h,b.previous_history_identity);hash(h,b.input_token);hash(h,b.boundary_dtype_id);state_hash(h,b.initial_global_state);
    for(const auto& p:b.peers) {
        hash(h,p.node);hash(h,p.first_step);hash(h,p.last_step);hash(h,p.runtime_compatibility_identity);
        hash(h,p.state_layout_identity);hash(h,p.context_capacity);state_hash(h,p.initial_state);
    }
    return h;
}
DistributedTokenTransaction::DistributedTokenTransaction(const CompiledSerialExecution& serial,
    const DistributedPlacementPlan& plan,const ModelManifest& manifest,const ExecutionTensorAccessPlan& access,
    DistributedTokenBinding binding):binding_(std::move(binding)),expected_global_(binding_.initial_global_state) {
    const auto& b=binding_;
    if(b.contract_version!=1U || !b.model_content_identity.valid() || !b.request_id || !b.connection_epoch || !b.generation || !b.boundary_dtype_id ||
        serial.logical_rows!=1U || serial.request_concurrency!=1U || serial.device_graph_inputs ||
        b.serial_execution_identity!=serial.execution_identity || !serial.execution_identity ||
        b.kernel_plan_identity!=serial.kernel_plan_identity || !serial.kernel_plan_identity ||
        b.placement_plan_identity!=plan.plan_identity || plan.local_node!=0U || plan.steps.empty() || serial.execution.steps.size()>std::numeric_limits<std::uint32_t>::max() ||
        b.input_token>=serial.vocab_size || b.position>=serial.max_position_embeddings ||
        b.position==std::numeric_limits<std::uint64_t>::max() || b.committed_tokens==std::numeric_limits<std::uint64_t>::max() ||
        (b.position!=0U && !b.previous_history_identity) || !b.initial_global_state.contains(all_state) ||
        (static_cast<std::uint32_t>(b.initial_global_state.valid)&~63U)!=0U)
        throw std::invalid_argument("distributed token transaction requires a bound known input and complete logical frontier");
    std::vector<std::uint32_t> nodes;for(const auto& step:plan.steps) nodes.push_back(step.node);
    const auto derived=build_distributed_placement_plan(serial,manifest,access,nodes,0U);
    if(!plan.executable || !plan.has_remote_work || plan.plan_identity!=derived.plan_identity ||
        plan.compiled_execution_identity!=derived.compiled_execution_identity ||
        plan.manifest_layout_identity!=derived.manifest_layout_identity || plan.tensor_access_identity!=derived.tensor_access_identity ||
        plan.steps.size()!=serial.execution.steps.size() || plan.boundaries.size()!=derived.boundaries.size())
        throw std::invalid_argument("distributed transaction placement is not derived from its serial source");
    for(std::size_t i=0;i<plan.steps.size();++i) {
        const auto& a=plan.steps[i];const auto& d=derived.steps[i];
        if(a.step_id!=d.step_id || a.op!=d.op || a.node!=d.node || a.remote!=d.remote || a.placement_ambiguous!=d.placement_ambiguous)
            throw std::invalid_argument("distributed token step placement mutated");
    }
    if(plan.boundary_payload_bytes_per_token!=derived.boundary_payload_bytes_per_token || plan.remote_nodes!=derived.remote_nodes)
        throw std::invalid_argument("distributed token placement summary mutated");
    for(std::size_t i=0;i<plan.boundaries.size();++i) {
        const auto& a=plan.boundaries[i];const auto& d=derived.boundaries[i];
        if(a.boundary_id!=d.boundary_id || a.source_step!=d.source_step || a.destination_step!=d.destination_step ||
            a.source_node!=d.source_node || a.destination_node!=d.destination_node || a.payload_bytes!=d.payload_bytes || a.payloads!=d.payloads)
            throw std::invalid_argument("distributed token boundary placement mutated");
    }
    for(std::uint32_t node=0;node<2U;++node) {
        const auto& p=b.peers[node];
        if(p.node!=node || !p.runtime_compatibility_identity || !p.state_layout_identity ||
            p.first_step>p.last_step || p.last_step>=serial.execution.steps.size() || b.position>=p.context_capacity ||
            !p.initial_state.contains(all_state) || (static_cast<std::uint32_t>(p.initial_state.valid)&~63U)!=0U)
            throw std::invalid_argument("distributed token peer stage or snapshot frontier is invalid");
        expected_local_[node]=p.initial_state;
        for(std::uint32_t step=p.first_step;step<=p.last_step;++step) {
            if(plan.steps[step].node!=node || plan.steps[step].step_id!=step ||
                plan.steps[step].op!=serial.execution.steps[step].op || plan.steps[step].placement_ambiguous)
                throw std::invalid_argument("distributed token stages do not match exact source placement");
            expected_local_[node].publish_write(serial.execution.steps[step].writes_state);
        }
    }
    if(b.peers[0].first_step!=0U || b.peers[0].last_step+1U!=b.peers[1].first_step ||
        b.peers[1].last_step+1ULL!=serial.execution.steps.size())
        throw std::invalid_argument("distributed token stages must cover the complete DAG once in order");
    for(const auto& step:serial.execution.steps) {
        if((static_cast<std::uint32_t>(step.reads_state)|static_cast<std::uint32_t>(step.writes_state))&~63U)
            throw std::invalid_argument("distributed token source has unknown persistent-state flags");
        expected_global_.publish_write(step.writes_state);
    }
    identity_=distributed_token_binding_identity(b);
    if(!identity_) throw std::invalid_argument("distributed token binding identity is empty");
}
std::uint64_t DistributedTokenTransaction::next_sequence(std::uint32_t node) const {
    if(node>1U) throw std::invalid_argument("distributed token peer is outside0/1");
    return sequence_[node];
}
PersistentStateGenerations DistributedTokenTransaction::expected_local_state(std::uint32_t node) const {
    if(node>1U) throw std::invalid_argument("distributed token peer is outside0/1");
    return expected_local_[node];
}
std::uint64_t DistributedTokenTransaction::expected_commit_decision() const {
    if(!both_ready()) throw std::logic_error("distributed commit requires both completed stage receipts");
    auto h=offset;hash(h,identity_);
    for(const auto& r:receipt_) {hash(h,r.snapshot_plan_identity);hash(h,r.snapshot_generation);state_hash(h,r.local_state);hash(h,r.completed_steps);}
    return h;
}
void DistributedTokenTransaction::observe(const DistributedTokenMessage& m) {
    try {
        if(phase_==DistributedTokenPhase::in_doubt || m.sender_node>1U || m.binding_identity!=identity_ ||
            distributed_token_binding_identity(m.binding)!=identity_ || m.sequence!=sequence_[m.sender_node] ||
            m.sequence==std::numeric_limits<std::uint64_t>::max())
            throw std::invalid_argument("distributed token message is stale, foreign or out of sequence");
        const auto node=m.sender_node;const auto& r=m.receipt;const auto& peer=binding_.peers[node];
        switch(m.kind) {
        case DistributedTokenMessageKind::prepared:
            if(phase_!=DistributedTokenPhase::preparing || prepared_[node] || m.decision_identity ||
                !r.snapshot_plan_identity || !r.snapshot_generation || r.completed_steps || !same_state(r.local_state,peer.initial_state))
                throw std::invalid_argument("distributed token preparation receipt is invalid");
            receipt_[node]=r;prepared_[node]=true;
            if(both_prepared()) phase_=DistributedTokenPhase::executing;
            break;
        case DistributedTokenMessageKind::ready:
            if(phase_!=DistributedTokenPhase::executing || !both_prepared() || ready_[node] || m.decision_identity ||
                r.snapshot_plan_identity!=receipt_[node].snapshot_plan_identity || r.snapshot_generation!=receipt_[node].snapshot_generation ||
                r.completed_steps!=peer.last_step-peer.first_step+1U || !same_state(r.local_state,expected_local_[node]))
                throw std::invalid_argument("distributed token completed stage receipt is invalid");
            receipt_[node]=r;ready_[node]=true;break;
        case DistributedTokenMessageKind::commit_decision:
            if(node!=0U || phase_!=DistributedTokenPhase::executing || !both_ready() || !zero_receipt(r) ||
                !m.decision_identity || m.decision_identity!=expected_commit_decision())
                throw std::invalid_argument("distributed token commit decision has no complete two-peer basis");
            decision_=m.decision_identity;phase_=DistributedTokenPhase::commit_decided;break;
        case DistributedTokenMessageKind::committed:
            if(phase_!=DistributedTokenPhase::commit_decided || acknowledged_[node] ||
                m.decision_identity!=decision_ || !zero_receipt(r))
                throw std::invalid_argument("distributed token commit acknowledgement is invalid");
            acknowledged_[node]=true;
            if(acknowledged_[0] && acknowledged_[1]) phase_=DistributedTokenPhase::committed;
            break;
        case DistributedTokenMessageKind::abort_decision:
            if(node!=0U || (phase_!=DistributedTokenPhase::executing) || !both_prepared() ||
                m.decision_identity || !zero_receipt(r))
                throw std::invalid_argument("distributed token abort cannot undo a commit decision or unknown preparation");
            phase_=DistributedTokenPhase::abort_decided;break;
        case DistributedTokenMessageKind::restored:
            if(phase_!=DistributedTokenPhase::abort_decided || restored_[node] || m.decision_identity ||
                r.snapshot_plan_identity!=receipt_[node].snapshot_plan_identity || r.snapshot_generation!=receipt_[node].snapshot_generation ||
                r.completed_steps || !same_state(r.local_state,peer.initial_state))
                throw std::invalid_argument("distributed token restore receipt is invalid");
            restored_[node]=true;if(restored_[0] && restored_[1]) phase_=DistributedTokenPhase::rolled_back;break;
        default:throw std::invalid_argument("distributed token message kind is outside the closed contract");
        }
        ++sequence_[node];
    } catch(...) {phase_=DistributedTokenPhase::in_doubt;throw;}
}
std::optional<DistributedTokenCommit> DistributedTokenTransaction::committed_frontier() const {
    if(phase_!=DistributedTokenPhase::committed) return std::nullopt;
    auto history=offset;hash(history,binding_.previous_history_identity);hash(history,identity_);hash(history,decision_);
    return DistributedTokenCommit{identity_,decision_,binding_.position+1U,binding_.committed_tokens+1U,history,
        expected_global_,expected_local_};
}
} // namespace strix
