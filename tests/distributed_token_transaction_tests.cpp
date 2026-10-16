#include "strix/distributed_token_transaction.hpp"
#include "strix/distributed_control_wire.hpp"
#include "strix/session_snapshot_transfer.hpp"
#include <algorithm>
#include <cstring>
#include <array>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using namespace strix;
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {
    bool rejected=false;try {f();} catch(const std::exception&) {rejected=true;}
    require(rejected,"invalid transaction was accepted");
}
struct Fixture {
    CompiledSerialExecution source;
    ModelManifest manifest;
    ExecutionTensorAccessPlan access;
    DistributedPlacementPlan plan;
    DistributedTokenBinding binding;
    Fixture() {
        source.execution_identity=0x100U;source.kernel_plan_identity=0x101U;
        source.vocab_size=32U;source.max_position_embeddings=64U;
        const auto none=PersistentState::none;
        source.execution.steps={
            {0,SerialOp::token_embedding,{}, {},none,PersistentState::ngram_history,true,false},
            {1,SerialOp::mlp_hyper_mix,{}, {0},none,PersistentState::recurrent_matrix,true,false},
            {2,SerialOp::moe_experts,{}, {1},none,PersistentState::convolution_history,true,false},
            {3,SerialOp::moe_combine,{}, {2},none,PersistentState::attention_kv,true,false}};
        const ExecutionBufferUse h0{{ExecutionBufferClass::hidden,0},1};
        const ExecutionBufferUse h1{{ExecutionBufferClass::hidden,1},1};
        const ExecutionBufferUse h2{{ExecutionBufferClass::hidden,2},1};
        source.io.hidden_slots=3;
        source.io.steps={{0,SerialOp::token_embedding,{}, {h0},{}},
            {1,SerialOp::mlp_hyper_mix,{h0},{h1},{}},
            {2,SerialOp::moe_experts,{h1},{h2},{}},
            {3,SerialOp::moe_combine,{h2},{},{}}};
        source.arena.alignment=256U;source.arena.total_bytes=1536U;
        source.arena.allocations={{h0,0,1,0,512},{h1,1,2,512,512},{h2,2,3,1024,512}};
        manifest.architecture="synthetic-transaction-contract";
        manifest.layout_identity=model_manifest_layout_identity(manifest);
        access.manifest_layout_identity=manifest.layout_identity;access.plan_identity=0x200U;
        const std::array<std::uint32_t,4> nodes{0,0,1,1};
        plan=build_distributed_placement_plan(source,manifest,access,nodes,0U);
        binding.serial_execution_identity=source.execution_identity;binding.kernel_plan_identity=source.kernel_plan_identity;
        binding.placement_plan_identity=plan.plan_identity;binding.model_content_identity.bytes[0]=1U;
        binding.request_id=9U;binding.connection_epoch=3U;binding.generation=7U;
        binding.position=2U;binding.committed_tokens=2U;binding.previous_history_identity=0xabcU;binding.input_token=5U;binding.boundary_dtype_id=1U;
        binding.initial_global_state.valid=static_cast<PersistentState>(63U);
        binding.initial_global_state.generations.fill(10U);
        for(std::uint32_t n=0;n<2U;++n) {
            auto& p=binding.peers[n];p.node=n;p.first_step=n*2U;p.last_step=p.first_step+1U;
            p.runtime_compatibility_identity=100U+n;p.state_layout_identity=200U+n;p.context_capacity=32U;
            p.initial_state=binding.initial_global_state;
        }
    }
    DistributedTokenTransaction make() const {return {source,plan,manifest,access,binding};}
};
DistributedTokenMessage message(const DistributedTokenTransaction& t,std::uint32_t node,DistributedTokenMessageKind kind) {
    DistributedTokenMessage m;m.binding=t.binding();m.binding_identity=t.binding_identity();
    m.sender_node=node;m.sequence=t.next_sequence(node);m.kind=kind;return m;
}
DistributedTokenReceipt receipt(const DistributedTokenTransaction& t,std::uint32_t node,bool ready) {
    const auto& p=t.binding().peers[node];
    return {700U+node,800U+node,ready?t.expected_local_state(node):p.initial_state,ready?p.last_step-p.first_step+1U:0U};
}
void prepare(DistributedTokenTransaction& t) {
    for(std::uint32_t n=0;n<2U;++n) {
        auto m=message(t,n,DistributedTokenMessageKind::prepared);m.receipt=receipt(t,n,false);t.observe(m);
    }
}
void ready(DistributedTokenTransaction& t) {
    for(std::uint32_t n=0;n<2U;++n) {
        auto m=message(t,n,DistributedTokenMessageKind::ready);m.receipt=receipt(t,n,true);t.observe(m);
    }
}
void commit_path(const Fixture& f) {
    auto t=f.make();prepare(t);ready(t);
    auto m=message(t,0,DistributedTokenMessageKind::commit_decision);m.decision_identity=t.expected_commit_decision();t.observe(m);
    require(t.phase()==DistributedTokenPhase::commit_decided && !t.committed_frontier(),"decision alone was reported globally committed");
    const auto decision=t.decision_identity();
    for(std::uint32_t n=0;n<2U;++n) {
        m=message(t,n,DistributedTokenMessageKind::committed);m.decision_identity=decision;t.observe(m);
        if(n==0U) require(!t.committed_frontier(),"single ACK published a global frontier");
    }
    const auto done=t.committed_frontier();require(done.has_value(),"both ACKs did not complete commit");
    require(done->next_position==3U && done->committed_tokens==3U && done->history_identity!=f.binding.previous_history_identity,
        "committed frontier lost position/input history binding");
    auto global=f.binding.initial_global_state;
    for(const auto& step:f.source.execution.steps) global.publish_write(step.writes_state);
    require(done->global_state.generations==global.generations,"global state counted nonlocal or omitted writes");
    require(done->local_state[0].generations==t.expected_local_state(0).generations &&
        done->local_state[1].generations==t.expected_local_state(1).generations,"local stage generations were retagged global");
}
void rollback_path(const Fixture& f) {
    auto t=f.make();prepare(t);
    auto m=message(t,0,DistributedTokenMessageKind::abort_decision);t.observe(m);
    m=message(t,0,DistributedTokenMessageKind::restored);m.receipt=receipt(t,0,false);t.observe(m);
    require(t.phase()==DistributedTokenPhase::abort_decided && !t.committed_frontier(),"partial restore authorized reuse");
    m=message(t,1,DistributedTokenMessageKind::restored);m.receipt=receipt(t,1,false);t.observe(m);
    require(t.phase()==DistributedTokenPhase::rolled_back && !t.committed_frontier(),"bilateral rollback manufactured a commit");
    auto failed=f.make();prepare(failed);m=message(failed,0,DistributedTokenMessageKind::abort_decision);failed.observe(m);
    m=message(failed,0,DistributedTokenMessageKind::restored);m.receipt=receipt(failed,0,false);
    ++m.receipt.local_state.generations[0];rejects([&]{failed.observe(m);});
    require(failed.phase()==DistributedTokenPhase::in_doubt,"partial/corrupt restore did not fail closed");
}
void wrong_messages(const Fixture& f) {
    using Edit=std::function<void(DistributedTokenMessage&)>;
    const std::vector<Edit> mutations={
        [](auto& m){++m.binding.connection_epoch;},[](auto& m){++m.binding.input_token;},
        [](auto& m){++m.binding.boundary_dtype_id;},[](auto& m){++m.binding.position;},[](auto& m){++m.binding.placement_plan_identity;},
        [](auto& m){++m.binding.generation;},[](auto& m){++m.binding.peers[1].state_layout_identity;},
        [](auto& m){++m.sequence;},[](auto& m){m.sender_node=2;},[](auto& m){++m.binding_identity;},
        [](auto& m){m.kind=DistributedTokenMessageKind::ready;},
        [](auto& m){m.receipt.snapshot_generation=0;},
        [](auto& m){m.receipt.completed_steps=1;},
        [](auto& m){m.kind=static_cast<DistributedTokenMessageKind>(99U);}};
    for(const auto& edit:mutations) {
        auto t=f.make();auto m=message(t,0,DistributedTokenMessageKind::prepared);m.receipt=receipt(t,0,false);edit(m);
        rejects([&]{t.observe(m);});require(t.phase()==DistributedTokenPhase::in_doubt && !t.committed_frontier(),"bad message left reusable state");
    }
    auto duplicate=f.make();auto m=message(duplicate,0,DistributedTokenMessageKind::prepared);m.receipt=receipt(duplicate,0,false);
    duplicate.observe(m);rejects([&]{duplicate.observe(m);});
    auto early=f.make();prepare(early);m=message(early,0,DistributedTokenMessageKind::commit_decision);m.decision_identity=1U;
    rejects([&]{early.observe(m);});
    auto decided=f.make();prepare(decided);ready(decided);
    m=message(decided,0,DistributedTokenMessageKind::commit_decision);m.decision_identity=decided.expected_commit_decision();decided.observe(m);
    const auto irreversible=decided.decision_identity();
    m=message(decided,0,DistributedTokenMessageKind::abort_decision);rejects([&]{decided.observe(m);});
    require(decided.decision_identity()==irreversible && decided.phase()==DistributedTokenPhase::in_doubt,
        "postdecision abort forgot the irreversible commit decision");
    auto dropped=f.make();prepare(dropped);ready(dropped);
    m=message(dropped,0,DistributedTokenMessageKind::commit_decision);m.decision_identity=dropped.expected_commit_decision();dropped.observe(m);
    dropped.transport_failed();require(dropped.decision_identity()!=0U && !dropped.committed_frontier(),"dropped decision/ACK was silently rolled back or committed");
}
// Inject an actual failure in the existing seven-state restore consumer after
// one physical range copied. This is not a fake successful restored receipt.
struct CopyBackend final : SessionSnapshotTransferBackend {
    bool fail_restore=false;std::uint32_t restored_ranges=0;
    static void copy(std::uintptr_t a,std::uintptr_t b,std::uint64_t n) {
        std::memcpy(reinterpret_cast<void*>(b),reinterpret_cast<const void*>(a),static_cast<std::size_t>(n));
    }
    void capture_device(std::uintptr_t a,std::uintptr_t b,std::uint64_t n) override {copy(a,b,n);}
    void capture_host(std::uintptr_t a,std::uintptr_t b,std::uint64_t n) override {copy(a,b,n);}
    void restore_device(std::uintptr_t a,std::uintptr_t b,std::uint64_t n) override {
        if(fail_restore && restored_ranges==1U) throw std::runtime_error("injected second-range restore failure");
        copy(a,b,n);++restored_ranges;
    }
    void restore_host(std::uintptr_t a,std::uintptr_t b,std::uint64_t n) override {restore_device(a,b,n);}
};
void actual_partial_restore(const Fixture& f) {
    auto model=ModelContract::qwen38_flash_next();
    model.vocab_size=32U;model.ple_eos_token_id=31U;model.hidden_size=8U;model.max_position_embeddings=8U;
    model.hidden_layers=2U;model.full_attention_layers=1U;model.gdn_layers=1U;model.full_attention_interval=2U;
    model.attention_heads=2U;model.kv_heads=1U;model.head_dim=8U;model.ple_embed_dim=8U;model.ple_conv_kernel_size=2U;
    model.ngram_size=2U;model.heads_per_ngram=1U;model.hc_lowrank=2U;
    model.gdn_key_heads=1U;model.gdn_value_heads=1U;model.gdn_key_head_dim=2U;model.gdn_value_head_dim=2U;model.gdn_conv_kernel=2U;
    model.indexer_n_heads=1U;model.indexer_kv_heads=1U;model.indexer_head_dim=8U;model.indexer_budget=8U;
    model.moe_intermediate_size=4U;model.shared_expert_intermediate_size=4U;model.num_experts=2U;model.num_experts_per_token=1U;
    SerialSessionStateLayoutRequest layout_request;layout_request.context_tokens=4U;
    layout_request.device_alignment=1U;layout_request.host_alignment=1U;
    const auto layout=build_serial_session_state_layout(model,layout_request);
    std::vector<std::byte> device(static_cast<std::size_t>(layout.device_total_bytes),std::byte{0x11});
    std::vector<std::byte> host(static_cast<std::size_t>(layout.host_total_bytes),std::byte{0x11});
    SessionStateMemoryBinding memory;memory.layout_identity=layout.layout_identity;
    memory.device_base=reinterpret_cast<std::uintptr_t>(device.data());memory.device_bytes=device.size();memory.device_allocation_generation=1U;
    memory.host_base=reinterpret_cast<std::uintptr_t>(host.data());memory.host_bytes=host.size();memory.host_allocation_generation=2U;
    SerialSessionSnapshotRequest request;request.compiled_execution_identity=f.source.execution_identity;
    request.runtime_compatibility_identity=0x123U;request.next_position=1U;request.committed_tokens=1U;
    request.state=f.binding.initial_global_state;request.alignment=1U;
    const auto snapshot=build_serial_session_snapshot_plan(model,layout,request);
    const auto restore=build_serial_session_restore_plan(model,layout,request.runtime_compatibility_identity,snapshot);
    require(snapshot.ranges.size()==7U,"partial-restore fixture lost a state component");
    std::vector<std::byte> saved_device(static_cast<std::size_t>(snapshot.device_bytes));
    std::vector<std::byte> saved_host(static_cast<std::size_t>(snapshot.host_bytes));
    SessionSnapshotMemoryBinding saved;saved.snapshot_plan_identity=snapshot.plan_identity;
    saved.device_section_base=reinterpret_cast<std::uintptr_t>(saved_device.data());saved.device_section_bytes=saved_device.size();
    saved.host_section_base=reinterpret_cast<std::uintptr_t>(saved_host.data());saved.host_section_bytes=saved_host.size();saved.allocation_generation=3U;
    CopyBackend backend;capture_serial_session_snapshot(snapshot,layout,memory,saved,backend);
    std::fill(device.begin(),device.end(),std::byte{0x44});std::fill(host.begin(),host.end(),std::byte{0x44});
    SerialSessionContext session;session.execution_identity=request.compiled_execution_identity;
    session.runtime_compatibility_identity=request.runtime_compatibility_identity;session.state_layout_identity=layout.layout_identity;
    session.next_position=1U;session.committed_tokens=1U;session.state=request.state;
    session.arithmetic_lineage=prepare_serial_row_lineage({},0U,session.execution_identity,0x567U);
    auto protocol=f.make();prepare(protocol);protocol.observe(message(protocol,0U,DistributedTokenMessageKind::abort_decision));
    backend.fail_restore=true;
    rejects([&]{restore_serial_session_snapshot(session,restore,layout,memory,saved,backend);});
    protocol.transport_failed();
    require(backend.restored_ranges==1U && device.front()==std::byte{0x11} && host.back()==std::byte{0x44} &&
        session.health==SerialSessionHealth::poisoned && session.arithmetic_lineage.tainted &&
        protocol.phase()==DistributedTokenPhase::in_doubt && !protocol.committed_frontier(),
        "partial physical restore was hidden by metadata publication or authorized reuse");
}
struct Loopback final : DistributedControlByteStream {
    std::vector<std::byte> frame;
    void send(std::span<const std::byte> bytes) override {frame.assign(bytes.begin(),bytes.end());}
    std::vector<std::byte> receive() override {return std::exchange(frame,{});}
};
void wire_path(const Fixture& f) {
    auto t=f.make();auto m=message(t,1,DistributedTokenMessageKind::prepared);m.receipt=receipt(t,1,false);
    Loopback stream;FramedDistributedTokenWire wire(stream);wire.send(m);
    require(inspect_distributed_control_frame_kind(stream.frame)==DistributedControlFrameKind::token_transaction,"transaction used a legacy frame kind");
    const auto original=stream.frame;const auto decoded=wire.receive();
    require(encode_distributed_token_message(decoded)==original,"transaction wire did not preserve all bound fields");
    auto truncated=original;truncated.pop_back();rejects([&]{(void)decode_distributed_token_message(truncated);});
    auto reserved=original;reserved[10]=std::byte{1};rejects([&]{(void)decode_distributed_token_message(reserved);});
    auto kind=original;kind[8]=std::byte{99};rejects([&]{(void)decode_distributed_token_message(kind);});
    auto trailing=original;trailing.push_back(std::byte{0});rejects([&]{(void)decode_distributed_token_message(trailing);});
    auto wrong=m;wrong.binding.initial_global_state.valid=static_cast<PersistentState>(64U);
    wrong.binding_identity=distributed_token_binding_identity(wrong.binding);rejects([&]{(void)encode_distributed_token_message(wrong);});
    wrong=m;wrong.kind=static_cast<DistributedTokenMessageKind>(0U);rejects([&]{(void)encode_distributed_token_message(wrong);});
    wrong=m;wrong.sequence=0U;rejects([&]{(void)encode_distributed_token_message(wrong);});
}
} // namespace
int main() {
    try {
        Fixture f;commit_path(f);rollback_path(f);wrong_messages(f);wire_path(f);actual_partial_restore(f);
        auto bad=f.binding;bad.input_token=32U;
        rejects([&]{DistributedTokenTransaction x(f.source,f.plan,f.manifest,f.access,bad);});
        bad=f.binding;bad.position=32U;rejects([&]{DistributedTokenTransaction x(f.source,f.plan,f.manifest,f.access,bad);});
        bad=f.binding;bad.peers[0].last_step=2U;rejects([&]{DistributedTokenTransaction x(f.source,f.plan,f.manifest,f.access,bad);});
        bad=f.binding;bad.initial_global_state.generations[0]=std::numeric_limits<std::uint64_t>::max();
        rejects([&]{DistributedTokenTransaction x(f.source,f.plan,f.manifest,f.access,bad);});
        std::cout<<"distributed token transaction source fixtures completed\n";return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
