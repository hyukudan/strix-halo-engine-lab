#include "../apps/deferred_moe_receipt.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/serial_host_state_shadow.hpp"
#include "strix/serial_session.hpp"

#include <array>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <utility>

namespace {
void expect(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
strix::kernels::KernelRegistry registry(const strix::ModelContract& model,const strix::ExecutionBindingRequest& request){
    strix::kernels::KernelRegistry out;std::set<std::string> seen;
    for(const auto& step:strix::build_serial_decode_execution_plan(model).steps){
        const auto family=strix::operator_family_for(step.op);if(!family)continue;
        const auto shape=strix::execution_shape_for(model,step.op,request);
        const auto id=std::to_string(static_cast<unsigned>(*family))+"."+std::to_string(shape.auxiliary_size);
        if(!seen.insert(id).second)continue;
        strix::kernels::KernelVariant v;v.id=id;v.family=*family;v.shape=shape;
        v.phase=strix::kernels::ExecutionPhase::decode;v.wave=strix::kernels::WaveSize::wave32;
        v.resources.workgroup_threads=256;v.numerics.input_precision="bf16";v.numerics.accumulator_precision="fp32";
        v.numerics.output_precision="bf16";v.numerics.deterministic=true;v.numerics.row_invariant=true;
        v.qualification=strix::kernels::QualificationStatus::oracle_matched;v.graph_safe=true;
        v.input_layout="test";v.output_layout="test";v.compiler_identity="test";out.add(std::move(v));
    }
    return out;
}
struct Dispatcher final:strix::EagerExecutionDispatcher{
    strix::DeferredMoeFirstError error;
    bool throw_during_dispatch=false,throw_after_device_error=false;
    unsigned calls=0,moe_layers=0;
    void run(const strix::ExecutionDispatchContext& c){
        ++calls;
        if(c.op==strix::SerialOp::position_prepare||c.op==strix::SerialOp::ple){
            const auto component=c.op==strix::SerialOp::ple?strix::SessionStateComponent::ngram_history:strix::SessionStateComponent::position_history;
            const auto address=strix::resolve_session_state(*c.session_state_layout,*c.session_state_memory,component);
            std::memset(reinterpret_cast<void*>(address.address),0x5a,static_cast<std::size_t>(address.bytes));
        }
        if(c.op==strix::SerialOp::moe_experts){
            ++moe_layers;
            if(moe_layers==17){
                if(throw_during_dispatch)throw std::runtime_error("injected middle-layer host failure");
                strix::merge_deferred_moe_first_error(error,5,strix::DeferredMoeDomain::q4k_a8);
            }
            if(moe_layers==30 && throw_after_device_error)throw std::runtime_error("injected later host failure");
            // A later failure must not replace the first domain or any status bits.
            if(moe_layers==29)strix::merge_deferred_moe_first_error(error,1,strix::DeferredMoeDomain::legacy);
            strix::merge_deferred_moe_first_error(error,0,strix::DeferredMoeDomain::none);
        }
    }
    void dispatch_kernel(const strix::ExecutionDispatchContext& c)override{run(c);}
    void dispatch_runtime_step(const strix::ExecutionDispatchContext& c)override{run(c);}
};
}
int main()try{
    strix::DeferredMoeFirstError first;
    strix::merge_deferred_moe_first_error(first,0,strix::DeferredMoeDomain::legacy);
    expect(first.domain==0 && first.bits==0,"successful layer leaves empty record unchanged");
    strix::merge_deferred_moe_first_error(first,5,strix::DeferredMoeDomain::q4k_a8);
    strix::merge_deferred_moe_first_error(first,1,strix::DeferredMoeDomain::legacy);
    strix::merge_deferred_moe_first_error(first,0,strix::DeferredMoeDomain::none);
    expect(first.domain==2 && first.bits==5,"product merge retains complete first A8 error over later legacy/zero status");

    const auto model=strix::ModelContract::qwen38_flash_next();strix::ExecutionBindingRequest binding;
    strix::ExecutionBufferRequest buffers;
    const auto compiled=strix::compile_serial_execution(model,registry(model,binding),binding,buffers,256);
    strix::SerialSessionStateLayoutRequest req;req.context_tokens=4;
    const auto layout=strix::build_serial_session_state_layout(model,req);
    alignas(64)std::array<std::byte,128> published{};
    expect(layout.host_total_bytes==published.size(),"fixture owns complete host history");
    strix::SessionStateMemoryBinding state{layout.layout_identity,0x31000000,layout.device_total_bytes,2,
        reinterpret_cast<std::uintptr_t>(published.data()),published.size(),3};
    strix::ExecutionMemoryBinding scratch{compiled.execution_identity,0x30000000,compiled.arena.total_bytes,1};
    strix::ModelManifest manifest;manifest.layout_identity=77;
    strix::ModelRuntimeBinding weights;weights.manifest_layout_identity=77;weights.model_content_identity.bytes[0]=1;
    strix::ExecutionTensorAccessPlan access;access.manifest_layout_identity=77;access.plan_identity=88;
    strix::ExecutionRuntimeResources resources;resources.scratch=&scratch;resources.model_manifest=&manifest;
    resources.model_weights=&weights;resources.tensor_access=&access;resources.session_state_layout=&layout;resources.session_state_memory=&state;
    strix::SerialHostStateShadow shadow(layout,state,9);auto transactional=resources;transactional.session_state_memory=&shadow.binding();
    const auto original=strix::build_runtime_execution_identities(compiled.execution_identity,compiled.arena,resources);
    const auto staged=strix::build_runtime_execution_identities(compiled.execution_identity,compiled.arena,transactional);
    expect(original.compatibility_identity==staged.compatibility_identity&&original.binding_identity!=staged.binding_identity,
           "host shadow retains actual runtime compatibility but has a distinct binding identity");
    strix::SerialSessionContext fresh;fresh.execution_identity=compiled.execution_identity;fresh.state_layout_identity=layout.layout_identity;
    fresh.runtime_compatibility_identity=original.compatibility_identity;
    for(const auto& step:compiled.execution.steps)fresh.state.valid=fresh.state.valid|step.reads_state|step.writes_state;
    for(unsigned failure:{0U,1U,2U,3U}){
        const bool throw_dispatch=failure==1U;
        auto session=fresh;Dispatcher d;d.throw_during_dispatch=throw_dispatch;d.throw_after_device_error=failure==3U;bool completion=false;shadow.begin();
        std::string error;
        try{(void)strix::execute_serial_token(compiled,session,transactional,7734,d,[&]{
            completion=true;
            expect(session.committed_tokens==0&&session.next_position==0,"completion precedes session publication");
            expect(published==std::array<std::byte,128>{},"history stays unpublished at completion boundary");
            if(failure==2U)throw std::runtime_error("injected token fence failure");
            strix::check_deferred_moe_status(d.error);shadow.publish();
        });}catch(const std::exception& e){error=e.what();shadow.cancel();}
        expect(session.health==strix::SerialSessionHealth::poisoned&&session.arithmetic_lineage.tainted,"post-dispatch error poisons session");
        expect(session.next_position==fresh.next_position&&session.committed_tokens==fresh.committed_tokens&&
               session.execution_identity==fresh.execution_identity&&session.state.generations==fresh.state.generations&&
               session.arithmetic_lineage.intervals==fresh.arithmetic_lineage.intervals&&
               session.arithmetic_lineage.history_identity==fresh.arithmetic_lineage.history_identity,"failure publishes no metadata, row ID or state generations");
        expect(published==std::array<std::byte,128>{},"failed middle layer publishes no ngram or position bytes");
        if(throw_dispatch)expect(!completion&&error=="injected middle-layer host failure","dispatch failure bypasses completion");
        else if(failure==3U)expect(!completion && error=="injected later host failure" && d.error.domain==2 && d.error.bits==5,
            "later host exception takes precedence over earlier device status and bypasses completion");
        else if(failure==2U)expect(completion&&error=="injected token fence failure","completion failure preserves its original error");
        else expect(completion&&d.moe_layers==48&&error=="Q4K A8 rejected status bits (route/weight=1,input=2,delta=4,pipeline=8): 5",
                    "deferred error retains first complete A8 bits and original message after all layers");
        bool rejected=false;try{(void)strix::execute_serial_token(compiled,session,transactional,7734,d);}catch(const std::logic_error&){rejected=true;}
        expect(rejected,"poisoned session rejects another token");
    }
    auto success=fresh;Dispatcher ok;shadow.begin();bool completion=false;
    struct CompletionView {
        const strix::SerialSessionContext* session;
        std::uint64_t completed_tokens() const { return session->committed_tokens; }
    } completed_view{&success};
    const auto prefix_snapshot = strix::decode_detail::deferred_moe_completed_tokens_receipt(&completed_view);
    expect(prefix_snapshot == "  \"deferred_moe_status_completed_tokens\": 0,\n",
           "before execution completion receipt is zero");
    (void)strix::execute_serial_token(compiled,success,transactional,7734,ok,[&]{completion=true;ok.error={};strix::check_deferred_moe_status(ok.error);shadow.publish();});
    expect(completion&&success.next_position==1&&success.committed_tokens==1&&success.health==strix::SerialSessionHealth::ready,"successful fence publishes exactly one token");
    expect(published!=std::array<std::byte,128>{},"successful fence publishes staged host history");
    expect(strix::decode_detail::deferred_moe_completed_tokens_receipt(&completed_view) ==
           "  \"deferred_moe_status_completed_tokens\": 1,\n" &&
           prefix_snapshot == "  \"deferred_moe_status_completed_tokens\": 0,\n",
           "post-execution reader follows actual committed session, not stale prefix snapshot");
    expect(strix::decode_detail::deferred_moe_completed_tokens_receipt<CompletionView>(nullptr) ==
           "  \"deferred_moe_status_completed_tokens\": 0,\n",
           "disabled deferred owner retains explicit zero completed tokens");
    auto preflight=fresh;Dispatcher untouched;bool ran=false;shadow.begin();
    try{(void)strix::execute_serial_token(compiled,preflight,transactional,static_cast<std::uint32_t>(model.vocab_size),untouched,[&]{ran=true;});}
    catch(const std::invalid_argument&){}shadow.cancel();
    expect(!ran&&untouched.calls==0&&preflight.health==strix::SerialSessionHealth::ready&&preflight.committed_tokens==0,"preflight failure stays ready without dispatch or publication");
    strix::check_deferred_moe_status({});
    for(const auto record:{strix::DeferredMoeFirstError{1,1},strix::DeferredMoeFirstError{2,15}}){
        bool rejected=false;try{strix::check_deferred_moe_status(record);}catch(const std::invalid_argument&){rejected=true;}
        expect(rejected,"legacy and A8 status failures preserve original exception class");
    }
    bool malformed=false;try{strix::check_deferred_moe_status({0,1});}catch(const std::runtime_error&){malformed=true;}
    expect(malformed,"malformed completion record fails closed");
    std::cout<<"deferred completion, first-error bits, shadow history, generations and poisoning tests passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
