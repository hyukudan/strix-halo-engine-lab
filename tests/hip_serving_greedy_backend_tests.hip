// Explicit full-model native qualification only. No automatic CTest/model run.
// Source implementation is not a correctness, quality or throughput result.
#include "strix/hip_serving_greedy_backend.hpp"
#include "strix/hip_serving_prefix_cache.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/greedy_serving_scheduler.hpp"
#include "strix/hip_generation_output_diagnostic.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/hip_runtime_allocator.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/mapped_model_file.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/model_tensor_index.hpp"
#include "strix/gguf_reader.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
constexpr std::uint64_t context_capacity = 64;
constexpr std::uint64_t generated_limit = 3;
constexpr std::size_t transfer_chunk = 8U * 1024U * 1024U;
void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}
void check(hipError_t error) {
    if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
}
std::string digest(const std::filesystem::path& path) {
    return strix::model_content_identity_hex(strix::sha256_file_content(path));
}
struct Args {
    std::filesystem::path model, prompt, output;
    std::string model_sha, prompt_sha;
};
Args parse(int argc, char** argv) {
    Args value;
    std::vector<std::string> seen;
    for (int i=1;i<argc;i+=2) {
        require(i+1<argc,"option missing value");
        const std::string key=argv[i];
        require(std::find(seen.begin(),seen.end(),key)==seen.end(),"duplicate option");
        seen.push_back(key);
        if(key=="--model") value.model=argv[i+1];
        else if(key=="--model-sha256") value.model_sha=argv[i+1];
        else if(key=="--prompt-u32le") value.prompt=argv[i+1];
        else if(key=="--prompt-sha256") value.prompt_sha=argv[i+1];
        else if(key=="--output") value.output=argv[i+1];
        else throw std::invalid_argument("unknown serving qualification option");
    }
    const auto sha=[](const std::string& text) {
        return text.size()==64 && std::all_of(text.begin(),text.end(),[](char c) {
            return (c>='0'&&c<='9')||(c>='a'&&c<='f');
        });
    };
    require(!value.model.empty()&&!value.prompt.empty()&&!value.output.empty()&&sha(value.model_sha)&&sha(value.prompt_sha),
        "required --model --model-sha256 --prompt-u32le --prompt-sha256 --output");
    require(!std::filesystem::exists(value.output),"qualification output must be fresh");
    return value;
}
std::vector<std::uint32_t> read_prompt(const Args& args, std::uint64_t vocabulary) {
    require(digest(args.prompt)==args.prompt_sha,"prompt SHA differs before loading");
    const auto bytes=std::filesystem::file_size(args.prompt);
    require(bytes>=16 && bytes<=128 && bytes%4U==0,"prompt requires 4..32 explicit u32le tokens");
    std::ifstream input(args.prompt,std::ios::binary);
    input.exceptions(std::ios::failbit|std::ios::badbit);
    std::vector<std::uint32_t> values;
    for(std::uint64_t offset=0;offset<bytes;offset+=4) {
        std::array<unsigned char,4> word{};
        input.read(reinterpret_cast<char*>(word.data()),4);
        const auto token=std::uint32_t(word[0])|(std::uint32_t(word[1])<<8U)|
            (std::uint32_t(word[2])<<16U)|(std::uint32_t(word[3])<<24U);
        require(token<vocabulary,"prompt token outside vocabulary");values.push_back(token);
    }
    require(digest(args.prompt)==args.prompt_sha,"prompt changed during loading");
    return values;
}
struct ModelOwner {
    std::shared_ptr<strix::HipSetupQuiescenceGuard> guard=std::make_shared<strix::HipSetupQuiescenceGuard>();
    strix::AllocationGenerationSource generations;
    strix::ModelContract model=strix::ModelContract::qwen38_flash_next();
    strix::ModelManifest manifest;
    strix::CompiledSerialExecution compiled;
    strix::ExecutionTensorAccessPlan access;
    strix::SerialSessionStateLayout layout;
    strix::HipLauncherCatalogConfig config;
    std::shared_ptr<strix::MappedModelFile> mapped;
    std::shared_ptr<strix::HipModelResidency> weights;
    ~ModelOwner() { guard->recover_or_fail_stop(); }
};
std::shared_ptr<ModelOwner> load_model(const Args& args) {
    require(digest(args.model)==args.model_sha,"model SHA differs before loading");
    auto owner=std::make_shared<ModelOwner>();
    owner->manifest=strix::build_model_manifest(strix::gguf::read_file(args.model));
    auto& cfg=owner->config;
    cfg.deferred_moe_status=true;
    cfg.parallel_hyper_mix=true;cfg.staged_hyper_projections=true;cfg.fused_staged_hyper_mix=true;
    cfg.typed_hyper_projection=true;cfg.prefetch_typed_hyper_rows=true;cfg.prefetch_q8_hyper_output=true;
    cfg.virtual_q8_gdn_input=true;cfg.prefetch_q8_gdn_input=true;cfg.virtual_q8_gdn_output=true;
    cfg.wave8_q8_gdn_input=true;cfg.wave8_q8_gdn_output=true;cfg.virtual_q8_lm_head=true;
    strix::ExecutionBindingRequest binding;
    binding.minimum_qualification=strix::kernels::QualificationStatus::experimental;
    binding.wave8_q8_gdn_input=true;binding.wave8_q8_gdn_output=true;
    strix::ExecutionBufferRequest buffers;
    buffers.activation_element_bytes=2;buffers.mlp_intermediate_element_bytes=4;
    strix::kernels::KernelRegistry registry;
    strix::register_complete_hip_kernel_catalog(owner->model,binding,cfg,registry);
    owner->compiled=strix::compile_serial_execution(owner->model,registry,binding,buffers,256);
    const auto index=strix::build_model_tensor_index(owner->manifest);
    owner->access=strix::compile_execution_tensor_access_plan(owner->compiled.execution,owner->manifest,index,
        strix::build_serial_execution_tensor_requirements(owner->compiled.execution));
    strix::SerialSessionStateLayoutRequest request;request.context_tokens=context_capacity;
    owner->layout=strix::build_serial_session_state_layout(owner->model,request);
    require(owner->layout.allocations.size()==7,"qualification requires seven full state classes");
    owner->mapped=std::make_shared<strix::MappedModelFile>(strix::MappedModelFile::open_read_only(
        args.model,"model",owner->generations));
    owner->weights=std::make_shared<strix::HipModelResidency>(strix::HipModelResidency::create(
        std::static_pointer_cast<const strix::MappedModelFile>(owner->mapped),owner->manifest,owner->generations));
    return owner;
}
struct SessionOwner {
    std::shared_ptr<ModelOwner> model;
    strix::AllocationGenerationSource generations;
    std::shared_ptr<strix::HipRuntimeAllocator> allocator=std::make_shared<strix::HipRuntimeAllocator>();
    std::shared_ptr<strix::RuntimeExecutionMemoryOwner> memory;
    strix::FileMappedPleLookupBackend ple;
    strix::PleHashContract ple_hash;
    strix::PleLookupPrefetchState prefetch;
    strix::ExecutionRuntimeResources resources;
    strix::SerialSessionContext session;
    strix::HipLaunchContext scope;
    ~SessionOwner() {
        model->guard->recover_or_fail_stop();
        if(scope.stream && hipStreamDestroy(scope.stream)!=hipSuccess) std::_Exit(3);
        if(scope.deferred_moe_status.arena_base)
            allocator->free_device(scope.deferred_moe_status.arena_base,sizeof(strix::DeferredMoeFirstError));
    }
};
void initialize_session(const std::shared_ptr<SessionOwner>& owner) {
    require(owner && owner->model && !owner->memory,"serving fixture session was initialized twice");
    const auto& model=owner->model;
    owner->memory=std::make_shared<strix::RuntimeExecutionMemoryOwner>(strix::RuntimeExecutionMemoryOwner::allocate(
        model->compiled,model->layout,*owner->allocator,owner->generations));
    owner->ple_hash=strix::build_ple_hash_contract(model->model);
    owner->resources={&owner->memory->scratch(),&model->manifest,&model->weights->binding(),&model->access,
        &model->layout,&owner->memory->state(),&owner->ple,&owner->ple_hash,&owner->prefetch};
    owner->scope.deferred_moe_status={model->compiled.execution_identity,
        owner->allocator->allocate_device(sizeof(strix::DeferredMoeFirstError),8),sizeof(strix::DeferredMoeFirstError),owner->generations.next()};
    model->guard->run([&] {
        check(hipStreamCreateWithFlags(&owner->scope.stream,hipStreamNonBlocking));
        strix::initialize_hip_fresh_runtime_memory(owner->memory->scratch(),owner->memory->state());
        check(hipMemsetAsync(reinterpret_cast<void*>(owner->scope.deferred_moe_status.arena_base),0,
            sizeof(strix::DeferredMoeFirstError),owner->scope.stream));
    });
    owner->session.execution_identity=model->compiled.execution_identity;
    owner->session.state_layout_identity=model->layout.layout_identity;
    owner->session.runtime_compatibility_identity=strix::build_runtime_execution_identities(
        model->compiled.execution_identity,model->compiled.arena,owner->resources).compatibility_identity;
    for(const auto& allocation:model->layout.allocations)
        owner->session.state.valid=owner->session.state.valid|allocation.state_flag;
}
std::shared_ptr<SessionOwner> fresh_session(const std::shared_ptr<ModelOwner>& model) {
    auto owner=std::make_shared<SessionOwner>();owner->model=model;initialize_session(owner);return owner;
}
std::unique_ptr<strix::HipServingGreedyBackend> backend(const std::shared_ptr<SessionOwner>& owner,bool prepared,
    std::shared_ptr<strix::HipServingPrefixCache> pool={}) {
    return std::make_unique<strix::HipServingGreedyBackend>(owner->model->compiled,owner->model->model,
        owner->model->config,owner->resources,owner->session,owner->scope,
        std::static_pointer_cast<const void>(owner),prepared,std::move(pool));
}
void independent(const SessionOwner& a,const SessionOwner& b) {
    using Range=std::pair<std::uintptr_t,std::uint64_t>;
    const auto ranges=[](const SessionOwner& owner) {
        return std::array<Range,4>{{{owner.memory->scratch().arena_base,owner.memory->scratch().arena_bytes},
            {owner.memory->state().device_base,owner.memory->state().device_bytes},
            {owner.memory->state().host_base,owner.memory->state().host_bytes},
            {owner.scope.deferred_moe_status.arena_base,owner.scope.deferred_moe_status.arena_bytes}}};
    };
    require(&a!=&b && a.scope.stream!=b.scope.stream && &a.ple!=&b.ple && &a.prefetch!=&b.prefetch,
        "serving requests share mutable stream/PLE owners");
    for(const auto& x:ranges(a)) for(const auto& y:ranges(b)) {
        require(x.first && y.first && x.second<=std::numeric_limits<std::uintptr_t>::max()-x.first &&
            y.second<=std::numeric_limits<std::uintptr_t>::max()-y.first &&
            !(x.first<y.first+y.second && y.first<x.first+x.second),"serving requests alias mutable allocation owners");
    }
}
void same_metadata(const strix::SerialSessionContext& a,const strix::SerialSessionContext& b) {
    require(a.execution_identity==b.execution_identity && a.state_layout_identity==b.state_layout_identity &&
        a.runtime_compatibility_identity==b.runtime_compatibility_identity && a.next_position==b.next_position &&
        a.committed_tokens==b.committed_tokens && a.health==b.health && a.state.valid==b.state.valid &&
        a.state.generations==b.state.generations && a.arithmetic_lineage.intervals==b.arithmetic_lineage.intervals &&
        a.arithmetic_lineage.history_identity==b.arithmetic_lineage.history_identity &&
        a.arithmetic_lineage.tainted==b.arithmetic_lineage.tainted,"serving session metadata differs from serial baseline");
}
using StateHashes=std::array<std::string,7>;
StateHashes capture_state(const std::shared_ptr<SessionOwner>& owner,const std::filesystem::path& directory) {
    require(!owner->prefetch.active,"state capture requires completed PLE");
    std::filesystem::create_directory(directory);
    StateHashes result;
    std::vector<std::byte> buffer(transfer_chunk);
    std::size_t ordinal=0;
    for(const auto& allocation:owner->model->layout.allocations) {
        require(ordinal<result.size(),"unexpected extra state component");
        const auto view=strix::resolve_session_state(owner->model->layout,owner->memory->state(),allocation.component);
        const auto path=directory/(std::to_string(static_cast<unsigned>(allocation.component))+".bin");
        std::ofstream output(path,std::ios::binary);output.exceptions(std::ios::failbit|std::ios::badbit);
        for(std::uint64_t offset=0;offset<view.bytes;) {
            const auto bytes=static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(),view.bytes-offset));
            owner->model->guard->run([&] {
                if(view.storage==strix::SessionStateStorage::device)
                    check(hipMemcpyAsync(buffer.data(),reinterpret_cast<const void*>(view.address+offset),bytes,
                        hipMemcpyDeviceToHost,owner->scope.stream));
                else std::memcpy(buffer.data(),reinterpret_cast<const void*>(view.address+offset),bytes);
            });
            output.write(reinterpret_cast<const char*>(buffer.data()),static_cast<std::streamsize>(bytes));offset+=bytes;
        }
        output.close();result[ordinal++]=digest(path);
    }
    require(ordinal==7,"missing full state component");return result;
}
std::uint64_t reservation(const ModelOwner& model) {
    // Requested payload reservation, not observed residency. Include response
    // growth, selector pool and conservative allocator/stream bookkeeping slack.
    return model.compiled.arena.total_bytes+model.layout.device_total_bytes+model.layout.host_total_bytes+
        strix::HipServingGreedyBackend::output_reservation_bytes(static_cast<std::uint32_t>(model.model.vocab_size))+1048576U;
}
strix::SerialGreedyRequest request(const ModelOwner& model,std::span<const std::uint32_t> prompt) {
    return {prompt,model.model.vocab_size,context_capacity,generated_limit,model.model.ple_eos_token_id};
}
strix::ServingCompatibility compatibility(const ModelOwner& model) {
    return {model.mapped->content_identity(),model.compiled.kernel_plan_identity,model.guard->device_index()};
}
strix::GreedyServingPolicy policy(const ModelOwner& model,std::uint32_t concurrency) {
    strix::GreedyServingPolicy result;
    result.concurrency=concurrency;result.prefill_chunk_rows=1;result.max_retained_requests=8;
    result.reservation_budget_bytes=reservation(model)*8U;
    return result;
}
void same_result(const strix::SerialGreedyRequestResult& a,const strix::SerialGreedyRequestResult& b) {
    require(a.generated_tokens==b.generated_tokens && a.steps_executed==b.steps_executed &&
        a.final_position==b.final_position && a.stop_reason==b.stop_reason,"serving greedy tokens/frontier differ");
}
void qualify_prefix_cache(const std::shared_ptr<ModelOwner>& model,const std::vector<std::uint32_t>& prompt,
    const std::shared_ptr<SessionOwner>& reference,const strix::SerialGreedyRequestResult& baseline,
    const StateHashes& baseline_state,const std::filesystem::path& directory) {
    strix::SerialSessionSnapshotRequest shape;shape.compiled_execution_identity=model->compiled.execution_identity;
    shape.runtime_compatibility_identity=reference->session.runtime_compatibility_identity;
    shape.next_position=reference->session.next_position;shape.committed_tokens=reference->session.committed_tokens;shape.state=reference->session.state;
    const auto snapshot=strix::build_serial_session_snapshot_plan(model->model,model->layout,shape);
    require(snapshot.device_bytes<=UINT64_MAX-snapshot.host_bytes,"cache snapshot budget overflow");
    const auto bytes=snapshot.device_bytes+snapshot.host_bytes;require(bytes && bytes<=UINT64_MAX/3U,"cache peak budget overflow");
    const strix::HipServingPrefixCachePolicy cache_policy{2U,bytes*2U,bytes*3U,context_capacity};
    const auto shared_bytes=strix::HipServingPrefixCache::requested_reservation_bytes(cache_policy);
    const auto scheduled_policy=[&](std::uint32_t concurrency) {
        auto p=policy(*model,concurrency);
        require(p.reservation_budget_bytes<=UINT64_MAX-shared_bytes,"cache shared reservation overflow");
        p.reservation_budget_bytes+=shared_bytes;p.shared_reservation_bytes=shared_bytes;return p;
    };
    // Charge the shared reservation before constructing a pool or request factory.
    (void)scheduled_policy(1U);
    auto pool=std::make_shared<strix::HipServingPrefixCache>(cache_policy,model->mapped->content_identity(),
        static_cast<std::uint32_t>(model->model.vocab_size));
    const auto unallocated=[&] {auto o=std::make_shared<SessionOwner>();o->model=model;return o;};
    const auto submit=[&](strix::GreedyServingScheduler& scheduler,const std::shared_ptr<SessionOwner>& owner,
        std::span<const std::uint32_t> tokens,bool prepared) {
        return scheduler.submit(request(*model,tokens),compatibility(*model),reinterpret_cast<std::uintptr_t>(&owner->session),
            1U,reservation(*model),owner,[owner,prepared,pool] {
                // Scheduler reserves BEFORE the first device state/scratch/selector allocation.
                initialize_session(owner);
                require(owner->memory->scratch().allocation_generation==1U,"future factory allocation epoch differed from its reservation");
                return backend(owner,prepared,pool);
            });
    };
    {
        strix::GreedyServingScheduler scheduler(scheduled_policy(1U),compatibility(*model),model);
        auto owner=unallocated();const std::weak_ptr<SessionOwner> source_lifetime=owner;
        auto submitted=submit(scheduler,owner,prompt,false);
        require(submitted.admission==strix::ServingAdmission::accepted,"cache prime admission failed");scheduler.run_until_idle();
        require(submitted.request.state()==strix::ServingRequestState::completed,"cache prime request failed");
        same_result(baseline,submitted.request.result());same_metadata(reference->session,owner->session);
        require(capture_state(owner,directory/"cache-prime")==baseline_state,"capture mutated the prime request's seven states");
        const auto stats=pool->stats();require(stats.entries==1U && stats.inserted==1U && stats.hits==0U,"prime did not publish exactly one actual finite frontier");
        submitted.request={};owner.reset();
        require(source_lifetime.expired(),"cache retained the originating request state/scratch/stream outside its reservation");
    }
    for(const std::uint32_t concurrency:{1U,2U,4U}) {
        strix::GreedyServingScheduler scheduler(scheduled_policy(concurrency),compatibility(*model),model);
        std::vector<std::shared_ptr<SessionOwner>> owners;std::vector<strix::ServingRequestHandle> handles;
        for(std::uint32_t n=0;n<concurrency;++n) {
            auto owner=unallocated();auto submitted=submit(scheduler,owner,prompt,(n%2U)!=0U);
            require(submitted.admission==strix::ServingAdmission::accepted,"cache-hit request was not reserved");
            owners.push_back(owner);handles.push_back(submitted.request);
        }
        scheduler.run_until_idle();
        for(std::uint32_t n=0;n<concurrency;++n) {
            require(handles[n].state()==strix::ServingRequestState::completed,"cache-hit serving request failed");
            const auto& result=handles[n].result();
            require(result.generated_tokens==baseline.generated_tokens && result.final_position==baseline.final_position &&
                result.stop_reason==baseline.stop_reason && result.restored_prefix_tokens==prompt.size() &&
                result.steps_executed+result.restored_prefix_tokens==baseline.steps_executed,
                "cache hit re-executed restored inputs, consumed the pending token twice or changed policy");
            same_metadata(reference->session,owners[n]->session);
            require(capture_state(owners[n],directory/("cache-hit-"+std::to_string(concurrency)+"-"+std::to_string(n)))==baseline_state,
                "cross-request cache restore changed a complete state component");
        }
    }
    {
        auto longer=prompt;longer.push_back(prompt[0]);longer.push_back(prompt[1]);
        auto serial=fresh_session(model);auto plain=backend(serial,false);
        const auto expected=strix::execute_serial_greedy_token_request(request(*model,longer),*plain);plain.reset();
        const auto expected_state=capture_state(serial,directory/"cache-longer-reference");
        strix::GreedyServingScheduler scheduler(scheduled_policy(1U),compatibility(*model),model);
        auto owner=unallocated();const auto submitted=submit(scheduler,owner,longer,true);
        require(submitted.admission==strix::ServingAdmission::accepted,"proper-prefix cache admission failed");scheduler.run_until_idle();
        require(submitted.request.state()==strix::ServingRequestState::completed,"proper-prefix restore failed");
        const auto& result=submitted.request.result();
        require(result.restored_prefix_tokens==prompt.size() && result.generated_tokens==expected.generated_tokens &&
            result.final_position==expected.final_position && result.stop_reason==expected.stop_reason &&
            result.steps_executed+result.restored_prefix_tokens==expected.steps_executed,
            "proper-prefix hit skipped or re-executed the explicit known suffix");
        same_metadata(serial->session,owner->session);
        require(capture_state(owner,directory/"cache-proper-hit")==expected_state,"proper-prefix hit changed seven-state results");
    }
    {
        auto prefix_reference=fresh_session(model);auto serial=backend(prefix_reference,false);
        for(std::size_t i=0;i+1U<prompt.size();++i) serial->execute_state(prompt[i],i);
        const auto pending=serial->execute_greedy(prompt.back(),prompt.size()-1U,model->model.vocab_size);serial.reset();
        const auto expected_state=capture_state(prefix_reference,directory/"cache-cancel-reference");
        auto owner=fresh_session(model);auto actual=backend(owner,true,pool);
        auto cancel_request=request(*model,prompt);
        // Explicit caller EOS distinct from this known scalar exercises pause/
        // cancel even when the model's default EOS is its first greedy output.
        cancel_request.eos_token_id=pending==0U?1U:0U;
        strix::SerialGreedyTokenCursor cursor(cancel_request);cursor.advance(*actual,1U);
        require(cursor.result().generated_tokens==std::vector<std::uint32_t>{pending} && cursor.result().steps_executed==0U &&
            cursor.result().restored_prefix_tokens==prompt.size(),"exact cache hit did not deliver its unconsumed bonus without a target call");
        cursor.cancel();cursor.advance(*actual,1U);
        require(cursor.cancelled() && cursor.result().steps_executed==0U,"cancelled cached bonus was later consumed");
        same_metadata(prefix_reference->session,owner->session);
        require(capture_state(owner,directory/"cache-cancelled")==expected_state,"cancel changed the restored prefix frontier");
    }
    {
        const strix::HipServingPrefixCachePolicy too_small{1U,1U,1U,context_capacity};
        auto bounded=std::make_shared<strix::HipServingPrefixCache>(too_small,model->mapped->content_identity(),
            static_cast<std::uint32_t>(model->model.vocab_size));
        auto owner=fresh_session(model);auto actual=backend(owner,false,bounded);
        const auto result=strix::execute_serial_greedy_token_request(request(*model,prompt),*actual);
        same_result(baseline,result);same_metadata(reference->session,owner->session);
        require(result.restored_prefix_tokens==0U && bounded->stats().entries==0U && bounded->stats().skipped==1U &&
            owner->session.health==strix::SerialSessionHealth::ready,"benign snapshot-budget admission miss changed normal inference");
        require(capture_state(owner,directory/"cache-budget-miss")==baseline_state,"budget miss changed complete target states");
    }
    {
        auto empty=std::make_shared<strix::HipServingPrefixCache>(cache_policy,model->mapped->content_identity(),
            static_cast<std::uint32_t>(model->model.vocab_size));
        auto owner=fresh_session(model);auto actual=backend(owner,false,empty);
        auto zero=request(*model,prompt);zero.max_new_tokens=0U;
        const auto result=strix::execute_serial_greedy_token_request(zero,*actual);
        require(result.generated_tokens.empty() && result.final_position==prompt.size() && empty->stats().capture_attempts==0U &&
            empty->stats().entries==0U,"zero-emission request published a cache frontier without requested head delivery");
    }
    {
        auto empty=std::make_shared<strix::HipServingPrefixCache>(cache_policy,model->mapped->content_identity(),
            static_cast<std::uint32_t>(model->model.vocab_size));
        auto owner=fresh_session(model);auto actual=backend(owner,false,empty);
        require(!actual->try_restore_greedy_prefix(request(*model,prompt)),"fresh failed-capture fixture unexpectedly hit");
        for(std::size_t i=0;i+1U<prompt.size();++i) actual->execute_state(prompt[i],i);
        const auto pending=actual->execute_greedy(prompt.back(),prompt.size()-1U,model->model.vocab_size);
        auto forged=prompt;forged[0]^=1U;bool rejected=false;
        try {actual->capture_greedy_prefix(forged,pending,model->model.vocab_size);}
        catch(const std::invalid_argument& error) {
            rejected=std::string_view(error.what())=="serving cache capture has no actual terminal finite-selection witness";
        }
        require(rejected && actual->failed() && owner->session.health==strix::SerialSessionHealth::poisoned &&
            owner->session.arithmetic_lineage.tainted && owner->session.next_position==prompt.size() && empty->stats().entries==0U,
            "postconsumption capture failure left a healthy reusable session or published a forged prefix");
    }

    {
        const auto device=model->guard->device_index();int devices=0;check(hipGetDeviceCount(&devices));
        auto release=std::async(std::launch::async,[retained=std::move(pool),device,devices]() mutable {
            if(devices>1) check(hipSetDevice((device+1)%devices));
            retained.reset(); // final entries/allocator freed on a different thread/device TLS
            int observed=-1;check(hipGetDevice(&observed));require(observed==device,"cache eviction/destruction did not bind its original device");
        });release.get();
    }

}
struct Gate {
    std::mutex mutex;
    std::condition_variable condition;
    bool entered=false,released=false;
    void pause() {
        std::unique_lock lock(mutex);entered=true;condition.notify_all();
        condition.wait(lock,[&]{return released;});
    }
    void release() {
        std::lock_guard lock(mutex);released=true;condition.notify_all();
    }
    void wait_and_cancel(const strix::ServingRequestHandle& handle) {
        std::unique_lock lock(mutex);
        const bool ready=condition.wait_for(lock,std::chrono::seconds(120),[&]{return entered;});
        const bool cancelled=ready && handle.cancel();
        released=true;condition.notify_all();
        require(ready&&cancelled,"gated request did not reach its deterministic cancellation seam");
    }
};
class GatedBackend final : public strix::SerialGreedyStepBackend {
public:
    GatedBackend(std::unique_ptr<strix::HipServingGreedyBackend> actual,std::shared_ptr<Gate> gate)
        :actual_(std::move(actual)),gate_(std::move(gate)) {}
    std::vector<float> execute(std::uint32_t t,std::uint64_t p) override { pause();return actual_->execute(t,p); }
    std::uint32_t execute_greedy(std::uint32_t t,std::uint64_t p,std::uint64_t v) override { pause();return actual_->execute_greedy(t,p,v); }
    void execute_state(std::uint32_t t,std::uint64_t p) override { pause();actual_->execute_state(t,p); }
private:
    void pause() { if(!paused_) {paused_=true;gate_->pause();} }
    std::unique_ptr<strix::HipServingGreedyBackend> actual_;
    std::shared_ptr<Gate> gate_;
    bool paused_=false;
};
int qualify(const Args& args) {
    const auto executable=std::filesystem::canonical("/proc/self/exe");
    const auto executable_sha=digest(executable);
    auto model=load_model(args);
    const auto prompt=read_prompt(args,model->model.vocab_size);
    std::filesystem::create_directory(args.output);
    auto base=fresh_session(model);
    auto base_backend=backend(base,false);
    const auto baseline=strix::execute_serial_greedy_request(request(*model,prompt),*base_backend);
    base_backend.reset();
    const auto base_state=capture_state(base,args.output/"baseline");
    const auto base_metadata=base->session;
    qualify_prefix_cache(model,prompt,base,baseline,base_state,args.output);
    for(const std::uint32_t concurrency:{1U,2U,4U}) {
        strix::GreedyServingScheduler scheduler(policy(*model,concurrency),compatibility(*model),model);
        std::vector<std::shared_ptr<SessionOwner>> owners;
        std::vector<std::shared_ptr<std::unique_ptr<strix::HipServingGreedyBackend>>> backends;
        std::vector<strix::ServingRequestHandle> handles;
        // Complete allocation/registration/pool setup before any request starts.
        // The copyable factory transfers its one unique backend exactly once.
        for(std::uint32_t i=0;i<concurrency;++i) {
            auto owner=fresh_session(model);independent(*owner,*base);
            for(const auto& previous:owners) independent(*owner,*previous);
            owners.push_back(owner);
            backends.push_back(std::make_shared<std::unique_ptr<strix::HipServingGreedyBackend>>(backend(owner,(i%2U)!=0)));
        }
        for(std::uint32_t i=0;i<concurrency;++i) {
            const auto& owner=owners[i];const auto retained=backends[i];
            const auto submission=scheduler.submit(request(*model,prompt),compatibility(*model),
                reinterpret_cast<std::uintptr_t>(&owner->session),owner->memory->state().device_allocation_generation,
                reservation(*model),owner,[retained] {return std::move(*retained);});
            require(submission.admission==strix::ServingAdmission::accepted,"serving C1/C2/C4 admission failed");
            handles.push_back(submission.request);
        }
        scheduler.run_until_idle();
        for(std::uint32_t i=0;i<concurrency;++i) {
            require(handles[i].state()==strix::ServingRequestState::completed,"serving worker failed");
            same_result(baseline,handles[i].result());same_metadata(base_metadata,owners[i]->session);
            require(capture_state(owners[i],args.output/("c"+std::to_string(concurrency)+"-"+std::to_string(i)))==base_state,
                "concurrent session full state differs from serial reference");
        }
        // Handles own the backend until terminal release on this main thread.
        // If available, select another device to exercise original-device dtor.
        int devices=0;check(hipGetDeviceCount(&devices));
        if(devices>1) check(hipSetDevice((model->guard->device_index()+1)%devices));
        handles.clear();owners.clear();
        check(hipSetDevice(model->guard->device_index()));model->guard->require_original();
    }
    {
        auto cancelled=fresh_session(model),healthy=fresh_session(model),prefix_reference=fresh_session(model);
        independent(*cancelled,*healthy);independent(*cancelled,*base);independent(*healthy,*base);
        auto prefix_backend=backend(prefix_reference,false);prefix_backend->execute_state(prompt[0],0);prefix_backend.reset();
        const auto prefix_hashes=capture_state(prefix_reference,args.output/"cancel-reference");
        auto gate=std::make_shared<Gate>();
        strix::GreedyServingScheduler scheduler(policy(*model,2),compatibility(*model),model);
        struct ReleaseGate {std::shared_ptr<Gate> gate;~ReleaseGate(){gate->release();}} release{gate};
        const auto a=scheduler.submit(request(*model,prompt),compatibility(*model),reinterpret_cast<std::uintptr_t>(&cancelled->session),
            cancelled->memory->state().device_allocation_generation,reservation(*model),cancelled,[cancelled,gate] {
                return std::make_unique<GatedBackend>(backend(cancelled,false),gate);
            });
        require(a.admission==strix::ServingAdmission::accepted,"cancel request was not admitted");
        const auto b=scheduler.submit(request(*model,prompt),compatibility(*model),reinterpret_cast<std::uintptr_t>(&healthy->session),
            healthy->memory->state().device_allocation_generation,reservation(*model),healthy,[healthy] {return backend(healthy,true);});
        require(b.admission==strix::ServingAdmission::accepted,"healthy sibling was not admitted");
        gate->wait_and_cancel(a.request);scheduler.run_until_idle();
        require(a.request.state()==strix::ServingRequestState::cancelled && b.request.state()==strix::ServingRequestState::completed,
            "one cancellation affected its independent sibling");
        same_metadata(cancelled->session,prefix_reference->session);
        require(capture_state(cancelled,args.output/"cancelled")==prefix_hashes,"cancelled request did not retain its completed prefix state");
        same_result(baseline,b.request.result());same_metadata(healthy->session,base_metadata);
        require(capture_state(healthy,args.output/"cancel-healthy")==base_state,"cancellation changed sibling full state");
    }
    {
        auto owner=fresh_session(model);auto actual=backend(owner,true);
        const auto saved=owner->session;
        // Existing BUILD_TESTING seam reports failure only after real selector
        // partial kernels have been queued. No fake token or mocked dispatcher.
        strix::arm_hip_greedy_output_diagnostic_fault(strix::HipGreedyOutputDiagnosticFault::after_partials);
        bool threw=false;
        try { (void)actual->execute_greedy(prompt[0],0,model->model.vocab_size); }
        catch(const std::runtime_error& error) {
            require(std::string_view(error.what())=="injected GPU greedy host boundary failure",
                "output fault fixture failed before its intended queued-selector boundary");
            threw=true;
        }
        strix::clear_hip_greedy_output_diagnostic_fault();
        require(threw && actual->failed() && owner->session.health==strix::SerialSessionHealth::poisoned &&
            owner->session.arithmetic_lineage.tainted && owner->session.next_position==saved.next_position &&
            owner->session.committed_tokens==saved.committed_tokens && owner->session.state.generations==saved.state.generations,
            "queued output failure published a token commit or failed to poison");
        check(hipStreamQuery(owner->scope.stream));
        bool rejected=false;
        try {actual->execute_state(prompt[0],0);} catch(const std::logic_error&) {rejected=true;}
        require(rejected,"failed serving backend was reused");
        actual.reset();model->guard->run([] {});
    }
    {
        auto owner=fresh_session(model);
        const auto first=prompt[0];
        auto future=std::async(std::launch::async,[owner,first] {
            check(hipSetDevice(owner->model->guard->device_index()));
            auto actual=backend(owner,false);actual->execute_state(first,0);return actual;
        });
        auto actual=future.get();
        int devices=0;check(hipGetDeviceCount(&devices));
        if(devices>1) check(hipSetDevice((model->guard->device_index()+1)%devices));
        actual.reset(); // constructed/used on worker, destroyed here on main
        model->guard->require_original();
        require(owner->session.next_position==1 && owner->session.health==strix::SerialSessionHealth::ready,
            "cross-thread backend destruction altered its completed session");
        (void)capture_state(owner,args.output/"cross-thread-destruction");
    }
    require(capture_state(base,args.output/"baseline-after-fault")==base_state,"another session's output error changed baseline owners");
    same_metadata(base->session,base_metadata);
    require(digest(args.prompt)==args.prompt_sha && digest(args.model)==args.model_sha && digest(executable)==executable_sha,
        "qualification model/prompt/executable pins changed");
    model->guard->run([] {});
    const auto pending=args.output/"result.pending.json";
    std::ofstream out(pending,std::ios::binary);out.exceptions(std::ios::failbit|std::ios::badbit);
    out<<"{\"schema\":\"serving-native-qualification-v1\",\"finished\":true,\"passed\":true,\"C\":[1,2,4],"
        "\"full_state_components\":7,\"complete_prefix_cache_cross_request\":true,\"cache_exact_and_proper_hits\":true,\"cache_pending_cancel_and_budget_miss\":true,\"baseline_full_F32_CPU_selection\":true,\"cancel_one_independent\":true,\"queued_selector_fault\":true,"
        "\"fault_is_reported_host_seam_not_driver_failure\":true,\"cross_thread_destruction\":true,\"native_model_qualification_only\":true,"
        "\"quality\":false,\"performance\":false,\"model_sha256\":\""<<args.model_sha<<"\",\"prompt_sha256\":\""<<args.prompt_sha
        <<"\",\"executable_sha256\":\""<<executable_sha<<"\",\"execution_identity\":"<<model->compiled.execution_identity
        <<",\"steps\":"<<baseline.steps_executed<<",\"final_position\":"<<baseline.final_position<<",\"tokens\":[";
    for(std::size_t i=0;i<baseline.generated_tokens.size();++i) out<<(i?",":"")<<baseline.generated_tokens[i];
    out<<"],\"baseline_committed\":"<<base_metadata.committed_tokens<<",\"baseline_lineage_identity\":"
        <<base_metadata.arithmetic_lineage.history_identity<<",\"baseline_generations\":[";
    for(std::size_t i=0;i<base_metadata.state.generations.size();++i) out<<(i?",":"")<<base_metadata.state.generations[i];
    out<<"],\"baseline_state_sha256\":[";
    for(std::size_t i=0;i<base_state.size();++i) out<<(i?",":"")<<'"'<<base_state[i]<<'"';
    out<<"]}\n";out.close();std::filesystem::rename(pending,args.output/"result.json");return 0;
}
} // namespace
int main(int argc,char** argv) {
    try {return qualify(parse(argc,argv));}
    catch(const std::exception& error) {std::cerr<<"serving native qualification failed: "<<error.what()<<'\n';return 1;}
}
