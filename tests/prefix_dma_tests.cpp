#include "../apps/hip_prefix_state_cache.hpp"
#include "strix/prompt_terminal_diagnostics.hpp"
#include "strix/mtp_target_window_payload.hpp"
#include <stdexcept>
#include <deque>
#include <map>
#include <iostream>

namespace {
bool queued = false, freed_pending = false, allocation_failure = false;
hipError_t drain_error = hipSuccess, copy_error = hipSuccess;
unsigned calls = 0, waits = 0, copies = 0;
bool snapshot_shim=false;
struct ShimEvent {bool ready=false;};
std::map<hipStream_t,std::deque<std::function<bool()>>> operations;
void pump(hipStream_t stream) {
    auto& q=operations[stream];
    while(!q.empty() && q.front()()) q.pop_front();
}

void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
}
extern "C" hipError_t hipStreamCreateWithFlags(hipStream_t* p, unsigned) { *p = reinterpret_cast<void*>(1); return hipSuccess; }
extern "C" hipError_t hipStreamDestroy(hipStream_t) { return hipSuccess; }
extern "C" hipError_t hipDeviceSynchronize() { return hipSuccess; }
extern "C" hipError_t hipStreamSynchronize(hipStream_t stream) {
    ++waits;
    if(snapshot_shim && drain_error==hipSuccess) pump(stream);
    if (drain_error == hipSuccess) queued = false;
    return drain_error;
}
extern "C" hipError_t hipMemcpyAsync(void* dest, const void* src, std::size_t n, hipMemcpyKind, hipStream_t stream) {
    ++copies;
    if(snapshot_shim) {
        if(copy_error==hipSuccess) operations[stream].push_back([=] {std::memcpy(dest,src,n);return true;});
    } else queued=true;
    return copy_error;
}
extern "C" hipError_t hipMalloc(void** p,std::size_t n) {
    if(allocation_failure) return hipErrorOutOfMemory;
    *p=new std::byte[n];return hipSuccess;
}
extern "C" hipError_t hipFree(void* p) {delete[] static_cast<std::byte*>(p);return hipSuccess;}
extern "C" hipError_t hipEventCreateWithFlags(hipEvent_t* p,unsigned) {*p=new ShimEvent;return hipSuccess;}
extern "C" hipError_t hipEventDestroy(hipEvent_t p) {delete static_cast<ShimEvent*>(p);return hipSuccess;}
extern "C" hipError_t hipEventRecord(hipEvent_t p,hipStream_t stream) {
    operations[stream].push_back([=] {static_cast<ShimEvent*>(p)->ready=true;return true;});return hipSuccess;
}
extern "C" hipError_t hipEventQuery(hipEvent_t p) {
    if(drain_error!=hipSuccess) return drain_error;
    return static_cast<ShimEvent*>(p)->ready?hipSuccess:hipErrorNotReady;
}
extern "C" hipError_t hipEventSynchronize(hipEvent_t p) {
    ++waits;
    for(unsigned i=0;i<3;++i) for(auto& [stream,q]:operations) { (void)q;pump(stream); }
    return hipEventQuery(p);
}
extern "C" hipError_t hipStreamWaitEvent(hipStream_t stream,hipEvent_t p,unsigned) {
    operations[stream].push_back([=] {return static_cast<ShimEvent*>(p)->ready;});return hipSuccess;
}
extern "C" const char* hipGetErrorString(hipError_t) { return "injected HIP failure"; }

void async_snapshot_tests();
void batch_snapshot_tests();

void prefix_dma_tests() {
    using namespace strix;
    using namespace strix::decode_cli;
    {
        const std::array<float,4> row{0.0F,-0.0F,1.0F,-2.0F};
        require(!raw_fp32_logits_digest({}),"missing terminal row fails closed");
        require(raw_fp32_logits_digest(row)==sha256_bytes(std::as_bytes(std::span(row))),
            "terminal digest hashes the full raw FP32 vocabulary row");
        OutputLogitsChain chain;chain.append(raw_fp32_logits_digest(row));
        const auto first=chain.digest;chain.append(raw_fp32_logits_digest(row));
        require(chain.rows==2 && chain.digest!=first,"every output row extends the logits chain");
        // Ticket snapshots must outlive owner reuse and terminal-stripe overwrite/reset.
        auto owner_digest=raw_fp32_logits_digest(row);
        OutputLogitsSnapshot ticket;ticket.capture(owner_digest);
        const auto saved=ticket.digest;
        // The window lease moved into the shared payload header. Keep both
        // independent diagnostic channels when tickets outlive scratch reuse.
        MtpTargetWindowLease window;window.tokens.resize(2);window.terminal_digest=saved;
        require(!window.terminal_logits_digest(0) && window.terminal_logits_digest(1)==saved,
            "uncertified window preserves terminal-only diagnostics");
        auto different=row;different[0]=7.0F;const auto second=raw_fp32_logits_digest(different);
        window.row_digests={saved,second};window.logit_digests={*saved,*second};
        require(window.terminal_logits_digest(0)==saved && window.terminal_logits_digest(1)==second &&
            !window.terminal_logits_digest(2),"certified moved lease resolves each row separately");
        OutputLogitsSnapshot accepted;accepted.capture(window.terminal_logits_digest(0));
        const auto committed_step=window.logit_digests[0];
        window.row_digests.clear();window.logit_digests.clear();window.terminal_digest.reset();
        require(accepted.digest==saved && committed_step==*saved,
            "moved lease preserves per-row terminal snapshots and committed-step hashes");
        auto stripe=row;stripe[0]=42.0F;
        owner_digest=raw_fp32_logits_digest(stripe);
        require(ticket.digest==saved && ticket.digest!=owner_digest,"ticket digest survives stripe reuse");
        owner_digest.reset();
        require(ticket.digest==saved,"ticket digest survives owner reset");
        // A missing row poisons just this request, even if later rows exist.
        chain.append({});chain.append(saved);
        require(chain.error && !chain.complete(2) && chain.rows==2,"missing output row fails closed without throwing");
        OutputLogitsChain other;other.append(ticket.digest);
        require(other.complete(1),"another request completes after a diagnostic failure");
        OutputLogitsChain short_chain;short_chain.append(saved);
        require(!short_chain.complete(2),"missing bonus row fails coverage");
        set_prefix_batch_certification_enabled(true);
        require(prefix_batch_certification_enabled(),"resolved policy crosses translation units");
        set_prefix_batch_certification_enabled(false);
        require(!prefix_batch_certification_enabled(),"default policy remains off");
        auto changed=row;changed[1]=0.0F;
        require(raw_fp32_logits_digest(changed)!=raw_fp32_logits_digest(row),"terminal digest preserves signed zero bits");
        SerialSessionContext session;
        session.arithmetic_lineage=prepare_prefill_chunk_lineage(session.arithmetic_lineage,0,2,1,2,1);
        const auto original=terminal_generations_lineage_digest(session);
        session.state.generations[0]++;
        require(terminal_generations_lineage_digest(session)!=original,"terminal metadata covers generations");
        session.state.generations[0]--;
        session.arithmetic_lineage.intervals.back().reduction_plan_identity++;
        require(terminal_generations_lineage_digest(session)!=original,"terminal metadata covers canonical lineage intervals");
        session.arithmetic_lineage.intervals.back().reduction_plan_identity--;
        session.arithmetic_lineage.tainted=true;
        require(terminal_generations_lineage_digest(session)!=original,"terminal metadata covers taint");
    }
    {
        PrefixStateCacheIdentityInputs in;
        in.model_content=sha256_bytes(std::as_bytes(std::span("model",5)));
        in.block_rows=1024; in.state_layout_identity=1; in.runtime_compatibility_identity=2;
        in.decode_execution_identity=3; in.prefill_execution_identity=4; in.tiled_execution_identity=5;
        const auto signature=[&] {return prefix_effective_execution_signature("effective-cli",false,false);};
        in.effective_execution=signature();
        const auto original=prefix_state_cache_identity(in);
        for(const char* name : {"STRIX_QSA_UNION_MERGE","STRIX_QSA_SCORE_ROWS64","STRIX_QSA_POOL_PACKED",
                "STRIX_QSA_DECODE_POOL","STRIX_QSA_DECODE_SELECT","STRIX_QSA_DECODE_POOL_BF16","STRIX_QSA_DECODE_POOL_CHECK",
                "STRIX_QPF2K_DEFAULTS","STRIX_BASELINE","STRIX_QPF2K_QSA_MASK"}) {
            const char* old=std::getenv(name);
            const std::optional<std::string> saved=old?std::optional<std::string>{old}:std::nullopt;
            ::setenv(name,"0",1); in.effective_execution=signature();
            const auto off=prefix_state_cache_identity(in);
            ::setenv(name,"1",1); in.effective_execution=signature();
            require(prefix_state_cache_identity(in)!=off,"one runtime control changes the prefix namespace");
            if(saved) ::setenv(name,saved->c_str(),1); else ::unsetenv(name);
        }
        in.effective_execution=prefix_effective_execution_signature("effective-cli",true,false);
        require(prefix_state_cache_identity(in)!=original,"CLI baseline suppression changes the key");
        in.effective_execution=prefix_effective_execution_signature("effective-cli",false,true);
        require(prefix_state_cache_identity(in)!=original,"GSQ suppression changes the key");
        {
            SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
            SessionStateMemoryBinding memory;memory.layout_identity=1;
            QwenTiledPlanConfig plan;plan.rows=2;plan.block={2,QwenPlanOutput::terminal};
            QwenBatchConfig batch;
            const auto binding=[&](const ModelContentIdentity& sig) {
                HipPrefixStateCacheSession adapter({},sig,2,2,ModelContract{},layout,memory);
                adapter.set_plan(plan,8,true,2,batch);
                return std::pair{adapter.active_identity(),adapter.execution_certificate(2)};
            };
            const auto base=binding(signature());
            for(const char* name : {"STRIX_QSA_UNION_MERGE","STRIX_QSA_SCORE_ROWS64","STRIX_QSA_POOL_PACKED",
                    "STRIX_QSA_DECODE_POOL","STRIX_QSA_DECODE_SELECT","STRIX_QSA_DECODE_POOL_BF16","STRIX_QSA_DECODE_POOL_CHECK",
                    "STRIX_CTX_EXACT_MEMORY","STRIX_QSA_SCORE_RING","STRIX_QSA_SCORE_RING_ROWS","STRIX_TERMINAL_LOGITS_ARENA",
                    "STRIX_QPF2K_DEFAULTS","STRIX_BASELINE","STRIX_QPF2K_QSA_MASK",
                    "STRIX_PROMPT_BATCH_SHARED","STRIX_BATCH_SERVICE_MS","STRIX_BATCH_SERVICE_LOOKAHEAD",
                    "STRIX_BATCH_PROMPT_SHARE","STRIX_BATCH_SHORT_FIRST_ROWS","STRIX_BATCH_INTERLEAVE"}) {
                const char* old=std::getenv(name);
                const std::optional<std::string> saved=old?std::optional<std::string>{old}:std::nullopt;
                ::setenv(name,"0",1);const auto off=binding(signature());
                ::setenv(name,"1",1);const auto on=binding(signature());
                require(off.first!=on.first && off.second!=on.second,"canonical v4 control binds batch namespace and certificate");
                if(saved) ::setenv(name,saved->c_str(),1);else ::unsetenv(name);
            }
            for(const auto sig : {prefix_effective_execution_signature("changed-cli",false,false),
                    prefix_effective_execution_signature("effective-cli",true,false),
                    prefix_effective_execution_signature("effective-cli",false,true),
                    prefix_effective_execution_signature("effective-cli",false,false,
                        RotaryIdentity(RotaryDescriptor::yarn(2),original))}) {
                const auto changed=binding(sig);
                require(changed.first!=base.first && changed.second!=base.second,"resolved execution and rotary bind batch proof");
            }
        }
        in.effective_execution={}; bool refused=false;
        try {(void)prefix_state_cache_identity(in);} catch(const std::invalid_argument&) {refused=true;}
        require(refused,"incomplete pre-v2 prefix identity refused");
    }
    {
        const auto table=sha256_bytes(std::as_bytes(std::span("table",5)));
        const std::array<RotaryIdentity,3> rotary{RotaryIdentity{},
            RotaryIdentity(RotaryDescriptor::yarn(2),table),RotaryIdentity(RotaryDescriptor::yarn(4),table)};
        std::array<ModelContentIdentity,3> signatures;
        for(std::size_t i=0;i<rotary.size();++i)
            signatures[i]=prefix_effective_execution_signature("same-effective-cli",false,false,rotary[i]);
        require(signatures[0]!=signatures[1] && signatures[1]!=signatures[2],"resolved rotary mode and factors bind signature");
        const auto other_table=sha256_bytes(std::as_bytes(std::span("other",5)));
        require(signatures[1]!=prefix_effective_execution_signature("same-effective-cli",false,false,
            RotaryIdentity(RotaryDescriptor::yarn(2),other_table)),"derived table identity binds signature");
        for(const unsigned route:{0U,1U,2U}) for(std::size_t producer=0;producer<rotary.size();++producer) {
            const bool certified=route!=0;
            auto alloc=[](std::uint64_t n) {return PrefixCacheBuffer{
                std::shared_ptr<std::byte>(new std::byte[n?n:1]{},std::default_delete<std::byte[]>()),n};};
            auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{4096,{},0},alloc);
            SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
            SessionStateMemoryBinding memory;memory.layout_identity=1;
            std::array<std::unique_ptr<HipPrefixStateCacheSession>,3> adapters;
            for(std::size_t i=0;i<rotary.size();++i) {
                adapters[i]=std::make_unique<HipPrefixStateCacheSession>(store,signatures[i],2,2,ModelContract{},layout,memory);
                if(certified) {QwenTiledPlanConfig plan;plan.rows=2;plan.block={2,QwenPlanOutput::terminal};
                    if(route==1) adapters[i]->set_plan(plan,8,true);
                    else adapters[i]->set_serial_plan(plan,8,{{0,2,2,true,false,false,false,true,false}},
                        [](auto) {return true;},true);}
            }
            const std::vector<std::uint32_t> tokens(8,1);
            const auto ns=adapters[producer]->active_identity();
            auto seg=std::make_shared<PrefixCacheSegment>();seg->identity=ns;seg->parent=ns;
            seg->digest=prefix_cache_block_digests(ns,tokens,2,1)[0];seg->block_rows=2;seg->tokens={1,1};seg->payload=alloc(2);
            auto cp=std::make_shared<PrefixCacheCheckpoint>();cp->identity=ns;cp->digest=seg->digest;
            cp->block_rows=2;cp->position=2;cp->segments={seg};cp->fixed=alloc(256);
            if(certified) cp->producer_certificate=adapters[producer]->execution_certificate(2);
            require(store->insert(cp),"rotary checkpoint captured");
            require(store->find_longest(ns,tokens,2,2)!=nullptr,"same rotary checkpoint hits");
            for(std::size_t reader=0;reader<rotary.size();++reader) if(reader!=producer) {
                require(!store->find_longest(adapters[reader]->active_identity(),tokens,2,2),"cross-rotary checkpoint misses both directions");
                if(certified) require(adapters[reader]->execution_certificate(2)!=cp->producer_certificate,
                    "cross-rotary producer certificate differs");
                SerialSessionContext fresh;fresh.state_layout_identity=1;
                require(adapters[reader]->begin_request(fresh,tokens)==0 && fresh.next_position==0,
                    "cross-rotary restore misses before state writes");
            }
        }
    }
    const auto id = sha256_bytes(std::as_bytes(std::span("dma-test", 8)));
    for (const auto failure : {hipSuccess, hipErrorOutOfMemory, hipErrorUnknown}) {
        for (unsigned scenario = 0; scenario < 3; ++scenario) {
            queued = freed_pending = false; calls = 0;
            drain_error = failure;
            allocation_failure = scenario == 0;
            copy_error = scenario == 1 ? hipErrorUnknown : hipSuccess;
            const auto before = PrefixDmaQuarantine::count.load();
            const auto before_bytes = PrefixDmaQuarantine::bytes.load();
            auto store = std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{4096, {}, 0}, [](std::uint64_t n) {
                if (++calls == 2 && allocation_failure) throw PrefixCacheAllocationFailure();
                return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n ? n : 1]{}, [](std::byte* p) {
                    if (queued) freed_pending = true;
                    delete[] p;
                }), n};
            });
            SerialSessionStateLayout layout; layout.layout_identity = 1; layout.context_tokens = 2;
            SessionStateMemoryBinding memory; memory.layout_identity = 1;
            {
                HipPrefixStateCacheSession capture(store, id, 1, 1, ModelContract{}, layout, memory);
                capture.add_rows(1, 1, 0); capture.add_fixed_device(1, 1);
                SerialSessionContext session; session.next_position = session.committed_tokens = 1; session.state_layout_identity = 1;
                bool threw = false;
                try { capture.after_block(session, std::vector<std::uint32_t>{1, 2}); }
                catch (const std::exception&) { threw = true; }
                require(threw == (failure != hipSuccess || scenario == 1), "capture fault propagation");
            }
            require(!freed_pending, "unresolved DMA destination freed");
            require(PrefixDmaQuarantine::count.load() == before + (failure == hipSuccess ? 0U : 1U), "quarantine owner count");
            require(PrefixDmaQuarantine::bytes.load() == before_bytes + (failure == hipSuccess ? 0U : scenario == 2 ? 257U : 1U), "quarantine byte count");
            require(store->counters().inserted == (failure == hipSuccess && scenario == 2 ? 1U : 0U), "partial capture published");
            require(store->counters().capture_failures == (failure == hipSuccess && scenario == 0 ? 1U : 0U), "capture fail-soft classification");
            queued = false;
        }
    }
    for (const auto failure : {hipErrorOutOfMemory, hipErrorUnknown}) {
        for (const bool fail_copy : {false, true}) {
            queued = freed_pending = allocation_failure = false; calls = 0;
            drain_error = copy_error = hipSuccess;
            const auto before = PrefixDmaQuarantine::count.load();
            const auto before_bytes = PrefixDmaQuarantine::bytes.load();
            {
                auto store = std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{4096, {}, 0}, [](std::uint64_t n) {
                    return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n ? n : 1]{}, [](std::byte* p) {
                        if (queued) freed_pending = true;
                        delete[] p;
                    }), n};
                });
                SerialSessionStateLayout layout; layout.layout_identity = 1; layout.context_tokens = 2;
                SessionStateMemoryBinding memory; memory.layout_identity = 1;
                HipPrefixStateCacheSession capture(store, id, 1, 1, ModelContract{}, layout, memory);
                capture.add_rows(1, 1, 0); capture.add_fixed_device(1, 1);
                SerialSessionContext session; session.next_position = session.committed_tokens = 1; session.state_layout_identity = 1;
                capture.after_block(session, std::vector<std::uint32_t>{1, 2});
                drain_error = failure; copy_error = fail_copy ? hipErrorUnknown : hipSuccess;
                SerialSessionContext fresh; fresh.state_layout_identity = 1;
                bool threw = false;
                try { (void)capture.begin_request(fresh, std::vector<std::uint32_t>{1, 2}); }
                catch (const std::exception&) { threw = true; }
                require(threw && fresh.health == SerialSessionHealth::poisoned, "restore failure poisons session");
            }
            require(!freed_pending && PrefixDmaQuarantine::count.load() == before + 1 &&
                PrefixDmaQuarantine::bytes.load() == before_bytes + 257, "restore source lease quarantine");
            queued = false;
        }
    }

    {
        queued=freed_pending=allocation_failure=false;drain_error=copy_error=hipSuccess;
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{4096,{},0},[](std::uint64_t n) {
            return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n?n:1]{},std::default_delete<std::byte[]>()),n};
        });
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        HipPrefixStateCacheSession adapter(store,id,2,2,ModelContract{},layout,memory);
        adapter.add_rows(1,1,0);adapter.add_fixed_device(1,1);
        QwenTiledPlanConfig config;config.rows=2;config.block={2,QwenPlanOutput::terminal};
        adapter.set_plan(config,8,true);
        auto other_id=id; other_id.bytes[0]^=1;
        HipPrefixStateCacheSession other(store,other_id,2,2,ModelContract{},layout,memory);
        other.set_plan(config,8,true);
        require(other.active_identity()!=adapter.active_identity(),"execution signature binds certified RAM namespace");
        require(other.execution_certificate(2)!=adapter.execution_certificate(2),"execution signature binds producer certificate");
        const std::vector<std::uint32_t> tokens(8,1);
        const auto digests=prefix_cache_block_digests(adapter.active_identity(),tokens,2,1);
        auto seg=std::make_shared<PrefixCacheSegment>();
        seg->identity=adapter.active_identity();seg->parent=seg->identity;seg->digest=digests[0];
        seg->block_rows=2;seg->tokens={1,1};seg->payload=store->allocate(2);
        auto cp=std::make_shared<PrefixCacheCheckpoint>();cp->identity=seg->identity;cp->digest=seg->digest;
        cp->block_rows=2;cp->position=2;cp->segments={seg};cp->fixed=store->allocate(256);
        cp->producer_certificate=adapter.execution_certificate(2);
        cp->producer_certificate[0]^=1U;
        require(store->insert(cp),"plan refusal seed");
        SerialSessionContext fresh;fresh.state_layout_identity=1;
        require(adapter.begin_request(fresh,tokens)==0 && !queued && fresh.next_position==0,
            "plan preflight miss performs zero live writes");
        // Correct producer proof but corrupt/truncated metadata must also miss before writes.
        cp->producer_certificate[0]^=1U;
        require(adapter.begin_request(fresh,tokens)==0 && !queued && fresh.next_position==0,
            "metadata preflight miss performs zero live writes");
        const auto stats=store->counters();
        require(stats.plan_misses==2 && stats.host_hits==0 && stats.restored_tokens==0,"preflight misses accounting");
    }

    {
        queued=freed_pending=allocation_failure=false;drain_error=copy_error=hipSuccess;
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{8192,{},0},[](std::uint64_t n) {
            return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n?n:1]{},std::default_delete<std::byte[]>()),n};
        });
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        HipPrefixStateCacheSession adapter(store,id,2,2,ModelContract{},layout,memory);
        adapter.add_rows(1,1,0);adapter.add_rows(2,1,1);adapter.add_fixed_device(1,1);
        QwenTiledPlanConfig config;config.rows=2;config.block={2,QwenPlanOutput::terminal};
        config.generations_per_row[0]=3;config.valid_state=1;
        config.block.execution=1;config.block.kernel=2;
        adapter.set_plan(config,8,true);
        const std::vector<std::uint32_t> tokens(8,1);
        SerialSessionContext session;session.state_layout_identity=1;
        for(std::uint64_t p=2;p<=4;p+=2) {
            session.arithmetic_lineage=prepare_prefill_chunk_lineage(session.arithmetic_lineage,p-2,p,1,2,1);
            session.next_position=session.committed_tokens=p;session.state.generations[0]=p*3U;
            session.state.valid=static_cast<PersistentState>(1);
            adapter.after_block(session,tokens);
        }
        const auto digests=prefix_cache_block_digests(adapter.active_identity(),tokens,2,2);
        require(store->find_segment(digests[0])->payload.bytes==3 && store->find_segment(digests[1])->payload.bytes==4,
            "shift-1 first/next segment geometry");
        // A valid warm checkpoint cannot authorize a policy/constraint verify request.
        adapter.set_policy_selection(true);
        SerialSessionContext refused;refused.state_layout_identity=1;queued=false;
        const auto checkpoints=store->counters().host_checkpoints;
        require(adapter.begin_request(refused,tokens)==0 && !queued && refused.next_position==0,
            "policy request refuses warm restore before device writes");
        require(adapter.request().refusal=="policy-selection-cache-unqualified","named policy cache refusal");
        adapter.after_block(session,tokens);
        require(!adapter.try_prepare_capture(session,tokens,nullptr,true) && !queued &&
            store->counters().host_checkpoints==checkpoints,"policy request refuses both capture paths");
        adapter.set_policy_selection(false);
        adapter.set_retention(false);
        SerialSessionContext fresh;fresh.state_layout_identity=1;
        require(adapter.begin_request(fresh,tokens)==4 && fresh.state.generations[0]==12,
            "certified no-store restore retains complete metadata");
        require(arithmetic_lineage_valid(fresh.arithmetic_lineage,4),"certified lineage restored");
        // Obtain the RAM lease and inject a generation-only corruption of the longest candidate.
        auto cp=store->find_longest(adapter.active_identity(),tokens,2,4);
        auto corrupt=std::const_pointer_cast<PrefixCacheCheckpoint>(cp);
        corrupt->metadata[9U*8U]^=std::byte{1};
        const auto before=store->counters();
        fresh={};fresh.state_layout_identity=1;queued=false;
        require(adapter.begin_request(fresh,tokens)==2,"generation mismatch falls back to shorter checkpoint");
        require(store->counters().plan_misses==before.plan_misses+1,"generation refusal counted");
        corrupt->metadata[9U*8U]^=std::byte{1};
        corrupt->segments[0].reset(); // Invalid section must be refused before token flattening.
        fresh={};fresh.state_layout_identity=1;queued=false;
        require(adapter.begin_request(fresh,tokens)==2,"malformed section safely falls back");
        // Different priming families cannot alias row segments even for identical tokens.
        config.priming=255;adapter.set_plan(config,8,true);
        const auto other=prefix_cache_block_digests(adapter.active_identity(),tokens,2,2);
        require(other[0]!=digests[0] && !store->find_segment(other[0]),"incompatible producer segment namespace");
        fresh={};fresh.state_layout_identity=1;queued=false;
        require(adapter.begin_request(fresh,tokens)==0 && !queued,"other family misses without writes");
    }

    {
        // Serial certificates use actual solo full-T units and a separate RAM namespace.
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{16384,{},0},[](std::uint64_t n) {
            return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n?n:1]{},std::default_delete<std::byte[]>()),n};
        });
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        QwenTiledPlanConfig config;config.rows=2;config.block={2,QwenPlanOutput::terminal,1,2};
        const std::vector<KnownPromptChunkUnit> trace{{0,2,2,true,false,false,false,true,false},
            {2,4,2,true,false,false,false,true,false},{4,6,2,true,false,false,false,true,false},
            {6,8,2,true,false,false,false,false,false}};
        const std::vector<std::uint32_t> tokens(8,1);
        HipPrefixStateCacheSession producer(store,id,2,2,ModelContract{},layout,memory);
        producer.add_rows(1,1,0);producer.add_fixed_device(1,1);
        producer.set_serial_plan(config,8,trace,[](auto) {return true;},true);
        SerialSessionContext session;session.state_layout_identity=1;
        (void)producer.begin_request(session,tokens);
        session.next_position=session.committed_tokens=2;
        session.arithmetic_lineage=prepare_prefill_chunk_lineage({},0,2,1,2,1);
        producer.observe_serial_unit(trace[0]);producer.after_block(session,tokens);
        require(store->counters().inserted==1,"serial actual full-T producer captured");
        auto cp=store->find_longest(producer.active_identity(),tokens,2,2);
        require(cp && cp->producer_certificate==producer.execution_certificate(2),"serial producer certificate stored");
        const auto serial_namespace=producer.active_identity();
        const auto attempt=[&](auto cold,bool resumable,const char* reason) {
            HipPrefixStateCacheSession consumer(store,id,2,2,ModelContract{},layout,memory);
            consumer.add_rows(1,1,0);consumer.add_fixed_device(1,1);
            consumer.set_serial_plan(config,8,cold,[=](auto) {return resumable;},true);
            SerialSessionContext fresh;fresh.state_layout_identity=1;queued=false;
            const auto before_copies=copies;
            const auto p=consumer.begin_request(fresh,tokens);
            if(reason) {
                require(!p && !queued && copies==before_copies && fresh.next_position==0 && fresh.committed_tokens==0 &&
                    fresh.arithmetic_lineage.intervals.empty(),"serial preflight miss has zero state writes");
                require(consumer.request().refusal==reason && consumer.request().certificate_declines.at(reason)>0,
                    "serial preflight decline is counted and named");
            } else require(p==2 && copies>before_copies && fresh.next_position==2,"compatible serial producer restores");
        };
        attempt(trace,true,nullptr);
        auto different=trace;different[0].headless=false;
        attempt(different,true,"serial-producer-plan-mismatch");
        // A consumer cursor unable to preserve its whole-prompt suffix must decline before DMA.
        attempt(trace,false,"serial-cursor-continuation-unsupported");
        auto corrupt=std::const_pointer_cast<PrefixCacheCheckpoint>(cp);
        const auto certificate=corrupt->producer_certificate;
        corrupt->producer_certificate=qwen_full_t_certificate(config,8,2);
        attempt(trace,true,"serial-producer-plan-mismatch");
        corrupt->producer_certificate=certificate;
        corrupt->metadata[0]^=std::byte{1};
        attempt(trace,true,"serial-checkpoint-preflight");
        corrupt->metadata[0]^=std::byte{1};
        for(unsigned variant=0;variant<4;++variant) {
            auto unsupported=trace;
            if(variant==0) unsupported[0].merged=true;
            if(variant==1) unsupported[0].balanced=true;
            if(variant==2) unsupported[0].deferred=true;
            if(variant==3) unsupported[0].end=4; // boundary lies inside a physical merged unit
            producer.set_serial_plan(config,8,unsupported,[](auto) {return true;},true);
            producer.observe_serial_unit(unsupported[0]);queued=false;
            producer.after_block(session,tokens);
            require(!queued && producer.request().refusal=="serial-capture-unit-unsupported",
                "merged/balanced/deferred/interior capture declines without writes");
        }
        producer.set_plan(config,8,true);
        require(producer.active_identity()!=serial_namespace,"MTP namespace differs from serial");
        SerialSessionContext fresh;fresh.state_layout_identity=1;queued=false;
        require(producer.begin_request(fresh,tokens)==0 && !queued,"MTP cannot restore serial checkpoint");
        producer.set_plan(config,8,false); // batch/uncertified route remains disabled
        require(producer.begin_request(fresh,tokens)==0 && !queued,"batch cannot restore serial checkpoint");
        const auto changed_id=sha256_bytes(std::as_bytes(std::span("changed-effective-signature",27)));
        HipPrefixStateCacheSession changed(store,changed_id,2,2,ModelContract{},layout,memory);
        changed.set_serial_plan(config,8,trace,[](auto) {return true;},true);
        require(changed.active_identity()!=serial_namespace && changed.begin_request(fresh,tokens)==0 && !queued,
            "effective signature change misses serial namespace without writes");
        require(store->counters().certificate_declines[5]>0,"named declines exported by store counters");
    }

    async_snapshot_tests();
    batch_snapshot_tests();

}


void async_snapshot_tests() {
    using namespace strix;
    using namespace strix::decode_cli;
    const ModelContract production;
    const auto fixed_device=gdn_matrix_state_bytes_per_session(production)+gdn_conv_state_bytes_per_session(production)+
        ple_conv_state_bytes_per_session(production)+production.hidden_size*production.hc_count*2U;
    require(fixed_device==116400128U,"production BF16 device fixed geometry");
    require(fixed_device+29952U*2048U==177741824U && fixed_device+29952U*8192U-2304U==361764608U,
        "production selected capture staging bytes");
    set_prefix_batch_certification_enabled(true);
    snapshot_shim=true;queued=false;drain_error=copy_error=hipSuccess;allocation_failure=false;
    const auto producer=reinterpret_cast<hipStream_t>(2), predictor=reinterpret_cast<hipStream_t>(3);
    const PrefixCaptureOrder ordered{producer,predictor,predictor};
    const auto id=sha256_bytes(std::as_bytes(std::span("snapshot",8)));
    const std::vector<std::uint32_t> tokens(8,1);
    // Repeated destruction, failed validation, cancellation and either completion ordering.
    for(unsigned scenario=0;scenario<8;++scenario) {
        operations.clear();
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{8192,{},0},[](std::uint64_t n) {
            return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n]{},std::default_delete<std::byte[]>()),n};
        });
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        auto adapter=std::make_unique<HipPrefixStateCacheSession>(store,id,2,2,ModelContract{},layout,memory);
        std::array<std::byte,8> target{},mtp{}; std::byte fixed{31},held{41};
        target[0]=std::byte{11};target[1]=std::byte{12};mtp[0]=std::byte{21};
        adapter->add_rows(reinterpret_cast<std::uintptr_t>(target.data()),1,0);
        adapter->add_rows(reinterpret_cast<std::uintptr_t>(mtp.data()),1,1);
        adapter->add_fixed_device(reinterpret_cast<std::uintptr_t>(&fixed),1);
        adapter->add_fixed_device(reinterpret_cast<std::uintptr_t>(&held),1,"mtp.held");
        QwenTiledPlanConfig config;config.rows=2;config.block={2,QwenPlanOutput::terminal};
        config.block.execution=1;config.block.kernel=2;config.priming=255;
        adapter->set_plan(config,8,true);
        SerialSessionContext session;session.state_layout_identity=1;
        session.next_position=session.committed_tokens=2;
        session.arithmetic_lineage=prepare_prefill_chunk_lineage(session.arithmetic_lineage,0,2,1,2,1);
        const auto before_waits=waits;
        adapter->set_retention(false);
        require(!adapter->try_prepare_capture(session,tokens,producer,true,ordered) && operations.empty(),"no-store skips before any submission");
        require(store->counters().snapshot_preparations==0,"no-store does not prepare a snapshot");
        adapter->set_retention(true);
        require(!adapter->try_prepare_capture(session,tokens,producer,false,ordered),"failed target validation skips");
        if(scenario==0) {
            const auto foreign=reinterpret_cast<hipStream_t>(4);
            for(const auto unsafe:{PrefixCaptureOrder{},PrefixCaptureOrder{foreign,predictor,predictor},
                PrefixCaptureOrder{producer,predictor,foreign}}) {
                const auto before=store->counters().capture_skips_cross_stream;
                require(!adapter->try_prepare_capture(session,tokens,producer,true,unsafe) && operations.empty(),
                    "unknown/foreign target or predictor refuses before snapshot submission");
                // Adversarial foreign-tail probe: run its write before pumping the old producer.
                operations[foreign].push_back([&] {fixed=std::byte{99};target.fill(std::byte{99});return true;});
                pump(foreign);pump(producer);PrefixAsyncCapture::reap();
                require(fixed==std::byte{99} && !PrefixAsyncCapture::slot && store->counters().inserted==0 &&
                    store->counters().capture_skips_cross_stream==before+1 && waits==before_waits,
                    "foreign-stream overwrite never publishes wrong bytes or adds a wait");
                operations.clear();fixed=std::byte{31};target[0]=std::byte{11};target[1]=std::byte{12};
            }
            // A suffix owner is refused even if a caller supplies matching stream handles.
            auto suffix=config;suffix.tiled_start=false;
            adapter->set_plan(suffix,3,true);
            require(!adapter->try_prepare_capture(session,std::span(tokens).first(3),producer,true,ordered) && operations.empty(),
                "next serial/window suffix cannot authorize a snapshot");
            adapter->set_plan(config,8,true);
        }
        if(scenario==6) adapter->set_staging_cap(1);
        if(scenario==7) allocation_failure=true;
        auto d=adapter->try_prepare_capture(session,tokens,producer,true,ordered);
        allocation_failure=false;
        require(store->counters().snapshot_preparations==1,"retained preparation counted even before failure or insertion");
        if(scenario>=6) {require(!d && store->counters().inserted==0,"pre-submit cap/allocation skip");continue;}
        require(bool(d),"snapshot prepared");
        require(!adapter->try_prepare_capture(session,tokens,producer,true,ordered),"global busy slot skips without wait");
        PrefixAsyncCapture::reap();
        require(store->counters().inserted==0,"worker not yet submitted cannot publish");
        // A next block cannot run ahead of reads already queued on its stream.
        bool next_block_ran=false;
        operations[producer].push_back([&] {fixed=std::byte{99};target.fill(std::byte{99});next_block_ran=true;return true;});
        require(!next_block_ran && fixed==std::byte{31},"same-stream next block cannot execute before snapshot pump");
        pump(producer);
        require(next_block_ran,"next same-stream block executed after snapshot");
        d->submit_predictor(predictor);
        operations[predictor].push_back([&] {mtp.fill(std::byte{99});held=std::byte{99};return true;});
        require(held==std::byte{41},"same-stream predictor writer cannot run before snapshot pump");
        pump(predictor);
        if(scenario%2==0) {
            pump(d->download);PrefixAsyncCapture::reap();
            require(store->counters().inserted==0,"D2H without commit cannot publish");
        }
        d->commit(scenario<2); // failed validation/reset
        adapter.reset(); // descriptor survives prompt/registry destruction
        PrefixAsyncCapture::reap();
        if(scenario%2) require(store->counters().inserted==0,"commit without D2H cannot publish");
        if(scenario==5) {PrefixAsyncCapture::shutdown();require(waits==before_waits+1U,"shutdown explicitly drains");}
        else {pump(d->download);PrefixAsyncCapture::reap();}
        require(!PrefixAsyncCapture::slot,"completed descriptor reaped");
        require(store->counters().staging_bytes==0 && store->counters().snapshot_preparations==1,"settled capture has no live staging and keeps preparation count");
        adapter=std::make_unique<HipPrefixStateCacheSession>(store,id,2,2,ModelContract{},layout,memory);
        adapter->add_rows(reinterpret_cast<std::uintptr_t>(target.data()),1,0);
        adapter->add_rows(reinterpret_cast<std::uintptr_t>(mtp.data()),1,1);
        adapter->add_fixed_device(reinterpret_cast<std::uintptr_t>(&fixed),1);
        adapter->add_fixed_device(reinterpret_cast<std::uintptr_t>(&held),1,"mtp.held");
        adapter->set_plan(config,8,true); adapter->set_retention(false);
        require(!adapter->try_prepare_capture(session,tokens,producer,true,ordered) &&
            store->counters().snapshot_preparations==1 && store->counters().staging_bytes==0,
            "no-store after a retaining high-water mark starts no preparation");
        require(store->counters().inserted==(scenario<2?1U:0U),"only committed successful DMA published");
        require(waits==before_waits+(scenario==5?1U:0U),"snapshot request path added a wait");
        if(scenario<2) {
            const auto& c=*d->checkpoint;
            require(c.fixed.data.get()[0]==std::byte{31} && c.fixed.data.get()[256]==std::byte{41},"stable fixed and held snapshots");
            require(c.segments[0]->payload.data.get()[0]==std::byte{11} &&
                c.segments[0]->payload.data.get()[2]==std::byte{21},"stable target and predictor row snapshots");
        }
    }
    // Invoke real capture hooks between numerical units; compare every descriptor and priming flag.
    for(const auto t:{2048U,8192U}) for(const auto n:{1280U,6000U,8300U,10300U,10400U,8192U,32768U}) for(bool roomy:{false,true}) {
        operations.clear();
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{1048576,{},0},[](std::uint64_t bytes) {
            return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[bytes]{},std::default_delete<std::byte[]>()),bytes};
        });
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=n;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        HipPrefixStateCacheSession adapter(store,id,t,std::max(4096U,t),ModelContract{},layout,memory);
        std::vector<std::byte> rows(n);
        adapter.add_rows(reinterpret_cast<std::uintptr_t>(rows.data()),1,0);
        if(roomy) adapter.set_staging_cap(2U*t);
        QwenTiledPlanConfig config;config.rows=t;config.block={t,QwenPlanOutput::terminal};
        config.block.execution=1;config.block.kernel=2;config.priming=255;config.schedule=12;
        config.ladder={1024,256,64,16};
        for(const auto r:config.ladder) config.tails.push_back({r,QwenPlanOutput::terminal});
        adapter.set_plan(config,n,true);
        QwenPlanCursor off,on;std::uint64_t position=0;
        SerialSessionContext session;session.state_layout_identity=1;
        const std::vector<std::uint32_t> prompt(n,1);
        unsigned admitted=0;
        while(auto unit=qwen_prompt_next(config,on,position,n)) {
            const auto reference=qwen_prompt_next(config,off,position,n);
            require(reference && *unit==*reference,"capture hooks preserve complete unit sequence and priming flags");
            session.arithmetic_lineage=prepare_prefill_chunk_lineage(session.arithmetic_lineage,position,unit->end,1,2,1);
            position=unit->end;session.next_position=session.committed_tokens=position;
            auto d=adapter.try_prepare_capture(session,prompt,producer,true,ordered,n-1U);
            if(d) {
                if(d->publish) ++admitted;
                d->submit_predictor(predictor);pump(producer);pump(predictor);pump(d->download);
                d->commit(true);PrefixAsyncCapture::reap();
            }
            if(position==n) break;
        }
        const auto expected=t==2048U?(n==32768U?8U:n==8192U?2U:n==8300U || n==6000U?1U:n>=10300U?2U:0U):
            (n==32768U?3U:n==10300U || n==10400U?1U:0U);
        require(admitted==expected && store->counters().inserted==expected,"default and enlarged-cap capture opportunity counts");
        const auto cross=t==2048U?(n==8300U || n==10300U || n==10400U?1U:0U):(n==8300U?1U:0U);
        require(store->counters().capture_skips_cross_stream==cross,"planner refuses tail/window/serial transitions even with matching handles");
        if(!roomy) std::cout<<"capture opportunities n="<<n<<" T="<<t<<" selected="<<admitted
            <<" private="<<store->counters().row_prefetches<<" cross_stream_skips="<<cross<<'\n';
    }
    // Credit failure before submission is a skip. Stream errors after submission quarantine.
    for(unsigned scenario=0;scenario<4;++scenario) {
        operations.clear();
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{scenario==0?1U:8192U,{},0},[](std::uint64_t n) {
            return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n]{},std::default_delete<std::byte[]>()),n};
        });
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        HipPrefixStateCacheSession adapter(store,id,2,2,ModelContract{},layout,memory);
        std::array<std::byte,8> target{};
        adapter.add_rows(reinterpret_cast<std::uintptr_t>(target.data()),1,0);
        QwenTiledPlanConfig config;config.rows=2;config.block={2,QwenPlanOutput::terminal};
        config.block.execution=1;config.block.kernel=2;adapter.set_plan(config,8,true);
        SerialSessionContext session;session.state_layout_identity=1;session.next_position=session.committed_tokens=2;
        session.arithmetic_lineage=prepare_prefill_chunk_lineage(session.arithmetic_lineage,0,2,1,2,1);
        auto d=adapter.try_prepare_capture(session,tokens,producer,true,ordered);
        if(scenario==0) {require(!d,"no credit skip");continue;}
        const auto before=PrefixDmaQuarantine::count.load();
        bool fatal=false;
        try {
            if(scenario==1) {copy_error=hipErrorUnknown;d->submit_predictor(predictor);}
            else {
                d->submit_predictor(predictor);
                if(scenario==2) {drain_error=hipErrorUnknown;PrefixAsyncCapture::reap();}
                else {d->fatal();throw PrefixCaptureHipError(hipErrorUnknown,"incumbent drain failure");}
            }
        } catch(const PrefixCaptureHipError&) {fatal=true;}
        require(fatal && PrefixDmaQuarantine::count.load()==before+1,"fatal error quarantines all descriptor backing");
        require(store->counters().inserted==0,"failed copy/drain never publishes");
        PrefixAsyncCapture::slot.reset();operations.clear();drain_error=copy_error=hipSuccess;
    }
    snapshot_shim=false;
    set_prefix_batch_certification_enabled(false);
}

// Real adapter + deferred HIP shim: namespace, restore preflight, hook routing and no-wait tripwire.
void batch_snapshot_tests() {
    using namespace strix;
    using namespace strix::decode_cli;
    snapshot_shim=true;drain_error=copy_error=hipSuccess;allocation_failure=false;
    const auto id=sha256_bytes(std::as_bytes(std::span("batch-dma",9)));
    const auto producer=reinterpret_cast<hipStream_t>(2),predictor=reinterpret_cast<hipStream_t>(3),foreign=reinterpret_cast<hipStream_t>(4);
    for(unsigned scenario=0;scenario<5;++scenario) {
        operations.clear();
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{8192,{},0},[](std::uint64_t n) {
            return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n?n:1]{},std::default_delete<std::byte[]>()),n};
        });
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        std::array<std::byte,8> rows{};std::byte fixed{31};
        HipPrefixStateCacheSession adapter(store,id,2,2,ModelContract{},layout,memory);
        adapter.add_rows(reinterpret_cast<std::uintptr_t>(rows.data()),1,0);
        adapter.add_fixed_device(reinterpret_cast<std::uintptr_t>(&fixed),1);
        QwenTiledPlanConfig config;config.rows=2;config.block={2,QwenPlanOutput::terminal};
        config.block.execution=1;config.block.kernel=2;
        QwenBatchConfig batch;batch.groups=false;batch.cohort=false;
        adapter.set_plan(config,8,true);
        const auto single=adapter.active_identity();
        adapter.set_plan(config,8,true,2,batch);
        const auto identity=adapter.active_identity();
        require(single!=identity,"batch and single namespaces differ");
        auto changed=batch;changed.cancel=true;adapter.set_plan(config,8,true,2,changed);
        require(identity!=adapter.active_identity(),"scheduler settings in batch namespace");
        for(unsigned field=0;field<6;++field) {
            auto scheduler=batch;
            switch(field) {
                case 0:scheduler.service_ms=5;break;
                case 1:scheduler.lookahead++;break;
                case 2:scheduler.prompt_share=0.5;break;
                case 3:scheduler.short_rows++;break;
                case 4:scheduler.interleave=!scheduler.interleave;break;
                default:scheduler.shared="none";break;
            }
            adapter.set_plan(config,8,true,2,scheduler);
            require(adapter.active_identity()!=identity,"resolved scheduler value binds batch namespace");
        }
        adapter.set_plan(config,8,true,2,batch);
        std::uint64_t bridge_frontier=0,slot_frontier=0;
        adapter.set_restore_hook([&](std::uint64_t p){bridge_frontier=p;});
        SerialSessionContext session;session.state_layout_identity=1;
        session.next_position=session.committed_tokens=2;
        session.arithmetic_lineage=prepare_prefill_chunk_lineage(session.arithmetic_lineage,0,2,1,2,1);
        const std::vector<std::uint32_t> tokens(8,1);
        const int tag=0,other_tag=0;
        adapter.set_retention(false);const auto no_store=store->counters().no_store_skips;
        require(!adapter.try_prepare_capture(session,tokens,producer,true,{producer,predictor,predictor},7,{},{},&tag) && operations.empty(),"batch no-store prepares nothing");
        require(store->counters().no_store_skips==no_store+1,"batch no-store counted");
        adapter.set_retention(true);
        auto d=adapter.try_prepare_capture(session,tokens,producer,true,{producer,predictor,predictor},7,{},{},&tag);
        require(bool(d),"batch tagged snapshot prepared");
        const auto before_waits=waits;
        PrefixAsyncCapture::before_writer(&other_tag,foreign);
        require(!d->poisoned,"another slot does not poison the snapshot");
        if(scenario==0 || scenario==3 || scenario==4) {
            PrefixAsyncCapture::before_writer(&tag,foreign);
            // Execute a genuinely foreign overwrite before the target snapshot reads.
            operations[foreign].push_back([&]{fixed=std::byte{99};return true;});pump(foreign);
        } else if(scenario==1) PrefixAsyncCapture::before_writer(&tag,producer);
        pump(producer);
        if(scenario==2) PrefixAsyncCapture::before_writer(&tag,foreign);
        d->submit_predictor(predictor);pump(predictor);pump(d->download);
        d->commit(true);
        if(scenario==3) d->commit(false);
        if(scenario==4) {adapter.before_target_reset();d->cancelled=true;d->commit(true);}
        PrefixAsyncCapture::reap();
        const bool poisoned=scenario==0 || scenario==3 || scenario==4;
        require(!PrefixAsyncCapture::slot && store->counters().inserted==(poisoned?0U:1U),"sticky tripwire prevents publication and retires");
        require(store->counters().capture_skips_tripwire==(poisoned?1U:0U) && waits==before_waits,"tripwire is once-only and never waits");
        require(store->counters().staging_bytes==0,"tripwire retirement releases staging accounting");
        if(poisoned) {
            const auto chain=d->row_chain.lock();
            require(chain && chain->empty(),"poisoned snapshot cannot publish private row segments");
            continue;
        }
        const auto digests=prefix_cache_block_digests(identity,tokens,2,1);
        const auto checkpoint=store->find_longest(identity,tokens,2,2);
        require(checkpoint && checkpoint->producer_certificate==adapter.execution_certificate(2),"batch certificate stored");
        // Another slot has the same identity but different live storage and predictor runtime.
        std::array<std::byte,8> other_rows{};std::byte other_fixed{};
        HipPrefixStateCacheSession other(store,id,2,2,ModelContract{},layout,memory);
        other.add_rows(reinterpret_cast<std::uintptr_t>(other_rows.data()),1,0);
        other.add_fixed_device(reinterpret_cast<std::uintptr_t>(&other_fixed),1);
        other.set_plan(config,8,true,2,batch);
        unsigned invalidations=0;bool target_m1_valid=true,predictor_m1_valid=true;
        other.set_before_restore_hook([&]{++invalidations;target_m1_valid=predictor_m1_valid=false;});
        other.set_restore_hook([&](std::uint64_t p){bridge_frontier=p;});
        other.bind_restore_runtime([&](std::uint64_t p){
            require(!target_m1_valid && !predictor_m1_valid,"M1 invalidates before rebound runtime frontier restore");
            slot_frontier=p;
        });
        other.set_retention(false); // no conversation: batch no-store still permits a validated hit
        SerialSessionContext fresh;fresh.state_layout_identity=1;
        require(other.begin_request(fresh,tokens)==2 && slot_frontier==2 && bridge_frontier==0 && other_fixed==std::byte{31},"slot-0 runtime hook rebound; cross-slot reuse");
        require(other.continuation()==qwen_full_t_continuation(config,8,2),"whole-prompt cold cursor installed");
        require(invalidations==1,"M1 pre-restore hook survives first runtime rebind");
        target_m1_valid=predictor_m1_valid=true;slot_frontier=0;
        other.bind_restore_runtime([&](std::uint64_t p){
            require(!target_m1_valid && !predictor_m1_valid,"M1 invalidation survives repeated named/slot runtime rebind");
            slot_frontier=p+1;
        });
        fresh={};fresh.state_layout_identity=1;
        require(other.begin_request(fresh,tokens)==2 && invalidations==2 && slot_frontier==3 && !bridge_frontier,
            "rebound restore retains independent M1 invalidation hook");
        // Restore is a zero-write miss during parked service admissions.
        other.set_plan(config,8,true,2,batch,false);fresh={};fresh.state_layout_identity=1;other_fixed=std::byte{77};
        const auto pending_ops=operations[reinterpret_cast<hipStream_t>(1)].size();
        require(other.begin_request(fresh,tokens)==0 && fresh.next_position==0 && other_fixed==std::byte{77} &&
            operations[reinterpret_cast<hipStream_t>(1)].size()==pending_ops,"service admission miss performs no writes");
        other.set_plan(config,8,true);fresh={};fresh.state_layout_identity=1;
        require(other.begin_request(fresh,tokens)==0 && !store->find_segment(prefix_cache_block_digests(single,tokens,2,1)[0]),"single route cannot share batch segments");
        // Corrupt proof fails before any restore enqueue or mutation.
        auto corrupt=std::const_pointer_cast<PrefixCacheCheckpoint>(checkpoint);corrupt->producer_certificate[4]^=1;
        other.set_plan(config,8,true,2,batch);fresh={};fresh.state_layout_identity=1;
        require(other.begin_request(fresh,tokens)==0 && other_fixed==std::byte{77},"batch corrupt proof is a zero-write miss");
        require(digests[0]==checkpoint->digest,"cross-slot token namespace stable");
    }
    operations.clear();snapshot_shim=false;
    set_prefix_batch_certification_enabled(false);
}
