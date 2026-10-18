#include "strix/host_phase_timing.hpp"
#include "strix/hip_mtp_runtime.hpp"
#include "strix/hip_mtp_attention.hpp"
#include "strix/hip_mtp_projection.hpp"
#include "strix/hip_mtp_native_validation.hpp"
#include "strix/hip_mtp_shared_expert.hpp"
#include "strix/hip_mtp_fast_draft.hpp"
#include "strix/hip_embedding_launcher.hpp"
#include "strix/hip_moe_route_launcher.hpp"
#include "strix/hip_shared_expert_launcher.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/mtp_sidecar_binding.hpp"
#include <cstring>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <array>
#include <limits>
#include <optional>
#include <stdexcept>

namespace strix {
namespace {
constexpr std::array<MtpDraftPlanStep,11> native_steps{MtpDraftPlanStep::input_fusion,MtpDraftPlanStep::attention_hyper_mix,
    MtpDraftPlanStep::qsa_indexer,MtpDraftPlanStep::sparse_attention,MtpDraftPlanStep::attention_combine,
    MtpDraftPlanStep::mlp_hyper_mix,MtpDraftPlanStep::routed_and_shared_moe,MtpDraftPlanStep::mlp_combine,
    MtpDraftPlanStep::publish_next_state,MtpDraftPlanStep::output_head_mix,MtpDraftPlanStep::lm_head};
void require(bool ok,const char* message) {if(!ok) throw std::invalid_argument(message);}
void check(hipError_t e,const char* message) {if(e!=hipSuccess) throw std::runtime_error(message);}
bool same_session(const SerialSessionContext& a,const SerialSessionContext& b) {
    return a.execution_identity==b.execution_identity && a.runtime_compatibility_identity==b.runtime_compatibility_identity &&
        a.state_layout_identity==b.state_layout_identity && a.next_position==b.next_position && a.committed_tokens==b.committed_tokens &&
        a.health==SerialSessionHealth::ready && b.health==SerialSessionHealth::ready && a.state.valid==b.state.valid &&
        a.state.generations==b.state.generations && arithmetic_lineage_matches(a.arithmetic_lineage,b.arithmetic_lineage,a.next_position);
}
bool same_input(const MtpDraftCycleInput& a,const MtpDraftCycleInput& b) {
    return a.proposal_identity==b.proposal_identity && a.activation_identity==b.activation_identity &&
        a.target_runtime_compatibility_identity==b.target_runtime_compatibility_identity && a.seed_token_id==b.seed_token_id &&
        a.seed_position==b.seed_position && a.hidden_source_position==b.hidden_source_position && a.hidden_address==b.hidden_address &&
        a.hidden_elements==b.hidden_elements && a.hidden_element_bytes==b.hidden_element_bytes &&
        a.hidden_allocation_generation==b.hidden_allocation_generation;
}
void require_chain(const HipMtpTargetHiddenTicket& a,const HipMtpTargetHiddenTicket& b) {
    require(a.valid() && b.valid() && a.same_producer(b) && a.source_position()!=UINT64_MAX &&
        b.source_position()==a.source_position()+1U,"MTP target tickets are not one actual consecutive producer");
    const auto& previous=a.committed_session();const auto& next=b.committed_session();
    if(a.same_commit(b)) {
        // Both snapshots belong to the same real whole-window commit. There is
        // deliberately no invented SessionContext for an intermediate row.
        require(a.is_window() && b.is_window() && a.chunk_begin()==b.chunk_begin() &&
            a.chunk_end()==b.chunk_end() && same_session(previous,next),
            "MTP adjacent rows do not retain the same actual whole-window commit");
        return;
    }
    // Consecutive rows keep one executor, except across the producer's admitted prompt handoff
    // (split prefill profile), where the lineage below still proves the exact continuation.
    const bool same_executor=previous.execution_identity==next.execution_identity &&
        previous.runtime_compatibility_identity==next.runtime_compatibility_identity;
    require(a.source_position()+1U==a.chunk_end() && b.source_position()==b.chunk_begin() &&
        (same_executor || a.admits_executor_change(b)) &&
        previous.state_layout_identity==next.state_layout_identity && previous.next_position==b.chunk_begin() &&
        b.chunk_end()>b.chunk_begin() && next.next_position==b.chunk_end() &&
        b.chunk_end()-b.chunk_begin()<=UINT64_MAX-previous.committed_tokens &&
        next.committed_tokens==previous.committed_tokens+(b.chunk_end()-b.chunk_begin()) &&
        !next.arithmetic_lineage.intervals.empty(),"MTP target ticket chunk frontier changed");
    const auto& actual=next.arithmetic_lineage.intervals.back();
    ArithmeticLineage history;
    if(b.is_window()) {
        require(actual.producer==ArithmeticProducer::known_window_chunk && actual.begin==b.chunk_begin() &&
            actual.end==b.chunk_end(),"MTP target window lacks its actual whole-chunk producer receipt");
        history=prepare_known_window_chunk_lineage(previous.arithmetic_lineage,b.chunk_begin(),b.chunk_end(),
            actual.numerical_plan_identity,actual.reduction_plan_identity,actual.state_update_identity);
    } else {
        require(b.chunk_end()==b.chunk_begin()+1U && actual.producer==ArithmeticProducer::serial_row,
            "MTP target serial ticket is not one actual serial row");
        const auto producer=b.producer_execution_identity();
        require(producer==next.execution_identity || (!same_executor && a.admits_executor_change(b)),
            "MTP target serial ticket producer differs from its session executor");
        history=prepare_serial_row_lineage(previous.arithmetic_lineage,b.source_position(),producer,
            actual.reduction_plan_identity);
    }
    require(arithmetic_lineage_matches(history,next.arithmetic_lineage,next.next_position),
        "MTP target ticket arithmetic history is not the actual producer continuation");
}
}
struct HipMtpRuntime::Impl {
    std::shared_ptr<HipMtpResources> owner;
    std::unique_ptr<HipGreedyOutput> selector;
    std::unique_ptr<HipMtpDraftHead> draft;
    std::unique_ptr<float,void(*)(float*)> fast_workspace{nullptr,[](float* p){(void)hipFree(p);}};
    // F32 copies of the attention/MLP HC injection matrices for the split-K draft mix.
    std::unique_ptr<float,void(*)(float*)> hc_inject{nullptr,[](float* p){(void)hipFree(p);}};
    bool draft_exact=false;
    std::optional<HipMtpResources::NativeValidationScope> validation;
    std::optional<HipMtpTargetHiddenTicket> last_target,cycle_base,base_previous_target,reconcile_final;
    std::optional<std::uint32_t> last_teacher_next_input,base_previous_input;
    std::vector<HipMtpReplayRow> replay;
    std::vector<std::uint32_t> candidates;
    std::vector<float> probabilities;
    // Sampled drafts (null: greedy drafts): one distribution per proposed draft of the last chain.
    std::shared_ptr<TokenSampler> sampler;
    std::vector<TokenDraft> sampled;
    std::vector<float> draft_logits;
    MtpDraftCycleInput current;
    HipMtpExecutionStats stats;
    std::uint64_t next_source=0,base_source=0,epoch=0,profile=0;
    // Rows [0, ticketless_frontier) were primed without target tickets (tiled prompt or restore):
    // the first ticketed teacher row may start there instead of at source zero.
    std::uint64_t ticketless_frontier=0;
    std::uint32_t step=0,pending_token=0;
    bool active=false,cycle=false,have_replay=false,broken=false,cache_only_active=false,first_cache_reusable=false;
    explicit Impl(std::shared_ptr<HipMtpResources> actual):owner(std::move(actual)) {
        require(owner && !owner.owner_before(owner->weak_from_this().lock()) &&
            !owner->weak_from_this().lock().owner_before(owner),"MTP runtime requires its actual retained native owner");
        owner->require_idle();
        require(owner->plan().steps.size()==native_steps.size() &&
            std::equal(native_steps.begin(),native_steps.end(),owner->plan().steps.begin()),"MTP native step order is unsupported");
        candidates.reserve(owner->budget().maximum_candidate_depth);replay.reserve(owner->budget().maximum_candidate_depth+1U);
        const auto pool=owner->view(HipMtpBuffer::wide_current);
        selector=std::make_unique<HipGreedyOutput>(static_cast<std::uint32_t>(owner->model().vocab_size),
            ExecutionMemoryBinding{owner->physical_identity(),pool.address,owner->reservation().device_pool_bytes,pool.allocation_generation},owner,owner->guard());
        const auto input=fusion();profile=hip_mtp_input_fusion_profile_identity(input);
        for(const auto word:{owner->plan().forward_semantics.contract_identity,owner->plan().plan_identity,
            hip_mtp_lm_head_profile_identity(owner->model(),owner->tensors().lm_head),
            std::uint64_t{0x4d54504341535431ULL}, // durable cache BF16-cast contract v1
            std::uint64_t{0x4d545052415731ULL}})
            for(unsigned shift=0;shift<64;shift+=8) profile=(profile^((word>>shift)&255U))*1099511628211ULL;
        if(!profile) profile=1U;
    }
    HipMtpInputFusionRequest fusion() const {
        const auto device=[](MtpDeviceView v) {return HipMtpInputFusionView{v.address,v.bytes,v.allocation_generation};};
        const auto tensor=[](MtpTensorView v) {return HipMtpInputFusionView{v.runtime->address,v.runtime->bytes,v.runtime->allocation_generation};};
        HipMtpInputFusionRequest r;const auto& m=owner->model();const auto& t=owner->tensors();
        r.hidden_size=static_cast<std::uint32_t>(m.hidden_size);r.hc_count=static_cast<std::uint32_t>(m.hc_count);r.epsilon=m.rms_norm_epsilon;
        r.embedding_storage=r.hidden_storage=r.output_storage=HipMtpInputStorage::bf16;
        r.normalized_rounding=owner->budget().normalized_rounding;r.projection_rounding=owner->budget().projection_rounding;
        r.norm_weight_encoding=owner->shared_target().source.norm_weight_encoding;r.norm_encoding_admitted=true;
        r.arithmetic=owner->plan().forward_semantics.input_arithmetic;
        r.embedding=device(owner->view(HipMtpBuffer::embedding));r.wide_hidden=device(owner->view(HipMtpBuffer::wide_current));
        r.embedding_norm=tensor(t.enorm);r.hidden_norm=tensor(t.hnorm);r.packed_projection=tensor(t.eh_projection);
        r.projection_row_stride_bytes=(m.hidden_size*2U/32U)*34U;
        r.normalized_scratch=device(owner->view(HipMtpBuffer::normalized_input));r.projection_scratch=device(owner->view(HipMtpBuffer::projected_input));
        r.output=device(owner->view(HipMtpBuffer::fused_wide));return r;
    }
    MtpNativeContext context(std::uint64_t position) const {return {&owner->model(),&owner->plan(),&owner->tensors(),owner.get(),position};}
    void preflight(std::uint64_t position,std::uint32_t token,bool already_admitted=false,bool cache_only=false) const {
        require(!broken && !active && !current_hip_launch_context(),"MTP native input requires one healthy idle host protocol");
        std::optional<HipMtpResources::NativeValidationScope> local_validation;
        if(!already_admitted && !validation) local_validation.emplace(owner->admit_native_entry());
        const auto c=context(position);mtp_native::context(c);
        hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
        check(hipStreamIsCapturing(static_cast<hipStream_t>(owner->stream()),&capture),"MTP capture query failed");
        require(capture==hipStreamCaptureStatusNone,"MTP native runtime is eager only");
        preflight_hip_mtp_embedding(c,token);preflight_hip_mtp_input_fusion(fusion());
        preflight_hip_mtp_hyper_mix(c,MtpHyperRole::attention);
        if(cache_only){preflight_hip_mtp_cache_rows(c);return;}
        preflight_hip_mtp_attention(c);
        preflight_hip_mtp_inject(c,MtpInjectRole::attention);preflight_hip_mtp_hyper_mix(c,MtpHyperRole::mlp);
        preflight_hip_mtp_route(c);preflight_hip_mtp_shared_expert(c);preflight_hip_mtp_routed_experts(c);
        preflight_hip_mtp_moe_combine(c);preflight_hip_mtp_inject(c,MtpInjectRole::mlp);
        preflight_hip_mtp_hyper_mix(c,MtpHyperRole::head);preflight_hip_mtp_lm_head(c);
    }
    void validate_ticket(const HipMtpTargetHiddenTicket& ticket) const {
        require(ticket.valid() && ticket.target_content_identity()==owner->shared_target().target_model_content_identity,
            "MTP target hidden is not an actual compatible committed ticket");
        const auto v=ticket.view();const auto& session=ticket.committed_session();
        require(v.bytes==owner->view(HipMtpBuffer::wide_current).bytes && v.address && v.allocation_generation &&
            ticket.source_position()<owner->context_capacity() && ticket.source_position()!=UINT64_MAX &&
            ticket.chunk_begin()<=ticket.source_position() && ticket.source_position()<ticket.chunk_end() &&
            session.next_position==ticket.chunk_end() && session.health==SerialSessionHealth::ready &&
            arithmetic_lineage_valid(session.arithmetic_lineage,session.next_position),"MTP target hidden width/frontier/history is invalid");
        const auto& actual=session.arithmetic_lineage.intervals.back();
        if(ticket.is_window()) {
            require(actual.producer==ArithmeticProducer::known_window_chunk && actual.begin==ticket.chunk_begin() &&
                actual.end==ticket.chunk_end(),"MTP target window ticket lacks its actual chunk history");
        } else {
            require(ticket.chunk_begin()==ticket.source_position() && ticket.chunk_end()==ticket.source_position()+1U &&
                actual.producer==ArithmeticProducer::serial_row,"MTP target serial ticket frontier differs from its actual row");
        }
    }
    void copy(MtpDeviceView destination,MtpDeviceView source) {
        require(destination.bytes==source.bytes,"MTP copy extents differ");
        check(hipMemcpyAsync(reinterpret_cast<void*>(destination.address),reinterpret_cast<const void*>(source.address),
            static_cast<std::size_t>(source.bytes),hipMemcpyDeviceToDevice,static_cast<hipStream_t>(owner->stream())),"MTP owned copy enqueue failed");
    }
    void failure() noexcept {broken=true;active=false;cache_only_active=false;first_cache_reusable=false;owner->fail();validation.reset();}
    void stage(const MtpForwardStepContext& call) {
        const auto& input=*call.cycle_input;
        require(active && !cache_only_active && step<native_steps.size() && call.step_index==step && call.step==native_steps[step] &&
            call.input_token==current.seed_token_id && call.target_position==current.seed_position &&
            call.predictor_position==current.hidden_source_position && call.plan_identity==owner->plan().plan_identity &&
            call.proposal_identity==owner->plan().proposal_identity && same_input(input,current),"MTP native dispatch order/input changed");
        const auto c=context(current.hidden_source_position);void* stream=owner->stream();
        switch(call.step) {
        case MtpDraftPlanStep::input_fusion:launch_hip_mtp_embedding(c,current.seed_token_id,stream);launch_hip_mtp_input_fusion(fusion(),stream);break;
        case MtpDraftPlanStep::attention_hyper_mix:launch_hip_mtp_hyper_mix(c,MtpHyperRole::attention,stream);break;
        case MtpDraftPlanStep::qsa_indexer:launch_hip_mtp_indexer(c,stream);break;
        case MtpDraftPlanStep::sparse_attention:launch_hip_mtp_sparse_attention(c,stream);break;
        case MtpDraftPlanStep::attention_combine:launch_hip_mtp_inject(c,MtpInjectRole::attention,stream);break;
        case MtpDraftPlanStep::mlp_hyper_mix:launch_hip_mtp_hyper_mix(c,MtpHyperRole::mlp,stream);break;
        case MtpDraftPlanStep::routed_and_shared_moe:
            launch_hip_mtp_route(c,stream);launch_hip_mtp_shared_expert(c,stream);launch_hip_mtp_routed_experts(c,stream);
            launch_hip_mtp_moe_combine(c,stream);break;
        case MtpDraftPlanStep::mlp_combine:launch_hip_mtp_inject(c,MtpInjectRole::mlp,stream);break;
        case MtpDraftPlanStep::publish_next_state:copy(owner->view(HipMtpBuffer::wide_current),owner->view(HipMtpBuffer::wide_next));break;
        case MtpDraftPlanStep::output_head_mix:launch_hip_mtp_hyper_mix(c,MtpHyperRole::head,stream);break;
        case MtpDraftPlanStep::lm_head:launch_hip_mtp_lm_head(c,stream);break;
        }
        ++step;
    }
    void start(MtpDraftCycleInput input,const HipMtpTargetHiddenTicket* source,bool cache_only=false) {
        require(input.hidden_source_position==next_source && input.seed_position==next_source+1U &&
            input.seed_token_id<owner->model().vocab_size,"MTP source position/seed is not the next native row");
        require(!active && !validation,"MTP input cannot replace an active native admission");
        validation.emplace(owner->admit_native_entry());
        try {
            preflight(next_source,input.seed_token_id,false,cache_only);
            require(cache_only?stats.cache_only_rows!=UINT64_MAX:stats.full_forward_rows!=UINT64_MAX,
                "MTP completed row counter exhausted");
            if(source) validate_ticket(*source);
            require(epoch!=UINT64_MAX,"MTP native epoch exhausted");
            owner->mark_submitted();
        } catch(...) {validation.reset();throw;}
        ++epoch;
        active=true;cache_only_active=cache_only;step=0;current=input;
        try {if(source) copy(owner->view(HipMtpBuffer::wide_current),source->view());} catch(...) {failure();throw;}
    }
    std::uint32_t finish() {
        require(active && !cache_only_active && step==native_steps.size(),"MTP native output requested before all eleven steps");
        try {
            owner->complete();const auto logits=owner->view(HipMtpBuffer::logits);
            const auto token=selector->select(ExecutionBufferAddress{{},logits.address,logits.bytes,logits.allocation_generation},
                static_cast<std::uint32_t>(owner->model().vocab_size),current.hidden_source_position,epoch);
            active=false;++next_source;++stats.full_forward_rows;validation.reset();return token;
        } catch(...) {failure();throw;}
    }
    std::uint32_t run_current() {
        for(std::size_t i=0;i<native_steps.size();++i) {
            MtpForwardStepContext call;call.step=native_steps[i];call.step_index=static_cast<std::uint32_t>(i);
            call.input_token=current.seed_token_id;call.target_position=current.seed_position;call.predictor_position=current.hidden_source_position;
            call.proposal_identity=current.proposal_identity;call.plan_identity=owner->plan().plan_identity;call.cycle_input=&current;stage(call);
        }
        return finish();
    }
    HipMtpNativeRowResult teacher(const HipMtpTargetHiddenTicket& ticket,std::uint32_t token) {
        validate_ticket(ticket);require(ticket.source_position()==next_source,"MTP teacher row is not contiguous");
        if(last_target) {
            require_chain(*last_target,ticket);
            require(last_teacher_next_input && *last_teacher_next_input==ticket.input_token(),"MTP prior teacher pair differs from the actual next target input");
        } else require(next_source==ticketless_frontier,"MTP initial history must start at source zero");
        const auto v=owner->view(HipMtpBuffer::wide_current);
        MtpDraftCycleInput input{owner->plan().proposal_identity,0U,ticket.committed_session().runtime_compatibility_identity,
            token,next_source+1U,next_source,v.address,owner->plan().state.state_width,2U,v.allocation_generation};
        start(input,&ticket);
        try {
            const auto selected=run_current();last_target=ticket;last_teacher_next_input=token;
            return {input.hidden_source_position,input.seed_position,profile,selected,static_cast<std::uint32_t>(native_steps.size()),
                owner->view(HipMtpBuffer::wide_current),owner->view(HipMtpBuffer::logits),true};
        } catch(...) {failure();throw;}
    }
    void teacher_cache(const HipMtpTargetHiddenTicket& ticket,std::uint32_t token) {
        validate_ticket(ticket);require(ticket.source_position()==next_source,"MTP teacher cache row is not contiguous");
        if(last_target) {
            require_chain(*last_target,ticket);
            require(last_teacher_next_input && *last_teacher_next_input==ticket.input_token(),
                "MTP prior cache teacher pair differs from the actual next target input");
        } else require(next_source==ticketless_frontier,"MTP initial cache history must start at source zero");
        const auto v=owner->view(HipMtpBuffer::wide_current);
        MtpDraftCycleInput input{owner->plan().proposal_identity,0U,ticket.committed_session().runtime_compatibility_identity,
            token,next_source+1U,next_source,v.address,owner->plan().state.state_width,2U,v.allocation_generation};
        // Copy the authoritative wide source on every teacher row. No previous
        // predictor-wide result, logits or scratch payload is a cache input.
        start(input,&ticket,true);
        try {
            const auto c=context(current.hidden_source_position);void* stream=owner->stream();
            launch_hip_mtp_embedding(c,current.seed_token_id,stream);
            cache_row_front(c,stream);
            launch_hip_mtp_cache_rows(c,stream);
            owner->complete();
            active=false;cache_only_active=false;++next_source;++stats.cache_only_rows;validation.reset();
            last_target=ticket;last_teacher_next_input=token;
        } catch(...) {failure();throw;}
    }
    void reuse_first_teacher_cache(const HipMtpReplayRow& row) {
        // The only caller is reconciliation after its full metadata admission.
        // The proof is private and minted after the first full proposal row's
        // native completion and finite selector, never by predicted-token match.
        require(first_cache_reusable && cycle_base && !broken && !active && !validation &&
            row.hidden.same_commit(*cycle_base) && row.hidden.source_position()==base_source &&
            row.hidden.view()==cycle_base->view() && row.next_input_token==pending_token && next_source==base_source,
            "MTP first cache reuse lacks its actual completed proposal source proof");
        validate_ticket(row.hidden);
        if(last_target) {
            require_chain(*last_target,row.hidden);
            require(last_teacher_next_input && *last_teacher_next_input==row.hidden.input_token(),
                "MTP reused cache teacher pair differs from the actual preceding input");
        } else require(next_source==ticketless_frontier,"MTP reused initial cache history must start at source zero");
        owner->require_idle();
        require(stats.cache_reused_rows!=UINT64_MAX,"MTP reused cache row counter exhausted");
        ++next_source;++stats.cache_reused_rows;
        last_target=row.hidden;last_teacher_next_input=row.next_input_token;
    }
    // Batched teacher cache rows: same per-row launches as teacher_cache, one
    // admission and one completion for the whole contiguous run.
    void teacher_cache_rows(std::span<const HipMtpReplayRow> rows) {
        if(rows.empty()) return;
        require(!active && !validation && stats.cache_only_rows<=UINT64_MAX-rows.size(),"MTP batched cache rows require an idle owner");
        validation.emplace(owner->admit_native_entry());
        try {
            for(std::size_t i=0;i<rows.size();++i) {
                const auto& row=rows[i];validate_ticket(row.hidden);
                require(row.hidden.source_position()==next_source+i,"MTP teacher cache row is not contiguous");
                if(i) {
                    require_chain(rows[i-1U].hidden,row.hidden);
                    require(rows[i-1U].next_input_token==row.hidden.input_token(),"MTP prior cache teacher pair differs from the actual next target input");
                } else if(last_target) {
                    require_chain(*last_target,row.hidden);
                    require(last_teacher_next_input && *last_teacher_next_input==row.hidden.input_token(),
                        "MTP prior cache teacher pair differs from the actual next target input");
                } else require(next_source==ticketless_frontier,"MTP initial cache history must start at source zero");
                preflight(row.hidden.source_position(),row.next_input_token,true,true);
            }
            owner->mark_submitted();
        } catch(...) {validation.reset();throw;}
        active=true;cache_only_active=true;
        try {
            void* stream=owner->stream();
            for(const auto& row:rows) {
                const auto c=context(row.hidden.source_position());
                copy(owner->view(HipMtpBuffer::wide_current),row.hidden.view());
                launch_hip_mtp_embedding(c,row.next_input_token,stream);cache_row_front(c,stream);
                launch_hip_mtp_cache_rows(c,stream);
            }
            owner->complete();
            active=false;cache_only_active=false;next_source+=rows.size();stats.cache_only_rows+=rows.size();validation.reset();
            last_target=rows.back().hidden;last_teacher_next_input=rows.back().next_input_token;
        } catch(...) {failure();throw;}
    }
    // Sync-free proposal chain: identical stage launches to run_current per row,
    // except the head/selector and the device-fed next input token.
    std::vector<std::uint32_t> chain(std::uint32_t depth,float min_probability,
        const std::function<bool(std::span<const float>)>& keep_drafting) {
        require(draft && cycle && cycle_base && !active && !validation && candidates.empty() &&
            depth<=owner->budget().maximum_candidate_depth,"MTP draft chain requires a fresh bounded proposal");
        sampled.clear();
        if(!depth) return {};
        require(stats.full_forward_rows<=UINT64_MAX-depth && epoch!=UINT64_MAX,"MTP draft chain counters exhausted");
        validation.emplace(owner->admit_native_entry());
        try {
            for(std::uint32_t k=0;k<depth;++k) preflight(next_source+k,k?0U:pending_token,true);
            validate_ticket(*cycle_base);owner->mark_submitted();
        } catch(...) {validation.reset();throw;}
        ++epoch;active=true;
        try {
            void* stream=owner->stream();
            const auto* hidden=reinterpret_cast<const std::uint16_t*>(owner->view(HipMtpBuffer::hidden0).address);
            copy(owner->view(HipMtpBuffer::wide_current),cycle_base->view());
            draft->begin_chain();
            const auto requested=depth;
            for(std::uint32_t k=0;k<requested;++k) {
                const auto c=context(next_source+k);
                if(k) launch_hip_mtp_embedding_device(c,draft->device_token(k-1U),stream);
                else launch_hip_mtp_embedding(c,pending_token,stream);
                if(draft_exact) launch_hip_mtp_input_fusion(fusion(),stream);
                else launch_hip_mtp_input_fusion_draft(fusion(),stream);
                hyper(c,MtpHyperRole::attention,stream);
                launch_hip_mtp_indexer(c,stream);
                if(draft_exact) launch_hip_mtp_sparse_attention(c,stream); else launch_hip_mtp_sparse_attention_draft(c,stream);
                launch_hip_mtp_inject(c,MtpInjectRole::attention,stream);hyper(c,MtpHyperRole::mlp,stream);
                launch_hip_mtp_route(c,stream);
                if(draft_exact) {
                    launch_hip_mtp_shared_expert(c,stream);launch_hip_mtp_routed_experts(c,stream);launch_hip_mtp_moe_combine(c,stream);
                } else launch_hip_mtp_fast_moe(fast_moe(),stream);
                launch_hip_mtp_inject(c,MtpInjectRole::mlp,stream);
                copy(owner->view(HipMtpBuffer::wide_current),owner->view(HipMtpBuffer::wide_next));
                hyper(c,MtpHyperRole::head,stream);
                draft->launch(hidden,k,stream);
                if(sampler) {
                    // Draft k proposes the token at next_source + k + 2 (row k's seed sits at next_source + k + 1).
                    draft->read_logits(draft_logits);
                    sampled.push_back(sampler->draft(draft_logits,draft->host_ids(),next_source+k+2U));
                    draft->set_token(k,sampled.back().token,stream);
                }
                if(min_probability>0.0F && k+1U<requested) {
                    if(!sampler) draft->wait_step(k,stream);
                    if(draft->probability(k)<min_probability) {depth=k+1U;break;}
                    if(keep_drafting) {
                        probabilities.resize(k+1U);for(std::uint32_t j=0;j<=k;++j) probabilities[j]=draft->probability(j);
                        if(!keep_drafting(probabilities)) {depth=k+1U;break;}
                    }
                }
            }
            owner->complete();
            auto tokens=draft->read(depth);
            probabilities.resize(depth);for(std::uint32_t k=0;k<depth;++k) probabilities[k]=draft->probability(k);
            // STRIX_MTP_DRAFT_LOG=1: every proposal (chain start position, step, token, probability bits) on stderr.
            static const bool log_drafts=std::getenv("STRIX_MTP_DRAFT_LOG")!=nullptr;
            if(log_drafts) for(std::uint32_t k=0;k<depth;++k) {
                std::uint32_t bits=0;std::memcpy(&bits,&probabilities[k],sizeof(bits));
                std::fprintf(stderr,"draft at=%llu k=%u token=%u p=%08x\n",static_cast<unsigned long long>(next_source),k,tokens[k],bits);
            }
            active=false;next_source+=depth;stats.full_forward_rows+=depth;validation.reset();
            // Row zero is the pending cycle input at base_source, as in read_greedy_token.
            first_cache_reusable=true;candidates=tokens;return tokens;
        } catch(...) {failure();throw;}
    }
    // Teacher cache rows only feed the predictor's own attention cache, so a
    // draft-only chain builds them with the same fast kernels as its proposals.
    void cache_row_front(const MtpNativeContext& c,void* stream) const {
        if(draft && !draft_exact) {launch_hip_mtp_input_fusion_draft(fusion(),stream);hyper(c,MtpHyperRole::attention,stream);return;}
        launch_hip_mtp_input_fusion(fusion(),stream);launch_hip_mtp_hyper_mix(c,MtpHyperRole::attention,stream);
    }
    void hyper(const MtpNativeContext& c,MtpHyperRole role,void* stream) const {
        if(draft_exact) {launch_hip_mtp_hyper_mix(c,role,stream);return;}
        constexpr std::uint64_t inject=4U*10240U;
        launch_hip_mtp_hyper_mix_draft(c,role,role==MtpHyperRole::attention?hc_inject.get():
            role==MtpHyperRole::mlp?hc_inject.get()+inject:nullptr,stream);
    }
    HipMtpFastMoeViews fast_moe() const {
        const auto& t=owner->tensors();const auto address=[](MtpTensorView v){return reinterpret_cast<const void*>(v.runtime->address);};
        HipMtpFastMoeViews v;
        v.input=reinterpret_cast<const std::uint16_t*>(owner->view(HipMtpBuffer::hidden0).address);
        v.routes=reinterpret_cast<const std::uint32_t*>(owner->view(HipMtpBuffer::routes).address);
        v.route_weights=reinterpret_cast<const float*>(owner->view(HipMtpBuffer::route_weights).address);
        v.gate=address(t.expert_gate);v.up=address(t.expert_up);v.down=address(t.expert_down);
        v.shared_gate=address(t.shared_gate);v.shared_up=address(t.shared_up);v.shared_down=address(t.shared_down);
        v.shared_scalar=static_cast<const float*>(address(t.shared_scalar_gate));v.workspace=fast_workspace.get();
        v.output=reinterpret_cast<std::uint16_t*>(owner->view(HipMtpBuffer::hidden0).address);
        v.status=reinterpret_cast<std::uint32_t*>(owner->view(HipMtpBuffer::status).address);
        return v;
    }
    void qualification(const QualifiedMtpDraftPlan& plan,const ActivatedMtpDraftPlan& activated) const {
        require_qualified_mtp_draft_plan(plan);
        require(plan.base.plan_identity==owner->plan().plan_identity && plan.base.proposal_identity==owner->plan().proposal_identity &&
            plan.native_arithmetic_profile_identity==profile && activated.native_arithmetic_profile_identity==profile &&
            activated.semantics_qualified && activated.structural_plan_identity==plan.base.plan_identity &&
            activated.proposal_identity==plan.base.proposal_identity && activated.activation_identity,
            "MTP native qualified entry does not match its actual source/profile plan");
    }
};
HipMtpRuntime::HipMtpRuntime(std::shared_ptr<HipMtpResources> owner):impl_(std::make_unique<Impl>(std::move(owner))) {}
HipMtpRuntime::~HipMtpRuntime()=default;
const HipMtpResources& HipMtpRuntime::resources() const noexcept {return *impl_->owner;}
std::uint64_t HipMtpRuntime::next_source_position() const noexcept {return impl_->next_source;}
bool HipMtpRuntime::failed() const noexcept {return impl_->broken || impl_->owner->failed();}
void HipMtpRuntime::cancel() noexcept {impl_->failure();}
std::uint64_t HipMtpRuntime::numerical_profile_identity() const noexcept {return impl_->profile;}
HipMtpExecutionStats HipMtpRuntime::execution_stats() const noexcept {return impl_->stats;}
HipMtpCacheStateView HipMtpRuntime::cache_state() const {
    const auto& p=*impl_;
    require(!p.broken && !p.active && !p.cycle && !p.validation,
        "MTP durable cache view requires an idle completed producer outside any proposal");
    p.owner->require_idle();
    const auto kv_stride=mtp_native::product(mtp_native::product(p.owner->model().kv_heads,p.owner->model().head_dim),4U);
    const auto index_stride=mtp_native::product(p.owner->model().indexer_head_dim,2U);
    auto kv=p.owner->view(HipMtpBuffer::kv),index=p.owner->view(HipMtpBuffer::raw_index);
    const auto kv_bytes=mtp_native::product(p.next_source,kv_stride),index_bytes=mtp_native::product(p.next_source,index_stride);
    require(p.next_source<=p.owner->context_capacity() && kv_bytes<=kv.bytes && index_bytes<=index.bytes,
        "MTP durable cache prefix exceeds actual retained backing");
    kv.bytes=kv_bytes;index.bytes=index_bytes;
    return {kv,index,p.next_source,kv_stride,index_stride};
}
HipMtpNativeRowResult HipMtpRuntime::execute_diagnostic(const HipMtpTargetHiddenTicket& ticket,std::uint32_t token) {
    require(!impl_->cycle,"MTP diagnostic cannot interleave a live production proposal");return impl_->teacher(ticket,token);
}
void HipMtpRuntime::prime_committed_rows(const QualifiedMtpDraftPlan& plan,const ActivatedMtpDraftPlan& activated,std::span<const HipMtpReplayRow> rows) {
    auto& p=*impl_;p.qualification(plan,activated);require(!p.cycle && !p.active,"MTP priming cannot interleave a proposal");
    {
    const auto admission=p.owner->admit_native_entry();
    for(std::size_t i=0;i<rows.size();++i) {
        p.validate_ticket(rows[i].hidden);require(rows[i].hidden.source_position()==p.next_source+i &&
            rows[i].next_input_token<p.owner->model().vocab_size,"MTP priming rows are invalid");
        if(i) {require_chain(rows[i-1U].hidden,rows[i].hidden);require(rows[i-1U].next_input_token==rows[i].hidden.input_token(),"MTP teacher priming token pair changed");}
        p.preflight(rows[i].hidden.source_position(),rows[i].next_input_token,true,true);
    }
    }
    if(p.draft) p.teacher_cache_rows(rows);
    else for(const auto& row:rows) p.teacher_cache(row.hidden,row.next_input_token);
}
void HipMtpRuntime::prime_hidden_rows(const QualifiedMtpDraftPlan& plan,const ActivatedMtpDraftPlan& activated,
    std::span<const MtpDeviceView> hidden,std::span<const std::uint32_t> next_tokens) {
    auto& p=*impl_;p.qualification(plan,activated);
    require(!p.cycle && !p.active && !p.validation && !p.broken && hidden.size()==next_tokens.size() &&
        p.stats.cache_only_rows<=UINT64_MAX-hidden.size(),"MTP hidden-row priming requires an idle owner and one token per row");
    if(hidden.empty()) return;
    const auto wide=p.owner->view(HipMtpBuffer::wide_current);
    p.validation.emplace(p.owner->admit_native_entry());
    try {
        for(std::size_t i=0;i<hidden.size();++i) {
            require(hidden[i].address && hidden[i].bytes==wide.bytes && next_tokens[i]<p.owner->model().vocab_size &&
                p.next_source+i+1U<p.owner->context_capacity(),"MTP hidden priming row is invalid");
            p.preflight(p.next_source+i,next_tokens[i],true,true);
        }
        p.owner->mark_submitted();
    } catch(...) {p.validation.reset();throw;}
    p.active=true;p.cache_only_active=true;
    try {
        void* stream=p.owner->stream();
        for(std::size_t i=0;i<hidden.size();++i) {
            const auto c=p.context(p.next_source+i);
            p.copy(wide,hidden[i]);
            launch_hip_mtp_embedding(c,next_tokens[i],stream);p.cache_row_front(c,stream);
            launch_hip_mtp_cache_rows(c,stream);
        }
        p.owner->complete();
        p.active=false;p.cache_only_active=false;p.next_source+=hidden.size();p.stats.cache_only_rows+=hidden.size();p.validation.reset();
        p.last_target.reset();p.last_teacher_next_input.reset();p.first_cache_reusable=false;p.ticketless_frontier=p.next_source;
    } catch(...) {p.failure();throw;}
}
void HipMtpRuntime::restore_cache_frontier(const QualifiedMtpDraftPlan& plan,const ActivatedMtpDraftPlan& activated,std::uint64_t source) {
    auto& p=*impl_;p.qualification(plan,activated);
    require(!p.cycle && !p.active && !p.validation && !p.broken && p.next_source==0U && !p.last_target &&
        source<p.owner->context_capacity(),"MTP cache frontier restore requires a fresh idle owner within capacity");
    p.owner->require_idle();
    p.next_source=source;p.ticketless_frontier=source;p.first_cache_reusable=false;
}
void HipMtpRuntime::begin_cycle(std::uint32_t token,const HipMtpTargetHiddenTicket& ticket) {
    auto& p=*impl_;p.validate_ticket(ticket);require(!p.cycle && !p.active && ticket.source_position()==p.next_source &&
        ticket.source_position()+1U==ticket.committed_session().next_position &&
        token==ticket.predicted_token(),"MTP cycle requires the actual pending target selection/frontier");
    if(p.last_target) {
        require_chain(*p.last_target,ticket);
        require(p.last_teacher_next_input && *p.last_teacher_next_input==ticket.input_token(),"MTP cycle does not continue its actual primed target input");
    }
    const auto admission=p.owner->admit_native_entry();
    p.preflight(p.next_source,token,true);p.owner->require_idle();
    // Metadata-only checkpoint: proposals overwrite rows starting at
    // base_source. All earlier rows are immutable, and every later reader is
    // bounded by its current source position after that row was overwritten.
    // The first proposal input always copies the authoritative cycle ticket.
    p.base_previous_target=p.last_target;p.base_previous_input=p.last_teacher_next_input;
    p.cycle_base=ticket;p.base_source=p.next_source;p.pending_token=token;p.candidates.clear();p.probabilities.clear();p.cycle=true;p.first_cache_reusable=false;
    p.replay.clear();p.reconcile_final.reset();p.have_replay=false;
    if(p.draft) p.draft->observe(std::span<const std::uint32_t>(&token,1));
}
void HipMtpRuntime::enable_draft_chain(HipMtpDraftHeadConfig config,bool exact) {
    auto& p=*impl_;require(!p.draft && !p.cycle && !p.active && !p.broken,"MTP draft chain must be enabled once on an idle owner");
    const auto& m=p.owner->model();const auto& head=p.owner->tensors().lm_head;
    require(select_hip_mtp_lm_head_recipe(m,head)==HipMtpLmHeadRecipe::prefetch8_wave8_q8,
        "MTP draft chain requires the Q8_0 prefetch8 head recipe");
    config.hidden_size=static_cast<std::uint32_t>(m.hidden_size);config.vocab_size=static_cast<std::uint32_t>(m.vocab_size);
    config.max_depth=p.owner->budget().maximum_candidate_depth;
    p.draft_exact=exact;
    if(!exact) {
        // The fast MoE is specialized to the canonical sidecar; preflight keeps the exact tensor checks.
        require(m.hidden_size==2560U && m.moe_intermediate_size==640U && m.shared_expert_intermediate_size==640U &&
            m.num_experts==512U && m.num_experts_per_token==10U,"MTP fast draft MoE requires the canonical sidecar geometry");
        float* workspace=nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&workspace),hip_mtp_fast_moe_workspace_floats*sizeof(float)),"MTP fast draft workspace");
        p.fast_workspace.reset(workspace);
        require(m.hc_count==4U && m.hc_lowrank==320U,"MTP fast draft HC mix requires the canonical sidecar geometry");
        constexpr std::uint64_t inject=4U*10240U;float* hc=nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&hc),2U*inject*sizeof(float)),"MTP draft HC injection copies");
        p.hc_inject.reset(hc);
        launch_hip_mtp_dequantize_q8(reinterpret_cast<const void*>(p.owner->tensors().attention_hc_inject.runtime->address),inject,hc,p.owner->stream());
        launch_hip_mtp_dequantize_q8(reinterpret_cast<const void*>(p.owner->tensors().mlp_hc_inject.runtime->address),inject,hc+inject,p.owner->stream());
        // The draft's dense and expert projections read Q4_0 copies (proposals only; verification
        // is unchanged). With {2,4,8} windows the cheaper deep drafts win: 46.8 -> 47.9 tok/s.
        // STRIX_MTP_DRAFT_Q4_ALL=0 keeps the Q8 sidecar.
        const char* q4_all=std::getenv("STRIX_MTP_DRAFT_Q4_ALL");
        if(!q4_all || q4_all[0]!='0') {
            const auto& t=p.owner->tensors();
            for(const auto* v:{&t.eh_projection,&t.query,&t.key,&t.value,&t.attention_output,&t.expert_gate,&t.expert_up,
                    &t.expert_down,&t.shared_gate,&t.shared_up,&t.shared_down}) {
                require(v->manifest && v->runtime && v->manifest->ggml_type==8U && v->runtime->bytes%34U==0U,
                    "MTP draft Q4 copies require Q8_0 sidecar projections");
                register_hip_mtp_draft_q4(reinterpret_cast<const void*>(v->runtime->address),v->runtime->bytes/34U*32U,p.owner->stream());
            }
        }
        check(hipStreamSynchronize(static_cast<hipStream_t>(p.owner->stream())),"MTP draft HC injection copies");
    }
    p.draft=std::make_unique<HipMtpDraftHead>(reinterpret_cast<const void*>(head.runtime->address),std::move(config),p.owner->stream());
}
const HipMtpDraftHead* HipMtpRuntime::draft_head() const noexcept {return impl_->draft.get();}
void HipMtpRuntime::observe_draft_context(std::span<const std::uint32_t> tokens) {
    auto& p=*impl_;require(p.draft && !p.active && !p.broken,"MTP draft context requires an enabled idle chain");p.draft->observe(tokens);
}
std::vector<std::uint32_t> HipMtpRuntime::propose_chain(const QualifiedMtpDraftPlan& plan,const ActivatedMtpDraftPlan& activated,
    const SerialSessionContext& target,const DraftProposalStateContext& proposal,std::uint32_t depth,float min_probability,
    const std::function<bool(std::span<const float>)>& keep_drafting) {
    auto& p=*impl_;p.qualification(plan,activated);
    validate_mtp_draft_proposal_phase(p.owner->binding(),proposal,target,DraftProposalStatePhase::proposal_active);
    require(p.cycle && p.cycle_base && same_session(target,p.cycle_base->committed_session()),"MTP draft chain does not match its native cycle");
    return p.chain(depth,min_probability,keep_drafting);
}
std::span<const float> HipMtpRuntime::draft_probabilities() const noexcept {return impl_->probabilities;}
void HipMtpRuntime::set_draft_sampler(std::shared_ptr<TokenSampler> sampler) {
    auto& p=*impl_;require(p.draft && sampler && !p.sampler && !p.cycle && !p.active && !p.broken,
        "MTP sampled drafts need an enabled idle draft chain, once before cycles");
    p.sampler=std::move(sampler);
}
std::span<const TokenDraft> HipMtpRuntime::sampled_drafts() const noexcept {return impl_->sampled;}
void HipMtpRuntime::dispatch_mtp_step(const MtpForwardStepContext& call) {
    auto& p=*impl_;
    try {
        require(call.cycle_input && call.binding && call.state_memory && call.manifest && call.runtime,
            "MTP dispatch lacks its typed physical source bindings");
        validate_mtp_draft_runtime_binding(*call.manifest,*call.runtime,*call.binding);
        validate_mtp_draft_state_memory_binding(plan_mtp_draft_state_memory(p.owner->plan().proposal_identity,p.owner->plan().state),*call.state_memory);
        require(call.binding->physical_binding_identity==p.owner->binding().physical_binding_identity &&
            call.binding->compatibility_identity==p.owner->binding().compatibility_identity &&
            call.state_memory->device_base==p.owner->state_memory().device_base &&
            call.state_memory->allocation_generation==p.owner->state_memory().allocation_generation &&
            call.state_memory->plan_identity==p.owner->state_memory().plan_identity,
            "MTP dispatch physical source differs from its native owner");
        p.stage(call);
    } catch(...) {if(p.active) p.failure();throw;}
}
MtpDraftCycleInput HipMtpRuntime::make_cycle_input(const QualifiedMtpDraftPlan& plan,const ActivatedMtpDraftPlan& activated,
    const MtpDraftRuntimeBinding& binding,const MtpDraftStateMemoryBinding& state,const SerialSessionContext& target,
    const DraftProposalStateContext& proposal,std::span<const std::uint32_t> prior) {
    auto& p=*impl_;p.qualification(plan,activated);
    validate_mtp_draft_proposal_phase(binding,proposal,target,DraftProposalStatePhase::proposal_active);
    validate_mtp_draft_runtime_binding(p.owner->manifest(),p.owner->runtime_binding(),binding);
    validate_mtp_draft_state_memory_binding(plan_mtp_draft_state_memory(plan.base.proposal_identity,plan.base.state),state);
    require(p.cycle && p.cycle_base && same_session(target,p.cycle_base->committed_session()) &&
        prior.size()==p.candidates.size() && prior.size()<p.owner->budget().maximum_candidate_depth &&
        std::equal(prior.begin(),prior.end(),p.candidates.begin()) && state.device_base==p.owner->state_memory().device_base &&
        state.allocation_generation==p.owner->state_memory().allocation_generation,"MTP cycle history/state does not match its native owner");
    const auto v=p.owner->view(HipMtpBuffer::wide_current);
    MtpDraftCycleInput input{plan.base.proposal_identity,activated.activation_identity,target.runtime_compatibility_identity,
        prior.empty()?p.pending_token:prior.back(),p.next_source+1U,p.next_source,v.address,plan.base.state.state_width,2U,v.allocation_generation};
    p.start(input,prior.empty()?&*p.cycle_base:nullptr);return input;
}
std::uint32_t HipMtpRuntime::read_greedy_token(const QualifiedMtpDraftPlan& plan,const MtpDraftCycleInput& input,
    const MtpForwardExecutionResult& result,std::uint32_t index) {
    auto& p=*impl_;
    try {
        require_qualified_mtp_draft_plan(plan);
        require(plan.base.plan_identity==p.owner->plan().plan_identity && p.cycle && index==p.candidates.size() && same_input(input,p.current) &&
            plan.native_arithmetic_profile_identity==p.profile &&
            result.completed_steps==native_steps.size() && result.next_state_published && result.logits_published,
            "MTP token read does not match a complete pending native forward");
        const auto token=p.finish();
        if(index==0U) p.first_cache_reusable=p.cycle_base && p.current.hidden_source_position==p.base_source &&
            p.current.seed_position==p.base_source+1U && p.current.seed_token_id==p.pending_token;
        p.candidates.push_back(token);return token;
    } catch(...) {if(p.active) p.failure();throw;}
}
void HipMtpRuntime::set_reconciliation_rows(std::span<const HipMtpReplayRow> rows,const HipMtpTargetHiddenTicket& final_target) {
    auto& p=*impl_;require(p.cycle && !p.active && p.cycle_base && rows.size()<=p.owner->budget().maximum_candidate_depth+1U,
        "MTP reconciliation requires its bounded completed proposal");p.validate_ticket(final_target);
    require(final_target.same_producer(*p.cycle_base) &&
        final_target.source_position()+1U==final_target.committed_session().next_position &&
        final_target.source_position()>=p.base_source &&
        final_target.source_position()-p.base_source==rows.size(),"MTP reconciliation target rows do not cover the checkpoint frontier");
    for(std::size_t i=0;i<rows.size();++i) {
        p.validate_ticket(rows[i].hidden);require(rows[i].hidden.source_position()==p.base_source+i &&
            rows[i].hidden.same_producer(*p.cycle_base),"MTP reconciliation source row changed");
        const auto& next=i+1U<rows.size()?rows[i+1U].hidden:final_target;
        require_chain(rows[i].hidden,next);require(rows[i].next_input_token==next.input_token(),"MTP reconciliation is not actual teacher-forced target input");
        if(!i) require(same_session(rows[i].hidden.committed_session(),p.cycle_base->committed_session()),"MTP reconciliation checkpoint history changed");
    }
    if(rows.empty()) require(same_session(final_target.committed_session(),p.cycle_base->committed_session()),"MTP empty reconciliation changed target history");
    p.replay.assign(rows.begin(),rows.end());p.reconcile_final=final_target;p.have_replay=true;
}
void HipMtpRuntime::reconcile_after_target(const QualifiedMtpDraftPlan& plan,const ActivatedMtpDraftPlan& activated,
    const MtpDraftRuntimeBinding& binding,const MtpDraftStateMemoryBinding& state,const DraftProposalStateContext& proposal,
    const SerialSessionContext& target,std::span<const std::uint32_t> emitted) {
    auto& p=*impl_;p.qualification(plan,activated);
    validate_mtp_draft_proposal_phase(binding,proposal,target,DraftProposalStatePhase::reconciliation_required);
    validate_mtp_draft_runtime_binding(p.owner->manifest(),p.owner->runtime_binding(),binding);
    validate_mtp_draft_state_memory_binding(plan_mtp_draft_state_memory(plan.base.proposal_identity,plan.base.state),state);
    require(p.cycle && !p.active && p.have_replay && p.reconcile_final && same_session(target,p.reconcile_final->committed_session()) &&
        state.device_base==p.owner->state_memory().device_base && state.allocation_generation==p.owner->state_memory().allocation_generation &&
        emitted.size()<=p.owner->budget().maximum_candidate_depth+1U,"MTP reconciliation requires actual committed target rows and owner state");
    // Output tokens do not attest draft-cache correctness; real committed
    // tickets and complete teacher-pair coverage above are the authority.
    host_phase_mark("9a reconcile validation");
    {
        const auto admission=p.owner->admit_native_entry();
        for(const auto& row:p.replay) p.preflight(row.hidden.source_position(),row.next_input_token,true,true);
    }
    host_phase_mark("9b reconcile preflight");
    try {
        p.next_source=p.base_source;
        // Earlier occupied rows were never overwritten. The whole proposal
        // suffix is now logically invalid until actual teacher pairs overwrite
        // its committed prefix in order. Bytes beyond next_source stay unused;
        // they are not restored, compared as committed data or read by a later
        // attention row before that row has produced its own cache values.
        p.last_target=p.base_previous_target;p.last_teacher_next_input=p.base_previous_input;
        std::size_t first=0;
        if(p.draft) {
            if(!p.replay.empty() && p.first_cache_reusable && p.replay[0].hidden.same_commit(*p.cycle_base) &&
               p.replay[0].hidden.source_position()==p.base_source && p.replay[0].hidden.view()==p.cycle_base->view() &&
               p.replay[0].next_input_token==p.pending_token) {p.reuse_first_teacher_cache(p.replay[0]);first=1;}
            host_phase_mark("9c reuse first cache");
            p.teacher_cache_rows(std::span<const HipMtpReplayRow>(p.replay).subspan(first));
            host_phase_mark("9d teacher cache rows (gpu+sync)");
            p.draft->observe(emitted);first=p.replay.size();
            host_phase_mark("9e draft observe");
        }
        for(std::size_t i=first;i<p.replay.size();++i) {
            const auto& row=p.replay[i];
            const bool reuse=i==0U && p.first_cache_reusable && p.cycle_base &&
                row.hidden.same_commit(*p.cycle_base) && row.hidden.source_position()==p.base_source &&
                row.hidden.view()==p.cycle_base->view() && row.next_input_token==p.pending_token;
            if(reuse)p.reuse_first_teacher_cache(row);
            else p.teacher_cache(row.hidden,row.next_input_token);
        }
        require(p.next_source==p.reconcile_final->source_position(),"MTP replay did not reach the actual committed target frontier");
        p.cycle=false;p.cycle_base.reset();p.base_previous_target.reset();p.base_previous_input.reset();
        p.replay.clear();p.reconcile_final.reset();p.have_replay=false;p.candidates.clear();p.first_cache_reusable=false;
    } catch(...) {p.failure();throw;}
}
} // namespace strix
