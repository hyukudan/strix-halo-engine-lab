#pragma once
#include "hip_decode_experiment.hpp"
#include "hip_decode_options.hpp"
#include "hip_decode_receipt_json.hpp"
#include "mtp_window_memory_reservation.hpp"
#include "hip_prefix_state_cache.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/hip_known_prefix_window.hpp"
#include "strix/hip_known_prompt_backend.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_mtp_asset_loader.hpp"
#include "strix/hip_launcher_catalog.hpp"
#include "strix/hip_mtp_forward_replay.hpp"
#include "strix/mtp_forward_golden_file.hpp"
#include "strix/hip_mtp_greedy_runtime.hpp"
#include "strix/hip_mtp_window_verifier.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include "strix/mtp_draft_activation.hpp"
#include "strix/mtp_draft_qualification.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/session_executor_handoff.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <ostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <utility>

namespace strix::decode_cli {
// These are the actual target parents. The PLE backend pins its source mapping;
// the private hash/prefetch metadata remains stable until all native owners die.
// No pointer to a caller's temporary PLE transaction is retained.
struct MtpDecodeTargetParents final {
    std::shared_ptr<HipDecodeExperimentResources> experiment;
    std::shared_ptr<HipLauncherCatalog> catalog;
    std::shared_ptr<PleLookupBackend> ple;
    PleHashContract hash;
    PleLookupPrefetchState prefetch;
    ExecutionRuntimeResources resources;
    // Split prefill numerical profile: the sibling serial execution that runs the prompt rows,
    // its launch dispatcher and `resources` with the sibling view of the target scratch. All are
    // retained by the caller through the bridge call. Null compiled: one profile for every row.
    struct PromptExecution {
        const CompiledSerialExecution* compiled = nullptr;
        DirectLaunchDispatcher* dispatcher = nullptr;
        ExecutionRuntimeResources resources;
    } prompt;
    // Tiled MTP prompt: the prompt runs through the tiled known-prefix prefill (complete T-row
    // blocks, optional smaller tail blocks, the rest serially through the MTP target producer) and
    // the predictor cache is built from every block row's wide hidden state. Null blocks: the
    // ordinary serial MTP prompt.
    struct TiledPrompt {
        HipKnownPromptBackend* blocks = nullptr;
        std::shared_ptr<HipKnownPrefixWindow> block_owner;
        struct Tail {
            HipKnownPromptBackend* backend = nullptr;
            std::shared_ptr<HipKnownPrefixWindow> owner;
            std::uint32_t rows = 0;
        };
        std::vector<Tail> tails; // descending rows
        std::uint32_t rows = 0;
        HipPrefixStateCacheSession* cache = nullptr;
        // Fresh target state and session, exactly as after setup (after the calibration replay).
        std::function<void()> reset_state;
        SerialSessionContext fresh;
    } tiled;
    struct WindowSetupReceipt {
        std::uint64_t union_source_steps = 0;
        std::uint64_t union_reserved_bytes = 0;
        std::uint64_t union_generation = 0;
        std::uint64_t retained_window_bytes = 0;
    } window_setup;
    MtpDecodeTargetParents(std::shared_ptr<HipDecodeExperimentResources> actual,
        std::shared_ptr<HipLauncherCatalog> launchers,std::shared_ptr<PleLookupBackend> lookup,const PleHashContract& contract)
        :experiment(std::move(actual)),catalog(std::move(launchers)),ple(std::move(lookup)),hash(contract) {
        if(!experiment || !catalog || !ple)throw std::invalid_argument("MTP CLI requires actual retained target/launcher/PLE parents");
        resources=experiment->resources;
        resources.ple_lookup=ple.get();resources.ple_hash_contract=&hash;resources.ple_prefetch=&prefetch;
    }
    MtpDecodeTargetParents(const MtpDecodeTargetParents&)=delete;
    MtpDecodeTargetParents& operator=(const MtpDecodeTargetParents&)=delete;
};
inline void write_mtp_union_observation(std::ostream& output,
    const std::optional<HipKnownPrefixUnionObservation>& observed) {
    if (!observed) { output << "null"; return; }
    const auto& group = observed->group;
    output << "{\"scope\":\"last-completed-window-last-eligible-layer-including-restored-windows-not-aggregate-reuse-or-traffic\","
        << "\"first_position\":" << observed->first_position << ",\"layer\":" << observed->layer
        << ",\"logical_rows\":" << group.logical_rows << ",\"status\":" << group.status
        << ",\"version\":" << group.version << ",\"assignments\":" << group.assignments
        << ",\"touched\":" << group.touched << ",\"batches\":" << group.batches
        << ",\"duplicate_assignments\":" << group.duplicate_assignments
        << ",\"maximum_assignments\":" << group.maximum_assignments << ",\"rows_per_expert_histogram\":[";
    for (unsigned i = 0; i < 8U; ++i) output << (i ? "," : "") << group.rows_per_expert_histogram[i];
    output << "],\"batch_size_histogram\":[";
    for (unsigned i = 0; i < 8U; ++i) output << (i ? "," : "") << group.batch_size_histogram[i];
    output << "]}";
}
inline ModelContentIdentity mtp_cli_digest(std::string_view text) {
    if(!mtp_sha256_text(text))throw std::invalid_argument("MTP SHA256 pin is invalid");
    const auto digit=[](char ch)->unsigned {if(ch>='0'&&ch<='9')return static_cast<unsigned>(ch-'0');
        if(ch>='a'&&ch<='f')return static_cast<unsigned>(ch-'a')+10U;
        return static_cast<unsigned>(ch-'A')+10U;};
    ModelContentIdentity out;
    for(std::size_t i=0;i<out.bytes.size();++i)out.bytes[i]=static_cast<std::uint8_t>((digit(text[i*2U])<<4U)|digit(text[i*2U+1U]));
    return out;
}
inline std::uint64_t mtp_cli_add(std::uint64_t a,std::uint64_t b) {
    if(b>UINT64_MAX-a)throw std::overflow_error("MTP CLI payload budget overflow");
    return a+b;
}
inline std::uint64_t mtp_cli_mul(std::uint64_t a,std::uint64_t b) {
    if(b && a>UINT64_MAX/b)throw std::overflow_error("MTP CLI payload budget overflow");
    return a*b;
}
using MtpDecodeWindowFactory=std::function<std::shared_ptr<HipMtpWindowVerifier>(std::shared_ptr<HipMtpTargetHidden>)>;
struct MtpDecodeQualificationContext {
    const Options& options;
    std::shared_ptr<MtpDecodeTargetParents> parents;
    std::shared_ptr<HipMtpResources> predictor;
    std::shared_ptr<HipMtpTargetHidden> target;
    std::shared_ptr<HipMtpWindowVerifier> window;
    SerialSessionContext& session;
    DirectLaunchDispatcher& dispatcher;
    AllocationGenerationSource& generations;
    HipMtpGreedyRuntime& greedy;
    HipMtpRuntime& runtime;
    HipMtpTargetHiddenTicket initial;
};
using MtpDecodeQualification=std::function<void(MtpDecodeQualificationContext&)>;
using MtpDecodeComponentDiagnostic=std::function<void(const std::shared_ptr<HipMtpResources>&,std::uint64_t)>;

// Tiled MTP prompt after the calibration replay. Complete T-row blocks (then complete tail blocks)
// run through the tiled prefill; a step observer copies every row's wide hidden state (the MTP
// target tap) out of the block arena, and the predictor's cache rows are built from those copies
// (pair (h_i, t_{i+1})). The remaining rows, at least the last prompt row, run serially through
// the MTP target producer, whose last ticket is the pending row of the first cycle. With a prefix
// state cache the longest cached block boundary P is restored first (target state, predictor
// rows [0, P-1) and h_{P-1}), and selected block boundaries are captured.
inline HipMtpTargetHiddenTicket run_tiled_mtp_prompt(const Options& options,const ModelContract& model,
    const CompiledSerialExecution& compiled,MtpDecodeTargetParents::TiledPrompt& tiled,SerialSessionContext& session,
    HipMtpTargetHidden& target,HipMtpRuntime& runtime,const QualifiedMtpDraftPlan& qualified,const ActivatedMtpDraftPlan& activated,
    const ExecutionRuntimeResources& resources,DirectLaunchDispatcher& dispatcher,std::byte* held,std::string& receipt) {
    const auto check=[](hipError_t e,const char* what){if(e!=hipSuccess)throw std::runtime_error(std::string(what)+": "+hipGetErrorString(e));};
    const std::span<const std::uint32_t> tokens(options.tokens);
    const auto n=static_cast<std::uint64_t>(tokens.size());
    const auto width=mtp_cli_mul(mtp_cli_mul(model.hidden_size,model.hc_count),2U);
    // The target tap: the last MLP injection before the final hyper mix (same step in every sibling).
    std::size_t tap=SIZE_MAX;
    for(std::size_t i=1;i<compiled.execution.steps.size();++i)
        if(compiled.execution.steps[i].op==SerialOp::final_hyper_mix && compiled.execution.steps[i-1U].op==SerialOp::mlp_inject) tap=i-1U;
    if(tap==SIZE_MAX)throw std::logic_error("tiled MTP prompt has no wide hidden tap step");
    const auto max_rows=tiled.rows;
    std::byte* block_hidden=nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&block_hidden),static_cast<std::size_t>(mtp_cli_mul(max_rows,width))),"tiled MTP hidden rows");
    struct Free{std::byte* p;~Free(){if(p)(void)hipFree(p);}} free_rows{block_hidden};
    const auto view=[&](const std::byte* p){return MtpDeviceView{reinterpret_cast<std::uintptr_t>(p),width,1U};};
    std::uint32_t block_rows=0;
    const HipKnownPrefixWindow::PrefillStepObserver observer=[&](std::size_t step,const std::function<ExecutionDispatchContext(std::uint32_t)>& row) {
        if(step!=tap) return;
        const auto stream=hip_execution_stream();
        for(std::uint32_t r=0;r<block_rows;++r) {
            const auto c=row(r);
            const auto source=resolve_dispatch_write(c,0);
            if(source.bytes<width)throw std::logic_error("tiled MTP tap row is undersized");
            check(hipMemcpyAsync(block_hidden+static_cast<std::size_t>(r)*width,reinterpret_cast<const void*>(source.address),
                static_cast<std::size_t>(width),hipMemcpyDeviceToDevice,stream),"tiled MTP tap copy");
        }
    };
    std::uint64_t restored=0,blocks=0,tail_blocks=0,serial_rows=0;
    const auto start=std::chrono::steady_clock::now();
    if(tiled.cache) restored=tiled.cache->begin_request(session,tokens);
    std::uint64_t pos=restored;
    std::vector<MtpDeviceView> views;std::vector<std::uint32_t> next;
    const auto run=[&](HipKnownPromptBackend& backend,HipKnownPrefixWindow& owner,std::uint32_t rows,bool capture) {
        owner.set_prefill_step_observer(observer);block_rows=rows;
        try {
            while(pos+rows<=n-1U) {
                backend.execute_state_rows(tokens.subspan(static_cast<std::size_t>(pos),rows),pos);
                views.clear();next.clear();
                if(pos){views.push_back(view(held));next.push_back(tokens[static_cast<std::size_t>(pos)]);}
                for(std::uint32_t r=0;r+1U<rows;++r){views.push_back(view(block_hidden+static_cast<std::size_t>(r)*width));
                    next.push_back(tokens[static_cast<std::size_t>(pos+r+1U)]);}
                runtime.prime_hidden_rows(qualified,activated,views,next);
                check(hipMemcpy(held,block_hidden+static_cast<std::size_t>(rows-1U)*width,static_cast<std::size_t>(width),hipMemcpyDeviceToDevice),
                    "tiled MTP held copy");
                pos+=rows;
                if(capture && tiled.cache) tiled.cache->after_block(session,tokens,n-1U);
                if(capture)++blocks;else ++tail_blocks;
            }
        } catch(...) {owner.set_prefill_step_observer({});throw;}
        owner.set_prefill_step_observer({});
    };
    run(*tiled.blocks,*tiled.block_owner,tiled.rows,true);
    for(const auto& tail:tiled.tails) run(*tail.backend,*tail.owner,tail.rows,false);
    HipMtpTargetHiddenTicket pending;
    while(pos<n) {
        auto ticket=target.execute_serial(session,resources,dispatcher,tokens[static_cast<std::size_t>(pos)]);
        if(pos) {
            const std::array<MtpDeviceView,1> one{view(held)};const std::array<std::uint32_t,1> token{tokens[static_cast<std::size_t>(pos)]};
            runtime.prime_hidden_rows(qualified,activated,one,token);
        }
        check(hipMemcpy(held,reinterpret_cast<const void*>(ticket.view().address),static_cast<std::size_t>(width),hipMemcpyDeviceToDevice),
            "tiled MTP held ticket copy");
        pending=std::move(ticket);++pos;++serial_rows;
    }
    if(runtime.next_source_position()+1U!=n || session.next_position!=n)
        throw std::logic_error("tiled MTP prompt did not reach the final prompt frontier");
    const auto prompt_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std::ostringstream out;
    out<<"\"rows\":"<<tiled.rows<<",\"tail_rows\":[";
    for(std::size_t t=0;t<tiled.tails.size();++t)out<<(t?",":"")<<tiled.tails[t].rows;
    out<<"],\"restored_position\":"<<restored
       <<",\"blocks\":"<<blocks<<",\"tail_blocks\":"<<tail_blocks<<",\"serial_rows\":"<<serial_rows
       <<",\"prompt_ms\":"<<prompt_ms;
    if(tiled.cache) {
        const auto& q=tiled.cache->request();
        out<<",\"prefix_cache\":{\"lookup_ms\":"<<q.lookup_ms<<",\"restore_ms\":"<<q.restore_ms<<",\"restored_bytes\":"<<q.restored_bytes
           <<",\"captured_checkpoints\":"<<q.captured_checkpoints<<",\"captured_segment_bytes\":"<<q.captured_segment_bytes
           <<",\"captured_fixed_bytes\":"<<q.captured_fixed_bytes<<",\"capture_ms\":"<<q.capture_ms
           <<",\"row_bytes_per_position\":"<<tiled.cache->row_bytes()<<",\"fixed_bytes_per_checkpoint\":"<<tiled.cache->fixed_bytes()<<'}';
        static const bool digest=[]{const char* v=std::getenv("STRIX_PREFIX_CACHE_DIGEST");return v && v[0]=='1';}();
        if(digest) out<<",\"state_digest\":\""<<model_content_identity_hex(tiled.cache->state_digest(n))<<'"';
    }
    receipt=out.str();
    return pending;
}

// One synchronous CLI request with externally pinned evidence. Setup/loading,
// independent replay, activation and generation have separate receipt scopes.
// There is no path from command-line booleans to a qualified/activated plan.
inline void run_mtp_decode_bridge(const Options& options,const ModelContract& model,
    const CompiledSerialExecution& compiled,const std::shared_ptr<MtpDecodeTargetParents>& parents,
    SerialSessionContext& session,DirectLaunchDispatcher& dispatcher,AllocationGenerationSource& generations,
    bool deferred_status,std::string_view resolved_profile,std::ostream& output,
    MtpDecodeWindowFactory make_window={},std::function<std::string()> completion_receipt={},
    MtpDecodeQualification qualify={},MtpDecodeComponentDiagnostic component_diagnostic={}) {
    validate_resolved_mtp_cli(options);
    if(!options.mtp.enabled || !parents || session.next_position || session.committed_tokens ||
       session.health!=SerialSessionHealth::ready || session.execution_identity!=compiled.execution_identity)
        throw std::invalid_argument("MTP CLI requires a fresh actual target session");
    const auto& m=options.mtp;const auto& r=parents->resources;const auto& actual=*parents->experiment;
    const auto width=mtp_cli_mul(model.hidden_size,model.hc_count);
    HipMtpForwardGolden golden;
    const bool dump=!m.golden_dump.empty();
    if(((m.metadata_only || dump) && (qualify || component_diagnostic)) || (qualify && component_diagnostic))
        throw std::invalid_argument("MTP metadata, golden dump, full fixture and component diagnostic routes are mutually exclusive");
    const auto pair=numerical_profile_pair(options);
    const auto prefill_profile_identity=numerical_profile_identity(pair.prefill);
    const auto decode_profile_identity=numerical_profile_identity(pair.decode);
    const bool split=pair.split();
    const bool tiled_prompt=parents && parents->tiled.blocks;
    const std::size_t calibration_rows=tiled_prompt ? mtp_calibration_rows(options) : options.tokens.size()-1U;
    if(tiled_prompt && (split || qualify || component_diagnostic || !parents->tiled.block_owner || !parents->tiled.reset_state ||
       !parents->tiled.rows || std::any_of(parents->tiled.tails.begin(),parents->tiled.tails.end(),
           [&](const auto& t){return !t.backend || !t.owner || t.rows>=parents->tiled.rows;})))
        throw std::invalid_argument("tiled MTP prompt requires one numerical profile, its block owners and a state reset");
    if(split!=(parents && parents->prompt.compiled) ||
       (split && (!parents->prompt.dispatcher || parents->prompt.resources.session_state_memory!=parents->resources.session_state_memory)))
        throw std::invalid_argument("MTP split prefill profile requires its actual prompt execution and only then");
    if(!m.metadata_only && !dump){
    golden=read_mtp_forward_golden_file(m.golden,mtp_cli_digest(m.golden_sha256),m.golden_max_bytes);
    // A golden records the target prefill/decode pair whose prefill produced its prompt rows. A
    // version-1 golden records none and is admitted only for a single-profile run, as before.
    if(golden.target_decode_profile_identity ?
           golden.target_prefill_profile_identity!=prefill_profile_identity || golden.target_decode_profile_identity!=decode_profile_identity :
           split)
        throw std::invalid_argument("MTP golden was dumped under another target prefill/decode numerical profile pair "
            "(a version-1 golden records none and requires one profile for both); dump a golden with this pair");
    if(tiled_prompt ? golden.rows.size()!=calibration_rows :
       golden.rows.size()+1U!=options.tokens.size())
        throw std::invalid_argument(tiled_prompt ? "MTP golden of a tiled prompt must cover exactly its calibration rows" :
            "MTP golden must cover every prompt predictor pair, leaving only its final pending target row");
    for(std::size_t i=0;i<golden.rows.size();++i) {
        const auto& row=golden.rows[i];
        if(row.source_position!=i || row.source_input_token!=options.tokens[i] || row.next_input_token!=options.tokens[i+1U] ||
           row.target_hidden_bf16.size()!=width || row.next_hidden_bf16.size()!=width || row.full_logits.size()!=model.vocab_size)
            throw std::invalid_argument("MTP independent golden prompt chain or full-row geometry differs before target execution");
    }
    }
    HipMtpAssetLoadRequest load;load.sidecar_path=m.sidecar;load.sidecar_logical_name=m.sidecar_logical_name;
    load.expected_target_file_sha256=mtp_cli_digest(m.target_sha256);load.expected_target_content=mtp_cli_digest(m.target_content);
    auto& evidence=load.expected_sidecar;evidence.repository=m.repository;evidence.revision=m.revision;evidence.relative_path=m.source_path;
    evidence.declared_file_sha256=mtp_cli_digest(m.sidecar_sha256);evidence.declared_file_bytes=m.sidecar_bytes;
    evidence.conversion=MtpSidecarConversionProfile::effective_norm_gain_embedding_first_packed_v1;
    evidence.converter_reference_sha256=mtp_cli_digest(m.converter_sha256);
    evidence.norm_converter_reference_sha256=mtp_cli_digest(m.norm_converter_sha256);evidence.converter_reference_revision=m.converter_revision;
    load.metadata_read_budget_bytes=m.header_bytes;
    load.residency.mode=m.sidecar_residency=="copy"?HipModelResidencyMode::resident_copy:HipModelResidencyMode::registered_file_mapping;
    load.residency.reservation_limits={m.sidecar_retained_bytes,m.sidecar_setup_bytes};
    const auto assets=HipMtpAssets::load(model,load,actual.mapped,actual.manifest,actual.model,generations);
    HipMtpBudget native_budget;native_budget.retained_bytes=m.runtime_bytes;native_budget.context_capacity=options.context_tokens;
    native_budget.maximum_candidate_depth=m.candidate_depth;
    native_budget.normalized_rounding=m.normalized_rounding=="fp32"?HipMtpInputRounding::fp32:HipMtpInputRounding::bf16_rne;
    native_budget.projection_rounding=m.projection_rounding=="fp32"?HipMtpInputRounding::fp32:HipMtpInputRounding::bf16_rne;
    auto native=HipMtpResources::create(model,assets->compiled(),assets->sidecar_manifest(),assets->sidecar_residency(),
        assets->target_manifest(),assets->target_residency(),assets->shared_target(),native_budget,actual.allocator);
    auto runtime=std::make_shared<HipMtpRuntime>(native);
    if(m.metadata_only){
        assets->validate_assets_unchanged();
        output<<"{\"mode\":\"mtp-metadata-only\",\"mtp_executed\":false,\"mtp_activated\":false,\"mtp_qualified\":false,"
            <<"\"proposal_identity\":"<<assets->compiled().proposal_identity<<",\"plan_identity\":"<<assets->compiled().plan_identity
            <<",\"semantic_identity\":"<<assets->compiled().forward_semantics.contract_identity<<",\"native_profile_identity\":"<<runtime->numerical_profile_identity()
            <<",\"head_profile_identity\":"<<hip_mtp_lm_head_profile_identity(model,native->tensors().lm_head)
            <<",\"head_recipe\":"<<static_cast<std::uint32_t>(select_hip_mtp_lm_head_recipe(model,native->tensors().lm_head))
            <<",\"target_content\":\""<<model_content_identity_hex(assets->shared_target().target_model_content_identity)
            <<"\",\"sidecar_content\":\""<<model_content_identity_hex(assets->shared_target().sidecar_model_content_identity)
            <<"\",\"target_file_sha256\":\""<<model_content_identity_hex(actual.mapped->file_sha256())
            <<"\",\"sidecar_file_sha256\":\""<<model_content_identity_hex(assets->sidecar_mapping()->file_sha256())
            <<"\",\"normalized_rounding\":\""<<m.normalized_rounding<<"\",\"projection_rounding\":\""<<m.projection_rounding<<"\"}\n";
        return;
    }
    MtpForwardReplayPolicy policy{m.policy_identity,m.max_hidden_error,m.max_logit_error,2U,true};
    (void)assess_mtp_forward_replay(policy,{});
    if(!dump && (golden.target_content!=assets->shared_target().target_model_content_identity ||
       golden.sidecar_content!=assets->shared_target().sidecar_model_content_identity ||
       golden.structural_plan_identity!=assets->compiled().plan_identity ||
       golden.semantic_identity!=assets->compiled().forward_semantics.contract_identity ||
       golden.numerical_profile_identity!=runtime->numerical_profile_identity()))
        throw std::invalid_argument("MTP golden does not bind actual verified assets, semantics and native cast profile");
    if(component_diagnostic){
        // Independent component inputs are admitted against these actual assets.
        // This diagnostic neither runs full replay nor creates qualified/active
        // greedy state, and cannot enter the ordinary generation path.
        try {assets->validate_assets_unchanged();component_diagnostic(native,runtime->numerical_profile_identity());assets->validate_assets_unchanged();}
        catch(...){runtime->cancel();throw;}
        return;
    }
    // Same complete known payload bound as the replay consumer; admit it before
    // the target prompt is mutated, rather than waiting until replay entry.
    auto replay_payload=mtp_cli_add(mtp_cli_mul(width,4U),mtp_cli_mul(model.vocab_size,4U));
    replay_payload=mtp_cli_add(replay_payload,mtp_cli_mul(golden.rows.size(),sizeof(HipMtpForwardReplayRowReceipt)+2U*sizeof(std::uint32_t)));
    replay_payload=mtp_cli_add(replay_payload,mtp_cli_mul(golden.rows.size(),mtp_cli_add(mtp_cli_mul(width,4U),mtp_cli_mul(model.vocab_size,4U))));
    replay_payload=mtp_cli_add(replay_payload,mtp_cli_mul(golden.rows.size()+1U,192U));
    if(!dump && replay_payload>m.replay_host_bytes)throw std::invalid_argument("MTP complete independent replay exceeds host payload budget before target mutation");
    HipMtpForwardReplayRequest replay_request{policy,hip_mtp_forward_golden_payload_identity(golden),golden.rows.size(),m.replay_host_bytes};
    // Tiled prompt: only the calibration rows and the serial remainder take capture slots.
    const auto retained_rows=static_cast<std::uint32_t>(tiled_prompt
        ? std::max<std::size_t>({calibration_rows+1U,static_cast<std::size_t>(parents->tiled.tails.empty()?parents->tiled.rows:parents->tiled.tails.back().rows)+1U,
            static_cast<std::size_t>(m.candidate_depth)+3U})
        : std::max<std::size_t>(options.tokens.size(),static_cast<std::size_t>(m.candidate_depth)+3U));
    HipMtpTargetDeferredConfig d1;
    if(deferred_status)d1={r.session_state_layout,&generations};
    auto target=HipMtpTargetHidden::create(model,compiled,actual.memory,actual.model,actual.allocator,
        retained_rows,m.target_capture_bytes,d1);
    if(split)target->set_prompt_execution(*parents->prompt.compiled);
    std::shared_ptr<HipMtpWindowVerifier> window;
    if(m.window_rows && !dump) {
        if(!make_window)throw std::invalid_argument("MTP requested target window has no actual prepared native factory");
        window=make_window(target);
        if(!window || window->target_owner()!=target || window->logical_rows()!=m.window_rows)
            throw std::invalid_argument("MTP target window factory returned incompatible actual owner");
    }
    assets->validate_assets_unchanged();
    std::vector<HipMtpReplayRow> replay_rows;replay_rows.reserve(golden.rows.size());
    std::vector<std::uint32_t> generated;generated.reserve(static_cast<std::size_t>(options.generate_tokens));
    std::uint64_t cycles=0,proposed=0,accepted=0,target_rows=0,window_attempts=0,window_accepts=0,window_restores=0,r2_dispatches=0;
    std::array<std::uint64_t,9> chosen_rows{};
    HipMtpTargetHiddenTicket pending;
    HipMtpNativeForwardReplayResult replay;
    QualifiedMtpDraftPlan qualified;ActivatedMtpDraftPlan activated;
    const std::array<std::uint32_t,1> eos{options.eos_token.value_or(static_cast<std::uint32_t>(model.ple_eos_token_id))};
    std::string stop="token-budget";
    double calibration_ms=0.0;
    std::string tiled_receipt;
    // h_{P-1} of the tiled prompt (one wide row), retained through the request.
    std::byte* tiled_held=nullptr;
    if(tiled_prompt && hipMalloc(reinterpret_cast<void**>(&tiled_held),static_cast<std::size_t>(width*2U))!=hipSuccess)
        throw std::runtime_error("tiled MTP held hidden allocation failed");
    struct HeldFree{std::byte* p;~HeldFree(){if(p)(void)hipFree(p);}} held_free{tiled_held};
    try {
        // The prompt rows are the prefill: under a split pair they run the sibling prompt
        // execution, and the session moves to the decode execution only after the replay.
        if(split)hand_off_serial_session(session,{&compiled,&r},{parents->prompt.compiled,&parents->prompt.resources});
        const auto prompt_rows=tiled_prompt ? calibration_rows+1U : options.tokens.size();
        const auto calibration_start=std::chrono::steady_clock::now();
        for(std::size_t i=0;i<prompt_rows;++i) {
            auto current=split ? target->execute_serial(session,parents->prompt.resources,*parents->prompt.dispatcher,options.tokens[i])
                               : target->execute_serial(session,r,dispatcher,options.tokens[i]);
            if(pending.valid())replay_rows.push_back({pending,options.tokens[i]});
            pending=std::move(current);
        }
        if(dump){
            // Same native forward as replay, recorded instead of compared. The
            // explicit self-generated pins keep it distinguishable from an
            // independent reference; nothing is qualified or generated here.
            golden=capture_hip_mtp_forward_golden(*runtime,replay_rows,mtp_self_generated_golden_source(),mtp_self_generated_golden_reference);
            golden.target_prefill_profile_identity=prefill_profile_identity;
            golden.target_decode_profile_identity=decode_profile_identity;
            const auto sha=write_mtp_forward_golden_file(m.golden_dump,golden,m.golden_max_bytes);
            assets->validate_assets_unchanged();
            output<<"{\"mode\":\"mtp-self-generated-golden-dump\",\"mtp_reference_independence\":\"self-generated-native-forward-not-independent\","
                <<"\"mtp_golden_path\":\""<<decode_receipt::escape_json_string(m.golden_dump)<<"\",\"mtp_golden_sha256\":\""<<model_content_identity_hex(sha)
                <<"\",\"mtp_golden_rows\":"<<golden.rows.size()<<",\"mtp_golden_payload_sha256\":\""<<model_content_identity_hex(hip_mtp_forward_golden_payload_identity(golden))
                <<"\",\"mtp_independent_source_content\":\""<<model_content_identity_hex(golden.independent_source_content)
                <<"\",\"mtp_independent_reference_identity\":"<<golden.independent_reference_identity
                <<",\"mtp_structural_plan_identity\":"<<golden.structural_plan_identity<<",\"mtp_semantic_identity\":"<<golden.semantic_identity
                <<",\"mtp_native_arithmetic_profile_identity\":"<<golden.numerical_profile_identity
                <<",\"mtp_golden_target_prefill_numerical_profile\":\""<<numerical_profile_name(pair.prefill)
                <<"\",\"mtp_golden_target_decode_numerical_profile\":\""<<numerical_profile_name(pair.decode)
                <<"\",\"mtp_golden_target_prefill_profile_identity\":"<<golden.target_prefill_profile_identity
                <<",\"mtp_golden_target_decode_profile_identity\":"<<golden.target_decode_profile_identity<<"}\n";
            return;
        }
        // This call executes all eleven native stages for every independent
        // reference pair. The resulting real observation is the sole evidence.
        replay=replay_hip_mtp_forward(*runtime,replay_rows,golden,replay_request);
        if(!replay.assessment.passes_declared_policy){
            std::ostringstream error;error<<std::setprecision(17)
                <<"MTP actual full-hidden/full-vocabulary replay failed its declared policy; generation is not admitted"
                <<"; hidden_max_abs_error="<<replay.observation.maximum_hidden_abs_error
                <<"; logit_max_abs_error="<<replay.observation.maximum_logit_abs_error
                <<"; hidden_values="<<replay.observation.hidden_values_compared
                <<"; logit_values="<<replay.observation.logit_values_compared
                <<"; tokens_match="<<replay.assessment.tokens_match
                <<"; hidden_within_policy="<<replay.assessment.hidden_within_policy
                <<"; logits_within_policy="<<replay.assessment.logits_within_policy
                <<"; token_pairs=";
            for(std::size_t i=0;i<replay.observation.engine_tokens.size();++i){
                if(i)error<<',';
                error<<replay.observation.reference_tokens.at(i)<<':'<<replay.observation.engine_tokens[i];
            }
            throw std::runtime_error(error.str());
        }
        qualified=qualify_mtp_draft_plan(assets->compiled(),policy,replay.observation);
        require_qualified_mtp_draft_plan(qualified);
        activated=activate_mtp_draft_plan(assets->compiled(),policy,replay.observation);
        if(runtime->next_source_position()+1U!=session.next_position)
            throw std::logic_error("MTP replay did not prime the actual final prompt frontier");
        replay_rows.clear(); // release prompt snapshots before bounded cycles
        const auto enable_draft=[&] {
            if(!m.draft_chain) return;
            // Drafts only: the subset (or full) head changes acceptance, never target output.
            HipMtpDraftHeadConfig draft;draft.full=m.draft_vocab.empty() && !m.draft_context_vocab;
            if(!m.draft_vocab.empty()) {
                std::string chosen;
                const auto spec=HipMtpDraftHead::resolve_vocabulary_spec(m.draft_vocab,options.tokens,
                    static_cast<std::uint32_t>(model.vocab_size),chosen);
                if(spec!=m.draft_vocab) std::cerr<<"MTP draft vocabulary: "<<chosen<<'\n';
                draft.base_tokens=HipMtpDraftHead::read_token_file(spec,static_cast<std::uint32_t>(model.vocab_size));
            }
            draft.context_capacity=m.draft_context_vocab;draft.q4=m.draft_q4;
            // Runner-up probabilities feed the survival controller; STRIX_MTP_DRAFT_TOP2=1 logs them (diagnostic).
            draft.runner_up=hip_mtp_survival_controller() || std::getenv("STRIX_MTP_DRAFT_TOP2")!=nullptr;
            runtime->enable_draft_chain(std::move(draft),m.draft_exact);runtime->observe_draft_context(options.tokens);
        };
        if(tiled_prompt) {
            calibration_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-calibration_start).count();
            // The calibration rows only qualified the predictor: start the real prompt from a fresh
            // target and a fresh predictor (the qualified/activated plans bind assets, not an owner).
            pending={};
            runtime.reset();
            runtime=std::make_shared<HipMtpRuntime>(native);
            parents->tiled.reset_state();
            session=parents->tiled.fresh;
            enable_draft();
            if(parents->tiled.cache) {
                // The predictor's durable rows and the held h_{P-1} travel with each checkpoint; the
                // identity gains everything that changes them (cache rows use the draft kernels
                // when the draft chain is not exact).
                auto& cache=*parents->tiled.cache;
                const auto kv=native->view(HipMtpBuffer::kv),index=native->view(HipMtpBuffer::raw_index);
                cache.add_rows(kv.address,mtp_cli_mul(mtp_cli_mul(model.kv_heads,model.head_dim),4U),1U);
                cache.add_rows(index.address,mtp_cli_mul(model.indexer_head_dim,2U),1U);
                cache.add_fixed_device(reinterpret_cast<std::uintptr_t>(tiled_held),width*2U);
                std::vector<std::byte> frame;
                const auto u64=[&](std::uint64_t v){for(unsigned b=0;b<64;b+=8)frame.push_back(static_cast<std::byte>((v>>b)&255U));};
                for(const auto byte:assets->shared_target().sidecar_model_content_identity.bytes)frame.push_back(static_cast<std::byte>(byte));
                u64(assets->compiled().plan_identity);u64(runtime->numerical_profile_identity());
                u64(m.draft_chain);u64(m.draft_exact);u64(m.draft_q4);
                const char* q4_all=std::getenv("STRIX_MTP_DRAFT_Q4_ALL");u64(!q4_all || q4_all[0]!='0');
                cache.extend_identity(sha256_bytes(frame));
                cache.set_restore_hook([&](std::uint64_t position){runtime->restore_cache_frontier(qualified,activated,position-1U);});
            }
            pending=run_tiled_mtp_prompt(options,model,compiled,parents->tiled,session,*target,*runtime,qualified,activated,
                r,dispatcher,tiled_held,tiled_receipt);
        } else
        // Prefill handoff: every later target row, window and checkpoint runs the decode execution
        // from the state this prefill produced. The pending prompt row is re-minted with the
        // handed-off session; the lineage keeps the prefill identity of every prompt row.
        {
            if(split)pending=target->hand_off_prompt(session,pending,parents->prompt.resources,r);
            enable_draft();
        }
        HipMtpGreedyRuntime greedy(runtime,qualified,activated,target,session,r,dispatcher,parents);
        if(window && !qualify)greedy.use_window_verifier(window);
        if(m.draft_chain)greedy.set_draft_min_probability(mtp_draft_min_probability(m));
        if(qualify){
            MtpDecodeQualificationContext context{options,parents,native,target,std::move(window),session,dispatcher,generations,greedy,*runtime,std::move(pending)};
            qualify(context);
            assets->validate_assets_unchanged();
            return;
        }
        // The final prompt target already selected this unconsumed pending
        // token. Emit it once, then each cycle returns only subsequent outputs.
        generated.push_back(pending.predicted_token());
        const auto start=std::chrono::steady_clock::now();
        while(generated.size()<options.generate_tokens && generated.back()!=eos.front()) {
            if(session.next_position>=options.context_tokens){stop="context-capacity";break;}
            const auto result=greedy.cycle(pending,m.candidate_depth,
                static_cast<std::uint32_t>(options.generate_tokens-generated.size()),eos);
            if(result.emitted_tokens.empty())throw std::runtime_error("MTP generation made no completed target progress");
            generated.insert(generated.end(),result.emitted_tokens.begin(),result.emitted_tokens.end());
            pending=result.final_target;++cycles;proposed+=result.proposed_candidates;accepted+=result.accepted_candidates;
            target_rows+=result.consumed_target_rows;window_attempts+=result.attempted_target_window;window_accepts+=result.accepted_target_window;
            window_restores+=result.target_checkpoint_restored;r2_dispatches+=result.target_r2_dispatches;
            ++chosen_rows[std::min<std::uint32_t>(result.target_window_rows,8U)];
        }
        const auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(generated.back()==eos.front())stop="eos";
        assets->validate_assets_unchanged();
        const auto native_stats=runtime->execution_stats();
        const auto draft_stats=runtime->draft_head()?runtime->draft_head()->stats():HipMtpDraftHeadStats{};
        std::optional<HipKnownPrefixUnionObservation> union_observation;
        if(window)for(const auto& owner:window->window_owners())
            if(auto observed=owner->latest_routed_union_observation())union_observation=observed;
        if(m.window_expert_union && window_attempts &&
            (!parents->window_setup.union_source_steps || !union_observation))
            throw std::logic_error("MTP expert union completed no actual native group for an attempted window");
        std::ostringstream union_receipt;
        union_receipt << "  \"mtp_window_expert_union_requested\":" << (m.window_expert_union ? "true" : "false")
            << ",\n  \"mtp_window_expert_union_source_steps\":" << parents->window_setup.union_source_steps
            << ",\n  \"mtp_window_expert_union_workspace_budget_bytes\":" << m.window_expert_union_bytes
            << ",\n  \"mtp_window_expert_union_workspace_reserved_bytes\":" << parents->window_setup.union_reserved_bytes
            << ",\n  \"mtp_window_expert_union_workspace_generation\":" << parents->window_setup.union_generation
            << ",\n  \"mtp_window_arena_completion_union_retained_bytes\":" << parents->window_setup.retained_window_bytes
            << ",\n  \"mtp_window_expert_union_latest_observation\":";
        write_mtp_union_observation(union_receipt, union_observation);
        union_receipt << ",\n";
        const auto mtp_d1_inventory=options.execution_inventory
            ? std::string("  \"mtp_target_d1_inventory\":{\"boundary\":\"after-mtp-target-setup-and-request-completion\",\"present\":")+
                (deferred_status?"true":"false")+",\"storage\":\""+(deferred_status?"unaccounted":"absent")+
                "\",\"included_in_base_runtime_backing_inventory\":false,\"physical_fit\":null,"
                "\"scope\":\"actual-target-private-D1-record-and-host-shadow-presence-not-allocation-census\"},\n"
            : std::string{};
        output<<"{\n"<<(completion_receipt?completion_receipt():std::string{})<<mtp_d1_inventory<<union_receipt.str()<<"  \"mode\":\"native-mtp-independent-replay-then-target-verified-greedy\",\n"
            <<"  \"mtp_native_hardware_qualified\":false,\n  \"mtp_performance_qualified\":false,\n"
            <<"  \"optimization_profile_selected\":\""<<decode_receipt::escape_json_string(resolved_profile)<<"\",\n"
            <<"  \"mtp_target_verification\":\""<<mtp_target_verification_name(window_attempts)<<"\",\n"
            <<"  \"mtp_target_window_configured\":"<<(window?"true":"false")<<",\n"
            <<"  \"mtp_requested_target_window_rows\":"<<m.window_rows<<",\n"
            <<"  \"mtp_reference_independence\":\""<<(golden.independent_source_content==mtp_self_generated_golden_source()?
                "self-generated-native-forward-not-independent":"externally-supplied-provenance-pinned-not-inferred-from-digest")<<"\",\n"
            <<"  \"mtp_golden_file_sha256\":\""<<model_content_identity_hex(mtp_cli_digest(m.golden_sha256))<<"\",\n"
            <<"  \"mtp_golden_payload_sha256\":\""<<model_content_identity_hex(replay.golden_payload)<<"\",\n"
            <<"  \"mtp_independent_source_content\":\""<<model_content_identity_hex(golden.independent_source_content)<<"\",\n"
            <<"  \"mtp_independent_reference_identity\":"<<golden.independent_reference_identity<<",\n"
            <<"  \"mtp_sidecar_file_sha256\":\""<<model_content_identity_hex(assets->sidecar_mapping()->file_sha256())<<"\",\n"
            <<"  \"mtp_sidecar_content_identity\":\""<<model_content_identity_hex(assets->shared_target().sidecar_model_content_identity)<<"\",\n"
            <<"  \"mtp_target_file_sha256\":\""<<model_content_identity_hex(actual.mapped->file_sha256())<<"\",\n"
            <<"  \"mtp_target_content_identity\":\""<<model_content_identity_hex(assets->shared_target().target_model_content_identity)<<"\",\n"
            <<"  \"mtp_source_identity\":"<<assets->source().source_identity<<",\n"
            <<"  \"mtp_source_repository\":\""<<decode_receipt::escape_json_string(m.repository)<<"\",\n"
            <<"  \"mtp_source_revision\":\""<<decode_receipt::escape_json_string(m.revision)<<"\",\n"
            <<"  \"mtp_source_path\":\""<<decode_receipt::escape_json_string(m.source_path)<<"\",\n"
            <<"  \"mtp_converter_reference_sha256\":\""<<model_content_identity_hex(evidence.converter_reference_sha256)<<"\",\n"
            <<"  \"mtp_norm_converter_reference_sha256\":\""<<model_content_identity_hex(evidence.norm_converter_reference_sha256)<<"\",\n"
            <<"  \"target_prefill_numerical_profile\":\""<<numerical_profile_name(pair.prefill)<<"\",\n"
            <<"  \"target_decode_numerical_profile\":\""<<numerical_profile_name(pair.decode)<<"\",\n"
            <<"  \"target_prefill_execution_identity\":"<<(split?parents->prompt.compiled->execution_identity:compiled.execution_identity)<<",\n"
            <<"  \"mtp_golden_target_profile_pair_recorded\":"<<(golden.target_decode_profile_identity?"true":"false")<<",\n"
            <<"  \"target_execution_identity\":"<<compiled.execution_identity<<",\n"
            <<"  \"target_kernel_plan_identity\":"<<compiled.kernel_plan_identity<<",\n"
            <<"  \"mtp_structural_plan_identity\":"<<assets->compiled().plan_identity<<",\n"
            <<"  \"mtp_semantic_identity\":"<<assets->compiled().forward_semantics.contract_identity<<",\n"
            <<"  \"mtp_native_arithmetic_profile_identity\":"<<runtime->numerical_profile_identity()<<",\n"
            <<"  \"mtp_qualification_identity\":"<<qualified.qualification_identity<<",\n  \"mtp_activation_identity\":"<<activated.activation_identity<<",\n"
            <<"  \"mtp_replay_policy_identity\":"<<policy.policy_identity<<",\n  \"mtp_replay_requires_exact_tokens\":true,\n"
            <<"  \"mtp_replay_hidden_error_limit\":"<<policy.max_hidden_abs_error<<",\n  \"mtp_replay_logit_error_limit\":"<<policy.max_logit_abs_error<<",\n"
            <<"  \"mtp_replay_maximum_hidden_abs_error\":"<<replay.observation.maximum_hidden_abs_error<<",\n"
            <<"  \"mtp_replay_maximum_logit_abs_error\":"<<replay.observation.maximum_logit_abs_error<<",\n"
            <<"  \"mtp_replay_rows\":"<<replay.rows.size()<<",\n  \"mtp_replay_passed_declared_policy\":true,\n"
            <<"  \"mtp_replay_native_steps\":"<<mtp_cli_mul(replay.rows.size(),assets->compiled().steps.size())<<",\n"
            <<"  \"mtp_requested_candidate_depth\":"<<m.candidate_depth<<",\n  \"mtp_target_capture_rows\":"<<retained_rows<<",\n"
            <<"  \"mtp_normalized_rounding\":\""<<m.normalized_rounding<<"\",\n  \"mtp_projection_rounding\":\""<<m.projection_rounding<<"\",\n"
            <<"  \"mtp_runtime_retained_payload_bytes\":"<<native->retained_payload_bytes()<<",\n"
            <<"  \"mtp_head_recipe\":"<<static_cast<std::uint32_t>(select_hip_mtp_lm_head_recipe(model,native->tensors().lm_head))<<",\n"
            <<"  \"mtp_head_profile_identity\":"<<hip_mtp_lm_head_profile_identity(model,native->tensors().lm_head)<<",\n"
            <<"  \"mtp_native_full_forward_rows\":"<<native_stats.full_forward_rows<<",\n"
            <<"  \"mtp_cache_only_rows\":"<<native_stats.cache_only_rows<<",\n"
            <<"  \"mtp_reused_cache_rows\":"<<native_stats.cache_reused_rows<<",\n"
            <<"  \"mtp_native_row_counter_scope\":\"completed-rows-including-prompt-independent-replay-and-generation\",\n"
            <<"  \"mtp_window_retained_payload_bytes\":"<<(window?window->retained_payload_bytes():0U)<<",\n"
            <<"  \"mtp_target_capture_budget_bytes\":"<<m.target_capture_bytes<<",\n  \"mtp_target_d1\":"<<(deferred_status?"true":"false")<<",\n"
            <<"  \"mtp_cycles\":"<<cycles<<",\n  \"mtp_proposed_candidates\":"<<proposed<<",\n  \"mtp_accepted_candidates\":"<<accepted<<",\n"
            <<"  \"mtp_consumed_generation_input_rows\":"<<target_rows<<",\n  \"mtp_attempted_target_windows\":"<<window_attempts<<",\n"
            <<"  \"mtp_accepted_target_windows\":"<<window_accepts<<",\n  \"mtp_restored_target_windows\":"<<window_restores<<",\n"
            <<"  \"mtp_target_r2_dispatches\":"<<r2_dispatches<<",\n"
            <<"  \"mtp_target_r2_dispatches_scope\":\"actual-pair-entry-dispatches-only-excludes-wide-R4-R8\",\n"
            <<"  \"mtp_target_wide_dispatches\":null,\n"
            <<"  \"mtp_target_wide_dispatches_scope\":\"not-exposed-by-current-verifier-result-not-inferred-from-geometry\",\n"
            <<"  \"mtp_target_window_rows_set\":[";
        if(window){const auto stats=window->owner_stats();for(std::size_t s=1;s<stats.size();++s)output<<(s>1?",":"")<<stats[s].rows;}
        output<<"],\n  \"mtp_spec_min_p\":"<<mtp_draft_min_probability(m)<<",\n"
            <<"  \"mtp_window_controller\":\""<<(window && window->survival_controller()?"survival-v1":"min-p")<<"\",\n"
            <<"  \"mtp_window_option_ms_ema\":[";
        if(window){const auto stats=window->option_stats();
            for(std::size_t s=0;s<stats.size();++s)output<<(s?",":"")<<"{\"rows\":"<<stats[s].rows<<",\"active\":"<<stats[s].active
                <<",\"samples\":"<<stats[s].samples<<",\"ema_ms\":"<<stats[s].ms_ema<<"}";}
        output<<"],\n"
            <<"  \"mtp_chosen_window_rows_histogram\":{";
        for(std::size_t t=0,first=1;t<chosen_rows.size();++t)if(chosen_rows[t]){output<<(first?"":",")<<"\""<<t<<"\":"<<chosen_rows[t];first=0;}
        output<<"},\n  \"mtp_chosen_window_rows_scope\":\"verifier-choice-per-cycle-1-is-serial-target-steps\",\n"
            <<"  \"mtp_window_owner_cycle_ms_ema\":{";
        if(window){const auto stats=window->owner_stats();
            for(std::size_t s=0;s<stats.size();++s)output<<(s?",":"")<<"\""<<stats[s].rows<<"\":{\"cycles\":"<<stats[s].cycles<<",\"ema_ms\":"<<stats[s].cycle_ms_ema
                <<",\"probes\":"<<stats[s].probes<<"}";}
        output<<"},\n"
            <<"  \"mtp_window_cost_model\":\""<<(window && !window->robust_cost_model()?"legacy-ema":"robust-median-reprobe-v1")<<"\",\n"
            <<"  \"mtp_draft_chain\":"<<(m.draft_chain?"true":"false")<<",\n"
            <<"  \"mtp_draft_exact_kernels\":"<<(m.draft_exact?"true":"false")<<",\n"
            <<"  \"mtp_draft_q4_head\":"<<(m.draft_q4?"true":"false")<<",\n"
            <<"  \"mtp_draft_head_rows\":"<<draft_stats.rows<<",\n  \"mtp_draft_base_rows\":"<<draft_stats.base_rows<<",\n"
            <<"  \"mtp_draft_context_rows\":"<<draft_stats.context_rows<<",\n  \"mtp_draft_context_dropped\":"<<draft_stats.context_dropped<<",\n"
            <<"  \"mtp_generation_cycle_elapsed_ms\":"<<elapsed<<",\n"
            <<"  \"mtp_timing_scope\":\"post-prompt-and-independent-replay-cycles-only-host-completed-wall-not-GPU-kernel-time\",\n"
            <<(tiled_prompt ? "  \"mtp_tiled_prompt\":{"+tiled_receipt+",\"calibration_rows\":"+std::to_string(calibration_rows)+
                ",\"calibration_ms\":"+std::to_string(calibration_ms)+"},\n" : std::string{})
            <<"  \"prefix_tokens_consumed\":"<<options.tokens.size()<<",\n  \"final_position\":"<<session.next_position<<",\n"
            <<"  \"stop_reason\":\""<<stop<<"\",\n  \"generated_tokens\":[";
        for(std::size_t i=0;i<generated.size();++i){if(i)output<<',';output<<generated[i];}
        output<<"]\n}\n";
    } catch(...) {if(runtime)runtime->cancel();throw;}
}
} // namespace strix::decode_cli
