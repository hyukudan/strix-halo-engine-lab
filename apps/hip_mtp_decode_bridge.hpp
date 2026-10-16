#pragma once
#include "hip_decode_experiment.hpp"
#include "hip_decode_options.hpp"
#include "hip_decode_receipt_json.hpp"
#include "strix/hip_mtp_asset_loader.hpp"
#include "strix/hip_launcher_catalog.hpp"
#include "strix/hip_mtp_forward_replay.hpp"
#include "strix/mtp_forward_golden_file.hpp"
#include "strix/hip_mtp_greedy_runtime.hpp"
#include "strix/hip_mtp_window_verifier.hpp"
#include "strix/mtp_draft_activation.hpp"
#include "strix/mtp_draft_qualification.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/ple_lookup.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <ostream>
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
inline ModelContentIdentity mtp_cli_digest(std::string_view text) {
    if(!mtp_sha256_text(text))throw std::invalid_argument("MTP SHA256 pin is invalid");
    const auto digit=[](char ch)->unsigned {if(ch>='0'&&ch<='9')return static_cast<unsigned>(ch-'0');
        if(ch>='a'&&ch<='f')return static_cast<unsigned>(ch-'a')+10U;return static_cast<unsigned>(ch-'A')+10U;};
    ModelContentIdentity out;
    for(std::size_t i=0;i<out.bytes.size();++i)out.bytes[i]=static_cast<std::uint8_t>((digit(text[i*2U])<<4U)|digit(text[i*2U+1U]));
    return out;
}
inline std::uint64_t mtp_cli_add(std::uint64_t a,std::uint64_t b) {
    if(b>UINT64_MAX-a)throw std::overflow_error("MTP CLI payload budget overflow");return a+b;
}
inline std::uint64_t mtp_cli_mul(std::uint64_t a,std::uint64_t b) {
    if(b && a>UINT64_MAX/b)throw std::overflow_error("MTP CLI payload budget overflow");return a*b;
}
using MtpDecodeWindowFactory=std::function<std::shared_ptr<HipMtpWindowVerifier>(std::shared_ptr<HipMtpTargetHidden>)>;

// One synchronous CLI request with externally pinned evidence. Setup/loading,
// independent replay, activation and generation have separate receipt scopes.
// There is no path from command-line booleans to a qualified/activated plan.
inline void run_mtp_decode_bridge(const Options& options,const ModelContract& model,
    const CompiledSerialExecution& compiled,const std::shared_ptr<MtpDecodeTargetParents>& parents,
    SerialSessionContext& session,DirectLaunchDispatcher& dispatcher,AllocationGenerationSource& generations,
    bool deferred_status,std::string_view resolved_profile,std::ostream& output,
    MtpDecodeWindowFactory make_window={},std::function<std::string()> completion_receipt={}) {
    validate_resolved_mtp_cli(options);
    if(!options.mtp.enabled || !parents || session.next_position || session.committed_tokens ||
       session.health!=SerialSessionHealth::ready || session.execution_identity!=compiled.execution_identity)
        throw std::invalid_argument("MTP CLI requires a fresh actual target session");
    const auto& m=options.mtp;const auto& r=parents->resources;const auto& actual=*parents->experiment;
    auto golden=read_mtp_forward_golden_file(m.golden,mtp_cli_digest(m.golden_sha256),m.golden_max_bytes);
    if(golden.rows.size()+1U!=options.tokens.size())
        throw std::invalid_argument("MTP golden must cover every prompt predictor pair, leaving only its final pending target row");
    const auto width=mtp_cli_mul(model.hidden_size,model.hc_count);
    for(std::size_t i=0;i<golden.rows.size();++i) {
        const auto& row=golden.rows[i];
        if(row.source_position!=i || row.source_input_token!=options.tokens[i] || row.next_input_token!=options.tokens[i+1U] ||
           row.target_hidden_bf16.size()!=width || row.next_hidden_bf16.size()!=width || row.full_logits.size()!=model.vocab_size)
            throw std::invalid_argument("MTP independent golden prompt chain or full-row geometry differs before target execution");
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
    MtpForwardReplayPolicy policy{m.policy_identity,m.max_hidden_error,m.max_logit_error,2U,true};
    (void)assess_mtp_forward_replay(policy,{});
    if(golden.target_content!=assets->shared_target().target_model_content_identity ||
       golden.sidecar_content!=assets->shared_target().sidecar_model_content_identity ||
       golden.structural_plan_identity!=assets->compiled().plan_identity ||
       golden.semantic_identity!=assets->compiled().forward_semantics.contract_identity ||
       golden.numerical_profile_identity!=runtime->numerical_profile_identity())
        throw std::invalid_argument("MTP golden does not bind actual verified assets, semantics and native cast profile");
    // Same complete known payload bound as the replay consumer; admit it before
    // the target prompt is mutated, rather than waiting until replay entry.
    auto replay_payload=mtp_cli_add(mtp_cli_mul(width,4U),mtp_cli_mul(model.vocab_size,4U));
    replay_payload=mtp_cli_add(replay_payload,mtp_cli_mul(golden.rows.size(),sizeof(HipMtpForwardReplayRowReceipt)+2U*sizeof(std::uint32_t)));
    replay_payload=mtp_cli_add(replay_payload,mtp_cli_mul(golden.rows.size(),mtp_cli_add(mtp_cli_mul(width,4U),mtp_cli_mul(model.vocab_size,4U))));
    replay_payload=mtp_cli_add(replay_payload,mtp_cli_mul(golden.rows.size()+1U,192U));
    if(replay_payload>m.replay_host_bytes)throw std::invalid_argument("MTP complete independent replay exceeds host payload budget before target mutation");
    HipMtpForwardReplayRequest replay_request{policy,hip_mtp_forward_golden_payload_identity(golden),golden.rows.size(),m.replay_host_bytes};
    const auto retained_rows=static_cast<std::uint32_t>(std::max<std::size_t>(options.tokens.size(),static_cast<std::size_t>(m.candidate_depth)+3U));
    HipMtpTargetDeferredConfig d1;
    if(deferred_status)d1={r.session_state_layout,&generations};
    auto target=HipMtpTargetHidden::create(model,compiled,actual.memory,actual.model,actual.allocator,
        retained_rows,m.target_capture_bytes,d1);
    std::shared_ptr<HipMtpWindowVerifier> window;
    if(m.window_rows) {
        if(!make_window)throw std::invalid_argument("MTP requested target window has no actual prepared native factory");
        window=make_window(target);
        if(!window || window->target_owner()!=target || window->logical_rows()!=m.window_rows)
            throw std::invalid_argument("MTP target window factory returned incompatible actual owner");
    }
    assets->validate_assets_unchanged();
    std::vector<HipMtpReplayRow> replay_rows;replay_rows.reserve(golden.rows.size());
    std::vector<std::uint32_t> generated;generated.reserve(static_cast<std::size_t>(options.generate_tokens));
    std::uint64_t cycles=0,proposed=0,accepted=0,target_rows=0,window_attempts=0,window_accepts=0,window_restores=0,r2_dispatches=0;
    HipMtpTargetHiddenTicket pending;
    HipMtpNativeForwardReplayResult replay;
    QualifiedMtpDraftPlan qualified;ActivatedMtpDraftPlan activated;
    const std::array<std::uint32_t,1> eos{options.eos_token.value_or(static_cast<std::uint32_t>(model.ple_eos_token_id))};
    std::string stop="token-budget";
    try {
        for(std::size_t i=0;i<options.tokens.size();++i) {
            auto current=target->execute_serial(session,r,dispatcher,options.tokens[i]);
            if(pending.valid())replay_rows.push_back({pending,options.tokens[i]});
            pending=std::move(current);
        }
        // This call executes all eleven native stages for every independent
        // reference pair. The resulting real observation is the sole evidence.
        replay=replay_hip_mtp_forward(*runtime,replay_rows,golden,replay_request);
        if(!replay.assessment.passes_declared_policy)
            throw std::runtime_error("MTP actual full-hidden/full-vocabulary replay failed its declared policy; generation is not admitted");
        qualified=qualify_mtp_draft_plan(assets->compiled(),policy,replay.observation);
        require_qualified_mtp_draft_plan(qualified);
        activated=activate_mtp_draft_plan(assets->compiled(),policy,replay.observation);
        if(runtime->next_source_position()+1U!=session.next_position)
            throw std::logic_error("MTP replay did not prime the actual final prompt frontier");
        replay_rows.clear(); // release prompt snapshots before bounded cycles
        HipMtpGreedyRuntime greedy(runtime,qualified,activated,target,session,r,dispatcher,parents);
        if(window)greedy.use_window_verifier(window);
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
        }
        const auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(generated.back()==eos.front())stop="eos";
        assets->validate_assets_unchanged();
        const auto mtp_d1_inventory=options.execution_inventory
            ? std::string("  \"mtp_target_d1_inventory\":{\"boundary\":\"after-mtp-target-setup-and-request-completion\",\"present\":")+
                (deferred_status?"true":"false")+",\"storage\":\""+(deferred_status?"unaccounted":"absent")+
                "\",\"included_in_base_runtime_backing_inventory\":false,\"physical_fit\":null,"
                "\"scope\":\"actual-target-private-D1-record-and-host-shadow-presence-not-allocation-census\"},\n"
            : std::string{};
        output<<"{\n"<<(completion_receipt?completion_receipt():std::string{})<<mtp_d1_inventory<<"  \"mode\":\"native-mtp-independent-replay-then-target-verified-greedy\",\n"
            <<"  \"mtp_native_hardware_qualified\":false,\n  \"mtp_performance_qualified\":false,\n"
            <<"  \"optimization_profile_selected\":\""<<decode_receipt::escape_json_string(resolved_profile)<<"\",\n"
            <<"  \"mtp_target_verification\":\""<<mtp_target_verification_name(window_attempts)<<"\",\n"
            <<"  \"mtp_target_window_configured\":"<<(window?"true":"false")<<",\n"
            <<"  \"mtp_requested_target_window_rows\":"<<m.window_rows<<",\n"
            <<"  \"mtp_reference_independence\":\"externally-supplied-provenance-pinned-not-inferred-from-digest\",\n"
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
            <<"  \"mtp_target_capture_budget_bytes\":"<<m.target_capture_bytes<<",\n  \"mtp_target_d1\":"<<(deferred_status?"true":"false")<<",\n"
            <<"  \"mtp_cycles\":"<<cycles<<",\n  \"mtp_proposed_candidates\":"<<proposed<<",\n  \"mtp_accepted_candidates\":"<<accepted<<",\n"
            <<"  \"mtp_consumed_generation_input_rows\":"<<target_rows<<",\n  \"mtp_attempted_target_windows\":"<<window_attempts<<",\n"
            <<"  \"mtp_accepted_target_windows\":"<<window_accepts<<",\n  \"mtp_restored_target_windows\":"<<window_restores<<",\n"
            <<"  \"mtp_target_r2_dispatches\":"<<r2_dispatches<<",\n"
            <<"  \"mtp_target_r2_dispatches_scope\":\"actual-pair-entry-dispatches-only-excludes-wide-R4-R8\",\n"
            <<"  \"mtp_target_wide_dispatches\":null,\n"
            <<"  \"mtp_target_wide_dispatches_scope\":\"not-exposed-by-current-verifier-result-not-inferred-from-geometry\",\n"
            <<"  \"mtp_generation_cycle_elapsed_ms\":"<<elapsed<<",\n"
            <<"  \"mtp_timing_scope\":\"post-prompt-and-independent-replay-cycles-only-host-completed-wall-not-GPU-kernel-time\",\n"
            <<"  \"prefix_tokens_consumed\":"<<options.tokens.size()<<",\n  \"final_position\":"<<session.next_position<<",\n"
            <<"  \"stop_reason\":\""<<stop<<"\",\n  \"generated_tokens\":[";
        for(std::size_t i=0;i<generated.size();++i){if(i)output<<',';output<<generated[i];}
        output<<"]\n}\n";
    } catch(...) {runtime->cancel();throw;}
}
} // namespace strix::decode_cli
