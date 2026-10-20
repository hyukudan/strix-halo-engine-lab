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
#include <span>
#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <ostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <utility>
#include <cstdlib>

namespace strix::decode_cli {
// Serving (--http-serve): after the calibration replay the bridge hands a request runner to the
// serve hook, which calls it for one request at a time on the bridge's thread until it returns.
// Each request starts from a fresh target state (restored from the prefix state cache when a
// cached block boundary matches) and decodes greedily with MTP verification.
struct MtpServeRequest {
    std::span<const std::uint32_t> prompt;
    std::uint64_t max_tokens = 0;
    std::span<const std::uint32_t> stop_tokens;
    // Newly verified tokens after each cycle (the first call carries the prompt's own prediction);
    // false stops the request.
    std::function<bool(std::span<const std::uint32_t>)> on_tokens;
};
struct MtpServeResult {
    std::uint64_t restored_tokens = 0, generated_tokens = 0, cycles = 0, proposed = 0, accepted = 0;
    double prompt_ms = 0.0, decode_ms = 0.0;
    std::string finish = "length"; // stop|length|context|cancelled
};
using MtpServeRunner = std::function<MtpServeResult(const MtpServeRequest&)>;
using MtpDecodeServe = std::function<void(const MtpServeRunner&)>;
class MtpBatchRunner;
struct MtpBatchSlotSetup;
// Batched decoding (--http-batch, cohort requests on the tiled prompt): after the calibration replay the
// bridge builds one slot per request it can hold and hands the runner to this hook.
using MtpDecodeBatchServe = std::function<void(MtpBatchRunner&)>;
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
        // Prompt groups (STRIX_PROMPT_BATCH): the owner the blocks of several requests run on, when it is not the
        // T-row block owner (STRIX_PROMPT_BATCH_ROWS above T). Null: the blocks backend and its owner.
        HipKnownPromptBackend* group = nullptr;
        std::shared_ptr<HipKnownPrefixWindow> group_owner;
    } tiled;
    // Set: serve requests after the calibration replay instead of the command-line request.
    MtpDecodeServe serve;
    // Set: batched decoding after the calibration replay. This request's state is slot 0; batch_slots are
    // the further slots (their own parents, session and dispatcher, window owners on a shared stream).
    MtpDecodeBatchServe serve_batch;
    std::vector<MtpBatchSlotSetup> batch_slots;
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
// Sampling receipt: the policy, its identity and the decision counts of this request.
inline std::string mtp_sampling_receipt(const MtpCliOptions& m,const TokenSampler* sampler) {
    std::ostringstream out;out<<std::setprecision(9);
    if(!sampler){out<<"{\"identity\":\"greedy-first-max\"}";return out.str();}
    const auto& p=sampler->policy();const auto& s=sampler->stats();
    out<<"{\"identity\":\"sampling-v1\",\"temperature\":"<<p.temperature<<",\"top_k\":"<<p.top_k<<",\"top_p\":"<<p.top_p
       <<",\"min_p\":"<<p.min_p<<",\"seed\":"<<p.seed<<",\"policy_identity\":"<<token_sampling_policy_identity(p)
       <<",\"speculation\":"<<(m.sample_speculation?"true":"false")<<",\"schedule\":\""<<(m.sample_timing_schedule?"adaptive-timed":"fixed")
       <<"\",\"drafts\":"<<s.drafts<<",\"plain_decisions\":"<<s.plain<<",\"accepted_drafts\":"<<s.accepted
       <<",\"rejected_drafts\":"<<s.rejected<<"}";
    return out.str();
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
// The tiled MTP prompt as resumable steps (one T-row block, one tail block or up to `serial_rows_per_step`
// serial rows per step), so that a batched run can interleave a joining request's prompt with the decode cycles
// of the others. The sequence of operations is run_tiled_mtp_prompt's, whatever the interleaving.
class TiledMtpPrompt {
public:
    TiledMtpPrompt(std::span<const std::uint32_t> tokens,const ModelContract& model,const CompiledSerialExecution& compiled,
        MtpDecodeTargetParents::TiledPrompt& tiled,SerialSessionContext& session,HipMtpTargetHidden& target,HipMtpRuntime& runtime,
        const QualifiedMtpDraftPlan& qualified,const ActivatedMtpDraftPlan& activated,const ExecutionRuntimeResources& resources,
        DirectLaunchDispatcher& dispatcher,std::byte* held,std::shared_ptr<TokenSampler> final_sampler={})
        :tokens_(tokens.begin(),tokens.end()),tiled_(tiled),session_(session),target_(target),runtime_(runtime),qualified_(qualified),
         activated_(activated),resources_(resources),dispatcher_(dispatcher),held_(held),final_sampler_(std::move(final_sampler)) {
        n_=static_cast<std::uint64_t>(tokens_.size());
        if(!n_)throw std::invalid_argument("tiled MTP prompt is empty");
        // Sampled decoding: only the final prompt row selects through the sampler (earlier rows' predictions
        // are never emitted), so the target holds it from that row on.
        if(final_sampler_ && target_.token_sampler()) throw std::invalid_argument("tiled MTP prompt sampler is set on the final row only");
        width_=mtp_cli_mul(mtp_cli_mul(model.hidden_size,model.hc_count),2U);
        // The target tap: the last MLP injection before the final hyper mix (same step in every sibling).
        for(std::size_t i=1;i<compiled.execution.steps.size();++i)
            if(compiled.execution.steps[i].op==SerialOp::final_hyper_mix && compiled.execution.steps[i-1U].op==SerialOp::mlp_inject) tap_=i-1U;
        if(tap_==SIZE_MAX)throw std::logic_error("tiled MTP prompt has no wide hidden tap step");
        check(hipMalloc(reinterpret_cast<void**>(&block_hidden_),static_cast<std::size_t>(mtp_cli_mul(tiled_.rows,width_))),"tiled MTP hidden rows");
        start_=std::chrono::steady_clock::now();
        if(tiled_.cache) restored_=tiled_.cache->begin_request(session_,tokens_);
        pos_=restored_;
    }
    ~TiledMtpPrompt(){if(block_hidden_)(void)hipFree(block_hidden_);}
    TiledMtpPrompt(const TiledMtpPrompt&)=delete;
    TiledMtpPrompt& operator=(const TiledMtpPrompt&)=delete;
    [[nodiscard]] bool done() const noexcept {return phase_==Phase::done;}
    [[nodiscard]] std::uint64_t position() const noexcept {return pos_;}
    [[nodiscard]] std::uint64_t restored() const noexcept {return restored_;}
    // The prompt's next unit when it is a tiled block (a T-row block or a tail block): its backend, owner, rows,
    // whether it is a T block (prefix cache capture) and whether it ends the prompt. Null when the next unit is
    // serial rows or the prompt is done. Only advances past exhausted block phases, as step() would.
    struct Block {
        HipKnownPromptBackend* backend=nullptr;
        HipKnownPrefixWindow* owner=nullptr;
        std::uint32_t rows=0;
        bool capture=false,final=false;
    };
    [[nodiscard]] std::optional<Block> next_block() {
        while(phase_==Phase::blocks || phase_==Phase::tails) {
            const bool blocks=phase_==Phase::blocks;
            if(!blocks && tail_>=tiled_.tails.size()){phase_=Phase::serial;break;}
            auto& backend=blocks?*tiled_.blocks:*tiled_.tails[tail_].backend;
            auto& owner=blocks?*tiled_.block_owner:*tiled_.tails[tail_].owner;
            const auto rows=blocks?tiled_.rows:tiled_.tails[tail_].rows;
            // STRIX_MTP_TILED_START=0: the last prompt row always runs serially (the previous handoff).
            // Otherwise a block that ends exactly at the last prompt row also computes that row's head, and
            // the first draft cycle starts from the block's last row (its tap and FP32 logits).
            static const bool tiled_start=[]{const char* v=std::getenv("STRIX_MTP_TILED_START");return !v || v[0]!='0';}();
            const bool heads=tiled_start && backend.output_policy()!=KnownWindowOutputPolicy::state_only;
            if(!adopted_ && (pos_+rows<=n_-1U || (heads && pos_+rows==n_))) return Block{&backend,&owner,rows,blocks,pos_+rows==n_};
            if(blocks)phase_=Phase::tails;else ++tail_;
        }
        return std::nullopt;
    }
    // The block's tokens (valid until the prompt is destroyed).
    [[nodiscard]] std::span<const std::uint32_t> block_tokens(const Block& block) const {
        return std::span<const std::uint32_t>(tokens_).subspan(static_cast<std::size_t>(pos_),block.rows);
    }
    // The step observer of a block: copies every row's wide hidden state (the MTP target tap) out of the arena.
    [[nodiscard]] HipKnownPrefixWindow::PrefillStepObserver block_observer(std::uint32_t rows) {
        return [this,rows](std::size_t step,const std::function<ExecutionDispatchContext(std::uint32_t)>& row) {
            if(step!=tap_) return;
            const auto stream=hip_execution_stream();
            for(std::uint32_t r=0;r<rows;++r) {
                const auto c=row(r);
                const auto source=resolve_dispatch_write(c,0);
                if(source.bytes<width_)throw std::logic_error("tiled MTP tap row is undersized");
                check(hipMemcpyAsync(block_hidden_+static_cast<std::size_t>(r)*width_,reinterpret_cast<const void*>(source.address),
                    static_cast<std::size_t>(width_),hipMemcpyDeviceToDevice,stream),"tiled MTP tap copy");
            }
        };
    }
    // After the block ran (alone or in a group): the predictor rows, the held row, the prefix cache capture and,
    // for the final block, the adoption of its last row (logits: its terminal logits).
    void finish_block(const Block& block,const std::vector<float>& logits,bool grouped=false) {
        const auto rows=block.rows;
        std::vector<MtpDeviceView> views;std::vector<std::uint32_t> next;
        if(pos_){views.push_back(view(held_));next.push_back(tokens_[static_cast<std::size_t>(pos_)]);}
        for(std::uint32_t r=0;r+1U<rows;++r){views.push_back(view(block_hidden_+static_cast<std::size_t>(r)*width_));
            next.push_back(tokens_[static_cast<std::size_t>(pos_+r+1U)]);}
        runtime_.prime_hidden_rows(qualified_,activated_,views,next);
        check(hipMemcpy(held_,block_hidden_+static_cast<std::size_t>(rows-1U)*width_,static_cast<std::size_t>(width_),hipMemcpyDeviceToDevice),
            "tiled MTP held copy");
        pos_+=rows;
        if(block.capture && tiled_.cache) tiled_.cache->after_block(session_,tokens_,n_-1U);
        if(block.capture)++blocks_;else ++tail_blocks_;
        if(grouped)++grouped_blocks_;
        if(block.final) {
            // STRIX_PREFIX_CACHE_DIGEST=1: also the SHA-256 of the final row's FP32 logits in the receipt.
            static const bool digest=[]{const char* v=std::getenv("STRIX_PREFIX_CACHE_DIGEST");return v && v[0]=='1';}();
            if(digest) {
                const auto bytes=std::as_bytes(std::span<const float>(logits));
                final_logits_digest_=sha256_bytes(std::vector<std::byte>(bytes.begin(),bytes.end()));
            }
            if(final_sampler_) target_.replace_token_sampler(final_sampler_);
            pending_=target_.adopt_prefill_row(session_,tokens_[static_cast<std::size_t>(n_-1U)],view(held_),logits);
            adopted_=true;
            phase_=Phase::done;
        }
    }
    // Runs the next unit of the prompt.
    void step(std::uint32_t serial_rows_per_step=UINT32_MAX) {
        for(;;) {
            if(phase_==Phase::blocks || phase_==Phase::tails) {
                if(const auto block=next_block()) {run_block(*block);return;}
                continue;
            }
            if(phase_==Phase::serial) {
                if(pos_>=n_){phase_=Phase::done;return;}
                for(std::uint32_t k=0;k<serial_rows_per_step && pos_<n_;++k) run_serial_row();
                if(pos_>=n_)phase_=Phase::done;
                return;
            }
            return;
        }
    }
    HipMtpTargetHiddenTicket finish(std::string& receipt) {
        if(!done())throw std::logic_error("tiled MTP prompt finished before its last row");
        if(runtime_.next_source_position()+1U!=n_ || session_.next_position!=n_)
            throw std::logic_error("tiled MTP prompt did not reach the final prompt frontier");
        const auto prompt_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start_).count();
        std::ostringstream out;
        out<<"\"rows\":"<<tiled_.rows<<",\"tail_rows\":[";
        for(std::size_t t=0;t<tiled_.tails.size();++t)out<<(t?",":"")<<tiled_.tails[t].rows;
        out<<"],\"restored_position\":"<<restored_
           <<",\"blocks\":"<<blocks_<<",\"tail_blocks\":"<<tail_blocks_<<",\"serial_rows\":"<<serial_rows_<<",\"tiled_start\":"<<(adopted_?"true":"false")
           <<",\"prompt_ms\":"<<prompt_ms<<",\"grouped_blocks\":"<<grouped_blocks_;
        if(final_logits_digest_) out<<",\"final_logits_digest\":\""<<model_content_identity_hex(*final_logits_digest_)<<'"';
        if(tiled_.cache) {
            const auto& q=tiled_.cache->request();
            out<<",\"prefix_cache\":{\"lookup_ms\":"<<q.lookup_ms<<",\"restore_ms\":"<<q.restore_ms<<",\"restored_bytes\":"<<q.restored_bytes
               <<",\"captured_checkpoints\":"<<q.captured_checkpoints<<",\"captured_segment_bytes\":"<<q.captured_segment_bytes
               <<",\"captured_fixed_bytes\":"<<q.captured_fixed_bytes<<",\"capture_ms\":"<<q.capture_ms
               <<",\"row_bytes_per_position\":"<<tiled_.cache->row_bytes()<<",\"fixed_bytes_per_checkpoint\":"<<tiled_.cache->fixed_bytes()<<'}';
            static const bool digest=[]{const char* v=std::getenv("STRIX_PREFIX_CACHE_DIGEST");return v && v[0]=='1';}();
            if(digest) out<<",\"state_digest\":\""<<model_content_identity_hex(tiled_.cache->state_digest(n_))<<'"';
        }
        receipt=out.str();
        return std::move(pending_);
    }
private:
    enum class Phase {blocks,tails,serial,done};
    static void check(hipError_t e,const char* what){if(e!=hipSuccess)throw std::runtime_error(std::string(what)+": "+hipGetErrorString(e));}
    [[nodiscard]] MtpDeviceView view(const std::byte* p) const {return MtpDeviceView{reinterpret_cast<std::uintptr_t>(p),width_,1U};}
    void run_block(const Block& block) {
        auto& owner=*block.owner;
        owner.set_prefill_step_observer(block_observer(block.rows));
        try {
            std::vector<float> logits;
            if(block.final) logits=block.backend->execute_rows(block_tokens(block),pos_);
            else block.backend->execute_state_rows(block_tokens(block),pos_);
            owner.set_prefill_step_observer({});
            finish_block(block,logits);
        } catch(...) {owner.set_prefill_step_observer({});throw;}
    }
    void run_serial_row() {
        if(final_sampler_ && pos_+1U==n_) target_.replace_token_sampler(final_sampler_);
        auto ticket=target_.execute_serial(session_,resources_,dispatcher_,tokens_[static_cast<std::size_t>(pos_)]);
        if(pos_) {
            const std::array<MtpDeviceView,1> one{view(held_)};const std::array<std::uint32_t,1> token{tokens_[static_cast<std::size_t>(pos_)]};
            runtime_.prime_hidden_rows(qualified_,activated_,one,token);
        }
        check(hipMemcpy(held_,reinterpret_cast<const void*>(ticket.view().address),static_cast<std::size_t>(width_),hipMemcpyDeviceToDevice),
            "tiled MTP held ticket copy");
        pending_=std::move(ticket);++pos_;++serial_rows_;
    }
    std::vector<std::uint32_t> tokens_;
    MtpDecodeTargetParents::TiledPrompt& tiled_;
    SerialSessionContext& session_;
    HipMtpTargetHidden& target_;
    HipMtpRuntime& runtime_;
    const QualifiedMtpDraftPlan& qualified_;
    const ActivatedMtpDraftPlan& activated_;
    const ExecutionRuntimeResources& resources_;
    DirectLaunchDispatcher& dispatcher_;
    std::byte* held_=nullptr;
    std::shared_ptr<TokenSampler> final_sampler_;
    std::uint64_t n_=0,width_=0,restored_=0,pos_=0,blocks_=0,tail_blocks_=0,serial_rows_=0,grouped_blocks_=0;
    std::optional<ModelContentIdentity> final_logits_digest_;
    std::size_t tap_=SIZE_MAX,tail_=0;
    std::byte* block_hidden_=nullptr;
    Phase phase_=Phase::blocks;
    bool adopted_=false;
    HipMtpTargetHiddenTicket pending_;
    std::chrono::steady_clock::time_point start_;
};

// Tiled MTP prompt after the calibration replay, in one call (see TiledMtpPrompt).
inline HipMtpTargetHiddenTicket run_tiled_mtp_prompt(std::span<const std::uint32_t> tokens,const ModelContract& model,
    const CompiledSerialExecution& compiled,MtpDecodeTargetParents::TiledPrompt& tiled,SerialSessionContext& session,
    HipMtpTargetHidden& target,HipMtpRuntime& runtime,const QualifiedMtpDraftPlan& qualified,const ActivatedMtpDraftPlan& activated,
    const ExecutionRuntimeResources& resources,DirectLaunchDispatcher& dispatcher,std::byte* held,std::string& receipt,
    std::shared_ptr<TokenSampler> final_sampler={}) {
    TiledMtpPrompt prompt(tokens,model,compiled,tiled,session,target,runtime,qualified,activated,resources,dispatcher,held,
        std::move(final_sampler));
    while(!prompt.done()) prompt.step();
    return prompt.finish(receipt);
}

// Batched decoding of independent requests on the tiled MTP route (--http-batch, cohort requests with the
// tiled prompt). Every slot owns what a single request owns: target state and session, launch plan, PLE
// backend, tiled prefill backends (the prefill arena is shared and used by one prompt at a time), prefix
// cache view, predictor, target producer and 2/4/8 window owners on one shared window stream. A request is
// admitted into a free slot between cycles: its prompt runs exactly as a single request's (fresh state,
// tiled blocks, the first token), then it joins the cycles of the others; it leaves when it finishes,
// without stopping them. With two or more active requests a cycle is a cohort cycle
// (hip_mtp_greedy_cohort_cycle); with one, the ordinary cycle.
struct MtpBatchRequest {
    std::vector<std::uint32_t> prompt;
    std::uint64_t max_tokens = 0;
    std::vector<std::uint32_t> stop_tokens;
    std::shared_ptr<TokenSampler> sampler; // null: first-max selection
    // Newly verified tokens (the first call carries the prompt's own prediction); false cancels.
    std::function<bool(std::span<const std::uint32_t>)> on_tokens;
    std::function<void(const MtpServeResult&)> on_finish;
};
struct MtpBatchSlotSetup {
    std::shared_ptr<MtpDecodeTargetParents> parents; // parents->tiled bound to this slot's state and session
    SerialSessionContext* session = nullptr;
    DirectLaunchDispatcher* dispatcher = nullptr;
    MtpDecodeWindowFactory make_window;
};
class MtpBatchRunner {
public:
    struct Slot {
        std::shared_ptr<MtpDecodeTargetParents> parents;
        SerialSessionContext* session = nullptr;
        DirectLaunchDispatcher* dispatcher = nullptr;
        std::shared_ptr<HipMtpResources> native;
        std::shared_ptr<HipMtpRuntime> runtime;
        std::shared_ptr<HipMtpTargetHidden> target;
        std::shared_ptr<HipMtpWindowVerifier> window;
        std::byte* held = nullptr; // h_{P-1} of the tiled prompt
        std::unique_ptr<HipMtpGreedyRuntime> greedy;
        HipMtpTargetHiddenTicket pending;
        bool active = false;     // the slot holds a request (prefilling or decoding)
        bool prefilling = false; // its prompt is still running (interleaved admission)
        std::unique_ptr<TiledMtpPrompt> prefill;
        std::uint64_t order = 0; // admission order: prompts advance oldest first
        std::chrono::steady_clock::time_point prompt_start;
        MtpBatchRequest request;
        MtpServeResult result;
        std::string prompt_receipt;
        std::chrono::steady_clock::time_point decode_start;
        std::array<std::uint64_t, 9> chosen_rows{};
    };
    struct Stats {
        std::uint64_t admitted = 0, finished = 0, steps = 0, cohort_steps = 0, solo_steps = 0, max_active = 0, prompt_steps = 0;
        // Prompt groups (STRIX_PROMPT_BATCH=1): segmented runs of several prompts' blocks, their blocks and rows.
        std::uint64_t prompt_groups = 0, grouped_blocks = 0, grouped_rows = 0;
        double prompt_group_ms = 0.0;
        HipMtpGreedyCohortStats cohort;
    };
    // The calibration replay that qualified the predictor for every slot (set by the bridge).
    struct Calibration {
        std::uint64_t rows = 0;
        double ms = 0.0, maximum_hidden_abs_error = 0.0, maximum_logit_abs_error = 0.0;
    } calibration;
    using EnableDraft = std::function<void(HipMtpRuntime&, std::span<const std::uint32_t>, const std::shared_ptr<TokenSampler>&)>;
    // interleave: a joining request's prompt advances one unit (a tiled block, a tail block or
    // `serial_rows_per_step` serial rows) per step, between the decode cycles of the others; otherwise
    // admit() runs the whole prompt.
    MtpBatchRunner(const Options& options, const ModelContract& model, const CompiledSerialExecution& compiled,
        const QualifiedMtpDraftPlan& qualified, const ActivatedMtpDraftPlan& activated, EnableDraft enable_draft,
        std::vector<std::unique_ptr<Slot>> slots, bool interleave = false, std::uint32_t serial_rows_per_step = 16)
        : options_(options), model_(model), compiled_(compiled), qualified_(qualified), activated_(activated),
          enable_draft_(std::move(enable_draft)), slots_(std::move(slots)), interleave_(interleave),
          serial_rows_per_step_(serial_rows_per_step ? serial_rows_per_step : 1U) {
        if (slots_.empty()) throw std::invalid_argument("batched MTP decoding needs a slot");
    }
    MtpBatchRunner(const MtpBatchRunner&) = delete;
    MtpBatchRunner& operator=(const MtpBatchRunner&) = delete;
    [[nodiscard]] std::size_t slots() const noexcept { return slots_.size(); }
    [[nodiscard]] std::size_t active() const noexcept {
        return static_cast<std::size_t>(std::count_if(slots_.begin(), slots_.end(), [](const auto& s) { return s->active; }));
    }
    [[nodiscard]] bool has_free_slot() const noexcept { return active() < slots_.size(); }
    [[nodiscard]] bool interleaved() const noexcept { return interleave_; }
    [[nodiscard]] static bool prompt_batching() noexcept { return prompt_batch_enabled(); }
    [[nodiscard]] std::size_t prefilling() const noexcept {
        return static_cast<std::size_t>(std::count_if(slots_.begin(), slots_.end(), [](const auto& s) { return s->prefilling; }));
    }
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
    [[nodiscard]] const Slot& slot(std::size_t index) const { return *slots_.at(index); }

    // Runs the request's prompt in a free slot and emits its first token; the request then joins the
    // next step (or finishes at once: stop token, budget of one, cancelled). Returns the slot.
    std::size_t admit(MtpBatchRequest request) {
        const auto free = std::find_if(slots_.begin(), slots_.end(), [](const auto& s) { return !s->active; });
        if (free == slots_.end()) throw std::logic_error("batched MTP admission without a free slot");
        const auto vocabulary = model_.vocab_size;
        if (request.prompt.empty() || request.prompt.size() >= options_.context_tokens || !request.max_tokens || !request.on_tokens)
            throw std::invalid_argument("batched request needs a prompt shorter than the context, a token budget and a sink");
        for (const auto t : request.prompt) if (t >= vocabulary) throw std::invalid_argument("batched prompt token outside the vocabulary");
        for (const auto t : request.stop_tokens) if (t >= vocabulary) throw std::invalid_argument("batched stop token outside the vocabulary");
        auto& s = **free;
        const auto& m = options_.mtp;
        const auto start = std::chrono::steady_clock::now();
        s.request = std::move(request);
        s.result = {};
        s.chosen_rows = {};
        // Exactly the single request's start: fresh target state and session, fresh predictor, no
        // deferred window rows, the request's sampler on the final prompt row and every later row.
        s.greedy.reset();
        s.pending = {};
        s.runtime = std::make_shared<HipMtpRuntime>(s.native);
        s.parents->tiled.reset_state();
        *s.session = s.parents->tiled.fresh;
        if (s.window) s.window->begin_session();
        s.target->replace_token_sampler({});
        enable_draft_(*s.runtime, s.request.prompt, s.request.sampler);
        s.prompt_start = start;
        s.prefill = std::make_unique<TiledMtpPrompt>(s.request.prompt, model_, compiled_, s.parents->tiled, *s.session, *s.target,
            *s.runtime, qualified_, activated_, s.parents->resources, *s.dispatcher, s.held, s.request.sampler);
        static_cast<void>(m);
        ++stats_.admitted;
        s.order = stats_.admitted;
        s.active = true;
        s.prefilling = true;
        stats_.max_active = std::max<std::uint64_t>(stats_.max_active, active());
        if (!interleave_) {
            while (!s.prefill->done()) s.prefill->step();
            complete_prompt(s);
        }
        return static_cast<std::size_t>(free - slots_.begin());
    }

    // Interleaved admission: one unit of the oldest running prompt, when the decode cycles have had at least
    // `prompt_share` times the time of the previous units (STRIX_BATCH_PROMPT_SHARE, default 0: one unit per
    // step; always when nothing decodes). Then one decode cycle
    // of every request past its prompt: a cohort cycle with two or more, the ordinary cycle with one.
    void step() {
        using clock = std::chrono::steady_clock;
        Slot* prompt = nullptr;
        bool decoding = false;
        for (auto& s : slots_) {
            if (s->prefilling && (!prompt || s->order < prompt->order)) prompt = s.get();
            if (s->active && !s->prefilling) decoding = true;
        }
        if (!prompt) prompt_debt_ms_ = 0.0;
        if (prompt && (!decoding || prompt_debt_ms_ <= 0.0)) {
            ++stats_.prompt_steps;
            const auto t0 = clock::now();
            if (!run_prompt_group(*prompt)) {
                prompt->prefill->step(serial_rows_per_step_);
                if (prompt->prefill->done()) complete_prompt(*prompt);
            }
            prompt_debt_ms_ += prompt_share_ * std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        }
        std::vector<Slot*> live;
        for (auto& s : slots_) if (s->active && !s->prefilling) live.push_back(s.get());
        if (live.empty()) return;
        const auto cycle_start = clock::now();
        struct Debt {double& debt; clock::time_point t0;
            ~Debt(){debt -= std::chrono::duration<double, std::milli>(clock::now() - t0).count();}} debt{prompt_debt_ms_, cycle_start};
        const auto& m = options_.mtp;
        const auto budget = [](const Slot& s) {
            return static_cast<std::uint32_t>(std::min<std::uint64_t>(s.request.max_tokens - s.result.generated_tokens, UINT32_MAX));
        };
        ++stats_.steps;
        if (live.size() == 1U) {
            auto& s = *live.front();
            ++stats_.solo_steps;
            apply(s, s.greedy->cycle(s.pending, m.candidate_depth, budget(s), s.request.stop_tokens));
            return;
        }
        ++stats_.cohort_steps;
        std::vector<HipMtpGreedyCohortMember> members;
        members.reserve(live.size());
        for (auto* s : live) members.push_back({s->greedy.get(), &s->pending, m.candidate_depth, budget(*s), s->request.stop_tokens});
        const auto results = hip_mtp_greedy_cohort_cycle(members, &stats_.cohort);
        for (std::size_t k = 0; k < live.size(); ++k) apply(*live[k], results[k]);
    }

    // After an engine failure: every active request ends (the caller reports the failure).
    void abandon() noexcept {
        for (auto& s : slots_) {
            if (!s->active) continue;
            s->active = false;
            s->prefilling = false;
            s->prefill.reset();
            s->greedy.reset();
            s->pending = {};
            if (s->runtime) s->runtime->cancel();
        }
    }

private:
    // STRIX_PROMPT_BATCH=1: the prompt unit of a step is a group. When the oldest running prompt's next unit is a
    // tiled block (a T-row or tail block), the next blocks of the other running prompts join it, oldest first,
    // while the rows fit the budget (the T-row block owner's arena, or STRIX_PROMPT_BATCH_ROWS if smaller). The
    // group runs as one segmented prefill on slot 0's block owner (HipKnownPromptBackend::execute_group): each
    // prompt's block is the one it would run alone, with the same per-row arithmetic. A prompt whose next unit
    // is serial rows, or a lone block, runs alone as before. Returns false when no group of two or more formed.
    static bool prompt_batch_enabled() {
        static const bool on = [] { const char* v = std::getenv("STRIX_PROMPT_BATCH"); return v && v[0] == '1'; }();
        return on;
    }
    static std::uint32_t prompt_batch_rows() {
        static const std::uint32_t rows = [] {
            const char* v = std::getenv("STRIX_PROMPT_BATCH_ROWS");
            const long n = v ? std::atol(v) : 0L;
            return n > 0L ? static_cast<std::uint32_t>(n) : UINT32_MAX;
        }();
        return rows;
    }
    bool run_prompt_group(Slot& oldest) {
        if (!prompt_batch_enabled() || prefilling() < 2U) return false;
        const auto& lead = slots_.front()->parents->tiled;
        auto& group = lead.group ? *lead.group : *lead.blocks;
        const auto budget = std::min(lead.group ? lead.group_owner->logical_rows() : lead.rows, prompt_batch_rows());
        struct Member { Slot* slot; TiledMtpPrompt::Block block; };
        std::vector<Member> members;
        std::uint32_t rows = 0;
        const auto join = [&](Slot& s) {
            const auto block = s.prefill->next_block();
            if (!block || block->rows > budget - rows) return;
            members.push_back({&s, *block});
            rows += block->rows;
        };
        join(oldest);
        if (members.empty()) return false;
        std::vector<Slot*> others;
        for (auto& s : slots_) if (s.get() != &oldest && s->prefilling) others.push_back(s.get());
        std::sort(others.begin(), others.end(), [](const Slot* a, const Slot* b) { return a->order < b->order; });
        for (auto* s : others) join(*s);
        if (members.size() < 2U) return false;
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<HipKnownPromptBackend::GroupMember> group_members;
        group_members.reserve(members.size());
        for (const auto& m : members)
            group_members.push_back({m.block.backend, m.slot->prefill->block_tokens(m.block), m.slot->prefill->position(), m.block.final,
                m.slot->prefill->block_observer(m.block.rows)});
        const auto logits = HipKnownPromptBackend::execute_group(group, group_members);
        for (std::size_t k = 0; k < members.size(); ++k) {
            auto& s = *members[k].slot;
            s.prefill->finish_block(members[k].block, logits[k], true);
            if (s.prefill->done()) complete_prompt(s);
        }
        ++stats_.prompt_groups;
        stats_.grouped_blocks += members.size();
        stats_.grouped_rows += rows;
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        stats_.prompt_group_ms += ms;
        // STRIX_PROMPT_BATCH_LOG=1: one stderr line per group (blocks, rows, wall).
        static const bool log = [] { const char* v = std::getenv("STRIX_PROMPT_BATCH_LOG"); return v && v[0] == '1'; }();
        if (log) {
            std::cerr << "prompt group " << stats_.prompt_groups << ": " << members.size() << " blocks, " << rows << " rows (";
            for (std::size_t k = 0; k < members.size(); ++k) std::cerr << (k ? "+" : "") << members[k].block.rows;
            std::cerr << "), " << ms << " ms" << std::endl;
        }
        return true;
    }
    // The end of a request's prompt: its pending row, its greedy runtime and its first token.
    void complete_prompt(Slot& s) {
        const auto& m = options_.mtp;
        s.pending = s.prefill->finish(s.prompt_receipt);
        s.result.restored_tokens = s.prefill->restored();
        s.prefill.reset();
        s.prefilling = false;
        s.decode_start = std::chrono::steady_clock::now();
        s.result.prompt_ms = std::chrono::duration<double, std::milli>(s.decode_start - s.prompt_start).count();
        s.greedy = std::make_unique<HipMtpGreedyRuntime>(s.runtime, qualified_, activated_, s.target, *s.session,
            s.parents->resources, *s.dispatcher, s.parents);
        if (s.window) s.greedy->use_window_verifier(s.window);
        if (m.draft_chain) s.greedy->set_draft_min_probability(mtp_draft_min_probability(m));
        const std::array<std::uint32_t, 1> first{s.pending.predicted_token()};
        s.result.generated_tokens = 1;
        emitted(s, first);
    }
    void apply(Slot& s, const HipMtpGreedyCycleResult& result) {
        if (result.emitted_tokens.empty()) throw std::runtime_error("batched MTP generation made no completed target progress");
        s.pending = result.final_target;
        ++s.result.cycles;
        s.result.proposed += result.proposed_candidates;
        s.result.accepted += result.accepted_candidates;
        s.result.generated_tokens += result.emitted_tokens.size();
        ++s.chosen_rows[std::min<std::uint32_t>(result.target_window_rows, 8U)];
        emitted(s, result.emitted_tokens);
    }
    // Publishes tokens and ends the request at a stop token, its budget, the context, or a cancel.
    void emitted(Slot& s, std::span<const std::uint32_t> tokens) {
        const auto& stops = s.request.stop_tokens;
        if (!s.request.on_tokens(tokens)) s.result.finish = "cancelled";
        else if (std::find(stops.begin(), stops.end(), tokens.back()) != stops.end()) s.result.finish = "stop";
        else if (s.result.generated_tokens >= s.request.max_tokens) s.result.finish = "length";
        else if (s.session->next_position >= options_.context_tokens) s.result.finish = "context";
        else return;
        s.result.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s.decode_start).count();
        s.active = false;
        s.greedy.reset();
        s.pending = {};
        ++stats_.finished;
        if (s.request.on_finish) s.request.on_finish(s.result);
    }

    const Options& options_;
    const ModelContract& model_;
    const CompiledSerialExecution& compiled_;
    const QualifiedMtpDraftPlan& qualified_;
    const ActivatedMtpDraftPlan& activated_;
    EnableDraft enable_draft_;
    std::vector<std::unique_ptr<Slot>> slots_;
    bool interleave_ = false;
    std::uint32_t serial_rows_per_step_ = 16;
    // Interleaved admission: decode time owed to the requests past their prompt for the prompt units run.
    double prompt_debt_ms_ = 0.0;
    double prompt_share_ = [] {
        const char* v = std::getenv("STRIX_BATCH_PROMPT_SHARE");
        const double x = v ? std::atof(v) : 0.0;
        return x >= 0.0 ? x : 0.0;
    }();
    Stats stats_;
};

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
    // The recorded target pair also binds the target KV identity (kv-int8-tok-v1 prompt rows differ
    // from the BF16 cache's); the canonical BF16 KV keeps the historical pair identities.
    const auto kv_bound=[&](std::uint64_t identity) {
        return model.attention_kv_storage==AttentionKvStorage::bf16 ? identity :
            (identity^0x4b564938544f4b31ULL)*0x100000001b3ULL+static_cast<std::uint32_t>(model.attention_kv_storage);
    };
    const auto prefill_profile_identity=kv_bound(numerical_profile_identity(pair.prefill));
    const auto decode_profile_identity=kv_bound(numerical_profile_identity(pair.decode));
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
           split || model.attention_kv_storage!=AttentionKvStorage::bf16)
        throw std::invalid_argument("MTP golden was dumped under another target prefill/decode numerical profile pair "
            "or KV identity (a version-1 golden records none and requires one profile for both and the BF16 KV); dump a golden with this pair");
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
    // Sampled decoding: the final prompt row and every generated row select by the sampler;
    // earlier prompt rows keep first-max selection (their predictions are never emitted).
    const auto sampler=mtp_sampling(m) && !dump && !qualify ? std::make_shared<TokenSampler>(mtp_sampling_policy(m))
                                                           : std::shared_ptr<TokenSampler>{};
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
            if(sampler && !tiled_prompt && i+1U==options.tokens.size())target->set_token_sampler(sampler);
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
        const auto enable_draft_on=[&](HipMtpRuntime& draft_runtime,std::span<const std::uint32_t> prompt,
            const std::shared_ptr<TokenSampler>& draft_sampler) {
            if(!m.draft_chain) return;
            // Drafts only: the subset (or full) head changes acceptance, never target output.
            HipMtpDraftHeadConfig draft;draft.full=m.draft_vocab.empty() && !m.draft_context_vocab;
            if(!m.draft_vocab.empty()) {
                std::string chosen;
                const auto spec=HipMtpDraftHead::resolve_vocabulary_spec(m.draft_vocab,prompt,
                    static_cast<std::uint32_t>(model.vocab_size),chosen);
                if(spec!=m.draft_vocab) std::cerr<<"MTP draft vocabulary: "<<chosen<<'\n';
                draft.base_tokens=HipMtpDraftHead::read_token_file(spec,static_cast<std::uint32_t>(model.vocab_size));
            }
            draft.context_capacity=m.draft_context_vocab;draft.q4=m.draft_q4;
            // Runner-up probabilities feed the survival controller; STRIX_MTP_DRAFT_TOP2=1 logs them (diagnostic).
            draft.runner_up=hip_mtp_survival_controller() || std::getenv("STRIX_MTP_DRAFT_TOP2")!=nullptr;
            draft_runtime.enable_draft_chain(std::move(draft),m.draft_exact);draft_runtime.observe_draft_context(prompt);
            if(draft_sampler && m.sample_speculation)draft_runtime.set_draft_sampler(draft_sampler);
        };
        const auto enable_draft=[&](std::span<const std::uint32_t> prompt) {enable_draft_on(*runtime,prompt,sampler);};
        if(tiled_prompt) {
            calibration_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-calibration_start).count();
            // The calibration rows only qualified the predictor: start the real prompt from a fresh
            // target and a fresh predictor (the qualified/activated plans bind assets, not an owner).
            pending={};
            runtime.reset();
            runtime=std::make_shared<HipMtpRuntime>(native);
            parents->tiled.reset_state();
            session=parents->tiled.fresh;
            enable_draft(options.tokens);
            // The predictor's durable rows and the held h_{P-1} travel with each checkpoint; the identity
            // gains everything that changes them (cache rows use the draft kernels when the draft chain
            // is not exact). Every slot of a batched run extends its own view identically.
            const auto extend_cache=[&](HipPrefixStateCacheSession& cache,const HipMtpResources& predictor,std::byte* held) {
                const auto kv=predictor.view(HipMtpBuffer::kv),index=predictor.view(HipMtpBuffer::raw_index);
                cache.add_rows(kv.address,mtp_cli_mul(mtp_cli_mul(model.kv_heads,model.head_dim),4U),1U);
                cache.add_rows(index.address,mtp_cli_mul(model.indexer_head_dim,2U),1U);
                cache.add_fixed_device(reinterpret_cast<std::uintptr_t>(held),width*2U);
                std::vector<std::byte> frame;
                const auto u64=[&](std::uint64_t v){for(unsigned b=0;b<64;b+=8)frame.push_back(static_cast<std::byte>((v>>b)&255U));};
                for(const auto byte:assets->shared_target().sidecar_model_content_identity.bytes)frame.push_back(static_cast<std::byte>(byte));
                u64(assets->compiled().plan_identity);u64(runtime->numerical_profile_identity());
                u64(m.draft_chain);u64(m.draft_exact);u64(m.draft_q4);
                const char* q4_all=std::getenv("STRIX_MTP_DRAFT_Q4_ALL");u64(!q4_all || q4_all[0]!='0');
                cache.extend_identity(sha256_bytes(frame));
            };
            if(parents->tiled.cache) {
                extend_cache(*parents->tiled.cache,*native,tiled_held);
                parents->tiled.cache->set_restore_hook([&](std::uint64_t position){runtime->restore_cache_frontier(qualified,activated,position-1U);});
            }
            if(parents->serve_batch) {
                // Slot 0 is this request's state; every further slot gets the same objects over its own state.
                std::vector<std::unique_ptr<MtpBatchRunner::Slot>> slots;
                std::vector<std::byte*> held_rows;
                struct HeldRowsFree{std::vector<std::byte*>& p;~HeldRowsFree(){for(auto* x:p)(void)hipFree(x);}} held_rows_free{held_rows};
                auto first=std::make_unique<MtpBatchRunner::Slot>();
                first->parents=parents;first->session=&session;first->dispatcher=&dispatcher;
                first->native=native;first->target=target;first->window=window;first->held=tiled_held;
                slots.push_back(std::move(first));
                for(const auto& setup:parents->batch_slots) {
                    if(!setup.parents || !setup.session || !setup.dispatcher || !setup.parents->tiled.blocks || !setup.parents->tiled.reset_state)
                        throw std::invalid_argument("batched MTP slot lacks its parents, session, dispatcher or tiled prompt");
                    auto slot=std::make_unique<MtpBatchRunner::Slot>();
                    slot->parents=setup.parents;slot->session=setup.session;slot->dispatcher=setup.dispatcher;
                    const auto& own=*setup.parents->experiment;
                    slot->native=HipMtpResources::create(model,assets->compiled(),assets->sidecar_manifest(),assets->sidecar_residency(),
                        assets->target_manifest(),assets->target_residency(),assets->shared_target(),native_budget,own.allocator);
                    HipMtpTargetDeferredConfig own_d1;
                    if(deferred_status)own_d1={setup.parents->resources.session_state_layout,&generations};
                    slot->target=HipMtpTargetHidden::create(model,compiled,own.memory,own.model,own.allocator,
                        retained_rows,m.target_capture_bytes,own_d1);
                    if(m.window_rows) {
                        if(!setup.make_window)throw std::invalid_argument("batched MTP slot has no window factory");
                        slot->window=setup.make_window(slot->target);
                        if(!slot->window || slot->window->target_owner()!=slot->target || slot->window->logical_rows()!=m.window_rows)
                            throw std::invalid_argument("batched MTP slot window factory returned an incompatible owner");
                    }
                    std::byte* held=nullptr;
                    if(hipMalloc(reinterpret_cast<void**>(&held),static_cast<std::size_t>(width*2U))!=hipSuccess)
                        throw std::runtime_error("batched MTP held hidden allocation failed");
                    held_rows.push_back(held);slot->held=held;
                    if(setup.parents->tiled.cache) extend_cache(*setup.parents->tiled.cache,*slot->native,held);
                    slots.push_back(std::move(slot));
                }
                for(auto& slot:slots)
                    if(slot->parents->tiled.cache) {
                        auto* s=slot.get();
                        slot->parents->tiled.cache->set_restore_hook([s,&qualified,&activated](std::uint64_t position){
                            s->runtime->restore_cache_frontier(qualified,activated,position-1U);});
                    }
                pending={};
                runtime.reset();
                // STRIX_BATCH_INTERLEAVE: 1/0 forces block-interleaved admission on/off (default: on when serving).
                const char* interleave_env=std::getenv("STRIX_BATCH_INTERLEAVE");
                const bool interleave=interleave_env?interleave_env[0]=='1':options.http_serve;
                MtpBatchRunner runner(options,model,compiled,qualified,activated,enable_draft_on,std::move(slots),interleave);
                runner.calibration={calibration_rows,calibration_ms,replay.observation.maximum_hidden_abs_error,
                    replay.observation.maximum_logit_abs_error};
                try {parents->serve_batch(runner);}
                catch(...) {runner.abandon();throw;}
                assets->validate_assets_unchanged();
                if(!options.http_serve) return; // the cohort run wrote its own receipt
                const auto& st=runner.stats();
                output<<"{\"mode\":\"native-mtp-batch\",\"slots\":"<<runner.slots()<<",\"admitted\":"<<st.admitted
                    <<",\"finished\":"<<st.finished<<",\"steps\":"<<st.steps<<",\"cohort_steps\":"<<st.cohort_steps
                    <<",\"solo_steps\":"<<st.solo_steps<<",\"max_active\":"<<st.max_active<<",\"prompt_steps\":"<<st.prompt_steps
                    <<",\"interleaved_admission\":"<<(runner.interleaved()?"true":"false")
                    <<",\"cohort_windows\":"<<st.cohort.cohort_windows<<",\"cohort_window_sessions\":"<<st.cohort.cohort_sessions
                    <<",\"cohort_window_rows\":"<<st.cohort.cohort_rows<<",\"cohort_solo_windows\":"<<st.cohort.solo_windows<<",\"cohort_chunks\":"<<st.cohort.cohort_chunks
                    <<",\"calibration_rows\":"<<calibration_rows<<",\"calibration_ms\":"<<calibration_ms<<"}\n";
                return;
            }
            if(parents->serve) {
                std::uint64_t served=0;
                const MtpServeRunner run_request=[&](const MtpServeRequest& q) {
                    const auto vocabulary=model.vocab_size;
                    if(q.prompt.empty() || q.prompt.size()>=options.context_tokens || !q.max_tokens || !q.on_tokens)
                        throw std::invalid_argument("served request needs a prompt shorter than the context, a token budget and a sink");
                    for(const auto t:q.prompt)if(t>=vocabulary)throw std::invalid_argument("served prompt token outside the vocabulary");
                    for(const auto t:q.stop_tokens)if(t>=vocabulary)throw std::invalid_argument("served stop token outside the vocabulary");
                    MtpServeResult out;
                    const auto request_start=std::chrono::steady_clock::now();
                    // Every request starts exactly like the command-line request after calibration:
                    // fresh target state and session, fresh predictor, no deferred window rows.
                    pending={};
                    ++served;
                    runtime.reset();
                    runtime=std::make_shared<HipMtpRuntime>(native);
                    parents->tiled.reset_state();
                    session=parents->tiled.fresh;
                    if(window)window->begin_session();
                    enable_draft(q.prompt);
                    std::string request_receipt;
                    pending=run_tiled_mtp_prompt(q.prompt,model,compiled,parents->tiled,session,*target,*runtime,qualified,activated,
                        r,dispatcher,tiled_held,request_receipt);
                    if(parents->tiled.cache)out.restored_tokens=parents->tiled.cache->request().restored_position;
                    const auto decode_start=std::chrono::steady_clock::now();
                    out.prompt_ms=std::chrono::duration<double,std::milli>(decode_start-request_start).count();
                    HipMtpGreedyRuntime greedy(runtime,qualified,activated,target,session,r,dispatcher,parents);
                    if(window)greedy.use_window_verifier(window);
                    if(m.draft_chain)greedy.set_draft_min_probability(mtp_draft_min_probability(m));
                    const auto is_stop=[&](std::uint32_t t){return std::find(q.stop_tokens.begin(),q.stop_tokens.end(),t)!=q.stop_tokens.end();};
                    const std::array<std::uint32_t,1> first{pending.predicted_token()};
                    out.generated_tokens=1;
                    if(!q.on_tokens(first))out.finish="cancelled";
                    else if(is_stop(first[0]))out.finish="stop";
                    while(out.finish=="length" && out.generated_tokens<q.max_tokens) {
                        if(session.next_position>=options.context_tokens){out.finish="context";break;}
                        const auto budget=std::min<std::uint64_t>(q.max_tokens-out.generated_tokens,UINT32_MAX);
                        const auto result=greedy.cycle(pending,m.candidate_depth,static_cast<std::uint32_t>(budget),q.stop_tokens);
                        if(result.emitted_tokens.empty())throw std::runtime_error("MTP generation made no completed target progress");
                        pending=result.final_target;++out.cycles;
                        out.proposed+=result.proposed_candidates;out.accepted+=result.accepted_candidates;
                        out.generated_tokens+=result.emitted_tokens.size();
                        if(!q.on_tokens(result.emitted_tokens)){out.finish="cancelled";break;}
                        if(is_stop(result.emitted_tokens.back()))out.finish="stop";
                    }
                    pending={};
                    out.decode_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-decode_start).count();
                    return out;
                };
                parents->serve(run_request);
                assets->validate_assets_unchanged();
                output<<"{\"mode\":\"native-mtp-http-serve\",\"served_requests\":"<<served
                    <<",\"calibration_rows\":"<<calibration_rows<<",\"calibration_ms\":"<<calibration_ms<<"}\n";
                return;
            }
            pending=run_tiled_mtp_prompt(options.tokens,model,compiled,parents->tiled,session,*target,*runtime,qualified,activated,
                r,dispatcher,tiled_held,tiled_receipt);
        } else
        // Prefill handoff: every later target row, window and checkpoint runs the decode execution
        // from the state this prefill produced. The pending prompt row is re-minted with the
        // handed-off session; the lineage keeps the prefill identity of every prompt row.
        {
            if(split)pending=target->hand_off_prompt(session,pending,parents->prompt.resources,r);
            enable_draft(options.tokens);
        }
        HipMtpGreedyRuntime greedy(runtime,qualified,activated,target,session,r,dispatcher,parents);
        if(window && !qualify)greedy.use_window_verifier(window);
        if(m.draft_chain)greedy.set_draft_min_probability(mtp_draft_min_probability(m));
        if(m.ngram_draft && !qualify) {
            // STRIX_MTP_NGRAM_FLOOR / STRIX_MTP_NGRAM_PRIOR tune the gate (defaults 0.5 / 0.2).
            MtpNgramDraftConfig ngram;
            if(const char* v=std::getenv("STRIX_MTP_NGRAM_FLOOR"))ngram.floor=std::strtod(v,nullptr);
            if(const char* v=std::getenv("STRIX_MTP_NGRAM_PRIOR"))ngram.prior=std::strtod(v,nullptr);
            greedy.enable_ngram_draft(options.tokens,ngram);
        }
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
            const auto result=greedy.cycle(pending,sampler && !m.sample_speculation?0U:m.candidate_depth,
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
            <<"  \"mtp_ngram_draft\":"<<[&]{
                const auto* s=greedy.ngram_stats();
                if(!s) return std::string("null");
                return "{\"proposal_cycles\":"+std::to_string(s->cycles)+",\"extended_cycles\":"+std::to_string(s->extended_cycles)+
                    ",\"extension_rows\":"+std::to_string(s->extension_rows)+",\"accepted_rows\":"+std::to_string(s->accepted_rows)+"}";
            }()<<",\n"
            <<"  \"mtp_draft_head_rows\":"<<draft_stats.rows<<",\n  \"mtp_draft_base_rows\":"<<draft_stats.base_rows<<",\n"
            <<"  \"mtp_draft_context_rows\":"<<draft_stats.context_rows<<",\n  \"mtp_draft_context_dropped\":"<<draft_stats.context_dropped<<",\n"
            <<"  \"sampling\":"<<mtp_sampling_receipt(m,sampler.get())<<",\n"
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
