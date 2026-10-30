#pragma once
#include "strix/hip_mtp_resources.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/serial_session.hpp"
#include "strix/token_sampling.hpp"
#include <functional>
#include <optional>
#include <span>
namespace strix {
class HipMtpTargetHidden;
class HipMtpTargetWindowBatch;
class HipMtpTargetWindowCapture;
// Minted by an actual completed target execution, never by pointer admission.
// Copies retain a snapshot slot and its actual allocation owner.
class HipMtpTargetHiddenTicket {
public:
    HipMtpTargetHiddenTicket()=default;
    [[nodiscard]] bool valid() const noexcept;
    // Request-scoped diagnostics; seed is the real terminal prompt head, never a rerun.
    void begin_step_logits_digest() const;
    void print_step_logits_summary(std::uint64_t committed_end) const;
    [[nodiscard]] bool same_producer(const HipMtpTargetHiddenTicket&) const noexcept;
    [[nodiscard]] bool same_commit(const HipMtpTargetHiddenTicket&) const noexcept;
    [[nodiscard]] bool is_window() const noexcept;
    // Last row of a tiled prefill block (adopt_prefill_row): its chunk is the block's lineage interval.
    [[nodiscard]] bool is_prefill_block() const noexcept;
    [[nodiscard]] std::uint64_t chunk_begin() const;
    [[nodiscard]] std::uint64_t chunk_end() const;
    [[nodiscard]] MtpDeviceView view() const;
    [[nodiscard]] std::uint64_t source_position() const;
    [[nodiscard]] std::uint32_t input_token() const;
    [[nodiscard]] std::uint32_t predicted_token() const;
    [[nodiscard]] const SerialSessionContext& committed_session() const;
    [[nodiscard]] ModelContentIdentity target_content_identity() const;
    [[nodiscard]] RotaryIdentity producer_rotary_identity() const;
    void admit_rotary_reader(const RotaryIdentity&) const;
    // STRIX_SERIAL_LOGITS_DIGEST=1: serial FP32 logits; STRIX_MTP_LOGITS_DIGEST=1 also captures verified rows.
    // STRIX_MTP_STEP_LOGITS_DIGEST=1 additionally retains the tiled terminal head.
    [[nodiscard]] std::optional<ModelContentIdentity> serial_logits_digest() const;
    [[nodiscard]] std::optional<ModelContentIdentity> terminal_logits_digest() const;
    // Execution identity of the serial execution that produced this row (a window ticket
    // reports its committed session's executor).
    [[nodiscard]] std::uint64_t producer_execution_identity() const;
    // True iff `next` is the row this producer re-minted at its admitted prompt handoff and this
    // ticket is the preceding prompt row: the one place where consecutive rows change executor.
    [[nodiscard]] bool admits_executor_change(const HipMtpTargetHiddenTicket& next) const noexcept;
private:
    friend class HipMtpTargetHidden;
    struct Lease;
    std::shared_ptr<const Lease> lease_;
};
struct HipMtpTargetDeferredConfig {
    const SerialSessionStateLayout* state_layout=nullptr;
    AllocationGenerationSource* generations=nullptr;
};
// Optional host selection is per-call and never retained by the producer. The
// callback sees a fully validated finite row and replaces the actual ticket token.
using HipMtpHostRowSelector = std::function<std::uint32_t(std::span<const float>)>;

// Serialized eager target producer. Owns its stream and bounded capture slots;
// direct dispatch uses that stream through the existing native launch scope.
// This producer supplies actual target rows; it does not qualify MTP arithmetic.
class HipMtpTargetHidden final : public std::enable_shared_from_this<HipMtpTargetHidden> {
public:
    [[nodiscard]] static std::shared_ptr<HipMtpTargetHidden> create(
        const ModelContract&,const CompiledSerialExecution&,
        std::shared_ptr<const RuntimeExecutionMemoryOwner>,
        std::shared_ptr<const HipModelResidency>,
        std::shared_ptr<HipRuntimeAllocator>,std::uint32_t retained_rows,
        std::uint64_t retained_byte_budget,HipMtpTargetDeferredConfig = {});
    ~HipMtpTargetHidden();
    [[nodiscard]] std::uint32_t retained_rows() const noexcept;
    [[nodiscard]] const ModelContract& model() const noexcept;
    [[nodiscard]] const CompiledSerialExecution& compiled() const noexcept;
    [[nodiscard]] std::shared_ptr<const RuntimeExecutionMemoryOwner> memory_owner() const noexcept;
    [[nodiscard]] std::shared_ptr<const HipModelResidency> weights_owner() const noexcept;
    [[nodiscard]] std::shared_ptr<HipRuntimeAllocator> allocator() const noexcept;
    [[nodiscard]] void* stream() const noexcept;
    [[nodiscard]] std::shared_ptr<const HipSetupQuiescenceGuard> guard() const noexcept;
    void validate_window_capture(const HipMtpTargetWindowCapture&) const;
    // A provisional window becomes committed row tickets only when its actual
    // input chain matches finite target predictions, EOS/budget permit all rows,
    // and the exact actual target state still equals the completed whole chunk.
    // Intermediate rows retain the REAL final chunk session, never fake prefixes.
    // Rows [0, n_keep) of a deferred-state window whose session was already rewound to the
    // accepted prefix (drafts 1..n_keep-1 matched their predecessors' predictions).
    [[nodiscard]] std::vector<HipMtpTargetHiddenTicket> accept_verified_prefix(
        std::shared_ptr<const HipMtpTargetWindowBatch> batch,const HipMtpTargetHiddenTicket& base,
        const SerialSessionContext& target,std::uint32_t n_keep);
    [[nodiscard]] std::vector<HipMtpTargetHiddenTicket> accept_verified_window(
        std::shared_ptr<const HipMtpTargetWindowBatch>,const HipMtpTargetHiddenTicket& base,
        const SerialSessionContext& actual_target,std::span<const std::uint32_t> eos_tokens,
        std::uint32_t output_budget);
    // Known prompt rows (HipMtpWindowVerifier::prompt_rows): every row of a deferred-state window whose inputs were
    // the prompt's own tokens, committed whole (target.next_position is the window's end). Each row's arithmetic is
    // the serial row's; unlike accept_verified_prefix no input needs to equal its predecessor's prediction.
    [[nodiscard]] std::vector<HipMtpTargetHiddenTicket> accept_known_rows(
        std::shared_ptr<const HipMtpTargetWindowBatch> batch,const SerialSessionContext& target);
    HipMtpTargetHidden(const HipMtpTargetHidden&)=delete;
    HipMtpTargetHidden& operator=(const HipMtpTargetHidden&)=delete;
    // With a token sampler, the row's selection is the sampled decision for the
    // token at input position + 1 (using `draft` when it proposes that token);
    // otherwise it is the finite first-max token and `draft` must be null.
    [[nodiscard]] HipMtpTargetHiddenTicket execute_serial(
        SerialSessionContext&,const ExecutionRuntimeResources&,
        DirectLaunchDispatcher&,std::uint32_t input_token,const TokenDraft* draft=nullptr,
        const HipMtpHostRowSelector& host_selector={});
    // Opt-in raw completed-head observer for session qualification (serial and adopted tiled terminal rows).
    void set_session_logits_observer(std::function<void(std::uint64_t, std::span<const float>)>);
    // Sampled decoding (temperature > 0): set once; later serial rows select by it.
    void set_token_sampler(std::shared_ptr<TokenSampler>);
    // Serving several requests on one producer: the sampler of the next request (null: first-max), on an idle
    // healthy producer. The final prompt row and every later row select by it.
    void replace_token_sampler(std::shared_ptr<TokenSampler>);
    [[nodiscard]] std::shared_ptr<TokenSampler> token_sampler() const noexcept;
    // Split prefill numerical profile. Prompt rows run a sibling serial execution (same steps,
    // arena, state and model bindings; different arithmetic) on the same scratch arena, selected
    // by the session's executor identity. Set once, before any row. hand_off_prompt then moves
    // the session to the decode execution and re-mints the pending prompt row with the handed-off
    // session; the returned ticket replaces `pending`. Later rows run the decode execution only.
    void set_prompt_execution(const CompiledSerialExecution& prompt);
    // Tiled MTP prompt handoff: mints the pending row from the last row of the tiled prefill block
    // that has just committed `session` (its last lineage interval, prefill or known-window chunk,
    // ending at session.next_position). `hidden` is that row's wide tap (copied into a retained
    // slot) and `logits` its full FP32 head row; the prediction uses the selector's finite
    // first-max rule. The caller guarantees both come from that completed block (the tiled
    // prefill computes every row exactly as a serial row of the same execution).
    [[nodiscard]] HipMtpTargetHiddenTicket adopt_prefill_row(const SerialSessionContext& session,
        std::uint32_t input_token,MtpDeviceView hidden,std::span<const float> logits,const RotaryIdentity& producer_rotary,
        const HipMtpHostRowSelector& host_selector={});
    [[nodiscard]] HipMtpTargetHiddenTicket hand_off_prompt(SerialSessionContext&,
        const HipMtpTargetHiddenTicket& pending,const ExecutionRuntimeResources& prompt_resources,
        const ExecutionRuntimeResources& decode_resources);
private:
    [[nodiscard]] HipMtpTargetHiddenTicket execute_serial_unselected(
        SerialSessionContext&,const ExecutionRuntimeResources&,DirectLaunchDispatcher&,std::uint32_t,const TokenDraft*,bool host_selection);
    HipMtpTargetHidden();
    friend class HipMtpTargetHiddenTicket;
    friend struct HipMtpTargetHiddenTicket::Lease;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
