#pragma once
#include "strix/hip_mtp_resources.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/serial_session.hpp"
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
    [[nodiscard]] bool same_producer(const HipMtpTargetHiddenTicket&) const noexcept;
    [[nodiscard]] bool same_commit(const HipMtpTargetHiddenTicket&) const noexcept;
    [[nodiscard]] bool is_window() const noexcept;
    [[nodiscard]] std::uint64_t chunk_begin() const;
    [[nodiscard]] std::uint64_t chunk_end() const;
    [[nodiscard]] MtpDeviceView view() const;
    [[nodiscard]] std::uint64_t source_position() const;
    [[nodiscard]] std::uint32_t input_token() const;
    [[nodiscard]] std::uint32_t predicted_token() const;
    [[nodiscard]] const SerialSessionContext& committed_session() const;
    [[nodiscard]] ModelContentIdentity target_content_identity() const;
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
    HipMtpTargetHidden(const HipMtpTargetHidden&)=delete;
    HipMtpTargetHidden& operator=(const HipMtpTargetHidden&)=delete;
    [[nodiscard]] HipMtpTargetHiddenTicket execute_serial(
        SerialSessionContext&,const ExecutionRuntimeResources&,
        DirectLaunchDispatcher&,std::uint32_t input_token);
    // Split prefill numerical profile. Prompt rows run a sibling serial execution (same steps,
    // arena, state and model bindings; different arithmetic) on the same scratch arena, selected
    // by the session's executor identity. Set once, before any row. hand_off_prompt then moves
    // the session to the decode execution and re-mints the pending prompt row with the handed-off
    // session; the returned ticket replaces `pending`. Later rows run the decode execution only.
    void set_prompt_execution(const CompiledSerialExecution& prompt);
    [[nodiscard]] HipMtpTargetHiddenTicket hand_off_prompt(SerialSessionContext&,
        const HipMtpTargetHiddenTicket& pending,const ExecutionRuntimeResources& prompt_resources,
        const ExecutionRuntimeResources& decode_resources);
private:
    HipMtpTargetHidden();
    friend class HipMtpTargetHiddenTicket;
    friend struct HipMtpTargetHiddenTicket::Lease;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
