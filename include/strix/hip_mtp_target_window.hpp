#pragma once
#include "strix/hip_mtp_resources.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/speculative_verify_runtime.hpp"
#include "strix/mtp_selection_transaction.hpp"
#include <functional>
#include <memory>
#include <optional>
#include <span>
namespace strix {
class HipKnownPrefixWindow;
class HipMtpTargetWindowCapture;
struct HipGreedyRowsReply;
struct HipMtpTargetWindowCapturePolicy {
    std::uint64_t retained_byte_budget=0,maximum_history_intervals=0;
};
// Actual completed WHOLE chunk only. Row snapshots are provisional evidence,
// not independent committed SerialSessionContexts or speculative acceptance.
// Target promotion must separately establish accepted/replayed authority.
class HipMtpTargetWindowBatch {
public:
    HipMtpTargetWindowBatch()=default;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint32_t logical_rows() const;
    [[nodiscard]] const MtpSelectionPrefix* selection_prefix() const;
    [[nodiscard]] std::uint64_t chunk_begin() const;
    [[nodiscard]] std::uint64_t chunk_end() const;
    [[nodiscard]] std::uint64_t source_position(std::uint32_t row) const;
    [[nodiscard]] std::uint32_t input_token(std::uint32_t row) const;
    [[nodiscard]] std::uint32_t predicted_token(std::uint32_t row) const;
    // Diagnostic snapshot of the actual last head, retained before scratch reuse.
    [[nodiscard]] std::optional<ModelContentIdentity> terminal_logits_digest(std::uint32_t row) const;
    [[nodiscard]] MtpDeviceView view(std::uint32_t row) const;
    [[nodiscard]] const SerialSessionContext& base_session() const;
    [[nodiscard]] const SerialSessionContext& committed_session() const;
    [[nodiscard]] ModelContentIdentity target_content_identity() const;
    [[nodiscard]] std::uint64_t window_execution_identity() const;
    [[nodiscard]] std::uint64_t window_kernel_plan_identity() const;
    [[nodiscard]] std::uint64_t epoch() const;
    [[nodiscard]] const SessionStateMemoryBinding& source_state_memory_binding() const;
    [[nodiscard]] const ExecutionMemoryBinding& source_scratch_binding() const;
    [[nodiscard]] const ModelRuntimeBinding& source_weights_binding() const;
    [[nodiscard]] const ExecutionMemoryBinding& deferred_status_binding() const;
    [[nodiscard]] void* source_stream() const;
private:
    friend class HipMtpTargetWindowCapture;
    struct Lease;
    std::shared_ptr<const Lease> lease_;
};
struct HipMtpTargetWindowExecution {
    KnownWindowResult execution;
    HipMtpTargetWindowBatch batch;
};
// Fixed GPU slots and finite-firstmax pools allocated BEFORE target execution.
// Reusable only after the previous batch/tickets release their lease. Batch
// retains Capture; Capture keeps a WEAK prior lease, never a retaining cycle.
// All views are valid only while their actual batch/ticket remains retained.
class HipMtpTargetWindowCapture final : public std::enable_shared_from_this<HipMtpTargetWindowCapture> {
public:
    ~HipMtpTargetWindowCapture();
    HipMtpTargetWindowCapture(const HipMtpTargetWindowCapture&)=delete;
    HipMtpTargetWindowCapture& operator=(const HipMtpTargetWindowCapture&)=delete;
    [[nodiscard]] std::uint64_t retained_payload_bytes() const noexcept;
    [[nodiscard]] bool ready_for_reuse() const noexcept;
    [[nodiscard]] const SessionStateMemoryBinding& source_state_memory_binding() const noexcept;
    [[nodiscard]] const ExecutionMemoryBinding& source_scratch_binding() const noexcept;
    [[nodiscard]] const ModelRuntimeBinding& source_weights_binding() const noexcept;
    [[nodiscard]] std::uint64_t source_serial_execution_identity() const noexcept;
    [[nodiscard]] std::uint32_t logical_rows() const noexcept;
    [[nodiscard]] std::uint64_t window_execution_identity() const noexcept;
    [[nodiscard]] std::uint64_t window_kernel_plan_identity() const noexcept;
    [[nodiscard]] const ExecutionMemoryBinding& deferred_status_binding() const noexcept;
    [[nodiscard]] void* source_stream() const noexcept;
    // Sampled decoding: called after the window completes and before any row is
    // published, with every row's FP32 vocabulary logits and finite first-max
    // tokens; it may replace tokens. Empty (the default) keeps first-max selection.
    using RowSelection=std::function<void(std::span<const ExecutionBufferAddress>,std::span<std::uint32_t>)>;
    void set_row_selection(RowSelection);
    // Rows' FP32 logits and, when the owner reduces them on the GPU, each row's unmasked finite first-max
    // reply (token, status); empty otherwise.
    using TransactionSelection=std::function<MtpSelectionPrefix(std::span<const ExecutionBufferAddress>,std::span<const HipGreedyRowsReply>)>;
    void set_transaction_selection(TransactionSelection);
private:
    friend class HipKnownPrefixWindow;
    friend class HipMtpTargetWindowBatch;
    HipMtpTargetWindowCapture();
    [[nodiscard]] static std::shared_ptr<HipMtpTargetWindowCapture> create(
        const void* source_owner,const ModelContract&,const CompiledSerialExecution&,
        const CompiledKnownPrefixWindow&,const ExecutionRuntimeResources&,
        ExecutionMemoryBinding,HipLaunchContext,std::shared_ptr<const void>,HipMtpTargetWindowCapturePolicy);
    void begin(const void* source_owner,const SerialSessionContext&,std::span<const std::uint32_t>);
    void preflight_row(const ExecutionDispatchContext&) const;
    void record_row(const ExecutionDispatchContext&);
    void completed_before_publish();
    // Host-mapped replies the window head rows fill (null: per-row selectors).
    void use_device_greedy(const HipGreedyRowsReply* replies) noexcept;
    [[nodiscard]] HipMtpTargetWindowBatch seal(const SerialSessionContext&);
    void abort() noexcept;
    [[nodiscard]] std::uint32_t tap_step() const noexcept;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
