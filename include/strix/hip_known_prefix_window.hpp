#pragma once

#include "strix/hip_launcher_catalog.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/speculative_verify_runtime.hpp"
#include "strix/q51_known_rows.hpp"
#include "strix/known_prefix_prefill.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>

namespace strix {
class HipMtpTargetWindowCapture;
struct HipMtpTargetWindowCapturePolicy;
struct HipMtpTargetWindowExecution;
struct HipKnownPrefixUnionObservation {
    std::uint64_t first_position = 0;
    std::uint32_t layer = 0;
    q51_known_rows::Summary group;
};

// Native eager owner for 2/4/8 KNOWN input tokens, not speculative generation.
// The caller's actual_owner retains every borrowed model/state/arena/stream
// allocation and descriptor. Metadata and allocation contents are exclusively
// owned from construction through destruction; no concurrent calls or aliases
// may write them. State changes only through the prescribed execution/PLE path.
//
// If deferred status is enabled, caller initializes the separate eight-byte
// device record to zero before construction. This owner never clears it. The
// constructor checks zero during guarded setup; every successful execute checks
// it again before session publication. A runtime/admission exception permanently
// disables this owner, even when the core correctly leaves a rejected session
// ready. Restore/recovery requires a newly constructed owner and a fresh zero
// status setup. There is no implicit single-token shadow or token counter.
//
// Construction/admission is not compiler, native or whole-model qualification.
class HipKnownPrefixWindow final {
public:
    HipKnownPrefixWindow(CompiledKnownPrefixWindow window,
        CompiledSerialExecution serial_snapshot, ModelContract model,
        HipLauncherCatalogConfig config,
        const ExecutionRuntimeResources& serial_resources,
        ExecutionMemoryBinding window_scratch, HipLaunchContext scope,
        std::shared_ptr<const void> actual_owner,
        q51_known_rows::Workspace routed_union_workspace = {});
    // Explicit T16..2048 step-major owner. Closed window construction above
    // retains its original guards and identities. Every projection tile still
    // uses a closed native 8/2 entry; this is not a general GEMM backend.
    HipKnownPrefixWindow(CompiledKnownPrefixPrefill prefill,
        CompiledSerialExecution serial_snapshot,ModelContract model,HipLauncherCatalogConfig config,
        const ExecutionRuntimeResources& serial_resources,ExecutionMemoryBinding prefill_scratch,HipLaunchContext scope,
        std::shared_ptr<const void> actual_owner,q51_known_rows::Workspace routed_union_workspace = {});
    ~HipKnownPrefixWindow();
    HipKnownPrefixWindow(const HipKnownPrefixWindow&) = delete;
    HipKnownPrefixWindow& operator=(const HipKnownPrefixWindow&) = delete;
    HipKnownPrefixWindow(HipKnownPrefixWindow&&) = delete;
    HipKnownPrefixWindow& operator=(HipKnownPrefixWindow&&) = delete;

    [[nodiscard]] KnownWindowResult execute(SerialSessionContext& session,
        std::array<std::uint32_t, 2> tokens);
    [[nodiscard]] KnownWindowResult execute(SerialSessionContext& session,
        std::span<const std::uint32_t> tokens);
    // Optional serving delivery validation AFTER producer/D1 completion but
    // BEFORE publishing any session state/generations. Callback is synchronous
    // on the owned eager scope; throws follow the existing poison/drain path.
    [[nodiscard]] KnownWindowResult execute_validated(SerialSessionContext&,
        std::span<const std::uint32_t>,const std::function<void()>& before_publish);
    // Typed opt-in target capture, closed R2/4/8 all-head windows only.
    // Preparation allocates all snapshot/selector payload before target work.
    // Returned batch proves only the ACTUAL committed whole chunk, not accepted
    // speculative rows or an invented intermediate SerialSessionContext.
    [[nodiscard]] std::shared_ptr<HipMtpTargetWindowCapture> prepare_mtp_target_capture(
        const HipMtpTargetWindowCapturePolicy&);
    [[nodiscard]] HipMtpTargetWindowExecution execute_mtp_captured(SerialSessionContext&,
        std::span<const std::uint32_t>,std::shared_ptr<HipMtpTargetWindowCapture>,
        const std::function<void()>& before_publish = {});
    [[nodiscard]] std::uint32_t logical_rows() const noexcept;
    [[nodiscard]] bool supports_tiled_prefill() const noexcept;
    // Immutable compiler-selected intent; verification consumers require all_rows.
    [[nodiscard]] KnownWindowOutputPolicy output_policy() const noexcept;
    // Last eligible layer in the last SUCCESSFULLY completed window only.
    // Not aggregate reuse, DRAM traffic, timing or model qualification.
    [[nodiscard]] std::optional<HipKnownPrefixUnionObservation> latest_routed_union_observation() const noexcept;
    // Prefix acceptance (deferred GDN / PLE rows windows). The window executed all rows but only
    // [0, n_keep) are committed; the caller rewinds its session frontier to match.
    [[nodiscard]] bool defers_state() const noexcept;
    void commit_prefix(std::uint32_t n_keep);
    // Applies pending GDN rows before any ordinary serial token touches GDN state.
    void flush_deferred();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strix
