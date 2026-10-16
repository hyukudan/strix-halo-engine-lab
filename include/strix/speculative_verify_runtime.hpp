#pragma once

#include "strix/compiled_speculative_verify.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_runtime.hpp"
#include "strix/serial_session.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <span>

namespace strix {

struct CompiledLaunchPlan;

// The caller retains both row registries/configs and every resource owner until
// completion or quiesce_or_fail_stop. The runtime invokes each selected R1
// launcher's pure preflight on both rows; a missing callback fails admission.
// preflight_plan optionally adds owner policy. preflight_scope proves an owned eager stream and
// deferred-status parity, and null graph-position arguments. complete drains/checks status before publication.
struct KnownWindowExecutionHooks {
    // Metadata/config/resource owners cannot change via aliases during a call.
    std::function<void()> preflight_plan;
    std::function<void()> preflight_scope;
    std::function<void()> complete;
    std::function<void()> quiesce_or_fail_stop;
    // Closed QSA implementation identities admitted by the native owner; a
    // test may inject a recording QSA identity. Runtime position uses its fixed
    // default implementation identity, never the verify-window implementation.
    std::span<const std::uint64_t> position_reader_implementations;
};

struct KnownWindowR2Entry {
    KnownWindowR2Variant variant = KnownWindowR2Variant::none;
    void* config = nullptr;
    void (*preflight)(std::span<const ExecutionDispatchContext, 2>, void*) = nullptr;
    void (*launch)(std::span<const ExecutionDispatchContext, 2>, void*) = nullptr;
};

struct KnownWindowResult {
    EagerExecutionResult execution;
    std::array<ExecutionBufferAddress, 2> logits;
    std::uint64_t logical_projection_weight_bytes_avoided = 0;
    std::uint32_t projection_launches_saved = 0;
    std::uint32_t logical_row_steps = 0;
    std::uint32_t r2_dispatches = 0;
    std::uint32_t row_loop_dispatches = 0;
    std::array<std::uint32_t, known_window_r2_variant_slots> r2_variant_dispatches{};
};

// Two KNOWN input tokens only. This API neither predicts a future greedy token
// nor authorises speculative reuse. Row launch plans must be freshly built from
// window.row_plan[r], preserving each serial incumbent kernel id.
[[nodiscard]] KnownWindowResult execute_known_prefix_window_eager(
    const CompiledKnownPrefixWindow& window, const CompiledSerialExecution& serial,
    const ExecutionRuntimeResources& serial_resources,
    const ExecutionMemoryBinding& window_scratch, SerialSessionContext& session,
    std::array<std::uint32_t, 2> tokens,
    std::array<const CompiledLaunchPlan*, 2> row_launch,
    std::span<const KnownWindowR2Entry> r2_entries,
    const KnownWindowExecutionHooks& hooks);

struct SpeculativeVerifyExecutionResult {
    EagerExecutionResult execution;
    std::uint64_t base_position = 0;
    std::uint64_t post_position = 0;
    std::uint64_t base_committed_tokens = 0;
    std::uint64_t post_committed_tokens = 0;
};

[[nodiscard]] SpeculativeVerifyExecutionResult
execute_compiled_speculative_verify_eager(
    const CompiledSpeculativeVerifyExecution& compiled,
    const ExecutionRuntimeResources& resources,
    SerialSessionContext& session,
    std::span<const std::uint32_t> candidate_tokens,
    EagerExecutionDispatcher& dispatcher);

}  // namespace strix
