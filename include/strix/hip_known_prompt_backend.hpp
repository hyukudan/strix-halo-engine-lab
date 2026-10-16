#pragma once

#include "strix/execution_memory.hpp"
#include "strix/compiled_speculative_verify.hpp"
#include "strix/serial_request.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace strix {
class HipKnownPrefixWindow;
class HipSetupQuiescenceGuard;
struct SerialSessionContext;

// Successful adapter returns only. Logical payload/launch ledgers are not
// measured memory traffic or timing. Overflow clamps telemetry and sets the
// flag; it never changes model execution, commit or request policy.
struct HipKnownPromptBackendStats {
    std::uint64_t windows = 0;
    std::uint64_t r2_dispatches = 0;
    std::uint64_t row_loop_dispatches = 0;
    std::uint64_t logical_projection_weight_bytes_avoided = 0;
    std::uint64_t projection_launches_saved = 0;
    std::array<std::uint64_t, known_window_r2_variant_slots> r2_variant_dispatches{};
    bool saturated = false;
    std::uint64_t completed_input_rows = 0;
    std::uint64_t shared_rows_dispatches = 0;
    std::uint64_t gdn_rows_dispatches = 0;
};

// Experimental eager prompt adapter, not speculative generation. Every input
// in the configured 2/4/8-row window is already known. The native owner computes
// complete heads, completes causal window/D1 checks and commits all rows before
// this adapter downloads only the terminal row's complete FP32 logits.
//
// window_scratch must be the exact retained physical owner passed to window.
// The caller retains session through destruction and excludes concurrent access
// to the session, window and all their reachable owners. The shared window
// retains its actual model/state/PLE/arena/stream resources; this class does not
// create a D1 shadow, clear a status record or publish another state generation.
//
// Any attempted-call exception disables this adapter. A failure after state
// consumption poisons the session and taints lineage without rolling it back;
// original-device drain is proved before a native-attempt exception returns. G0 rejection
// without consumption preserves the native owner's ready-session semantics.
class HipKnownPromptBackend final : public KnownPromptWindowBackend {
public:
    HipKnownPromptBackend(std::shared_ptr<HipKnownPrefixWindow> window,
        SerialSessionContext& session, ExecutionMemoryBinding window_scratch,
        std::uint64_t vocab_size, std::shared_ptr<const HipSetupQuiescenceGuard> guard);
    ~HipKnownPromptBackend();
    HipKnownPromptBackend(const HipKnownPromptBackend&) = delete;
    HipKnownPromptBackend& operator=(const HipKnownPromptBackend&) = delete;
    HipKnownPromptBackend(HipKnownPromptBackend&&) = delete;
    HipKnownPromptBackend& operator=(HipKnownPromptBackend&&) = delete;

    [[nodiscard]] std::vector<float> execute_window(
        std::array<std::uint32_t, 2> tokens, std::uint64_t position) override;
    [[nodiscard]] std::uint32_t logical_rows() const noexcept override;
    [[nodiscard]] std::vector<float> execute_rows(
        std::span<const std::uint32_t> tokens, std::uint64_t position) override;
    // A failed output copy/validation does not publish new telemetry, even
    // though its session may already have consumed the configured native rows.
    [[nodiscard]] HipKnownPromptBackendStats stats() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
