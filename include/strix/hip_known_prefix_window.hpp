#pragma once

#include "strix/hip_launcher_catalog.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/speculative_verify_runtime.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace strix {

// Native eager owner for two KNOWN input tokens, not speculative generation.
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
        std::shared_ptr<const void> actual_owner);
    ~HipKnownPrefixWindow();
    HipKnownPrefixWindow(const HipKnownPrefixWindow&) = delete;
    HipKnownPrefixWindow& operator=(const HipKnownPrefixWindow&) = delete;
    HipKnownPrefixWindow(HipKnownPrefixWindow&&) = delete;
    HipKnownPrefixWindow& operator=(HipKnownPrefixWindow&&) = delete;

    [[nodiscard]] KnownWindowResult execute(SerialSessionContext& session,
        std::array<std::uint32_t, 2> tokens);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strix
