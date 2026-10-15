#pragma once

#include "strix/execution_memory.hpp"
#include <cstdint>
#include <memory>
#include <string_view>

namespace strix {
class HipSetupQuiescenceGuard;

inline constexpr std::string_view gpu_greedy_output_identity =
    "gpu-greedy-firstmax-finite-v1";

struct HipGreedyOutputStats {
    std::uint64_t calls = 0;
    std::uint64_t completed = 0;
    std::uint64_t reduction_kernel_submissions = 0;
    std::uint64_t result_d2h_bytes = 0;
    std::uint64_t poison_h2d_bytes = 0;
    std::uint64_t stream_completions = 0;
    std::uint64_t failures = 0;
    std::uint64_t pool_allocations = 0;
};

// Serialized C1 output owner. Caller MUST complete the original producer before
// select(), retain the actual logits allocation via logits_owner, and prevent any
// writer/remap until select returns. It never rolls back an already consumed token.
// On selection failure caller poisons/taints its session and publishes no output.
class HipGreedyOutput final {
public:
    HipGreedyOutput(std::uint32_t vocabulary_size, ExecutionMemoryBinding scratch,
        std::shared_ptr<const void> logits_owner,
        std::shared_ptr<const HipSetupQuiescenceGuard> guard);
    ~HipGreedyOutput();
    HipGreedyOutput(const HipGreedyOutput&) = delete;
    HipGreedyOutput& operator=(const HipGreedyOutput&) = delete;
    HipGreedyOutput(HipGreedyOutput&&) = delete;
    HipGreedyOutput& operator=(HipGreedyOutput&&) = delete;

    // Scratch is the retained compiled arena binding; every view must be an exact
    // vocabulary-sized slice with its allocation generation. First call binds the
    // selected view address for this
    // owner. Epoch must be nonzero and strictly increasing, including after error.
    // No device or transfer-buffer allocation; all-vocabulary finite checking
    // and first-max numeric ties.
    // Position is the consumed INPUT position, not the predicted token position.
    [[nodiscard]] std::uint32_t select(const ExecutionBufferAddress& logits,
        std::uint32_t vocabulary_size, std::uint64_t position, std::uint64_t epoch);
    [[nodiscard]] HipGreedyOutputStats stats() const noexcept;
    [[nodiscard]] bool failed() const noexcept;
    // Total retained payload bytes: device partials + device reply + pinned host
    // reply. Excludes allocator/runtime overhead, which is not observed here.
    // Requested allocation sizes, including the pinned reply; driver allocation
    // granularity and stream bookkeeping are not a measured physical peak.
    [[nodiscard]] std::uint64_t scratch_bytes() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
