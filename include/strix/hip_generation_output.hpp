#pragma once

#include "strix/execution_memory.hpp"
#include <cstdint>
#include <memory>
#include <span>
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
    // Pure requested payload reservation; no HIP calls or allocation. Includes
    // device partials + device reply + pinned reply. Rejects zero vocabulary
    // and arithmetic overflow; excludes driver/stream/allocator overhead.
    [[nodiscard]] static std::uint64_t required_scratch_bytes(std::uint32_t vocabulary_size);
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
    // Split form for independently retained rows: submit all owners first, then
    // complete all before publishing any token. The caller keeps every logits
    // allocation immutable until its completion. Each owner allows one pending
    // request; host-only rejection does not invalidate that pending request.
    void submit(const ExecutionBufferAddress& logits, std::uint32_t vocabulary_size,
        std::uint64_t position, std::uint64_t epoch);
    [[nodiscard]] std::uint32_t complete();
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

// Window greedy selection: the same all-vocabulary first-max finite rule as
// HipGreedyOutput (numeric zeros equal, lowest token on ties) for 1..8 rows in
// two launches. replies must be a host-mapped coherent allocation of rows
// entries (hipHostMalloc mapped|coherent): after the stream drains, replies[r]
// is row r's result with no copy. status bit0: a non-finite logit was seen;
// bit1: no finite logit (token 0). Callers keep logits/partials immutable until
// the stream completes; partials is device scratch of hip_greedy_rows_scratch_bytes.
inline constexpr std::uint32_t hip_greedy_rows_max = 8;
struct HipGreedyRowsReply { std::uint32_t token, status; };
[[nodiscard]] std::uint64_t hip_greedy_rows_scratch_bytes(std::uint32_t rows, std::uint32_t vocabulary);
void enqueue_hip_greedy_rows(std::span<const ExecutionBufferAddress> logits, std::uint32_t vocabulary,
    ExecutionBufferAddress partials, HipGreedyRowsReply* replies, void* stream);
} // namespace strix
