#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strix::ds41::prefill_tuning {

inline bool enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

// Rows of the prefill run in progress (0: unknown), set by the prefill forward at the start of each run.
inline std::atomic<std::uint32_t>& run_rows() {
    static std::atomic<std::uint32_t> rows{0U};
    return rows;
}
inline void set_run_rows(std::uint32_t rows) { run_rows().store(rows, std::memory_order_relaxed); }

// STRIX_DS41_PREFILL_TILES=1 (opt-in; off by default): the unsplit prefill tiles, for runs of at least
// STRIX_DS41_PREFILL_TILES_MIN_ROWS rows (default 2048; below that they leave too few CTAs per expert). Measured on the
// current kernels, tiles on vs off: fresh-process TTFT 2K +1.0 %, 8K +3.4 %, 32K +1.9 % (order-balanced pair); same-process
// A/B 4K +0.89 %, 8K +0.66 %, 16K +0.82 %: slower or equal, so the split-K kernels are the default. Same bits either way.
inline bool tiles() {  // read on every call: tests switch it within one process
    const char* value = std::getenv("STRIX_DS41_PREFILL_TILES");
    const bool on = value && std::strcmp(value, "1") == 0;
    const char* min = std::getenv("STRIX_DS41_PREFILL_TILES_MIN_ROWS");
    const std::uint32_t min_rows = min && *min ? static_cast<std::uint32_t>(std::strtoul(min, nullptr, 10)) : 2048U;
    const auto rows = run_rows().load(std::memory_order_relaxed);
    return on && (rows == 0U || rows >= min_rows);
}

inline unsigned tile_override(const char* name, unsigned selected) {
    if (!tiles()) return selected;
    const char* value = std::getenv(name);
    if (!value || !*value) return selected;
    if (std::strcmp(value, "16") == 0) return 16U;
    if (std::strcmp(value, "32") == 0) return 32U;
    throw std::invalid_argument("ds41 prefill tile override must be 16 or 32");
}

// Resource-based candidates, not a measured performance table. Keep opt-in until
// the complete workload campaign selects a policy on the target device.
constexpr unsigned attention_rows(unsigned positions, unsigned chunk_rows, unsigned heads,
                                  unsigned context_positions = 0U, unsigned lanes = 4U) {
    unsigned rows = positions >= 8192U && heads >= 32U ? 256U : positions >= 1024U ? 128U : 64U;
    const std::uint64_t context = context_positions ? context_positions : positions;
    // The current caller allocates partials for 64 heads even with head TP.
    // Bound all attention-lane scratch to 2 GiB (scores + keys + query prep +
    // partials + tickets). Candidate masks and row buffers are separate.
    const auto scratch = [&](unsigned r) {
        return std::uint64_t{r} * (64U * 10U * 514U * 4U + 8U * (context + 4U * r + 1U) + 4736U + 32U);
    };
    while (rows > 1U && (rows > chunk_rows || chunk_rows % rows != 0U ||
                         scratch(rows) * lanes > (std::uint64_t{2} << 30U))) rows /= 2U;
    return rows;
}

constexpr unsigned dense_outputs(unsigned width, unsigned outputs, unsigned rows) {
    // A 16-output tile halves the live class accumulators for long K chains.
    return width >= 4096U && outputs >= 512U && rows >= 16U ? 16U : 32U;
}

constexpr unsigned moe_rows(unsigned hidden, unsigned down_width, unsigned rows, unsigned top_k, unsigned experts) {
    // Two row tiles amortise weight decoding only when there are enough entries
    // and both activation stages are small. Long-K shapes keep the split tile.
    return hidden <= 1280U && down_width <= 1280U &&
                   std::uint64_t{rows} * top_k >= std::uint64_t{experts} * 32U ? 32U : 16U;
}

constexpr bool down_stage(unsigned width, unsigned outputs) {
    return width <= 1280U && outputs >= 512U;
}

}  // namespace strix::ds41::prefill_tuning
