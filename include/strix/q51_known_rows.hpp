#pragma once

#include "strix/execution_memory.hpp"
#include "strix/known_prefill_limits.hpp"
#include <cstdint>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace strix::q51_known_rows {
inline constexpr std::string_view organization_identity =
    "q51-known-rows-csr-v1:row-slot-stable:virtual256:ordered-slot-fma";
// Last submitted eligible group only. This is GPU-produced metadata, not a
// traffic/timing counter. The owner may copy it explicitly at its existing
// completion boundary; host return from launch does not make it readable.
// Histograms use index n-1 for n=1..8 (unique source rows/expert and batch size).
struct Summary {
    std::uint32_t status = 0, touched = 0, batches = 0, assignments = 0;
    std::uint32_t duplicate_assignments = 0, maximum_assignments = 0, logical_rows = 0, version = 1;
    std::uint32_t rows_per_expert_histogram[8]{};
    std::uint32_t batch_size_histogram[8]{};
    std::uint32_t reserved[8]{};
};
static_assert(sizeof(Summary) == 128U);
static_assert(offsetof(Summary, status) == 0U && offsetof(Summary, batch_size_histogram) == 64U);
// All counts are bounded before arithmetic. Offsets index uint32 payloads;
// the host MoeGroupingPlan uses uint64 offsets with the same CSR semantics.
struct ScratchLayout {
    std::uint64_t header = 0, offsets = 0, touched = 0, rows = 0, slots = 0;
    std::uint64_t batch_experts = 0, batch_begins = 0, bytes = 0;
};
// Known-prompt prefill: one CSR over all T rows of a step (T a multiple of 8).
inline constexpr std::uint32_t prefill_max_rows = known_prefill_max_rows;
[[nodiscard]] inline ScratchLayout scratch_layout(std::uint32_t rows,
    std::uint32_t experts, std::uint32_t top_k, bool prefill = false) {
    if ((prefill ? rows < 16U || rows % 8U || rows > prefill_max_rows : rows != 2U && rows != 4U && rows != 8U) ||
        !experts || experts > 512U || !top_k || top_k > 10U || top_k > experts)
        throw std::invalid_argument("Q5_1 known rows CSR geometry is invalid");
    const auto align = [](std::uint64_t n) { return (n + 255U) / 256U * 256U; };
    ScratchLayout out;
    out.offsets = 256U;
    out.touched = out.offsets + align(4ULL * (experts + 1U));
    out.rows = out.touched + align(4ULL * experts);
    out.slots = out.rows + align(4ULL * rows * top_k);
    out.batch_experts = out.slots + align(4ULL * rows * top_k);
    out.batch_begins = out.batch_experts + align(4ULL * rows * top_k);
    out.bytes = out.batch_begins + align(4ULL * (rows * top_k + 1U));
    return out;
}
[[nodiscard]] inline std::uint64_t required_scratch_bytes(std::uint32_t rows,
    std::uint32_t experts, std::uint32_t top_k) {
    return scratch_layout(rows, experts, top_k).bytes;
}
[[nodiscard]] inline std::uint64_t prefill_scratch_bytes(std::uint32_t rows,
    std::uint32_t experts, std::uint32_t top_k) {
    return scratch_layout(rows, experts, top_k, true).bytes;
}
// An exclusive allocation, never a slice of window scratch/state/weights/D1.
// The actual native window owner creates and leases this allocation. This
// strong owner retains it; its independent generation identifies rebinding.
// No call is complete until its stream has quiesced, including on exceptions.
struct Workspace {
    ExecutionMemoryBinding memory;
    std::shared_ptr<const void> owner;
};
} // namespace strix::q51_known_rows
