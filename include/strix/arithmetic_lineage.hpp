#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

enum class ArithmeticProducer { serial_row, prefill_chunk, known_window_chunk };

struct ArithmeticRowInterval {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    ArithmeticProducer producer = ArithmeticProducer::serial_row;
    std::uint64_t numerical_plan_identity = 0;
    std::uint64_t reduction_plan_identity = 0;
    std::uint64_t state_update_identity = 0;

    friend bool operator==(const ArithmeticRowInterval&, const ArithmeticRowInterval&) = default;
};

// Serial intervals compress consecutive one-row calls with identical arithmetic.
// Prefill intervals each represent one complete canonical chunk: cuts are never
// coalesced. A recorded endpoint alone cannot stand in for the preceding history.
struct ArithmeticLineage {
    std::vector<ArithmeticRowInterval> intervals;
    std::uint64_t history_identity = 0;
    bool tainted = false;
};

[[nodiscard]] std::uint64_t arithmetic_history_identity(
    std::span<const ArithmeticRowInterval> intervals) noexcept;
[[nodiscard]] bool arithmetic_lineage_valid(const ArithmeticLineage& lineage,
                                           std::uint64_t resume_position) noexcept;
[[nodiscard]] bool arithmetic_lineage_matches(const ArithmeticLineage& recorded,
                                             const ArithmeticLineage& requested,
                                             std::uint64_t resume_position) noexcept;
[[nodiscard]] bool arithmetic_lineage_serial_only(const ArithmeticLineage& lineage,
                                                 std::uint64_t execution_identity,
                                                 std::uint64_t resume_position) noexcept;

// The serial producer stages this metadata before dispatch and publishes it only
// after state completion. Unknown earlier rows remain unknown, never synthesized.
[[nodiscard]] ArithmeticLineage prepare_serial_row_lineage(
    const ArithmeticLineage& previous, std::uint64_t position,
    std::uint64_t execution_identity, std::uint64_t reduction_plan_identity);

// A known-prefix window is a distinct producer. Preserve a validated prefix,
// append one bounded chunk, and never coalesce adjacent chunk boundaries or
// synthesize unknown earlier producer history.
[[nodiscard]] ArithmeticLineage prepare_prefill_chunk_lineage(
    const ArithmeticLineage& previous, std::uint64_t begin, std::uint64_t end,
    std::uint64_t numerical_plan_identity, std::uint64_t reduction_plan_identity,
    std::uint64_t state_update_identity);

// One actually completed closed known window, distinct from tiled prefill and
// serial rows. Each call retains its full chunk boundary. Unknown prior history
// returns empty; it is never replaced by fabricated per-row frontiers.
[[nodiscard]] ArithmeticLineage prepare_known_window_chunk_lineage(
    const ArithmeticLineage& previous, std::uint64_t begin, std::uint64_t end,
    std::uint64_t numerical_plan_identity, std::uint64_t reduction_plan_identity,
    std::uint64_t state_update_identity);

} // namespace strix
