#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strix::decode_cli {
struct MtpWindowMemoryReservation {
    std::uint64_t retained_window_bytes = 0;
    std::uint64_t verifier_budget_bytes = 0;
};

// The window owns its arena, eight-byte D1 record and optional union scratch.
// Both capture pools and the checkpoint must fit in the remaining total budget.
inline MtpWindowMemoryReservation reserve_mtp_window_memory(
    std::uint64_t arena_bytes, std::uint64_t arena_budget,
    std::uint64_t total_budget, std::uint64_t union_bytes = 0,
    std::uint64_t union_budget = 0) {
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    if (!arena_bytes || arena_bytes > maximum - 8U ||
        union_bytes > 65536U || union_bytes > union_budget)
        throw std::invalid_argument("MTP window arena/union payload is invalid or exceeds its explicit union budget");
    const auto arena_and_status = arena_bytes + 8U;
    if (union_bytes > maximum - arena_and_status)
        throw std::invalid_argument("MTP window arena/union retained payload overflows");
    const auto retained = arena_and_status + union_bytes;
    if (retained > arena_budget || retained >= total_budget)
        throw std::invalid_argument("MTP window arena/completion/union exceeds its explicit arena or total budget");
    return {retained, total_budget - retained};
}
} // namespace strix::decode_cli
