#pragma once
// Pair schedule: independent work advanced into a staged exchange's exposed wait (schedule only: the same kernels
// on the same inputs, so the same bits). A sequence of `units` (for example a layer's attention chunks) normally
// produces unit u into one buffer just before consuming it. With `banks` output buffers, units [0, ahead()) are
// produced before the exchange's wait, unit u into bank u % banks. A later unit u reuses bank u % banks, last
// written by unit u - banks; the caller produces unit u only after unit u - 1 was consumed in stream order, so the
// bank's previous contents are always consumed before they are overwritten (u - banks <= u - 1).
// Bound users: the DS41 row-split prefill (q_b chunks while the final K/V entries cross,
// src/hip_deepseek41_prefill.hip). The same plan applies to any striped or chunked producer whose consumer waits
// on a peer message (MiniMax exchange_post stripes: units = row stripes, banks = transmit slots).
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strix::xport {

struct AheadPlan {
    std::uint32_t banks = 1;  // output banks the caller allocated (0 is treated as 1)
    std::uint32_t units = 0;  // units of this sequence
    std::uint32_t limit = 1;  // units the caller may produce before the wait (lever value, task parking)

    [[nodiscard]] constexpr std::uint32_t bank_count() const noexcept { return banks == 0U ? 1U : banks; }
    // units produced before the wait (0: none, the incumbent order)
    [[nodiscard]] constexpr std::uint32_t ahead() const noexcept { return std::min({bank_count(), units, limit}); }
    [[nodiscard]] constexpr std::uint32_t bank(std::uint32_t unit) const noexcept { return unit % bank_count(); }
    [[nodiscard]] constexpr bool produced_ahead(std::uint32_t unit) const noexcept { return unit < ahead(); }
    // unit writes a bank that an earlier unit of the sequence used (that unit must be consumed first)
    [[nodiscard]] constexpr bool reuses_bank(std::uint32_t unit) const noexcept { return unit >= bank_count(); }
};

// Banks for a lever value `want` (units ahead; 0 = off) when a sequence has at most `max_units` units and every bank
// beyond the first costs `bank_bytes` of a `budget_bytes` scratch allowance (0: no extra bank). At least 1.
[[nodiscard]] constexpr std::uint32_t ahead_banks(std::uint32_t want, std::uint32_t max_units, std::size_t bank_bytes,
                                                  std::size_t budget_bytes) noexcept {
    std::uint32_t b = std::max<std::uint32_t>(1U, std::min(want, max_units));
    if (bank_bytes != 0U) {
        const std::size_t extra = budget_bytes / bank_bytes;
        if (extra + 1U < b) b = static_cast<std::uint32_t>(extra + 1U);
    }
    return b;
}

// Planning estimate only (never a measured saving): the part of one exposed wait of `gap_ms` that `extra_units`
// advanced units of `unit_ms` each can cover. It is bounded by the gap: waits summed over layers are not savings.
[[nodiscard]] constexpr double ahead_fill_ms(double gap_ms, std::uint32_t extra_units, double unit_ms) noexcept {
    const double work = static_cast<double>(extra_units) * unit_ms;
    return gap_ms <= 0.0 ? 0.0 : (work < gap_ms ? work : gap_ms);
}

} // namespace strix::xport
