#pragma once

#include <cstdint>
#include <limits>

namespace strix::transport::visibility {
inline constexpr std::uint64_t completion_mask = 0xa6d2b8439f5071ceULL;
enum class Stage : std::uint64_t { warm = 1, negative_control = 2, raw_rx = 3, consumer = 4 };

// Fixed-size GPU-written evidence. A zeroed/stale/partly written result cannot
// pass. All payload bytes are compared on the GPU, not inferred from a checksum.
struct Result {
    unsigned long long started = 0, finished = 0, epoch = 0, sequence = 0;
    unsigned long long stage = 0, checked = 0, mismatches = 0;
    unsigned long long first_bad = std::numeric_limits<unsigned long long>::max();
};
static_assert(sizeof(Result) == 64);
struct Expected {
    std::uint64_t nonce = 0, epoch = 0, sequence = 0, bytes = 0;
    Stage stage = Stage::raw_rx;
};
constexpr bool complete(const Result& r, const Expected& e) noexcept {
    return e.nonce && e.epoch && e.bytes && r.started == e.nonce &&
        r.finished == (e.nonce ^ completion_mask) && r.epoch == e.epoch &&
        r.sequence == e.sequence && r.stage == static_cast<std::uint64_t>(e.stage) &&
        r.checked == e.bytes && r.mismatches <= e.bytes &&
        (r.mismatches ? r.first_bad < e.bytes : r.first_bad == std::numeric_limits<unsigned long long>::max());
}
constexpr bool passed(const Result& r, const Expected& e) noexcept {
    return complete(r, e) && r.mismatches == 0;
}
struct Observation {
    bool executed = false;
    Expected expected;
    Result result;
    [[nodiscard]] constexpr bool passed() const noexcept {
        return executed && visibility::passed(result, expected);
    }
};
struct TransferObservation { Observation raw_rx, consumer; };
} // namespace strix::transport::visibility
