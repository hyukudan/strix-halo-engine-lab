#pragma once
// CPU-only optional-pin policy. No CUDA or OS dependencies.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

namespace strix::ds41 {
enum class HostPinReason { none, physical, commit, alloc, telemetry };
inline const char* host_pin_reason_name(HostPinReason r) {
    switch (r) {
    case HostPinReason::physical: return "physical";
    case HostPinReason::commit: return "commit";
    case HostPinReason::alloc: return "alloc";
    case HostPinReason::telemetry: return "telemetry";
    default: return "none";
    }
}
inline bool valid_host_pin_policy(double floor, double wait) {
    return std::isfinite(floor) && floor >= 0 && std::isfinite(wait) && wait >= 0;
}
inline std::chrono::steady_clock::time_point host_pin_deadline(double wait) {
    const auto now = std::chrono::steady_clock::now();
    const auto remaining = std::chrono::steady_clock::time_point::max() - now;
    const long double millis = static_cast<long double>(wait) * 1000;
    if (millis >= std::chrono::duration<long double, std::milli>(remaining).count())
        return std::chrono::steady_clock::time_point::max();
    return now + std::chrono::milliseconds(static_cast<std::int64_t>(millis));
}
inline HostPinReason host_pin_admission(bool known, double physical, double commit, double n, double floor) {
    if (!known) return HostPinReason::telemetry;
    if (physical < n + floor + double(4ULL << 30)) return HostPinReason::physical;
    if (commit < 2 * n + floor) return HostPinReason::commit;
    return HostPinReason::none;
}
struct HostPinStats {
    std::uint64_t requested_bytes = 0, effective_bytes = 0;
    std::vector<std::uint32_t> layers;
    struct Demotion { std::uint32_t layer; HostPinReason reason; };
    std::vector<Demotion> demotions;
};
// The automatic path calls neither grow nor pause, including allocation failure.
// Explicit policy may grow commit/retry, bounded by a shared deadline. No later
// layer starts a probe/allocation after that deadline (wait=0 allows one attempt).
template<class Room, class Alloc, class Grow, class Pause, class Now>
HostPinReason allocate_host_pin(std::uint64_t n, double floor_gib, bool implicit,
        bool allow_first, std::chrono::steady_clock::time_point deadline,
        std::uint32_t& waits, Room room, Alloc alloc, Grow grow, Pause pause, Now now) {
    HostPinReason reason = HostPinReason::commit;
    bool first = true;
    for (;;) {
        if (!implicit && (!first || !allow_first) && now() >= deadline) return reason;
        first = false;
        const auto r = room();
        reason = host_pin_admission(r.known, double(r.free_physical), double(r.free_commit),
                                    double(n), floor_gib * double(1ULL << 30));
        if (reason == HostPinReason::physical || reason == HostPinReason::telemetry) return reason;
        if (reason == HostPinReason::none) {
            if (alloc()) return HostPinReason::none;
            reason = HostPinReason::alloc;
        }
        if (implicit || now() >= deadline) return reason;
        if (reason == HostPinReason::commit) grow(double(r.free_commit) + double(n) + double(8ULL << 30));
        if (now() >= deadline) return reason;
        ++waits;
        pause();
    }
}
} // namespace strix::ds41
