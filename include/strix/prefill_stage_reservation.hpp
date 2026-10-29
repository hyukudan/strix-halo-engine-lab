#pragma once
#include <cstdint>
namespace strix {
struct PrefillStageReservation {
    std::uint32_t rows = 0;
    std::uint64_t context = 0;
    std::uint32_t ratio = 0;
    int device = -1;
    bool compact = false;
    constexpr bool fits(const PrefillStageReservation& request) const {
        return rows && request.rows <= rows && request.context <= context &&
            request.ratio == ratio && request.device == device && request.compact == compact;
    }
};
} // namespace strix
