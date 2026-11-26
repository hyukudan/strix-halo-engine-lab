#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

// Low-bit resident route switch points of a device: the optional `lowbit_rows` key in its declaration entry or
// selected catalogue profile (docs/devices.md), and the per-process record of the points
// bound to each runtime device ordinal. The device gate binds them with the device (hip_gate.hpp bind_hip); the
// resident projection launchers read them (core/resident_row_plan.hpp). Catalogue points cover group-scale q4/q6.
// {0, 0} = not declared / not bound: one route.
namespace strix::device {

struct LowBitRows {
    std::uint32_t small_rows = 0;   // launches of at most this many rows run the small-M schedule
    std::uint32_t wide_panels = 0;  // small-M launches of >= this many 64-column panels run 17..32 rows as one block
    bool from_profile = false;     // catalogue schedule applies only to the measured arithmetic and storage
};

inline constexpr int kLowBitRowSlots = 16;

namespace detail {
inline constexpr std::uint64_t kLowBitRowsFromProfile = std::uint64_t{1} << 63U;
inline std::array<std::atomic<std::uint64_t>, kLowBitRowSlots>& lowbit_row_slots() noexcept {
    static std::array<std::atomic<std::uint64_t>, kLowBitRowSlots> slots{};
    return slots;
}
}  // namespace detail

// Binding happens with the device, before any launch. Switch points fit the declaration's 16-bit range.
inline bool bind_lowbit_rows(int ordinal, const LowBitRows& rows) noexcept {
    if (ordinal < 0 || ordinal >= kLowBitRowSlots || rows.small_rows > 65535U || rows.wide_panels > 65535U) return false;
    detail::lowbit_row_slots()[static_cast<std::size_t>(ordinal)].store(
        std::uint64_t{rows.small_rows} | (std::uint64_t{rows.wide_panels} << 16U) |
            (rows.from_profile ? detail::kLowBitRowsFromProfile : 0U),
        std::memory_order_relaxed);
    return true;
}

inline LowBitRows bound_lowbit_rows(int ordinal, bool allow_profile = true) noexcept {
    if (ordinal < 0 || ordinal >= kLowBitRowSlots) return {};
    const auto packed = detail::lowbit_row_slots()[static_cast<std::size_t>(ordinal)].load(std::memory_order_relaxed);
    const bool from_profile = (packed & detail::kLowBitRowsFromProfile) != 0U;
    if (from_profile && !allow_profile) return {};
    return {static_cast<std::uint32_t>(packed & 0xffffU), static_cast<std::uint32_t>((packed >> 16U) & 0xffffU), from_profile};
}

}  // namespace strix::device
