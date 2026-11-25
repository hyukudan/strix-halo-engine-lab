#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

// Low-bit resident route switch points of a device: the optional `lowbit_rows` key of its declaration entry
// (strix/device/declaration.hpp Descriptor::lowbit_rows, docs/devices.md), and the per-process record of the points
// bound to each runtime device ordinal. The device gate binds them with the device (hip_gate.hpp bind_hip); the
// resident projection launchers read them (core/resident_row_plan.hpp). Speed only: every route gives the same bits.
// {0, 0} = not declared / not bound: one route.
namespace strix::device {

struct LowBitRows {
    std::uint32_t small_rows = 0;   // launches of at most this many rows run the small-M schedule
    std::uint32_t wide_panels = 0;  // small-M launches of >= this many 64-column panels run 17..32 rows as one block
};

inline constexpr int kLowBitRowSlots = 16;

namespace detail {
inline std::array<std::atomic<std::uint64_t>, kLowBitRowSlots>& lowbit_row_slots() noexcept {
    static std::array<std::atomic<std::uint64_t>, kLowBitRowSlots> slots{};
    return slots;
}
}  // namespace detail

// Binding happens with the device, before any launch; false (nothing bound) for an ordinal out of range.
inline bool bind_lowbit_rows(int ordinal, const LowBitRows& rows) noexcept {
    if (ordinal < 0 || ordinal >= kLowBitRowSlots) return false;
    detail::lowbit_row_slots()[static_cast<std::size_t>(ordinal)].store(
        std::uint64_t{rows.small_rows} | std::uint64_t{rows.wide_panels} << 32U, std::memory_order_relaxed);
    return true;
}

inline LowBitRows bound_lowbit_rows(int ordinal) noexcept {
    if (ordinal < 0 || ordinal >= kLowBitRowSlots) return {};
    const auto packed = detail::lowbit_row_slots()[static_cast<std::size_t>(ordinal)].load(std::memory_order_relaxed);
    return {static_cast<std::uint32_t>(packed & 0xffffffffU), static_cast<std::uint32_t>(packed >> 32U)};
}

}  // namespace strix::device
