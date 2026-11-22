#pragma once
// Backend-neutral plan of the EXACT screened greedy LM head: the device-state records, the plane view, the scratch layout,
// the launch-argument contract and the resident-screen build logic (shape admission, host plane, bound, scratch size).
// No HIP or CUDA type appears here: a backend contributes only kernel bodies and its allocator / copy calls
// (core/hip_screened_head.hpp, cuda/core/head_screened.cuh). The math and the invariant are in core/screened_head.hpp.
#include "core/screened_head.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace strix::core::screened_head_plan {

inline constexpr unsigned kCompactThreads = 256U;
inline constexpr unsigned kPass1Waves = 8U;      // waves / warps (rows in flight) per pass-1 block
inline constexpr unsigned kAllRows = 0xffffffffU;
inline constexpr unsigned kMaxWidth = 8192U;     // x is staged in shared memory (width * 4 bytes <= 32 KiB)
inline constexpr unsigned kCapacity = 16384U;    // candidate capacity of a resident screen

struct State {
    float nx;
    unsigned flag, count, lkey;
};

// Device view of a built plane (screened_head::Plane uploaded as three buffers).
struct PlaneView {
    const std::uint8_t* data;
    const float* residual;
    const float* amplitude;
    std::size_t row_bytes, scales_offset;
    float max_amplitude;
    unsigned rows, width, bits;
};

struct HeadSpec {
    const std::uint16_t* weights;  // incumbent BF16 rows [>= rows][width], 16-byte aligned
    unsigned rows, width;
    float softcap;
};

struct Grids {
    unsigned pass1 = 640U, compact = 256U, pass2 = 1024U;
};

// Scratch: State | ids[capacity] | upper[rows] | y[max(capacity, rows)], each part 256-byte aligned.
struct ScratchLayout {
    std::size_t ids, upper, y, bytes;
};
constexpr ScratchLayout scratch_layout(unsigned rows, unsigned capacity) noexcept {
    const auto up = [](std::size_t v) { return (v + 255U) / 256U * 256U; };
    ScratchLayout l{};
    l.ids = 256U;
    l.upper = l.ids + up(std::size_t{capacity} * 4U);
    l.y = l.upper + up(std::size_t{rows} * 4U);
    l.bytes = l.y + up(std::size_t{rows > capacity ? rows : capacity} * 4U);
    return l;
}

// True when a screen may be built for this head shape (softcap must be finite and > 0: the transform the screen proves
// monotone).
inline bool shape_supported(const std::uint16_t* device_weights, unsigned rows, unsigned width, float softcap,
                            unsigned bits) noexcept {
    return device_weights && rows != 0U && width <= kMaxWidth && width % 16U == 0U && softcap > 0.0F &&
           std::isfinite(softcap) && screened_head::layout_supported(width, bits) && rows < (1U << 30U);
}

// Launch-argument contract shared by every backend (all pointers are device pointers; alignment is checked on the value).
inline bool launch_args_valid(const PlaneView& plane, const screened_head::Bound& bound, const HeadSpec& head,
                              const float* x, const std::uint8_t* scratch, unsigned capacity, const Grids& grids) noexcept {
    return !(!x || !scratch || !head.weights || capacity == 0U || plane.rows != head.rows ||
             plane.width != head.width || bound.width != head.width || head.width > kMaxWidth ||
             !screened_head::layout_supported(head.width, plane.bits) || head.width % 16U != 0U ||
             reinterpret_cast<std::uintptr_t>(x) % 16U != 0U ||
             reinterpret_cast<std::uintptr_t>(head.weights) % 16U != 0U ||
             reinterpret_cast<std::uintptr_t>(plane.data) % 16U != 0U || plane.row_bytes % 16U != 0U ||
             reinterpret_cast<std::uintptr_t>(scratch) % 256U != 0U || grids.pass1 == 0U || grids.compact == 0U ||
             grids.pass2 == 0U || head.rows >= (1U << 30U));
}

// Host result of the resident-screen build: the plane (to upload as three buffers), the bound, the scratch size.
struct ResidentPlan {
    screened_head::Plane plane;
    screened_head::Bound bound{};
    ScratchLayout layout{};
    bool usable = false;
    [[nodiscard]] std::size_t device_bytes() const noexcept { return plane.bytes() + layout.bytes; }
};

// host_weights: the live rows [rows][width] BF16 bits (copied from the resident head once at load).
inline ResidentPlan plan_resident(std::span<const std::uint16_t> host_weights, unsigned rows, unsigned width,
                                  unsigned bits) {
    ResidentPlan plan;
    plan.plane = screened_head::build_plane(host_weights, width, rows, bits);
    if (!plan.plane.usable) return plan;
    plan.layout = scratch_layout(rows, kCapacity);
    plan.bound = screened_head::make_bound(width);
    plan.usable = true;
    return plan;
}

// The device view of an uploaded plan.
inline PlaneView make_view(const ResidentPlan& plan, const void* data, const void* residual, const void* amplitude,
                           unsigned rows, unsigned width, unsigned bits) noexcept {
    return {static_cast<const std::uint8_t*>(data), static_cast<const float*>(residual),
            static_cast<const float*>(amplitude), plan.plane.layout.row_bytes, plan.plane.layout.scales_offset,
            plan.plane.max_amplitude, rows, width, bits};
}

}  // namespace strix::core::screened_head_plan
