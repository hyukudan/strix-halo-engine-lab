#pragma once

// Device view of the lanes of one multi-request decode step: backend-neutral, no device headers.
//
// Lane l is one query row of its own session. Its absolute position, its slot and its write flag are
// strided device columns (row l at column[l * stride]; the X3 step block projects onto these columns,
// strix/exec/step_lanes.hpp). Per-session state is addressed through a slot directory in the StateTable
// layout (strix/state/table.hpp: state-major, then slot-major, SlotEntry::base): plane p of lane l is
// entries[p * slots + slot(l)].base + offset, where the offset of a visit is the same for every slot of
// one uniform layout, so it is a launch constant.
//
// The view travels by value as a kernel argument and holds no host position or slot address: a launch
// captured once replays for new positions and slot assignments written into the same device columns.
// A lane whose write flag is clear (padding of a wider program) never writes session state; kernels may
// let it read the slot it names so its rows stay finite and deterministic.

#include "strix/state/table.hpp"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_LANE_VIEW_HD __host__ __device__
#else
#define STRIX_LANE_VIEW_HD
#endif

namespace strix::state {

struct LaneView {
    const std::uint32_t* position;  // device column: lane l's absolute position at position[l * stride]
    const std::uint32_t* slot;      // device column: lane l's slot in the directory
    const std::uint32_t* flags;     // device column: bit 0 set = lane l writes its session state
    const SlotEntry* entries;       // device slot directory, state-major then slot-major
    std::uint32_t stride;           // 32-bit words between consecutive lanes of a column
    std::uint32_t slots;            // slots per state in the directory
    std::uint32_t lanes;            // lanes of the launch (program width, padding included)
};

inline constexpr std::uint32_t kLaneWrites = 1U;

// One state plane of a visit: directory state index plus the byte offset of the visit inside a slot.
struct LanePlane {
    std::uint32_t state;
    std::uint64_t offset;
};

[[nodiscard]] STRIX_LANE_VIEW_HD inline bool lane_writes(const LaneView& view, unsigned lane) {
    return (view.flags[static_cast<std::size_t>(lane) * view.stride] & kLaneWrites) != 0U;
}

[[nodiscard]] STRIX_LANE_VIEW_HD inline std::uint32_t lane_position(const LaneView& view, unsigned lane) {
    return view.position[static_cast<std::size_t>(lane) * view.stride];
}

template <class T>
[[nodiscard]] STRIX_LANE_VIEW_HD inline T* lane_plane(const LaneView& view, LanePlane plane, unsigned lane) {
    const auto slot = view.slot[static_cast<std::size_t>(lane) * view.stride];
    const auto& entry = view.entries[static_cast<std::size_t>(plane.state) * view.slots + slot];
    return reinterpret_cast<T*>(static_cast<std::uintptr_t>(entry.base + plane.offset));
}

// Typed state accessor for a lane-batched kernel: the state of lane l and whether lane l may write it.
// Generic lane kernels (short convolution, cache store) take any type of this shape.
template <class T>
struct LaneState {
    LaneView view;
    LanePlane plane;
    [[nodiscard]] STRIX_LANE_VIEW_HD T* operator()(unsigned lane) const { return lane_plane<T>(view, plane, lane); }
    [[nodiscard]] STRIX_LANE_VIEW_HD bool writes(unsigned lane) const { return lane_writes(view, lane); }
};

}  // namespace strix::state

#undef STRIX_LANE_VIEW_HD
