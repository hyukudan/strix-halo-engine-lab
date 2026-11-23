#pragma once
// Backend-neutral plan of the EXACT screened greedy LM head: the device-state records, the plane view, the scratch layout,
// the launch-argument contract and the resident-screen build logic (shape admission, host plane, bound, scratch size).
// No HIP or CUDA type appears here: a backend contributes only kernel bodies and its allocator / copy calls
// (core/hip_screened_head.hpp, cuda/core/head_screened.cuh). The math and the invariant are in core/screened_head.hpp.
#include "core/screened_head.hpp"
#include "strix/env_flags.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <thread>
#include <vector>

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
    // Rows are independent: built on up to 16 host threads (same bytes as one thread).
    const unsigned threads = std::min(16U, std::max(1U, std::thread::hardware_concurrency()));
    plan.plane = screened_head::build_plane(host_weights, width, rows, bits, threads);
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

// Off switch of every binding (EXACT: same token and status either way). STRIX_SCREENED_HEAD=0 restores the full head.
inline bool enabled() { return env_flag_default_on("STRIX_SCREENED_HEAD"); }

// ---- Lanes: several independent activation rows against the same head (cohort lanes, verify rows). ----
// Lane l has its own x (x + l * x_stride), its own State, candidate list, bounds and stored G(y), and its own pick
// (emitted as row l). Pass 1 reads each plane row once for a group of up to kMaxPass1Lanes lanes (the bytes that make the
// multi-row head worth screening); everything after pass 1 is per lane. One lane is the R=1 form, with the same
// scratch addresses as scratch_layout(rows, capacity).
inline constexpr unsigned kMaxLanes = 32U;
inline constexpr unsigned kMaxPass1Lanes = 8U;

// Scratch: State[lanes] | ids[lanes][capacity] | upper[lanes][rows] | y[lanes][max(capacity, rows)] | key2[lanes]; every
// lane's part starts on 256 bytes. Strides are in elements. key2 is the runner-up trait's second-largest lower bound.
struct LanesLayout {
    std::size_t ids, upper, y, key2, bytes;
    std::size_t ids_stride, upper_stride, y_stride;
};
constexpr LanesLayout lanes_layout(unsigned rows, unsigned capacity, unsigned lanes) noexcept {
    const auto up = [](std::size_t v) { return (v + 255U) / 256U * 256U; };
    LanesLayout l{};
    l.ids_stride = up(std::size_t{capacity} * 4U) / 4U;
    l.upper_stride = up(std::size_t{rows} * 4U) / 4U;
    l.y_stride = up(std::size_t{rows > capacity ? rows : capacity} * 4U) / 4U;
    l.ids = up(std::size_t{lanes} * sizeof(State));
    l.upper = l.ids + std::size_t{lanes} * l.ids_stride * 4U;
    l.y = l.upper + std::size_t{lanes} * l.upper_stride * 4U;
    l.key2 = l.y + std::size_t{lanes} * l.y_stride * 4U;
    l.bytes = l.key2 + up(std::size_t{lanes} * 4U);
    return l;
}
static_assert(sizeof(State) == 16U, "State is four 32-bit words (lane states are packed)");
static_assert(lanes_layout(200058U, kCapacity, 1U).ids == scratch_layout(200058U, kCapacity).ids &&
                  lanes_layout(200058U, kCapacity, 1U).upper == scratch_layout(200058U, kCapacity).upper &&
                  lanes_layout(200058U, kCapacity, 1U).y == scratch_layout(200058U, kCapacity).y &&
                  lanes_layout(200058U, kCapacity, 1U).key2 == scratch_layout(200058U, kCapacity).bytes,
              "one lane is the R=1 scratch layout (the runner-up key follows it)");

// Lanes per pass-1 launch row: the largest power of two <= min(lanes, kMaxPass1Lanes) (rounded up for an odd tail)
// whose staged activations (pass1_lanes * width * 4 bytes) fit the backend's shared-memory limit per block. 0: none fits.
constexpr unsigned pass1_lanes(unsigned width, unsigned lanes, std::size_t shared_limit) noexcept {
    if (lanes == 0U || width == 0U) return 0U;
    unsigned n = 1U;
    while (n < lanes && n < kMaxPass1Lanes) n *= 2U;
    while (n != 0U && std::size_t{n} * width * sizeof(float) > shared_limit) n /= 2U;
    return n;
}
constexpr unsigned pass1_groups(unsigned lanes, unsigned per_group) noexcept {
    return per_group == 0U ? 0U : (lanes + per_group - 1U) / per_group;
}

// One enqueue of the screened head over `lanes` activation rows (all pointers are device pointers). pass1_lanes is the
// owner's plan (pass1_lanes() against its device's shared-memory limit, set up once at load), so binding a launch never
// queries a device.
struct Launch {
    PlaneView plane;
    screened_head::Bound bound;
    HeadSpec head;
    const float* x;
    std::uint8_t* scratch;
    unsigned capacity, lanes, x_stride;  // x_stride in floats (ignored for one lane)
    Grids grids;
    unsigned pass1_lanes = 1U;
    // Runner-up trait: candidates are every row that can be the first OR the second (threshold = the second-largest
    // lower bound L(2): two distinct rows reach it, so the true runner-up does, and a row with G(U) < G(L(2)) is
    // strictly below it). The pick may then fold the shared top-2 (head_argmax::Pick2) bit for bit.
    bool runner_up = false;
};

constexpr bool pass1_lanes_valid(unsigned per_group, unsigned lanes) noexcept {
    unsigned cover = 1U;
    while (cover < lanes && cover < kMaxPass1Lanes) cover *= 2U;
    return per_group != 0U && per_group <= cover && (per_group & (per_group - 1U)) == 0U;
}

inline bool launch_valid(const Launch& a) noexcept {
    if (a.lanes == 0U || a.lanes > kMaxLanes || !pass1_lanes_valid(a.pass1_lanes, a.lanes)) return false;
    if (a.lanes > 1U && (a.x_stride < a.head.width || a.x_stride % 4U != 0U)) return false;
    return launch_args_valid(a.plane, a.bound, a.head, a.x, a.scratch, a.capacity, a.grids);
}

// Diagnostics of a finished enqueue (host copy of one lane's State): the candidate count pass 2 ran, or kAllRows-like
// fallback (count > capacity: every row ran).
struct LaneOutcome {
    unsigned candidates;
    bool fallback;
};
inline LaneOutcome outcome(const State& s, unsigned rows, unsigned capacity) noexcept {
    const bool fallback = s.flag != 0U || s.count > capacity;
    return {fallback ? rows : s.count, fallback};
}

// Host log of outcomes (diagnostics of a run: how many rows pass 2 recomputed per pick).
class OutcomeLog {
public:
    void add(const State& s, unsigned rows, unsigned capacity) {
        const auto o = outcome(s, rows, capacity);
        if (o.fallback) ++fallbacks_;
        else candidates_.push_back(o.candidates);
    }
    [[nodiscard]] bool empty() const noexcept { return fallbacks_ == 0U && candidates_.empty(); }
    // "picks N screened S fallback F candidates p50 A p90 B p99 C max D"
    [[nodiscard]] std::string summary() const {
        auto sorted = candidates_;
        std::sort(sorted.begin(), sorted.end());
        const auto at = [&](double q) {
            return sorted.empty() ? 0U : sorted[static_cast<std::size_t>(q * static_cast<double>(sorted.size() - 1U))];
        };
        return "picks " + std::to_string(sorted.size() + fallbacks_) + " screened " + std::to_string(sorted.size()) +
               " fallback " + std::to_string(fallbacks_) + " candidates p50 " + std::to_string(at(0.5)) + " p90 " +
               std::to_string(at(0.9)) + " p99 " + std::to_string(at(0.99)) + " max " +
               std::to_string(sorted.empty() ? 0U : sorted.back());
    }

private:
    std::vector<unsigned> candidates_;
    std::size_t fallbacks_ = 0;
};

}  // namespace strix::core::screened_head_plan
