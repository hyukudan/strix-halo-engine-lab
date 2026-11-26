#pragma once
// Backend-neutral plan of the EXACT screened greedy LM head: the device-state records, the plane view, the scratch layout,
// the launch-argument contract and the resident-screen build logic (shape admission, host plane, bound, scratch size).
// No HIP or CUDA type appears here: a backend contributes only kernel bodies and its allocator / copy calls
// (core/hip_screened_head.hpp, cuda/core/head_screened.cuh). The math and the invariant are in core/screened_head.hpp.
// The rescore trait (end of this file) is the FAST form: exact rows for the largest scores of an approximate head.
#include "strix/device/decode_geometry.hpp"
#include "core/byte_range.hpp"
#include "core/screened_head.hpp"
#include "strix/env_flags.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
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

// ---- Pass-1 schedule (backend-neutral; the CUDA and HIP bodies call these from device code). ----
// The schedule only decides which warp computes which (row, lane) bound; every bound is computed by the same per-row
// arithmetic (thread t of the warp folds units t, t + 32, ... in order, then the warp sum), so y_hat, U, Lstar and the
// runner-up key are bitwise independent of it.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_SHP_HD __host__ __device__
#else
#define STRIX_SHP_HD
#endif

// Row tile: plane rows one warp carries at once. The activations a thread reads from shared memory for one unit serve
// all of them, so shared-memory traffic per weight is per_group / tile_rows activations instead of per_group (the
// limit of a group of 8 lanes with one row per warp: 32 B of LDS per lane per 8 weights).
STRIX_SHP_HD constexpr unsigned pass1_tile_rows(unsigned per_group) noexcept {
    return per_group >= 2U ? per_group / 2U : 1U;
}

// Pass-1 blocks: one wave of the instantiation (resident = blocks the device holds at once, 0 when unknown) capped by
// the owner's grid, rounded down to whole lane groups (at least one block per group). Lane groups are folded into the
// block index (block % groups), so the sibling groups of one row block run at the same time over the same plane rows:
// the plane is read from memory once and the other groups find it in the last-level cache.
constexpr unsigned pass1_blocks(unsigned cap, unsigned resident, unsigned groups) noexcept {
    if (groups == 0U) return 0U;
    unsigned blocks = resident != 0U && resident < cap ? resident : cap;
    blocks = blocks / groups * groups;
    return blocks < groups ? groups : blocks;
}

struct Pass1Map {
    unsigned group, row_block, row_blocks;  // lane group, this block's row block, row blocks per group
};
STRIX_SHP_HD constexpr Pass1Map pass1_map(unsigned block, unsigned blocks, unsigned groups) noexcept {
    return {block % groups, block / groups, blocks / groups};
}
// Warp `warp` of a block visits row tiles first_tile, first_tile + tile_step, ... (tile t = rows [t * tile_rows,
// (t + 1) * tile_rows) of the plane, the tail clipped at rows).
STRIX_SHP_HD constexpr unsigned pass1_first_tile(const Pass1Map& m, unsigned warp) noexcept {
    return m.row_block * kPass1Waves + warp;
}
STRIX_SHP_HD constexpr unsigned pass1_tile_step(const Pass1Map& m) noexcept { return m.row_blocks * kPass1Waves; }

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
    // Pass-1 blocks the device holds at once for this launch's instantiation (SMs x blocks per SM, measured by the
    // owner at load); 0: unknown, grids.pass1 alone sizes the grid. See pass1_blocks().
    unsigned pass1_resident = 0U;
};

// The pass-1 launch of `a`: a 1-D grid (lane groups folded into the block index, pass1_blocks) of kPass1Waves warps,
// each block staging pass1_lanes rows of x. The one source of that geometry: the CUDA launcher enqueues it and the exec
// catalog binds it (a strict graph capture compares the two node by node).
struct Pass1Grid {
    unsigned blocks, threads;
    std::size_t shared_bytes;
};
inline Pass1Grid pass1_grid(const Launch& a) noexcept {
    return {pass1_blocks(a.grids.pass1, a.pass1_resident, pass1_groups(a.lanes, a.pass1_lanes)), kPass1Waves * 32U,
            std::size_t{a.pass1_lanes} * a.head.width * sizeof(float)};
}

inline device::LaunchGeometry lane_grid(const Launch& launch) {
    return device::decode_geometry::row_groups(launch.lanes);
}
inline device::LaunchGeometry compact_grid(const Launch& launch) {
    return device::decode_geometry::row_groups(launch.grids.compact, launch.lanes, kCompactThreads);
}
inline device::LaunchGeometry pass2_grid(const Launch& launch) {
    return device::decode_geometry::mapped_dense(launch.grids.pass2, launch.lanes);
}

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

// ---- Rescore trait: exact rows for the largest scores of an approximate head (a screen, not a bound). ----
// Input: scores [lanes][pitch] already written by an approximate head (e.g. a low-bit image of the same rows) with the
// head's post-transform applied. Per lane the trait selects S = { i : key(s_i) >> 8 >= key(s_(k)) >> 8 }, where key is
// the order-preserving key below and s_(k) the k-th largest score: every row of the exact top-k plus the rows sharing
// the k-th one's 24-bit key prefix (so S never depends on the visit order). Each row of S is recomputed from the
// incumbent BF16 rows with the incumbent tight-row body and the incumbent post-transform, and written back in place:
// a selected row holds the bits of the full incumbent head; every other row keeps its approximate score. |S| > capacity
// (degenerate scores: many equal keys) recomputes every row, i.e. the incumbent head. Not EXACT: the order of the
// approximate scores decides S. The selection is two passes of 12-bit digits (key >> 20, then (key >> 8) & 0xfff).
// All NaNs share the highest key, so every NaN is selected. Signed zeros are a numeric tie.
inline constexpr unsigned kRescoreBins = 4096U;
inline constexpr unsigned kRescoreMaxK = 4096U;
inline constexpr unsigned kRescorePass2Threads = 128U;

#if defined(__HIPCC__) || defined(__CUDACC__)
__host__ __device__
#endif
constexpr std::uint32_t rescore_order_key_bits(std::uint32_t b) noexcept {
    const auto magnitude = b & 0x7fffffffU;
    if (magnitude > 0x7f800000U) return 0xffffffffU;
    if (magnitude == 0U) return 0x80000000U;
    return (b & 0x80000000U) != 0U ? ~b : (b | 0x80000000U);
}

constexpr std::uint32_t order_key(float v) noexcept {
    return rescore_order_key_bits(std::bit_cast<std::uint32_t>(v));
}

// The selection threshold of one lane (host reference of the device digits): the k-th largest key with its low 8 bits
// cleared; 0 (every row) when the lane has fewer than k rows.
inline std::uint32_t rescore_threshold(std::span<const float> scores, unsigned k) {
    if (k == 0U || scores.size() < k) return 0U;
    std::vector<std::uint32_t> keys(scores.size());
    std::transform(scores.begin(), scores.end(), keys.begin(), [](float v) { return order_key(v) >> 8U; });
    const auto nth = keys.begin() + static_cast<std::ptrdiff_t>(k - 1U);
    std::nth_element(keys.begin(), nth, keys.end(), std::greater<>{});
    return *nth << 8U;
}

// Per lane state: the first digit, the rank still needed inside it, the threshold key, the selected count.
struct RescoreState {
    unsigned digit, need, key, count;
};
static_assert(sizeof(RescoreState) == 16U, "rescore state is four 32-bit words");

// Scratch: RescoreState[lanes] | hist[lanes][2][kRescoreBins] | ids[lanes][capacity] | y[lanes][max(capacity, rows)].
// The state and histogram parts (`cleared` bytes from the start) are zeroed by every enqueue before its first launch.
struct RescoreLayout {
    std::size_t hist, ids, y, cleared, bytes;
    std::size_t hist_stride, ids_stride, y_stride;
};
constexpr RescoreLayout rescore_layout(unsigned rows, unsigned capacity, unsigned lanes) noexcept {
    if (lanes == 0U || lanes > kMaxLanes) return {};
    const auto up = [](std::uint64_t v) { return (v + 255U) / 256U * 256U; };
    constexpr std::size_t hist_stride = 2U * kRescoreBins;
    const auto ids_bytes = up(std::uint64_t{capacity} * 4U);
    const auto y_bytes = up(std::uint64_t{std::max(rows, capacity)} * 4U);
    const auto hist = up(std::uint64_t{lanes} * sizeof(RescoreState));
    const auto ids = hist + std::uint64_t{lanes} * hist_stride * 4U;
    const auto y = ids + lanes * ids_bytes;
    const auto bytes = y + lanes * y_bytes;
    if (bytes > std::numeric_limits<std::size_t>::max()) return {};
    return {static_cast<std::size_t>(hist), static_cast<std::size_t>(ids), static_cast<std::size_t>(y),
            static_cast<std::size_t>(ids), static_cast<std::size_t>(bytes), hist_stride,
            static_cast<std::size_t>(ids_bytes / 4U), static_cast<std::size_t>(y_bytes / 4U)};
}

// Blocks per lane of the histogram and compaction passes (each thread visits 8 rows of a 256-thread block).
constexpr unsigned rescore_blocks(unsigned rows) noexcept {
    return rows == 0U ? 1U : std::min(128U, 1U + (rows - 1U) / 2048U);
}

// One enqueue of the rescore trait over `lanes` rows of scores (all pointers are device pointers).
// The required prefix of scratch must be disjoint from weights, activations and the score image (including pitch
// tails). The enqueue owns this prefix exclusively until stream completion. The score image (including pitch tails) is
// rewritten in place, so it must be disjoint from the weights and the activations too.
struct RescoreLaunch {
    HeadSpec head;          // incumbent BF16 rows and the post-transform cap (> 0)
    const float* x;         // activations [lanes][x_stride], 16-byte aligned
    float* scores;          // [lanes][pitch]: approximate scores in, selected rows replaced in place
    std::uint8_t* scratch;  // rescore_layout(head.rows, capacity, lanes).bytes
    unsigned lanes, x_stride, pitch, k, capacity;
    unsigned pass2_blocks = 1024U;
    std::size_t scratch_bytes = 0U;  // available bytes at scratch
};

inline bool rescore_launch_valid(const RescoreLaunch& a) noexcept {
    const auto aligned = [](const void* p, std::uintptr_t n) {
        return p != nullptr && reinterpret_cast<std::uintptr_t>(p) % n == 0U;
    };
    if (a.lanes == 0U || a.lanes > kMaxLanes || a.k == 0U || a.k > kRescoreMaxK || a.capacity < a.k ||
        a.capacity > kCapacity || a.pass2_blocks == 0U ||
        a.pass2_blocks > std::numeric_limits<unsigned>::max() / kRescorePass2Threads)
        return false;
    // The mapped BF16 body consumes 16 columns at a time. The incumbent row bound leaves room for
    // grid-stride increments, histogram totals and rounding the all-row mapped launch.
    if (a.head.rows == 0U || a.head.rows >= (1U << 30U) ||
        a.head.width == 0U || a.head.width > kMaxWidth || a.head.width % 16U != 0U ||
        !(a.head.softcap > 0.0F) || !std::isfinite(a.head.softcap))
        return false;
    if (a.pitch < a.head.rows || (a.lanes > 1U && (a.x_stride < a.head.width || a.x_stride % 4U != 0U))) return false;
    if (!aligned(a.head.weights, 16U) || !aligned(a.x, 16U) || !aligned(a.scores, 4U) || !aligned(a.scratch, 256U))
        return false;
    const auto bytes = rescore_layout(a.head.rows, a.capacity, a.lanes).bytes;
    if (bytes > a.scratch_bytes) return false;
    const auto weight_bytes = std::uint64_t{a.head.rows} * a.head.width * sizeof(std::uint16_t);
    const auto activation_bytes = (std::uint64_t{a.lanes - 1U} * a.x_stride + a.head.width) * sizeof(float);
    const auto score_bytes = std::uint64_t{a.lanes} * a.pitch * sizeof(float);
    constexpr auto max_size = std::numeric_limits<std::size_t>::max();
    if (weight_bytes > max_size || activation_bytes > max_size || score_bytes > max_size) return false;
    const ByteRange weights{a.head.weights, static_cast<std::size_t>(weight_bytes)};
    const ByteRange activations{a.x, static_cast<std::size_t>(activation_bytes)};
    const ByteRange scores{a.scores, static_cast<std::size_t>(score_bytes)};
    const ByteRange scratch{a.scratch, bytes};
    return byte_range_valid(weights) && byte_range_valid(activations) && byte_range_valid(scores) &&
           byte_range_valid(scratch) && reads_disjoint_from_writes({weights, activations}, {scores, scratch}) &&
           !byte_ranges_overlap(scores, scratch);
}

}  // namespace strix::core::screened_head_plan
