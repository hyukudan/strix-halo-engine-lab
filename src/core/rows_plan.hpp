#pragma once

#include "strix/core/rows.hpp"

#include <array>

namespace strix::core::rows {

struct RoleFormat {
    weight::Role role;
    weight::FormatId format;
};

// How an MXFP8 job with `splits == 0` gets its count. `incumbent` is the default of every binding: the count of
// the incumbent rows plan (split_k::factor), so the arithmetic is the incumbent tree. `occupancy` is an explicit,
// opt-in numerical identity (kMxfp8OccupancySplitIdentity): the count that fills the device wave target.
enum class SplitPolicy : std::uint8_t { incumbent, occupancy };

struct RoleJob {
    weight::Role role;
    Job job;
    // Zero selects the count of `policy`. A pinned count replays an MXFP8 identity.
    std::uint32_t splits = 0;
    SplitPolicy policy = SplitPolicy::incumbent;
};

struct SplitLimits {
    std::uint32_t max_splits; // backend grid limit along K
};

struct ChunkRange {
    std::uint32_t begin, end;
};

// Contiguous, floor-divided chunk ranges. Each partial starts at +0, and
// partials merge from the first partial by rounded FP32 adds in ascending order.
// The family's partial-dot identity, chunk geometry and split count together
// specify the arithmetic. Output geometry is independent of that identity.
struct SplitTree {
    std::string_view arithmetic;
    std::uint32_t chunk_in, chunks, splits;

    [[nodiscard]] constexpr bool valid() const { return splits != 0 && splits <= chunks && chunk_in != 0; }
    // Precondition: valid() and split < splits.
    [[nodiscard]] constexpr ChunkRange range(std::uint32_t split) const {
        return {static_cast<std::uint32_t>(std::uint64_t{split} * chunks / splits),
                static_cast<std::uint32_t>((std::uint64_t{split} + 1) * chunks / splits)};
    }
    constexpr bool operator==(const SplitTree&) const = default;
};

// Numerical identity of the occupancy-selected MXFP8 tree (SplitPolicy::occupancy); the incumbent tree keeps
// kMxfp8SplitIdentity.
inline constexpr std::string_view kMxfp8OccupancySplitIdentity = "mxfp8-f32-block-dot-splitk-occupancy-v1";

struct PlannedLaunch {
    Launch launch;
    std::array<SplitTree, kMaxJobs> trees; // Empty for unsplit row-tile forms.
    std::array<weight::Role, kMaxJobs> roles;

    [[nodiscard]] bool reaches_wave_target(std::uint64_t target) const { return launch.useful_blocks >= target; }
};

class RowsPlan {
public:
    [[nodiscard]] std::span<const PlannedLaunch> launches() const { return launches_; }
    [[nodiscard]] std::uint32_t rows() const { return rows_; }
    [[nodiscard]] std::uint64_t target_blocks() const { return target_blocks_; }

private:
    RowsPlan() = default;
    friend RowsPlan plan_rows(std::span<const RoleJob>, std::span<const RoleFormat>, std::uint32_t,
                              const device::Profile&, SplitLimits);

    std::vector<PlannedLaunch> launches_;
    std::uint32_t rows_ = 0;
    std::uint64_t target_blocks_ = 0;
};

// Formats are supplied by the caller's resolved weight-role policy. This plan
// takes FP32 activations. An MXFP8 job's count is a function of its shape (and, for the opt-in occupancy policy,
// the device) so grouping or a different row count cannot change its bits. The default is the incumbent count.
[[nodiscard]] RowsPlan plan_rows(std::span<const RoleJob> jobs, std::span<const RoleFormat> formats,
                                 std::uint32_t rows, const device::Profile& device, SplitLimits limits);

// CPU oracle for one output of the declared MXFP8 tree, before any epilogue.
[[nodiscard]] float mxfp8_rows_reference(const SplitTree&, std::span<const std::uint8_t> codes,
                                         std::span<const std::uint8_t> scales, std::span<const float> x);

// Planar block32 activation view. Codes are signed bytes; delta is FP32 and
// x ~= code * delta. Stride is in codes. Delta and optional signed code sums
// use stride / 32 elements per row. Producer owns storage and synchronization.
inline constexpr std::string_view kA8Block32Identity = "a8.s8.delta-f32.block32.planar-v1";
struct A8ReadView {
    const std::int8_t* codes;
    const float* deltas;
    const std::int32_t* sums; // optional for consumers of symmetric Q8_0 weights
    std::uint32_t rows, width, stride;
};

// Validate the draft Q8_0 consumer binding. Admission is explicit from the
// caller's numerical policy; this contract does not resolve a lever.
void validate_a8_consumer(const A8ReadView&, std::uint32_t rows, std::uint32_t width,
                          weight::FormatId weights, weight::Exactness admission);

}  // namespace strix::core::rows
