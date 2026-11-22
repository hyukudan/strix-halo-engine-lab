#pragma once

#include "strix/decode_projection_group.hpp"
#include "strix/device/profiles.hpp"
#include "strix/weight/types.hpp"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::weight {

// Bound, borrowed row-major image. The format name and image must outlive uses
// of the descriptor (including any plan that copies it).
struct ViewRef {
    const void* data = nullptr;
    FormatId format{};

    [[nodiscard]] bool has_format(FormatId other) const {
        return format.revision == other.revision && format.name == other.name;
    }
};

}  // namespace strix::weight

namespace strix::core::rows {

using ChainId = std::uint16_t;  // 0 is the empty chain

struct BufferRef {
    const void* data = nullptr;
};

struct Job {
    weight::ViewRef w;
    std::uint32_t out = 0, in = 0;
    BufferRef x, y;  // y must refer to writable storage
    ChainId epilogue = 0;
    BufferRef aux[2];
};

// All inputs are ready together. Outputs are disjoint and no job consumes
// another job's output. A job can use its own output as its residual.
struct Group {
    std::span<const Job> jobs;
    std::uint32_t rows = 0;
    ChainId prologue = 0;
};

// First binding of F1: the row-tile tensor form, with a single CTA per output tile.
inline constexpr unsigned kMaxJobs = 8;
inline constexpr std::uint32_t kMaxRows = 32;
inline constexpr std::uint32_t kTileOut = 16;
inline constexpr std::uint32_t kBlockIn = 32;
inline constexpr std::uint32_t kTargetWaves = 2;

struct Launch {
    Job jobs[kMaxJobs];
    decode::GroupIndex<kMaxJobs> tiles;
};

class Plan {
public:
    [[nodiscard]] std::span<const Launch> launches() const { return launches_; }
    [[nodiscard]] std::uint32_t rows() const { return rows_; }
    [[nodiscard]] ChainId prologue() const { return prologue_; }
    [[nodiscard]] std::uint64_t target_tiles() const { return target_tiles_; }

private:
    Plan() = default;
    friend Plan plan(const Group&, const device::Profile&);

    std::vector<Launch> launches_;
    std::uint64_t target_tiles_ = 0;
    std::uint32_t rows_ = 0;
    ChainId prologue_ = 0;
};

// Pack in input order until the target wave count is reached. Format changes,
// the job-table limit and the end of the group can close a smaller launch.
// Jobs are indivisible. Undersized launches retain the one-CTA tile form.
[[nodiscard]] inline Plan plan(const Group& group, const device::Profile& device) {
    if (group.jobs.empty() || group.rows == 0 || group.rows > kMaxRows || device.units == 0)
        throw std::invalid_argument("rows: expected jobs, 1..32 rows and a nonzero device unit count");

    Plan result;
    result.rows_ = group.rows;
    result.prologue_ = group.prologue;
    result.target_tiles_ = static_cast<std::uint64_t>(device.units) * kTargetWaves;
    for (const auto& job : group.jobs) {
        if (!job.w.data || job.w.format.name.empty() || !job.x.data || !job.y.data ||
            job.out == 0 || job.out % kTileOut != 0 || job.in == 0 || job.in % kBlockIn != 0)
            throw std::invalid_argument("rows: invalid row-tile operands or dimensions");

        if (result.launches_.empty() ||
            result.launches_.back().tiles.count == kMaxJobs ||
            result.launches_.back().tiles.total() >= result.target_tiles_ ||
            !result.launches_.back().jobs[0].w.has_format(job.w.format))
            result.launches_.emplace_back();

        auto& launch = result.launches_.back();
        const auto member = launch.tiles.count;
        launch.tiles.add(job.out / kTileOut);
        launch.jobs[member] = job;
    }
    return result;
}

}  // namespace strix::core::rows
