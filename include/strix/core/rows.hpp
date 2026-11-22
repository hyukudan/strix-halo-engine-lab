#pragma once

#include "strix/decode_projection_group.hpp"
#include "strix/device/profiles.hpp"
#include "strix/split_k_policy.hpp"
#include "strix/weight/types.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix::weight {

// Bound, borrowed weight image. The format name and image must outlive uses
// of the descriptor (including any plan that copies it).
struct ViewRef {
    const void* data = nullptr;
    FormatId format{};
    const void* scales = nullptr;  // MXFP8 E8M0 plane
    float outer_scale = 1.0f;     // NVFP4 tensor scale

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

enum class Packing : std::uint8_t { wave_batches, fewest_launches };

// All inputs are ready together. Outputs are disjoint and no job consumes
// another job's output. A job can use its own output as its residual.
struct Group {
    std::span<const Job> jobs;
    std::uint32_t rows = 0;
    ChainId prologue = 0;
    Packing packing = Packing::wave_batches;
};

// NVFP4 is the 1152-byte tile record image. MXFP8 has contiguous row-major
// E4M3 codes and a separate row-major E8M0 scale plane.
inline constexpr weight::FormatId kNvfp4Records{"cuda.nvfp4.records", 1};
inline constexpr weight::FormatId kMxfp8Rows{"mxfp8.e4m3.e8m0.rows", 1};
inline constexpr std::string_view kMxfp8SplitIdentity = "mxfp8-f32-block-dot-splitk-v1";

inline constexpr unsigned kMaxJobs = 8;
inline constexpr std::uint32_t kMaxRows = 32;
inline constexpr std::uint32_t kTileOut = 16;
inline constexpr std::uint32_t kBlockIn = 32;
inline constexpr std::uint32_t kTargetWaves = 2;

enum class Form : std::uint8_t { row_tile, nvfp4_split_k, mxfp8_split_k };

[[nodiscard]] constexpr std::uint32_t output_blocks(std::uint32_t tiles, std::uint32_t tiles_per_block) {
    return (tiles + tiles_per_block - 1) / tiles_per_block;
}

struct Launch {
    Job jobs[kMaxJobs];
    decode::GroupIndex<kMaxJobs> tiles;
    Form form = Form::row_tile;
    std::uint32_t tiles_per_block = 1;
    std::uint32_t chunk_in = kBlockIn;
    std::uint32_t splits[kMaxJobs] = {};
    std::uint64_t useful_blocks = 0;

    [[nodiscard]] std::string_view split_identity() const {
        if (form == Form::nvfp4_split_k) return split_k::kExactName;
        if (form == Form::mxfp8_split_k) return kMxfp8SplitIdentity;
        return {};
    }
};

class Plan {
public:
    [[nodiscard]] std::span<const Launch> launches() const { return launches_; }
    [[nodiscard]] std::uint32_t rows() const { return rows_; }
    [[nodiscard]] ChainId prologue() const { return prologue_; }
    [[nodiscard]] std::uint64_t target_tiles() const { return target_tiles_; }
    // CTA work across the group's launches. Empty split-grid cells do not
    // count, and a format boundary can keep an individual launch below target.
    [[nodiscard]] std::uint64_t useful_blocks() const {
        std::uint64_t count = 0;
        for (const auto& launch : launches_) count += launch.useful_blocks;
        return count;
    }
    [[nodiscard]] bool reaches_wave_target() const { return useful_blocks() >= target_tiles_; }
    // Retain the original plan metadata when handing a homogeneous launch to
    // a backend entry that consumes a Plan.
    [[nodiscard]] Plan select_launch(std::size_t index) const {
        Plan selected;
        selected.rows_ = rows_;
        selected.prologue_ = prologue_;
        selected.target_tiles_ = target_tiles_;
        selected.launches_.push_back(launches_.at(index));
        return selected;
    }

private:
    Plan() = default;
    friend Plan plan(const Group&, const device::Profile&);

    std::vector<Launch> launches_;
    std::uint64_t target_tiles_ = 0;
    std::uint32_t rows_ = 0;
    ChainId prologue_ = 0;
};

// Keep input order. Wave batches close after reaching the device target.
// Fewest launches keeps appending compatible jobs up to the job-table limit,
// so reaching the target does not strand independent jobs in a short tail.
// A decoder change still starts a new launch. Split counts are independent of
// group membership and row count.
[[nodiscard]] inline Plan plan(const Group& group, const device::Profile& device) {
    if (group.jobs.empty() || group.rows == 0 || group.rows > kMaxRows || device.units == 0)
        throw std::invalid_argument("rows: expected jobs, 1..32 rows and a nonzero device unit count");

    Plan result;
    result.rows_ = group.rows;
    result.prologue_ = group.prologue;
    result.target_tiles_ = static_cast<std::uint64_t>(device.units) * kTargetWaves;
    for (const auto& job : group.jobs) {
        const bool nvfp4 = job.w.has_format(kNvfp4Records);
        const bool mxfp8 = job.w.has_format(kMxfp8Rows);
        const auto chunk_in = nvfp4 ? split_k::Geometry{}.chunk_in : kBlockIn;
        if (!job.w.data || job.w.format.name.empty() || !job.x.data || !job.y.data ||
            job.out == 0 || job.out % kTileOut != 0 || job.in == 0 || job.in % chunk_in != 0 ||
            (mxfp8 && !job.w.scales))
            throw std::invalid_argument("rows: invalid row-tile operands or dimensions");

        if (result.launches_.empty() ||
            result.launches_.back().tiles.count == kMaxJobs ||
            (group.packing == Packing::wave_batches && result.launches_.back().useful_blocks >= result.target_tiles_) ||
            !result.launches_.back().jobs[0].w.has_format(job.w.format)) {
            result.launches_.emplace_back();
            auto& launch = result.launches_.back();
            launch.form = nvfp4 ? Form::nvfp4_split_k : mxfp8 ? Form::mxfp8_split_k : Form::row_tile;
            launch.tiles_per_block = nvfp4 ? (group.rows <= 16 ? 16U : 8U) : mxfp8 ? 4U : 1U;
            launch.chunk_in = chunk_in;
        }

        auto& launch = result.launches_.back();
        const auto member = launch.tiles.count;
        const auto tiles = job.out / kTileOut;
        const auto blocks = output_blocks(tiles, launch.tiles_per_block);
        const auto splits = nvfp4 ? split_k::factor(split_k::kExactTarget, job.out, job.in) :
                            mxfp8 ? split_k::factor(split_k::kExactTarget, job.out, job.in,
                                                   {kTileOut, launch.tiles_per_block, chunk_in}) : 1U;
        if (tiles > std::numeric_limits<std::uint32_t>::max() - launch.tiles.total())
            throw std::invalid_argument("rows: launch exceeds grid limits");
        launch.tiles.add(tiles);
        launch.splits[member] = splits;
        launch.useful_blocks += static_cast<std::uint64_t>(blocks) * splits;
        launch.jobs[member] = job;
    }
    return result;
}

}  // namespace strix::core::rows
