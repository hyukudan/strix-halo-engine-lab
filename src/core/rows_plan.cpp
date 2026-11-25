#include "core/rows_plan.hpp"

#include "strix/weight/formats/scalar_float.hpp"
#include "strix/weight/registry.hpp"

#include <algorithm>

namespace strix::core::rows {
namespace {

constexpr std::uint32_t kOccupancyWaves = 4;

float add(float a, float b) { const volatile float value = a + b; return value; }
float mul(float a, float b) { const volatile float value = a * b; return value; }

bool aligned(const void* pointer, std::size_t bytes) {
    return reinterpret_cast<std::uintptr_t>(pointer) % bytes == 0;
}

}  // namespace

RowsPlan plan_rows(std::span<const RoleJob> jobs, std::span<const RoleFormat> formats, std::uint32_t rows,
                   const device::Profile& device, SplitLimits limits) {
    if (limits.max_splits == 0) throw std::invalid_argument("rows: expected a nonzero backend split limit");

    std::vector<Job> bound;
    bound.reserve(jobs.size());
    for (const auto& entry : jobs) {
        const auto match = [&](const RoleFormat& f) { return f.role == entry.role; };
        const auto format = std::find_if(formats.begin(), formats.end(), match);
        if (format == formats.end() || !entry.job.w.has_format(format->format) ||
            std::find_if(format + 1, formats.end(), match) != formats.end())
            throw std::invalid_argument("rows: expected one matching weight format for the role");
        if (!entry.job.w.has_format(weight::kBf16WeightStorage) && !entry.job.w.has_format(weight::kQ8WeightStorage) &&
            !entry.job.w.has_format(kMxfp8Rows) && !entry.job.w.has_format(kNvfp4Records))
            throw std::invalid_argument("rows: unsupported weight format");
        bound.push_back(entry.job);
    }

    const auto base = plan({bound, rows, 0, Packing::fewest_launches}, device);
    RowsPlan result;
    result.rows_ = rows;
    result.target_blocks_ = std::uint64_t{device.units} * kOccupancyWaves;
    result.launches_.reserve(base.launches().size());
    std::size_t index = 0;
    for (const auto& launch : base.launches()) {
        auto& planned = result.launches_.emplace_back(PlannedLaunch{launch, {}, {}});
        planned.launch.useful_blocks = 0;
        for (unsigned member = 0; member < launch.tiles.count; ++member, ++index) {
            const auto& entry = jobs[index];
            const auto blocks = output_blocks(entry.job.out / kTileOut, launch.tiles_per_block);
            const auto chunks = entry.job.in / launch.chunk_in;
            const auto cap = std::min(chunks, limits.max_splits);
            auto splits = launch.splits[member];
            auto identity = launch.split_identity();
            if (launch.form == Form::mxfp8_split_k) {
                if (entry.splits != 0) {
                    splits = entry.splits;
                } else if (entry.policy == SplitPolicy::occupancy) {
                    const auto needed = (result.target_blocks_ + blocks - 1) / blocks;
                    splits = static_cast<std::uint32_t>(std::min<std::uint64_t>(needed, cap));
                    identity = kMxfp8OccupancySplitIdentity;
                }
            } else if (entry.splits != 0 && entry.splits != splits) {
                throw std::invalid_argument("rows: requested split changes the incumbent arithmetic");
            }
            if (splits == 0 || splits > cap) throw std::invalid_argument("rows: split count exceeds usable chunks or grid");

            planned.roles[member] = entry.role;
            if (launch.form != Form::row_tile)
                planned.trees[member] = {identity, launch.chunk_in, chunks, splits};
            planned.launch.splits[member] = splits;
            planned.launch.useful_blocks += std::uint64_t{blocks} * splits;
        }
    }
    return result;
}

float mxfp8_rows_reference(const SplitTree& tree, std::span<const std::uint8_t> codes,
                           std::span<const std::uint8_t> scales, std::span<const float> x) {
    if (!tree.valid() || (tree.arithmetic != kMxfp8SplitIdentity && tree.arithmetic != kMxfp8OccupancySplitIdentity) || tree.chunk_in != kBlockIn ||
        codes.size() != std::uint64_t{tree.chunks} * tree.chunk_in || x.size() != codes.size() ||
        scales.size() != tree.chunks)
        throw std::invalid_argument("rows: invalid MXFP8 reference tree or input");

    float sum = 0.0f;
    for (std::uint32_t split = 0; split < tree.splits; ++split) {
        const auto range = tree.range(split);
        float part = 0.0f;
        for (auto c = range.begin; c < range.end; ++c) {
            float dot = 0.0f;
            const auto begin = std::size_t{c} * tree.chunk_in;
            for (std::size_t k = begin; k < begin + tree.chunk_in; ++k)
                dot = add(dot, mul(weight::scalar_float::e4m3(codes[k]), x[k]));
            part = add(part, mul(dot, weight::scalar_float::e8m0(scales[c])));
        }
        sum = split == 0 ? part : add(sum, part);
    }
    return sum;
}

void validate_a8_consumer(const A8ReadView& a, std::uint32_t rows, std::uint32_t width, weight::FormatId weights,
                          weight::Exactness admission) {
    if (admission != weight::Exactness::fast || weights.name != weight::kQ8WeightStorage.name ||
        weights.revision != weight::kQ8WeightStorage.revision)
        throw std::invalid_argument("rows: A8 requires explicit bit-changing admission and Q8_0 weights");
    if (!a.codes || !a.deltas || rows == 0 || rows > kMaxRows || a.rows != rows || width == 0 || a.width != width ||
        width % kBlockIn != 0 || a.stride < width || a.stride % kBlockIn != 0 ||
        !aligned(a.codes, 8) || !aligned(a.deltas, alignof(float)) || !aligned(a.sums, alignof(std::int32_t)))
        throw std::invalid_argument("rows: invalid planar block32 A8 binding");
}

}  // namespace strix::core::rows
