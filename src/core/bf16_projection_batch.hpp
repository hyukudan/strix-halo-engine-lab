#pragma once

#include "core/dense_row_plan.hpp"
#include "strix/core/rows.hpp"

#include <array>
#include <limits>

namespace strix::core {

// Projection calls share a readiness boundary but can have different inputs.
// Store offsets, not spans into either the caller's arrays or this object.
class Bf16ProjectionBatch {
public:
    struct Group {
        const float* input;
        unsigned width;
        std::size_t first, count;
    };

    bool append(const float* input, unsigned width, std::span<const Bf16RowProjection> outputs) {
        if (outputs.empty() || outputs.size() > projections_.size() - count_) return false;
        groups_[group_count_++] = {input, width, count_, outputs.size()};
        std::ranges::copy(outputs, projections_.begin() + static_cast<std::ptrdiff_t>(count_));
        count_ += outputs.size();
        return true;
    }

    [[nodiscard]] auto groups() const { return std::span(groups_).first(group_count_); }
    [[nodiscard]] auto outputs(const Group& group) const {
        return std::span(projections_).subspan(group.first, group.count);
    }

    [[nodiscard]] bool valid_r1_operands() const {
        struct Range { std::uintptr_t first, end; };
        const auto range = [](const void* p, std::size_t count, std::size_t size) -> Range {
            const auto address = reinterpret_cast<std::uintptr_t>(p);
            if (!address || address % 16U || !count || count > (std::numeric_limits<std::uintptr_t>::max() - address) / size)
                return {};
            return {address, address + count * size};
        };
        const auto overlaps = [](Range a, Range b) { return a.first < b.end && b.first < a.end; };
        std::array<Range, 2U * rows::kMaxJobs> reads{};
        std::array<Range, rows::kMaxJobs> writes{};
        std::size_t nr = 0, nw = 0;
        for (const auto& group : groups()) {
            if (!group.width || group.width % 8U) return false;
            const auto input = range(group.input, group.width, sizeof(float));
            if (!input.first) return false;
            reads[nr++] = input;
            for (const auto& p : outputs(group)) {
                if (group.width > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) return false;
                const auto weight = range(p.weights, p.outputs, std::size_t{group.width} * sizeof(std::uint16_t));
                const auto output = range(p.output, p.outputs, sizeof(float));
                if (!weight.first || !output.first) return false;
                reads[nr++] = weight;
                writes[nw++] = output;
            }
        }
        for (std::size_t i = 0; i < nw; ++i) {
            for (const auto read : std::span(reads).first(nr)) if (overlaps(writes[i], read)) return false;
            for (const auto write : std::span(writes).first(i)) if (overlaps(writes[i], write)) return false;
        }
        return nw != 0;
    }

    // Only consume an independently qualified store-round for a matching R1
    // tile. Scalar fallbacks and rounds checking a failure word keep their tail.
    bool try_round_store(float* output, std::size_t elements, unsigned* failure = nullptr) {
        if (!output || failure || !elements || elements % rows::kTileOut) return false;
        for (const auto& group : groups()) {
            if (group.width % rows::kBlockIn) continue;
            for (auto& p : std::span(projections_).subspan(group.first, group.count)) {
                if (p.output == output && p.outputs == elements && !p.round_bf16) {
                    p.round_bf16 = true;
                    return true;
                }
            }
        }
        return false;
    }

private:
    std::array<Group, rows::kMaxJobs> groups_{};
    std::array<Bf16RowProjection, rows::kMaxJobs> projections_{};
    std::size_t group_count_{}, count_{};
};

}  // namespace strix::core
