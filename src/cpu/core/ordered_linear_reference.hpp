#pragma once

#include "core/streamed_linear_reference.hpp"
#include "cpu/core/dense.hpp"

namespace strix::cpu {

// Keep workers on independent output rows. A worker may cross a group
// boundary, at which point it advances to the group's input stripe.
inline void parallel_ordered_matvec(const OrderedFma& kernel, core::HostLane& lane, unsigned threads,
                                    std::span<const float> matrix, std::span<const float> input,
                                    std::span<float> output, unsigned groups = 1) {
    if (groups == 0 || input.empty() || output.empty() || input.size() % groups || output.size() % groups)
        throw std::invalid_argument("ordered matvec: invalid projection groups");
    const auto width = input.size() / groups;
    const auto group_rows = output.size() / groups;
    if (matrix.size() % width || matrix.size() / width != output.size())
        throw std::invalid_argument("ordered matvec: invalid matrix geometry");
    core::reference_row_ranges(lane, threads, output.size(), [&](std::size_t first, std::size_t last) {
        while (first < last) {
            const auto group = first / group_rows;
            const auto end = std::min(last, (group + 1) * group_rows);
            kernel.matvec(matrix.data() + first * width, input.data() + group * width, width,
                          output.data() + first, end - first);
            first = end;
        }
    });
}

// A reusable input image for a streamed vocabulary matrix. Pack once, apply
// to successive blocks of output rows. Each lane preserves the scalar order.
class OrderedLinearBatch {
public:
    OrderedLinearBatch(const std::vector<std::vector<float>>& rows, const OrderedFma& kernel)
        : kernel_(kernel), count_(rows.size()), width_(rows.empty() ? 0 : rows.front().size()) {
        if (width_ == 0 || std::ranges::any_of(rows, [&](const auto& row) { return row.size() != width_; }))
            throw std::invalid_argument("ordered batch: inconsistent input rows");
        packed_.resize(((count_ + kLanes - 1) / kLanes) * width_ * kLanes);
        for (std::size_t row = 0; row < count_; ++row)
            for (std::size_t k = 0; k < width_; ++k)
                packed_[(row / kLanes * width_ + k) * kLanes + row % kLanes] = rows[row][k];
    }

    void column(std::span<const float> weights, std::vector<std::vector<float>>& logits, std::size_t col) const {
        for (std::size_t row = 0; row < count_; row += kLanes) {
            float values[kLanes];
            kernel_.dot8(weights.data(), packed_.data() + row * width_, width_, values);
            for (std::size_t i = 0; i < std::min(kLanes, count_ - row); ++i)
                logits[row + i][col] = values[i];
        }
    }

private:
    static constexpr std::size_t kLanes = 8;
    OrderedFma kernel_;
    std::size_t count_, width_;
    std::vector<float> packed_;
};

} // namespace strix::cpu
