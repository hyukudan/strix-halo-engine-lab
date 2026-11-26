#pragma once

#include "moe_core/expert_major_plan.hpp"
#include "moe_core/grouped_policy.hpp"
#include "strix/weight/lowbit_pack.hpp"
#include "strix/weight/host_q8.hpp"
#include "strix/weight/pack_source.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace strix::moe_core::grouped_strixpack {

namespace wei = strix::weight;
namespace sp = wei::sp;

struct MatrixView {
    wei::FormatId storage{};
    std::size_t rows{}, columns{};
    sp::Header header{};
    std::span<const std::uint8_t> bytes;
};

class Matrix {
public:
    wei::Status load(const wei::LowbitPack& pack, const wei::LowbitPackEntry& entry,
                     const wei::PackMatrixSource& source) {
        if (!wei::lowbit_storage_has_header(entry.storage) || !entry.sp_header)
            return {wei::Error::unknown_format, "grouped reference needs a Strix pack matrix"};
        std::optional<wei::Slice> slice;
        if (const auto status = pack.find(entry.tensor, entry.storage, source, slice); !status) return status;
        if (!slice) return {wei::Error::malformed, "missing grouped matrix " + entry.tensor};
        if (slice->range.bytes > std::numeric_limits<std::size_t>::max())
            return {wei::Error::unsupported_shape, "grouped matrix exceeds host address space"};
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(slice->range.bytes));
        if (const auto status = wei::read_slice(*slice, std::as_writable_bytes(std::span(bytes))); !status) return status;
        bytes_ = std::move(bytes);
        view_ = {entry.storage, source.rows, source.columns, *entry.sp_header, {}};
        return {wei::Error::ok, {}};
    }

    [[nodiscard]] MatrixView view() const { return {view_.storage, view_.rows, view_.columns, view_.header, bytes_}; }

private:
    MatrixView view_;
    std::vector<std::uint8_t> bytes_;
};

enum class InputRows { token, assignment };

namespace detail {

template <class Storage>
struct Operations {
    static float dot(const MatrixView& matrix, std::size_t row, std::span<const float> x) {
        const auto* block = matrix.bytes.data() + row * Storage::row_bytes(matrix.columns);
        float sum = 0.0F;
        for (std::size_t col = 0; col < matrix.columns; col += Storage::elements, block += Storage::bytes) {
            for (unsigned group = 0; group < Storage::groups; ++group) {
                const float scale = sp::scale_value(Storage::scale_byte(block, group), matrix.header.base);
                const unsigned start = group * Storage::group;
                for (unsigned lane = start; lane < start + Storage::group; ++lane)
                    sum = std::fma(x[col + lane], sp::decode_scaled<Storage>(block, lane, matrix.header, scale), sum);
            }
        }
        return sum;
    }
};

using SP2 = GroupedWeightDecode<GroupedWeightFormat::sp2, Operations<sp::SP2Storage>>;
using SP3 = GroupedWeightDecode<GroupedWeightFormat::sp3, Operations<sp::SP3Storage>>;
using SP4 = GroupedWeightDecode<GroupedWeightFormat::sp4, Operations<sp::SP4Storage>>;

inline auto dot_for(wei::FormatId format) {
    if (format == wei::kSP2WeightStorage) return &SP2::dot;
    if (format == wei::kSP3WeightStorage) return &SP3::dot;
    return &SP4::dot;
}

}  // namespace detail

// Single activation row of an already admitted matrix. The output range may
// be a prefix (for a vocabulary with unused checkpoint rows).
inline void matvec(const Matrix& matrix, std::span<const float> input, std::span<float> output, std::size_t first = 0U) {
    const auto view = matrix.view();
    if (input.size() != view.columns || first > view.rows || output.size() > view.rows - first)
        throw std::invalid_argument("invalid Strix pack projection geometry");
    const auto dot = detail::dot_for(view.storage);
    for (std::size_t row = 0; row < output.size(); ++row) output[row] = dot(view, first + row, input);
}

inline void lookup(const Matrix& matrix, std::size_t row, std::span<float> output) {
    const auto view = matrix.view();
    if (row >= view.rows || output.size() != view.columns)
        throw std::invalid_argument("invalid Strix pack lookup geometry");
    const auto decode = [&]<class Storage>() {
        const auto* block = view.bytes.data() + row * Storage::row_bytes(view.columns);
        for (std::size_t col = 0; col < view.columns; col += Storage::elements, block += Storage::bytes)
            for (unsigned lane = 0; lane < Storage::elements; ++lane)
                output[col + lane] = sp::decode<Storage>(block, lane, view.header);
    };
    if (view.storage == wei::kSP2WeightStorage) decode.template operator()<sp::SP2Storage>();
    else if (view.storage == wei::kSP3WeightStorage) decode.template operator()<sp::SP3Storage>();
    else decode.template operator()<sp::SP4Storage>();
}

// CPU arithmetic oracle for any projection in the expert-major schedule.
// Inputs can be token rows (gate/up) or already grouped rows (down).
// The output remains expert-major for the caller's activation/combine step.
inline wei::Status project(const ExpertMajorPlan& plan, std::span<const MatrixView> experts,
                           InputRows layout, std::size_t inputs, std::size_t outputs,
                           std::span<const float> x, std::span<float> y) {
    const auto& assignments = plan.grouping.assignment_rows;
    const auto& offsets = plan.grouping.expert_offsets;
    if (inputs == 0 || outputs == 0 || x.size() % inputs != 0 || y.size() % outputs != 0 ||
        y.size() / outputs != assignments.size() || offsets.size() != experts.size() + 1U ||
        (layout == InputRows::assignment && x.size() / inputs != assignments.size()) ||
        (layout == InputRows::token &&
         std::ranges::any_of(assignments, [&](auto row) { return row >= x.size() / inputs; })))
        return {wei::Error::unsupported_shape, "invalid grouped activation geometry"};

    std::size_t first = 0;
    for (const auto& tile : plan.tiles) {
        if (tile.expert >= experts.size() || tile.rows == 0 || tile.first_assignment != first ||
            tile.rows > assignments.size() - first || first < offsets[tile.expert] ||
            first + tile.rows > offsets[std::size_t{tile.expert} + 1U])
            return {wei::Error::malformed, "invalid expert-major tile"};
        const auto& matrix = experts[tile.expert];
        wei::Shape shape{};
        shape.rank = 2;
        shape.dims = {outputs, inputs, 0, 0};
        wei::MatrixLoadGeometry geometry{};
        const auto storage = matrix.storage == wei::kSP4TWeightStorage ? wei::kSP4WeightStorage : matrix.storage;
        if (!wei::strixpack_matrix_geometry(storage, shape, geometry) || matrix.columns != inputs ||
            matrix.rows != outputs || matrix.bytes.size() != geometry.bytes ||
            !wei::lowbit_header_defect(matrix.storage, matrix.header).empty())
            return {wei::Error::malformed, "invalid grouped Strix pack operand"};
        first += tile.rows;
    }
    if (first != assignments.size()) return {wei::Error::malformed, "incomplete expert-major tiles"};

    for (const auto& tile : plan.tiles) {
        const auto& matrix = experts[tile.expert];
        const auto dot = detail::dot_for(matrix.storage);
        for (std::size_t row = 0; row < tile.rows; ++row) {
            const std::size_t assignment = std::size_t{tile.first_assignment} + row;
            const auto input_row = layout == InputRows::token ? assignments[assignment] : assignment;
            const auto input = x.subspan(input_row * inputs, inputs);
            for (std::size_t out = 0; out < outputs; ++out)
                y[assignment * outputs + out] = dot(matrix, out, input);
        }
    }
    return {wei::Error::ok, {}};
}

}  // namespace strix::moe_core::grouped_strixpack
