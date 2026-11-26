#pragma once

#include "moe_core/expert_major_plan.hpp"
#include "moe_core/grouped_policy.hpp"
#include "strix/weight/lowbit_pack.hpp"
#include "strix/weight/host_q8.hpp"
#include "strix/weight/pack_source.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
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
        return load(pack, entry.tensor);
    }

    // The pack has already been admitted by the caller's checkpoint or pack identity.
    wei::Status load(const wei::LowbitPack& pack, std::string_view tensor) {
        const auto* entry = pack.entry(tensor);
        if (!entry || !entry->sp_header)
            return {wei::Error::unknown_format, "grouped reference needs a Strix pack matrix: " + std::string(tensor)};
        wei::MappedBytes mapped;
        if (const auto status = pack.map_tensor(tensor, mapped); !status) return status;
        mapped_ = std::move(mapped);
        view_ = {entry->storage, static_cast<std::size_t>(entry->rows), static_cast<std::size_t>(entry->columns),
                 *entry->sp_header, {reinterpret_cast<const std::uint8_t*>(mapped_.bytes.data()), mapped_.bytes.size()}};
        return {wei::Error::ok, {}};
    }

    [[nodiscard]] MatrixView view() const { return view_; }

private:
    MatrixView view_{};
    wei::MappedBytes mapped_{};
};

// A bank binds a single projection and role. Its cache holds only referenced expert views.
// Keep the admitted pack alive through the bank. The route plan keeps its original arithmetic order.
class ExpertBank {
public:
    ExpertBank(const wei::LowbitPack& pack, std::string role, std::vector<std::string> tensors)
        : pack_(pack), role_(std::move(role)), tensors_(std::move(tensors)), matrices_(tensors_.size()) {}

    [[nodiscard]] std::size_t size() const { return matrices_.size(); }

    wei::Status load(std::size_t expert, MatrixView& out) {
        out = {};
        if (expert >= size()) return {wei::Error::unsupported_shape, "expert outside Strix pack bank"};
        const std::lock_guard lock(mutex_);
        auto& matrix = matrices_[expert];
        if (matrix.view().bytes.empty()) {
            const auto* entry = pack_.entry(tensors_[expert]);
            if (!entry || (pack_.version() == wei::PackVersion::checked && entry->coding.role != role_))
                return {wei::Error::unsupported_role, "missing packed expert in role " + role_ + ": " + tensors_[expert]};
            if (const auto status = matrix.load(pack_, tensors_[expert]); !status) return status;
        }
        out = matrix.view();
        return {wei::Error::ok, {}};
    }

    // Counts may be measured before admission. Load only positive-count hot experts, highest count first.
    wei::Status warm(std::span<const std::uint64_t> route_counts, std::size_t limit) {
        if (route_counts.size() != size())
            return {wei::Error::unsupported_shape, "route counts disagree with Strix pack bank size"};
        std::vector<std::size_t> order(size());
        std::iota(order.begin(), order.end(), 0U);
        std::ranges::sort(order, [&](auto a, auto b) {
            return route_counts[a] == route_counts[b] ? a < b : route_counts[a] > route_counts[b];
        });
        for (const auto expert : std::span(order).first(std::min(limit, order.size()))) {
            if (!route_counts[expert]) break;
            MatrixView view;
            if (const auto status = load(expert, view); !status) return status;
        }
        return {wei::Error::ok, {}};
    }

private:
    const wei::LowbitPack& pack_;
    std::string role_;
    std::vector<std::string> tensors_;
    std::vector<Matrix> matrices_;
    std::mutex mutex_;
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
                const auto scale = sp::scale_value(Storage::scale_byte(block, group), matrix.header.base);
                for (unsigned lane = group * Storage::group; lane < (group + 1U) * Storage::group; ++lane) {
                    const auto code = Storage::code(block, lane);
                    const float m = static_cast<float>(matrix.header.table[code & (Storage::levels - 1U)]) * scale;
                    sum = std::fma(x[col + lane], (code >> (Storage::bits - 1U)) != 0U ? -m : m, sum);
                }
            }
        }
        return sum;
    }
};

using SP2 = GroupedWeightDecode<GroupedWeightFormat::sp2, Operations<sp::SP2Storage>>;
using SP3 = GroupedWeightDecode<GroupedWeightFormat::sp3, Operations<sp::SP3Storage>>;
using SP4 = GroupedWeightDecode<GroupedWeightFormat::sp4, Operations<sp::SP4Storage>>;

inline bool valid_matrix(const MatrixView& matrix) {
    wei::Shape shape{};
    shape.rank = 2;
    shape.dims = {matrix.rows, matrix.columns, 0, 0};
    wei::MatrixLoadGeometry geometry{};
    const auto storage = matrix.storage == wei::kSP4TWeightStorage ? wei::kSP4WeightStorage : matrix.storage;
    return wei::strixpack_matrix_geometry(storage, shape, geometry) &&
           matrix.bytes.size() == geometry.bytes &&
           wei::lowbit_header_defect(matrix.storage, matrix.header).empty();
}

inline auto dot_for(wei::FormatId format) {
    if (format == wei::kSP2WeightStorage) return &SP2::dot;
    if (format == wei::kSP3WeightStorage) return &SP3::dot;
    return &SP4::dot;
}

}  // namespace detail

// An admitted packed view can also feed a single row directly. The output may
// select a row prefix, for example when the head has padded vocabulary rows.
inline wei::Status project_row(const MatrixView& matrix, std::span<const float> x, std::span<float> y) {
    if (x.size() != matrix.columns || y.size() > matrix.rows || !detail::valid_matrix(matrix))
        return {wei::Error::unsupported_shape, "packed projection dimensions disagree with the operand"};
    const auto dot = detail::dot_for(matrix.storage);
    for (std::size_t row = 0; row < y.size(); ++row) y[row] = dot(matrix, row, x);
    return {wei::Error::ok, {}};
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
        if (matrix.columns != inputs || matrix.rows != outputs || !detail::valid_matrix(matrix))
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

// Fetch the routed expert union, then use the same grouped arithmetic as prebound views.
inline wei::Status project(const ExpertMajorPlan& plan, ExpertBank& bank, InputRows layout,
                           std::size_t inputs, std::size_t outputs, std::span<const float> x, std::span<float> y) {
    std::vector<MatrixView> views(bank.size());
    for (const auto& tile : plan.tiles) {
        if (tile.expert >= views.size()) return {wei::Error::malformed, "expert outside Strix pack bank"};
        auto& view = views[tile.expert];
        if (view.bytes.empty()) {
            if (const auto status = bank.load(tile.expert, view); !status) return status;
        }
    }
    return project(plan, views, layout, inputs, outputs, x, y);
}

}  // namespace strix::moe_core::grouped_strixpack
