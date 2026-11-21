#pragma once

#include "strix/weight/formats/microscaled.hpp"
#include "strix/weight/source.hpp"

#include <bit>
#include <limits>
#include <utility>

namespace strix::weight::microscaled {

struct MatrixView {
    MatrixFormat format{};
    std::size_t rows{};
    std::size_t columns{};
    Slice codes;
    Slice block_scales;
    Slice amax;
    Slice denominator;
};

// Owns the packed bytes for just one selected matrix. Its physical row order
// matches the source, so the same staging is usable by host and device callers.
class StagedMatrix {
public:
    Status load(const MatrixView& view) {
        const bool nvfp4 = view.format == MatrixFormat::nvfp4;
        if (!nvfp4 && view.format != MatrixFormat::mxfp8)
            return {Error::unknown_format, "unknown microscaled matrix format"};
        const auto block_columns = nvfp4 ? kNvfp4BlockColumns : kMxfp8BlockColumns;
        if (view.rows == 0 || view.columns == 0 || view.columns % block_columns != 0 ||
            view.rows > std::numeric_limits<std::size_t>::max() / view.columns)
            return {Error::unsupported_shape, "invalid microscaled matrix geometry"};

        StagedMatrix loaded;
        loaded.format_ = view.format;
        loaded.rows_ = view.rows;
        loaded.columns_ = view.columns;
        const auto elements = view.rows * view.columns;
        const auto codes_bytes = nvfp4 ? elements / 2U : elements;
        const auto scales_bytes = elements / block_columns;
        if (view.codes.range.bytes != codes_bytes || view.block_scales.range.bytes != scales_bytes)
            return {Error::malformed, "microscaled matrix payload length disagrees with its geometry"};

        if (nvfp4) {
            if (const auto status = read_f32(view.amax, loaded.amax_); !status) return status;
            if (const auto status = read_f32(view.denominator, loaded.denominator_); !status) return status;
            if (!std::isfinite(loaded.amax_) || loaded.amax_ < 0.0F ||
                !std::isfinite(loaded.denominator_) || loaded.denominator_ <= 0.0F ||
                !std::isfinite(loaded.amax_ / loaded.denominator_))
                return {Error::malformed, "invalid NVFP4 outer scale operands"};
        } else if (view.amax.source || view.denominator.source) {
            return {Error::malformed, "unexpected outer scale on MXFP8 matrix"};
        }
        loaded.codes_.resize(codes_bytes);
        loaded.scales_.resize(scales_bytes);
        if (const auto status = read_slice(view.codes, std::as_writable_bytes(std::span(loaded.codes_))); !status)
            return status;
        if (const auto status = read_slice(view.block_scales, std::as_writable_bytes(std::span(loaded.scales_))); !status)
            return status;
        *this = std::move(loaded);
        return {Error::ok, {}};
    }

    [[nodiscard]] std::size_t rows() const { return codes_.empty() ? 0 : rows_; }
    [[nodiscard]] std::size_t columns() const { return columns_; }
    [[nodiscard]] MatrixFormat format() const { return format_; }
    [[nodiscard]] std::span<const std::uint8_t> codes() const { return codes_; }
    [[nodiscard]] std::span<const std::uint8_t> block_scales() const { return scales_; }
    [[nodiscard]] PackedMatrix packed(RowOrder order) const {
        const float outer = rows() != 0 && format_ == MatrixFormat::nvfp4 ? amax_ / denominator_ : 1.0F;
        return {codes_.data(), scales_.data(), rows(), columns(), outer, order};
    }

    Status decode_row(std::size_t row, std::span<float> out) const {
        if (row >= rows() || out.size() != columns_)
            return {Error::unsupported_shape, "microscaled matrix row outside its geometry"};
        const bool nvfp4 = format_ == MatrixFormat::nvfp4;
        const auto code_bytes = nvfp4 ? columns_ / 2U : columns_;
        const auto scale_bytes = columns_ / (nvfp4 ? kNvfp4BlockColumns : kMxfp8BlockColumns);
        const auto codes = std::span(codes_).subspan(row * code_bytes, code_bytes);
        const auto scales = std::span(scales_).subspan(row * scale_bytes, scale_bytes);
        try {
            if (nvfp4) decode_nvfp4_row(codes, scales, amax_, denominator_, out);
            else decode_mxfp8_row(codes, scales, out);
        } catch (const std::invalid_argument& error) {
            return {Error::malformed, error.what()};
        }
        return {Error::ok, {}};
    }

private:
    static Status read_f32(const Slice& slice, float& out) {
        std::array<Byte, 4> bytes{};
        if (slice.range.bytes != bytes.size()) return {Error::malformed, "expected one F32 outer scale"};
        if (const auto status = read_slice(slice, bytes); !status) return status;
        std::uint32_t bits = 0;
        for (unsigned i = 0; i < bytes.size(); ++i)
            bits |= std::to_integer<std::uint32_t>(bytes[i]) << (8U * i);
        out = std::bit_cast<float>(bits);
        return {Error::ok, {}};
    }

    MatrixFormat format_{};
    std::size_t rows_{};
    std::size_t columns_{};
    float amax_{};
    float denominator_{};
    std::vector<std::uint8_t> codes_;
    std::vector<std::uint8_t> scales_;
};

}  // namespace strix::weight::microscaled
