#pragma once

#include "strix/weight/formats/microscaled.hpp"
#include "strix/weight/source.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace strix::weight { class TensorLookup; }

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

namespace detail {

inline bool valid_row_order(RowOrder order, std::size_t rows) {
    return order == RowOrder::contiguous || (order == RowOrder::interleaved_halves && rows % 2U == 0);
}

inline Status read_f32(const Slice& slice, float& out) {
    std::array<Byte, 4> bytes{};
    if (slice.range.bytes != bytes.size()) return {Error::malformed, "expected one F32 outer scale"};
    if (const auto status = read_slice(slice, bytes); !status) return status;
    out = decode_f32_le(bytes.data());
    return {Error::ok, {}};
}

}  // namespace detail

struct ExpertSlabNames {
    std::string codes;
    std::string block_scales;
    std::string amax;
    std::string denominator;
};

// Raw source layout for one physical projection's [expert][row][column]
// checkpoints. Codes and scales can be uploaded in two spans with no widening.
struct ExpertSlabView {
    MatrixFormat format{};
    RowOrder row_order{};
    std::size_t experts{};
    std::size_t rows{};
    std::size_t columns{};
    Slice codes;
    Slice block_scales;
    Slice amax;
    Slice denominator;

    // Attach named source slices to the declared geometry without reading
    // payloads. Only NVFP4 uses the two outer-scale sidecar names.
    Status bind_sources(const TensorLookup&, const ExpertSlabNames&);

    Status matrix(std::size_t expert, MatrixView& out) const {
        if (expert >= experts) return {Error::unsupported_shape, "expert outside microscaled slab"};
        std::size_t code_stride = 0, scale_stride = 0;
        if (const auto status = strides(code_stride, scale_stride); !status) return status;
        MatrixView result{format, rows, columns, {}, {}, {}, {}};
        if (const auto status = subslice(codes, {expert * code_stride, code_stride}, result.codes); !status)
            return status;
        if (const auto status = subslice(block_scales, {expert * scale_stride, scale_stride}, result.block_scales);
            !status) return status;
        if (format == MatrixFormat::nvfp4) {
            if (const auto status = subslice(amax, {expert * 4U, 4U}, result.amax); !status) return status;
            result.denominator = denominator;
        }
        out = std::move(result);
        return {Error::ok, {}};
    }

    // The code/scale spans may name device allocations and are not read here.
    // Load only the small F32 NVFP4 sidecars and attach the actual device pointers.
    Status bind_storage(std::span<const std::uint8_t> resident_codes,
                        std::span<const std::uint8_t> resident_scales,
                        std::span<PackedMatrix> out) const {
        std::size_t code_stride = 0, scale_stride = 0;
        if (const auto status = strides(code_stride, scale_stride); !status) return status;
        if (resident_codes.size() != codes.range.bytes || resident_scales.size() != block_scales.range.bytes ||
            out.size() != experts)
            return {Error::unsupported_shape, "resident microscaled slab dimensions disagree with source"};
        std::vector<float> outer(experts, 1.0F);
        if (format == MatrixFormat::nvfp4) {
            float divisor = 0.0F;
            if (const auto status = detail::read_f32(denominator, divisor); !status) return status;
            if (!std::isfinite(divisor) || divisor <= 0.0F)
                return {Error::malformed, "invalid NVFP4 outer scale denominator"};
            std::vector<Byte> maxima(experts * 4U);
            if (const auto status = read_slice(amax, maxima); !status) return status;
            for (std::size_t expert = 0; expert < experts; ++expert) {
                const float maximum = decode_f32_le(maxima.data() + 4U * expert);
                outer[expert] = maximum / divisor;
                if (!std::isfinite(outer[expert]) || maximum < 0.0F)
                    return {Error::malformed, "invalid per-expert NVFP4 outer scale"};
            }
        }
        for (std::size_t expert = 0; expert < experts; ++expert) {
            out[expert] = {resident_codes.data() + expert * code_stride, resident_scales.data() + expert * scale_stride,
                           rows, columns, outer[expert], row_order};
        }
        return {Error::ok, {}};
    }

private:
    Status strides(std::size_t& code_stride, std::size_t& scale_stride) const {
        const bool nvfp4 = format == MatrixFormat::nvfp4;
        if (!nvfp4 && format != MatrixFormat::mxfp8)
            return {Error::unknown_format, "unknown microscaled slab format"};
        if (!detail::valid_row_order(row_order, rows))
            return {Error::unsupported_shape, "invalid microscaled slab row order"};
        const auto block = nvfp4 ? kNvfp4BlockColumns : kMxfp8BlockColumns;
        constexpr auto limit = std::numeric_limits<std::size_t>::max();
        if (experts == 0 || rows == 0 || columns == 0 || columns % block != 0 ||
            rows > limit / columns || experts > limit / (rows * columns))
            return {Error::unsupported_shape, "invalid microscaled slab geometry"};
        const auto elements = rows * columns;
        code_stride = nvfp4 ? elements / 2U : elements;
        scale_stride = elements / block;
        if (codes.range.bytes != experts * code_stride || block_scales.range.bytes != experts * scale_stride)
            return {Error::malformed, "microscaled slab slice length disagrees with geometry"};
        if (nvfp4 && (amax.range.bytes != experts * 4U || denominator.range.bytes != 4U))
            return {Error::malformed, "invalid NVFP4 outer scale sidecar lengths"};
        if (!nvfp4 && (amax.source || denominator.source || amax.range.bytes != 0 || denominator.range.bytes != 0))
            return {Error::malformed, "unexpected outer scale on MXFP8 slab"};
        return {Error::ok, {}};
    }
};

// Owns the packed bytes for just one selected matrix. Its physical row order
// matches the source, so the same staging is usable by host and device callers.
class StagedMatrix {
public:
    Status load(const MatrixView& view) { return load(view, RowOrder::contiguous); }

    Status load(const MatrixView& view, RowOrder order) {
        if (!detail::valid_row_order(order, view.rows))
            return {Error::unsupported_shape, "invalid microscaled matrix row order"};
        const bool nvfp4 = view.format == MatrixFormat::nvfp4;
        if (!nvfp4 && view.format != MatrixFormat::mxfp8)
            return {Error::unknown_format, "unknown microscaled matrix format"};
        const auto block_columns = nvfp4 ? kNvfp4BlockColumns : kMxfp8BlockColumns;
        if (view.rows == 0 || view.columns == 0 || view.columns % block_columns != 0 ||
            view.rows > std::numeric_limits<std::size_t>::max() / view.columns)
            return {Error::unsupported_shape, "invalid microscaled matrix geometry"};

        StagedMatrix loaded;
        loaded.format_ = view.format;
        loaded.order_ = order;
        loaded.rows_ = view.rows;
        loaded.columns_ = view.columns;
        const auto elements = view.rows * view.columns;
        const auto codes_bytes = nvfp4 ? elements / 2U : elements;
        const auto scales_bytes = elements / block_columns;
        if (view.codes.range.bytes != codes_bytes || view.block_scales.range.bytes != scales_bytes)
            return {Error::malformed, "microscaled matrix payload length disagrees with its geometry"};

        if (nvfp4) {
            if (const auto status = detail::read_f32(view.amax, loaded.amax_); !status) return status;
            if (const auto status = detail::read_f32(view.denominator, loaded.denominator_); !status) return status;
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
    [[nodiscard]] PackedMatrix packed() const { return packed(order_); }
    [[nodiscard]] PackedMatrix packed(RowOrder order) const {
        const float outer = rows() != 0 && format_ == MatrixFormat::nvfp4 ? amax_ / denominator_ : 1.0F;
        return {codes_.data(), scales_.data(), rows(), columns(), outer, order};
    }

    Status decode_row(std::size_t row, std::span<float> out) const {
        if (row >= rows() || out.size() != columns_)
            return {Error::unsupported_shape, "microscaled matrix row outside its geometry"};
        row = physical_row(order_, rows_, row);
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
    MatrixFormat format_{};
    RowOrder order_{};
    std::size_t rows_{};
    std::size_t columns_{};
    float amax_{};
    float denominator_{};
    std::vector<std::uint8_t> codes_;
    std::vector<std::uint8_t> scales_;
};

}  // namespace strix::weight::microscaled
