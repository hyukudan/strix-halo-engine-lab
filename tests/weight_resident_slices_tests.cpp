#include "strix/weight/resident_slices.hpp"

#include <array>
#include <iostream>
#include <span>

namespace {

namespace wei = strix::weight;

bool expects(const wei::Status& status, wei::Error expected) {
    if (status.code == expected) return true;
    std::cerr << status.detail << '\n';
    return false;
}

bool check_resident_bindings() {
    using wei::microscaled::ExpertSlabView;
    using wei::microscaled::MatrixFormat;
    using wei::microscaled::PackedMatrix;
    using wei::microscaled::RowOrder;
    constexpr std::size_t experts = 3, rows = 4, columns = 32;
    std::array<std::uint8_t, experts * rows * columns> codes{};
    std::array<std::uint8_t, experts * rows * columns / 16U> scales{};
    std::array<PackedMatrix, experts> matrices{};
    std::array<std::uint8_t, 16> outer{
        0, 0, 0, 0x41,     // amax 8.0 -> 2.0
        0, 0, 0x40, 0x41,  // amax 12.0 -> 3.0
        0, 0, 0x80, 0x3f,  // amax 1.0 -> 0.25
        0, 0, 0x80, 0x40,  // denominator 4.0
    };
    const auto storage = wei::memory_source(std::as_bytes(std::span(outer)));
    const auto fp4_codes = std::span(codes).first(codes.size() / 2U);
    // Metadata-only code and scale slices intentionally have no readable source.
    // The device binder may inspect extents but must never read those payloads.
    ExpertSlabView fp4{MatrixFormat::nvfp4, RowOrder::interleaved_halves, experts, rows, columns,
                       {{}, {0, fp4_codes.size()}}, {{}, {0, scales.size()}},
                       {storage, {0, 12}}, {storage, {12, 4}}};
    if (!expects(fp4.bind_storage(fp4_codes, scales, matrices), wei::Error::ok)) return false;
    constexpr std::array outer_scale{2.0F, 3.0F, 0.25F};
    for (std::size_t expert = 0; expert < experts; ++expert) {
        const auto& m = matrices[expert];
        if (m.codes != codes.data() + expert * rows * columns / 2U ||
            m.block_scales != scales.data() + expert * rows * columns / 16U ||
            m.rows != rows || m.columns != columns || m.outer_scale != outer_scale[expert] ||
            m.row_order != RowOrder::interleaved_halves) return false;
    }
    if (!expects(fp4.bind_storage(fp4_codes.first(fp4_codes.size() - 1U), scales, matrices),
                 wei::Error::unsupported_shape)) return false;
    outer[7] = 0xc1U;  // negative second amax
    fp4.amax.source = wei::memory_source(std::as_bytes(std::span(outer)));
    if (!expects(fp4.bind_storage(fp4_codes, scales, matrices), wei::Error::malformed) ||
        matrices[1].outer_scale != 3.0F) return false;

    const auto fp8_scales = std::span(scales).first(scales.size() / 2U);
    ExpertSlabView fp8{MatrixFormat::mxfp8, RowOrder::contiguous, experts, rows, columns,
                       {{}, {0, codes.size()}}, {{}, {0, fp8_scales.size()}}, {}, {}};
    return expects(fp8.bind_storage(codes, fp8_scales, matrices), wei::Error::ok) &&
           matrices.back().codes == codes.data() + 2U * rows * columns &&
           matrices.back().block_scales == scales.data() + 2U * rows &&
           matrices.back().outer_scale == 1.0F && matrices.back().row_order == RowOrder::contiguous;
}

bool check_aligned_plan() {
    constexpr std::size_t alignment = 16;
    alignas(alignment) std::array<std::uint8_t, 20> arena{};
    const auto source = wei::memory_source(std::as_bytes(std::span(arena)));
    wei::RawTensor bf16;
    bf16.wire_dtype = "BF16";
    bf16.payload = {source, {0, 2}};
    wei::RawTensor f32;
    f32.wire_dtype = "F32";
    f32.payload = {source, {2, 4}};
    const auto plan = wei::ResidentSlicePlan::make({{"bias", &f32}, {"gain", &bf16}}, alignment);
    if (plan.bytes() != arena.size() || plan.ranges().size() != 2 ||
        plan.ranges()[0].source.range.offset != 0 || plan.ranges()[1].destination_offset != alignment)
        return false;
    const auto storage = wei::ResidentSlices::bind(plan, arena);
    if (std::as_bytes(storage.bf16("gain")).data() != std::as_bytes(std::span(arena)).data() ||
        std::as_bytes(storage.f32("bias")).data() != std::as_bytes(std::span(arena)).data() + alignment)
        return false;
    try {
        (void)storage.f32("gain");
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    if (!check_resident_bindings()) return 1;
    return check_aligned_plan() ? 0 : 2;
}
