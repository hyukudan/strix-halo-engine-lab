#include "strix/weight/resident_slices.hpp"
#include "strix/weight/row_pitch.hpp"

#include <array>
#include <bit>
#include <iostream>

namespace {

namespace wei = strix::weight;

static_assert(wei::valid_row_pitch_rule({}));
static_assert(wei::valid_row_pitch_rule(wei::kGfx1151RowPitch));
static_assert(!wei::valid_row_pitch_rule({0, 16}));
static_assert(!wei::valid_row_pitch_rule({3072, 256}));
static_assert(!wei::valid_row_pitch_rule({2048, 17}));
static_assert(!wei::valid_row_pitch_rule({2048, 2048}));
static_assert(wei::pitched_row_bytes(2048, wei::kGfx1151RowPitch) == 2304);
static_assert(wei::pitched_row_bytes(4096, wei::kGfx1151RowPitch) == 4352);
static_assert(wei::pitched_row_bytes(6144, wei::kGfx1151RowPitch) == 6400);
static_assert(wei::pitched_row_bytes(4608, wei::kGfx1151RowPitch) == 4608);
static_assert(wei::pitched_row_bytes(2304, wei::kGfx1151RowPitch) == 2304);
static_assert(wei::pitched_row_bytes(2048, {}) == 2048);
static_assert(wei::row_pitch_rule_for_arch("gfx1151").collision_stride == wei::kGfx1151RowPitch.collision_stride &&
              wei::row_pitch_rule_for_arch("gfx1151").pad == wei::kGfx1151RowPitch.pad);
static_assert(wei::row_pitch_rule_for_arch("gfx1100").collision_stride == 0U && wei::row_pitch_rule_for_arch("").pad == 0U);

template <class Call>
bool rejects(const Call& call) {
    try {
        call();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

float dot(const std::uint16_t* weights, std::span<const float> input) {
    float sum = 0.0F;
    for (std::size_t col = 0; col < input.size(); ++col)
        sum += std::bit_cast<float>(std::uint32_t{weights[col]} << 16U) * input[col];
    return sum;
}

bool check_resident_rows() {
    constexpr std::size_t rows = 3, columns = 8, natural = columns * sizeof(std::uint16_t);
    constexpr std::size_t pitch = 32, alignment = 16, matrix_offset = alignment;
    constexpr std::uint16_t sentinel = 0x7fc0;
    constexpr std::array<float, columns> input{0.5F, 3.0F, -0.25F, 1.0F, 2.0F, 0.75F, -2.0F, 4.0F};
    std::array<std::uint16_t, 2 + rows * columns + 1> values{0x3f80, 0x4000};
    for (std::size_t i = 0; i < rows * columns; ++i)
        values[2 + i] = static_cast<std::uint16_t>(0x3f80U + 11U * i);
    values.back() = 0x4040;
    const auto source = wei::memory_source(std::as_bytes(std::span(values)));
    wei::RawTensor before, matrix, after;
    before.wire_dtype = matrix.wire_dtype = after.wire_dtype = "BF16";
    before.payload = {source, {0, 4}};
    matrix.shape = {{rows, columns}, 2};
    matrix.payload = {source, {4, rows * natural}};
    after.payload = {source, {4 + rows * natural, 2}};

    const auto make_plan = [&](std::size_t row_pitch) {
        return wei::ResidentSlicePlan::make(
            {{"after", &after}, {"matrix", &matrix, {}, row_pitch}, {"before", &before}}, alignment);
    };
    const auto contiguous = make_plan(0);
    const auto explicit_contiguous = make_plan(natural);
    const auto pitched = make_plan(pitch);
    const auto matrix_only = wei::ResidentSlicePlan::make({{"matrix", &matrix, {}, pitch}}, alignment);
    if (matrix_only.ranges().size() != rows || matrix_only.bytes() != rows * pitch ||
        contiguous.ranges().size() != 3 || explicit_contiguous.ranges().size() != 3 ||
        pitched.ranges().size() != rows + 2 || pitched.bytes() != matrix_offset + rows * pitch + 2 ||
        contiguous.bytes() != matrix_offset + rows * natural + 2 || explicit_contiguous.bytes() != contiguous.bytes())
        return false;
    constexpr std::array contiguous_offsets{std::size_t{0}, matrix_offset, matrix_offset + rows * natural};
    for (std::size_t i = 0; i < contiguous_offsets.size(); ++i)
        if (contiguous.ranges()[i].destination_offset != contiguous_offsets[i] ||
            explicit_contiguous.ranges()[i].destination_offset != contiguous_offsets[i]) return false;
    for (std::size_t row = 0; row < rows; ++row) {
        const auto& range = pitched.ranges()[1 + row];
        if (range.destination_offset != matrix_offset + row * pitch || range.bytes != natural ||
            range.source.range.offset != matrix.payload.range.offset + row * natural || range.source.range.bytes != natural)
            return false;
    }

    alignas(alignment) std::array<std::uint16_t, (matrix_offset + rows * pitch + 2) / 2> arena;
    alignas(alignment) std::array<std::uint16_t, (matrix_offset + rows * natural + 2) / 2> baseline;
    arena.fill(sentinel);
    const auto load = [](const auto& plan, std::span<std::uint16_t> storage) {
        for (const auto& range : plan.ranges())
            if (!wei::read_slice(range.source, std::as_writable_bytes(storage).subspan(range.destination_offset, range.bytes)))
                return false;
        return true;
    };
    if (!load(pitched, arena) || !load(contiguous, baseline)) return false;
    const auto bound = wei::ResidentSlices::bind(pitched, {reinterpret_cast<const std::uint8_t*>(arena.data()), sizeof(arena)});
    const auto plain = wei::ResidentSlices::bind(contiguous,
        {reinterpret_cast<const std::uint8_t*>(baseline.data()), sizeof(baseline)});
    const auto view = bound.bf16_rows("matrix");
    const auto reference = plain.bf16_rows("matrix");
    if (view.data != arena.data() + matrix_offset / 2 || view.rows != rows || view.columns != columns || view.row_bytes != pitch ||
        reference.rows != rows || reference.columns != columns || reference.row_bytes != natural ||
        reference.data != plain.bf16("matrix").data() || plain.bf16("matrix").size() != rows * columns ||
        bound.bf16("before").data() != arena.data() || bound.bf16("after").front() != values.back() ||
        !rejects([&] { (void)bound.bf16("matrix"); }) || !rejects([&] { (void)bound.bf16_rows("before"); })) return false;
    for (std::size_t row = 0; row < rows; ++row) {
        const auto* weights = view.data + row * (view.row_bytes / 2);
        const auto* expected = reference.data + row * (reference.row_bytes / 2);
        if (!std::equal(weights, weights + columns, expected) ||
            !std::all_of(weights + columns, weights + pitch / 2, [](auto bits) { return bits == sentinel; }) ||
            std::bit_cast<std::uint32_t>(dot(weights, input)) != std::bit_cast<std::uint32_t>(dot(expected, input)))
            return false;
    }

    auto non_bf16 = matrix;
    non_bf16.wire_dtype = "F32";
    auto non_matrix = matrix;
    non_matrix.shape.rank = 3;
    const auto rejects_pitch = [&](const wei::RawTensor& tensor, std::uint64_t row_pitch,
                                   std::optional<wei::FormatId> storage = {}) {
        return rejects([&] {
            (void)wei::ResidentSlicePlan::make({{"matrix", &tensor, storage, row_pitch}}, alignment);
        });
    };
    return rejects_pitch(non_bf16, pitch) && rejects_pitch(non_matrix, pitch) &&
           rejects_pitch(matrix, 8) && rejects_pitch(matrix, 24) &&
           rejects_pitch(matrix, pitch, wei::kBf16WeightStorage);
}

}  // namespace

int main() {
    if (check_resident_rows()) return 0;
    std::cerr << "resident BF16 row pitch check failed\n";
    return 1;
}
