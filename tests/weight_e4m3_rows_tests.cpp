// Row-local image of a 128 x 128 block-scaled E4M3 matrix (BlockScaledE4M3::pack_row_blocks) read through the shared
// row-sweep chunk decoder (core/weight_decode.hpp Format::e4m3_b128): every weight is the host decode_row weight bit for
// bit (all 256 codes, NaN codes, subnormals, power-of-two and arbitrary FP32 scales, a partial row tile), and the
// image size is the decoder's row_bytes.
#include "core/weight_decode.hpp"
#include "strix/weight/formats/fp8_block.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

namespace wd = strix::core::weight_decode;
using Matrix = strix::weight::BlockScaledE4M3<128, 128, float>;

int failures = 0;
void expect(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}

bool same_weight(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}

void check_matrix(std::size_t rows, std::size_t columns, bool power_of_two, std::mt19937& random) {
    std::vector<std::uint8_t> codes(rows * columns);
    for (std::size_t i = 0; i < codes.size(); ++i)
        codes[i] = static_cast<std::uint8_t>(i < 256U ? i : random() & 255U);  // every code at least once
    std::vector<float> scales(Matrix::scale_rows(rows) * Matrix::scale_columns(columns));
    std::uniform_int_distribution<int> exponent(-13, -7);
    std::uniform_real_distribution<float> mantissa(0.5F, 2.0F);
    for (auto& s : scales)
        s = power_of_two ? std::ldexp(1.0F, exponent(random)) : mantissa(random) * std::ldexp(1.0F, exponent(random));
    scales.back() = 1.0e-38F;  // subnormal products in the last tile
    const Matrix matrix(codes, scales, rows, columns);

    const auto row_bytes = wd::row_bytes(wd::Format::e4m3_b128, columns);
    expect(row_bytes == Matrix::row_block_bytes(columns), "row_bytes equals the packed row size");
    std::vector<std::uint8_t> image(rows * row_bytes);
    std::vector<float> host(columns);
    std::size_t compared = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        const std::span<std::uint8_t> row(image.data() + r * row_bytes, row_bytes);
        matrix.pack_row_blocks(r, row);
        matrix.decode_row(r, host);
        for (unsigned i = 0; i < columns; i += 16U) {
            const wd::Chunk<wd::Format::e4m3_b128> chunk(row.data(), i);
            for (unsigned e = 0; e < 16U; ++e) {
                if (!same_weight(chunk.at(e), host[i + e])) {
                    std::fprintf(stderr, "row %zu col %u: chunk %a host %a\n", r, i + e, double(chunk.at(e)),
                                 double(host[i + e]));
                    ++failures;
                    return;
                }
                ++compared;
            }
        }
    }
    expect(compared == rows * columns, "every weight compared");
}

} // namespace

int main() {
    std::mt19937 random(20261127U);
    check_matrix(130, 384, true, random);   // two row tiles, the second partial; scales are powers of two
    check_matrix(64, 1024, false, random);  // arbitrary FP32 scales: still the host's one FP32 product
    bool rejected = false;
    try {
        const std::vector<std::uint8_t> codes(2 * 136);
        const std::vector<float> scales(2, 1.0F);
        std::vector<std::uint8_t> out(Matrix::row_block_bytes(136));
        Matrix(codes, scales, 2, 136).pack_row_blocks(0, out);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "a partial column block is refused");
    if (failures) return 1;
    std::puts("weight-e4m3-rows: PASS");
    return 0;
}
