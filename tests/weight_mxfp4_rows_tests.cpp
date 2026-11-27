// Row-local image of split MXFP4 rows (microscaled.hpp pack_mxfp4_row_blocks) read through the shared row-sweep chunk
// decoder (core/weight_decode.hpp Format::mxfp4_b32): every weight is the host decode_mxfp4_row weight bit for bit
// (all 16 codes in both nibble positions, every finite E8M0 scale incl. the subnormal 0 and 1), and the image size is
// the decoder's row_bytes. A 16-chunk is half a 32-block; chunks at both halves are read.
#include "core/weight_decode.hpp"
#include "strix/weight/formats/microscaled.hpp"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

namespace wd = strix::core::weight_decode;
namespace ms = strix::weight::microscaled;

int failures = 0;
void expect(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}

void check_rows(std::size_t rows, std::size_t columns, std::mt19937& random) {
    const std::size_t blocks = columns / 32U;
    std::vector<std::uint8_t> codes(rows * columns / 2U), scales(rows * blocks);
    for (std::size_t i = 0; i < codes.size(); ++i)
        codes[i] = static_cast<std::uint8_t>(i < 256U ? i : random() & 255U);  // every nibble pair at least once
    for (std::size_t i = 0; i < scales.size(); ++i)
        scales[i] = static_cast<std::uint8_t>(i < 254U ? i : 2U + random() % 250U);  // E8M0 0..253 (255 is NaN)
    const auto row_bytes = wd::row_bytes(wd::Format::mxfp4_b32, columns);
    expect(row_bytes == ms::mxfp4_row_block_bytes(columns), "row_bytes equals the packed row size");
    expect(wd::block_elements(wd::Format::mxfp4_b32) == 32U, "K granularity is one block");
    std::vector<std::uint8_t> image(rows * row_bytes);
    std::vector<float> host(columns);
    std::size_t compared = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        const std::span<std::uint8_t> row(image.data() + r * row_bytes, row_bytes);
        const std::span<const std::uint8_t> row_codes(codes.data() + r * columns / 2U, columns / 2U);
        const std::span<const std::uint8_t> row_scales(scales.data() + r * blocks, blocks);
        ms::pack_mxfp4_row_blocks(row_codes, row_scales, row);
        try {
            ms::decode_mxfp4_row(row_codes, row_scales, host);
        } catch (const std::invalid_argument&) {
            continue;  // a scale that overflows FP32 for the largest code: the host refuses it, nothing to compare
        }
        for (unsigned i = 0; i < columns; i += 16U) {
            const wd::Chunk<wd::Format::mxfp4_b32> chunk(row.data(), i);
            for (unsigned e = 0; e < 16U; ++e) {
                if (std::bit_cast<std::uint32_t>(chunk.at(e)) != std::bit_cast<std::uint32_t>(host[i + e])) {
                    std::fprintf(stderr, "row %zu col %u: chunk %a host %a\n", r, i + e, double(chunk.at(e)),
                                 double(host[i + e]));
                    ++failures;
                    return;
                }
                ++compared;
            }
        }
    }
    expect(compared > 0, "weights compared");
}

} // namespace

int main() {
    std::mt19937 random(20261127U);
    check_rows(8, 2048, random);
    check_rows(130, 96, random);
    bool rejected = false;
    try {
        const std::vector<std::uint8_t> codes(16), scales(1);
        std::vector<std::uint8_t> out(16);
        ms::pack_mxfp4_row_blocks(codes, scales, out);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "an inconsistent image size is refused");
    if (failures) return 1;
    std::puts("weight-mxfp4-rows: PASS");
    return 0;
}
