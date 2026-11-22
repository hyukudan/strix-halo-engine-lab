// A8 operand row pitch (include/strix/a8_row_pitch.hpp): the padded pitch never aliases, and the dense A8 product
// (policy dense-q8a8-f16-v1, CPU oracle src/intel/core/dense_q8a8_ref.hpp) gives the same bits at any row pitch,
// including the head-batched K_b / V_b views (codes + z * batch_in, deltas + z * batch_in / 32).
#include "strix/a8_row_pitch.hpp"
#include "strix/core/a8g32.hpp"
#include "dense_q8a8_ref.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}

// Host image of quantize_a8_kernel (src/hip_ds41_formats.hip): input row r at r * width, output row r at r * pitch
// (codes) and r * (pitch / 32) (deltas, sums); the a8g32-v1 reference (strix/core/a8g32.hpp).
void quantize_rows(const std::vector<float>& x, std::uint32_t width, std::uint32_t rows, std::uint32_t pitch,
                   std::vector<std::int8_t>& codes, std::vector<float>& deltas, std::vector<std::int32_t>& sums) {
    codes.assign(std::size_t{pitch} * rows, std::int8_t{0x55});
    deltas.assign(std::size_t{pitch / 32U} * rows, -7.0F);
    sums.assign(std::size_t{pitch / 32U} * rows, -7);
    strix::core::a8g32::quantize_rows(x.data(), width, {rows, width, pitch}, codes.data(), deltas.data(), sums.data());
}

std::uint32_t next(std::uint32_t& s) {
    s = s * 1664525U + 1013904223U;
    return s >> 8;
}

// Q8_0 weight rows {FP16 scale, 32 x int8} per 32 inputs, `heads` products of `count` rows each.
std::vector<std::uint8_t> weights(std::uint32_t in, std::uint32_t count, std::uint32_t heads, std::uint32_t seed) {
    const std::uint32_t row_bytes = in / 32U * 34U;
    std::vector<std::uint8_t> w(std::size_t{row_bytes} * count * heads);
    for (std::size_t b = 0; b < w.size() / 34U; ++b) {
        const std::uint16_t scale = static_cast<std::uint16_t>(0x2000U + next(seed) % 0x0800U);  // small positive FP16
        std::memcpy(&w[b * 34U], &scale, 2);
        for (std::size_t j = 0; j < 32U; ++j) w[b * 34U + 2U + j] = static_cast<std::uint8_t>(next(seed));
    }
    return w;
}

// Head-batched product over activation rows of `width` values: head z reads inputs [z * batch_in, z * batch_in + in).
std::vector<float> product(const std::vector<float>& x, std::uint32_t width, std::uint32_t rows, std::uint32_t pitch,
                           const std::vector<std::uint8_t>& w, std::uint32_t in, std::uint32_t count, std::uint32_t heads,
                           std::uint32_t batch_in) {
    std::vector<std::int8_t> codes;
    std::vector<float> deltas;
    std::vector<std::int32_t> sums;
    quantize_rows(x, width, rows, pitch, codes, deltas, sums);
    std::vector<float> out(std::size_t{rows} * count * heads, 0.0F);
    const std::uint32_t row_bytes = in / 32U * 34U;
    for (std::uint32_t z = 0; z < heads; ++z) {
        strix::intel::dense::Q8A8View v;
        v.weight = w.data() + std::size_t{z} * count * row_bytes;
        v.codes = codes.data() + std::size_t{z} * batch_in;
        v.deltas = deltas.data() + std::size_t{z} * (batch_in / 32U);
        v.out = out.data() + std::size_t{z} * count;
        v.row_bytes = row_bytes;
        v.in = in;
        v.count = count;
        v.rows = rows;
        v.x_stride = pitch;
        v.out_stride = count * heads;
        expect(strix::intel::dense::supported(v), "view supported at this pitch");
        strix::intel::dense::run_reference(v);
    }
    return out;
}

void same_bits_at_any_pitch(std::uint32_t in, std::uint32_t count, std::uint32_t heads, std::uint32_t pad, const char* what) {
    const std::uint32_t rows = 5U, width = in * heads;
    std::uint32_t seed = 12345U + in;
    std::vector<float> x(std::size_t{width} * rows);
    for (float& v : x) v = static_cast<float>(static_cast<std::int32_t>(next(seed) % 2001U) - 1000) / 37.0F;
    for (std::size_t k = 0; k < 32U; ++k) x[64U + k] = 0.0F;  // one all-zero group (delta 1)
    const auto w = weights(in, count, heads, seed);
    const auto dense = product(x, width, rows, width, w, in, count, heads, in);
    const auto padded = product(x, width, rows, width + pad, w, in, count, heads, in);
    expect(dense.size() == padded.size() && std::memcmp(dense.data(), padded.data(), dense.size() * sizeof(float)) == 0, what);
}

}  // namespace

int main() {
    using strix::a8_row_pitch;
    // Policy: the aliasing pitches of the probe get the pad, the others keep the dense layout.
    expect(a8_row_pitch(16384U) == 16512U && a8_row_pitch(32768U) == 32896U && a8_row_pitch(8192U) == 8320U, "pad aliasing");
    expect(a8_row_pitch(3072U) == 3072U && a8_row_pitch(512U) == 512U && a8_row_pitch(7168U) == 7168U, "dense elsewhere");
    for (std::uint32_t width = 32U; width <= (1U << 20); width += 32U) {
        const std::uint32_t p = a8_row_pitch(width);
        if (p < width || p % 32U != 0U || p % strix::kA8PitchAliasBytes == 0U || p - width > strix::kA8PitchPadBytes) {
            std::fprintf(stderr, "FAIL pitch invariant width %u pitch %u\n", width, p);
            ++failures;
            break;
        }
    }
    expect(strix::a8_codes_bytes(16384U, 2048U) == std::size_t{16512U} * 2048U, "codes bytes");
    expect(strix::a8_group_count(16384U, 2048U) == std::size_t{516U} * 2048U, "group count");
    // Bits: K_b form (256 in per head -> outputs), V_b form (512 in per head), one wide unbatched product.
    same_bits_at_any_pitch(256U, 24U, 4U, 128U, "K_b form same bits dense vs padded pitch");
    same_bits_at_any_pitch(512U, 16U, 3U, 128U, "V_b form same bits dense vs padded pitch");
    same_bits_at_any_pitch(1024U, 8U, 1U, 128U, "wide form same bits dense vs padded pitch");
    if (failures) return 1;
    std::printf("a8_row_pitch_tests: PASS\n");
    return 0;
}
