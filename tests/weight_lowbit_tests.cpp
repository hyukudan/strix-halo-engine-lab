// Low-bit resident row storage (strix/weight/host_lowbit.hpp): the host encoder, the loader that produces it from BF16
// rows (whole and partial reads), and the device chunk decoder of the dense rows family (core/weight_decode.hpp,
// host build) agree element by element; hand-written bytes decode to their specified codes; ties round to even with
// fixed bytes in any rounding mode; a scale beyond FP16 is refused; the chosen scale never reconstructs worse than the
// unclipped scale; the Q8_0 loader is unchanged by the shared block source.
#include "core/iu4_onfly.hpp"
#include "core/weight_decode.hpp"
#include "strix/weight/host_lowbit.hpp"
#include "strix/weight/host_q8.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

namespace wei = strix::weight;
namespace wd = strix::core::weight_decode;

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

std::uint16_t to_bf16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, 4);
    return static_cast<std::uint16_t>((x + 0x7fffU + ((x >> 16U) & 1U)) >> 16U);
}

std::vector<std::uint16_t> random_matrix(std::size_t n, std::size_t k, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 0.02F);
    std::vector<std::uint16_t> bits(n * k);
    for (std::size_t i = 0; i < bits.size(); ++i) {
        float v = normal(rng);
        if (i % 97U == 5U) v *= 12.0F;  // outliers exercise the clip search
        bits[i] = to_bf16(v);
    }
    // One all-zero group (row 0, group 1) and one single-signed group (row 1, group 0).
    for (std::size_t j = 64; j < 128; ++j) bits[j] = 0;
    for (std::size_t j = 0; j < 64; ++j) bits[k + j] = to_bf16(std::fabs(strix::core::onfly::bf16_bits_to_f32(bits[k + j])));
    return bits;
}

template <class Storage, wd::Format F>
void check_format(const char* name) {
    constexpr std::size_t n = 3, k = 384;
    const auto bits = random_matrix(n, k, 7U + Storage::high_bytes);
    const wei::Shape shape{{n, k, 0, 0}, 2};
    const auto storage = F == wd::Format::q4_g64 ? wei::kQ4G64WeightStorage : wei::kQ6G64WeightStorage;

    wei::MatrixLoadGeometry geometry{};
    check(bool(wei::matrix_load_geometry(storage, shape, geometry)), "geometry");
    check(geometry.row_bytes == k / 128U * Storage::bytes && geometry.bytes == n * geometry.row_bytes, "row bytes");
    check(geometry.row_bytes == wd::row_bytes(F, k), "dense family row bytes");
    wei::MatrixLoadGeometry odd{};
    check(!wei::matrix_load_geometry(storage, {{n, 192, 0, 0}, 2}, odd), "K % 128 refused");

    const auto source = wei::memory_source(std::as_bytes(std::span(bits)));
    const wei::Slice slice{source, {0, bits.size() * 2U}};
    std::vector<std::byte> loaded(geometry.bytes);
    check(bool(wei::load_bf16_matrix(slice, shape, storage, loaded)), "load");

    // Direct encoder == loader; reference decode == device chunk decoder; codes and error bounds.
    std::vector<std::uint8_t> direct(geometry.bytes);
    std::vector<float> w(n * k);
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = strix::core::onfly::bf16_bits_to_f32(bits[i]);
    for (std::size_t b = 0; b < w.size() / 128U; ++b)
        check(wei::lowbit::encode_block<Storage>(w.data() + b * 128U, direct.data() + b * Storage::bytes), "encode");
    check(std::memcmp(direct.data(), loaded.data(), direct.size()) == 0, "loader bytes == encoder bytes");
    // The bytes do not depend on the caller's rounding mode, and the caller's mode survives the encoder.
    for (const int mode : {FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
        std::vector<std::uint8_t> moded(geometry.bytes);
        if (std::fesetround(mode) != 0) continue;
        for (std::size_t b = 0; b < w.size() / 128U; ++b)
            check(wei::lowbit::encode_block<Storage>(w.data() + b * 128U, moded.data() + b * Storage::bytes), "encode moded");
        check(std::fegetround() == mode, "rounding mode restored");
        std::fesetround(FE_TONEAREST);
        check(moded == direct, "bytes independent of rounding mode");
    }

    for (std::size_t r = 0; r < n; ++r) {
        const auto* row = direct.data() + r * geometry.row_bytes;
        for (unsigned i = 0; i < k; i += 16U) {
            const wd::Chunk<F> chunk(row, i);
            for (unsigned e = 0; e < 16U; ++e) {
                const unsigned col = i + e;
                const auto* block = row + (col / 128U) * Storage::bytes;
                const float ref = wei::lowbit::decode<Storage>(block, col % 128U);
                const float dev = chunk.at(e);
                check(std::memcmp(&ref, &dev, 4) == 0, "chunk decode == reference decode");
            }
        }
        for (unsigned g = 0; g < k / 64U; ++g) {
            const auto* block = row + (g / 2U) * Storage::bytes;
            const float s = wei::lowbit::f16_bits_to_f32(Storage::scale_bits(block, g % 2U));
            const float* src = w.data() + r * k + g * 64U;
            float pos = 0, neg = 0, err = 0, err_full = 0;
            for (unsigned j = 0; j < 64; ++j) {
                pos = std::max(pos, src[j]);
                neg = std::max(neg, -src[j]);
            }
            const float base = std::max(pos / float(Storage::qmax), neg / float(-Storage::qmin));
            if (base == 0.0F) {
                check(s == 0.0F, "zero group has zero scale");
                continue;
            }
            const float full = wei::lowbit::f16_bits_to_f32(wei::lowbit::f32_to_f16_bits(base));
            for (unsigned j = 0; j < 64; ++j) {
                const float d = src[j] - wei::lowbit::decode<Storage>(block, (g % 2U) * 64U + j);
                err += d * d;
                const float df = src[j] - full * float(wei::lowbit::quantise<Storage>(src[j], full));
                err_full += df * df;
            }
            check(err <= err_full, "chosen scale no worse than unclipped");
        }
    }

    // Partial reads of the staged source (start and end inside blocks) equal the whole image.
    wei::Slice packed{};
    check(bool(wei::prepare_bf16_matrix_slice(slice, shape, storage, packed)), "prepare");
    check(packed.range.bytes == geometry.bytes, "packed size");
    for (std::uint64_t offset : {0ULL, 1ULL, 67ULL, 99ULL, 150ULL}) {
        const std::uint64_t count = std::min<std::uint64_t>(geometry.bytes - offset, 211ULL);
        std::vector<std::byte> part(count);
        check(bool(packed.source->read({offset, count}, part)), "partial read");
        check(std::memcmp(part.data(), loaded.data() + offset, count) == 0, "partial read bytes");
    }

    // Non-finite input is refused.
    auto bad = bits;
    bad[200] = 0x7f80U;
    const auto bad_source = wei::memory_source(std::as_bytes(std::span(bad)));
    check(!wei::load_bf16_matrix({bad_source, {0, bad.size() * 2U}}, shape, storage, loaded), "nonfinite refused");
    std::cout << name << " ok\n";
}

void check_q8_unchanged() {
    constexpr std::size_t n = 2, k = 96;
    const auto bits = random_matrix(n, k + 32U, 3U);  // reuse the generator, keep the first n * k values
    std::vector<std::uint16_t> m(bits.begin(), bits.begin() + static_cast<std::ptrdiff_t>(n * k));
    const wei::Shape shape{{n, k, 0, 0}, 2};
    const auto source = wei::memory_source(std::as_bytes(std::span(m)));
    const wei::Slice slice{source, {0, m.size() * 2U}};
    const auto expected = strix::core::onfly::bf16_to_q8_0_host(m.data(), n, k);
    std::vector<std::byte> loaded(expected.size());
    check(bool(wei::load_bf16_matrix(slice, shape, wei::kQ8WeightStorage, loaded)), "q8 load");
    check(std::memcmp(loaded.data(), expected.data(), expected.size()) == 0, "q8 bytes unchanged");
    wei::Slice packed{};
    check(bool(wei::prepare_bf16_matrix_slice(slice, shape, wei::kQ8WeightStorage, packed)), "q8 prepare");
    std::vector<std::byte> part(50);
    check(bool(packed.source->read({17, 50}, part)), "q8 partial read");
    check(std::memcmp(part.data(), expected.data() + 17, 50) == 0, "q8 partial bytes unchanged");
}


// Independently specified superblock bytes: signed codes at the ends of the planes (j = 0, 127) and at the scale-group
// boundary (j = 63 | 64), read through the storage accessor, the reference decode and the device chunk decoder.
struct Expect {
    unsigned j;
    int code;
    float value;
};
template <class Storage, wd::Format F>
void check_layout_bytes(const char* name, const std::array<Expect, 4>& expect) {
    std::array<std::uint8_t, Storage::bytes> block{};
    block[0] = 0x01;  // j = 0: +1
    if constexpr (Storage::high_bytes == 0U) {
        block[31] = 0x80;  // j = 63: high nibble of byte 31 = 8 -> -8
        block[32] = 0x07;  // j = 64: +7
        block[63] = 0xF0;  // j = 127: 15 -> -1
    } else {
        block[79] = 0x80;  // j = 63: low 0, high 2 (byte 64 + 15, bits 6..7) -> 32 -> -32
        block[32] = 0x0F;  // j = 64: low 15
        block[80] = 0x01;  //         high 1 (byte 64 + 16, bits 0..1) -> +31
        block[63] = 0xF0;  // j = 127: low 15
        block[95] = 0xC0;  //          high 3 (byte 64 + 31, bits 6..7) -> 63 -> -1
    }
    block[Storage::scale_offset + 1U] = 0x3C;  // group 0 scale 1.0
    block[Storage::scale_offset + 3U] = 0x40;  // group 1 scale 2.0
    for (unsigned j = 0; j < 128U; ++j) {
        int code = 0;
        float value = 0.0F;
        for (const auto& e : expect)
            if (e.j == j) {
                code = e.code;
                value = e.value;
            }
        check(Storage::code(block.data(), j) == code, "layout code");
        check(wei::lowbit::decode<Storage>(block.data(), j) == value, "layout reference decode");
        const wd::Chunk<F> chunk(block.data(), j / 16U * 16U);
        check(chunk.at(j % 16U) == value, "layout chunk decode");
    }
    std::cout << name << " layout ok\n";
}

// Fixed-byte round-to-nearest-even ties: quotients of exactly k + 1/2 against the stored scale (u = 2^-24).
void check_ties() {
    constexpr float u = 0x1p-24F;
    using Q4 = wei::Q4G64Storage;
    using Q6 = wei::Q6G64Storage;
    {  // q4: 63 x 49u and one 45.5u -> scale 7u, codes 7 and 6 (45.5 / 7 = 6.5)
        std::array<float, 128> w{};
        for (unsigned j = 0; j < 63U; ++j) w[j] = 49.0F * u;
        w[63] = 45.5F * u;
        std::array<std::uint8_t, Q4::bytes> expected{}, out{};
        for (unsigned b = 0; b < 31U; ++b) expected[b] = 0x77;
        expected[31] = 0x67;
        expected[64] = 0x07;
        check(wei::lowbit::encode_block<Q4>(w.data(), out.data()), "q4 tie encode");
        check(out == expected, "q4 tie bytes");
        check(wei::lowbit::quantise<Q4>(45.5F * u, 7.0F * u) == 6, "q4 tie 6.5 -> 6");
        check(wei::lowbit::quantise<Q4>(-45.5F * u, 7.0F * u) == -6, "q4 tie -6.5 -> -6");
        check(wei::lowbit::quantise<Q4>(38.5F * u, 7.0F * u) == 6, "q4 tie 5.5 -> 6");
        check(wei::lowbit::quantise<Q4>(-59.5F * u, 7.0F * u) == -8, "q4 tie -8.5 -> -8");
    }
    {  // q6: 62 x 93u, +37.5u, -37.5u -> scale 3u, codes 31, 12, -12 (37.5 / 3 = 12.5)
        std::array<float, 128> w{};
        for (unsigned j = 0; j < 62U; ++j) w[j] = 93.0F * u;
        w[62] = 37.5F * u;
        w[63] = -37.5F * u;
        std::array<std::uint8_t, Q6::bytes> expected{}, out{};
        for (unsigned b = 0; b < 31U; ++b) expected[b] = 0xFF;
        expected[31] = 0x4C;
        for (unsigned b = 64; b < 79U; ++b) expected[b] = 0x55;
        expected[79] = 0xC5;
        expected[96] = 0x03;
        check(wei::lowbit::encode_block<Q6>(w.data(), out.data()), "q6 tie encode");
        check(out == expected, "q6 tie bytes");
    }
    std::cout << "ties ok\n";
}

// A nonzero group whose unclipped scale exceeds FP16 is refused by the encoder, the direct load and the staged read;
// the largest power-of-two scale that fits still encodes exactly.
template <class Storage>
void check_scale_overflow(wei::FormatId storage, const char* name) {
    std::array<float, 128> w{};
    w[5] = 0x1p22F;
    std::array<std::uint8_t, Storage::bytes> out{};
    out.fill(0xA5);
    const auto status = wei::lowbit::encode_block<Storage>(w.data(), out.data());
    check(!status && status.code == wei::lowbit::EncodeStatus::scale_overflow, "overflow status");
    check(std::all_of(out.begin(), out.end(), [](std::uint8_t b) { return b == 0xA5; }), "output untouched on overflow");

    std::vector<std::uint16_t> bits(128, 0);
    bits[5] = to_bf16(0x1p22F);
    const wei::Shape shape{{1, 128, 0, 0}, 2};
    const auto source = wei::memory_source(std::as_bytes(std::span(bits)));
    const wei::Slice slice{source, {0, bits.size() * 2U}};
    std::vector<std::byte> loaded(Storage::bytes);
    const auto direct = wei::load_bf16_matrix(slice, shape, storage, loaded);
    check(!direct && direct.detail.find("FP16") != std::string::npos, "direct load refuses scale overflow");
    wei::Slice packed{};
    const bool prepared = bool(wei::prepare_bf16_matrix_slice(slice, shape, storage, packed));
    check(prepared, "prepare staged slice for scale overflow read");
    if (prepared) {
        std::vector<std::byte> part(Storage::bytes);
        const auto staged = packed.source->read({0, Storage::bytes}, part);
        check(!staged && staged.detail.find("FP16") != std::string::npos, "staged read refuses scale overflow");
    }

    w[5] = 0x1p15F * static_cast<float>(Storage::qmax);  // scale 2^15 fits FP16
    check(wei::lowbit::encode_block<Storage>(w.data(), out.data()), "largest scale encodes");
    check(wei::lowbit::decode<Storage>(out.data(), 5) == w[5] && wei::lowbit::decode<Storage>(out.data(), 6) == 0.0F,
          "largest scale exact");
    std::cout << name << " overflow ok\n";
}
}  // namespace

int main() {
    check_format<wei::Q4G64Storage, wd::Format::q4_g64>("q4_g64");
    check_format<wei::Q6G64Storage, wd::Format::q6_g64>("q6_g64");
    check_q8_unchanged();
    check_layout_bytes<wei::Q4G64Storage, wd::Format::q4_g64>(
        "q4_g64", {{{0, 1, 1.0F}, {63, -8, -8.0F}, {64, 7, 14.0F}, {127, -1, -2.0F}}});
    check_layout_bytes<wei::Q6G64Storage, wd::Format::q6_g64>(
        "q6_g64", {{{0, 1, 1.0F}, {63, -32, -32.0F}, {64, 31, 62.0F}, {127, -1, -2.0F}}});
    check_ties();
    check_scale_overflow<wei::Q4G64Storage>(wei::kQ4G64WeightStorage, "q4_g64");
    check_scale_overflow<wei::Q6G64Storage>(wei::kQ6G64WeightStorage, "q6_g64");
    if (failures) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "weight lowbit tests passed\n";
    return 0;
}
