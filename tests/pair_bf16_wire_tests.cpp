// strix/pair/bf16_wire.hpp: a BF16-rounded FP32 word crosses as its high half and widens back to the same 32 bits
// (normals, subnormals, -0, infinities); a word with a non-zero low half is reported; the placer puts slot q's rows into
// columns [q count, (q + 1) count) of the one-node rows, honouring the slot offset and row pitch.
#include "strix/ds41_layer.hpp"  // round_to_bf16 = glu Round::bf16 (the routed experts' down epilogue)
#include "strix/pair/bf16_wire.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}
std::uint32_t bits(float x) {
    std::uint32_t b = 0;
    std::memcpy(&b, &x, 4);
    return b;
}
float from_bits(std::uint32_t b) {
    float x = 0.0F;
    std::memcpy(&x, &b, 4);
    return x;
}
}  // namespace

int main() {
    namespace w = strix::pair::bf16_wire;
    // every high half: the word with a zero low half round-trips bit for bit (NaN included: bits, not values)
    std::uint32_t mismatches = 0;
    for (std::uint32_t h = 0; h <= 0xFFFFU; ++h) {
        const float x = from_bits(h << 16U);
        bool exact = false;
        const auto e = w::encode(x, exact);
        if (!exact || e != h || bits(w::decode(e)) != (h << 16U)) ++mismatches;
    }
    check(mismatches == 0U, "all 65536 bfloat16 words round-trip exactly");
    bool exact = true;
    (void)w::encode(1.0F + std::numeric_limits<float>::epsilon(), exact);
    check(!exact, "a word with low bits is reported");
    (void)w::encode(from_bits(0x3F800001U), exact);
    check(!exact, "low bit 0 is reported");
    (void)w::encode(-0.0F, exact);
    check(exact && bits(w::decode(w::encode(-0.0F, exact))) == 0x80000000U, "-0 keeps its sign");
    check(w::wire_bytes(10U) == 20U, "wire bytes");

    std::vector<float> src{1.0F, -2.5F, 0.15625F, 3.0F};
    std::vector<std::uint16_t> half(src.size());
    check(w::pack_ref(src.data(), src.size(), half.data()), "pack_ref exact rows");
    src[2] = 0.1F;  // not a bfloat16
    check(!w::pack_ref(src.data(), src.size(), half.data()), "pack_ref reports an inexact row");

    // placer: R = 2 rows, count 3 per rank, slot = 2 x pitch 4 (+ at 1), cols 6
    const std::size_t slot = 9, at = 1, pitch = 4, count = 3, cols = 6;
    const std::uint32_t R = 2;
    std::vector<std::uint16_t> wire(2U * slot);
    for (std::size_t q = 0; q < 2U; ++q)
        for (std::uint32_t r = 0; r < R; ++r)
            for (std::size_t j = 0; j < count; ++j) {
                bool e = true;
                wire[q * slot + at + r * pitch + j] = w::encode(static_cast<float>(100U * q + 10U * r + j), e);
            }
    std::vector<float> full(R * cols, -1.0F);
    w::place_ref(wire.data(), slot, at, pitch, count, R, full.data(), cols);
    bool placed = true;
    for (std::uint32_t r = 0; r < R; ++r)
        for (std::size_t c = 0; c < cols; ++c) {
            const std::size_t q = c / count, j = c % count;
            placed = placed && full[r * cols + c] == static_cast<float>(100U * q + 10U * r + j);
        }
    check(placed, "place_ref puts slot q rows into columns q count .. (q + 1) count");

    // A TP rank's routed-expert slots: unrounded down outputs are refused by the exact wire; rounded by the bf16 epilogue
    // (glu Round::bf16 = ds41::round_to_bf16) they cross exactly, and the combine's rounding on read is then the identity
    // (the joined value equals what one node reads).
    {
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> mag(-30.0F, 30.0F), sgn(-1.0F, 1.0F);
        std::vector<float> raw(4096), rounded(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) {
            raw[i] = std::ldexp(sgn(rng), static_cast<int>(mag(rng)));
            rounded[i] = strix::ds41::round_to_bf16(raw[i]);
        }
        std::vector<std::uint16_t> wire16(raw.size());
        check(!w::pack_ref(raw.data(), raw.size(), wire16.data()), "unrounded expert slots are refused by the exact wire");
        check(w::pack_ref(rounded.data(), rounded.size(), wire16.data()), "bf16-rounded expert slots cross exactly");
        bool same = true;
        for (std::size_t i = 0; i < raw.size(); ++i)
            same = same && bits(strix::ds41::round_to_bf16(w::decode(wire16[i]))) == bits(rounded[i]);
        check(same, "rounding a joined bf16 slot again is the identity (one-node combine bits)");
    }

    if (failures != 0) return 1;
    std::printf("pair bf16 wire tests: PASS\n");
    return 0;
}
