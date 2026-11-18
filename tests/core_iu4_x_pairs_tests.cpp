// Pre-paired activation layout of the W4A16 core (src/core/iu4_x_pairs.hpp): the words the scalar-activation path loads
// must be exactly the dot2 operands the vector path builds with pair_activations, for every row and k-tile, so both
// paths issue the same v_dot2 chain on the same operands (bitwise equal outputs).
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "core/iu4_x_pairs.hpp"

namespace {
int failures = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                       \
        }                                                                     \
    } while (0)

using strix::core::iu4_pairs::pair_words;
using strix::core::iu4_pairs::paired_index;

std::uint32_t word(const std::vector<std::uint16_t>& h, std::size_t e) { return h[e] | (std::uint32_t(h[e + 1]) << 16U); }
}  // namespace

int main() {
    // Permutation: inside its own 8-element block, a bijection over [0, rows * inputs).
    for (const std::size_t inputs : {64U, 128U, 5120U}) {
        const std::size_t n = 3U * inputs;
        std::vector<int> seen(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t d = paired_index(i);
            CHECK(d < n && d / 8U == i / 8U);
            if (d < n) ++seen[d];
        }
        bool once = true;
        for (const int s : seen) once = once && s == 1;
        CHECK(once);
    }
    // Producer epilogues (dense_iu4::PreparedStore, decode launch fusion): a norm block of row r stores element k at
    // paired_index(r * inputs + k), a flat SiLU pass at paired_index(i). Both equal prepare()'s flat layout for a whole
    // window of 1..16 rows only if the permutation is row-local: paired_index(r * inputs + k) == r * inputs + paired_index(k).
    std::size_t row_local = 0;
    for (const std::size_t inputs : {5120U, 6144U, 17408U})
        for (std::size_t r = 0; r < 16U; ++r)
            for (std::size_t k = 0; k < inputs; k += 7U, ++row_local)
                CHECK(paired_index(r * inputs + k) == r * inputs + paired_index(k));
    CHECK(row_local > 0U);
    // Spelled-out order of one block: (x0,x4) (x1,x5) (x2,x6) (x3,x7).
    const std::size_t expect[8] = {0U, 2U, 4U, 6U, 1U, 3U, 5U, 7U};
    for (std::size_t j = 0; j < 8U; ++j) CHECK(paired_index(40U + j) == 40U + expect[j]);

    // Operand identity over random FP16 bit patterns (NaN / Inf / denormal patterns included: pure bit moves).
    constexpr std::size_t rows = 5U, inputs = 1024U;
    std::mt19937 rng(11);
    std::vector<std::uint16_t> natural(rows * inputs), paired(rows * inputs);
    for (auto& v : natural) v = static_cast<std::uint16_t>(rng());
    for (std::size_t i = 0; i < natural.size(); ++i) paired[paired_index(i)] = natural[i];
    std::size_t tiles = 0;
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t k = 0; k < inputs; k += 16U, ++tiles) {
            const std::size_t e = r * inputs + k;  // one k-tile: lo = k..k+7, hi = k+8..k+15 (the core's xr[0..3] / xr[4..7])
            for (std::size_t half = 0; half < 2U; ++half) {
                const std::size_t b = e + 8U * half;
                const std::uint32_t a[4] = {word(natural, b), word(natural, b + 2U), word(natural, b + 4U), word(natural, b + 6U)};
                std::uint32_t p[4];
                pair_words(a, p);
                for (std::size_t j = 0; j < 4U; ++j) {
                    CHECK(p[j] == word(paired, b + 2U * j));                               // scalar path operand == vector path
                    CHECK((p[j] & 0xFFFFU) == natural[b + j]);                              // low half = x j
                    CHECK((p[j] >> 16U) == natural[b + j + 4U]);                            // high half = x j + 4 (nibble_pair)
                }
            }
        }
    CHECK(tiles == rows * inputs / 16U);
    if (failures) {
        std::fprintf(stderr, "core_iu4_x_pairs_tests: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("core_iu4_x_pairs_tests: OK (%zu k-tiles, 3 permutation shapes)\n", tiles);
    return 0;
}
