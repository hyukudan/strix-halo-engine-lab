#include "strix/hip/legacy_intrinsics.hpp"
#include "glm/prefill/glm_moe_traits.hpp"
#include "glm/prefill/glm_pf_aux.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>

using Policy = strix::glm::pf::MoePolicy;
static_assert(Policy::kSplits == 1);
static_assert(Policy::kRowBlock == 16 && Policy::kGateUpInterleave == 2);
static_assert(Policy::kAscendingRouteOrder);
static_assert(Policy::kG3Pitch == 136 && Policy::kG3Blocks == 8);
static_assert(!Policy::accepts_bits(2) && Policy::accepts_bits(3));
static_assert(Policy::accepts_bits(6) && !Policy::accepts_bits(7));
namespace {
void require(bool value) { if (!value) throw std::runtime_error("aux policy mismatch"); }
void set_switch(const char* value) {
#ifdef _WIN32
    require(_putenv_s("STRIX_GLM_PF_AUX_TEST", value ? value : "") == 0);
#else
    require(value ? setenv("STRIX_GLM_PF_AUX_TEST", value, 1) == 0 : unsetenv("STRIX_GLM_PF_AUX_TEST") == 0);
#endif
}
void test_norm(unsigned width, unsigned seed) {
    namespace A = strix::glm::pf::aux;
    std::array<float, 4096> x{};
    std::array<float, 256> old{};
    std::array<float, 128> folded{};
    // Cancellation is absent in squares, but exponent gaps, signed zeros,
    // subnormals and overflow exercise rounding at every tree level.
    const unsigned kind = seed % 3U;
    for (unsigned i = 0; i < width; ++i) {
        seed = seed * 1664525U + 1013904223U;
        unsigned bits = kind ? seed & 0xbfffffffU : seed;
        if ((bits & 0x7f800000U) == 0x7f800000U) bits ^= 0x00800000U;
        x[i] = std::bit_cast<float>(bits);
    }
    for (unsigned lane = 0; lane < 256; ++lane)
        for (unsigned i = lane; i < width; i += 256) old[lane] = std::fma(x[i], x[i], old[lane]);
    for (unsigned lane = 0; lane < 128; ++lane)
        folded[lane] = A::norm_pair(x.data(), width, lane,
                                   [](float a, float b, float c) { return std::fma(a, b, c); });
    for (unsigned h = 128; h; h /= 2)
        for (unsigned lane = 0; lane < h; ++lane) old[lane] = old[lane] + old[lane + h];
    for (unsigned h = 64; h; h /= 2)
        for (unsigned lane = 0; lane < h; ++lane) folded[lane] = folded[lane] + folded[lane + h];
    require(std::bit_cast<unsigned>(old[0]) == std::bit_cast<unsigned>(folded[0]));
}
}
int main() {
    namespace A = strix::glm::pf::aux;
    static_assert(!A::input_admitted(0) && A::input_admitted(128) && A::input_admitted(512));
    static_assert(!A::input_admitted(513) && !A::input_admitted(1024) && !A::input_admitted(129));
    for (unsigned a : {0U, 1U, 32U, 63U, 64U, 65U, 128U, 288U})
        for (unsigned b : {0U, 32U, 127U, 128U, 289U}) {
            const unsigned widths[]{a, b, 17U};
            require(A::dense_columns(widths, 0) == 0);
            require(A::dense_columns(widths, 2) == (a > b ? a : b));
            require(A::dense_columns(widths, 3) >= 17U);
        }
    for (unsigned width : {512U, 768U, 1024U, 1536U, 4096U})
        for (unsigned seed = 0; seed < 128; ++seed) test_norm(width, seed);
    set_switch(nullptr); require(!A::enabled("STRIX_GLM_PF_AUX_TEST"));
    set_switch("0"); require(!A::enabled("STRIX_GLM_PF_AUX_TEST"));
    set_switch("1"); require(A::enabled("STRIX_GLM_PF_AUX_TEST"));
    for (const char* bad : {"", "2", "on", "-1", "1x"}) {
        // Windows removes a variable when assigned an empty string.
#ifdef _WIN32
        if (!*bad) continue;
#endif
        set_switch(bad);
        bool rejected = false;
        try { (void)A::enabled("STRIX_GLM_PF_AUX_TEST"); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected);
    }
    set_switch(nullptr);
    std::puts("GLM auxiliary policies: PASS");
}
