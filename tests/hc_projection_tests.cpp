// Normalized HC projection (core/hc_core.hpp NormalizedProjection): bitwise against an independent scalar of the
// R1b order (256 lane-strided squares + pairwise tree, rr = 1 / sqrt(ss / n + eps), f = x * rr, glm r1b_dot over 64
// blocks of 256), and the 8-lane distributed tree against the serial 64-leaf tree.
#include "core/hc_core.hpp"
#include "glm/common/glm_math.hpp"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
namespace hc = strix::core::hc;
namespace glm = strix::glm;

struct Math {
    static float rstd(float ss, unsigned n, float eps) { return glm::inv_sqrt(ss / static_cast<float>(n) + eps); }
};
using P = hc::NormalizedProjection<256, 256, 64>;
constexpr unsigned kColumns = 4 * 4096, kMixes = 24;
static_assert(P::columns == kColumns);

void same(float actual, float expected, const char* stage, unsigned index) {
    if (std::bit_cast<std::uint32_t>(actual) != std::bit_cast<std::uint32_t>(expected)) {
        std::fprintf(stderr, "%s[%u]: %a != %a\n", stage, index, static_cast<double>(actual), static_cast<double>(expected));
        std::exit(1);
    }
}

struct Lcg {
    std::uint32_t s;
    float next(int exp_span) {
        s = s * 1664525U + 1013904223U;
        const auto mant = static_cast<float>(s >> 8U) * 0x1p-24F - 0.5F;
        s = s * 1664525U + 1013904223U;
        const int e = static_cast<int>(s >> 27U) % (2 * exp_span + 1) - exp_span;
        return mant * static_cast<float>(1U << (e + exp_span)) / static_cast<float>(1U << exp_span);
    }
};

// Independent scalar of the R1b producer (written from the private kernels, not from the trait).
void reference(const float* x, const float* fn, float* f, float* z) {
    float s[256];
    for (unsigned l = 0; l < 256; ++l) {
        float a = 0.0F;
        for (unsigned i = l; i < kColumns; i += 256) a = __builtin_fmaf(x[i], x[i], a);
        s[l] = a;
    }
    for (unsigned h = 128; h > 0; h >>= 1U)
        for (unsigned l = 0; l < h; ++l) s[l] = s[l] + s[l + h];
    const float rr = glm::inv_sqrt(s[0] / static_cast<float>(kColumns) + glm::kEps);
    for (unsigned i = 0; i < kColumns; ++i) f[i] = x[i] * rr;
    for (unsigned o = 0; o < kMixes; ++o) {
        const float* w = fn + static_cast<std::size_t>(o) * kColumns;
        z[o] = glm::r1b_dot([&](unsigned i) { return f[i]; }, [&](unsigned i) { return w[i]; }, kColumns);
    }
}

// Host emulation of lanes8_tree: lane a holds partials a + 8 k; local levels 4, 2, 1; then shfl_down 4, 2, 1 in width 8.
float lanes8_emulated(const float* src) {
    float acc[8];
    for (unsigned lane = 0; lane < 8; ++lane) {
        float p[8];
        for (unsigned k = 0; k < 8; ++k) p[k] = src[lane + 8 * k];
        for (unsigned h = 4; h > 0; h >>= 1U)
            for (unsigned k = 0; k < h; ++k) p[k] = p[k] + p[k + h];
        acc[lane] = p[0];
    }
    for (unsigned h = 4; h > 0; h >>= 1U) {
        float next[8];
        for (unsigned lane = 0; lane < 8; ++lane) next[lane] = lane < h ? acc[lane] + acc[lane + h] : acc[lane];
        for (unsigned lane = 0; lane < 8; ++lane) acc[lane] = next[lane];
    }
    return acc[0];
}

}  // namespace

int main() {
    std::vector<float> x(kColumns), fn(static_cast<std::size_t>(kMixes) * kColumns), f(kColumns), fr(kColumns);
    float z[kMixes], zr[kMixes];
    unsigned rows = 0;
    for (std::uint32_t seed = 1; seed <= 12; ++seed) {
        Lcg g{seed * 2654435761U};
        const int span = static_cast<int>(seed % 4U) * 4 + 2;
        for (auto& v : x) v = g.next(span);
        for (auto& v : fn) v = g.next(3);
        if (seed == 12)
            for (auto& v : x) v = 0.0F;  // all-zero row: rr = 1 / sqrt(eps)
        P::project_row<Math>(x.data(), fn.data(), kMixes, glm::kEps, f.data(), z);
        reference(x.data(), fn.data(), fr.data(), zr);
        for (unsigned i = 0; i < kColumns; ++i) same(f[i], fr[i], "normalized", i);
        for (unsigned o = 0; o < kMixes; ++o) same(z[o], zr[o], "projected", o);
        for (unsigned l = 0; l < 256; l += 17) same(P::lane_squares_staged(x.data(), l), P::lane_squares(x.data(), l), "staged", l);
        for (unsigned o = 0; o < kMixes; ++o) {
            float p[64], q[64];
            for (unsigned j = 0; j < 64; ++j)
                q[j] = p[j] = P::block_dot(f.data() + j * 256, fn.data() + static_cast<std::size_t>(o) * kColumns + j * 256);
            same(lanes8_emulated(q), P::tree<64>(p), "lanes8", o);
        }
        ++rows;
    }
    std::printf("Normalized HC projection bit comparisons passed (%u rows x %u mixes, squares, staged squares, lanes8 tree)\n", rows,
                kMixes);
    return 0;
}
