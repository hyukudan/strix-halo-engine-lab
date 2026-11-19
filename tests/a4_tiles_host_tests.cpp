// Host-only: the A4 activation-group rule of src/core/a4_tiles.hpp / integer_math (a4_step_f, a4_code_f, reference).
//  1. the FP32 form is the fast_i4 rule: for every int8 value and every integer amax 1..127 it gives the integer form's
//     code (a4_code(v, a4_step(amax)));
//  2. a group: codes in [-7, 7], the amax element maps to +-7, |v - value| <= step / 2 (the rounding bound), an all-zero
//     group stays zero, a NaN gives code 0 instead of undefined behaviour;
//  3. groups are independent: changing one group of a row leaves the other groups' values unchanged;
//  4. the Hadamard-128 is orthonormal and its own inverse (norm kept, H(H x) == x within 4 ulp of the row scale).
#include "core/a4_tiles.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); ++failures; }
}
namespace im = strix::core::integer_math;
namespace a4 = strix::core::a4_tiles;
}  // namespace

int main() {
    {  // 1
        bool same = true;
        for (int amax = 1; amax <= 127; ++amax)
            for (int v = -amax; v <= amax; ++v)
                same = same && im::a4_code(v, im::a4_step(amax)) == im::a4_code_f(float(v), im::a4_step_f(float(amax)));
        check(same, "FP32 A4 rule != the fast_i4 integer rule on int8 inputs");
    }
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    for (unsigned group : {32U, 64U}) {  // 2 and 3
        std::vector<float> x(512), y;
        for (auto& v : x) v = nd(rng) * 3.0F;
        x[7] = 55.0F;                                           // an outlier
        for (unsigned k = 128; k < 128U + group; ++k) x[k] = 0.0F;  // an all-zero group
        y = x;
        a4::reference(y.data(), 512U, group, false);
        bool bounded = true, ends = true, grid = true;
        for (unsigned g = 0; g < 512U; g += group) {
            float amax = 0.0F;
            for (unsigned k = g; k < g + group; ++k) amax = std::fmax(amax, std::fabs(x[k]));
            const float step = im::a4_step_f(amax);
            for (unsigned k = g; k < g + group; ++k) {
                const int code = im::a4_code_f(x[k], step);
                grid = grid && code >= -7 && code <= 7 && y[k] == float(code) * step;
                bounded = bounded && std::fabs(x[k] - y[k]) <= 0.5F * step * (1.0F + 1e-6F);
                if (std::fabs(x[k]) == amax && amax > 0.0F) ends = ends && std::abs(code) == 7;
            }
        }
        check(grid, "values off the code * step grid or codes outside +-7");
        check(bounded, "|v - value| > step / 2");
        check(ends, "the amax element is not coded +-7");
        bool zero = true;
        for (unsigned k = 128; k < 128U + group; ++k) zero = zero && y[k] == 0.0F;
        check(zero, "an all-zero group is not zero");
        auto x2 = x;
        x2[300] = 1000.0F;  // group of input 300 only
        a4::reference(x2.data(), 512U, group, false);
        bool independent = true;
        for (unsigned k = 0; k < 512U; ++k)
            if (k / group != 300U / group) independent = independent && x2[k] == y[k];
        check(independent, "a group's values depend on another group");
    }
    check(im::a4_code_f(std::nanf(""), 1.0F) == 0, "NaN input does not give code 0");
    check(im::a4_step_f(0.0F) == 1.0F && im::a4_step_f(1e-45F) == 1.0F, "zero / underflowing step is not 1");
    {  // 4
        std::vector<float> v(128), w;
        double n0 = 0.0, n1 = 0.0, worst = 0.0, scale = 0.0;
        for (auto& e : v) e = nd(rng), n0 += double(e) * e, scale = std::fmax(scale, std::fabs(e));
        w = v;
        a4::fwht128(w.data());
        for (float e : w) n1 += double(e) * e;
        a4::fwht128(w.data());
        for (unsigned i = 0; i < 128U; ++i) worst = std::fmax(worst, std::fabs(double(w[i]) - v[i]));
        check(std::fabs(n1 / n0 - 1.0) < 1e-5, "Hadamard-128 does not keep the norm");
        check(worst <= 4.0 * scale * 0x1p-23, "Hadamard-128 is not its own inverse");
    }
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
