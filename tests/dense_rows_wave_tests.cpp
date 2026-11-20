// Host checks of the wave form of the head-family dense kernel (strix/dense_rows_wave.hpp): the (lane, local) leaf map and
// the cross-lane / local tree reproduce the block form of lm_head_kernel BIT FOR BIT. Both forms are emulated with the
// device arithmetic (one fused multiply-add per leaf element, one FP32 add per tree pair) on random data, at the Hy4
// head / router / indexer widths, a width that is not a multiple of 256 and widths below 256; plus the admissibility rule.
#include "strix/dense_rows_wave.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace drw = strix::dense_rows_wave;

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
std::uint32_t lcg(std::uint32_t& s) {
    s = s * 1664525U + 1013904223U;
    return s;
}
float random_float(std::uint32_t& s) {
    const auto u = lcg(s);
    const float m = static_cast<float>(u >> 8U) / 16777216.0F - 0.5F;
    return (u & 7U) == 0U ? m * 64.0F : m;
}

// lm_head_kernel: thread t sums features t + 256 k (k order), then s[t] += s[t + d] for t < d, d = 128 .. 1.
float block_form(const std::vector<float>& x, const std::vector<float>& w) {
    const auto width = static_cast<std::uint32_t>(x.size());
    std::vector<float> s(drw::kLeaves, 0.0F);
    for (std::uint32_t t = 0; t < drw::kLeaves; ++t)
        for (std::uint32_t f = t; f < width; f += drw::kLeaves) s[t] = std::fma(x[f], w[f], s[t]);
    for (std::uint32_t d = drw::kLeaves / 2U; d > 0; d >>= 1U)
        for (std::uint32_t t = 0; t < d; ++t) s[t] = s[t] + s[t + d];
    return s[0];
}

// dense_rows_wave_kernel: lane l keeps leaves leaf(l, j); chunk c loads features c + leaf(l, 0) .. + kLocal - 1 when the
// first is in range; cross-lane stages (lane distance 16 .. 1, the low lanes add the partner) then the local stages.
float wave_form(const std::vector<float>& x, const std::vector<float>& w) {
    const auto width = static_cast<std::uint32_t>(x.size());
    std::vector<std::vector<float>> p(drw::kLanes, std::vector<float>(drw::kLocal, 0.0F));
    for (std::uint32_t chunk = 0; chunk < width; chunk += drw::kLeaves)
        for (std::uint32_t l = 0; l < drw::kLanes; ++l) {
            const std::uint32_t base = chunk + drw::leaf(l, 0U);
            if (base >= width) continue;
            for (std::uint32_t j = 0; j < drw::kLocal; ++j) p[l][j] = std::fma(x[base + j], w[base + j], p[l][j]);
        }
    for (std::uint32_t dl = drw::kLanes / 2U; dl > 0; dl >>= 1U)
        for (std::uint32_t l = 0; l < dl; ++l)
            for (std::uint32_t j = 0; j < drw::kLocal; ++j) p[l][j] = p[l][j] + p[l + dl][j];
    for (std::uint32_t d = drw::kLocal / 2U; d > 0; d >>= 1U)
        for (std::uint32_t j = 0; j < d; ++j) p[0][j] = p[0][j] + p[0][j + d];
    return p[0][0];
}

}  // namespace

int main() {
    // The leaf map is a bijection onto the 256 block leaves, and the cross-lane stages are exactly the block distances >= 8.
    std::vector<int> seen(drw::kLeaves, 0);
    for (std::uint32_t l = 0; l < drw::kLanes; ++l)
        for (std::uint32_t j = 0; j < drw::kLocal; ++j) ++seen[drw::leaf(l, j)];
    bool bijective = true;
    for (const int c : seen) bijective = bijective && c == 1;
    check(bijective, "leaf(lane, local) covers each of the 256 block leaves once");
    bool pairs = true;
    for (std::uint32_t d = drw::kLeaves / 2U; d > 0; d >>= 1U)
        for (std::uint32_t v = 0; v < d; ++v) {
            const std::uint32_t l = v / drw::kLocal, j = v % drw::kLocal, pv = v + d;
            if (drw::cross_lane(d)) pairs = pairs && pv / drw::kLocal == l + d / drw::kLocal && pv % drw::kLocal == j;
            else pairs = pairs && l == 0U && pv / drw::kLocal == 0U && pv % drw::kLocal == j + d;
        }
    check(pairs, "block tree pair (v, v + d) = cross-lane (lane + d / 8, same local) for d >= 8, local (j + d) below");

    std::uint32_t seed = 0x5eed1234U;
    const std::uint32_t widths[] = {3072U, 1024U, 128U, 8U, 40U, 264U, 3000U, 6144U, 16384U};
    bool equal = true;
    std::uint32_t cases = 0;
    for (const auto width : widths)
        for (int rep = 0; rep < 24; ++rep) {
            std::vector<float> x(width), w(width);
            for (auto& v : x) v = random_float(seed);
            for (auto& v : w) v = std::bit_cast<float>(std::bit_cast<std::uint32_t>(random_float(seed)) & 0xffff0000U);  // BF16 values
            if (rep % 3 == 2)
                for (auto& v : w) v = random_float(seed);  // F32 weights
            const float b = block_form(x, w), v = wave_form(x, w);
            equal = equal && std::bit_cast<std::uint32_t>(b) == std::bit_cast<std::uint32_t>(v);
            ++cases;
        }
    std::printf("cases %u\n", cases);
    check(equal, "wave form == block form, bitwise, widths 8 .. 16384 (incl. 40, 264, 3000: not multiples of 256)");

    check(drw::admissible(3072U, 3072U, 2U, 0x1000U, 0x2000U, 4U), "admissible: BF16 head, fp32 input, aligned");
    check(!drw::admissible(3004U, 3004U, 2U, 0x1000U, 0x2000U, 4U), "not admissible: width % 8 != 0");
    check(!drw::admissible(3072U, 3076U, 4U, 0x1000U, 0x2000U, 4U), "not admissible: activation row stride % 8 != 0");
    check(!drw::admissible(3072U, 3072U, 2U, 0x1008U, 0x2000U, 4U), "not admissible: activation not 16-byte aligned");
    check(!drw::admissible(3072U, 3072U, 2U, 0x1000U, 0x2004U, 4U), "not admissible: weights not 16-byte aligned");
    check(!drw::admissible(3072U, 3072U, 1U, 0x1000U, 0x2000U, 4U), "not admissible: 1-byte weights (quantised kinds)");
    check(!drw::admissible(0U, 0U, 4U, 0x1000U, 0x2000U, 4U), "not admissible: width 0");
    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
