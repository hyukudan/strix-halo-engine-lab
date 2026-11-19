// Decode weight-read geometry (include/strix/decode_weight_sweep.hpp).
// Invariants:
//   1. The HIP IU4 scalar-activation form reads the weights ONCE for every window of 1..16 rows (one sweep), with the
//      activation operands of a 16-row sweep in two 8-row chunks; the vector form keeps 8 rows per sweep (16 -> 2).
//   2. The loop nest of core/w4a16.hpp wave_rows (sweep -> operand chunk -> row) visits every row of the bucket exactly
//      once, in ascending order inside a chunk.
//   3. Per-row results do not depend on the geometry: a CPU model of the per-row op sequence (group dot chain, then
//      fmaf(scale, dot, acc), groups ascending) gives bitwise the same F32 for rows-per-sweep 1, 4, 8, 16 and operand
//      chunks 4, 8, 16, because each row's operations and their order are the same; only the grouping changes.
#include "strix/decode_weight_sweep.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

using strix::decode::WeightSweepTraits;

// Rows visited by the wave_rows loop nest for one bucket: sweeps of sweep_rows, operand chunks of operand_rows.
std::vector<unsigned> visit_order(unsigned bucket, WeightSweepTraits t) {
    std::vector<unsigned> rows;
    const unsigned per = strix::decode::sweep_rows(bucket, t);
    for (unsigned s = 0; s < strix::decode::weight_sweeps(bucket, t); ++s)
        for (unsigned xb = 0; xb < per; xb += t.operand_rows)
            for (unsigned i = 0; i < t.operand_rows && xb + i < per; ++i) rows.push_back(s * per + xb + i);
    return rows;
}

// CPU model of one output column over `groups` K64 groups for `bucket` rows, computed in the given geometry.
std::vector<float> model(const std::vector<float>& w, const std::vector<float>& scale, const std::vector<float>& x, unsigned groups,
                         unsigned bucket, WeightSweepTraits t) {
    std::vector<float> out(bucket, 0.0F);
    const unsigned per = strix::decode::sweep_rows(bucket, t);
    for (unsigned s = 0; s < strix::decode::weight_sweeps(bucket, t); ++s) {
        std::vector<float> acc(per, 0.0F);
        for (unsigned g = 0; g < groups; ++g) {
            std::vector<float> dot(per, 0.0F);
            for (unsigned k = 0; k < 64U; ++k)  // one weight word decoded once, used by every row of the sweep
                for (unsigned xb = 0; xb < per; xb += t.operand_rows)
                    for (unsigned i = 0; i < t.operand_rows && xb + i < per; ++i) {
                        const unsigned r = s * per + xb + i;
                        dot[xb + i] = std::fma(w[g * 64U + k], x[r * groups * 64U + g * 64U + k], dot[xb + i]);
                    }
            for (unsigned i = 0; i < per; ++i) acc[i] = std::fma(scale[g], dot[i], acc[i]);
        }
        for (unsigned i = 0; i < per; ++i) out[s * per + i] = acc[i];
    }
    return out;
}

}  // namespace

int main() {
    using namespace strix::decode;
    for (unsigned b : {1U, 2U, 4U, 8U, 16U}) {
        check(weight_sweeps(b, kHipIu4ScalarX) == 1U, "scalar-X: one weight stream for every bucket");
        check(sweep_rows(b, kHipIu4ScalarX) == b, "scalar-X: the whole bucket in one sweep");
        check(operand_chunks(b, kHipIu4ScalarX) == (b > 8U ? 2U : 1U), "scalar-X: 8-row operand chunks");
        check(weight_sweeps(b, kHipIu4VectorX) == (b > 8U ? 2U : 1U), "vector-X: 8 rows per sweep");
    }
    check(kHipIu4ScalarX.operand_rows * 8U <= 64U, "scalar-X operand words fit 64 SGPRs");

    const WeightSweepTraits geometries[] = {{1U, 1U}, {4U, 4U}, {8U, 8U}, {8U, 4U}, {16U, 8U}, {16U, 4U}, {16U, 16U}};
    for (const WeightSweepTraits t : geometries)
        for (unsigned b : {1U, 2U, 4U, 8U, 16U}) {
            const std::vector<unsigned> v = visit_order(b, t);
            bool once = v.size() == b;
            for (unsigned i = 0; once && i < v.size(); ++i) once = v[i] == i;
            check(once, "every row visited once, ascending");
        }

    std::mt19937 rng(1118U);
    std::uniform_real_distribution<float> u(-2.0F, 2.0F);
    unsigned cases = 0;
    for (unsigned groups : {10U, 12U, 34U})
        for (int rep = 0; rep < 8; ++rep, ++cases) {
            std::vector<float> w(groups * 64U), scale(groups), x(16U * groups * 64U);
            for (float& v : w) v = std::nearbyint(u(rng) * 4.0F);  // signed nibble values
            for (float& v : scale) v = u(rng) * 0.01F;
            for (float& v : x) v = u(rng);
            const std::vector<float> ref = model(w, scale, x, groups, 16U, {1U, 1U});
            for (const WeightSweepTraits t : geometries) {
                const std::vector<float> got = model(w, scale, x, groups, 16U, t);
                check(std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) == 0, "per-row result bitwise for any geometry");
            }
        }
    if (failures != 0) {
        std::fprintf(stderr, "decode weight sweep: %d failures\n", failures);
        return 1;
    }
    std::printf("decode weight sweep: PASS (%u random columns x 7 geometries)\n", cases);
    return 0;
}
