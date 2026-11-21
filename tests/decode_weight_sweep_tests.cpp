// Decode weight-read geometry (include/strix/decode_weight_sweep.hpp).
// Invariants:
//   1. The HIP IU4 scalar-activation form reads the weights ONCE for every window of 1..16 rows (one sweep), with the
//      activation operands of a 16-row sweep in two 8-row chunks; the vector form keeps 8 rows per sweep (16 -> 2).
//   2. The loop nest of core/w4a16.hpp wave_rows (sweep -> operand chunk -> row) visits every row of the bucket exactly
//      once, in ascending order inside a chunk.
//   3. Per-row results do not depend on the geometry: a CPU model of the per-row op sequence (group dot chain, then
//      fmaf(scale, dot, acc), groups ascending) gives bitwise the same F32 for rows-per-sweep 1, 4, 8, 16 and operand
//      chunks 4, 8, 16, because each row's operations and their order are the same; only the grouping changes.
//   4. Read-ahead (readahead_checks below): the dispatch-order pieces are exactly the GEMV's reads, each record once,
//      and the lead prefix is the longest one within min(bytes, cache / 4); consumers get a side node only for rows
//      1..stream_bound_rows (readahead_rows_ok).
#include "strix/decode_weight_sweep.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <random>
#include <string>
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

// Read-ahead geometry (decode_weight_sweep.hpp): for the split-K tiled GEMV grids of the CUDA NVFP4 verify shapes,
//   (a) the pieces of all dispatched blocks cover every record of every weight exactly once (so a prefetched piece is a
//       piece the GEMV reads, and nothing outside the weights is touched);
//   (b) the piece of (block, t) equals the GEMV's own nested enumeration (z, y, x; tile (x - x0) tpb + t; records
//       [y chunks / ks, (y + 1) chunks / ks));
//   (c) the lead blocks are the longest dispatch-order prefix within the budget, and the budget is min(bytes, cache / 4).
void readahead_case(const strix::decode::ReadaheadGrid& g, const char* name) {
    using namespace strix::decode;
    constexpr unsigned long long kRec = 1152ULL;
    std::vector<std::vector<unsigned char>> hits;  // [job * 2 + plane][tile * chunks + record]
    hits.resize(2U * g.jobs);
    unsigned long long total_bytes = 0;
    for (unsigned j = 0; j < g.jobs; ++j)
        for (unsigned z = 0; z < g.planes[j]; ++z) {
            hits[2U * j + z].assign(static_cast<std::size_t>(g.tiles[j]) * g.chunks, 0U);
            total_bytes += static_cast<unsigned long long>(g.tiles[j]) * g.chunks * kRec;
        }
    bool inside = true, same = true;
    for (unsigned z = 0; z < g.gz; ++z)
        for (unsigned y = 0; y < g.gy; ++y)
            for (unsigned x = 0; x < g.gx; ++x) {
                const unsigned long long lb = x + static_cast<unsigned long long>(g.gx) * (y + static_cast<unsigned long long>(g.gy) * z);
                unsigned j = 0, x0 = 0;
                for (; j < g.jobs; ++j) {
                    const unsigned nb = (g.tiles[j] + g.tiles_per_block - 1U) / g.tiles_per_block;
                    if (x < x0 + nb) break;
                    x0 += nb;
                }
                for (unsigned t = 0; t < g.tiles_per_block; ++t) {
                    const ReadaheadPiece p = readahead_piece(g, lb, t);
                    const unsigned tile = j < g.jobs ? (x - x0) * g.tiles_per_block + t : 0U;
                    const bool live = j < g.jobs && y < g.ks[j] && z < g.planes[j] && tile < g.tiles[j];
                    if (!live) {
                        same = same && p.c1 == p.c0;
                        continue;
                    }
                    const unsigned c0 = static_cast<unsigned>(static_cast<unsigned long long>(y) * g.chunks / g.ks[j]);
                    const unsigned c1 = static_cast<unsigned>(static_cast<unsigned long long>(y + 1U) * g.chunks / g.ks[j]);
                    same = same && p.job == j && p.plane == z && p.tile == tile && p.c0 == c0 && p.c1 == c1;
                    for (unsigned c = p.c0; c < p.c1; ++c) {
                        inside = inside && p.tile < g.tiles[p.job] && c < g.chunks;
                        if (inside) ++hits[2U * p.job + p.plane][static_cast<std::size_t>(p.tile) * g.chunks + c];
                    }
                }
            }
    bool once = true;
    for (const auto& h : hits)
        for (unsigned char v : h) once = once && v == 1U;
    std::string tag(name);
    check(inside, (tag + ": pieces inside the weights").c_str());
    check(same, (tag + ": piece == the GEMV's own block enumeration").c_str());
    check(once, (tag + ": every record read by exactly one piece").c_str());

    const unsigned long long cache = 128ULL << 20U;
    const unsigned long long budget = readahead_budget(total_bytes, cache);
    check(budget == std::min(total_bytes, cache / 4ULL), (tag + ": budget = min(bytes, cache / 4)").c_str());
    const unsigned long long lead = readahead_lead_blocks(g, kRec, budget);
    const unsigned long long blocks = static_cast<unsigned long long>(g.gx) * g.gy * g.gz;
    unsigned long long sum = 0;
    for (unsigned long long b = 0; b < lead; ++b) sum += readahead_block_records(g, b) * kRec;
    check(sum <= budget, (tag + ": lead blocks within the budget").c_str());
    check(lead == blocks || sum + readahead_block_records(g, lead) * kRec > budget, (tag + ": lead prefix is maximal").c_str());
    check(lead > 0U, (tag + ": a non-empty head").c_str());
}

void readahead_checks() {
    using strix::decode::ReadaheadGrid;
    auto grid = [](std::initializer_list<unsigned> tiles, std::initializer_list<unsigned> ks, std::initializer_list<unsigned> planes,
                   unsigned chunks, unsigned tpb) {
        ReadaheadGrid g{};
        unsigned j = 0;
        for (unsigned t : tiles) g.tiles[j++] = t;
        g.jobs = j;
        j = 0;
        for (unsigned k : ks) {
            g.ks[j++] = k;
            g.gy = std::max(g.gy, k);
        }
        j = 0;
        for (unsigned p : planes) {
            g.planes[j++] = p;
            g.gz = std::max(g.gz, p);
        }
        g.chunks = chunks;
        g.tiles_per_block = tpb;
        for (unsigned q = 0; q < g.jobs; ++q) g.gx += (g.tiles[q] + tpb - 1U) / tpb;
        return g;
    };
    // verify shapes (rows <= 16: 4 warps x 4 tiles = 16 tiles per block; rows > 16: 8)
    readahead_case(grid({320}, {48}, {1}, 48, 16), "out 6144->5120");
    readahead_case(grid({320}, {50}, {1}, 136, 16), "down 17408->5120");
    readahead_case(grid({1088}, {15}, {2}, 40, 16), "gate+up fused");
    readahead_case(grid({640, 384, 3, 3}, {4, 7, 40, 40}, {1, 1, 1, 1}, 40, 16), "multi in-proj (4 jobs, uneven ks)");
    readahead_case(grid({15520}, {1}, {1}, 40, 16), "lm head ks 1");
    readahead_case(grid({320}, {48}, {1}, 48, 8), "out, rows > 16");
    readahead_case(grid({7}, {3}, {1}, 10, 16), "ragged: one partial block, ks not dividing chunks");
    check(strix::decode::readahead_budget(100ULL << 20U, 128ULL << 20U) == (32ULL << 20U), "budget caps at cache / 4");
    // Consumer rows: eligible exactly for 1..stream_bound_rows (CUDA NVFP4: 24 = the last stream-bound tile, RB3).
    for (unsigned rows = 0U; rows <= 40U; ++rows) {
        const bool want = rows >= 1U && rows <= 24U;
        check(strix::decode::readahead_rows_ok(rows, 24U) == want, "rows rule: side node only for rows 1..24");
    }
    static_assert(strix::decode::readahead_rows_ok(24U, 24U) && !strix::decode::readahead_rows_ok(25U, 24U),
                  "rows rule is constexpr and inclusive at the bound");
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
    readahead_checks();
    if (failures != 0) {
        std::fprintf(stderr, "decode weight sweep: %d failures\n", failures);
        return 1;
    }
    std::printf("decode weight sweep: PASS (%u random columns x 7 geometries; read-ahead 7 grids)\n", cases);
    return 0;
}
