// Expert-major rows (core/dense_linear enqueue_expert_rows + moe_core/expert_rows_plan.hpp), host side of the invariant: a
// (row, expert) slot's projection value is the declared row-sweep dot of its weight row and its input row, so it does not depend
// on the tile it lands in or on the rows that share the launch. The test builds the CSR-by-expert tile list the device planner
// builds (assignments of an expert in ascending flat order, tiles of `cap`), runs the kernel's slot mapping (flat = row * top_k
// + slot, input row = flat / divisor, unused tile rows alias live row 0 and are not stored) with the declared row_sweep_dot, and
// requires the bits of the per-slot evaluation for every cap, for a full batch and for one-row batches; it also checks the
// planner's tile capacity bound on uniform, same-experts and rotating routes.
#include "attn_core/declared_order_reference.hpp"
#include "moe_core/expert_rows_plan.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace {
namespace er = strix::moe_core::expert_rows;
namespace dor = strix::attn_core::declared_order;
constexpr unsigned kExperts = 512U, kTopK = 10U, kIn = 64U, kOut = 5U;

#define ER_CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(1); } } while (0)

struct Plan {
    std::vector<unsigned> offsets, tile_expert, tile_begin, rows, slots;
};

Plan plan(const std::vector<std::uint32_t>& ids, unsigned rows, unsigned cap) {
    Plan p;
    p.offsets.assign(kExperts + 1U, 0U);
    for (const auto e : ids) ++p.offsets[e + 1U];
    for (unsigned e = 0; e < kExperts; ++e) p.offsets[e + 1U] += p.offsets[e];
    p.rows.resize(ids.size());
    p.slots.resize(ids.size());
    std::vector<unsigned> at(p.offsets.begin(), p.offsets.end() - 1);
    for (unsigned a = 0; a < rows * kTopK; ++a) {
        const unsigned i = at[ids[a]]++;
        p.rows[i] = a / kTopK;
        p.slots[i] = a % kTopK;
    }
    for (unsigned e = 0; e < kExperts; ++e)
        for (unsigned b = p.offsets[e]; b < p.offsets[e + 1U]; b += cap) {
            p.tile_expert.push_back(e);
            p.tile_begin.push_back(b);
        }
    return p;
}

std::vector<std::uint32_t> route(std::mt19937& rng, unsigned rows, int mode) {
    std::vector<std::uint32_t> ids(std::size_t{rows} * kTopK);
    for (unsigned r = 0; r < rows; ++r) {
        std::vector<std::uint32_t> all(kExperts);
        std::iota(all.begin(), all.end(), 0U);
        std::shuffle(all.begin(), all.end(), rng);
        std::vector<std::uint32_t> pick(all.begin(), all.begin() + kTopK);
        if (mode == 1) std::iota(pick.begin(), pick.end(), 7U);  // every row the same 10 experts
        if (mode == 2)                                           // rotating experts
            for (unsigned k = 0; k < kTopK; ++k) pick[k] = (r * 3U + k) % kExperts;
        std::sort(pick.begin(), pick.end());
        std::copy(pick.begin(), pick.end(), ids.begin() + std::ptrdiff_t{r} * kTopK);
    }
    return ids;
}

float bf16(float v) {
    std::uint32_t b;
    std::memcpy(&b, &v, 4);
    b &= 0xffff0000U;
    std::memcpy(&v, &b, 4);
    return v;
}
}  // namespace

int main() {
    std::mt19937 rng(11U);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> weights(std::size_t{kExperts} * kOut * kIn);
    for (auto& w : weights) w = dist(rng);
    for (const unsigned rows : {1U, 5U, 16U, 37U, 300U}) {
        for (int mode = 0; mode < 3; ++mode) {
            const auto ids = route(rng, rows, mode);
            std::vector<float> x(std::size_t{rows} * kIn);
            for (auto& v : x) v = bf16(dist(rng));
            const auto value = [&](unsigned expert, unsigned row, unsigned o) {
                return dor::row_sweep_dot(weights.data() + (std::size_t{expert} * kOut + o) * kIn, x.data() + std::size_t{row} * kIn, kIn);
            };
            for (const unsigned cap : {4U, 8U, 16U}) {
                const Plan p = plan(ids, rows, cap);
                ER_CHECK(p.tile_expert.size() <= er::tile_capacity(rows, kTopK, kExperts, 4U),
                         "tile capacity %zu > bound %u (rows %u mode %d cap %u)", p.tile_expert.size(),
                         er::tile_capacity(rows, kTopK, kExperts, 4U), rows, mode, cap);
                std::vector<float> got(std::size_t{rows} * kTopK * kOut, -1.0F);
                std::vector<unsigned> stored(std::size_t{rows} * kTopK, 0U);
                for (std::size_t t = 0; t < p.tile_expert.size(); ++t) {
                    const unsigned e = p.tile_expert[t], begin = p.tile_begin[t];
                    const unsigned count = std::min(cap, p.offsets[e + 1U] - begin);
                    for (unsigned r = 0; r < cap; ++r) {  // unused rows alias row 0 and are not stored
                        const unsigned a = begin + (r < count ? r : 0U), flat = p.rows[a] * kTopK + p.slots[a];
                        for (unsigned o = 0; o < kOut; ++o) {
                            const float v = value(e, flat / kTopK, o);
                            if (r < count) got[std::size_t{flat} * kOut + o] = v;
                        }
                        if (r < count) ++stored[flat];
                    }
                }
                for (unsigned f = 0; f < rows * kTopK; ++f) {
                    ER_CHECK(stored[f] == 1U, "slot %u stored %u times (rows %u mode %d cap %u)", f, stored[f], rows, mode, cap);
                    for (unsigned o = 0; o < kOut; ++o) {
                        const float want = value(ids[f], f / kTopK, o);
                        ER_CHECK(std::memcmp(&want, &got[std::size_t{f} * kOut + o], 4) == 0, "slot %u output %u differs", f, o);
                    }
                }
                // Batch invariance: a row planned alone (a one-row batch) stores the bits it has in the full batch.
                for (unsigned r = 0; r < std::min(rows, 3U); ++r) {
                    const std::vector<std::uint32_t> one(ids.begin() + std::ptrdiff_t{r} * kTopK, ids.begin() + std::ptrdiff_t{r + 1U} * kTopK);
                    const Plan q = plan(one, 1U, cap);
                    for (std::size_t t = 0; t < q.tile_expert.size(); ++t) {
                        const unsigned e = q.tile_expert[t], a = q.tile_begin[t], slot = q.slots[a];
                        for (unsigned o = 0; o < kOut; ++o) {
                            const float v = value(e, r, o);
                            ER_CHECK(std::memcmp(&v, &got[std::size_t{r * kTopK + slot} * kOut + o], 4) == 0, "single row %u differs", r);
                        }
                    }
                }
            }
        }
    }
    ER_CHECK(er::tile_capacity(2048U, kTopK, kExperts, 4U) == 512U + 20479U / 4U, "capacity 2048 rows");
    ER_CHECK(er::tile_capacity(1U, kTopK, kExperts, 4U) == 10U, "capacity 1 row (10 assignments)");
    std::printf("expert rows plan tests PASS\n");
    return 0;
}
