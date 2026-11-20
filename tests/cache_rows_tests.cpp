// Shared attention cache-row contract (strix/attn/cache_rows.hpp): RNE encoding vs an independent nearest-value
// reference, slot rule (full cache / ring), the Window reader's ring size, ring wrap overwrite, non-finite flag, and
// the MODEL_B geometries (MLA 576-wide rows in a full cache and in the SWA ring, 128-wide token keys in the indexer ring).
#include "strix/attn/cache_rows.hpp"
#include "strix/attn/group_indexer.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace cr = strix::attn::cache_rows;

namespace {

int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

struct Rng {
    std::uint64_t s;
    std::uint32_t bits() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(s >> 32U);
    }
    float uniform() { return static_cast<float>(static_cast<double>(bits() >> 8U) / 8388608.0 - 1.0); }
};

float as_float(std::uint32_t u) {
    float f = 0.0F;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

// Independent reference: the bf16 value nearest to x (exact in double), ties to the even mantissa; overflow past
// the largest finite bf16 by at least half an ulp -> infinity (IEEE RNE).
std::uint16_t nearest_bf16(float x) {
    std::uint32_t u = 0;
    std::memcpy(&u, &x, sizeof u);
    const auto lo = static_cast<std::uint16_t>(u >> 16U);
    if ((u & 0xffffU) == 0U) return lo;
    const auto hi = static_cast<std::uint16_t>(lo + 1U);  // next magnitude (same sign); 0x7f80 / 0xff80 = inf
    const double ax = std::fabs(static_cast<double>(x));
    const double vl = std::fabs(static_cast<double>(cr::decode(lo)));
    // IEEE RNE rounds as if the exponent were unbounded: infinity stands for 2^128 (max finite has an odd mantissa)
    const double vh = (hi & 0x7f80U) == 0x7f80U ? std::ldexp(1.0, 128) : std::fabs(static_cast<double>(cr::decode(hi)));
    const double dl = ax - vl, dh = vh - ax;
    if (dl < dh) return lo;
    if (dh < dl) return hi;
    return (lo & 1U) == 0U ? lo : hi;
}

void encoding() {
    // hand cases: ties to even, carry into the exponent, signed zero, max finite, overflow to inf
    CHECK(cr::encode(1.0F) == 0x3f80U);
    CHECK(cr::encode(as_float(0x3f808000U)) == 0x3f80U);  // tie, even below
    CHECK(cr::encode(as_float(0x3f818000U)) == 0x3f82U);  // tie, even above
    CHECK(cr::encode(as_float(0x3f808001U)) == 0x3f81U);
    CHECK(cr::encode(as_float(0x3fffffffU)) == 0x4000U);  // carry
    CHECK(cr::encode(-0.0F) == 0x8000U);
    CHECK(cr::encode(as_float(0x7f7fffffU)) == 0x7f80U);  // rounds past max finite -> inf: flagged by finite()
    CHECK(!cr::finite(cr::encode(as_float(0x7f7fffffU))));
    Rng r{7};
    for (int k = 0; k < 2000000; ++k) {
        std::uint32_t u = r.bits();
        if ((u & 0x7f800000U) == 0x7f800000U) continue;  // finite inputs only (the contract's domain)
        const float x = as_float(u);
        if (cr::encode(x) != nearest_bf16(x)) {
            std::fprintf(stderr, "FAIL encode 0x%08x -> 0x%04x, nearest 0x%04x\n", u, cr::encode(x), nearest_bf16(x));
            ++failures;
            break;
        }
        CHECK(cr::decode(cr::encode(cr::decode(cr::encode(x)))) == cr::decode(cr::encode(x)));  // idempotent
    }
    CHECK(cr::finite(1.0F) && !cr::finite(std::numeric_limits<float>::infinity()) && !cr::finite(std::nanf("")));
}

void slots_and_rings() {
    cr::Rows full{576U, 0U};  // runtime values: the checks are not constant expressions
    CHECK(cr::slot(full, 123456U) == 123456U && cr::row_offset(full, 3U) == 3U * 576U);
    // the sparse flash Window reader (sparse_flash_mla.hpp key_row: pos & ring_mask) and the writer agree for a
    // power-of-two ring
    constexpr unsigned ring = cr::window_ring(512U, 1U);
    static_assert(ring == 512U && cr::is_pow2(ring));
    static_assert(cr::window_ring(512U, 64U) == 1024U && cr::window_ring(512U, 2U) == 1024U && cr::window_ring(1U, 1U) == 1U);
    cr::Rows swa{576U, ring};
    for (std::uint64_t p = 0; p < 5000U; p += 7U) CHECK(cr::slot(swa, p) == (p & (ring - 1U)));
    // a window of `w` keys plus a chunk of c rows never shares a slot inside one launch
    for (unsigned c : {1U, 3U, 64U, 513U}) {
        const unsigned rr = cr::window_ring(512U, c);
        CHECK(rr >= 512U - 1U + c);
    }
    // indexer token-key ring (not a power of two): slot = position % ring
    static_assert(strix::attn::group_indexer::token_ring_slots(
        strix::attn::group_indexer::Shape{4U, 16U, 128U, 512U, strix::attn::group_indexer::Pooling::GatedSoftmaxApe,
                                          strix::attn::group_indexer::OwnGroup::AlwaysOwn,
                                          strix::attn::group_indexer::ScoreScale::InHeadWeight},
        4U) == 7U);
    cr::Rows keys{128U, 7U};
    CHECK(cr::slot(keys, 20U) == 6U && cr::row_offset(keys, 21U) == 0U);
}

void store_reference() {
    // SWA ring of 8 rows of width 5: 13 positions written one per launch; the ring holds the last 8, each at p % 8.
    constexpr cr::Rows r{5U, 8U};
    std::vector<std::uint16_t> cache(8U * 5U, 0xdeadU);
    Rng g{11};
    std::vector<float> all(13U * 6U);
    for (auto& v : all) v = g.uniform() * 4.0F;
    for (unsigned p = 0; p < 13U; ++p) CHECK(cr::store_rows_ref(all.data() + p * 6U, 6U, nullptr, p, 1U, r, cache.data()));
    for (unsigned p = 5; p < 13U; ++p)
        for (unsigned i = 0; i < 5U; ++i) CHECK(cache[(p % 8U) * 5U + i] == cr::encode(all[p * 6U + i]));
    // explicit positions (a verify window), full cache; one non-finite element flags the call, bits still stored
    constexpr cr::Rows f{4U, 0U};
    std::vector<std::uint16_t> full(32U * 4U, 0U);
    const std::uint32_t pos[3] = {9U, 30U, 2U};
    std::vector<float> src(3U * 4U, 0.5F);
    src[5] = std::numeric_limits<float>::infinity();
    CHECK(!cr::store_rows_ref(src.data(), 4U, pos, 0U, 3U, f, full.data()));
    CHECK(full[30U * 4U + 1U] == 0x7f80U && full[9U * 4U] == cr::encode(0.5F) && full[2U * 4U + 3U] == cr::encode(0.5F));
    src[5] = 0.25F;
    CHECK(cr::store_rows_ref(src.data(), 4U, pos, 0U, 3U, f, full.data()));
    // the GLM C3 case: a value past the largest finite bf16 encodes to inf and flags
    src[0] = as_float(0x7f7fffffU);
    CHECK(!cr::store_rows_ref(src.data(), 4U, pos, 0U, 1U, f, full.data()));
}

}  // namespace

int main() {
    encoding();
    slots_and_rings();
    store_reference();
    if (failures != 0) {
        std::fprintf(stderr, "cache_rows: %d failures\n", failures);
        return 1;
    }
    std::printf("cache_rows: PASS\n");
    return 0;
}
