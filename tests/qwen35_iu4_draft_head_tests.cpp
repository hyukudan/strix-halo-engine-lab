// Draft-only subset of an IU4 g64 head (include/strix/qwen35/iu4_draft_head.hpp): the gather plan moves whole tile columns,
// so every compact column holds exactly its source column's nibbles and scales; incremental and re-layout gathers agree;
// the compact argmax mapped through the slot table equals the full-head argmax whenever that token is a subset row.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "strix/qwen35/iu4_draft_head.hpp"
#include "strix/spec/draft_rows.hpp"

namespace {
int failures = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                       \
        }                                                                     \
    } while (0)

using strix::qwen35::Iu4Copy;

// Layout of the shared pack: tiles [K/16][N/16][16 columns][8 bytes] (low nibble = even k), then FP16 scales [N][K/64].
int nibble(const std::vector<std::uint8_t>& h, std::uint32_t N, std::uint32_t K, std::uint32_t n, std::uint32_t k) {
    (void)K;
    const std::size_t at = (std::size_t(k / 16U) * (N / 16U) + n / 16U) * 128U + (n % 16U) * 8U + (k % 16U) / 2U;
    const int v = (k % 2U == 0U) ? (h[at] & 0xF) : (h[at] >> 4);
    return v >= 8 ? v - 16 : v;
}
std::uint16_t scale(const std::vector<std::uint8_t>& h, std::uint32_t N, std::uint32_t K, std::uint32_t n, std::uint32_t k) {
    std::uint16_t s;
    std::memcpy(&s, h.data() + std::size_t(N) * K / 2U + (std::size_t(n) * (K / 64U) + k / 64U) * 2U, 2);
    return s;
}
void apply(const std::vector<Iu4Copy>& plan, const std::vector<std::uint8_t>& src, std::vector<std::uint8_t>& dst) {
    for (const Iu4Copy& c : plan)
        for (std::size_t r = 0; r < c.height; ++r) {
            const std::size_t so = c.src + r * c.spitch, d = c.dst + r * c.dpitch;
            if (so + c.width > src.size() || d + c.width > dst.size()) throw std::out_of_range("copy past a buffer");
            std::memcpy(dst.data() + d, src.data() + so, c.width);
        }
}
// Host logit of column n (deterministic decode of the raw scale bits: the test is about byte identity, not FP16).
double logit(const std::vector<std::uint8_t>& h, std::uint32_t N, std::uint32_t K, std::uint32_t n, const std::vector<double>& x) {
    double acc = 0.0;
    for (std::uint32_t k = 0; k < K; ++k) acc += nibble(h, N, K, n, k) * (double(scale(h, N, K, n, k) & 0x3FFU) / 1024.0 + 0.5) * x[k];
    return acc;
}
bool same_column(const std::vector<std::uint8_t>& a, std::uint32_t na, std::uint32_t ca, const std::vector<std::uint8_t>& b,
                 std::uint32_t nb, std::uint32_t cb, std::uint32_t K) {
    for (std::uint32_t k = 0; k < K; ++k)
        if (nibble(a, na, K, ca, k) != nibble(b, nb, K, cb, k) || scale(a, na, K, ca, k) != scale(b, nb, K, cb, k)) return false;
    return true;
}
}  // namespace

int main() {
    constexpr std::uint32_t K = 128U, N = 2048U;  // 128 groups of 16 ids
    std::mt19937 rng(7);
    std::vector<std::uint8_t> head(strix::qwen35::iu4_head_bytes(N, K));
    for (auto& b : head) b = std::uint8_t(rng());

    // Column count: rows + headroom rounded to 64, capped at the head; source must be a multiple of 64.
    CHECK(strix::qwen35::iu4_draft_columns(96U, N) == 1152U);
    CHECK(strix::qwen35::iu4_draft_columns(96U, 1024U) == 1024U);
    CHECK(strix::qwen35::iu4_draft_columns(5000U, 248320U) % 64U == 0U);
    bool threw = false;
    try { (void)strix::qwen35::iu4_draft_columns(16U, 1000U); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    // Slots: a run (0..2), singles, a second run (9, 10), the last group.
    const std::vector<std::uint32_t> groups{0U, 1U, 2U, 7U, 9U, 10U, 127U};
    const std::uint32_t slots = std::uint32_t(groups.size());
    const std::uint32_t C = strix::qwen35::iu4_draft_columns(slots * 16U, N);
    std::vector<std::uint8_t> one(strix::qwen35::iu4_head_bytes(C, K), 0), inc(one.size(), 0);
    const auto plan = strix::qwen35::iu4_group_gather(K, N, C, groups, 0U, slots);
    CHECK(plan.size() == 2U * 4U);  // runs {0,1,2} {7} {9,10} {127}: tiles + scales each
    apply(plan, head, one);
    bool exact = true;
    for (std::uint32_t s = 0; s < slots; ++s)
        for (std::uint32_t j = 0; j < 16U; ++j) exact = exact && same_column(one, C, s * 16U + j, head, N, groups[s] * 16U + j, K);
    CHECK(exact);

    // Incremental gathers (as context admission does) give the same bytes as one gather.
    apply(strix::qwen35::iu4_group_gather(K, N, C, groups, 0U, 4U), head, inc);
    apply(strix::qwen35::iu4_group_gather(K, N, C, groups, 4U, slots), head, inc);
    bool same_used = true;
    for (std::uint32_t c = 0; c < slots * 16U; ++c) same_used = same_used && same_column(one, C, c, inc, C, c, K);
    CHECK(same_used);

    // Re-layout at another column count (the subset outgrew the head): per-column content unchanged.
    const std::uint32_t C2 = C + 64U;
    std::vector<std::uint8_t> big(strix::qwen35::iu4_head_bytes(C2, K), 0);
    apply(strix::qwen35::iu4_group_gather(K, N, C2, groups, 0U, slots), head, big);
    bool relayout = true;
    for (std::uint32_t c = 0; c < slots * 16U; ++c) relayout = relayout && same_column(one, C, c, big, C2, c, K);
    CHECK(relayout);

    // Logits: compact row r == full row map(r); compact argmax mapped == full argmax when the full argmax is a subset row.
    std::vector<double> x(K);
    std::normal_distribution<double> nd;
    int agreed = 0, inside = 0;
    for (int trial = 0; trial < 20; ++trial) {
        for (auto& v : x) v = nd(rng);
        std::uint32_t best_full = 0, best_sub = 0;
        double vf = -1e300, vs = -1e300;
        for (std::uint32_t n = 0; n < N; ++n) {
            const double l = logit(head, N, K, n, x);
            if (l > vf) { vf = l; best_full = n; }
        }
        bool rows_equal = true;
        for (std::uint32_t r = 0; r < slots * 16U; ++r) {
            const double l = logit(one, C, K, r, x);
            rows_equal = rows_equal && l == logit(head, N, K, groups[r / 16U] * 16U + r % 16U, x);
            if (l > vs) { vs = l; best_sub = r; }
        }
        CHECK(rows_equal);
        {  // a subset that holds the full argmax's group (placed second, after an unrelated group): mapped argmax == full
            const std::uint32_t gf = best_full / 16U;
            const std::vector<std::uint32_t> g2{(gf + 64U) % 128U, gf};
            const std::uint32_t C4 = strix::qwen35::iu4_draft_columns(32U, N);
            std::vector<std::uint8_t> s2(strix::qwen35::iu4_head_bytes(C4, K), 0);
            apply(strix::qwen35::iu4_group_gather(K, N, C4, g2, 0U, 2U), head, s2);
            std::uint32_t b2 = 0;
            double v2 = -1e300;
            for (std::uint32_t r = 0; r < 32U; ++r) {
                const double l = logit(s2, C4, K, r, x);
                if (l > v2) { v2 = l; b2 = r; }
            }
            ++inside;
            agreed += g2[b2 / 16U] * 16U + b2 % 16U == best_full;
        }
        const std::uint32_t mapped = groups[best_sub / 16U] * 16U + best_sub % 16U;
        bool member = false;
        for (std::uint32_t g : groups) member = member || g == best_full / 16U;
        if (member) {
            ++inside;
            agreed += mapped == best_full;
        }
    }
    CHECK(agreed == inside);

    // The shared DraftRows slot table drives the plan unchanged (group 16 = one tile column).
    strix::spec::DraftRowsConfig dc;
    dc.vocab = N;
    dc.group = 16U;
    dc.core_groups = 4U;
    dc.fallback_groups = 0U;
    strix::spec::DraftRows rows(dc);
    const std::vector<std::uint32_t> prompt{100U, 101U, 900U, 2047U};
    rows.begin(prompt);
    const std::uint32_t C3 = strix::qwen35::iu4_draft_columns(rows.rows(), N);
    std::vector<std::uint8_t> sub(strix::qwen35::iu4_head_bytes(C3, K), 0);
    apply(strix::qwen35::iu4_group_gather(K, N, C3, rows.groups(), 0U, rows.slots()), head, sub);
    bool table = rows.slots() >= 4U;
    for (std::uint32_t r = 0; r < rows.rows(); ++r) table = table && same_column(sub, C3, r, head, N, rows.token(r), K);
    CHECK(table);
    for (std::uint32_t id : prompt) CHECK(rows.contains(id));

    // Guards: a group past the head, slots past the compact head, K not a multiple of 64.
    const std::vector<std::uint32_t> bad{128U};
    threw = false;
    try { (void)strix::qwen35::iu4_group_gather(K, N, C, bad, 0U, 1U); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { (void)strix::qwen35::iu4_group_gather(K, N, 16U, groups, 0U, 2U); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { (void)strix::qwen35::iu4_group_gather(96U, N, C, groups, 0U, 1U); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("qwen35 iu4 draft head: OK (subset argmax agreed %d/%d inside the rows)\n", agreed, inside);
    return 0;
}
