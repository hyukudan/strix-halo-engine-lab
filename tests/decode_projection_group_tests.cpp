// Decode step plan: projection group index (include/strix/decode_projection_group.hpp).
// Invariant: for every unit u < total, locate() returns the member m with first[m] <= u < first[m + 1] and
// local = u - first[m]; the units of the group are a bijection onto {(m, local < units[m])}, in member order. It equals
// the incumbent per-site loop of src/hip_q8_rows_gemv.hpp (segment_row, segment_row_q6, the dense A8 dispatch), which
// now calls it, including units past the total (last member). The Qwen 27B dense groups (64-column blocks of the HIP
// dense_iu4 group launch) are checked by name.
#include "strix/decode_projection_group.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

// The incumbent loop (before the shared index), kept here as the reference.
void incumbent(const std::uint32_t* first, unsigned count, std::uint32_t unit, unsigned& member, std::uint32_t& local) {
    member = 0U;
    while (member + 1U < count && unit >= first[member + 1U]) ++member;
    local = unit - first[member];
}

template <unsigned Max>
void check_group(const std::vector<std::uint32_t>& units, const char* name) {
    strix::decode::GroupIndex<Max> g;
    for (std::uint32_t u : units) g.add(u);
    std::uint32_t total = 0;
    for (std::uint32_t u : units) total += u;
    check(g.total() == total && g.count == units.size(), name);
    std::vector<std::vector<unsigned>> seen(units.size());
    for (unsigned m = 0; m < units.size(); ++m) seen[m].assign(units[m], 0U);
    unsigned prev_member = 0;
    for (std::uint32_t u = 0; u < total + 3U; ++u) {
        unsigned m = 0, rm = 0;
        std::uint32_t local = 0, rlocal = 0;
        strix::decode::locate(g.first, g.count, u, m, local);
        incumbent(g.first, g.count, u, rm, rlocal);
        check(m == rm && local == rlocal, name);
        if (u >= total) {
            check(m + 1U == g.count, name);  // past the total: the last member, as the incumbent loop
            continue;
        }
        check(m < units.size() && local < units[m] && g.first[m] + local == u, name);
        check(m >= prev_member, name);  // member order
        prev_member = m;
        if (m < units.size() && local < units[m]) ++seen[m][local];
    }
    for (const auto& v : seen)
        for (unsigned c : v) check(c == 1U, name);  // bijection
}

}  // namespace

int main() {
    // Qwen 27B dense (HIP dense_iu4 group launch, units = 64-column blocks): gate+up, GDN qkv+z, attention q+k+v.
    check_group<4>({17408U / 64U, 17408U / 64U}, "gate+up");
    check_group<4>({10240U / 64U, 6144U / 64U}, "qkv+z");
    check_group<4>({12288U / 64U, 1024U / 64U, 1024U / 64U}, "q+k+v");
    check_group<4>({1U}, "single");
    // q8_rows sites (units = output rows, up to 5 members).
    check_group<5>({10240U, 6144U, 48U, 48U}, "gdn input site");
    std::mt19937 rng(20261118U);
    unsigned cases = 0;
    for (int t = 0; t < 4000; ++t) {
        const unsigned count = 1U + static_cast<unsigned>(rng() % 5U);
        std::vector<std::uint32_t> units(count);
        for (auto& u : units) u = 1U + static_cast<std::uint32_t>(rng() % 97U);
        check_group<5>(units, "random");
        ++cases;
    }
    bool threw = false;
    try {
        strix::decode::GroupIndex<4> g;
        for (int i = 0; i < 5; ++i) g.add(1U);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "fifth member of a 4-member group throws");
    threw = false;
    try {
        strix::decode::GroupIndex<4> g;
        g.add(0U);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "empty member throws");
    if (failures) {
        std::fprintf(stderr, "decode projection group: %d failures\n", failures);
        return 1;
    }
    std::printf("decode projection group: PASS (%u random groups + 5 named)\n", cases);
    return 0;
}
