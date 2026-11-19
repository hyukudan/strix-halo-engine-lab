// Host tests of strix/pair/column_split.hpp (exact tensor-parallel decode shares):
//   - split_units: every unit owned by exactly one rank, contiguous in rank order, each share within one unit of its
//     exact quota, equal rates -> equal shares, deterministic;
//   - joined shares == one rank: a CPU model of the IU4 decode column (K-major tile pack + scales after the tiles, as
//     core/dense_linear_iu4.hpp addresses it with the FULL N) computed per rank share and joined equals the full rows
//     word for word, for N ranks = 1..4 and uneven rates;
//   - estimate: one rank alone == group of one; the group wins only when the link term is below the saved compute.
#include "strix/pair/column_split.hpp"

#include <algorithm>
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
        std::printf("FAIL %s\n", what);
    }
}
namespace tp = strix::pair::tp;

// One output column of the IU4 g64 pack, ascending k in float (the pack addressing of the core's reference form).
float iu4_column(const std::vector<std::uint8_t>& w, unsigned n_total, unsigned k_total, const std::vector<float>& x, unsigned n) {
    const std::size_t packed = std::size_t(n_total) * k_total / 2U;
    float acc = 0.0F;
    for (unsigned k = 0; k < k_total; ++k) {
        const std::size_t at = ((std::size_t(k / 16U) * (n_total / 16U) + n / 16U) * 16U + n % 16U) * 8U + (k % 16U) / 2U;
        const unsigned nib = (unsigned(w[at]) >> ((k & 1U) * 4U)) & 15U;
        const float q = static_cast<float>(nib < 8U ? int(nib) : int(nib) - 16);
        std::uint16_t raw = 0;
        std::memcpy(&raw, &w[packed + (std::size_t(n) * (k_total / 64U) + k / 64U) * 2U], 2U);
        const float s = static_cast<float>(raw & 0x3FFU) / 1024.0F + 1.0F;  // any fixed decode of the scale word
        acc += q * s * x[k];
    }
    return acc;
}

void test_split() {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> rate(0.3, 3.0);
    for (unsigned ranks = 1; ranks <= 4; ++ranks) {
        for (const std::uint32_t units : {0U, 1U, 3U, 4U, 16U, 80U, 272U, 3880U}) {
            for (int trial = 0; trial < 50; ++trial) {
                std::vector<double> r(ranks);
                double total = 0.0;
                for (auto& v : r) total += (v = trial == 0 ? 1.0 : rate(rng));
                const auto s = tp::split_units(units, r);
                bool ok = s.size() == ranks && s.front().begin == 0U && s.back().end == units;
                for (std::size_t i = 0; i + 1 < s.size(); ++i) ok = ok && s[i].end == s[i + 1].begin;
                for (std::size_t i = 0; i < s.size(); ++i) {
                    ok = ok && s[i].begin <= s[i].end;
                    ok = ok && std::fabs(double(s[i].size()) - double(units) * r[i] / total) < 1.0 + 1e-9;
                }
                check(ok, "split_units covers [0, units) once, contiguous, within one unit of the quota");
                const auto again = tp::split_units(units, r);
                bool same = true;
                for (std::size_t i = 0; i < s.size(); ++i) same = same && s[i].begin == again[i].begin && s[i].end == again[i].end;
                check(same, "split_units deterministic");
                if (trial == 0 && units % ranks == 0U)
                    for (const auto& x : s) check(x.size() == units / ranks, "equal rates -> equal shares");
            }
        }
    }
    bool threw = false;
    try {
        const double bad[2] = {1.0, 0.0};
        (void)tp::split_units(8U, bad);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a zero rate is refused");
}

void test_join_bitwise() {
    std::mt19937_64 rng(11);
    const unsigned n_total = 512U, k_total = 256U;
    std::vector<std::uint8_t> w(std::size_t(n_total) * k_total / 2U + std::size_t(n_total) * (k_total / 64U) * 2U);
    for (auto& b : w) b = std::uint8_t(rng());
    std::vector<float> x(k_total);
    std::normal_distribution<float> xd(0.0F, 1.0F);
    for (auto& v : x) v = xd(rng);
    std::vector<float> full(n_total);
    for (unsigned n = 0; n < n_total; ++n) full[n] = iu4_column(w, n_total, k_total, x, n);
    for (unsigned ranks = 1; ranks <= 4; ++ranks) {
        std::vector<double> r(ranks);
        for (unsigned i = 0; i < ranks; ++i) r[i] = 1.0 + 0.37 * i;
        const auto s = tp::split_units(n_total / 64U, r);
        std::vector<float> joined(n_total, -1.0F);
        for (const auto& share : s) {  // each rank: only its columns, full-N addressing, stored in place
            const auto c = tp::scaled(share, 64U);
            for (unsigned n = c.begin; n < c.end; ++n) joined[n] = iu4_column(w, n_total, k_total, x, n);
        }
        check(std::memcmp(full.data(), joined.data(), full.size() * sizeof(float)) == 0, "joined column shares == one rank, bitwise");
    }
}

void test_estimate() {
    tp::StepModel m;
    m.split_bytes = 13.9e9;
    m.replicated_bytes = 0.05e9;
    m.fixed_ms = 1.3;
    for (int l = 0; l < 64; ++l)
        for (const double b : {12288.0, 10240.0, 34816.0, 10240.0}) m.trips.push_back({b});
    m.trips.push_back({8.0});
    const double one[1] = {243e6 * 0.78}, one_f[1] = {1.0};
    const auto e1 = tp::estimate(m, one, one_f, {0.031, 4.6e6});
    check(!e1.group_wins && e1.group_ms == e1.one_rank_ms, "one rank: group == alone");
    const double two[2] = {243e6 * 0.78, 243e6 * 0.78};
    const auto s = tp::split_units(80U, two);
    const auto f = tp::fractions(s);
    const auto fast = tp::estimate(m, two, f, {0.031, 4.6e6});
    check(fast.group_wins && fast.group_ms < 0.75 * fast.one_rank_ms, "two ranks at a 31 us trip: group wins");
    const auto slow = tp::estimate(m, two, f, {0.200, 4.6e6});
    check(!slow.group_wins, "two ranks at a 200 us trip: one rank wins");
    std::printf("estimate: one %.1f ms, two ranks %.1f ms (compute %.1f + link %.1f) at 31 us/trip, %.1f ms at 200 us/trip\n",
                fast.one_rank_ms, fast.group_ms, fast.compute_ms, fast.link_ms, slow.group_ms);
}

// 3 and 4 synthetic ranks of unequal speed (host only: two nodes exist and they measure the same rate).
// Shares follow the rates (each within one unit of its exact quota), so the rank times are balanced to within one unit
// of the slowest rank and never worse than an equal split; choose_ranks keeps the fastest ranks and drops a rank whose
// replicated work costs more than its share saves.
void test_unequal_ranks() {
    const std::vector<std::vector<double>> sets{{1.0, 0.6, 1.4}, {1.0, 0.5, 1.2, 0.8}, {0.7, 0.7, 1.3}};
    for (const auto& rel : sets) {
        std::vector<double> rates;
        double sum = 0.0, slowest = 1e30;
        for (const double x : rel) {
            rates.push_back(190e6 * x);
            sum += 190e6 * x;
            slowest = std::min(slowest, 190e6 * x);
        }
        for (const std::uint32_t units : {80U, 272U, 3880U}) {
            const auto s = tp::split_units(units, rates);
            check(s.size() == rates.size() && s.front().begin == 0U && s.back().end == units, "N-rank shares cover the units");
            double tmax = 0.0, tmin = 1e30, emax = 0.0;
            for (std::size_t r = 0; r < s.size(); ++r) {
                const double quota = units * rates[r] / sum;
                check(std::fabs(double(s[r].size()) - quota) <= 1.0, "N-rank share within one unit of its quota");
                const double t = s[r].size() / rates[r];
                tmax = std::max(tmax, t);
                tmin = std::min(tmin, t);
                emax = std::max(emax, (double(units) / double(rates.size())) / rates[r]);
            }
            check(tmax - tmin <= 2.0 / slowest + 1e-12, "N-rank times balanced within one unit of the slowest rank");
            check(tmax <= emax + 1e-12, "rate shares never slower than an equal split");
        }
    }
    tp::StepModel m;
    m.split_bytes = 13.9e9;
    m.replicated_bytes = 0.05e9;
    m.fixed_ms = 1.3;
    for (int l = 0; l < 64; ++l)
        for (const double b : {12288.0, 10240.0, 34816.0, 10240.0}) m.trips.push_back({b});
    const tp::TripCost link{0.039, 2.7e6};
    const double even3[3] = {181e6, 190e6, 200e6};
    const auto a = tp::choose_ranks(m, even3, 3880U, link);
    check(a.ranks.size() == 3U && a.ranks[0] == 2U && a.ranks[2] == 0U, "3 similar ranks: all decode, fastest first");
    check(a.est.group_wins, "3 similar ranks: the group beats the fastest alone");
    const double one_slow[4] = {190e6, 190e6, 190e6, 3e6};  // a rank 60x slower: its replicated reads cost ~17 ms
    const auto b = tp::choose_ranks(m, one_slow, 3880U, link);
    check(b.ranks.size() == 3U && std::find(b.ranks.begin(), b.ranks.end(), 3U) == b.ranks.end(), "4 ranks, one very slow: it is dropped");
    const auto c = tp::choose_ranks(m, one_slow, 3880U, tp::TripCost{0.4, 2.7e6});
    check(c.ranks.size() == 1U && c.ranks[0] == 0U, "slow link: the fastest rank decodes alone (first of equal rates)");
    std::printf("choose_ranks: 3 similar -> %zu ranks %.1f ms (one %.1f); 4 with a slow one -> %zu ranks %.1f ms; slow link -> %zu\n",
                a.ranks.size(), a.est.group_ms, a.est.one_rank_ms, b.ranks.size(), b.est.group_ms, c.ranks.size());
}
}  // namespace

int main() {
    test_split();
    test_join_bitwise();
    test_estimate();
    test_unequal_ranks();
    std::printf("pair column split tests: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
