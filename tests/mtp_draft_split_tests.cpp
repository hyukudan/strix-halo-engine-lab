// strix::mtp::DraftSplit (include/strix/mtp_draft_split.hpp): first-segment choice of a split draft chain.
#include "strix/mtp_draft_split.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}
}  // namespace

int main() {
    using strix::mtp::DraftSplit;
    {
        DraftSplit off(0);
        check(off.first_segment(7) == 7, "mode 0 keeps the whole chain");
        DraftSplit fixed(3);
        check(fixed.first_segment(7) == 3 && fixed.first_segment(2) == 2, "fixed N, capped by depth");
        DraftSplit any;
        check(any.first_segment(1) == 1 && any.first_segment(0) == 0, "depth < 2 is never split");
        check(any.first_segment(7) == 2, "auto without samples starts at 2");
    }
    {
        // prose-like: 54 % keep 1, 26 % keep 2, 14 % keep 3, 6 % keep 7 -> a = 2 (expected steps 2 + .2 * 5.25 = 3.05)
        DraftSplit s;
        for (int i = 0; i < 54; ++i) s.note(1);
        for (int i = 0; i < 26; ++i) s.note(2);
        for (int i = 0; i < 14; ++i) s.note(3);
        for (int i = 0; i < 6; ++i) s.note(7);
        check(s.first_segment(7) == 2, "prose-like histogram picks 2");
    }
    {
        // every chain keeps the full depth: splitting only adds a host turn -> no split
        DraftSplit s;
        for (int i = 0; i < 20; ++i) s.note(7);
        check(s.first_segment(7) == 7, "full-depth chains are not split");
        check(s.kept(7) == 20 && s.kept(99) == 0, "histogram counts");
    }
    {
        // kept counts beyond the bins land in the last bin
        DraftSplit s;
        s.note(1000);
        check(s.kept(DraftSplit::kBins - 1) == 1, "overflow bin");
    }
    {
        // receipt tally: the three-argument note feeds the same histogram (same split choice) and counts the step work
        DraftSplit a, b;
        const std::uint32_t kept[6] = {1, 2, 7, 3, 1, 7};
        for (int r = 0; r < 4; ++r)
            for (const auto k : kept) {
                const std::uint32_t first = b.first_segment(7);
                check(a.first_segment(7) == first, "tally note keeps the split choice");
                const bool second = first < 7 && k > first;
                a.note(k);
                b.note(k, second ? 7U : first, second ? 1U : 0U);
            }
        const auto& t = b.tally();
        check(t.chains == 24 && t.kept == 4 * (1 + 2 + 7 + 3 + 1 + 7), "tally chains / kept");
        check(t.launched >= t.kept && t.turns <= t.chains, "tally launched >= kept");
        for (std::uint32_t n = 0; n < DraftSplit::kBins; ++n) check(a.kept(n) == b.kept(n), "tally histogram equal");
    }
    {
        // segment ends: cap 2 == the single split; the DP equals a brute force over every boundary set; mode 0 / fixed
        std::uint64_t seed = 12345;
        const auto rnd = [&]() { seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<std::uint32_t>(seed >> 33); };
        for (int trial = 0; trial < 400; ++trial) {
            const std::uint32_t depth = 2 + rnd() % 7;  // 2..8
            DraftSplit one(-1, 0.25, 8, 2), any(-1, 0.25, 8, 0);
            const std::uint32_t chains = 8 + rnd() % 60;
            for (std::uint32_t c = 0; c < chains; ++c) {
                const std::uint32_t k = 1 + rnd() % depth;
                one.note(k);
                any.note(k);
            }
            const auto e2 = one.segment_ends(depth);
            const std::uint32_t a = one.first_segment(depth);
            check(e2.back() == depth && (a == depth ? e2.size() == 1 : (e2.size() == 2 && e2[0] == a)), "cap 2 = single split");
            // expected steps of a boundary set (same model as the header)
            double total = 0;
            for (std::uint32_t n = 0; n <= depth; ++n) total += static_cast<double>(any.kept(n));
            const auto alive = [&](std::uint32_t s) {
                if (s == 0) return 1.0;
                double more = 0;
                for (std::uint32_t n = s + 1; n <= depth; ++n) more += static_cast<double>(any.kept(n));
                return more / total;
            };
            const auto cost = [&](const std::vector<std::uint32_t>& ends) {
                double c = 0;
                std::uint32_t s = 0;
                for (const auto e : ends) {
                    c += alive(s) * (e - s + (s > 0 ? 0.25 : 0.0));
                    s = e;
                }
                return c;
            };
            double brute = 1e300;
            for (std::uint32_t mask = 0; mask < (1U << (depth - 1)); ++mask) {
                std::vector<std::uint32_t> ends;
                for (std::uint32_t b = 1; b < depth; ++b)
                    if (mask >> (b - 1) & 1U) ends.push_back(b);
                ends.push_back(depth);
                brute = std::min(brute, cost(ends));
            }
            const auto ed = any.segment_ends(depth);
            bool ascending = ed.back() == depth;
            for (std::size_t i = 1; i < ed.size(); ++i) ascending = ascending && ed[i - 1] < ed[i];
            check(ascending, "ends ascending, last == depth");
            check(std::fabs(cost(ed) - brute) < 1e-9, "DP ends reach the brute-force minimum");
            check(cost(ed) <= cost(e2) + 1e-12, "more segments never cost more");
        }
        DraftSplit off(0), fixed(3);
        check(off.segment_ends(7) == std::vector<std::uint32_t>{7}, "mode 0: one segment");
        check(fixed.segment_ends(7) == (std::vector<std::uint32_t>{3, 7}), "fixed first segment, then the rest");
        DraftSplit fresh;
        check(fresh.segment_ends(7) == (std::vector<std::uint32_t>{2, 7}), "too few samples: first segment 2");
        // the measured prose histogram (RTX receipt: kept 1..7 = 46 38 28 15 7 3 3, depth 7)
        DraftSplit prose;
        const std::uint32_t h[8] = {0, 46, 38, 28, 15, 7, 3, 3};
        for (std::uint32_t n = 1; n <= 7; ++n)
            for (std::uint32_t c = 0; c < h[n]; ++c) prose.note(n);
        const auto ep = prose.segment_ends(7);
        std::printf("prose histogram: first segment %u, segment ends", prose.first_segment(7));
        for (const auto e : ep) std::printf(" %u", e);
        std::printf("\n");
        check(ep.size() > 2, "prose: more than two segments pay");
        DraftSplit full;
        for (int c = 0; c < 50; ++c) full.note(7);
        check(full.segment_ends(7) == std::vector<std::uint32_t>{7}, "full-depth chains: one segment");
        // contexts: prose-like chains in context 0, full chains in context 1 -> each keeps its own ends
        DraftSplit mixed;
        for (std::uint32_t n = 1; n <= 7; ++n)
            for (std::uint32_t c = 0; c < h[n]; ++c) mixed.note(n, 7, 0, 0);
        for (int c = 0; c < 50; ++c) mixed.note(7, 7, 0, 1);
        check(mixed.segment_ends(7, 0) == ep, "context 0 = the prose ends");
        check(mixed.segment_ends(7, 1) == std::vector<std::uint32_t>{7}, "context 1 = one segment");
        check(mixed.kept(7) == 53 && mixed.kept(7, 1) == 50, "receipt histogram sums the contexts");
    }
    std::printf("%s mtp_draft_split_tests\n", failures ? "FAIL" : "PASS");
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
