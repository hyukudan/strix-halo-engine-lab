// Draft-only head rows (strix/spec/draft_rows.hpp): core identity mapping, pack choice by prompt, context admission,
// request reset, capacity, and the invariant the CUDA subset head relies on: the argmax over the compact rows, mapped back
// to token ids, equals the full-head argmax whenever the full argmax is one of the rows (drafts only; the target verifies).
#include "strix/spec/draft_rows.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <vector>

namespace sp = strix::spec;

namespace {
#define DR_CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(1); } } while (0)

struct Rng {
    std::uint64_t s;
    std::uint32_t next(std::uint32_t n) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return std::uint32_t((s >> 33) % n); }
};

void check_map(const sp::DraftRows& r) {
    std::set<std::uint32_t> seen;
    for (std::uint32_t row = 0; row < r.rows(); ++row) {
        const auto id = r.token(row);
        DR_CHECK(r.contains(id), "row %u -> id %u not contained", row, id);
        DR_CHECK(seen.insert(id).second, "id %u mapped twice", id);
    }
}

void layout_tests() {
    sp::DraftRowsConfig c;
    c.vocab = 4096;
    c.group = 16;
    c.core_groups = 32;  // ids < 512
    c.fallback_groups = 40;  // ids < 640 when no pack joins
    c.always = {4095};
    c.packs = {{3000, 3001, 3017, 3100, 3200, 3300}};
    c.pack_min_hits = 2;
    sp::DraftRows r(c);
    for (std::uint32_t row = 0; row < 512; ++row) DR_CHECK(r.token(row) == row, "core row %u maps to %u", row, r.token(row));
    DR_CHECK(r.contains(4095) && r.contains(4080) && !r.contains(3000), "always group / no pack before a prompt");
    DR_CHECK(r.contains(639) && !r.contains(640), "fallback groups without a pack");
    DR_CHECK(r.materialised() == 0 && r.slots() == 41, "fresh: %u slots, %u materialised", r.slots(), r.materialised());
    r.mark_materialised();
    // One pack id in the prompt: below pack_min_hits, only its own group (prompt context) joins.
    r.begin(std::vector<std::uint32_t>{5, 3001, 700});
    DR_CHECK(r.packs_chosen() == 0 && r.contains(3001) && r.contains(700) && !r.contains(3300), "pack below its hit count");
    DR_CHECK(r.materialised() == 32, "begin keeps the core materialised (%u)", r.materialised());
    check_map(r);
    // Two pack ids outside the core: the pack joins.
    r.begin(std::vector<std::uint32_t>{3000, 3200});
    DR_CHECK(r.packs_chosen() == 1 && r.contains(3300) && r.contains(3100) && !r.contains(700), "pack by prompt + reset of context");
    DR_CHECK(!r.contains(600), "a joined pack replaces the fallback groups");
    const auto before = r.slots();
    DR_CHECK(r.admit(std::vector<std::uint32_t>{700, 701, 3300, 9999}) == 1 && r.slots() == before + 1, "context admits one new group");
    check_map(r);
    // Capacity: dropped, never past it.
    sp::DraftRowsConfig small = c;
    small.capacity_groups = 34;
    small.fallback_groups = 0;
    sp::DraftRows s(small);
    s.begin(std::vector<std::uint32_t>{600, 900, 1500, 2500});
    DR_CHECK(s.slots() == 34 && s.dropped() == 3, "capacity: %u slots, %llu dropped", s.slots(), static_cast<unsigned long long>(s.dropped()));
    bool threw = false;
    try { (void)s.token(s.rows()); } catch (const std::out_of_range&) { threw = true; }
    DR_CHECK(threw, "token past rows must throw");
    threw = false;
    try { sp::DraftRowsConfig bad = c; bad.vocab = 4100; sp::DraftRows b(bad); } catch (const std::invalid_argument&) { threw = true; }
    DR_CHECK(threw, "vocab not a multiple of the group must throw");
}

// Subset argmax (lowest compact row on ties, as the device argmax) mapped back == full argmax when the full argmax is a row.
void argmax_tests() {
    for (const std::uint32_t group : {1U, 16U}) {
        sp::DraftRowsConfig c;
        c.vocab = 16 * 512;
        c.group = group;
        c.core_groups = 2048 / group;
        c.fallback_groups = 0;
        sp::DraftRows r(c);
        Rng rng{group};
        std::vector<std::uint32_t> prompt;
        for (int i = 0; i < 40; ++i) prompt.push_back(rng.next(c.vocab));
        r.begin(prompt);
        std::vector<float> logits(c.vocab), compact;
        std::uint32_t inside = 0, agree = 0;
        for (int trial = 0; trial < 2000; ++trial) {
            for (auto& l : logits) l = float(rng.next(100000)) / 1000.0f;
            std::uint32_t full = 0;
            for (std::uint32_t i = 1; i < c.vocab; ++i) if (logits[i] > logits[full]) full = i;
            compact.assign(r.rows(), 0.0f);
            for (std::uint32_t row = 0; row < r.rows(); ++row) compact[row] = logits[r.token(row)];
            std::uint32_t best = 0;
            for (std::uint32_t row = 1; row < r.rows(); ++row) if (compact[row] > compact[best]) best = row;
            const auto drafted = r.token(best);
            DR_CHECK(r.contains(drafted), "drafted id outside the rows");
            if (r.contains(full)) {
                ++inside;
                // Ties: both pick a maximal logit; equal ids unless another id has the same maximal value.
                DR_CHECK(logits[drafted] == logits[full], "subset argmax %u (%.3f) != full %u (%.3f)", drafted, logits[drafted], full,
                         logits[full]);
                agree += drafted == full;
            }
            r.admit(std::vector<std::uint32_t>{full});  // the emitted token becomes a context row
        }
        std::printf("draft rows group %u: %u rows of %u, full argmax inside %u of 2000, same id %u\n", group, r.rows(), c.vocab, inside, agree);
        DR_CHECK(inside > 0 && agree + 20 >= inside, "argmax agreement %u of %u", agree, inside);
    }
}
} // namespace

int main() {
    layout_tests();
    argmax_tests();
    std::printf("PASS strix-spec-draft-rows-tests\n");
    return 0;
}
