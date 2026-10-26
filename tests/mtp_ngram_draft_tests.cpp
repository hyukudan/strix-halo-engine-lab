#include "strix/mtp_ngram_draft.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what.c_str()); }
}
using V = std::vector<std::uint32_t>;
}  // namespace

int main() {
    using strix::MtpNgramDraft;
    using strix::MtpNgramDraftConfig;
    {
        // Lookup: longest suffix, most recent occurrence, continuation through the extra tokens.
        MtpNgramDraft d;
        d.reset(V{1, 2, 3, 4, 5, 9, 2, 3, 7, 8, 6});
        std::uint32_t suffix = 0, support = 0;
        auto out = d.lookup(V{2, 3}, 4, suffix, support);
        check(out == V{7, 8, 6, 2} && suffix == 2 && support == 1, "most recent occurrence of a 2-gram");
        out = d.lookup(V{9, 2, 3}, 2, suffix, support);
        check(out == V{7, 8} && suffix == 3, "longer suffix preferred");
        out = d.lookup(V{4, 5}, 3, suffix, support);
        check(out == V{9, 2, 3} && suffix == 2, "continuation of an older occurrence");
        out = d.lookup(V{42, 43}, 3, suffix, support);
        check(out.empty() && suffix == 0, "no match");
        // Support: two earlier occurrences followed by the same token.
        MtpNgramDraft s;
        s.reset(V{5, 6, 1, 5, 6, 1, 0, 5, 6});
        out = s.lookup({}, 2, suffix, support);
        check(out == V{1, 0} && support == 2, "support counts agreeing earlier occurrences");
    }
    {
        // Gate: with the default prior (0.2) the extension is withheld until shadow scoring has seen it
        // right; a request that copies opens it, one that does not keeps it closed.
        MtpNgramDraftConfig config;
        MtpNgramDraft d(config);
        const V text{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
        V prompt = text;
        prompt.push_back(99);
        d.reset(prompt);
        std::vector<double> acc;
        // The output copies `text`: pending 10, the MTP chain keeps one confident draft (11).
        std::size_t at = 0;
        bool opened = false;
        std::uint32_t pending = 10;
        for (int cycle = 0; cycle < 8 && at + 3 < text.size(); ++cycle) {
            const V drafts{text[at + 1], 777, 778};
            auto ext = d.begin_cycle(pending, drafts, 1, 7, acc);
            if (!ext.empty()) {
                opened = true;
                check(ext.front() == text[at + 2] && acc.size() == ext.size(), "extension continues the copied text");
                d.used(ext.size());
            }
            // The target emits the next three copied tokens.
            const V emitted{text[at + 1], text[at + 2], text[at + 3]};
            d.finish_cycle(emitted);
            pending = emitted.back();
            at += 3;
        }
        check(opened, "copying opens the gate");
        check(d.stats().accepted_rows > 0 && d.stats().extended_cycles > 0, "accepted extension rows are counted");
        check(d.stats().shadow_right_rows >= d.stats().accepted_rows && d.stats().scored_cycles > 0, "shadow scoring is counted");
    }
    {
        MtpNgramDraft d;
        d.reset(V{1, 2, 3, 4, 1, 2, 9});
        std::vector<double> acc;
        std::uint32_t pending = 1;
        int extended = 0;
        // The context suggests "1 2 -> 3/9" but the output never follows it.
        for (int cycle = 0; cycle < 20; ++cycle) {
            auto ext = d.begin_cycle(pending, V{2, 50, 51}, 1, 7, acc);
            if (!ext.empty()) { ++extended; d.used(ext.size()); }
            const V emitted{2, static_cast<std::uint32_t>(60 + cycle), 1};
            d.finish_cycle(emitted);
            pending = 1;
        }
        check(extended == 0, "a request that does not copy never gets extension rows");
        check(d.stats().cycles > 0, "shadow proposals are still scored");
        check(d.stats().scored_cycles == d.stats().cycles && d.stats().shadow_right_rows == 0 && d.stats().extension_rows == 0,
            "wrong shadow proposals are scored, counted as not right and never verified");
        check(d.acceptance(2, false, 0) < 0.2, "shadow scoring lowers the learned acceptance");
    }
    {
        // No extension when the chain is not cut or nothing is confident.
        MtpNgramDraft d;
        d.reset(V{1, 2, 3, 1, 2});
        std::vector<double> acc;
        check(d.begin_cycle(2, V{3, 1, 2}, 3, 3, acc).empty(), "full chain is not extended");
        d.finish_cycle(V{3});
        check(d.begin_cycle(3, V{1, 2}, 0, 7, acc).empty(), "no confident draft, no extension");
    }
    {
        // Cross-request lookup: request B copies text that only request A's context holds.
        MtpNgramDraftConfig config;
        config.prior = 0.6;  // open from the start for this test (the default prior keeps it closed until learned)
        MtpNgramDraft a(config), b(config), c(config);
        const V text{40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
        a.reset(text);
        b.reset(V{7, 8, 9});
        c.reset(V{7, 8, 9});
        std::vector<double> acc;
        // B has no own match: without peers nothing, with peers A's continuation (4-token suffix 40..43 + draft 44).
        check(c.begin_cycle(40, V{41, 42, 43, 44}, 4, 7, acc).empty(), "no peers, no cross-request extension");
        b.set_peers({&a, &b});
        auto ext = b.begin_cycle(40, V{41, 42, 43, 44}, 4, 7, acc);
        check(ext.size() == 3 && ext[0] == 45 && ext[2] == 47 && acc.size() == 3, "peer context continues the chain");
        b.used(ext.size());
        b.finish_cycle(V{41, 42, 43, 44, 45, 46, 47, 48});
        check(b.stats().peer_cycles == 1 && b.stats().peer_extension_rows == 3 && b.stats().peer_accepted_rows == 3,
            "peer rows are counted and accepted");
        check(b.peer_acceptance(0) > config.prior && b.acceptance(4, false, 0) == config.prior,
            "peer rows learn in their own cells");
        // A wrong peer continuation closes the peer cells, not the own ones.
        MtpNgramDraft d(config);
        d.reset(V{7, 8, 9});
        d.set_peers({&a});
        for (int cycle = 0; cycle < 12; ++cycle) {
            auto e = d.begin_cycle(40, V{41, 42, 43, 44}, 4, 7, acc);
            if (!e.empty()) d.used(e.size());
            d.finish_cycle(V{41, 42, 43, 44, static_cast<std::uint32_t>(900 + cycle)});
        }
        check(d.peer_acceptance(0) < config.floor, "peer rows that are never right stop being proposed");
        // Peers never shorten an own match: A's own lookup is unchanged by peers.
        MtpNgramDraft e(config);
        e.reset(V{1, 2, 3, 4, 5, 1, 2});
        e.set_peers({&a});
        auto own = e.begin_cycle(2, V{3, 4}, 2, 7, acc);
        check(!own.empty() && own[0] == 5 && e.stats().peer_cycles == 0, "own context first");
    }
    std::printf("mtp n-gram draft tests: failures=%d\n", failures);
    return failures ? 1 : 0;
}
