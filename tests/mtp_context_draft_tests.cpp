#include "strix/mtp_context_draft.hpp"
#include "strix/mtp_window_controller.hpp"
#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what.c_str()); }
}
using V = std::vector<std::uint32_t>;
using strix::MtpContextDraft;
using strix::MtpContextDraftConfig;
V bytes(const std::string& s) { V out; for (const unsigned char c : s) out.push_back(c); return out; }
}
int main() {
    {
        MtpContextDraftConfig c; c.max_suffix = 2;
        MtpContextDraft d(c);
        // The most recent occurrence disagrees with two older ones; vote must win.
        d.reset(V{5,6,1,8,5,6,1,9,5,6,7,0,5,6});
        std::uint32_t suffix = 0, support = 0;
        auto out = d.lookup({}, 2, suffix, support);
        check(out == V{1,9} && suffix == 2 && support == 2, "majority continuation beats recent minority; ties use recency");
        check(d.lookup(V{42,43}, 3, suffix, support).empty() && suffix == 0 && support == 0, "missing key resets metadata");
        check(d.lookup({}, 0, suffix, support).empty(), "zero length produces no rows");
        // Speculative extras can form a key, but never supply historical continuation tokens.
        d.reset(V{1,2});
        check(d.lookup(V{1,2}, 3, suffix, support).empty(), "end-of-context cannot invent continuation from extra drafts");
    }
    {
        MtpContextDraft d;
        V text; for (std::uint32_t t = 10; t < 40; ++t) text.push_back(t);
        V prompt = text; prompt.push_back(99); d.reset(prompt);
        std::vector<double> acc;
        auto ext = d.begin_cycle(10, V{11,777}, 1, 7, acc, 0.7F);
        check(ext == V({12,13,14,15,16,17}) && acc.size() == ext.size(), "cold request immediately proposes beyond confident prefix");
        // R4 carries three candidate inputs: one MTP and two copy. The third copy match is a bonus,
        // not an accepted input. Remaining selected rows did not reach this owner.
        d.finish_cycle(V{11,12,13,14}, 3, 3);
        check(d.stats().selected_rows == 6 && d.stats().extension_rows == 2 && d.stats().accepted_rows == 2,
            "delivery follows actual owner, excludes serial bonus and undelivered rows");
        check(d.stats().scored_cycles == 1 && d.stats().shadow_right_rows == 3,
            "counter patch retains scored primary shadows independently of delivery");
        auto full = d.begin_cycle(14, V{15,16}, 2, 7, acc, 0.7F);
        check(!full.empty() && full.front() == 17, "complete MTP chain can be extended within bounded capacity");
    }
    {
        MtpContextDraft d; d.reset(V{1,2,3,4,5,6,99,1});
        std::vector<double> acc;
        auto ext = d.begin_cycle(2, V{777}, 0, 7, acc, 0.7F);
        check(!ext.empty() && ext.front() == 3 && d.kept_drafts() == 0, "zero confident drafts still propose from pending");
        d.finish_cycle(V{777}, 1, 0);
        check(d.stats().extension_rows == 1 && d.stats().accepted_rows == 0, "rejected copy input counted as delivered, never accepted");
        check(d.acceptance(2, true, 0) < 0.7, "a cold miss immediately lowers posterior below gate (single occurrence = unanimous)");
    }
    {
        MtpContextDraft d; d.reset(V{1,2,3,4,5,6,99,1});
        std::vector<double> acc;
        d.begin_cycle(2, V{3}, 1, 7, acc, 0.7F);
        // Bad MTP prefix: primary cannot be scored, pending-only lookup must still learn.
        d.finish_cycle(V{88}, 1, 0);
        check(d.stats().scored_cycles == 0 && d.stats().learning_rows == 1,
            "pending-only shadow learns when confident MTP prefix is rejected");
    }
    {
        MtpContextDraftConfig c; c.copy_edit = true; c.copy_anchor = 4;
        MtpContextDraft d(c); d.reset(V{1,2,3,4,5,6,7,8,9,99,1,2,3});
        std::vector<double> acc;
        auto ext = d.begin_cycle(4, V{700,701,702}, 3, 3, acc, 0.7F);
        check(ext == V({5,6,7}) && d.kept_drafts() == 0, "copy/edit can replace a full confident chain using a long span");
        check(d.stats().replaced_cycles == 1, "replacing a confident prefix is counted");
        {
            // The confident distillation chain is kept when it is expected to deliver more rows than the span
            // (cold span: 0.75 + 0.5625 + 0.42 < 0.99 + 0.98 + 0.97).
            MtpContextDraft k(c); k.reset(V{1,2,3,4,5,6,7,8,9,99,1,2,3});
            std::vector<double> kept;
            check(k.begin_cycle(4, V{700,701,702}, 3, 3, kept, 0.7F, 2.94).empty() && k.kept_drafts() == 3 &&
                k.stats().replaced_cycles == 0, "a more confident MTP chain wins over a cold context span");
            k.reset(V{1,2,3,4,5,6,7,8,9,99,1,2,3});
            check(k.begin_cycle(4, V{700,701,702}, 3, 3, kept, 0.7F, 0.9) == V({5,6,7}) && k.kept_drafts() == 0,
                "a weak MTP chain is replaced by the context span");
        }
        strix::MtpMinPWindowController controller({2,4,8}, 0.7F,
            strix::MtpWindowCostEstimate::Model::robust, 0, true);
        controller.set_external_rows(0, acc);
        std::vector<float> probabilities; for (const auto p : acc) probabilities.push_back(static_cast<float>(p));
        const std::vector<std::uint8_t> fits{1,1,1};
        const auto owner = controller.choose(probabilities, probabilities.size(), fits);
        check(owner == 1, "cold copy probabilities open the actual min-p R4 owner");
        d.finish_cycle(V{5,6,7,8}, 3, 3);
        check(d.stats().extension_rows == 3 && d.stats().accepted_rows == 3, "full replacement accounts accepted copy inputs");
        d.reset(V{1,2,3,4,5,6,99,1});
        check(d.begin_cycle(2, V{}, 0, 7, acc, 0.9).empty() && acc.empty(), "gate respects stricter window min-p without inflating estimates");
        d.finish_cycle(V{3});
        check(d.stats().extension_rows == 0 && d.stats().scored_cycles == 1, "gate-closed shadows do not claim delivery");
    }
    {
        MtpContextDraftConfig c; c.copy_edit = true; c.copy_anchor = 4;
        MtpContextDraft d(c); d.reset(V{1,2,3,4,5,6,7,8,9,10,11,99,1,2,3});
        std::vector<double> acc;
        const auto ext = d.begin_copy_cycle(4, 7, acc, 0.7F);
        check(ext == V({5,6,7,8,9,10,11}) && d.stats().copy_first_cycles == 1,
            "long copy anchor can supply the complete chain before MTP drafting");
        {
            MtpContextDraft k(c); k.reset(V{1,2,3,4,5,6,7,8,9,10,11,99,1,2,3});
            std::vector<double> kept;
            check(k.begin_copy_cycle(4, 7, kept, 0.7F, 5.0).empty() && kept.empty() && k.stats().cycles == 0 &&
                k.stats().copy_first_declined == 1, "copy-first falls back to MTP when MTP cycles accept more rows");
            check(k.begin_cycle(4, V{}, 0, 7, kept, 0.7F) == V({5,6,7,8,9,10,11}),
                "a declined copy-first probe leaves the ordinary proposal intact");
            MtpContextDraft w(c); w.reset(V{1,2,3,4,5,6,7,8,9,10,11,99,1,2,3});
            check(w.begin_copy_cycle(4, 7, kept, 0.7F, 1.0).size() == 7, "copy-first wins over weak MTP cycles");
        }
        d.finish_cycle(V{5,6,7,8}, 3, 3);
        check(d.stats().extension_rows == 3 && d.stats().accepted_rows == 3,
            "copy-first still waits for actual verifier delivery and acceptance");
        d.reset(V{1,2,3,4,5,6,99,1});
        check(d.begin_copy_cycle(2, 7, acc, 0.7F).empty() && d.stats().cycles == 0,
            "short anchor does not skip MTP or count a selected proposal");
        const auto fallback = d.begin_cycle(2, {}, 0, 7, acc, 0.7F);
        check(!fallback.empty() && fallback.front() == 3 && d.stats().copy_first_cycles == 0,
            "unsuccessful copy-first probe does not duplicate pending in fallback");
    }
    {
        MtpContextDraft d; d.reset(V{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,99});
        std::uint32_t suffix = 0, support = 0;
        auto out = d.lookup(V{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, 1, suffix, support);
        check(out == V{17} && suffix == 16, "long exact suffix extends beyond four tokens");
        d.reset({}); std::vector<double> acc;
        check(d.begin_cycle(42, {}, 0, 0, acc).empty(), "zero capacity is safe");
        d.finish_cycle(V{43});
        check(d.stats().cycles == 0 && d.stats().extension_rows == 0, "reset clears request learning and counters");
    }
    {
        // All rejection depths and candidate capacities: deterministic target simulation checks that
        // speculative candidates only change batching, never the target sequence or accepted count.
        for (std::size_t room = 1; room <= 7; ++room) for (std::size_t reject = 0; reject <= room; ++reject) {
            MtpContextDraftConfig c; c.copy_edit = true; c.copy_anchor = 4; MtpContextDraft d(c);
            V prompt{10,11,12,13,14,15,16,17,18,19,99,10,11,12}; d.reset(prompt);
            std::vector<double> acc; const auto ext = d.begin_cycle(13, {}, 0, room, acc, 0.7F);
            check(ext.size() == room, "proposal obeys each capacity");
            V target(ext.begin(), ext.end()); target.push_back(20);
            if (reject < room) target[reject] = 800;
            V emitted; std::size_t accepted = 0;
            for (std::size_t row = 0; row <= room; ++row) {
                emitted.push_back(target[row]);
                if (row == room || ext[row] != target[row]) break;
                ++accepted;
            }
            check(std::equal(emitted.begin(), emitted.end(), target.begin()) && accepted == reject,
                "simulated exact verifier commits only target prefix at every rejection depth");
            d.finish_cycle(emitted, room, accepted);
            check(d.stats().extension_rows == room && d.stats().accepted_rows == reject, "actual delivery and rejection depth agree");
        }
    }
    {
        // Own EN/ES edit fixtures, encoded as byte token IDs. Copy resumes after a one-token edit.
        for (const auto& text : {std::string("tool output: alpha beta gamma delta. quote the exact result."),
                                 std::string("salida de herramienta: alfa beta gamma delta. cita el resultado exacto.")}) {
            const auto source = bytes(text);
            MtpContextDraftConfig c; c.copy_edit = true; c.copy_anchor = 4; MtpContextDraft d(c);
            V prompt = source; prompt.push_back(999); prompt.insert(prompt.end(), source.begin(), source.begin()+3);
            d.reset(prompt); std::vector<double> acc;
            const auto ext = d.begin_cycle(source[3], V{888,889}, 0, 7, acc, 0.7F);
            check(ext.size() == 7 && std::equal(ext.begin(), ext.end(), source.begin()+4), "EN/ES prompt span supplies real copy inputs");
            V emitted(ext); emitted.push_back(source[11]); d.finish_cycle(emitted, 7, 7);
            check(d.stats().accepted_rows == 7, "EN/ES accepted inputs are counted");
            // A target edit, then a fresh four-token anchor in the source.
            d.finish_cycle(V{777,source[20],source[21],source[22],source[23]});
            const auto resumed = d.begin_cycle(source[23], V{888}, 0, 7, acc, 0.7F);
            check(!resumed.empty() && resumed.front() == source[24], "copy resumes after a target edit at a new exact anchor");
        }
    }
    {
        for (const auto bad : {std::numeric_limits<double>::quiet_NaN(), -0.1, 1.1}) {
            MtpContextDraftConfig c; c.floor = bad; bool threw = false;
            try { MtpContextDraft d(c); } catch (const std::invalid_argument&) { threw = true; }
            check(threw, "invalid gate rejected");
        }
    }
    {
        // Wide copy verify: exactly `rows` tokens from a long unanimous anchor whose acceptance cells are confident.
        V text; for (std::uint32_t t = 100; t < 140; ++t) text.push_back(t);
        V prompt = text; prompt.push_back(999); prompt.insert(prompt.end(), text.begin(), text.begin() + 8);
        const V expected(text.begin() + 9, text.end()); // 109..139: 31 tokens after the pending 108
        MtpContextDraftConfig cold; cold.copy_edit = true;
        MtpContextDraft d(cold); d.reset(prompt);
        check(d.propose_wide(108, 31).empty(), "cold acceptance cells (prior 0.75) do not open a wide cycle");
        MtpContextDraftConfig c = cold; c.prior = 0.95;
        MtpContextDraft w(c); w.reset(prompt);
        const auto wide = w.propose_wide(108, 31);
        check(wide == expected, "a long unanimous anchor with confident cells proposes exactly the 31 copied rows");
        V emitted = wide; emitted.push_back(999);
        w.finish_cycle(emitted, 31, 31);
        check(w.stats().wide_cycles == 1 && w.stats().wide_rows_accepted == 31 && w.stats().wide_abandoned == 0,
            "a fully accepted wide cycle is counted");
        check(w.stats().extension_rows == 0 && w.stats().accepted_rows == 0 && w.stats().selected_rows == 0,
            "wide rows stay out of the window delivery counters");
        check(w.acceptance(9, true, 3) > 0.95 && w.stats().scored_cycles == 1, "a wide cycle learns through finish_cycle");
        MtpContextDraft r(c); r.reset(prompt);
        const auto rejected = r.propose_wide(108, 31);
        check(rejected == expected, "the same proposal on a fresh request");
        V truth(rejected.begin(), rejected.begin() + 10); truth.push_back(800);
        const auto before = r.acceptance(9, true, 3);
        r.abandon_wide(truth);
        check(r.stats().wide_cycles == 0 && r.stats().wide_rows_accepted == 0 && r.stats().wide_abandoned == 1,
            "an abandoned wide proposal is counted apart from committed wide cycles");
        check(r.acceptance(9, true, 3) < before && r.acceptance(9, true, 3) < w.acceptance(9, true, 3),
            "an abandoned proposal is scored as a shadow: its rejected row lowers the learned acceptance");
        check(r.stats().scored_cycles == 0 && r.stats().learning_rows == 11, "abandon learns rows 0..10 without a scored cycle");
        r.abandon_wide(truth);
        check(r.stats().wide_abandoned == 1, "abandon without an open wide proposal does nothing");
        // Nothing was committed, but row 10's context token (119) is now an edited token: a span containing it is not
        // proposed again, and copy proposals stop before it.
        check(r.propose_wide(108, 31, 0.0).empty() && r.stats().wide_blocked == 1,
            "a wide span containing a token the output edited is refused");
        std::vector<double> cut_acc;
        const auto cut = r.begin_cycle(108, V{}, 0, 7, cut_acc, 0.0);
        check(cut == V({109,110,111,112,113,114,115}) && r.stats().edit_cuts == 0, "rows before the edited token are still proposed");
        r.finish_cycle(V{109}, 1, 1);
        MtpContextDraft e(c); e.reset(prompt);
        (void)e.propose_wide(108, 31); e.abandon_wide(truth);
        for (std::uint32_t t = 109; t < 116; ++t) { std::vector<double> acc2; (void)e.begin_cycle(t - 1, V{}, 0, 7, acc2, 0.0); e.finish_cycle(V{t}, 0, 0); }
        std::vector<double> acc3;
        const auto stop = e.begin_cycle(115, V{}, 0, 7, acc3, 0.0);
        check(stop == V({116,117,118}) && e.stats().edit_cuts >= 1, "a copy proposal stops before the edited token");
        MtpContextDraft s(c); s.reset(prompt);
        check(s.propose_wide(108, 60).empty(), "a context continuation shorter than the requested rows is refused");
        check(s.propose_wide(108, 0).empty(), "zero rows propose nothing");
        MtpContextDraft a(c);
        a.reset(V{1,2,3,4,5,6,7,8,50,77, 1,2,3,4,5,6,7,8,60,88, 1,2,3,4,5,6,7});
        check(a.propose_wide(8, 2).empty(), "an ambiguous anchor is refused");
        MtpContextDraft h(c);
        V brief{1,2,3,4}; for (std::uint32_t t = 200; t < 240; ++t) brief.push_back(t);
        brief.push_back(999); brief.insert(brief.end(), {1,2,3});
        h.reset(brief);
        check(h.propose_wide(4, 31).empty(), "an anchor below copy_anchor is refused");
        MtpContextDraftConfig ctx = c; ctx.copy_edit = false;
        MtpContextDraft x(ctx); x.reset(prompt);
        check(x.propose_wide(108, 31).empty(), "wide proposals are copy/edit only");
        MtpContextDraft o(c); o.reset(prompt);
        check(o.propose_wide(108, 31) == expected, "a proposal before an ordinary cycle");
        std::vector<double> acc;
        (void)o.begin_cycle(108, {}, 0, 7, acc, 0.7F);
        o.finish_cycle(V{109}, 1, 1);
        check(o.stats().wide_cycles == 0, "an ordinary cycle after an abandoned wide proposal is not a wide cycle");
        bool threw = false;
        try { (void)o.propose_wide(109, 31, 1.5); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "an invalid wide minimum acceptance is rejected");
        MtpContextDraftConfig strict = c; strict.prior = 0.85;
        MtpContextDraft t(strict); t.reset(prompt);
        check(t.propose_wide(108, 31).empty() && t.propose_wide(108, 31, 0.8) == expected,
            "the minimum acceptance gates every long-unanimous cell");
        MtpContextDraftConfig anchored = c; anchored.wide_anchor = 16;
        MtpContextDraft wa(anchored); wa.reset(prompt);
        check(wa.propose_wide(108, 31).empty(), "an anchor shorter than wide_anchor opens no wide cycle");
        MtpContextDraftConfig bad = c; bad.wide_anchor = 17; bool rejected_anchor = false;
        try { MtpContextDraft x2(bad); } catch (const std::invalid_argument&) { rejected_anchor = true; }
        check(rejected_anchor, "a wide anchor above max_suffix is rejected");
    }
    {
        // Chain copy: fresh copies need chain_fresh (12) matching tokens, a resumed copy chain_resume (5).
        MtpContextDraftConfig c; c.copy_edit = true; c.chain_fresh = 12; c.chain_resume = 5;
        V text; for (std::uint32_t t = 300; t < 360; ++t) text.push_back(t);
        // Prompt: the source, a separator, then the first 11 source tokens repeated (pending = text[11] is pushed).
        V prompt = text; prompt.push_back(999); prompt.insert(prompt.end(), text.begin(), text.begin() + 11);
        MtpContextDraft d(c); d.reset(prompt);
        std::vector<double> acc;
        check(d.chain_extend(text[11], {}, 7, acc) == V(text.begin() + 12, text.begin() + 19) && acc.size() == 7 && acc[0] > 0.8,
            "a 12-token match copies the continuation");
        MtpContextDraft s11(c); V short_prompt = text; short_prompt.push_back(999);
        short_prompt.insert(short_prompt.end(), text.begin(), text.begin() + 10); s11.reset(short_prompt);
        check(s11.chain_extend(text[10], {}, 7, acc).empty(), "an 11-token match is below the fresh threshold");
        // Kept drafts extend the match: 10 history tokens + pending + 1 kept draft = 12.
        check(s11.chain_extend(text[10], V{text[11]}, 6, acc) == V(text.begin() + 12, text.begin() + 18),
            "kept drafts are part of the matched suffix");
        // Full acceptance establishes the copy; chain_first continues it without a predictor forward.
        auto first = d.chain_extend(text[11], {}, 7, acc);
        V emitted(first); emitted.push_back(text[19]);
        d.finish_cycle(emitted, 7, 7);
        check(d.stats().chain_accepted_rows == 7, "accepted chain rows are counted");
        check(d.chain_first(text[19], 7, acc) == V(text.begin() + 20, text.begin() + 27) && d.stats().chain_first_cycles == 1,
            "an established copy continues copy-first");
        // An edit interrupts it: 5 matching tokens resume the same source.
        d.finish_cycle(V{800}, 7, 0);
        check(d.chain_first(800, 7, acc).empty(), "a rejected copy is not established");
        for (std::uint32_t t = 21; t < 25; ++t) d.finish_cycle(V{text[t]}, 0, 0);
        check(d.chain_extend(text[24], {}, 3, acc).empty(), "4 tokens after the edit do not resume");
        d.finish_cycle(V{text[25]}, 0, 0);
        check(d.chain_extend(text[25], {}, 3, acc) == V({text[26], text[27], text[28]}), "a resumed copy needs 5 tokens");
        // Self-repetition: a period-2 pattern copies past the end of history from its own rows.
        MtpContextDraftConfig r = c; r.chain_fresh = 4; r.chain_resume = 4;
        MtpContextDraft rep(r); rep.reset(V{7,8,7,8,7,8,7});
        check(rep.chain_extend(8, {}, 6, acc) == V({7,8,7,8,7,8}), "self-repetition continues through the copy's own rows");
        // A predictor-drafted cycle that copied (tail matches >= 12) establishes the copy for the next cycle.
        MtpContextDraft e(c); e.reset(prompt);
        std::vector<double> acc2;
        check(e.chain_first(text[11], 7, acc2).empty(), "nothing is established before a copy");
        V copied(text.begin() + 12, text.begin() + 20);
        e.finish_cycle(copied, 7, 7);
        // Established, but the prior (0.9) is below chain_first_accept: the predictor drafts once more, and the copy
        // it would have proposed is scored against what it emitted (8 of 8 right).
        check(e.stats().chain_established == 1 && e.chain_first(text[19], 7, acc2).empty(),
            "an established copy waits for its learned acceptance");
        e.finish_cycle(V(text.begin() + 20, text.begin() + 28), 7, 7);
        check(e.chain_first(text[27], 7, acc2) == V(text.begin() + 28, text.begin() + 35),
            "a drafted cycle that copied establishes copy-first");
        // Copy-first needs the learned chain acceptance: a request whose copies keep failing drafts with the predictor.
        MtpContextDraftConfig strict = c; strict.chain_first_accept = 0.95;
        MtpContextDraft f(strict); f.reset(prompt);
        auto p1 = f.chain_extend(text[11], {}, 7, acc2);
        f.finish_cycle(V{text[12], 801}, 7, 1);                      // 1 of 7 accepted: acceptance (0.9+1)/(1+2)
        for (std::uint32_t t2 = 0; t2 < 13; ++t2) f.finish_cycle(V{text[t2]}, 0, 0);
        check(f.chain_first(text[12], 7, acc2).empty(), "copy-first needs the learned chain acceptance");
        // With a wide anchor, a match that reaches it is left to the wide window.
        MtpContextDraftConfig wa2 = c; wa2.wide_anchor = 12;
        MtpContextDraft w2(wa2); w2.reset(prompt);
        (void)w2.chain_extend(text[11], {}, 7, acc2);
        w2.finish_cycle(V(text.begin() + 12, text.begin() + 20), 7, 7);
        check(w2.chain_first(text[19], 7, acc2).empty(), "copy-first leaves matches at the wide anchor to the wide window");
        MtpContextDraftConfig bad = c; bad.chain_fresh = 2; bool threw = false;
        try { MtpContextDraft x(bad); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "a chain threshold below the 3-token key is rejected");
        MtpContextDraftConfig off; off.copy_edit = true;
        MtpContextDraft o(off); o.reset(prompt);
        check(o.chain_extend(text[11], {}, 7, acc).empty() && !o.chain_enabled(), "chain copy is off by default");
    }
    std::printf("mtp n-gram/copy draft tests: failures=%d\n", failures);
    {
        // Records share long anchors but differ in their values: copy-first and replacement need a unanimous
        // anchor of copy_anchor tokens; an ambiguous anchor can only extend the MTP chain.
        MtpContextDraftConfig c; c.copy_edit = true;
        MtpContextDraft d(c);
        d.reset(V{1,2,3,4,5,6,7,8,50,77, 1,2,3,4,5,6,7,8,60,88, 1,2,3,4,5,6,7});
        std::vector<double> acc;
        check(d.begin_copy_cycle(8, 7, acc, 0.7F).empty(), "an ambiguous 8-token anchor does not skip the predictor");
        MtpContextDraft u(c);
        u.reset(V{1,2,3,4,5,6,7,8,50,9,10,11,12,13,14, 99, 1,2,3,4,5,6,7});
        check(u.begin_copy_cycle(8, 7, acc, 0.7F) == V({50,9,10,11,12,13,14}), "a unanimous 8-token anchor does");
        MtpContextDraft s4(c);
        s4.reset(V{1,2,3,4,50,9,10,11,12,13,14, 99, 1,2,3});
        check(s4.begin_copy_cycle(4, 7, acc, 0.7F).empty(), "a 4-token anchor is below the default copy anchor");
    }
    return failures ? 1 : 0;
}
