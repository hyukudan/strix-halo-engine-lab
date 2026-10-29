#include "strix/mtp_frame_draft.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what.c_str()); }
}
using V = std::vector<std::uint32_t>;
// Token stand-ins: 100 open, 101 close, 10 '\n', 11 '<function=', 12 '>', 13 '<parameter=', 14 '</parameter>',
// 15 '</function>', names 20 (bash) 21 (str_replace_editor), keys 30 (command) 31 (path), values >= 50.
constexpr std::uint32_t O = 100, C = 101;
}  // namespace

int main() {
    using strix::MtpFrameDraft;
    using strix::MtpFrameDraftConfig;
    MtpFrameDraftConfig config;
    config.open = O; config.close = C;
    const V call_bash{O, 10, 11, 20, 12, 10, 13, 30, 12, 10, 50, 51, 10, 14, 10, 15, 10, C};
    const V call_edit{O, 10, 11, 21, 12, 10, 13, 30, 12, 10, 60, 10, 14, 10, 13, 31, 12, 10, 61, 62, 10, 14, 10, 15, 10, C};
    {
        MtpFrameDraft d(config);
        V prompt{1, 2, 3};
        prompt.insert(prompt.end(), call_bash.begin(), call_bash.end());
        prompt.insert(prompt.end(), call_edit.begin(), call_edit.end());
        prompt.insert(prompt.end(), {4, 5});
        d.reset(prompt);
        bool ambiguous = false;
        std::size_t support = 0;
        // Outside any frame: nothing.
        check(d.lookup(V{6}, 7, ambiguous, support).empty(), "no proposal outside a frame");
        // Just opened: every earlier frame continues "\n<function=" then disagrees on the name.
        auto out = d.lookup(V{O}, 7, ambiguous, support);
        check(out == V{10, 11} && support == 2, "frame opening is the common prefix of all frames");
        // The name typed: the rest of that frame's header, up to where the frames with this name disagree.
        out = d.lookup(V{O, 10, 11, 21}, 7, ambiguous, support);
        check(out == V{12, 10, 13, 30, 12, 10, 60} && support == 1, "a typed name continues its single frame");
        // A parameter value is never invented when two frames disagree right after the anchor.
        out = d.lookup(V{O, 10, 11, 20, 12, 10, 13, 30, 12, 10, 77}, 7, ambiguous, support);
        check(out.empty(), "a new value has no continuation");
        // After a value: the closing of the parameter is shared ("\n</parameter>\n") and then disagrees.
        out = d.lookup(V{O, 10, 11, 20, 12, 10, 13, 30, 12, 10, 77, 10, 14}, 7, ambiguous, support);
        check(out.size() >= 1 && out[0] == 10, "parameter close continues");
        // After </function>: "\n" and the close token.
        out = d.lookup(V{O, 10, 11, 20, 12, 10, 15}, 7, ambiguous, support);
        check(out == V{10, C}, "function close ends the frame");
    }
    {
        // A continuation seen in a single earlier frame starts closed (single_prior below the floor).
        {
            auto low = config;
            low.single_prior = 0.4;
            MtpFrameDraft closed(low);
            closed.reset(call_bash);
            std::vector<double> none;
            check(closed.begin_cycle(O, V{10, 99}, 1, 7, none).empty() && closed.stats().cycles == 1,
                "single-frame continuation is only a shadow at first");
        }
        // Learning: a request whose frames never repeat stops proposing; a repeating one is accepted.
        config.single_prior = 0.8;
        MtpFrameDraft d(config);
        V prompt = call_bash;
        d.reset(prompt);
        std::vector<double> acc;
        int proposed = 0;
        std::uint32_t pending = O;
        // The output repeats call_bash: one confident draft (10) each cycle.
        auto ext = d.begin_cycle(pending, V{10, 99}, 1, 7, acc);
        check(!ext.empty() && ext.front() == 11 && acc.size() == ext.size(), "frame proposal after the open token");
        if (!ext.empty()) { d.used(ext.size()); ++proposed; }
        d.finish_cycle(V{10, 11, 20, 12, 10, 13, 30, 12});
        check(d.stats().accepted_rows == 6 && d.stats().scored_cycles == 1, "accepted frame rows are counted");
        // Wrong proposals lower the learned acceptance until nothing is proposed.
        MtpFrameDraft e(config);
        e.reset(call_bash);
        int extended = 0;
        for (int cycle = 0; cycle < 12; ++cycle) {
            auto x = e.begin_cycle(O, V{10, 99}, 1, 7, acc);
            if (!x.empty()) { e.used(x.size()); ++extended; }
            e.finish_cycle(V{10, 90, C});  // frames that never follow the earlier one
            // the cycle's emitted close ends the frame; reopen next cycle (pending O is pushed only once, so push it)
            e.finish_cycle(V{O});
        }
        check(extended > 0 && extended < 12, "unconfirmed frame proposals stop");
    }
    {
        // Committed-prefix gate: nothing is proposed until the frame's first prefix_gate committed tokens equal the
        // start of an earlier frame; a frame opened inside the unverified drafts is held back too.
        auto gated = config;
        gated.prefix_gate = 4;
        std::vector<double> acc;
        MtpFrameDraft g(gated);
        g.reset(call_bash);
        check(g.begin_cycle(O, V{10, 11}, 2, 7, acc).empty() && g.stats().gated_cycles == 1,
            "gate holds a frame with fewer committed tokens than the gate");
        g.finish_cycle(V{10, 11, 20});  // committed O 10 11 20 = the start of call_bash
        auto out = g.begin_cycle(O, V{12}, 0, 7, acc);
        check(!out.empty() && out.front() == 12 && g.stats().gated_cycles == 1, "matching committed prefix opens the gate");
        MtpFrameDraft h(gated);
        h.reset(call_bash);
        h.begin_cycle(O, V{}, 0, 7, acc);
        h.finish_cycle(V{10, 11, 21});  // O 10 11 21: no earlier frame starts this way
        check(h.begin_cycle(O, V{12}, 0, 7, acc).empty() && h.stats().gated_cycles == 2 && h.stats().cycles == 0,
            "a frame that starts like no earlier frame proposes nothing");
        MtpFrameDraft k(gated);
        k.reset(call_bash);
        check(k.begin_cycle(4, V{O, 10}, 2, 7, acc).empty() && k.stats().gated_cycles == 1,
            "a frame opened in the drafts is gated");
        auto off = gated;
        off.prefix_gate = 0;
        MtpFrameDraft u(off);
        u.reset(call_bash);
        check(!u.begin_cycle(4, V{O, 10}, 2, 7, acc).empty() && u.stats().gated_cycles == 0, "gate 0 is the old behaviour");
    }
    {
        // Gate edge cases (production gate 16, one-token gate, prompt-carried frames, resets, reopen, short frames,
        // duplicate openers, stray closes, rejected drafts).
        std::vector<double> acc;
        auto g16 = config;
        g16.prefix_gate = 16;
        MtpFrameDraft a(g16);
        a.reset(call_edit);
        check(a.begin_cycle(O, V{}, 0, 7, acc).empty(), "gate 16: just opened");
        a.finish_cycle(V(call_edit.begin() + 1, call_edit.begin() + 15));  // 15 committed tokens
        check(a.begin_cycle(O, V{}, 0, 7, acc).empty(), "gate 16: 15 committed tokens stay gated");
        a.finish_cycle(V{call_edit[15]});  // the 16th
        auto out = a.begin_cycle(O, V{}, 0, 7, acc);
        check(!out.empty() && out.front() == call_edit[16], "gate 16 opens on the 16th matching committed token");

        auto g1 = config;
        g1.prefix_gate = 1;
        MtpFrameDraft one(g1);
        one.reset(call_bash);
        out = one.begin_cycle(O, V{}, 0, 7, acc);
        check(!out.empty() && out.front() == 10 && one.stats().gated_cycles == 0, "a one-token gate opens on the opener");

        auto g4 = config;
        g4.prefix_gate = 4;
        V carried = call_bash;
        carried.insert(carried.end(), {O, 10, 11, 20});
        MtpFrameDraft p(g4);
        p.reset(carried);
        check(!p.begin_cycle(12, V{}, 0, 7, acc).empty(), "a prompt-carried matching prefix opens the gate");
        p.reset(carried);
        p.reset(call_bash);
        check(p.begin_cycle(O, V{10}, 1, 7, acc).empty() && p.stats().gated_cycles == 1, "reset clears the gate");
        V wrong = call_bash;
        wrong.insert(wrong.end(), {O, 10, 11, 21});
        MtpFrameDraft q(g4);
        q.reset(wrong);
        check(q.begin_cycle(12, V{}, 0, 7, acc).empty(), "a prompt-carried mismatching prefix stays gated");

        MtpFrameDraft r(g4);
        r.reset(carried);
        r.begin_cycle(12, V{}, 0, 7, acc);
        r.finish_cycle(V{10, 15, 10, C, O, 10, 11, 21});  // close the open frame, reopen with a new start
        check(r.begin_cycle(0, V{12}, 0, 7, acc).empty(), "a reopened frame is gated again");

        MtpFrameDraft s(g4);
        s.reset(V{O, 10, C, O, 10, 11, 20});  // the only earlier frame is shorter than the gate
        check(s.begin_cycle(12, V{}, 0, 7, acc).empty(), "short earlier frames never open the gate");

        V dup = call_bash;
        dup.insert(dup.end(), {O, 10, O, 10, 11, 20});
        MtpFrameDraft d(g4);
        d.reset(dup);
        check(!d.begin_cycle(12, V{}, 0, 7, acc).empty(), "a duplicate opener restarts the committed prefix");

        V stray = call_bash;
        stray.insert(stray.end(), {C, 7, O, 10, 11, 20});
        MtpFrameDraft t(g4);
        t.reset(stray);
        check(!t.begin_cycle(12, V{}, 0, 7, acc).empty(), "a stray close is ignored");

        MtpFrameDraft x(g4);
        x.reset(call_bash);
        x.begin_cycle(O, V{}, 0, 7, acc);
        x.finish_cycle(V{10, 11});  // committed O 10 11
        check(x.begin_cycle(0, V{20, 12}, 2, 7, acc).empty(), "drafts never complete the committed prefix");
        x.finish_cycle(V{21});      // the draft was rejected
        check(x.begin_cycle(0, V{12}, 0, 7, acc).empty(), "a rejected draft leaves the frame gated");
    }
    {
        using strix::parse_frame_draft_prefix_gate;
        check(parse_frame_draft_prefix_gate(nullptr) == 16 && parse_frame_draft_prefix_gate("") == 16, "unset/empty keep 16");
        check(parse_frame_draft_prefix_gate("0") == 0 && parse_frame_draft_prefix_gate("8") == 8 &&
                  parse_frame_draft_prefix_gate("4096") == 4096, "valid values");
        check(parse_frame_draft_prefix_gate("-1") == 16 && parse_frame_draft_prefix_gate("16junk") == 16 &&
                  parse_frame_draft_prefix_gate(" 8") == 16 && parse_frame_draft_prefix_gate("4097") == 16 &&
                  parse_frame_draft_prefix_gate("4294967296") == 16 && parse_frame_draft_prefix_gate("99999999999999999999") == 16,
            "invalid values keep 16");
    }
    std::printf("mtp frame draft tests: failures=%d\n", failures);
    return failures ? 1 : 0;
}
