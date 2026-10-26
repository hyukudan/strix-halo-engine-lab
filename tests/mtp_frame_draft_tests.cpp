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
    std::printf("mtp frame draft tests: failures=%d\n", failures);
    return failures ? 1 : 0;
}
