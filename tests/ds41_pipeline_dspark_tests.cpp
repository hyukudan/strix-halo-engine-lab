// strix/ds41-pipeline-v1 speculative extension (ds41_pipeline_dspark.hpp): message round trips, the verify rule, EOS /
// max_new cuts, the TAIL's decision log and a FRONT <-> TAIL session over a lossy in-memory channel (VERIFY dropped,
// duplicated, sent with a stale epoch; SPEC_ACCEPT dropped and duplicated) against a toy target and drafter: the
// published tokens equal greedy decoding, both sides end at the same committed position, and no window is verified twice.
#include "strix/ds41_pipeline_dspark.hpp"

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

namespace sp = strix::ds41::pipe;

namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

// toy target: the greedy token after a context (vocabulary 23, token 22 = EOS)
std::uint32_t greedy(const std::vector<std::uint32_t>& ctx) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const auto t : ctx) h = (h ^ t) * 1099511628211ULL;
    return static_cast<std::uint32_t>((h >> 17U) % 23U);
}
// toy drafter: the true continuation, wrong at a depth that depends on the position
std::vector<std::uint32_t> draft(std::vector<std::uint32_t> ctx, std::uint32_t t, std::uint32_t k) {
    std::vector<std::uint32_t> d;
    ctx.push_back(t);
    const std::uint32_t bad = static_cast<std::uint32_t>(ctx.size() * 7U % 6U);  // 0..5
    for (std::uint32_t j = 0; j < k; ++j) {
        std::uint32_t g = greedy(ctx);
        if (j + 1U == bad) g = (g + 1U) % 21U;
        d.push_back(g);
        ctx.push_back(g);
    }
    return d;
}

struct Msg {
    bool verify = false;  // FRONT -> TAIL: VERIFY (else SPEC_ACCEPT, TAIL -> FRONT)
    std::uint64_t request = 0;
    std::vector<std::uint32_t> rows;  // VERIFY: the window's tokens
    std::uint32_t position = 0;
    std::vector<std::uint8_t> payload;  // SPEC_ACCEPT
};

struct Faults {
    std::uint32_t drop_verify = 0, dup_verify = 0, stale_verify = 0, drop_accept = 0, dup_accept = 0;  // every n-th (0: never)
};

// One sequence: prompt context, then max_new tokens. Returns the published tokens.
std::vector<std::uint32_t> session(const std::vector<std::uint32_t>& prompt, std::uint32_t max_new, const Faults& f, std::uint32_t* verified_rows,
                                   std::uint32_t* resends, bool* same_position) {
    const std::vector<std::uint32_t> eos{22U};
    // ---- TAIL state
    std::vector<std::uint32_t> tail_ctx = prompt;
    sp::SpecDecisionLog log;
    std::uint32_t tail_remaining = 0, tail_next = 0;
    std::uint32_t computed = 0;
    // ---- FRONT state
    std::vector<std::uint32_t> front_ctx = prompt;
    std::vector<std::uint32_t> out;
    const std::uint32_t first = greedy(prompt);  // the prompt fill's last argmax
    out.push_back(first);
    std::deque<Msg> to_tail, to_front;
    const std::uint32_t epoch = 7;
    const auto plan_for = [&](std::uint32_t next, std::uint32_t remaining) {
        const std::uint32_t k = std::min<std::uint32_t>(static_cast<std::uint32_t>(tail_ctx.size() % 6U), remaining - 1U);  // the "controller": 0..5 drafts
        std::vector<std::uint32_t> plan{next};
        const auto d = draft(tail_ctx, next, k);
        plan.insert(plan.end(), d.begin(), d.end());
        return plan;
    };
    // BEGIN (no loss modelled: a lost BEGIN is a failed session start)
    {
        sp::SpecBegin b{epoch, static_cast<std::uint32_t>(prompt.size()), first, max_new, 0U, eos};
        const auto bb = sp::decode_spec_begin(sp::encode(b));
        log.begin(bb.epoch);
        tail_remaining = bb.max_new - 1U;
        tail_next = bb.next_token;
        sp::SpecAccept a;
        a.epoch = epoch;
        a.window = 0;
        a.committed = static_cast<std::uint32_t>(tail_ctx.size());
        a.next_token = tail_next;
        a.plan = plan_for(tail_next, tail_remaining);
        auto bytes = sp::encode(a);
        log.record(0, bytes);
        to_front.push_back({false, 0, {}, 0, bytes});
    }
    sp::SpecFrontState fs{epoch, 0};
    std::vector<std::uint32_t> plan;
    Msg last_verify;
    std::uint32_t sent = 0, accepts = 0;
    bool done = first == 22U || max_new == 1U;
    std::uint32_t guard = 0;
    while (!done && ++guard < 100000U) {
        // ---- TAIL: serve one request
        if (!to_tail.empty()) {
            const Msg m = to_tail.front();
            to_tail.pop_front();
            const auto kind = log.classify(m.request);
            if (kind == sp::SpecDecisionLog::Kind::duplicate) {
                to_front.push_back({false, 0, {}, 0, log.last()});
            } else if (kind == sp::SpecDecisionLog::Kind::stale) {
                sp::SpecAccept a;
                a.epoch = sp::spec_epoch(m.request);
                a.window = sp::spec_window(m.request);
                a.status = sp::SpecStatus::stale;
                to_front.push_back({false, 0, {}, 0, sp::encode(a)});
            } else {
                expect(m.position == tail_ctx.size(), "VERIFY at the TAIL's committed position");
                ++computed;
                *verified_rows += static_cast<std::uint32_t>(m.rows.size());
                std::vector<std::uint32_t> g;
                auto ctx = tail_ctx;
                for (const auto t : m.rows) {
                    ctx.push_back(t);
                    g.push_back(greedy(ctx));
                }
                const auto o = sp::spec_outcome(m.rows, g, tail_remaining, eos);
                tail_ctx.insert(tail_ctx.end(), m.rows.begin(), m.rows.begin() + o.kept);
                tail_remaining -= static_cast<std::uint32_t>(o.published.size());
                tail_next = o.published.back();
                sp::SpecAccept a;
                a.epoch = epoch;
                a.window = sp::spec_window(m.request);
                a.position = m.position;
                a.rows = static_cast<std::uint32_t>(m.rows.size());
                a.kept = o.kept;
                a.argmax = g;
                a.committed = static_cast<std::uint32_t>(tail_ctx.size());
                a.next_token = tail_next;
                a.published = o.published;
                a.done = o.done;
                if (o.done == sp::SpecDone::more) a.plan = plan_for(tail_next, tail_remaining);
                auto bytes = sp::encode(a);
                log.record(a.window, bytes);
                ++accepts;
                if (!(f.drop_accept && accepts % f.drop_accept == 0U)) to_front.push_back({false, 0, {}, 0, bytes});
                if (f.dup_accept && accepts % f.dup_accept == 0U) to_front.push_back({false, 0, {}, 0, bytes});
            }
        }
        // ---- FRONT: a decision, or a timeout (nothing to read): resend the last VERIFY
        if (to_front.empty()) {
            if (to_tail.empty() && sent != 0U) {
                ++*resends;
                to_tail.push_back(last_verify);
            }
            continue;
        }
        const Msg m = to_front.front();
        to_front.pop_front();
        const auto a = sp::decode_spec_accept(m.payload);
        if (a.status != sp::SpecStatus::ok || !fs.expected(a)) continue;  // stale answers, duplicates
        if (a.window != 0U) {
            front_ctx.insert(front_ctx.end(), plan.begin(), plan.begin() + a.kept);
            out.insert(out.end(), a.published.begin(), a.published.end());
            expect(a.committed == front_ctx.size(), "committed positions agree");
            if (a.done != sp::SpecDone::more) {
                done = true;
                break;
            }
        }
        plan = a.plan;
        fs.window = a.window + 1U;
        last_verify = {true, sp::spec_request(epoch, fs.window), plan, static_cast<std::uint32_t>(front_ctx.size()), {}};
        ++sent;
        if (!(f.drop_verify && sent % f.drop_verify == 0U)) to_tail.push_back(last_verify);
        if (f.dup_verify && sent % f.dup_verify == 0U) to_tail.push_back(last_verify);
        if (f.stale_verify && sent % f.stale_verify == 0U) {
            auto st = last_verify;
            st.request = sp::spec_request(epoch - 1U, fs.window);
            to_tail.push_front(st);
        }
    }
    expect(guard < 100000U, "session ends");
    *same_position = tail_ctx == front_ctx;
    expect(computed == fs.window || done, "every window verified once");
    return out;
}

}  // namespace

int main() {
    // ---- round trips
    {
        sp::SpecBegin b{3, 2048, 10316, 128, sp::kStepWantRowFnv, {1, 2}};
        const auto d = sp::decode_spec_begin(sp::encode(b));
        expect(d.epoch == 3 && d.position == 2048 && d.next_token == 10316 && d.max_new == 128 && d.flags == 2 && d.eos == b.eos, "SPEC_BEGIN round trip");
        sp::SpecAccept a;
        a.epoch = 3;
        a.window = 5;
        a.position = 2050;
        a.rows = 3;
        a.kept = 2;
        a.argmax = {7, 8, 9};
        a.row_fnv = {1, 2, 3};
        a.committed = 2052;
        a.next_token = 8;
        a.published = {7, 8};
        a.plan = {8, 1, 2};
        a.draft_ms = 6.5F;
        const auto e = sp::encode(a);
        const auto x = sp::decode_spec_accept(e);
        expect(sp::encode(x) == e, "SPEC_ACCEPT round trip");
        auto bad = e;
        bad.push_back(0);
        bool threw = false;
        try {
            (void)sp::decode_spec_accept(bad);
        } catch (const sp::WireError&) {
            threw = true;
        }
        expect(threw, "SPEC_ACCEPT trailing byte refused");
        const auto c = sp::decode_spec_commit(sp::encode_spec_commit(3, 9));
        expect(c.first == 3 && c.second == 9, "SPEC_COMMIT round trip");
        expect(sp::spec_epoch(sp::spec_request(4, 77)) == 4 && sp::spec_window(sp::spec_request(4, 77)) == 77, "request packing");
    }
    // ---- the verify rule and the cuts
    {
        const std::vector<std::uint32_t> in{5, 6, 7, 8}, g{6, 7, 9, 1};
        expect(sp::accepted_prefix(in, g) == 2U, "accepted prefix");
        auto o = sp::spec_outcome(in, g, 100, {});
        expect(o.kept == 3U && o.published == std::vector<std::uint32_t>({6, 7, 9}) && o.done == sp::SpecDone::more, "outcome");
        o = sp::spec_outcome(in, g, 2, {});
        expect(o.kept == 2U && o.published.size() == 2U && o.done == sp::SpecDone::max_new, "max_new cut");
        o = sp::spec_outcome(in, g, 100, std::vector<std::uint32_t>{7});
        expect(o.kept == 2U && o.published == std::vector<std::uint32_t>({6, 7}) && o.done == sp::SpecDone::eos, "EOS cut");
        const std::vector<std::uint32_t> one{5}, g1{4};
        o = sp::spec_outcome(one, g1, 1, {});
        expect(o.kept == 1U && o.done == sp::SpecDone::max_new, "R1 window");
    }
    // ---- decision log
    {
        sp::SpecDecisionLog log;
        expect(log.classify(sp::spec_request(1, 1)) == sp::SpecDecisionLog::Kind::stale, "inactive log is stale");
        log.begin(1);
        log.record(0, {1});
        expect(log.classify(sp::spec_request(1, 1)) == sp::SpecDecisionLog::Kind::fresh, "next window fresh");
        expect(log.classify(sp::spec_request(1, 0)) == sp::SpecDecisionLog::Kind::duplicate, "same window duplicate");
        expect(log.classify(sp::spec_request(0, 1)) == sp::SpecDecisionLog::Kind::stale, "other epoch stale");
        expect(log.classify(sp::spec_request(1, 3)) == sp::SpecDecisionLog::Kind::stale, "future window stale");
    }
    // ---- sessions over a lossy channel == greedy
    const std::vector<Faults> faults{{}, {3, 0, 0, 0, 0}, {0, 2, 0, 0, 0}, {0, 0, 2, 0, 0}, {0, 0, 0, 3, 0}, {0, 0, 0, 0, 2}, {5, 4, 3, 7, 2}, {2, 2, 2, 2, 2}};
    std::uint32_t sessions = 0;
    for (std::uint32_t pi = 0; pi < 12U; ++pi) {
        std::vector<std::uint32_t> prompt;
        for (std::uint32_t i = 0; i < 5U + pi; ++i) prompt.push_back((pi * 31U + i * 7U) % 21U);
        const std::uint32_t max_new = 1U + pi * 11U;
        // greedy reference
        std::vector<std::uint32_t> ref, ctx = prompt;
        while (ref.size() < max_new) {
            const auto t = greedy(ctx);
            ref.push_back(t);
            if (t == 22U) break;
            ctx.push_back(t);
        }
        for (std::size_t fi = 0; fi < faults.size(); ++fi) {
            std::uint32_t rows = 0, resends = 0;
            bool same = false;
            const auto out = session(prompt, max_new, faults[fi], &rows, &resends, &same);
            expect(out == ref, "prompt " + std::to_string(pi) + " faults " + std::to_string(fi) + ": tokens == greedy (" + std::to_string(out.size()) + " vs " +
                                   std::to_string(ref.size()) + ")");
            expect(same, "prompt " + std::to_string(pi) + " faults " + std::to_string(fi) + ": same committed context");
            ++sessions;
        }
    }
    std::printf("ds41 pipeline dspark: %u sessions over lossy channels, %s\n", sessions, failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
