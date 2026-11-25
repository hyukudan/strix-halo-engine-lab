#include "strix/serve/pair_sampling.hpp"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <thread>

namespace sv = strix::serve;

namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void test_selection() {
    sv::GenerationJob job;
    job.sampling = {0.0, 0.98, 4, 0.05, 33, true};
    const std::array<float, 2> local{1.0F, 1.1F};
    std::vector<float> remote(16386, -1000.0F);
    remote[2] = 1.3F;
    remote[16384] = 1.2F;
    remote.back() = 100.0F;  // beyond the tokenizer's vocabulary
    const std::size_t vocab = local.size() + remote.size() - 1;
    std::vector<float> full(local.begin(), local.end());
    full.insert(full.end(), remote.begin(), remote.end() - 1);
    const auto greedy = static_cast<std::uint32_t>(std::max_element(full.begin(), full.end()) - full.begin());
    std::array<std::uint32_t, 2> ids{greedy, greedy};
    const auto no_read = [](std::size_t) -> std::span<const float> {
        throw std::runtime_error("greedy route read logits");
    };
    const auto no_exchange = [](std::span<const float>, std::span<float>) {
        throw std::runtime_error("greedy route exchanged logits");
    };
    check(!sv::make_pair_sampler(job), "temperature zero must retain the engine's greedy choice");
    sv::sample_pair_rows(0, true, remote.size(), vocab, {}, ids, no_read, no_exchange);
    check(ids[0] == greedy && ids[1] == greedy, "temperature zero changed the engine's greedy IDs");

    job.sampling.temperature = 0.8;
    auto first = sv::make_pair_sampler(job);
    auto repeat = sv::make_pair_sampler(job);
    ++job.sampling.seed;
    auto different = sv::make_pair_sampler(job);
    check(first && repeat && different, "positive temperature must construct a shared sampler");
    const auto& p = first->policy();
    check(p.temperature == 0.8F && p.top_k == 4 && p.top_p == 0.98F && p.min_p == 0.05F && p.seed == 33,
          "the adapter must forward every sampling parameter");

    std::vector<std::vector<float>> packets;
    sv::sample_pair_rows(2, false, remote.size(), 0, {}, {},
        [&](std::size_t row) { check(row == 1, "follower read a greedy row"); return std::span(remote); },
        [&](auto send, auto receive) {
            packets.emplace_back(send.begin(), send.end());
            std::fill(receive.begin(), receive.end(), 0.0F);
        });
    check(packets.size() == 2 && packets[0].size() == 16384 && packets[1].size() == 2,
          "follower must exchange its complete slice in bounded frames");

    const auto run = [&](strix::TokenSampler& sampler, std::uint64_t position) {
        std::size_t packet = 0;
        const std::array<sv::PairSamplingRow, 2> choices{{{}, {&sampler, position}}};
        sv::sample_pair_rows(2, true, remote.size(), vocab, choices, ids,
            [&](std::size_t row) { check(row == 1, "lead read a greedy row"); return std::span(local); },
            [&](auto send, auto receive) {
                const auto& wire = packets.at(packet++);
                check(wire.size() == send.size(), "ranks disagree on frame width");
                std::copy(wire.begin(), wire.end(), receive.begin());
            });
        check(ids[0] == greedy && packet == packets.size(), "sampled row changed a greedy cohort member");
        check(ids[1] < vocab, "sampling selected padding");
        return ids[1];
    };
    strix::TokenSampler oracle(p, {});
    bool diverged = false;
    for (std::uint64_t pos = 32; pos < 64; ++pos) {
        const auto token = run(*first, pos);
        check(token == oracle.decide(full, pos, nullptr).token, "the lead did not sample the assembled row at its absolute position");
        check(token == run(*repeat, pos), "identical seeds must reproduce the same sequence");
        diverged |= token != run(*different, pos);
    }
    check(diverged, "different seeds must diverge on this fixture");
    check(!first->coupled(), "the pair sampler keeps rejection acceptance by default");
    strix::TokenSamplerOptions coupled_options;
    coupled_options.acceptance = strix::SampledAcceptance::coupled;
    const auto coupled = sv::make_pair_sampler(job, coupled_options);
    check(coupled && coupled->coupled(), "the pair sampler must forward the acceptance rule");
}

// In-process lock-step duplex: each side posts its message and takes the peer's (sizes must agree), as the control wire.
struct Duplex {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::vector<std::byte>> q[2];
    std::size_t float_messages[2]{}, word_messages[2]{};
    void xchg(int side, std::span<const std::byte> out, std::span<std::byte> in) {
        std::unique_lock<std::mutex> l(m);
        q[side].emplace_back(out.begin(), out.end());
        cv.notify_all();
        if (!cv.wait_for(l, std::chrono::seconds(10), [&] { return !q[1 - side].empty(); }))
            throw std::runtime_error("duplex: peer timeout");
        auto b = std::move(q[1 - side].front());
        q[1 - side].pop_front();
        if (b.size() != in.size()) throw std::runtime_error("duplex: message sizes differ");
        std::memcpy(in.data(), b.data(), b.size());
    }
    auto floats(int side) {
        return [this, side](std::span<const float> out, std::span<float> in) {
            ++float_messages[side];
            xchg(side, std::as_bytes(out), std::as_writable_bytes(in));
        };
    }
    auto words(int side) {
        return [this, side](std::span<const std::uint32_t> out, std::span<std::uint32_t> in) {
            ++word_messages[side];
            xchg(side, std::as_bytes(out), std::as_writable_bytes(in));
        };
    }
};

// Rows of a window: the lead's slice is [0, kLead), the follower's [kLead, kLead + kFollow); the vocabulary ends
// before the follower's padding.
constexpr std::size_t kLead = 3000, kFollow = 20000, kVocab = kLead + kFollow - 7;
struct WindowRows {
    std::vector<std::vector<float>> lead, follow, full;
    explicit WindowRows(unsigned rows, unsigned seed) {
        std::mt19937 g(seed);
        std::normal_distribution<float> n(0.0F, 3.0F);
        for (unsigned r = 0; r < rows; ++r) {
            std::vector<float> row(kLead + kFollow);
            for (auto& x : row) x = n(g);
            lead.emplace_back(row.begin(), row.begin() + kLead);
            follow.emplace_back(row.begin() + kLead, row.end());
            full.emplace_back(row.begin(), row.begin() + kVocab);
        }
    }
};

strix::TokenSamplingPolicy window_policy() {
    strix::TokenSamplingPolicy p;
    p.temperature = 0.9F;
    p.seed = 77;
    return p;
}
strix::TokenSamplerOptions window_options(strix::SampledAcceptance a) {
    strix::TokenSamplerOptions o;
    o.acceptance = a;
    return o;
}

struct WindowRun {
    std::vector<std::uint32_t> lead, follow;
    std::uint32_t lead_accepted = 0, follow_accepted = 0;
    std::size_t follower_rows = 0;
};
// Runs both ranks of one window on two threads.
WindowRun run_window(strix::TokenSampler& sampler, const WindowRows& rows, std::uint64_t first,
                     std::span<const strix::TokenDraft> drafts, std::uint32_t max_rows) {
    Duplex d;
    WindowRun out;
    std::exception_ptr error;
    std::thread follower([&] {
        try {
            auto w = sv::decide_pair_window(false, kFollow, 0, nullptr, 0, {}, max_rows,
                [&](std::size_t r) { ++out.follower_rows; return std::span<const float>(rows.follow.at(r)); },
                d.floats(1), d.words(1));
            out.follow = std::move(w.tokens);
            out.follow_accepted = w.accepted;
        } catch (...) {
            error = std::current_exception();
        }
    });
    try {
        auto w = sv::decide_pair_window(true, kFollow, kVocab, &sampler, first, drafts, max_rows,
            [&](std::size_t r) { return std::span<const float>(rows.lead.at(r)); }, d.floats(0), d.words(0));
        out.lead = std::move(w.tokens);
        out.lead_accepted = w.accepted;
    } catch (...) {
        follower.join();
        throw;
    }
    follower.join();
    if (error) std::rethrow_exception(error);
    return out;
}

void test_window_coupled() {
    const WindowRows rows(4, 5);
    const std::uint64_t first = 100;
    strix::TokenSampler oracle(window_policy(), window_options(strix::SampledAcceptance::coupled));
    std::vector<std::uint32_t> plain;
    for (unsigned r = 0; r < 4; ++r) plain.push_back(oracle.decide(rows.full[r], first + r, nullptr).token);
    for (unsigned wrong = 0; wrong <= 3; ++wrong) {  // the draft at this depth is wrong (3: none)
        std::vector<strix::TokenDraft> drafts;
        for (unsigned r = 0; r < 3; ++r) {
            strix::TokenDraft dr;
            dr.position = first + r;
            dr.token = r == wrong ? (plain[r] + 1U) % static_cast<std::uint32_t>(kVocab) : plain[r];
            drafts.push_back(std::move(dr));
        }
        strix::TokenSampler sampler(window_policy(), window_options(strix::SampledAcceptance::coupled));
        const auto run = run_window(sampler, rows, first, drafts, 4);
        const std::vector<std::uint32_t> expect(plain.begin(), plain.begin() + wrong + 1);
        check(run.lead == expect, "coupled window: the lead's tokens differ from plain sampling");
        check(run.follow == expect, "coupled window: the follower's tokens differ from the lead's");
        check(run.follower_rows == wrong + 1, "coupled window: the follower sent rows the lead never decided");
        check(run.lead_accepted == wrong && run.follow_accepted == wrong, "coupled window: accepted drafts not agreed");
        check(sampler.stats().accepted == wrong && sampler.stats().rejected == (wrong < 3 ? 1U : 0U),
              "coupled window: decisions not counted");
    }
}

void test_window_cuts() {
    const WindowRows rows(4, 6);
    const std::uint64_t first = 7;
    strix::TokenSampler oracle(window_policy(), window_options(strix::SampledAcceptance::coupled));
    std::vector<strix::TokenDraft> drafts;
    std::vector<std::uint32_t> plain;
    for (unsigned r = 0; r < 3; ++r) {
        plain.push_back(oracle.decide(rows.full[r], first + r, nullptr).token);
        strix::TokenDraft dr;
        dr.position = first + r;
        dr.token = plain.back();
        drafts.push_back(std::move(dr));
    }
    strix::TokenSampler sampler(window_policy(), window_options(strix::SampledAcceptance::coupled));
    const auto run = run_window(sampler, rows, first, drafts, 2);
    check(run.lead == std::vector<std::uint32_t>(plain.begin(), plain.begin() + 2) && run.follow == run.lead,
          "max_rows 2: exactly two rows decided on both ranks");
    check(run.follower_rows == 2, "max_rows 2: the follower sent two rows");
    bool refused = false;
    try {
        (void)sv::decide_pair_window(true, kFollow, kVocab, &sampler, first, drafts, 0,
            [](std::size_t) -> std::span<const float> { throw std::runtime_error("read"); },
            [](std::span<const float>, std::span<float>) { throw std::runtime_error("exchange"); },
            [](std::span<const std::uint32_t>, std::span<std::uint32_t>) { throw std::runtime_error("words"); });
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    check(refused, "max_rows 0 must be refused before any exchange");
}

void test_window_world1() {
    const WindowRows rows(4, 8);
    strix::TokenSampler oracle(window_policy(), window_options(strix::SampledAcceptance::coupled));
    strix::TokenSampler sampler(window_policy(), window_options(strix::SampledAcceptance::coupled));
    std::vector<strix::TokenDraft> drafts(1);
    drafts[0].position = 3;
    drafts[0].token = oracle.decide(rows.full[0], 3, nullptr).token;
    const auto expect1 = oracle.decide(rows.full[1], 4, nullptr).token;
    const auto tokens = sv::decide_pair_window(true, 0, kVocab, &sampler, 3, drafts, 2,
        [&](std::size_t r) { return std::span<const float>(rows.full.at(r)); },
        [](std::span<const float>, std::span<float>) { throw std::runtime_error("one node exchanged logits"); },
        [](std::span<const std::uint32_t>, std::span<std::uint32_t>) { throw std::runtime_error("one node exchanged words"); });
    check(tokens.tokens.size() == 2 && tokens.tokens[0] == drafts[0].token && tokens.tokens[1] == expect1 && tokens.accepted == 1,
          "one node: rows decided locally");
}

// A rejected row can keep a token equal to its draft (coupled acceptance keeps the plain sample; here the test-only
// forced rejection of row 0). The window must report 0 accepted drafts on both ranks, so a caller never commits the
// draft by token equality and then the next row's undecided token.
void test_window_same_token_rejection() {
    const WindowRows rows(4, 11);
    const std::uint64_t first = 30;
    strix::TokenSampler oracle(window_policy(), window_options(strix::SampledAcceptance::coupled));
    std::vector<strix::TokenDraft> drafts;
    for (unsigned r = 0; r < 3; ++r) {
        strix::TokenDraft dr;
        dr.position = first + r;
        dr.token = oracle.decide(rows.full[r], first + r, nullptr).token;
        drafts.push_back(std::move(dr));
    }
    auto options = window_options(strix::SampledAcceptance::coupled);
    options.force_reject_row = 0;
    strix::TokenSampler sampler(window_policy(), options);
    const auto run = run_window(sampler, rows, first, drafts, 4);
    check(run.lead.size() == 1 && run.lead[0] == drafts[0].token && run.follow == run.lead,
          "same-token rejection: one row decided, its token is the draft's");
    check(run.lead_accepted == 0 && run.follow_accepted == 0, "same-token rejection: no draft accepted on either rank");
    check(run.follower_rows == 1 && sampler.stats().rejected == 1 && sampler.stats().forced == 1,
          "same-token rejection: one row read, counted as a rejection");
}

// The follower holds the agreed max_rows: a lead that asks for one more row is refused before the follower reads it.
void test_window_follower_bound() {
    const WindowRows rows(4, 12);
    std::size_t reads = 0;
    bool refused = false;
    try {
        (void)sv::decide_pair_window(false, kFollow, 0, nullptr, 0, {}, 2,
            [&](std::size_t r) { ++reads; return std::span<const float>(rows.follow.at(r)); },
            [](std::span<const float>, std::span<float>) {},
            [](std::span<const std::uint32_t>, std::span<std::uint32_t> in) { in[0] = sv::kPairWindowMore; });
    } catch (const std::runtime_error&) {
        refused = true;
    }
    check(refused && reads == 2, "follower: MORE at max_rows must be refused without reading row max_rows");
    refused = false;
    try {
        (void)sv::decide_pair_window(false, kFollow, 0, nullptr, 0, {}, 9,
            [](std::size_t) -> std::span<const float> { throw std::logic_error("read"); },
            [](std::span<const float>, std::span<float>) {}, [](std::span<const std::uint32_t>, std::span<std::uint32_t>) {});
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    check(refused, "follower: max_rows above 8 refused before any read");
    // A DONE that claims every decided row accepted its draft before max_rows leaves the next row undecided: refused.
    // At max_rows the same claim is the protocol's all-accepted window.
    for (const std::uint32_t max_rows : {3U, 1U}) {
        refused = false;
        reads = 0;
        sv::PairWindowDecision got;
        try {
            got = sv::decide_pair_window(false, kFollow, 0, nullptr, 0, {}, max_rows,
                [&](std::size_t r) { ++reads; return std::span<const float>(rows.follow.at(r)); },
                [](std::span<const float>, std::span<float>) {},
                [](std::span<const std::uint32_t>, std::span<std::uint32_t> in) {
                    in[0] = sv::kPairWindowDone;
                    in[1] = 1;  // n
                    in[2] = 1;  // accepted
                    in[3] = 7;
                });
        } catch (const std::runtime_error&) {
            refused = true;
        }
        if (max_rows == 3U) check(refused && reads == 1, "follower: all-accepted DONE before max_rows refused");
        else check(!refused && got.accepted == 1 && got.tokens.size() == 1 && got.tokens[0] == 7,
                   "follower: all-accepted DONE at max_rows accepted");
    }
}

void test_window_rejection() {
    const WindowRows rows(4, 9), draft_rows(3, 10);
    const std::uint64_t first = 50;
    unsigned residual = 0;
    for (std::uint64_t seed = 1; seed <= 24; ++seed) {
        auto policy = window_policy();
        policy.seed = seed;
        strix::TokenSampler sampler(policy, window_options(strix::SampledAcceptance::rejection));
        strix::TokenSampler oracle(policy, window_options(strix::SampledAcceptance::rejection));
        std::vector<strix::TokenDraft> drafts;
        for (unsigned r = 0; r < 3; ++r) drafts.push_back(sampler.draft(draft_rows.full[r], {}, first + r));
        const auto want = oracle.decide_rows(4, first, drafts, {}, [&](std::uint32_t r) { return std::span<const float>(rows.full.at(r)); });
        const auto run = run_window(sampler, rows, first, drafts, 4);
        check(run.lead.size() == want.size() && run.follow == run.lead, "rejection window: row counts");
        for (std::size_t r = 0; r < want.size() && r < run.lead.size(); ++r)
            check(run.lead[r] == want[r].token, "rejection window: tokens differ from decide_rows");
        residual += static_cast<unsigned>(sampler.stats().rejected);
    }
    check(residual > 0, "rejection window: the fixture never rejected");
}

// Both ranks of one draft pick on two threads; returns {lead token, follower token}.
std::pair<std::uint32_t, std::uint32_t> run_draft(strix::TokenSampler& sampler, const WindowRows& rows, unsigned row,
                                                  std::uint64_t position, strix::TokenDraft* kept, std::size_t* exchanges) {
    Duplex d;
    std::uint32_t follower_token = 0;
    std::exception_ptr error;
    std::thread follower([&] {
        try {
            follower_token = sv::pick_pair_draft(false, kFollow, 0, nullptr, 0,
                [&] { return std::span<const float>(rows.follow.at(row)); }, d.floats(1), d.words(1));
        } catch (...) {
            error = std::current_exception();
        }
    });
    std::uint32_t lead_token = 0;
    try {
        lead_token = sv::pick_pair_draft(true, kFollow, kVocab, &sampler, position,
            [&] { return std::span<const float>(rows.lead.at(row)); }, d.floats(0), d.words(0), kept);
    } catch (...) {
        follower.join();
        throw;
    }
    follower.join();
    if (error) std::rethrow_exception(error);
    if (exchanges) *exchanges = d.float_messages[0] + d.word_messages[0];
    return {lead_token, follower_token};
}

void test_draft_pick() {
    const WindowRows rows(4, 11);
    for (const auto acceptance : {strix::SampledAcceptance::coupled, strix::SampledAcceptance::rejection}) {
        strix::TokenSampler sampler(window_policy(), window_options(acceptance));
        strix::TokenSampler oracle(window_policy(), window_options(acceptance));
        for (unsigned r = 0; r < 4; ++r) {
            strix::TokenDraft kept;
            std::size_t exchanges = 0;
            const auto [lead, follow] = run_draft(sampler, rows, r, 40 + r, &kept, &exchanges);
            const auto want = oracle.draft(rows.full[r], {}, 40 + r);
            check(lead == want.token && follow == lead, "draft pick: both ranks hold the oracle's draft");
            check(kept.token == lead && kept.position == 40 + r && kept.distribution.probability(lead) == want.distribution.probability(lead),
                  "draft pick: the lead keeps the draft's distribution");
            check(exchanges == 3, "draft pick: two slice frames and one decision message");
        }
        check(sampler.stats().drafts == 4, "draft pick: drafts counted");
    }
    // coupled: a draft drawn from the target's own row is the token that row decides (same uniform)
    strix::TokenSampler coupled(window_policy(), window_options(strix::SampledAcceptance::coupled));
    strix::TokenSampler plain(window_policy(), window_options(strix::SampledAcceptance::coupled));
    for (unsigned r = 0; r < 4; ++r)
        check(run_draft(coupled, rows, r, 60 + r, nullptr, nullptr).first == plain.decide(rows.full[r], 60 + r, nullptr).token,
              "coupled draft from the target row differs from the row's decision");
    // one node: no exchange
    strix::TokenSampler solo(window_policy(), window_options(strix::SampledAcceptance::coupled));
    const auto token = sv::pick_pair_draft(true, 0, kVocab, &solo, 9, [&] { return std::span<const float>(rows.full[0]); },
        [](std::span<const float>, std::span<float>) { throw std::runtime_error("one node exchanged logits"); },
        [](std::span<const std::uint32_t>, std::span<std::uint32_t>) { throw std::runtime_error("one node exchanged words"); });
    check(token == plain.decide(rows.full[0], 9, nullptr).token, "one node: draft drawn locally");
}
}  // namespace

int main() try {
    test_selection();
    test_window_coupled();
    test_window_cuts();
    test_window_world1();
    test_window_same_token_rejection();
    test_window_follower_bound();
    test_window_rejection();
    test_draft_pick();
    std::puts("pair draft: both ranks hold the lead's draft, distribution kept, coupled draft == row decision, one node local");
    std::puts("pair window: coupled == plain sampling on both ranks, cuts honored, one node local, rejection == decide_rows");
    std::puts("pair sampling: greedy IDs preserved, policy forwarded, same seeds matched, different seeds diverged");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
