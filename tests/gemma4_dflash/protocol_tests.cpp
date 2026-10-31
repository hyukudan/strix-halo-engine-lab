#include "fixture.hpp"

#include <iostream>
#include <numeric>

namespace g4 = strix::gemma4;
namespace df = g4::dflash;
using fixture::check;

struct FakeTxn final : g4::KvTxn {
    std::uint64_t position, digest;
    std::vector<std::uint32_t> pending, committed;
    std::size_t commits = 0;
    explicit FakeTxn(std::uint64_t p) : position(p), digest(17 + p) {}
    std::uint64_t frontier() const override { return position; }
    void commit(std::size_t count) override {
        check(count > 0 && count <= pending.size(), "bad prefix commit");
        for (std::size_t i = 0; i < count; ++i) {
            digest = digest * 257 + pending[i] + position;
            committed.push_back(pending[i]);
            ++position;
        }
        pending.clear();
        ++commits;
    }
};

struct FakeTarget final : g4::Target {
    std::size_t calls = 0;
    std::uint64_t stream = 0, generation = 0;
    g4::ForwardResult forward_rows(std::span<const g4::RowPlan> rows) override {
        ++calls;
        stream = rows.front().stream; generation = rows.front().generation;
        auto& txn = dynamic_cast<FakeTxn&>(*rows.front().transaction);
        check(txn.pending.empty(), "old overlay was not discarded");
        g4::ForwardResult out;
        out.rows = rows.size(); out.vocab = 128; out.tap_width = 2;
        out.logits.assign(rows.size() * out.vocab, -10.0F);
        out.taps.resize(rows.size() * 2);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto& r = rows[i];
            check(r.active && r.output_row == i && r.position == txn.position + i && r.transaction == &txn,
                  "bad row plan");
            txn.pending.push_back(r.token);
            const auto next = (r.token + 1) % out.vocab;
            out.logits[i * out.vocab + next] = 5.0F;
            out.taps[2 * i] = static_cast<float>(r.position);
            out.taps[2 * i + 1] = static_cast<float>(r.token);
        }
        return out;
    }
};

struct Run { std::vector<std::uint32_t> emitted, processed; std::uint64_t digest = 0, frontier = 0; };

Run generate(std::uint32_t k, int reject, std::uint64_t start, std::size_t budget,
             std::span<const std::uint32_t> eos, bool off = false) {
    FakeTarget target;
    FakeTxn txn(start);
    df::Acceptance stats;
    Run run;
    std::uint32_t anchor = 10;
    std::size_t draft_calls = 0, drafted_cycles = 0;
    while (run.emitted.size() < budget && txn.frontier() < df::position_limit) {
        auto draft = [&] {
            ++draft_calls;
            std::array<std::uint32_t, df::max_drafts> tokens{};
            for (std::uint32_t d = 0; d < df::max_drafts; ++d) tokens[d] = (anchor + d + 1) % 128;
            if (reject >= 0 && reject < static_cast<int>(k)) tokens[static_cast<std::size_t>(reject)] = 100;
            return tokens;
        };
        const auto result = df::run_cycle(target, txn, anchor, k, draft, budget - run.emitted.size(), eos, &stats);
        const auto& v = result.verdict;
        check(v.processed == v.emitted.size() && result.committed_positions.size() == v.processed, "truncated commit count");
        check(result.committed_taps.size() == v.processed * 2, "rejected taps leaked");
        for (std::size_t i = 0; i < v.processed; ++i) {
            check(result.committed_positions[i] == txn.frontier() - v.processed + i, "tap absolute position");
            check(result.committed_taps[2 * i + 1] == static_cast<float>(txn.committed[txn.committed.size() - v.processed + i]),
                  "tap/input alignment");
        }
        if (v.eligible) ++drafted_cycles;
        run.emitted.insert(run.emitted.end(), v.emitted.begin(), v.emitted.end());
        anchor = result.next_anchor;
        if (v.stopped) break;
    }
    check(draft_calls == drafted_cycles, "k=0 or truncated admission ran drafting");
    if (off) check(draft_calls == 0, "OFF switch ran drafting");
    check(stats.emitted == run.emitted.size() && stats.eligible >= stats.accepted && stats.attempted >= stats.eligible,
          "acceptance counters");
    const auto eligible = std::accumulate(stats.depth_eligible.begin(), stats.depth_eligible.end(), std::uint64_t{0});
    const auto accepted = std::accumulate(stats.depth_accepted.begin(), stats.depth_accepted.end(), std::uint64_t{0});
    check(eligible == stats.eligible && accepted == stats.accepted, "per-depth survival counters");
    run.processed = txn.committed; run.digest = txn.digest; run.frontier = txn.frontier();
    return run;
}

void equal(const Run& a, const Run& b) {
    check(a.emitted == b.emitted, "speculative tokens differ from serial greedy");
    check(a.processed == b.processed && a.digest == b.digest && a.frontier == b.frontier,
          "speculative committed state differs from serial");
}

int main(int argc, char** argv) {
    try {
        const bool off = argc > 1 && std::string(argv[1]) == "--off";
        for (std::uint32_t k = 0; k <= df::max_drafts; ++k)
            for (int reject = -1; reject < static_cast<int>(k); ++reject)
                for (std::uint64_t start : {0ULL, 508ULL, 511ULL, 512ULL, 516ULL, 1020ULL, 1023ULL, 1024ULL, 1028ULL, 65528ULL, 65529ULL, 65535ULL})
                    for (std::size_t budget = 1; budget <= 25; ++budget) {
                        equal(generate(k, reject, start, budget, {}, off), generate(0, -1, start, budget, {}, off));
                        for (std::uint32_t depth = 0; depth < df::block; ++depth) {
                            const std::array<std::uint32_t, 2> eos{11 + depth, 127};
                            equal(generate(k, reject, start, budget, eos, off), generate(0, -1, start, budget, eos, off));
                        }
                    }
        FakeTarget target; FakeTxn txn(0); std::size_t calls = 0;
        const auto result = df::run_cycle(target, txn, 10, 7, [&] { ++calls; return std::array<std::uint32_t, 7>{}; }, 0);
        check(result.verdict.emitted.empty() && calls == 0 && target.calls == 0 && txn.commits == 0, "empty budget mutated state");
        (void)df::run_cycle(target, txn, 10, 0, {}, 1, {}, nullptr, 42, 7);
        check(target.stream == 42 && target.generation == 7, "stream/generation identity was lost");
        check(df::admit_window(7, 0, 20, true, false, 0, true) == 0, "auto without evidence");
        check(df::admit_window(7, 0, 20, true, false, 1.049, true) == 0, "auto below 5 percent");
        check(df::admit_window(7, 0, 20, true, false, 1.05, true) == 7, "measured auto admission");
        check(df::admit_window(7, 0, 20, true, true) == 0, "policy did not force serial");
        check(df::admit_window(7, 65529, 20, true, false) == 0, "full-block position reservation");
        check(df::admit_window(7, 0, 2, true, false) == 1, "budget admission");
        fixture::rejects([&] { (void)df::verify_greedy(std::array<std::uint32_t, 1>{1}, std::array<std::uint32_t, 1>{1}, 10); });
        const std::array<std::uint32_t, 3> drafts{1, 2, 3};
        const std::array<std::uint32_t, 4> greedy{1, 2, 3, 4};
        df::Acceptance stats;
        const auto truncated = df::verify_greedy(drafts, greedy, 2, {}, &stats);
        check(truncated.accepted == 2 && truncated.processed == 2 && stats.eligible == 2 && stats.attempted == 3,
              "truncation inspected ineligible depths");
        if (!off) {
            df::Acceptance all;
            const auto v = df::verify_greedy(drafts, greedy, 10, {}, &all);
            check(v.emitted == std::vector<std::uint32_t>({1, 2, 3, 4}) && all.all_accepted == 1 && all.accepted == 3,
                  "all-accepted bonus/counters");
        }
        std::cout << "DFlash protocol matches serial tokens and committed state\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
