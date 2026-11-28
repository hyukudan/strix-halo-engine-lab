// Device-sample hook of the shared multi-request decoder (CohortBackend::samples_on_device / sample_final), on the host.
// Invariants:
// - A backend without the hook gets the same calls at the same points as one with it, read_final_logits where the
//   other gets sample_final, and never sample_final itself; ids and sampling receipts are equal when the device
//   decision equals the host one.
// - With the hook, a request's sampled ids are its solo run's in a 4-lane cohort (C1 == C4), on the prefill-emits-first
//   site and the decode site.
// - Replicated (collective_reads): rank 1's backend receives the same sample_final calls (lane, position, sampler
//   options) in the same order relative to every other call; without collective_reads it receives none.
// - Round forms (CohortBackend::samples_in_round / sample_round, propose_round): a backend that decides a step's sampled
//   lanes in ONE sample_round call gets the same decisions (lane, position, sampler, id) in the same order as one that
//   gets sample_final per lane, at the same point of the call sequence; ids and receipts are equal. propose_round comes
//   once per round, before the round's propose calls, listing exactly those lanes. Replicated with collective_reads,
//   rank 1 receives the same round calls in the same order.
// - Sampled drafts (CohortConfig::sampled_drafts, coupled acceptance): a sampled lane that verifies device drafts emits
//   the ids and receipt of its run without drafts (host and device sampling, both prefill sites, C1 == C4), decides
//   no row after a rejected draft, and drafts nothing under rejection acceptance; replicated with collective_reads,
//   rank 1 gets the same window-row calls in the same order.
#include "strix/serve/replicated_cohort.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace strix::serve;
using strix::TokenSampler;

namespace {
int failures = 0;
void check(bool value, const std::string& label) {
    if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}

constexpr std::uint32_t kVocab = 23;

std::uint32_t next_id(const std::vector<std::uint32_t>& ctx) {
    const std::uint32_t a = ctx.empty() ? 0U : ctx.back();
    return (a * 5U + 1U + static_cast<std::uint32_t>(ctx.size() % 3U)) % kVocab;
}
std::vector<float> logits_of(const std::vector<std::uint32_t>& ctx) {
    std::vector<float> l(kVocab);
    const std::uint32_t a = ctx.empty() ? 0U : ctx.back();
    const auto n = static_cast<std::uint32_t>(ctx.size());
    for (std::uint32_t i = 0; i < kVocab; ++i) l[i] = -0.25F * static_cast<float>((i * 7U + a * 3U + n) % 11U);
    return l;
}
// The device path through the sampler's public device API: decide(logits, position, nullptr) computed elsewhere.
std::uint32_t device_decide(TokenSampler& s, std::span<const float> logits, std::uint64_t position) {
    const auto p = strix::make_token_distribution(logits, {}, s.policy());
    strix::TokenDecision d;
    d.token = p.sample(s.uniforms(position, false).plain);
    s.count(d);
    return d.token;
}

class Backend final : public CohortBackend {
public:
    Backend(bool device, bool emits_first) : device_(device), emits_first_(emits_first), ctx_(4), window_(4), final_(4) {}
    std::uint32_t lanes() const override { return 4; }
    std::uint32_t rows_per_forward() const override { return 8; }
    std::uint32_t max_positions() const override { return 4096; }
    bool prefill_emits_first() const override { return emits_first_; }
    bool supports_final_logits() const override { return !device_; }
    bool samples_on_device() const override { return device_; }
    bool samples_in_round() const override { return round_; }
    bool device_drafts() const override { return round_ || drafts_; }
    bool reads_window_rows() const override { return drafts_; }
    std::span<const float> read_row_logits(std::uint32_t lane, std::uint32_t row) override {
        if (device_) throw std::logic_error("device backend read on the host");
        trace.push_back("W" + std::to_string(lane) + "/" + std::to_string(row));
        logits_ = logits_of(row_context(lane, row));
        return logits_;
    }
    std::uint32_t sample_row(std::uint32_t lane, std::uint32_t row, TokenSampler& sampler,
                             std::uint64_t position) override {
        if (!device_) throw std::logic_error("sample_row on a host backend");
        trace.push_back("W" + std::to_string(lane) + "/" + std::to_string(row));
        return decide_on(lane, row_context(lane, row), sampler, position);
    }
    void propose_round(std::span<const CohortProposal> lanes) override {
        if (!round_ && !drafts_) throw std::logic_error("propose_round without device drafts");
        if (!asked_.empty()) throw std::logic_error("propose_round before the last round's lanes were asked");
        std::string e = "Q";
        for (const auto& p : lanes) e += std::to_string(p.lane) + "/" + std::to_string(p.room) + ",";
        asked_.assign(lanes.begin(), lanes.end());
        trace.push_back(e);
    }
    void propose(std::uint32_t lane, std::uint32_t room, std::vector<std::uint32_t>& tokens,
                 std::vector<double>& probability) override {
        if (asked_.empty() || asked_.front().lane != lane || asked_.front().room != room)
            throw std::logic_error("propose outside its round, or another lane / room than the round listed");
        asked_.erase(asked_.begin());
        tokens.clear();
        probability.clear();
        trace.push_back("D" + std::to_string(lane));
        if (drafts_) {
            // the argmax chain from the lane's committed rows (its pending id is the argmax under top_k = 1)
            auto c = ctx_.at(lane);
            c.push_back(argmax(logits_of(c)));
            for (std::uint32_t r = 0; r < room; ++r) {
                tokens.push_back(argmax(logits_of(c)));
                c.push_back(tokens.back());
            }
        }
    }
    void sample_round(std::span<CohortSample> lanes) override {
        if (!round_) throw std::logic_error("sample_round without the round trait");
        std::string e = "S";
        for (auto& s : lanes) {
            s.token = decide(s.lane, *s.sampler, s.position);
            e += std::to_string(s.lane) + ",";
        }
        trace.push_back(e);
        ++round_calls;
    }
    void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override {
        if (first) ctx_.at(lane).clear();
        ctx_[lane].insert(ctx_[lane].end(), tokens.begin(), tokens.end());
        final_[lane] = ctx_[lane];
        trace.push_back("P" + std::to_string(lane) + ":" + std::to_string(tokens.size()));
    }
    std::uint32_t prefill_top(std::uint32_t lane) override { return next_id(ctx_.at(lane)); }
    void verify(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                std::span<std::uint32_t> top) override {
        if (!asked_.empty()) throw std::logic_error("verify before every lane of the round proposal was asked");
        std::size_t at = 0;
        std::string e = "V";
        groups_.assign(groups.begin(), groups.end());
        // every request samples: host and device sampling both read the final scores of every group
        for (const auto& g : groups) unread_groups += g.reads_scores ? 0U : 1U;
        for (const auto& g : groups) {
            window_[g.lane] = ctx_[g.lane];
            for (std::uint32_t r = 0; r < g.rows; ++r, ++at) {
                window_[g.lane].push_back(tokens[at]);
                top[at] = next_id(window_[g.lane]);
            }
            final_[g.lane] = window_[g.lane];
            e += std::to_string(g.lane) + "x" + std::to_string(g.rows) + ",";
        }
        trace.push_back(e);
    }
    std::span<const float> read_final_logits(std::uint32_t lane) override {
        if (device_) throw std::logic_error("device backend read on the host");
        trace.push_back("F" + std::to_string(lane));
        logits_ = logits_of(final_.at(lane));
        return logits_;
    }
    std::uint32_t sample_final(std::uint32_t lane, TokenSampler& sampler, std::uint64_t position) override {
        if (!device_) throw std::logic_error("sample_final on a host backend");
        trace.push_back("F" + std::to_string(lane));
        return decide(lane, sampler, position);
    }
    void commit(std::span<const std::uint32_t> accept) override {
        std::string e = "C";
        for (std::size_t g = 0; g < accept.size(); ++g) {
            const auto lane = groups_.at(g).lane;
            window_[lane].resize(ctx_[lane].size() + accept[g]);
            ctx_[lane] = window_[lane];
            e += std::to_string(accept[g]) + ",";
        }
        trace.push_back(e);
    }
    void release(std::uint32_t lane) override { trace.push_back("R" + std::to_string(lane)); }

    std::vector<std::string> trace, samples;
    unsigned unread_groups = 0, round_calls = 0;
    bool round_ = false, drafts_ = false;

private:
    static std::uint32_t argmax(const std::vector<float>& l) {
        return static_cast<std::uint32_t>(std::ranges::max_element(l) - l.begin());
    }
    // row r of the lane's pending window reads the committed rows and the window's rows 0..r
    std::vector<std::uint32_t> row_context(std::uint32_t lane, std::uint32_t row) const {
        const auto& w = window_.at(lane);
        const std::size_t n = ctx_.at(lane).size() + row + 1U;
        if (n > w.size()) throw std::logic_error("a window row beyond the lane's group");
        return {w.begin(), w.begin() + static_cast<std::ptrdiff_t>(n)};
    }
    std::uint32_t decide(std::uint32_t lane, TokenSampler& sampler, std::uint64_t position) {
        return decide_on(lane, final_.at(lane), sampler, position);
    }
    std::uint32_t decide_on(std::uint32_t lane, const std::vector<std::uint32_t>& ctx, TokenSampler& sampler,
                            std::uint64_t position) {
        const auto l = logits_of(ctx);
        const std::uint32_t id = device_decide(sampler, l, position);
        samples.push_back(std::to_string(lane) + ":" + std::to_string(position) + ":" +
                          std::to_string(sampler.policy().seed) + ":" +
                          std::to_string(sampler.options().request_identity) + ":" + std::to_string(id));
        return id;
    }
    std::vector<CohortProposal> asked_;
    bool device_, emits_first_;
    std::vector<std::vector<std::uint32_t>> ctx_, window_, final_;
    std::vector<CohortGroup> groups_;
    std::vector<float> logits_;
};

CohortRequest request(std::uint64_t id, std::uint32_t seed, std::size_t prompt, std::uint32_t n) {
    CohortRequest r;
    r.id = id;
    r.prompt.resize(prompt);
    for (std::size_t i = 0; i < prompt; ++i) r.prompt[i] = static_cast<std::uint32_t>((seed + 3U * i) % kVocab);
    r.max_tokens = n;
    r.sampling.temperature = 0.8F;
    r.sampling.seed = 1000U + seed;
    r.sampler_options.request_identity = 77U + seed;
    return r;
}
std::vector<CohortRequest> requests() {
    return {request(1, 1, 5, 9), request(2, 2, 3, 12), request(3, 3, 7, 6), request(4, 4, 4, 10)};
}

std::vector<CohortResult> run(CohortBackend& be, const std::vector<CohortRequest>& reqs, bool sampled_drafts = true) {
    CohortConfig config;
    config.sampled_drafts = sampled_drafts;
    CohortDecoder d(be, config);
    d.set_fake_clock(1.0);
    for (const auto& r : reqs) d.submit(r);
    d.run();
    auto out = d.take_finished();
    std::ranges::sort(out, [](const CohortResult& a, const CohortResult& b) { return a.id < b.id; });
    return out;
}

void same_results(const std::vector<CohortResult>& a, const std::vector<CohortResult>& b, const std::string& n) {
    check(a.size() == b.size(), n + ": result count");
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        const std::string k = n + " request " + std::to_string(a[i].id);
        check(a[i].error.empty() && b[i].error.empty(), k + ": served (" + a[i].error + b[i].error + ")");
        check(!a[i].ids.empty() && a[i].ids == b[i].ids, k + ": ids");
        check(!a[i].sampling_receipt.empty() && a[i].sampling_receipt == b[i].sampling_receipt, k + ": receipt");
    }
}

// (1) and (4): the host backend's call trace is the device backend's with the read in place of the device decision.
void host_trace_unchanged(bool emits_first) {
    const std::string n = std::string("host vs device, emits_first ") + (emits_first ? "1" : "0");
    Backend host(false, emits_first), dev(true, emits_first);
    const auto a = run(host, requests());
    const auto b = run(dev, requests());
    same_results(a, b, n);
    check(host.trace == dev.trace, n + ": same calls at the same points");
    check(host.samples.empty(), n + ": the host backend never gets sample_final");
    check(!dev.samples.empty(), n + ": the device backend decides the sampled ids");
    check(host.unread_groups == 0U && dev.unread_groups == 0U, n + ": host and device sampled lanes declare score reads");
}

// (2): C1 == C4 with the hook.
void solo_equals_cohort(bool emits_first) {
    const std::string n = std::string("C1 == C4, emits_first ") + (emits_first ? "1" : "0");
    Backend four(true, emits_first);
    const auto together = run(four, requests());
    std::vector<CohortResult> alone;
    for (const auto& r : requests()) {
        Backend one(true, emits_first);
        alone.push_back(run(one, {r}).front());
    }
    same_results(alone, together, n);
}

// Round forms: the requests plus one greedy request (a lane the round proposal asks).
std::vector<CohortRequest> round_requests() {
    auto out = requests();
    auto greedy = request(5, 5, 6, 8);
    greedy.sampling = {};
    out.push_back(greedy);
    return out;
}
void same_ids(const std::vector<CohortResult>& a, const std::vector<CohortResult>& b, const std::string& n) {
    check(a.size() == b.size(), n + ": result count");
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        const std::string k = n + " request " + std::to_string(a[i].id);
        check(a[i].error.empty() && b[i].error.empty(), k + ": served (" + a[i].error + b[i].error + ")");
        check(!a[i].ids.empty() && a[i].ids == b[i].ids, k + ": ids");
        check(a[i].sampling_receipt == b[i].sampling_receipt, k + ": receipt");
    }
}
// A round trace with each S entry spelled as the per-lane F entries and the propose calls (Q, D) left out.
std::vector<std::string> per_lane_form(const std::vector<std::string>& trace) {
    std::vector<std::string> out;
    for (const auto& e : trace) {
        if (e.empty() || e[0] == 'Q' || e[0] == 'D') continue;
        if (e[0] != 'S') {
            out.push_back(e);
            continue;
        }
        for (std::size_t at = 1; at < e.size();) {
            const auto comma = e.find(',', at);
            out.push_back("F" + e.substr(at, comma - at));
            at = comma + 1U;
        }
    }
    return out;
}
void round_forms(bool emits_first) {
    const std::string n = std::string("round forms, emits_first ") + (emits_first ? "1" : "0");
    Backend lane(true, emits_first), round(true, emits_first);
    round.round_ = true;
    const auto a = run(lane, round_requests());
    const auto b = run(round, round_requests());
    same_ids(a, b, n);
    check(round.round_calls > 0U, n + ": sampled lanes were decided in round calls");
    check(lane.samples == round.samples, n + ": the same decisions (lane, position, sampler, id) in the same order");
    check(per_lane_form(round.trace) == lane.trace, n + ": at the same point of the call sequence");
    std::size_t rounds = 0;
    for (const auto& e : round.trace) rounds += e[0] == 'Q' ? 1U : 0U;
    check(rounds > 0U, n + ": the greedy lane was asked through a round proposal");
}

// Sampled drafts: coupled requests (half of them top_k = 1, so the argmax drafts are kept) and the same set under
// rejection acceptance.
std::vector<CohortRequest> drafted_requests(bool coupled) {
    auto out = requests();
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i].sampler_options.acceptance =
            coupled ? strix::SampledAcceptance::coupled : strix::SampledAcceptance::rejection;
        if (i % 2U == 0U) out[i].sampling.top_k = 1U;
        out[i].max_tokens += 8U;
    }
    return out;
}
std::uint64_t sum_of(const std::vector<CohortResult>& rs, std::uint64_t CohortResult::*field) {
    std::uint64_t n = 0;
    for (const auto& r : rs) n += r.*field;
    return n;
}
void sampled_drafts(bool device, bool emits_first) {
    const std::string n = std::string("sampled drafts, device ") + (device ? "1" : "0") + ", emits_first " +
                          (emits_first ? "1" : "0");
    Backend off(device, emits_first), on(device, emits_first);
    off.drafts_ = on.drafts_ = true;
    const auto plain = run(off, drafted_requests(true), false);
    const auto drafted = run(on, drafted_requests(true));
    same_results(plain, drafted, n);
    check(sum_of(plain, &CohortResult::drafted) == 0U, n + ": the off switch drafts nothing");
    check(sum_of(drafted, &CohortResult::drafted) > 0U && sum_of(drafted, &CohortResult::accepted) > 0U,
          n + ": sampled lanes verified and kept device drafts");
    if (device) check(on.samples.size() == off.samples.size(), n + ": no row decided beyond the run without drafts");
    // C1 == C4 with drafts
    std::vector<CohortResult> alone;
    for (const auto& r : drafted_requests(true)) {
        Backend one(device, emits_first);
        one.drafts_ = true;
        alone.push_back(run(one, {r}).front());
    }
    same_results(alone, drafted, n + " C1 == C4");
    // rejection acceptance keeps one row per step
    Backend rej(device, emits_first);
    rej.drafts_ = true;
    const auto rejected = run(rej, drafted_requests(false));
    check(sum_of(rejected, &CohortResult::drafted) == 0U, n + ": rejection acceptance drafts nothing");
}

// In-memory two-rank transport: whole messages, in order, per direction.
struct Wire {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::vector<std::byte>> q[2];
    bool aborted = false;
};
class QueueTransport final : public strix::dist::Transport {
public:
    QueueTransport(std::shared_ptr<Wire> wire, int rank) : wire_(std::move(wire)), rank_(rank) {}
    void send(std::span<const std::byte> message) override {
        std::lock_guard lock(wire_->m);
        if (wire_->aborted) throw std::runtime_error("link aborted");
        wire_->q[1 - rank_].emplace_back(message.begin(), message.end());
        wire_->cv.notify_all();
    }
    void receive(std::span<std::byte> message) override {
        std::unique_lock lock(wire_->m);
        wire_->cv.wait(lock, [&] { return wire_->aborted || !wire_->q[rank_].empty(); });
        if (wire_->q[rank_].empty()) throw std::runtime_error("link aborted");
        auto got = std::move(wire_->q[rank_].front());
        wire_->q[rank_].pop_front();
        if (got.size() != message.size()) throw std::runtime_error("message size mismatch");
        std::memcpy(message.data(), got.data(), got.size());
    }
    void exchange(std::span<const std::byte> out, std::span<std::byte> in) override {
        send(out);
        receive(in);
    }
    std::string name() const override { return "test-queue"; }
    bool supports_peer_abort() const noexcept override { return true; }
    void abort(const std::string&) override {
        std::lock_guard lock(wire_->m);
        wire_->aborted = true;
        wire_->cv.notify_all();
    }

private:
    std::shared_ptr<Wire> wire_;
    int rank_;
};

// (3): rank 1 replays sample_final in rank 0's order (collective_reads), or never sees it.
void replicated(bool collective, bool emits_first, bool in_round = false, bool drafts = false) {
    const std::string n = std::string("replicated, collective ") + (collective ? "1" : "0") + ", emits_first " +
                          (emits_first ? "1" : "0") + (in_round ? ", round forms" : "") + (drafts ? ", drafts" : "");
    auto wire = std::make_shared<Wire>();
    QueueTransport t0(wire, 0), t1(wire, 1);
    Backend be0(true, emits_first), be1(true, emits_first);
    be0.round_ = be1.round_ = in_round;
    be0.drafts_ = be1.drafts_ = drafts;
    const auto reqs = drafts ? drafted_requests(true) : in_round ? round_requests() : requests();
    ReplicaOptions options;
    options.collective_reads = collective;
    std::string error;
    std::thread follower([&] {
        try {
            (void)follow_cohort(be1, t1, options);
        } catch (const std::exception& e) {
            error = e.what();
        }
    });
    std::vector<CohortResult> res;
    {
        ReplicatedCohort rep(be0, t0, options);
        check(rep.samples_on_device(), n + ": device sampling is a replicated trait");
        res = run(rep, reqs);
        rep.close();
    }
    follower.join();
    check(error.empty(), n + ": rank 1 finished cleanly (" + error + ")");
    Backend solo(true, emits_first);
    solo.round_ = in_round;
    solo.drafts_ = drafts;
    if (in_round) same_ids(run(solo, reqs), res, n);
    else same_results(run(solo, reqs), res, n);
    if (collective) {
        check(be1.samples == be0.samples, n + ": rank 1 gets the same sample_final calls, options and ids");
        check(be1.trace == be0.trace, n + ": in the same order relative to every other call");
    } else {
        check(be1.samples.empty(), n + ": rank 1 never sees a rank-0 sample query");
        if (in_round) check(be1.round_calls == 0U, n + ": nor a rank-0 round sample");
    }
}
}  // namespace

int main() {
    try {
        for (const bool emits_first : {false, true}) {
            host_trace_unchanged(emits_first);
            solo_equals_cohort(emits_first);
            replicated(true, emits_first);
            replicated(false, emits_first);
            round_forms(emits_first);
            replicated(true, emits_first, true);
            replicated(false, emits_first, true);
            sampled_drafts(true, emits_first);
            sampled_drafts(false, emits_first);
            replicated(true, emits_first, false, true);
            replicated(false, emits_first, false, true);
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL: exception " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) return 1;
    std::cout << "cohort sample hook tests: PASS\n";
    return 0;
}
