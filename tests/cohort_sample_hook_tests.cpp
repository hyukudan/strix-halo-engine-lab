// Device-sample hook of the shared multi-request decoder (CohortBackend::samples_on_device / sample_final), on the host.
// Invariants:
// - A backend without the hook gets the same calls at the same points as one with it, read_final_logits where the
//   other gets sample_final, and never sample_final itself; ids and sampling receipts are equal when the device
//   decision equals the host one.
// - With the hook, a request's sampled ids are its solo run's in a 4-lane cohort (C1 == C4), on the prefill-emits-first
//   site and the decode site.
// - Replicated (collective_reads): rank 1's backend receives the same sample_final calls (lane, position, sampler
//   options) in the same order relative to every other call; without collective_reads it receives none.
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
    void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override {
        if (first) ctx_.at(lane).clear();
        ctx_[lane].insert(ctx_[lane].end(), tokens.begin(), tokens.end());
        final_[lane] = ctx_[lane];
        trace.push_back("P" + std::to_string(lane) + ":" + std::to_string(tokens.size()));
    }
    std::uint32_t prefill_top(std::uint32_t lane) override { return next_id(ctx_.at(lane)); }
    void verify(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                std::span<std::uint32_t> top) override {
        std::size_t at = 0;
        std::string e = "V";
        groups_.assign(groups.begin(), groups.end());
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
        const auto l = logits_of(final_.at(lane));
        const std::uint32_t id = device_decide(sampler, l, position);
        samples.push_back(std::to_string(lane) + ":" + std::to_string(position) + ":" +
                          std::to_string(sampler.policy().seed) + ":" +
                          std::to_string(sampler.options().request_identity) + ":" + std::to_string(id));
        return id;
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

private:
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

std::vector<CohortResult> run(CohortBackend& be, const std::vector<CohortRequest>& reqs) {
    CohortDecoder d(be, CohortConfig{});
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
void replicated(bool collective, bool emits_first) {
    const std::string n = std::string("replicated, collective ") + (collective ? "1" : "0") + ", emits_first " +
                          (emits_first ? "1" : "0");
    auto wire = std::make_shared<Wire>();
    QueueTransport t0(wire, 0), t1(wire, 1);
    Backend be0(true, emits_first), be1(true, emits_first);
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
        res = run(rep, requests());
        rep.close();
    }
    follower.join();
    check(error.empty(), n + ": rank 1 finished cleanly (" + error + ")");
    Backend solo(true, emits_first);
    same_results(run(solo, requests()), res, n);
    if (collective) {
        check(be1.samples == be0.samples, n + ": rank 1 gets the same sample_final calls, options and ids");
        check(be1.trace == be0.trace, n + ": in the same order relative to every other call");
    } else {
        check(be1.samples.empty(), n + ": rank 1 never sees a rank-0 sample query");
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
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL: exception " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) return 1;
    std::cout << "cohort sample hook tests: PASS\n";
    return 0;
}
