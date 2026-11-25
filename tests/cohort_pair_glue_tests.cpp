// Pair glue of the shared multi-request scheduler, on the host.
// Invariants:
// - dist::ExchangeTransport: send / receive / exchange over a two-sided exchange deliver whole messages in order; a
//   size mismatch and a use after abort throw; abort runs the abort callback once.
// - serve::ReplicatedCohort over two ExchangeTransports: rank 1's backend makes rank 0's calls in rank 0's order and
//   the served ids equal the single-rank run (C1 == C4 on the pair).
// - CohortSessionHooks::idle: while nothing runs the owner calls idle() at most once per idle_interval and never blocks
//   in the queue; a submitted job is served at once; while a request runs idle() is not called; unset = unchanged.
#include "strix/dist/exchange_transport.hpp"
#include "strix/serve/cohort_session.hpp"
#include "strix/serve/replicated_cohort.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
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

namespace sv = strix::serve;
namespace sd = strix::dist;

namespace {
int failures = 0;
void check(bool value, const std::string& label) {
    if (!value) {
        ++failures;
        std::cerr << "FAIL: " << label << '\n';
    }
}

// Two-sided in-process exchange (a communicator's host path): each side posts its message and takes the peer's.
struct Pipe {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::vector<std::byte>> q[2];
    std::uint64_t exchanges[2] = {0, 0};
    bool aborted = false;
    sd::ExchangeTransport::Exchange side(int rank) {
        return [this, rank](std::span<const std::byte> out, std::span<std::byte> in) {
            std::unique_lock lock(m);
            if (aborted) throw std::runtime_error("pipe: aborted");
            q[1 - rank].emplace_back(out.begin(), out.end());
            ++exchanges[rank];
            cv.notify_all();
            if (!cv.wait_for(lock, std::chrono::seconds(10), [&] { return aborted || !q[rank].empty(); }))
                throw std::runtime_error("pipe: the peer never exchanged");
            if (aborted) throw std::runtime_error("pipe: aborted");
            auto got = std::move(q[rank].front());
            q[rank].pop_front();
            if (got.size() != in.size()) throw std::runtime_error("pipe: the ranks exchanged different sizes");
            std::memcpy(in.data(), got.data(), got.size());
        };
    }
    sd::ExchangeTransport transport(int rank) {
        return sd::ExchangeTransport(side(rank), "pipe", [this](const std::string&) {
            std::lock_guard lock(m);
            aborted = true;
            cv.notify_all();
        });
    }
};

void exchange_transport() {
    Pipe pipe;
    std::string aborted;
    sd::ExchangeTransport t0(pipe.side(0), "pipe", [&](const std::string& why) { aborted += why; });
    sd::ExchangeTransport t1(pipe.side(1));
    std::vector<std::uint32_t> got(3, 0), sent{7, 8, 9};
    std::uint64_t a = 1, b = 2, ra = 0, rb = 0;
    std::thread peer([&] {
        t1.receive(std::as_writable_bytes(std::span(got)));
        t1.exchange(std::as_bytes(std::span(&b, 1)), std::as_writable_bytes(std::span(&rb, 1)));
    });
    t0.send(std::as_bytes(std::span(sent)));
    t0.exchange(std::as_bytes(std::span(&a, 1)), std::as_writable_bytes(std::span(&ra, 1)));
    peer.join();
    check(got == sent, "exchange transport: a send reaches the peer's receive");
    check(ra == 2 && rb == 1, "exchange transport: exchange moves both messages");
    check(pipe.exchanges[0] == 2 && pipe.exchanges[1] == 2, "exchange transport: one exchange per call");
    check(t0.name() == "pipe", "exchange transport: name");
    bool threw = false;
    try {
        std::uint32_t x = 0;
        std::uint64_t y = 0;
        t0.exchange(std::as_bytes(std::span(&x, 1)), std::as_writable_bytes(std::span(&y, 1)));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "exchange transport: different out / in sizes are refused before the wire");
    t0.abort("first");
    t0.abort("second");
    check(aborted == "first" && t0.aborted(), "exchange transport: abort runs its callback once");
    threw = false;
    try {
        t0.send(std::as_bytes(std::span(sent)));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "exchange transport: no message after abort");
}

constexpr std::uint32_t kVocab = 31;
std::uint32_t next_id(const std::vector<std::uint32_t>& ctx) {
    std::uint32_t v = 7;
    for (const auto t : ctx) v = (v * 5U + t) % kVocab;
    return v;
}

// Greedy backend that drafts nothing itself; records every call so both ranks' sequences can be compared.
class Backend final : public sv::CohortBackend {
public:
    std::uint32_t lanes() const override { return 4; }
    std::uint32_t rows_per_forward() const override { return 8; }
    std::uint32_t max_positions() const override { return 256; }
    void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override {
        if (first) {
            ctx_.at(lane).clear();
            ++live;
        }
        ctx_[lane].insert(ctx_[lane].end(), tokens.begin(), tokens.end());
        trace.push_back("P" + std::to_string(lane) + ":" + std::to_string(tokens.size()));
    }
    void verify(std::span<const sv::CohortGroup> groups, std::span<const std::uint32_t> tokens,
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
            e += std::to_string(g.lane) + "x" + std::to_string(g.rows) + ",";
        }
        trace.push_back(e);
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
    void release(std::uint32_t lane) override {
        trace.push_back("R" + std::to_string(lane));
        --live;
    }
    std::vector<std::string> trace;
    int live = 0;  // lanes holding a request (owner thread only)

private:
    std::vector<std::vector<std::uint32_t>> ctx_ = std::vector<std::vector<std::uint32_t>>(4), window_ = ctx_;
    std::vector<sv::CohortGroup> groups_;
};

std::vector<sv::CohortRequest> requests() {
    std::vector<sv::CohortRequest> out;
    for (std::uint32_t i = 0; i < 4; ++i) {
        sv::CohortRequest r;
        r.id = i + 1;
        for (std::uint32_t k = 0; k < 3 + 2 * i; ++k) r.prompt.push_back((i * 11U + k * 3U) % kVocab);
        // repeating prompts so the context drafter proposes rows
        r.prompt.insert(r.prompt.end(), r.prompt.begin(), r.prompt.end());
        r.max_tokens = 10 + 3 * i;
        out.push_back(std::move(r));
    }
    return out;
}
std::vector<sv::CohortResult> run(sv::CohortBackend& be, const std::vector<sv::CohortRequest>& reqs) {
    sv::CohortDecoder d(be, sv::CohortConfig{});
    d.set_fake_clock(1.0);
    for (const auto& r : reqs) d.submit(r);
    d.run();
    auto out = d.take_finished();
    std::ranges::sort(out, [](const sv::CohortResult& a, const sv::CohortResult& b) { return a.id < b.id; });
    return out;
}

void replicated_over_exchange() {
    Pipe pipe;
    auto t0 = pipe.transport(0), t1 = pipe.transport(1);
    Backend be0, be1;
    std::string error;
    std::thread follower([&] {
        try {
            (void)sv::follow_cohort(be1, t1, {});
        } catch (const std::exception& e) {
            error = e.what();
        }
    });
    std::vector<sv::CohortResult> pair;
    {
        sv::ReplicatedCohort rep(be0, t0, {});
        pair = run(rep, requests());
        rep.close();
    }
    follower.join();
    check(error.empty(), "replicated over exchange: rank 1 ended cleanly (" + error + ")");
    check(!be0.trace.empty() && be1.trace == be0.trace, "replicated over exchange: rank 1 makes rank 0's calls in order");
    Backend solo;
    const auto alone = run(solo, requests());
    check(alone.size() == pair.size(), "replicated over exchange: result count");
    for (std::size_t i = 0; i < std::min(alone.size(), pair.size()); ++i)
        check(!alone[i].ids.empty() && alone[i].ids == pair[i].ids && pair[i].error.empty(),
              "replicated over exchange: request " + std::to_string(i + 1) + " ids equal the single-rank run");
    // the frames of a long verify body ride one-sided send / receive (frame_words smaller than a message)
    Pipe pipe2;
    auto u0 = pipe2.transport(0), u1 = pipe2.transport(1);
    Backend c0, c1;
    sv::ReplicaOptions small;
    small.frame_words = 3;
    std::thread f2([&] {
        try {
            (void)sv::follow_cohort(c1, u1, small);
        } catch (const std::exception& e) {
            error = e.what();
        }
    });
    {
        sv::ReplicatedCohort rep(c0, u0, small);
        (void)run(rep, requests());
        rep.close();
    }
    f2.join();
    check(error.empty() && c1.trace == c0.trace, "replicated over exchange: framed payloads arrive in order");
}

sv::GenerationJob job(std::vector<std::uint32_t> prompt, std::uint32_t count) {
    sv::GenerationJob j;
    j.prompt = std::move(prompt);
    j.max_tokens = count;
    return j;
}

void idle_hook() {
    Backend backend;
    sv::GenerationQueue queue;
    sv::CohortConfig config;
    config.context_draft = false;
    sv::CohortSessionHooks hooks;
    std::atomic<unsigned> idles{0};
    std::atomic<bool> idle_while_running{false};
    hooks.idle = [&] {  // owner thread: the backend's lane count is safe to read here
        ++idles;
        if (backend.live != 0) idle_while_running = true;
    };
    hooks.idle_interval = std::chrono::milliseconds(20);
    sv::CohortSession session(queue, backend, config, hooks);
    std::thread owner([&] { session.runtime().run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(210));
    const unsigned before = idles.load();
    check(before >= 3 && before <= 13, "idle hook: about one call per interval while idle (" + std::to_string(before) + ")");
    const auto t = queue.submit(job({3, 4, 5}, 6));
    bool done = false;
    std::vector<std::uint32_t> ids;
    const auto start = std::chrono::steady_clock::now();
    while (!done && std::chrono::steady_clock::now() - start < std::chrono::seconds(5)) {
        auto g = t->wait(std::chrono::milliseconds(50), done);
        ids.insert(ids.end(), g.begin(), g.end());
    }
    check(done && ids.size() == 6, "idle hook: a job submitted while idle is served");
    queue.shutdown();
    owner.join();
    check(!idle_while_running.load() && backend.live == 0, "idle hook: not called while a request holds a lane");
    // a non-positive interval with a hook is refused
    bool threw = false;
    try {
        Backend b2;
        sv::GenerationQueue q2;
        sv::CohortSessionHooks bad;
        bad.idle = [] {};
        bad.idle_interval = std::chrono::milliseconds(0);
        sv::CohortSession s2(q2, b2, config, bad);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "idle hook: a non-positive interval is refused");
}

void annotate_hook() {
    Backend backend;
    sv::GenerationQueue queue;
    sv::CohortConfig config;
    config.context_draft = false;
    sv::CohortSessionHooks hooks;
    std::atomic<unsigned> calls{0};
    hooks.annotate = [&](const sv::GenerationJob& j, const sv::CohortResult& r, sv::GenerationStats& stats) {
        ++calls;
        // the shared fields are already filled when the owner annotates
        if (stats.finish == "length" && stats.generated_tokens == r.ids.size() && stats.prompt_tokens == j.prompt.size())
            stats.engine_extra.emplace_back("owner", "seen");
    };
    sv::CohortSession session(queue, backend, config, hooks);
    std::thread owner([&] { session.runtime().run(); });
    const auto t = queue.submit(job({3, 4, 5}, 4));
    bool done = false;
    const auto start = std::chrono::steady_clock::now();
    while (!done && std::chrono::steady_clock::now() - start < std::chrono::seconds(5)) (void)t->wait(std::chrono::milliseconds(50), done);
    queue.shutdown();
    owner.join();
    const auto stats = t->stats();
    const bool seen = std::ranges::any_of(stats.engine_extra, [](const auto& kv) { return kv.first == "owner" && kv.second == "seen"; });
    check(done && calls.load() == 1 && seen, "annotate hook: called once after the shared stats, its fields are published");
}
}  // namespace

int main() {
    try {
        exchange_transport();
        replicated_over_exchange();
        idle_hook();
        annotate_hook();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: exception " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) return 1;
    std::cout << "cohort pair glue tests: PASS\n";
    return 0;
}
