// strix/pair/relayout.hpp: after the exchange every rank holds every layer's state equal to the owner's bytes (the
// non-owner's stale bytes are replaced, the owner's own bytes are untouched); a header disagreement makes both ranks
// throw before any payload moves; the plan follows StageMap::owns.
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "strix/pair/relayout.hpp"

namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what.c_str());
        ++failures;
    }
}

// Ordered in-memory duplex link: q[r] carries bytes from rank r to rank 1 - r.
struct Wire {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::byte> q[2];
    bool aborted = false;
    strix::pair::relayout::Link link(unsigned rank) {
        strix::pair::relayout::Link l;
        l.send = [this, rank](std::span<const std::byte> b) {
            std::lock_guard<std::mutex> g(m);
            q[rank].insert(q[rank].end(), b.begin(), b.end());
            cv.notify_all();
        };
        l.receive = [this, rank](std::span<std::byte> b) {
            std::unique_lock<std::mutex> g(m);
            auto& in = q[1U - rank];
            for (auto& x : b) {
                cv.wait(g, [&] { return aborted || !in.empty(); });
                if (in.empty()) throw std::runtime_error("wire aborted");
                x = in.front();
                in.pop_front();
            }
        };
        l.abort = [this](const std::string&) {
            std::lock_guard<std::mutex> g(m);
            aborted = true;
            cv.notify_all();
        };
        return l;
    }
};

std::byte truth(std::uint32_t layer, std::size_t i) { return static_cast<std::byte>((layer * 131U + i * 7U + 3U) & 0xFFU); }

struct Rank {
    std::vector<std::vector<std::byte>> state;  // per layer
    strix::pair::relayout::Stats stats;
    std::string error;
};

// Each rank: owned layers hold the truth, the others stale 0xEE bytes; sizes(rank, layer) = that rank's byte count.
template <class Sizes>
void run(const strix::pair::StageMap& map, std::uint64_t id0, std::uint64_t id1, Sizes&& sizes, Rank (&r)[2]) {
    Wire w;
    std::thread th[2];
    for (unsigned k = 0; k < 2U; ++k) {
        r[k].state.assign(map.layers(), {});
        for (std::uint32_t l = 0; l < map.layers(); ++l) {
            r[k].state[l].assign(static_cast<std::size_t>(sizes(k, l)), std::byte{0xEE});
            if (map.owns(k, l))
                for (std::size_t i = 0; i < r[k].state[l].size(); ++i) r[k].state[l][i] = truth(l, i);
        }
    }
    for (unsigned k = 0; k < 2U; ++k)
        th[k] = std::thread([&, k] {
            try {
                const auto p = strix::pair::relayout::plan(map, k, [&](std::uint32_t l) { return sizes(k, l); });
                strix::pair::relayout::Io io;
                io.save = [&](std::uint32_t l, std::span<std::byte> out) {
                    if (out.size() != r[k].state[l].size()) throw std::runtime_error("save size");
                    std::memcpy(out.data(), r[k].state[l].data(), out.size());
                };
                io.load = [&](std::uint32_t l, std::span<const std::byte> in) {
                    if (in.size() != r[k].state[l].size()) throw std::runtime_error("load size");
                    std::memcpy(r[k].state[l].data(), in.data(), in.size());
                };
                r[k].stats = strix::pair::relayout::exchange(map, k, k == 0U ? id0 : id1, p, w.link(k), io);
            } catch (const std::exception& e) {
                r[k].error = e.what();
                w.link(k).abort(e.what());
            }
        });
    for (auto& t : th) t.join();
}

void test_moves_every_layer(const char* text) {
    const auto map = strix::pair::StageMap::parse(text, 42U);
    const auto sizes = [](unsigned, std::uint32_t l) -> std::uint64_t { return l % 5U == 3U ? 0U : 1000U + 37U * l; };
    Rank r[2];
    run(map, 77U, 77U, sizes, r);
    const std::string tag = std::string("map ") + text;
    expect(r[0].error.empty() && r[1].error.empty(), tag + ": no error (" + r[0].error + " / " + r[1].error + ")");
    for (unsigned k = 0; k < 2U; ++k)
        for (std::uint32_t l = 0; l < map.layers(); ++l) {
            bool same = true;
            for (std::size_t i = 0; i < r[k].state[l].size(); ++i) same = same && r[k].state[l][i] == truth(l, i);
            expect(same, tag + ": rank " + std::to_string(k) + " layer " + std::to_string(l) + " holds the owner's bytes");
        }
    expect(r[0].stats.sent == r[1].stats.received && r[1].stats.sent == r[0].stats.received, tag + ": byte counts agree");
    const auto p0 = strix::pair::relayout::plan(map, 0U, [&](std::uint32_t l) { return sizes(0U, l); });
    for (const auto l : p0.send) expect(map.owns(0U, l), tag + ": rank 0 sends only its layers");
    for (const auto l : p0.receive) expect(map.owns(1U, l), tag + ": rank 0 receives only rank 1's layers");
    expect(p0.send.size() + p0.receive.size() + 8U == map.layers(), tag + ": stateless layers are skipped");
}

void test_identity_refused() {
    const auto map = strix::pair::StageMap::parse("0,7,14,21,28,35,42", 42U);
    Rank r[2];
    run(map, 1U, 2U, [](unsigned, std::uint32_t l) -> std::uint64_t { return 64U + l; }, r);
    expect(!r[0].error.empty() && !r[1].error.empty(), "identity mismatch: both ranks throw");
    expect(r[0].stats.received == 0U && r[1].stats.received == 0U, "identity mismatch: no payload moved");
    bool stale = true;
    for (std::uint32_t l = 0; l < map.layers(); ++l)
        if (!map.owns(0U, l)) stale = stale && r[0].state[l][0] == std::byte{0xEE};
    expect(stale, "identity mismatch: rank 0's non-owned layers untouched");
}

void test_bytes_refused() {
    const auto map = strix::pair::StageMap::parse("0,21,42", 42U);
    Rank r[2];
    // rank 1 counts one more position in layer 30 (another prompt length / cache layout): refused before the payload
    run(map, 5U, 5U, [](unsigned k, std::uint32_t l) -> std::uint64_t { return 256U + (k == 1U && l == 30U ? 8U : 0U); }, r);
    expect(!r[0].error.empty() && !r[1].error.empty(), "byte mismatch: both ranks throw");
    expect(r[0].stats.received == 0U && r[1].stats.received == 0U, "byte mismatch: no payload moved");
}

}  // namespace

int main() {
    test_moves_every_layer("0,7,14,21,28,35,42");
    test_moves_every_layer("0,21,42");
    test_identity_refused();
    test_bytes_refused();
    using strix::pair::relayout::stage_prime_wins;
    expect(!stage_prime_wins(1U, 2048U) && stage_prime_wins(2U, 2048U) && stage_prime_wins(4U, 1024U), "route rule: two full windows");
    expect(stage_prime_wins(8U, 256U) && !stage_prime_wins(6U, 16U) && !stage_prime_wins(9U, 248U),
           "route rule: measured from 256-row windows on, smaller windows stay one-node");
    if (failures != 0) {
        std::fprintf(stderr, "pair relayout tests: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("pair relayout tests: PASS\n");
    return 0;
}
