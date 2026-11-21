// strix/pair/window_ring.hpp: a two-rank window-ring prefill of a causal toy model (one by-position cache and one
// sliding ring cache per layer, rows resident between layers) leaves BOTH ranks with every cache slot equal to the
// one-rank sequential prefill, and the last window's output rows equal; odd and even window counts, a short last window,
// a ring that wraps inside a window (attn/cache_rows.hpp slot_spans); one window is refused; an identity disagreement
// makes both ranks throw. The ranks talk through LinkMux over an in-memory byte stream (the binding's link).
#include "strix/attn/cache_rows.hpp"
#include "strix/pair/link_mux.hpp"
#include "strix/pair/window_ring.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what.c_str());
        ++failures;
    }
}

namespace cr = strix::attn::cache_rows;
namespace layers = strix::pair::layers;
namespace ring = strix::pair::ring;

std::uint64_t mix(std::uint64_t a, std::uint64_t b) {
    std::uint64_t h = a * 0x9E3779B97F4A7C15ULL + b + 0x632BE59BD9B4E019ULL;
    h ^= h >> 29U;
    h *= 0xBF58476D1CE4E5B9ULL;
    return h ^ (h >> 32U);
}

// One rank's toy model: layer l writes full[l][p] (by position) and rng[l][p % ring] (sliding window `win`), then each
// row reads every earlier full row and its window of ring rows (causal, inclusive).
struct Toy {
    std::uint32_t P = 0, layers_n = 0, win = 0;
    cr::Rows full_rows{1U, 0U}, ring_rows{1U, 0U};
    std::vector<std::uint32_t> tok;
    std::vector<std::vector<std::uint64_t>> full, rng;
    std::vector<std::uint64_t> x;
    Toy(std::uint32_t p, std::uint32_t width, std::uint32_t nl, std::uint32_t w) : P(p), layers_n(nl), win(w) {
        ring_rows.ring = cr::window_ring(w, width);
        for (std::uint32_t i = 0; i < P; ++i) tok.push_back((i * 2654435761U) % 50021U);
        full.assign(nl, std::vector<std::uint64_t>(P, 0U));
        rng.assign(nl, std::vector<std::uint64_t>(ring_rows.ring, 0U));
        x.assign(P, 0U);
    }
    void run(const layers::Window& w, std::uint32_t l) {
        const std::uint32_t p1 = w.start + w.rows;
        if (l == 0U)
            for (std::uint32_t p = w.start; p < p1; ++p) x[p] = mix(tok[p], 7U);
        for (std::uint32_t p = w.start; p < p1; ++p) {
            full[l][p] = mix(x[p], 2U * l + 1U);
            rng[l][cr::slot(ring_rows, p)] = mix(x[p], 2U * l + 2U);
        }
        std::vector<std::uint64_t> nx(w.rows);
        for (std::uint32_t p = w.start; p < p1; ++p) {
            std::uint64_t acc = x[p];
            for (std::uint32_t q = 0; q <= p; ++q) acc = mix(acc, full[l][q]);
            for (std::uint32_t q = p + 1U > win ? p + 1U - win : 0U; q <= p; ++q) acc = mix(acc, rng[l][cr::slot(ring_rows, q)]);
            nx[p - w.start] = acc;
        }
        std::copy(nx.begin(), nx.end(), x.begin() + w.start);
    }
    // a window's rows of layer l: its full rows, then its ring slots (slot_spans order)
    std::size_t bytes(const layers::Window& w, std::uint32_t) const {
        std::uint64_t n = w.rows;
        const auto s = cr::slot_spans(ring_rows, w.start, std::uint64_t{w.start} + w.rows);
        for (unsigned i = 0; i < s.count; ++i) n += s.span[i].rows;
        return static_cast<std::size_t>(n) * sizeof(std::uint64_t);
    }
    template <class Copy> void pieces(const layers::Window& w, std::uint32_t l, Copy&& copy) {
        std::size_t at = 0;
        copy(full[l].data() + w.start, w.rows, at);
        at += w.rows;
        const auto s = cr::slot_spans(ring_rows, w.start, std::uint64_t{w.start} + w.rows);
        for (unsigned i = 0; i < s.count; ++i) {
            copy(rng[l].data() + s.span[i].slot, static_cast<std::size_t>(s.span[i].rows), at);
            at += static_cast<std::size_t>(s.span[i].rows);
        }
    }
};

// Ordered in-memory byte stream both ways: q[r] carries bytes from rank r.
struct Wire {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::byte> q[2];
    bool aborted = false;
    void send(unsigned r, std::span<const std::byte> b) {
        std::lock_guard<std::mutex> g(m);
        if (aborted) throw std::runtime_error("wire aborted");
        q[r].insert(q[r].end(), b.begin(), b.end());
        cv.notify_all();
    }
    void receive(unsigned r, std::span<std::byte> b) {
        std::unique_lock<std::mutex> g(m);
        auto& in = q[1U - r];
        for (auto& v : b) {
            cv.wait(g, [&] { return aborted || !in.empty(); });
            if (in.empty()) throw std::runtime_error("wire aborted");
            v = in.front();
            in.pop_front();
        }
    }
    void abort() {
        std::lock_guard<std::mutex> g(m);
        aborted = true;
        cv.notify_all();
    }
};

struct Result {
    Toy toy;
    ring::Stats st;
    std::string error;
};

void ring_rank(Wire& wire, unsigned r, std::uint64_t identity, std::uint32_t width, Result& out) {
    try {
        strix::pair::LinkMux mux([&wire, r](std::span<const std::byte> d) { wire.send(r, d); },
                                 [&wire, r](std::span<std::byte> d) { wire.receive(r, d); }, [&wire](const std::string&) { wire.abort(); }, 1U,
                                 std::size_t{64});
        ring::Link link{[&mux](std::span<const std::byte> d) { mux.send(0U, d); }, [&mux](std::span<std::byte> d) { mux.receive(0U, d); },
                        [&mux](const std::string& why) { mux.abort(why); }};
        Toy& t = out.toy;
        ring::Ops ops;
        ops.run = [&t](std::size_t, const layers::Window& w, std::uint32_t l) { t.run(w, l); };
        ops.bytes = [&t](const layers::Window& w, std::uint32_t l) { return t.bytes(w, l); };
        ops.save = [&t](const layers::Window& w, std::uint32_t l, std::span<std::byte> b) {
            t.pieces(w, l, [&b](const std::uint64_t* src, std::size_t n, std::size_t at) {
                std::memcpy(b.data() + at * sizeof(std::uint64_t), src, n * sizeof(std::uint64_t));
            });
        };
        ops.load = [&t](const layers::Window& w, std::uint32_t l, std::span<const std::byte> b) {
            t.pieces(w, l, [&b](std::uint64_t* dst, std::size_t n, std::size_t at) {
                std::memcpy(dst, b.data() + at * sizeof(std::uint64_t), n * sizeof(std::uint64_t));
            });
        };
        out.st = ring::prefill(layers::windows(t.P, width), t.layers_n, r, identity, link, ops, 2U);
        mux.close();
    } catch (const std::exception& e) {
        out.error = e.what();
        wire.abort();
    }
}

void check_case(std::uint32_t P, std::uint32_t width, std::uint32_t nl, std::uint32_t win) {
    const std::string tag = "P=" + std::to_string(P) + " R=" + std::to_string(width) + " L=" + std::to_string(nl) + " win=" + std::to_string(win);
    Toy ref(P, width, nl, win);
    const auto ws = layers::windows(P, width);
    for (const auto& w : ws)
        for (std::uint32_t l = 0; l < nl; ++l) ref.run(w, l);
    Wire wire;
    Result res[2]{{Toy(P, width, nl, win), {}, {}}, {Toy(P, width, nl, win), {}, {}}};
    std::thread a(ring_rank, std::ref(wire), 0U, 77ULL, width, std::ref(res[0]));
    std::thread b(ring_rank, std::ref(wire), 1U, 77ULL, width, std::ref(res[1]));
    a.join();
    b.join();
    for (unsigned r = 0; r < 2U; ++r) {
        expect(res[r].error.empty(), tag + " rank " + std::to_string(r) + " error: " + res[r].error);
        expect(res[r].toy.full == ref.full, tag + " rank " + std::to_string(r) + " by-position caches equal one rank");
        expect(res[r].toy.rng == ref.rng, tag + " rank " + std::to_string(r) + " ring caches equal one rank");
        expect(res[r].st.windows_here == (ws.size() + 1U - r) / 2U, tag + " windows on rank " + std::to_string(r));
    }
    const auto& last = ws.back();
    const auto& own = res[ring::owner(ws.size() - 1U)].toy;
    expect(std::equal(ref.x.begin() + last.start, ref.x.begin() + last.start + last.rows, own.x.begin() + last.start),
           tag + " last window output rows equal one rank");
    expect(res[0].st.sent == res[1].st.received && res[1].st.sent == res[0].st.received, tag + " every sent byte received");
}

}  // namespace

int main() {
    // slot_spans: by position; a ring without wrap; a ring that wraps; a range longer than the ring keeps its last rows
    {
        const auto a = cr::slot_spans({4U, 0U}, 10U, 30U);
        expect(a.count == 1U && a.span[0].slot == 10U && a.span[0].rows == 20U, "slot_spans by position");
        const auto b = cr::slot_spans({4U, 32U}, 32U, 48U);
        expect(b.count == 1U && b.span[0].slot == 0U && b.span[0].rows == 16U, "slot_spans ring no wrap");
        const auto c = cr::slot_spans({4U, 32U}, 24U, 40U);
        expect(c.count == 2U && c.span[0].slot == 24U && c.span[0].rows == 8U && c.span[1].slot == 0U && c.span[1].rows == 8U,
               "slot_spans ring wrap");
        const auto d = cr::slot_spans({4U, 16U}, 0U, 40U);
        expect(d.count == 2U && d.span[0].slot == 8U && d.span[0].rows == 8U && d.span[1].slot == 0U && d.span[1].rows == 8U,
               "slot_spans keeps the ring's last rows");
        expect(cr::slot_spans({4U, 16U}, 5U, 5U).count == 0U, "slot_spans empty");
    }
    check_case(32U, 16U, 3U, 8U);   // 2 windows (the 8K form)
    check_case(48U, 16U, 2U, 8U);   // 3 windows: the last on rank 0, rank 1 receives it at the end
    check_case(64U, 16U, 3U, 12U);  // 4 windows, the ring (32) holds two windows
    check_case(40U, 16U, 3U, 8U);   // short last window
    check_case(72U, 24U, 2U, 16U);  // ring 64 with 24-row windows: a window's ring slots wrap
    check_case(128U, 16U, 2U, 8U);  // 8 windows (the 32K form)
    expect(!ring::admits(16U, 16U) && ring::admits(17U, 16U), "admits needs two windows");
    {  // one window: refused before any message
        bool threw = false;
        try {
            ring::Link l{[](std::span<const std::byte>) {}, [](std::span<std::byte>) {}, [](const std::string&) {}};
            ring::Ops o{[](std::size_t, const layers::Window&, std::uint32_t) {}, [](const layers::Window&, std::uint32_t) { return std::size_t{0}; },
                        [](const layers::Window&, std::uint32_t, std::span<std::byte>) {},
                        [](const layers::Window&, std::uint32_t, std::span<const std::byte>) {}};
            (void)ring::prefill(layers::windows(16U, 16U), 2U, 0U, 1ULL, l, o);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        expect(threw, "one window refused");
    }
    {  // identity disagreement: both ranks throw
        Wire wire;
        Result res[2]{{Toy(48U, 16U, 2U, 8U), {}, {}}, {Toy(48U, 16U, 2U, 8U), {}, {}}};
        std::thread a(ring_rank, std::ref(wire), 0U, 5ULL, 16U, std::ref(res[0]));
        std::thread b(ring_rank, std::ref(wire), 1U, 6ULL, 16U, std::ref(res[1]));
        a.join();
        b.join();
        expect(!res[0].error.empty() && !res[1].error.empty(), "identity disagreement fails both ranks");
    }
    if (failures == 0) std::printf("pair window ring tests: OK\n");
    return failures == 0 ? 0 : 1;
}
