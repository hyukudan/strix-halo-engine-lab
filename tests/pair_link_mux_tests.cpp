// Ordered logical channels over one ordered duplex transport (include/strix/pair/link_mux.hpp): per channel and direction
// the bytes arrive in order and unchanged whatever the send / receive / piece sizes and however the four streams interleave;
// close() leaves the raw transport aligned for plain messages; a failure on one rank fails the peer's pending receive.
#include "strix/pair/link_mux.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", what.c_str()); ++failures; }
}

class Stream {   // one ordered byte stream of a loopback transport
public:
    void send(std::span<const std::byte> d) {
        std::lock_guard lock(mu_);
        if (aborted_) throw std::runtime_error("loopback aborted");
        q_.insert(q_.end(), d.begin(), d.end());
        cv_.notify_all();
    }
    void receive(std::span<std::byte> d) {
        std::unique_lock lock(mu_);
        cv_.wait(lock, [&] { return aborted_ || q_.size() >= d.size(); });
        if (q_.size() < d.size()) throw std::runtime_error("loopback aborted");
        std::copy_n(q_.begin(), d.size(), d.begin());
        q_.erase(q_.begin(), q_.begin() + static_cast<std::ptrdiff_t>(d.size()));
    }
    void abort() { std::lock_guard lock(mu_); aborted_ = true; cv_.notify_all(); }
private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::byte> q_;
    bool aborted_ = false;
};
struct Wire {   // full duplex: side 0 sends on ab, side 1 on ba
    Stream ab, ba;
    Stream& out(unsigned side) { return side == 0U ? ab : ba; }
    Stream& in(unsigned side) { return side == 0U ? ba : ab; }
    std::unique_ptr<strix::pair::LinkMux> mux(unsigned side, std::uint32_t channels, std::size_t piece) {
        return std::make_unique<strix::pair::LinkMux>([this, side](std::span<const std::byte> d) { out(side).send(d); },
                                                      [this, side](std::span<std::byte> d) { in(side).receive(d); },
                                                      [this](const std::string&) { ab.abort(); ba.abort(); }, channels, piece);
    }
};

std::uint32_t next(std::uint32_t& s) { s = s * 1664525U + 1013904223U; return s >> 8; }
std::vector<std::byte> pattern(std::size_t n, std::uint32_t seed) {
    std::vector<std::byte> v(n);
    for (auto& b : v) b = static_cast<std::byte>(next(seed) & 0xFFU);
    return v;
}
// Sizes of the sends (or receives) of one stream: some empty, some under a piece, some spanning several pieces.
std::vector<std::size_t> sizes(std::size_t total, std::uint32_t seed, std::size_t max) {
    std::vector<std::size_t> out;
    for (std::size_t left = total; left != 0U;) {
        const std::size_t n = std::min<std::size_t>(left, next(seed) % (max + 1U));
        out.push_back(n);
        left -= n;
    }
    return out;
}
}  // namespace

int main() {
    // 1. Four streams at once (2 channels x 2 directions), independent send and receive splits, pieces of 1000 bytes.
    for (const std::size_t piece : {std::size_t{1000}, std::size_t{1} << 20U}) {
        Wire w;
        auto m0 = w.mux(0U, 2U, piece);
        auto m1 = w.mux(1U, 2U, piece);
        strix::pair::LinkMux* m[2] = {m0.get(), m1.get()};
        constexpr std::size_t total = 300000;
        std::vector<std::byte> got[2][2];   // [receiving side][channel]
        std::vector<std::thread> threads;
        std::string error;
        std::mutex error_mu;
        const auto guard = [&](auto&& body) {
            try { body(); } catch (const std::exception& e) { std::lock_guard lock(error_mu); error = e.what(); }
        };
        for (unsigned side = 0; side < 2U; ++side)
            for (std::uint32_t ch = 0; ch < 2U; ++ch) {
                threads.emplace_back([&, side, ch] { guard([&] {
                    const auto data = pattern(total, 17U * side + ch);
                    std::size_t at = 0;
                    for (const auto n : sizes(total, 5U + side + 2U * ch, 7000)) {
                        m[side]->send(ch, std::span(data).subspan(at, n));
                        at += n;
                    }
                }); });
                threads.emplace_back([&, side, ch] { guard([&] {
                    auto& g = got[side][ch];
                    g.resize(total);
                    std::size_t at = 0;
                    for (const auto n : sizes(total, 91U + side + 2U * ch, 13000)) {
                        m[side]->receive(ch, std::span(g).subspan(at, n));
                        at += n;
                    }
                }); });
            }
        for (auto& t : threads) t.join();
        expect(error.empty(), "four streams ran (" + error + ")");
        for (unsigned side = 0; side < 2U; ++side)
            for (std::uint32_t ch = 0; ch < 2U; ++ch)
                expect(got[side][ch] == pattern(total, 17U * (1U - side) + ch),
                       "side " + std::to_string(side) + " channel " + std::to_string(ch) + " piece " + std::to_string(piece) +
                           ": bytes in order and unchanged");
        // 2. close() on both ranks leaves the raw transport aligned: a plain message goes through unmuxed.
        std::thread c1([&] { guard([&] { m1->close(); }); });
        m0->close();
        c1.join();
        expect(error.empty(), "close handshake (" + error + ")");
        const auto plain = pattern(4096, 3U);
        std::vector<std::byte> back(plain.size());
        w.out(0U).send(plain);
        w.in(1U).receive(back);
        expect(back == plain, "raw transport aligned after close");
    }
    {   // 3. a rank that fails aborts the transport: the peer's pending receive fails instead of hanging
        Wire w;
        auto m0 = w.mux(0U, 2U, 1000);
        auto m1 = w.mux(1U, 2U, 1000);
        bool threw = false;
        std::thread r([&] {
            std::vector<std::byte> b(64);
            try { m1->receive(1U, b); } catch (const std::runtime_error&) { threw = true; }
        });
        m0->abort("rank 0 failed");
        r.join();
        expect(threw, "peer receive fails on abort");
        bool send_threw = false;
        try { const std::byte one[1] = {}; m0->send(0U, one); } catch (const std::runtime_error&) { send_threw = true; }
        expect(send_threw, "send after abort fails");
    }
    {   // 4. refusal: an unknown channel
        Wire w;
        auto m0 = w.mux(0U, 1U, 1000);
        auto m1 = w.mux(1U, 1U, 1000);
        bool threw = false;
        try { const std::byte one[1] = {}; m0->send(1U, one); } catch (const std::out_of_range&) { threw = true; }
        expect(threw, "unknown channel refused");
        std::thread c1([&] { m1->close(); });
        m0->close();
        c1.join();
    }
    if (failures == 0) std::printf("PASS pair link mux (4 streams x 2 piece sizes, close alignment, abort, refusal)\n");
    return failures == 0 ? 0 : 1;
}
