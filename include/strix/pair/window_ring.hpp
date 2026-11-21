#pragma once
// Pair context-parallel prefill: the window ring (shared by every engine's pair route; backend-neutral, host only).
//
// Window k of layers::windows(prompt, width) runs on rank k % 2 through EVERY layer. Before layer L of window k its rank
// loads the peer's layer-L cache rows of window k-1 (Ops::load); after layer L it saves the layer-L rows of window k and
// sends them to the peer (Ops::save). Every other operation of a layer is row-local (norms, Linears, experts, row mixes),
// so each window runs the one-node kernels on the one-node inputs: the bits equal the one-node prefill at the same width,
// provided "the rows of a window" are every cache slot the window writes (attn/cache_rows.hpp slot_spans of its
// positions in each cache). Then the receiver's caches at the start of window k equal the one-node caches: by-position
// caches hold every earlier window (own or loaded), and a ring's slots older than window k-1 hold the receiver's own
// window k-2 (or slots no window has written yet, as on one node).
// After the prefill the last window's rows cross once more, so both ranks end with every window's cache rows: an exact
// replicated-state (TP) decode starts with no relayout.
//
// Message order: a rank consumes every peer message in the order the peer sends it (peer windows ascending, layers
// ascending; the last window's messages last), so ONE ordered byte stream carries them and the next message a rank waits
// for is always the head of the peer's stream: no circular wait on any link, even an unbuffered one. Each message is a
// header {magic, identity, window, layer, bytes} + payload, checked on receipt (a mismatch aborts the link and throws).
// Sends run on a sender thread with a bounded queue (pooled buffers), so the compute thread does not wait on the link.
// The link's send and receive are called from two threads (LinkMux, strix/pair/link_mux.hpp, is the binding's link).
//
// Makespan with one layer-window time h, L layers, K windows: rank of window k starts layer l at max(own layer l-1 done,
// peer's window k-1 layer l done + message): even K -> (K/2 x L + 1) h against K x L x h on one node (2K: 1.95x).
// Home: shared (host shared). Bound by: the Hy4 TP prime (apps/hy4_rows_main.cpp --prefill-route ring).
#include "strix/pair/layer_pipeline.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strix::pair::ring {

[[nodiscard]] constexpr unsigned owner(std::size_t window) noexcept { return static_cast<unsigned>(window % 2U); }

// The ring needs two windows (one window = one node's work on one rank).
[[nodiscard]] inline bool admits(std::size_t prompt, std::uint32_t width) {
    return width != 0U && layers::windows(prompt, width).size() >= 2U;
}

struct Link {
    std::function<void(std::span<const std::byte>)> send;
    std::function<void(std::span<std::byte>)> receive;
    std::function<void(const std::string&)> abort;
};

struct Ops {
    // layer `layer` of window `k` (= ws[k]) on this rank; the window's rows stay resident between its layers
    std::function<void(std::size_t k, const layers::Window&, std::uint32_t layer)> run;
    // bytes of the layer's cache rows of a window (both ranks must agree)
    std::function<std::size_t(const layers::Window&, std::uint32_t layer)> bytes;
    std::function<void(const layers::Window&, std::uint32_t layer, std::span<std::byte>)> save;
    std::function<void(const layers::Window&, std::uint32_t layer, std::span<const std::byte>)> load;
};

struct Stats {
    std::size_t windows_here = 0;
    std::uint64_t sent = 0, received = 0;
    double run_ms = 0, wait_ms = 0, load_ms = 0, save_ms = 0, send_wait_ms = 0;
};

struct Header {
    std::uint64_t magic = 0, identity = 0;
    std::uint32_t window = 0, layer = 0;
    std::uint64_t bytes = 0;
};
inline constexpr std::uint64_t kMagic = 0x57494E5252494E47ULL;  // "WINRRING"

namespace detail {
// One sender thread: frames go out in push order; buffers come back to a pool. A send failure aborts the link and is
// rethrown by the next push / finish on the compute thread.
class Sender {
public:
    Sender(const Link& link, std::size_t max_queued) : link_(link), max_(max_queued == 0U ? 1U : max_queued), th_([this] { loop(); }) {}
    ~Sender() {
        {
            std::lock_guard<std::mutex> g(m_);
            stop_ = true;
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }
    Sender(const Sender&) = delete;
    Sender& operator=(const Sender&) = delete;
    std::vector<std::byte> acquire(std::size_t bytes) {
        std::lock_guard<std::mutex> g(m_);
        std::vector<std::byte> b;
        if (!pool_.empty()) {
            b = std::move(pool_.back());
            pool_.pop_back();
        }
        b.resize(bytes);
        return b;
    }
    // returns the ms the caller waited for queue room
    double push(const Header& h, std::vector<std::byte> payload) {
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> g(m_);
        cv_.wait(g, [this] { return error_ || q_.size() < max_; });
        if (error_) std::rethrow_exception(error_);
        q_.push_back({h, std::move(payload)});
        cv_.notify_all();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    void finish() {
        std::unique_lock<std::mutex> g(m_);
        cv_.wait(g, [this] { return error_ || (q_.empty() && !busy_); });
        if (error_) std::rethrow_exception(error_);
    }

private:
    struct Frame {
        Header h;
        std::vector<std::byte> payload;
    };
    void loop() {
        for (;;) {
            Frame f;
            {
                std::unique_lock<std::mutex> g(m_);
                cv_.wait(g, [this] { return stop_ || !q_.empty(); });
                if (q_.empty()) return;
                f = std::move(q_.front());
                q_.pop_front();
                busy_ = true;
            }
            try {
                link_.send(std::as_bytes(std::span(&f.h, 1)));
                if (!f.payload.empty()) link_.send(std::span<const std::byte>(f.payload));
            } catch (...) {
                std::lock_guard<std::mutex> g(m_);
                error_ = std::current_exception();
                busy_ = false;
                q_.clear();
                cv_.notify_all();
                try { link_.abort("window ring send failed"); } catch (...) {}
                return;
            }
            std::lock_guard<std::mutex> g(m_);
            busy_ = false;
            pool_.push_back(std::move(f.payload));
            cv_.notify_all();
        }
    }
    const Link& link_;
    const std::size_t max_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<Frame> q_;
    std::vector<std::vector<std::byte>> pool_;
    std::exception_ptr error_;
    bool stop_ = false, busy_ = false;
    std::thread th_;  // last: starts after every member above is constructed
};
}  // namespace detail

// One ring prefill on this rank (both ranks call it with the same windows, layer count and identity). Throws (after
// aborting the link) on any header disagreement or a failure of either rank.
inline Stats prefill(const std::vector<layers::Window>& ws, std::uint32_t layer_count, unsigned rank, std::uint64_t identity,
                     const Link& link, const Ops& ops, std::size_t max_queued = 4U) {
    using clock = std::chrono::steady_clock;
    const auto ms = [](clock::time_point a, clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    if (rank > 1U) throw std::invalid_argument("window ring: rank 0 or 1");
    if (ws.size() < 2U) throw std::invalid_argument("window ring: needs at least two windows");
    if (layer_count == 0U) throw std::invalid_argument("window ring: no layers");
    if (!link.send || !link.receive || !link.abort || !ops.run || !ops.bytes || !ops.save || !ops.load)
        throw std::invalid_argument("window ring: link and ops must be complete");
    Stats st;
    std::vector<std::byte> in;
    const auto fail = [&link](const std::string& why) {
        try { link.abort(why); } catch (...) {}
        throw std::runtime_error(why);
    };
    const auto receive_one = [&](std::size_t j, std::uint32_t l) {
        auto t0 = clock::now();
        Header h;
        link.receive(std::as_writable_bytes(std::span(&h, 1)));
        const std::size_t want = ops.bytes(ws[j], l);
        if (h.magic != kMagic || h.identity != identity || h.window != j || h.layer != l || h.bytes != want)
            fail("window ring: peer message " + std::to_string(h.window) + ":" + std::to_string(h.layer) + " (" + std::to_string(h.bytes) +
                 " B) where window " + std::to_string(j) + " layer " + std::to_string(l) + " (" + std::to_string(want) +
                 " B) was due (identity / order / size differ)");
        in.resize(want);
        if (want != 0U) link.receive(std::span<std::byte>(in));
        auto t1 = clock::now();
        st.wait_ms += ms(t0, t1);
        ops.load(ws[j], l, std::span<const std::byte>(in));
        st.load_ms += ms(t1, clock::now());
        st.received += sizeof(Header) + want;
    };
    detail::Sender sender(link, max_queued);
    try {
        for (std::size_t k = rank; k < ws.size(); k += 2U) {
            ++st.windows_here;
            for (std::uint32_t l = 0; l < layer_count; ++l) {
                if (k != 0U) receive_one(k - 1U, l);
                auto t0 = clock::now();
                ops.run(k, ws[k], l);
                auto t1 = clock::now();
                st.run_ms += ms(t0, t1);
                const std::size_t n = ops.bytes(ws[k], l);
                auto buf = sender.acquire(n);
                ops.save(ws[k], l, std::span<std::byte>(buf));
                st.save_ms += ms(t1, clock::now());
                Header h{kMagic, identity, static_cast<std::uint32_t>(k), l, n};
                st.send_wait_ms += sender.push(h, std::move(buf));
                st.sent += sizeof(Header) + n;
            }
        }
        const std::size_t last = ws.size() - 1U;
        if (owner(last) != rank)
            for (std::uint32_t l = 0; l < layer_count; ++l) receive_one(last, l);
        sender.finish();
    } catch (...) {
        try { link.abort("window ring stopped"); } catch (...) {}
        throw;
    }
    return st;
}

}  // namespace strix::pair::ring
