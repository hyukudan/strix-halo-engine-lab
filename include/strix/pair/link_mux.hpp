#pragma once
// Ordered logical byte channels over ONE ordered duplex transport (shared by every engine's pair route).
// StagePipes (strix/pair/stage_map.hpp) runs one Pipe per direction, each over its own ordered link; a pair connected by
// one transport (the striped USB4 host path opens every stream device once) carries both through this mux.
// Wire: every send is cut into pieces of at most `piece_bytes`; a piece is two transport messages, a 16-byte header
// {magic | channel, bytes} and the payload, sent under one lock, so a small message of one channel (an ACK) waits at most
// one piece of another channel's frame. One reader thread per rank reads the transport; a piece goes straight into the
// waiting receive() of its channel when nothing older is queued for that channel, else into a pooled buffer (reused, so a
// steady stream of frames does not fault fresh pages) that receive() drains first. A channel is a byte stream: a receive
// may span pieces and sends. close(): each rank sends a close message and the reader stops on the peer's, so the raw
// transport is left at a message boundary and can carry plain messages again (decode after a muxed prefill).
// Failures (a bad header, a transport error, abort()) fail every waiting and later call of both channels.
// Home: shared (host shared). Bound by: the Hy4 pair prefill on interleaved virtual stages (apps/hy4_rank_main.cpp).
#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

namespace strix::pair {

class LinkMux {
public:
    using Send = std::function<void(std::span<const std::byte>)>;
    using Receive = std::function<void(std::span<std::byte>)>;
    using Abort = std::function<void(const std::string&)>;
    static constexpr std::uint64_t magic = 0x4C4D5558ULL << 32U;   // "LMUX" in the high word, channel in the low word
    static constexpr std::uint32_t close_channel = 0xFFFFFFFFU;

    LinkMux(Send send, Receive receive, Abort abort, std::uint32_t channels, std::size_t piece_bytes = std::size_t{8} << 20U)
        : send_(std::move(send)), receive_(std::move(receive)), abort_(std::move(abort)), piece_(piece_bytes), ch_(channels) {
        // abort must stop a blocked transport receive: it is how a failed rank releases its reader (and the peer).
        if (!send_ || !receive_ || !abort_ || channels == 0U || channels >= close_channel || piece_ == 0U)
            throw std::invalid_argument("link mux needs send, receive, abort, channels and a piece size");
        reader_ = std::thread([this] { read_loop(); });
    }
    LinkMux(const LinkMux&) = delete;
    LinkMux& operator=(const LinkMux&) = delete;
    // Not closed (an exception unwound past the owner): abort, so the peer fails instead of waiting for frames.
    ~LinkMux() {
        if (!closed_) abort("link mux destroyed before close");
        if (reader_.joinable()) reader_.join();
    }

    [[nodiscard]] std::uint32_t channels() const noexcept { return static_cast<std::uint32_t>(ch_.size()); }

    void send(std::uint32_t channel, std::span<const std::byte> data) {
        if (channel >= ch_.size()) throw std::out_of_range("link mux: no such channel");
        for (std::size_t at = 0; at < data.size(); at += piece_) {
            const std::size_t n = std::min(piece_, data.size() - at);
            std::lock_guard lock(send_mu_);
            throw_if_failed();
            put_header(channel, n);
            send_(data.subspan(at, n));
        }
    }

    void receive(std::uint32_t channel, std::span<std::byte> data) {
        if (channel >= ch_.size()) throw std::out_of_range("link mux: no such channel");
        std::unique_lock lock(mu_);
        auto& c = ch_[channel];
        if (c.want != nullptr) throw std::logic_error("link mux: one receiver per channel");
        c.want = data.data();
        c.want_size = data.size();
        c.filled = 0;
        for (;;) {
            while (!c.queue.empty() && c.filled < c.want_size) {   // older bytes first (pieces arrive in order)
                auto& front = c.queue.front();
                const std::size_t n = std::min(front.size() - c.offset, c.want_size - c.filled);
                std::memcpy(c.want + c.filled, front.data() + c.offset, n);
                c.filled += n;
                c.offset += n;
                if (c.offset == front.size()) {
                    pool_.push_back(std::move(front));
                    c.queue.pop_front();
                    c.offset = 0;
                }
            }
            if (c.filled == c.want_size) break;
            if (failed_ || peer_closed_) {
                cv_.wait(lock, [&] { return !c.direct; });   // the reader no longer writes into `data`
                c.want = nullptr;
                throw std::runtime_error(failed_ ? "link mux failed: " + why_ : "link mux: peer closed with a receive pending");
            }
            cv_.wait(lock);
        }
        c.want = nullptr;
    }

    // Fails every waiting and later call on this rank and aborts the transport (the peer's calls fail on its side).
    void abort(const std::string& why) {
        fail(why);
        abort_(why);
    }

    // Both ranks call close() once their channels are idle; returns when the peer's close has arrived. Bytes still queued
    // for a channel are a protocol error.
    void close() {
        {
            std::lock_guard lock(send_mu_);
            throw_if_failed();
            put_header(close_channel, 0U);
        }
        if (reader_.joinable()) reader_.join();
        closed_ = true;
        std::lock_guard lock(mu_);
        if (failed_) throw std::runtime_error("link mux failed: " + why_);
        for (const auto& c : ch_)
            if (!c.queue.empty()) throw std::runtime_error("link mux: closed with unread bytes on a channel");
    }

private:
    struct Header {
        std::uint64_t word = 0;    // magic | channel
        std::uint64_t bytes = 0;
    };
    struct Channel {
        std::deque<std::vector<std::byte>> queue;  // pieces not consumed yet; the front is consumed up to `offset`
        std::size_t offset = 0;
        std::byte* want = nullptr;                 // a waiting receive(): destination, size, bytes filled
        std::size_t want_size = 0, filled = 0;
        bool direct = false;                       // the reader is writing a piece straight into `want`
    };

    void put_header(std::uint32_t channel, std::size_t bytes) {
        Header h;
        h.word = magic | channel;
        h.bytes = bytes;
        send_(std::as_bytes(std::span(&h, 1)));
    }
    void throw_if_failed() {
        std::lock_guard lock(mu_);
        if (failed_) throw std::runtime_error("link mux failed: " + why_);
    }
    void fail(const std::string& why) {
        std::lock_guard lock(mu_);
        if (!failed_) { failed_ = true; why_ = why; }
        cv_.notify_all();
    }
    void read_loop() {
        try {
            for (;;) {
                Header h;
                receive_(std::as_writable_bytes(std::span(&h, 1)));
                if ((h.word & 0xFFFFFFFF00000000ULL) != magic) throw std::runtime_error("link mux: bad header magic");
                const auto channel = static_cast<std::uint32_t>(h.word & 0xFFFFFFFFULL);
                if (channel == close_channel) {
                    std::lock_guard lock(mu_);
                    peer_closed_ = true;
                    cv_.notify_all();
                    return;
                }
                if (channel >= ch_.size() || h.bytes == 0U || h.bytes > piece_) throw std::runtime_error("link mux: bad header");
                const auto n = static_cast<std::size_t>(h.bytes);
                std::byte* dst = nullptr;
                std::vector<std::byte> buf;
                {
                    std::lock_guard lock(mu_);
                    if (failed_) return;
                    auto& c = ch_[channel];
                    if (c.queue.empty() && c.want != nullptr && c.want_size - c.filled >= n) {
                        dst = c.want + c.filled;
                        c.direct = true;
                    } else {
                        for (auto it = pool_.begin(); it != pool_.end(); ++it)
                            if (it->capacity() >= n) { buf = std::move(*it); pool_.erase(it); break; }
                    }
                }
                if (dst != nullptr) {
                    try {
                        receive_(std::span(dst, n));
                    } catch (...) {
                        std::lock_guard lock(mu_);
                        ch_[channel].direct = false;
                        throw;
                    }
                    std::lock_guard lock(mu_);
                    auto& c = ch_[channel];
                    c.direct = false;
                    c.filled += n;
                    cv_.notify_all();
                } else {
                    buf.resize(n);
                    receive_(std::span(buf));
                    std::lock_guard lock(mu_);
                    ch_[channel].queue.push_back(std::move(buf));
                    cv_.notify_all();
                }
            }
        } catch (const std::exception& e) {
            fail(e.what());
        } catch (...) {
            fail("link mux reader failed");
        }
    }

    Send send_;
    Receive receive_;
    Abort abort_;
    std::size_t piece_;
    std::mutex send_mu_, mu_;
    std::condition_variable cv_;
    std::vector<Channel> ch_;
    std::vector<std::vector<std::byte>> pool_;
    bool failed_ = false, peer_closed_ = false, closed_ = false;
    std::string why_;
    std::thread reader_;
};

}  // namespace strix::pair
