#pragma once

// Integrity of one USB4 host stream (a byte stream between the two ranks).
//
// The link below the stream can lose a whole frame without any error on either side (the NHI reports damaged frames,
// but a packet dropped on the way never arrives), so a byte stream with a hole either stalls (the tail never comes)
// or shifts every later byte into the wrong message. Every byte of the host path therefore travels in units:
//
//   DATA    header (sequence number, message size and offset, payload CRC32C, header CRC32C) + payload (one piece)
//   ACK     cumulative: the receiver accepted every piece before `seq`
//   NACK    the receiver saw damage (bad header, payload CRC, a gap in the sequence, or a unit that stopped arriving)
//           and wants every piece again from `seq`
//   RESYNC  the sender's answer to a NACK: the resend round `epoch` starts here, at piece `seq`
//
// The receiver accepts pieces strictly in order, so a lost frame costs one NACK and a resend from the damaged piece
// (go-back-N) and never reaches the caller. After a NACK the receiver drops bytes until it finds the RESYNC of a
// newer round (headers are found by magic + header CRC at any byte offset). A sender keeps every piece until it is
// acknowledged: in the caller's buffer while the call lasts, then (if the peer has not acknowledged it within a short
// linger, i.e. the peer is not receiving yet) in a copy, so `send` keeps its meaning (it returns when the bytes are
// handed over, not when the peer has read them). A piece that needs more than `attempts` resends is a hard error,
// and so is damage to a small message sent without a copy when nothing was retained (see begin_send).
//
// Threads: at most one sending and one receiving thread per link (the transport's contract), and either may pump the
// other direction (a sender reads ACKs, a receiver writes ACKs). Locks: tx_mutex_ and rx_mutex_ are never nested;
// ctl_mutex_ is a leaf.

#include "strix/dist/communicator.hpp"

#include <poll.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::dist::link {

inline constexpr std::uint64_t unit_magic = 0x31544e55584e5453ULL;  // "STNXUNT1"
inline constexpr std::uint32_t kind_data = 1U, kind_ack = 2U, kind_nack = 3U, kind_resync = 4U;
inline constexpr std::uint32_t flag_ack = 1U;   // DATA: kept by the sender until acknowledged (resendable)
inline constexpr std::uint32_t flag_lazy = 2U;  // DATA: nobody waits for this ACK: it may ride on the next unit out

struct Unit {
    std::uint64_t magic = unit_magic;
    std::uint32_t kind = 0;
    std::uint32_t flags = 0;
    std::uint64_t seq = 0;      // DATA: piece number; ACK: pieces accepted (cumulative); NACK / RESYNC: next piece
    std::uint64_t epoch = 0;    // RESYNC: resend round (increasing per direction); NACK: the last round received
    std::uint64_t message = 0;  // DATA: bytes of the message the piece belongs to
    std::uint64_t offset = 0;   // DATA: offset of the piece in that message
    std::uint32_t size = 0;     // DATA: payload bytes after the header
    std::uint32_t payload_crc = 0;
    std::uint32_t attempt = 0;  // DATA: 0 first transmission, n the n-th resend of this piece
    std::uint32_t header_crc = 0;
};
static_assert(sizeof(Unit) == 64);

// What a stream errno means for the pair (the transport's failure texts).
inline std::string errno_text(int error) {
    switch (error) {
    case ENODEV: return "ENODEV: the stream device is gone (link unplugged or controller reset)";
    case ETIMEDOUT: return "ETIMEDOUT: the driver timed out (peer or domain gone)";
    case ENXIO: return "ENXIO: no stream attached (peer service gone)";
    case ENOMEM: return "ENOMEM: the driver could not allocate the stream rings (memory pressure)";
    case EPIPE: return "EPIPE: the peer closed the stream (its rank failed or exited)";
    case ECONNRESET: return "ECONNRESET: the peer closed the stream (its rank failed or exited)";
    default: return std::strerror(error);
    }
}

inline std::uint32_t unit_crc(const Unit& u) noexcept {
    return crc32c(std::span<const std::byte>(reinterpret_cast<const std::byte*>(&u), offsetof(Unit, header_crc)));
}
inline Unit sealed(Unit u) noexcept {
    u.header_crc = unit_crc(u);
    return u;
}
inline bool valid(const Unit& u) noexcept {
    if (u.magic != unit_magic || u.kind < kind_data || u.kind > kind_resync || u.header_crc != unit_crc(u)) return false;
    return u.kind != kind_data || (u.offset <= u.message && u.size <= u.message - u.offset);
}

struct Options {
    std::size_t piece = std::size_t{256} << 10U;       // payload bytes per DATA unit
    std::chrono::milliseconds timeout{120000};         // no progress at all: the peer is gone
    std::chrono::milliseconds loss{1000};              // a unit that stops arriving half-way: a lost frame
    std::chrono::milliseconds ack{4000};               // everything written, no acknowledgement: resend (doubles)
    std::chrono::microseconds linger{2000};            // send(): wait this long for the ACK before keeping a copy
    std::size_t retain_cap = std::size_t{64} << 20U;   // copies kept for unacknowledged pieces, at most
    unsigned attempts = 4;                             // resends of one piece after damage, then a hard error
    bool spin = false;                                 // busy-wait instead of poll()
};

// A failure of the stream (the message names the stream, the piece and what was seen).
struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};
// The transport was aborted (another stream failed or the owner gave up): leave at once.
struct Stopped {};

struct Counters {
    std::atomic<std::uint64_t> pieces_out{0}, pieces_in{0}, bytes_out{0}, bytes_in{0}, resends{0}, resent_bytes{0},
        crc_failures{0}, header_failures{0}, losses{0}, gaps{0}, duplicates{0}, nacks_out{0}, nacks_in{0}, ack_timeouts{0},
        retained{0};
};

class Link {
public:
    using clock = std::chrono::steady_clock;

    Link(int fd, std::string name, Options options, const std::atomic<bool>* abort = nullptr)
        : fd_(fd), name_(std::move(name)), opt_(options), abort_(abort) {
        if (opt_.piece == 0U || opt_.piece > (std::size_t{1} << 30U)) opt_.piece = std::size_t{256} << 10U;
        if (opt_.attempts == 0U) opt_.attempts = 1U;
        discard_.resize(std::size_t{64} << 10U);
        last_rx_ = clock::now();
        last_ack_change_ = last_rx_;
    }
    Link(const Link&) = delete;
    Link& operator=(const Link&) = delete;

    void set_abort(const std::atomic<bool>* abort) noexcept { abort_ = abort; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] const Counters& counters() const noexcept { return counters_; }

    // ---- Stepping interface (the transport's eager loop drives several links on one thread) ----

    // Queues a message. Pieces go out with flag_ack (kept until acknowledged) when `retain` is set or when earlier
    // pieces are still unacknowledged (their numbers must stay resendable in one run); otherwise without, and the
    // caller's buffer is only read until send_written() (damage to such a piece is reported, not repaired).
    // `copy`: keep a copy at once (the caller's buffer is released when the call returns); without it a retained
    // message stays in the caller's buffer until keep_copies() (the blocking send does that before it returns).
    void begin_send(std::span<const std::byte> message, bool retain, bool copy = false) {
        std::lock_guard lock(tx_mutex_);
        sent_through_target_ = next_seq_;
        if (message.empty()) return;
        TxMessage m;
        m.first = next_seq_;
        m.pieces = (message.size() + opt_.piece - 1U) / opt_.piece;
        m.ack = retain || !queue_.empty();
        m.user = true;
        m.data = message;
        m.lazy = m.ack && !retain;  // nobody waits in this call for the ACK of a message that is not retained by it
        if (copy && retain) m.lazy = true;  // eager copies: the call returns once written
        if (m.ack && (copy || !retain)) {  // a small message (or one behind unacknowledged pieces): keep a copy now
            m.owned.assign(message.begin(), message.end());
            m.data = m.owned;
            m.user = false;
            counters_.retained.fetch_add(m.data.size(), std::memory_order_relaxed);
        }
        next_seq_ += m.pieces;
        sent_through_target_ = next_seq_;
        queue_.push_back(std::move(m));
    }
    // Every piece of the last begun message was handed to the driver at least once.
    [[nodiscard]] bool send_written() {
        std::lock_guard lock(tx_mutex_);
        return written_through_ >= sent_through_target_;
    }
    void begin_receive(std::span<std::byte> message) {
        std::lock_guard lock(rx_mutex_);
        posted_ = message;
        posted_on_ = !message.empty();
        posted_complete_ = message.empty();
    }
    [[nodiscard]] bool received() {
        std::lock_guard lock(rx_mutex_);
        return posted_complete_;
    }
    // One non-blocking pass over both directions. need_rx: wait for the receive side lock (a receiver); else try it.
    // Returns whether any byte moved. Sets rx_busy() when another thread holds the receive side.
    bool progress(bool need_rx, bool* rx_busy = nullptr) {
        if (abort_ && abort_->load(std::memory_order_acquire)) throw Stopped{};
        bool moved = false;
        if (rx_busy) *rx_busy = false;
        {
            std::unique_lock lock(rx_mutex_, std::defer_lock);
            if (need_rx) lock.lock();
            else if (!lock.try_lock() && rx_busy) *rx_busy = true;
            // While another thread of this link sleeps in poll() for input, it is the only reader (bytes taken here would
            // not wake it): this thread waits for its news (bump) instead.
            if (lock.owns_lock() && !polling_.load()) moved |= pump_rx();
        }
        moved |= step_tx();
        return moved;
    }
    // Events worth polling for: POLLIN always (acknowledgements, data), POLLOUT while a unit waits for the driver.
    [[nodiscard]] short events() {
        std::lock_guard lock(tx_mutex_);
        return static_cast<short>(POLLIN | (tx_pending() ? POLLOUT : 0));
    }
    [[nodiscard]] bool retaining() {
        std::lock_guard lock(tx_mutex_);
        return !queue_.empty();
    }
    [[nodiscard]] int fd() const noexcept { return fd_; }
    // A calling thread drives this link (blocking send/receive, or the eager loop) while the guard lives: the service
    // thread leaves it alone.
    class Use {
    public:
        explicit Use(Link& l) noexcept : l_(&l) { l_->users_.fetch_add(1); }
        Use(Use&& o) noexcept : l_(std::exchange(o.l_, nullptr)) {}
        Use(const Use&) = delete;
        Use& operator=(const Use&) = delete;
        Use& operator=(Use&&) = delete;
        ~Use() {
            if (!l_) return;
            l_->users_.fetch_sub(1);
            l_->bump();  // a thread that waited for this one's pump now pumps itself
        }

    private:
        Link* l_;
    };
    [[nodiscard]] Use hold() noexcept { return Use(*this); }
    [[nodiscard]] bool in_use() const noexcept { return users_.load() > 0; }

    // ---- Blocking interface (stream threads, the open probe) ----

    // Returns once every piece is acknowledged, or after the linger with a copy of what is not (bounded by retain_cap).
    void send(std::span<const std::byte> message) {
        if (message.empty()) return;
        const auto use = hold();
        begin_send(message, true);
        auto deadline = clock::now() + opt_.timeout;
        while (!send_written()) wait_round(false, deadline, "write");
        const auto end = sent_through_target_;
        const auto linger_end = clock::now() + opt_.linger;
        for (;;) {
            {
                std::lock_guard lock(ctl_mutex_);
                if (acked_ >= end) break;
            }
            if (clock::now() >= linger_end) {
                std::lock_guard lock(tx_mutex_);
                if (retained_bytes() + message.size() <= opt_.retain_cap) {
                    keep_copies();
                    break;
                }
            }
            wait_round(false, deadline, "acknowledgement");
        }
        std::lock_guard lock(tx_mutex_);
        release_acked();
        keep_copies();  // whatever is still user-backed (acknowledged meanwhile or not) must not outlive the call
    }
    void receive(std::span<std::byte> message) {
        if (message.empty()) return;
        const auto use = hold();
        begin_receive(message);
        auto deadline = clock::now() + opt_.timeout;
        while (!received()) wait_round(true, deadline, "read");
        // Hand the acknowledgement to the driver now unless a sender of this rank will (it writes at unit boundaries).
        for (int i = 0; i < 64 && urgent_control() && !sender_active(); ++i) {
            if (!step_tx()) wait_io(POLLOUT, std::chrono::milliseconds(1));
        }
    }
    // Something for the transport's service thread: retained pieces (ACKs to read, a lost tail to resend), control
    // units to write, or a resend the peer asked for.
    [[nodiscard]] bool needs_service() { return !in_use() && (control_pending() || retaining()); }
    // Non-blocking: acknowledgements and requests for retained pieces, so a peer that waits for a resend gets it even
    // when this rank's next transfers use other streams. Links a calling thread is driving are left to it (a second
    // reader would take the bytes that thread waits for in poll()).
    void service() {
        if (in_use() || (!retaining() && !control_pending())) return;
        (void)progress(false);
        check_ack_timeout();
    }
    // Before close: give the peer the retained pieces it still needs (bounded).
    void drain(std::chrono::milliseconds budget) noexcept {
        try {
            const auto end = clock::now() + budget;
            while (retaining() && clock::now() < end) {
                if (!progress(false)) wait_io(POLLIN, std::chrono::milliseconds(5));
                check_ack_timeout();
                std::lock_guard lock(tx_mutex_);
                release_acked();
            }
        } catch (...) {
        }
    }
    [[nodiscard]] std::string summary() const {
        const auto v = [](const std::atomic<std::uint64_t>& a) { return a.load(std::memory_order_relaxed); };
        std::ostringstream out;
        out << name_ << " pieces out " << v(counters_.pieces_out) << " in " << v(counters_.pieces_in) << ", resends " << v(counters_.resends)
            << " (" << v(counters_.resent_bytes) << " B), crc failures " << v(counters_.crc_failures) << ", header failures "
            << v(counters_.header_failures) << ", losses " << v(counters_.losses) << ", gaps " << v(counters_.gaps) << ", duplicates "
            << v(counters_.duplicates) << ", nacks out " << v(counters_.nacks_out) << " in " << v(counters_.nacks_in) << ", ack timeouts "
            << v(counters_.ack_timeouts) << ", retained " << v(counters_.retained) << " B";
        return out.str();
    }
    [[nodiscard]] bool clean() const noexcept {
        return !counters_.resends.load() && !counters_.crc_failures.load() && !counters_.header_failures.load() && !counters_.losses.load() &&
               !counters_.gaps.load() && !counters_.duplicates.load() && !counters_.nacks_out.load() && !counters_.nacks_in.load() &&
               !counters_.ack_timeouts.load();
    }
    [[nodiscard]] std::string state() {
        std::ostringstream out;
        std::uint64_t acked = 0;
        {
            std::lock_guard lock(ctl_mutex_);
            acked = acked_;
        }
        {
            std::lock_guard lock(tx_mutex_);
            out << "tx next " << next_seq_ << " cursor " << send_seq_ << " acked " << acked << " queued " << queue_.size();
        }
        {
            std::lock_guard lock(rx_mutex_);
            out << ", rx expected " << expected_ << " mode " << (mode_ == Mode::header ? "header" : mode_ == Mode::payload ? "payload" : "scan")
                << (parked_ ? " parked" : "");
        }
        return out.str();
    }

private:
    enum class Mode { header, payload, scan };
    struct TxMessage {
        std::uint64_t first = 0, pieces = 0;
        std::span<const std::byte> data;
        std::vector<std::byte> owned;
        bool ack = false, user = false, lazy = false;
    };
    struct TxUnit {
        std::array<Unit, 2> heads{};  // [0] a lazy ACK riding in front (when nheads == 2), then the unit's header
        std::size_t nheads = 1;
        std::span<const std::byte> payload;
        std::size_t done = 0;
        bool busy = false;
        bool data = false;
        std::uint64_t seq = 0;
    };

    [[noreturn]] void fail(const std::string& what) const { throw Error("usb4 stream " + name_ + ": " + what); }
    std::string piece_text(std::uint64_t seq) const { return "piece " + std::to_string(seq); }

    // ---- send side (tx_mutex_ held) ----
    bool tx_pending() const {
        return cur_.busy || send_seq_ < next_seq_ || ctl_ack_out_.load(std::memory_order_relaxed) ||
               ctl_nack_out_.load(std::memory_order_relaxed) || ctl_resend_.load(std::memory_order_relaxed);
    }
    bool sender_active() {
        std::lock_guard lock(tx_mutex_);
        return cur_.busy || send_seq_ < next_seq_;
    }
    bool urgent_control() const {
        return ctl_ack_out_.load(std::memory_order_relaxed) || ctl_nack_out_.load(std::memory_order_relaxed) ||
               ctl_resend_.load(std::memory_order_relaxed);
    }
    bool control_pending() const {
        return ctl_ack_out_.load(std::memory_order_relaxed) || ctl_nack_out_.load(std::memory_order_relaxed) ||
               ctl_resend_.load(std::memory_order_relaxed) || ctl_lazy_ack_.load(std::memory_order_relaxed);
    }
    std::size_t retained_bytes() const {
        std::size_t total = 0;
        for (const auto& m : queue_)
            if (!m.user) total += m.data.size();
        return total;
    }
    // Copies every user-backed message still queued (the caller's buffer is about to be released).
    void keep_copies() {
        for (auto& m : queue_) {
            if (!m.user) continue;
            const auto* old_begin = m.data.data();
            m.owned.assign(m.data.begin(), m.data.end());
            m.data = m.owned;
            m.user = false;
            counters_.retained.fetch_add(m.data.size(), std::memory_order_relaxed);
            if (cur_.busy && cur_.data && !cur_.payload.empty() && cur_.payload.data() >= old_begin &&
                cur_.payload.data() < old_begin + m.owned.size())
                cur_.payload = std::span<const std::byte>(m.owned.data() + (cur_.payload.data() - old_begin), cur_.payload.size());
        }
    }
    void release_acked() {
        std::uint64_t acked = 0;
        {
            std::lock_guard lock(ctl_mutex_);
            acked = acked_;
        }
        while (!queue_.empty()) {
            auto& m = queue_.front();
            const auto end = m.first + m.pieces;
            const bool written = written_through_ >= end && !(cur_.busy && cur_.data && cur_.seq < end);
            // Pieces without flag_ack leave once written (nothing will acknowledge them alone).
            if (!(written && (!m.ack ? true : acked >= end))) break;
            for (auto it = attempts_.begin(); it != attempts_.end() && it->first < end;) it = attempts_.erase(it);
            queue_.pop_front();
        }
    }
    const TxMessage* find(std::uint64_t seq) const {
        for (const auto& m : queue_)
            if (seq >= m.first && seq < m.first + m.pieces) return &m;
        return nullptr;
    }
    // Writes what it can (one unit at a time, whole units only). Returns whether bytes moved.
    bool step_tx() {
        std::lock_guard lock(tx_mutex_);
        bool moved = false;
        for (;;) {
            if (cur_.busy) {
                const std::size_t head = cur_.nheads * sizeof(Unit);
                const std::size_t total = head + cur_.payload.size();
                std::array<iovec, 2> iov{};
                int count = 0;
                if (cur_.done < head) {
                    iov[0] = iovec{reinterpret_cast<std::byte*>(cur_.heads.data()) + cur_.done, head - cur_.done};
                    count = 1;
                    if (!cur_.payload.empty()) iov[count++] = iovec{const_cast<std::byte*>(cur_.payload.data()), cur_.payload.size()};
                } else {
                    const auto off = cur_.done - head;
                    iov[0] = iovec{const_cast<std::byte*>(cur_.payload.data() + off), cur_.payload.size() - off};
                    count = 1;
                }
                const auto n = ::writev(fd_, iov.data(), count);
                if (n < 0) {
                    if (errno == EAGAIN || errno == EINTR) return moved;
                    fail("write: " + errno_text(errno));
                }
                if (n == 0) fail("write returned 0");
                moved = true;
                last_tx_ = clock::now();
                cur_.done += static_cast<std::size_t>(n);
                if (cur_.done < total) continue;
                cur_.busy = false;
                bump();  // a writer waiting for send_written() while another thread wrote its unit
                if (cur_.data) {
                    counters_.pieces_out.fetch_add(1, std::memory_order_relaxed);
                    counters_.bytes_out.fetch_add(cur_.payload.size(), std::memory_order_relaxed);
                    written_through_ = std::max(written_through_, cur_.seq + 1U);
                    if (written_through_ >= next_seq_) {
                        std::lock_guard c(ctl_mutex_);
                        last_ack_change_ = clock::now();  // the acknowledgement clock starts with the last piece
                    }
                }
                release_acked();
                continue;
            }
            // Unit boundary: control first (the peer may be waiting on it), then resends, then new pieces.
            if (ctl_nack_out_.load(std::memory_order_acquire)) {
                Unit u;
                u.kind = kind_nack;
                {
                    std::lock_guard c(ctl_mutex_);
                    u.seq = nack_seq_;
                    u.epoch = nack_epoch_;
                    ctl_nack_out_.store(false, std::memory_order_release);
                }
                start_unit(sealed(u), {}, false, 0);
                continue;
            }
            if (ctl_ack_out_.load(std::memory_order_acquire)) {
                Unit u;
                u.kind = kind_ack;
                {
                    std::lock_guard c(ctl_mutex_);
                    u.seq = ack_value_;
                    ctl_ack_out_.store(false, std::memory_order_release);
                    ctl_lazy_ack_.store(false, std::memory_order_release);
                }
                start_unit(sealed(u), {}, false, 0);
                continue;
            }
            if (ctl_lazy_ack_.load(std::memory_order_acquire) && send_seq_ >= next_seq_.load()) {
                std::lock_guard c(ctl_mutex_);
                if (clock::now() - lazy_since_ >= std::chrono::milliseconds(1)) {
                    ctl_lazy_ack_.store(false, std::memory_order_release);
                    ack_value_ = std::max(ack_value_, lazy_value_);
                    ctl_ack_out_.store(true, std::memory_order_release);
                    continue;
                }
            }
            if (ctl_resend_.load(std::memory_order_acquire)) {
                std::uint64_t from = 0;
                bool damage = false;
                {
                    std::lock_guard c(ctl_mutex_);
                    from = resend_from_;
                    damage = resend_damage_;
                    ctl_resend_.store(false, std::memory_order_release);
                }
                if (from > next_seq_) fail("the peer asks for " + piece_text(from) + ", beyond the last sent (" + std::to_string(next_seq_.load()) + ")");
                if (from < next_seq_ && !find(from))
                    fail("the peer lost or received damaged " + piece_text(from) +
                         " of a small message sent without a copy (not repairable; STRIX_DIST_USB4_EAGER=0 sends every message resendable)");
                if (from < next_seq_) {
                    const auto* m = find(from);
                    if (!m->ack)
                        fail("the peer lost or received damaged " + piece_text(from) +
                             " of a small message sent without a copy (not repairable; STRIX_DIST_USB4_EAGER=0 sends every message resendable)");
                    if (damage && ++attempts_[from] > opt_.attempts)
                        fail(piece_text(from) + " still damaged after " + std::to_string(opt_.attempts) + " resends (the link loses data steadily)");
                }
                Unit u;
                u.kind = kind_resync;
                u.epoch = ++tx_epoch_;
                u.seq = from;
                start_unit(sealed(u), {}, false, 0);
                if (from < send_seq_) {
                    std::uint64_t bytes = 0;
                    for (auto s = from; s < send_seq_; ++s)
                        if (const auto* m = find(s)) bytes += piece_span(*m, s).size();
                    counters_.resends.fetch_add(send_seq_ - from, std::memory_order_relaxed);
                    counters_.resent_bytes.fetch_add(bytes, std::memory_order_relaxed);
                    send_seq_ = from;
                }
                continue;
            }
            if (send_seq_ < next_seq_) {
                const auto* m = find(send_seq_);
                if (!m) fail("internal: " + piece_text(send_seq_) + " is not queued");
                const auto payload = piece_span(*m, send_seq_);
                Unit u;
                u.kind = kind_data;
                u.flags = (m->ack ? flag_ack : 0U) | (m->lazy ? flag_lazy : 0U);
                u.seq = send_seq_;
                u.message = m->data.size();
                u.offset = (send_seq_ - m->first) * opt_.piece;
                u.size = static_cast<std::uint32_t>(payload.size());
                u.payload_crc = crc32c(payload);
                const auto it = attempts_.find(send_seq_);
                u.attempt = it == attempts_.end() ? 0U : it->second;
                start_unit(sealed(u), payload, true, send_seq_);
                if (ctl_lazy_ack_.load(std::memory_order_acquire)) {  // a pending lazy ACK rides in front, same writev
                    std::lock_guard c(ctl_mutex_);
                    if (ctl_lazy_ack_.exchange(false, std::memory_order_acq_rel)) {
                        Unit a;
                        a.kind = kind_ack;
                        a.seq = lazy_value_;
                        cur_.heads[1] = cur_.heads[0];
                        cur_.heads[0] = sealed(a);
                        cur_.nheads = 2;
                    }
                }
                ++send_seq_;
                continue;
            }
            return moved;
        }
    }
    std::span<const std::byte> piece_span(const TxMessage& m, std::uint64_t seq) const {
        const auto offset = static_cast<std::size_t>(seq - m.first) * opt_.piece;
        return m.data.subspan(offset, std::min(opt_.piece, m.data.size() - offset));
    }
    void start_unit(const Unit& header, std::span<const std::byte> payload, bool data, std::uint64_t seq) {
        cur_.heads[0] = header;
        cur_.nheads = 1;
        cur_.payload = payload;
        cur_.done = 0;
        cur_.busy = true;
        cur_.data = data;
        cur_.seq = seq;
    }

    // ---- receive side (rx_mutex_ held) ----
    // Bytes already read but not yet parsed (after a scan) come first.
    long read_some(std::byte* into, std::size_t size) {
        if (carry_pos_ < carry_.size()) {
            const auto n = std::min(size, carry_.size() - carry_pos_);
            std::memcpy(into, carry_.data() + carry_pos_, n);
            carry_pos_ += n;
            if (carry_pos_ == carry_.size()) {
                carry_.clear();
                carry_pos_ = 0;
            }
            return static_cast<long>(n);
        }
        const auto n = ::read(fd_, into, size);
        if (n == 0) fail("the peer closed the stream (its rank failed or exited)");
        if (n < 0) {
            if (errno == EAGAIN || errno == EINTR) return -1;
            fail("read: " + errno_text(errno));
        }
        last_rx_ = clock::now();
        return static_cast<long>(n);
    }
    void lazy_ack() {
        std::lock_guard c(ctl_mutex_);
        lazy_value_ = expected_;
        if (!ctl_lazy_ack_.exchange(true, std::memory_order_acq_rel)) lazy_since_ = clock::now();
    }
    void request_ack() {
        std::lock_guard c(ctl_mutex_);
        ack_value_ = expected_;
        ctl_ack_out_.store(true, std::memory_order_release);
    }
    // Damage seen: ask for everything again from the expected piece and drop bytes until the sender's RESYNC.
    void start_scan(std::span<const std::byte> already, const char* why) {
        mode_ = Mode::scan;
        parked_ = false;
        scan_.assign(already.begin(), already.end());
        scan_since_ = clock::now();
        if (++nack_rounds_ > opt_.attempts + 2U)
            fail("cannot repair " + piece_text(expected_) + " after " + std::to_string(nack_rounds_ - 1U) + " requests (last: " + why + ")");
        counters_.nacks_out.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard c(ctl_mutex_);
        nack_seq_ = expected_;
        nack_epoch_ = rx_epoch_;
        ctl_nack_out_.store(true, std::memory_order_release);
    }
    void on_control(const Unit& u) {
        if (u.kind == kind_ack) {
            std::lock_guard c(ctl_mutex_);
            if (u.seq > acked_) {
                acked_ = u.seq;
                last_ack_change_ = clock::now();
                ack_wait_ = opt_.ack;
            }
        } else if (u.kind == kind_nack) {
            counters_.nacks_in.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard c(ctl_mutex_);
            if (!ctl_resend_.load(std::memory_order_relaxed) || u.seq < resend_from_) resend_from_ = u.seq;
            resend_damage_ = true;
            ctl_resend_.store(true, std::memory_order_release);
        }
        bump();
    }
    // Scan mode: look for a valid unit header at any offset; a RESYNC of a newer round ends the scan.
    bool scan() {
        bool moved = false;
        for (;;) {
            const auto keep = scan_.size();
            scan_.resize(keep + (std::size_t{64} << 10U));
            const auto n = read_some(scan_.data() + keep, scan_.size() - keep);
            scan_.resize(keep + (n > 0 ? static_cast<std::size_t>(n) : 0U));
            if (n > 0) moved = true;
            std::size_t p = 0;
            while (p + sizeof(Unit) <= scan_.size()) {
                if (std::memcmp(scan_.data() + p, &unit_magic, sizeof unit_magic) != 0) {
                    ++p;
                    continue;
                }
                Unit u;
                std::memcpy(&u, scan_.data() + p, sizeof u);
                if (!valid(u)) {
                    ++p;
                    continue;
                }
                if (u.kind == kind_ack || u.kind == kind_nack) {
                    on_control(u);
                    p += sizeof(Unit);
                    continue;
                }
                if (u.kind == kind_resync && u.epoch > rx_epoch_) {
                    rx_epoch_ = u.epoch;
                    // What follows the RESYNC is the next unit: parse it normally.
                    std::vector<std::byte> rest(scan_.begin() + static_cast<std::ptrdiff_t>(p + sizeof(Unit)), scan_.end());
                    rest.insert(rest.end(), carry_.begin() + static_cast<std::ptrdiff_t>(carry_pos_), carry_.end());
                    carry_ = std::move(rest);
                    carry_pos_ = 0;
                    scan_.clear();
                    mode_ = Mode::header;
                    hdone_ = 0;
                    return true;
                }
                p += 1;
            }
            // Keep the bytes that may start a header cut by the read.
            const auto tail = std::min<std::size_t>(scan_.size(), sizeof(Unit) - 1U);
            scan_.erase(scan_.begin(), scan_.end() - static_cast<std::ptrdiff_t>(tail));
            if (n <= 0) break;
        }
        if (clock::now() - scan_since_ > 2 * opt_.loss) {  // the RESYNC (or our NACK) was lost too: ask again
            const std::vector<std::byte> kept = scan_;
            start_scan(kept, "no resend arrived");
        }
        return moved;
    }
    bool pump_rx() {
        bool moved = false;
        // Silence only counts while somebody watches: after a gap without pumping, the loss clocks restart.
        const auto entered = clock::now();
        if (entered - last_pump_ > opt_.loss / 4) {
            last_rx_ = entered;
            if (mode_ == Mode::scan) scan_since_ = entered;
        }
        last_pump_ = entered;
        for (;;) {
            if (mode_ == Mode::scan) {
                const bool m = scan();
                moved |= m;
                if (mode_ == Mode::scan) return moved;
                continue;
            }
            if (mode_ == Mode::header) {
                if (parked_) {
                    if (!posted_on_ || posted_complete_) return moved;
                    parked_ = false;
                    if (!start_data()) continue;
                    continue;
                }
                std::size_t spec = 0, spec_at = 0;
                const auto n = read_header(spec, spec_at);
                if (n < 0) return check_loss(moved);
                moved = true;
                hdone_ += static_cast<std::size_t>(n);
                if (hdone_ < sizeof(Unit)) continue;
                hdone_ = 0;
                if (spec) settle_speculation(spec, spec_at);
                if (!valid(hbuf_)) {
                    counters_.header_failures.fetch_add(1, std::memory_order_relaxed);
                    const std::span<const std::byte> bytes(reinterpret_cast<const std::byte*>(&hbuf_) + 1, sizeof(Unit) - 1U);
                    start_scan(bytes, "bad unit header");
                    continue;
                }
                if (hbuf_.kind == kind_ack || hbuf_.kind == kind_nack) {
                    on_control(hbuf_);
                    continue;
                }
                if (hbuf_.kind == kind_resync) {
                    if (hbuf_.epoch > rx_epoch_) rx_epoch_ = hbuf_.epoch;
                    continue;
                }
                unit_ = hbuf_;
                if (unit_.seq < expected_) {  // a resend of a piece already accepted (its ACK was late or lost)
                    counters_.duplicates.fetch_add(1, std::memory_order_relaxed);
                    discarding_ = true;
                    pdone_ = 0;
                    mode_ = Mode::payload;
                    continue;
                }
                if (unit_.seq > expected_) {
                    counters_.gaps.fetch_add(1, std::memory_order_relaxed);
                    start_scan({}, "a piece is missing");
                    continue;
                }
                if (in_size_ == 0U) {
                    if (unit_.offset != 0U) {
                        counters_.gaps.fetch_add(1, std::memory_order_relaxed);
                        start_scan({}, "a message started mid-way");
                        continue;
                    }
                    if (!posted_on_ || posted_complete_) {  // the next message, before this rank asks for it: wait here
                        parked_ = true;
                        return moved;
                    }
                }
                if (!start_data()) continue;
                pdone_ = std::exchange(spec_payload_, std::size_t{0});
                continue;
            }
            // payload
            const auto want = static_cast<std::size_t>(unit_.size) - pdone_;
            if (want) {
                std::byte* into = discarding_ ? discard_.data() : posted_.data() + unit_.offset + pdone_;
                const auto n = read_some(into, discarding_ ? std::min(want, discard_.size()) : want);
                if (n < 0) return check_loss(moved);
                moved = true;
                pdone_ += static_cast<std::size_t>(n);
                if (pdone_ < unit_.size) continue;
            }
            mode_ = Mode::header;
            if (discarding_) {
                discarding_ = false;
                if (unit_.flags & flag_ack) request_ack();
                continue;
            }
            const auto piece = std::span<const std::byte>(posted_.data() + unit_.offset, unit_.size);
            if (crc32c(piece) != unit_.payload_crc) {
                counters_.crc_failures.fetch_add(1, std::memory_order_relaxed);
                if (!(unit_.flags & flag_ack))  // the sender kept no copy: say so now instead of asking in vain
                    fail("damaged " + piece_text(unit_.seq) + " (payload CRC) of a small message sent without a copy: not repairable "
                         "(STRIX_DIST_USB4_SMALL_RESEND=0)");
                start_scan({}, "payload CRC mismatch");
                continue;
            }
            ++expected_;
            nack_rounds_ = 0;
            in_done_ += unit_.size;
            counters_.pieces_in.fetch_add(1, std::memory_order_relaxed);
            counters_.bytes_in.fetch_add(unit_.size, std::memory_order_relaxed);
            if (in_done_ == in_size_) {
                in_size_ = 0;
                in_done_ = 0;
                posted_complete_ = true;
                posted_on_ = false;
            }
            if (unit_.flags & flag_ack) {
                if (unit_.flags & flag_lazy) lazy_ack();
                else request_ack();
            }
            bump();
        }
    }
    // Reads (the rest of) a unit header. With a receive posted and nothing carried over, one readv also reads what follows
    // the header straight into the place the next expected piece of the posted message goes (most units are that piece):
    // `spec` bytes landed at posted_[spec_at...]; settle_speculation() keeps them or hands them to carry_.
    long read_header(std::size_t& spec, std::size_t& spec_at) {
        auto* head = reinterpret_cast<std::byte*>(&hbuf_) + hdone_;
        const std::size_t want = sizeof(Unit) - hdone_;
        if (carry_pos_ < carry_.size() || !posted_on_ || posted_complete_) return read_some(head, want);
        spec_at = in_size_ ? static_cast<std::size_t>(in_done_) : 0U;
        const std::size_t room = std::min(posted_.size() - spec_at, opt_.piece);
        std::array<iovec, 2> iov{iovec{head, want}, iovec{posted_.data() + spec_at, room}};
        const auto n = ::readv(fd_, iov.data(), room ? 2 : 1);
        if (n == 0) fail("the peer closed the stream (its rank failed or exited)");
        if (n < 0) {
            if (errno == EAGAIN || errno == EINTR) return -1;
            fail("read: " + errno_text(errno));
        }
        last_rx_ = clock::now();
        if (static_cast<std::size_t>(n) <= want) return static_cast<long>(n);
        spec = static_cast<std::size_t>(n) - want;
        return static_cast<long>(want);
    }
    void settle_speculation(std::size_t spec, std::size_t spec_at) {
        const Unit& u = hbuf_;
        const bool ours = valid(u) && u.kind == kind_data && u.seq == expected_ && u.offset == spec_at &&
                          (in_size_ == 0U ? u.message == posted_.size() : u.message == in_size_);
        const std::size_t keep = ours ? std::min<std::size_t>(spec, u.size) : 0U;
        if (spec > keep)  // the next unit(s) began in the posted bytes: parse them from carry_
            carry_.assign(posted_.data() + spec_at + keep, posted_.data() + spec_at + spec);
        carry_pos_ = 0;
        spec_payload_ = keep;
    }
    // The DATA unit in unit_ (expected piece) goes into the posted message. False when it was refused (scan started).
    bool start_data() {
        if (in_size_ == 0U) {
            if (unit_.message != posted_.size())
                fail("message size mismatch: the peer sends " + std::to_string(unit_.message) + " bytes, this rank expects " +
                     std::to_string(posted_.size()) + " (" + piece_text(unit_.seq) + ")");
            in_size_ = unit_.message;
            in_done_ = 0;
        } else if (unit_.message != in_size_ || unit_.offset != in_done_) {
            counters_.gaps.fetch_add(1, std::memory_order_relaxed);
            start_scan({}, "a piece out of place");
            return false;
        }
        pdone_ = 0;
        discarding_ = false;
        mode_ = Mode::payload;
        return true;
    }
    // Nothing to read now. Half a unit that stops arriving is a lost frame: ask for a resend.
    bool check_loss(bool moved) {
        const bool mid_unit = (mode_ == Mode::header && hdone_ > 0U) || mode_ == Mode::payload;
        if (!moved && mid_unit && clock::now() - last_rx_ > opt_.loss) {
            counters_.losses.fetch_add(1, std::memory_order_relaxed);
            hdone_ = 0;
            discarding_ = false;
            start_scan({}, "a unit stopped arriving (lost frame)");
            return true;
        }
        return moved;
    }

    // ---- waiting ----
    void wait_io(short events, std::chrono::milliseconds slice) {
        if (opt_.spin) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
            return;
        }
        pollfd p{fd_, events, 0};
        (void)::poll(&p, 1, static_cast<int>(slice.count()));
    }
    // Everything written, retained pieces unacknowledged for a while: resend from the first one (the tail, or its
    // ACK, was lost). Backs off; not counted as damage (a peer that is only slow to read must not fail).
    void check_ack_timeout() {
        std::uint64_t first = 0;
        {
            std::lock_guard lock(tx_mutex_);
            if (queue_.empty() || send_seq_ < next_seq_.load() || cur_.busy) return;
            first = queue_.front().first;  // pieces before it left the queue: written without a copy, or acknowledged
        }
        std::lock_guard c(ctl_mutex_);
        if (acked_ >= next_seq_.load() || clock::now() - last_ack_change_ < ack_wait_) return;
        counters_.ack_timeouts.fetch_add(1, std::memory_order_relaxed);
        last_ack_change_ = clock::now();
        ack_wait_ = std::min(ack_wait_ * 2, opt_.timeout);
        const auto from = std::max(acked_, first);
        if (!ctl_resend_.load(std::memory_order_relaxed) || from < resend_from_) resend_from_ = from;
        resend_damage_ = false;
        ctl_resend_.store(true, std::memory_order_release);
    }
    // One round of a blocking call: progress, then wait for the stream or for the other thread's pump.
    // Several threads may drive one link (a stream's writer and reader). One of them at a time sleeps in poll() and is
    // then the only reader of the descriptor; the others sleep on the condition variable until a pump, a written unit
    // or the poller's return changes something (bump), or poll() themselves for write room only.
    void wait_round(bool need_rx, clock::time_point& deadline, const char* what) {
        const auto seen = gen_.load();
        const bool moved = progress(need_rx);
        check_ack_timeout();
        const auto now = clock::now();
        if (moved) {
            deadline = now + opt_.timeout;
            return;
        }
        if (now >= deadline) fail(std::string(what) + ": no progress within " + std::to_string(opt_.timeout.count()) + " ms (" + state() + ")");
        if (opt_.spin) {
            wait_io(POLLIN, std::chrono::milliseconds(0));
            return;
        }
        const auto slice = std::chrono::duration_cast<std::chrono::milliseconds>(std::min<clock::duration>(deadline - now, std::chrono::milliseconds(20))) +
                           std::chrono::milliseconds(1);
        bool poller = false;
        {
            std::lock_guard lock(rx_mutex_);  // no pump runs between this check and the flag
            if (gen_.load() != seen) return;
            poller = !polling_.exchange(true);
        }
        if (poller) {
            const short ev = events();
            if (gen_.load() == seen) wait_io(ev, slice);  // a unit written meanwhile (no rx lock) must not sleep here
            polling_.store(false);
            bump();
            return;
        }
        if (events() & POLLOUT) {  // write room is not taken by another thread's poll
            wait_io(POLLOUT, std::chrono::milliseconds(1));
            return;
        }
        waiters_.fetch_add(1);
        {
            std::unique_lock c(ctl_mutex_);
            ctl_cv_.wait_for(c, slice, [&] { return gen_.load() != seen; });
        }
        waiters_.fetch_sub(1);
    }
    // Something changed (input pumped, a unit written, a poller returned, a user left): waiters re-check.
    void bump() {
        gen_.fetch_add(1);
        if (waiters_.load() == 0) return;
        { std::lock_guard c(ctl_mutex_); }
        ctl_cv_.notify_all();
    }

    int fd_;
    std::string name_;
    Options opt_;
    const std::atomic<bool>* abort_ = nullptr;
    Counters counters_;

    // send side (tx_mutex_)
    std::mutex tx_mutex_;
    std::deque<TxMessage> queue_;
    std::atomic<std::uint64_t> next_seq_{0};
    std::uint64_t send_seq_ = 0, written_through_ = 0, sent_through_target_ = 0, tx_epoch_ = 0;
    std::map<std::uint64_t, unsigned> attempts_;
    TxUnit cur_;
    clock::time_point last_tx_{};

    // control (ctl_mutex_, leaf)
    std::mutex ctl_mutex_;
    std::condition_variable ctl_cv_;
    std::atomic<bool> ctl_ack_out_{false}, ctl_nack_out_{false}, ctl_resend_{false}, ctl_lazy_ack_{false};
    std::uint64_t acked_ = 0, ack_value_ = 0, nack_seq_ = 0, nack_epoch_ = 0, resend_from_ = 0, lazy_value_ = 0;
    clock::time_point lazy_since_{};
    bool resend_damage_ = false;
    clock::time_point last_ack_change_{};
    std::chrono::milliseconds ack_wait_{opt_.ack};

    // waiting (wait_round): change counter, the poll() owner flag, sleepers on ctl_cv_, threads driving the link
    std::atomic<std::uint64_t> gen_{0};
    std::atomic<bool> polling_{false};
    std::atomic<int> waiters_{0}, users_{0};

    // receive side (rx_mutex_)
    std::mutex rx_mutex_;
    Mode mode_ = Mode::header;
    Unit hbuf_{}, unit_{};
    std::size_t hdone_ = 0, pdone_ = 0;
    bool discarding_ = false, parked_ = false;
    std::vector<std::byte> discard_, carry_, scan_;
    std::size_t carry_pos_ = 0, spec_payload_ = 0;
    std::uint64_t expected_ = 0, rx_epoch_ = 0, in_size_ = 0, in_done_ = 0;
    unsigned nack_rounds_ = 0;
    std::span<std::byte> posted_;
    bool posted_on_ = false, posted_complete_ = false;
    clock::time_point last_rx_{}, scan_since_{}, last_pump_{};
};

}  // namespace strix::dist::link
