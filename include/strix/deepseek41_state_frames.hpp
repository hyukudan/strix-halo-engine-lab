#pragma once
// The receiving half of the streamed DeepSeek state hand-off (streaming protocol):
// one strix::handoff session carries the state file of deepseek41_state_pieces.hpp as kind 1 pieces (index = layer,
// that layer's record, any order of layers) and one kind 2 piece (head | tail, last). receive_state_file() drains the
// session and returns the very bytes serialize_attention_state() wrote on the sender (the caller deserializes them,
// which checks the file's checksum and digest). HELLO: model kModel, meta = u32 layers | u32 position | u32 reserve.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "strix/deepseek41_state_pieces.hpp"
#include "strix/handoff.hpp"

namespace strix::deepseek41::pieces {

inline constexpr const char* kModel = "deepseek-v4.1/attn-state-v1";
inline constexpr std::uint32_t kKindRecord = 1, kKindHeadTail = 2;

struct StateMeta {
    std::uint32_t layers = 0, position = 0, reserve_tokens = 0;
};

inline std::vector<std::uint8_t> meta_bytes(const StateMeta& m) {
    std::vector<std::uint8_t> v(12);
    std::memcpy(v.data(), &m, 12);
    return v;
}

inline StateMeta read_meta(const handoff::Hello& h) {
    if (h.model != kModel || h.meta.size() != 12U || h.world != 1U) throw std::runtime_error("not a deepseek-v4.1 attention state session");
    StateMeta m;
    std::memcpy(&m, h.meta.data(), 12);
    return m;
}

struct Received {
    std::vector<std::uint8_t> bytes;  // the state file
    StateMeta meta;
    double first_piece_s = -1.0, received_s = 0.0;
};

// Throws on a session error, a missing / repeated / out-of-range record, a second head-tail or an unexpected piece.
inline Received receive_state_file(handoff::Session& s, std::uint32_t layers) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    Received out;
    out.meta = read_meta(s.hello());
    if (out.meta.layers != layers) throw std::runtime_error("the state has " + std::to_string(out.meta.layers) + " layers, the server " + std::to_string(layers));
    std::vector<std::vector<std::uint8_t>> records(layers);
    std::vector<bool> got(layers, false);
    std::vector<std::uint8_t> head_tail;
    handoff::Piece p;
    while (s.next(p)) {
        if (out.first_piece_s < 0.0) out.first_piece_s = std::chrono::duration<double>(Clock::now() - t0).count();
        if (p.header.kind == kKindRecord) {
            if (p.header.index >= layers || got[p.header.index]) throw std::runtime_error("layer record out of range or repeated");
            got[p.header.index] = true;
            records[p.header.index].swap(p.plain);
        } else if (p.header.kind == kKindHeadTail && head_tail.empty() && p.plain.size() > kHeadBytes) {
            head_tail.swap(p.plain);
        } else {
            throw std::runtime_error("unexpected piece");
        }
    }
    for (std::uint32_t l = 0; l < layers; ++l)
        if (!got[l]) throw std::runtime_error("layer " + std::to_string(l) + " record missing");
    if (head_tail.empty()) throw std::runtime_error("head and tail missing");
    out.received_s = std::chrono::duration<double>(Clock::now() - t0).count();
    const auto ht = std::span<const std::uint8_t>(head_tail);
    out.bytes = assemble(ht.first(kHeadBytes), records, ht.subspan(kHeadBytes));
    return out;
}

// The node's frames listener: one session at a time, each handed to `serve` (which receives the state and hands the
// import to the server through publish()). Shutdown (destructor) and publication are serialized by one mutex: either
// shutdown comes first (publish() refuses, nothing reaches the server) or publication does (that session is not
// cancelled). The destructor stops the listener (also waking a stream still in its HELLO handshake), cancels the
// unpublished session being served (its next() throws at once) and joins. A session that throws is aborted with the
// reason (the sender's finish() reports it).
class FramesImportListener {
public:
    using Serve = std::function<void(handoff::Session&, FramesImportListener&)>;
    FramesImportListener(std::uint16_t port, Serve serve) : listener_(port), serve_(std::move(serve)) {
        thread_ = std::thread([this] { loop(); });
    }
    ~FramesImportListener() {
        {
            std::lock_guard<std::mutex> g(m_);
            stop_ = true;
            if (current_ && !published_) current_->cancel("server shutting down");
        }
        listener_.shutdown();
        if (thread_.joinable()) thread_.join();
    }
    FramesImportListener(const FramesImportListener&) = delete;
    FramesImportListener& operator=(const FramesImportListener&) = delete;
    [[nodiscard]] std::uint16_t port() const noexcept { return listener_.port(); }
    [[nodiscard]] bool stopping() {
        std::lock_guard<std::mutex> g(m_);
        return stop_;
    }
    // Runs `hand_over` (e.g. queue the import job) under the shutdown mutex unless shutdown began: false = refused.
    bool publish(const std::function<void()>& hand_over) {
        std::lock_guard<std::mutex> g(m_);
        if (stop_) return false;
        hand_over();
        published_ = true;
        return true;
    }
    [[nodiscard]] std::uint64_t sessions() const noexcept { return sessions_.load(); }

private:
    void loop() {
        while (!stopping()) {
            std::unique_ptr<handoff::Session> s;
            try {
                s = listener_.accept();
            } catch (const std::exception& e) {
                if (stopping()) return;
                std::fprintf(stderr, "import-frames: accept: %s\n", e.what());
                continue;
            }
            {
                std::lock_guard<std::mutex> g(m_);
                if (stop_) {
                    s->abort("server shutting down");
                    return;
                }
                current_ = s.get();
                published_ = false;
            }
            ++sessions_;
            try {
                serve_(*s, *this);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "import-frames: session failed: %s\n", e.what());
                s->abort(e.what());
            }
            std::lock_guard<std::mutex> g(m_);
            current_ = nullptr;
            published_ = false;
        }
    }

    handoff::Listener listener_;
    Serve serve_;
    std::mutex m_;
    bool stop_ = false;                    // under m_
    handoff::Session* current_ = nullptr;  // under m_
    bool published_ = false;               // under m_: current_'s import was handed to the server
    std::atomic<std::uint64_t> sessions_{0};
    std::thread thread_;
};

// The sending half's last piece: the head and tail of the final file, after checking that every record already sent
// (their XXH64 in `sent`, by layer) is the file's record.
inline std::vector<std::uint8_t> head_tail_after_check(std::span<const std::uint8_t> file, const std::vector<std::uint64_t>& sent) {
    const auto lay = split(file);
    if (lay.records.size() != sent.size()) throw std::runtime_error("the final state has another layer count");
    for (std::size_t l = 0; l < sent.size(); ++l) {
        handoff::Xxh64 h;
        h.update(lay.records[l].data(), lay.records[l].size());
        if (h.digest() != sent[l]) throw std::runtime_error("streamed layer " + std::to_string(l) + " differs from the final state");
    }
    std::vector<std::uint8_t> ht(lay.head.begin(), lay.head.end());
    ht.insert(ht.end(), lay.tail.begin(), lay.tail.end());
    return ht;
}

}  // namespace strix::deepseek41::pieces
