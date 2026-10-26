#pragma once

// The TAIL's side of a split decode with several requests (strix/ds41-pipeline-multi-v1, ds41_pipeline_multi.hpp), for
// strix-deepseek41-batch and strix-deepseek41-serve. A FRONT of sequence slots (strix-cuda-ds41-multi-front on the
// workstation GPU) runs layers [0, cut) of every request; the pair runs [cut, layers) and the head as one batch step
// (BatchForward::step_boundary). Rank 0 drives the FRONT: submit() sends one SEQ_STEP per request segment of a step as
// soon as its rows are known, step() takes the answers of the oldest submission, forwards them to rank 1 over the
// communicator (host path: a 64-byte header, then the payloads; rank 1 sends zeros of the same size) and runs the TAIL
// step on both ranks. Submitting a group's next rows before stepping another group keeps the FRONT and the pair busy
// at the same time (micro-batches in flight).
//
// Exactness: the FRONT runs each slot's rows as that request alone, the TAIL batch step keeps every row's arithmetic
// (BatchForward's invariance), so each request's tokens and logits are those of running it alone, split or whole.
//
// --hetero-local (gates on the pair, no FRONT process): the FRONT half runs on this BatchForward
// (BatchForward::step_front), its boundaries go through the wire's encode / decode, then the TAIL half runs from them;
// the step's rows end bitwise as BatchForward::step's.

#include "strix/deepseek41_attention_state.hpp"
#include "strix/deepseek41_batch_forward.hpp"
#include "strix/ds41_pipeline_multi.hpp"
#include "strix/ds41_pipeline_wire.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::ds41::hetero {

namespace pipe = strix::ds41::pipe;
using strix::deepseek41::BatchForward;
using strix::deepseek41::BatchRow;

struct StageOptions {
    std::uint32_t cut = 22;
    std::uint32_t rank = 0, world = 1;
    std::string listen;          // rank 0: HOST:PORT the FRONT connects to (empty with local)
    std::string token_file;      // the session secret (64 hex characters)
    std::uint32_t slots = 8;     // the FRONT must hold at least this many sequence slots
    std::chrono::seconds connect_wait{600};
    std::chrono::seconds timeout{120};  // every wait for a FRONT answer
    bool local = false;          // --hetero-local: the FRONT half on this forward
    std::string engine = "strix-deepseek41-batch";
};

struct StageStats {
    std::uint64_t steps = 0, segments = 0, rows = 0, bytes_in = 0, bytes_forwarded = 0;
    double wait_seconds = 0.0;     // rank 0: waiting for the FRONT's boundaries of a step (receive included)
    double forward_seconds = 0.0;  // the forward to rank 1 (both ranks)
    double tail_seconds = 0.0;     // the TAIL step (BatchForward::step_boundary)
};

class Stage {
public:
    // exchange: host bytes rank 0 <-> rank 1 (same size on both ranks), between forward steps; unused with world 1.
    using Exchange = std::function<void(std::span<const std::byte>, std::span<std::byte>)>;

    Stage(BatchForward& fwd, StageOptions o, Exchange exchange) : fwd_(fwd), o_(std::move(o)), exchange_(std::move(exchange)) {
        const auto& cfg = fwd_.model().config;
        geo_ = pipe::boundary_geometry(cfg);
        if (o_.cut == 0U || o_.cut >= cfg.num_hidden_layers) throw std::invalid_argument("--hetero-cut must be in [1, layers)");
    }
    Stage(const Stage&) = delete;
    Stage& operator=(const Stage&) = delete;

    // Rank 0: waits for the FRONT and checks its MULTI_HELLO (token, cut, geometry, model, slots, numeric identity).
    // World 2: both ranks call it; while rank 0 waits, a 64-byte host exchange every 5 s keeps rank 1 (and the link) in
    // step, so no rank sits in a device exchange past the transport's no-progress limit; it ends with rank 0's verdict.
    void connect() {
        if (o_.local) return;
        const auto beat = [&](std::uint32_t status) {  // 0 waiting, 1 connected, 2 failed (rank 1 gets rank 0's)
            if (o_.world != 2U) return status;
            std::array<std::byte, 64> out{}, in{};
            std::memcpy(out.data(), &status, 4);
            exchange_(out, in);
            if (o_.rank == 1U) std::memcpy(&status, in.data(), 4);
            return status;
        };
        if (o_.rank != 0U) {
            std::uint32_t st = 0;
            while ((st = beat(0U)) == 0U) {
            }
            if (st != 1U) throw std::runtime_error("hetero: rank 0 has no FRONT");
            return;
        }
        const auto [host, port] = pipe::split_host_port(o_.listen);
        listener_ = std::make_unique<pipe::Listener>(host, port);
        std::printf("hetero: waiting for the FRONT on %s (cut %u, %u slots)\n", o_.listen.c_str(), o_.cut, o_.slots);
        std::fflush(stdout);
        const auto token = pipe::read_token_file(o_.token_file);
        const auto deadline = std::chrono::steady_clock::now() + o_.connect_wait;
        while (!conn_ && std::chrono::steady_clock::now() < deadline) {
            conn_ = listener_->accept(std::chrono::milliseconds(5000), std::chrono::duration_cast<std::chrono::milliseconds>(o_.timeout));
            if (!conn_) (void)beat(0U);
        }
        try {
            if (!conn_) throw std::runtime_error("hetero: no FRONT connected");
            accept_front(token);
            listener_.reset();  // one FRONT per session: a FRONT's next session must not reach this process
        } catch (...) {
            (void)beat(2U);
            throw;
        }
        (void)beat(1U);
    }

private:
    void accept_front(const std::array<std::uint8_t, 32>& token) {
        const auto first = conn_->receive(1U << 20U);
        if (first.header.type != pipe::MsgType::multi_hello) throw pipe::WireError("hetero: expected MULTI_HELLO");
        const auto mh = pipe::decode_multi_hello(first.payload);
        const auto& h = mh.hello;
        const auto& c = fwd_.model().config;
        const auto identity = fwd_.model().manifest.layout_identity;
        pipe::HelloAck ack;
        ack.first_layer = o_.cut;
        ack.end_layer = c.num_hidden_layers;
        ack.world = o_.world;
        ack.max_positions = h.max_positions;
        ack.model_identity = identity;
        ack.engine = o_.engine;
        ack.numeric_identity = fwd_.expert_tp() ? "ds41-expert-tp2-i1152-v1" : "ds41-exact-v1";
        if (!pipe::same_token(h.token, token)) ack.status = 1, ack.message = "wrong session token";
        else if (h.cut != o_.cut) ack.status = 2, ack.message = "the FRONT cuts at " + std::to_string(h.cut) + ", this TAIL starts at " + std::to_string(o_.cut);
        else if (h.layers != c.num_hidden_layers || h.hidden != c.hidden_size || h.hc != c.hc_mult || h.vocab != c.vocab_size)
            ack.status = 3, ack.message = "model geometry or layer count differs";
        else if (h.model_identity != identity) ack.status = 4, ack.message = "the model files differ (layout identity)";
        else if (mh.slots < o_.slots) ack.status = 5, ack.message = "the FRONT holds " + std::to_string(mh.slots) + " slots, " + std::to_string(o_.slots) + " needed";
        else if (h.numeric_identity.rfind(ack.numeric_identity, 0) != 0)
            ack.status = 6, ack.message = "numeric identity " + h.numeric_identity + ", the TAIL runs " + ack.numeric_identity;
        conn_->send(pipe::MsgType::hello_ack, first.header.request, pipe::encode(ack));
        std::printf("hetero: FRONT %s backend %s (%s), %u slots x %u rows, max positions %u: %s\n", conn_->peer().c_str(), h.backend.c_str(),
                    h.numeric_identity.c_str(), mh.slots, mh.max_rows, h.max_positions, ack.status == 0U ? "accepted" : ack.message.c_str());
        std::fflush(stdout);
        if (ack.status != 0U) throw std::runtime_error("hetero: FRONT refused: " + ack.message);
        front_rows_ = mh.max_rows;
        front_positions_ = h.max_positions;
        conn_->set_busy_poll(true);
    }

public:
    [[nodiscard]] std::uint32_t front_positions() const noexcept { return front_positions_; }

    // The rows of a later step (call order = step order). commit(slot, kept) before it: the FRONT drops the rows of the
    // slot's previous step after the first `kept` (a verify window), as BatchForward::commit does here.
    void submit(std::span<const BatchRow> rows) {
        if (o_.local) return;
        for (const auto& g : segments(rows)) {
            if (o_.rank == 0U) {
                pipe::SeqStep q;
                q.slot = g.slot;
                q.position = g.position;
                q.tokens = g.tokens;
                const auto it = commits_.find(g.slot);
                if (it != commits_.end()) q.commit = it->second;
                conn_->send(pipe::MsgType::seq_step, next_request_, pipe::encode(q));
                pending_.push_back({pipe::MsgType::seq_step, next_request_++, g.slot, g.position, static_cast<std::uint32_t>(g.tokens.size())});
            }
            commits_.erase(g.slot);
        }
    }
    void commit(std::uint32_t slot, std::uint32_t kept) {
        if (!o_.local) commits_[slot] = kept;
    }
    // A slot's new sequence at position 0 (the TAIL's slot is reset by the caller).
    void reset(std::uint32_t slot) {
        commits_.erase(slot);
        if (o_.local || o_.rank != 0U) return;
        conn_->send(pipe::MsgType::seq_reset, next_request_, pipe::encode_u32(slot));
        pending_.push_back({pipe::MsgType::seq_reset, next_request_++, slot, 0U, 0U});
    }
    // A slot continues `state` (a prefill's, all layers): the FRONT keeps its layers [0, cut) (the TAIL's slot imports the
    // same state, by the caller).
    void import_state(std::uint32_t slot, const strix::deepseek41::AttentionState& state) {
        commits_.erase(slot);
        if (o_.local || o_.rank != 0U) return;
        pipe::StateMsg sm;
        sm.manifest = "{\"layers\":[0," + std::to_string(state.layers.size()) + "],\"position\":" + std::to_string(state.position) + "}";
        sm.state = strix::deepseek41::serialize_attention_state(state);
        conn_->send(pipe::MsgType::seq_import, next_request_, pipe::encode_seq_import(slot, sm));
        pending_.push_back({pipe::MsgType::seq_import, next_request_++, slot, 0U, 0U});
    }

    // The TAIL step of `rows` (submitted earlier, the oldest submission still pending) on both ranks.
    const std::vector<std::uint32_t>& step(std::span<const BatchRow> rows) {
        const auto segs = segments(rows);
        if (o_.local) {  // the FRONT half here, its boundaries through the wire's encode / decode
            auto front = fwd_.step_front(rows, o_.cut);
            payloads_.clear();
            for (const auto& b : front) {
                pipe::StepMsg sm;
                sm.boundary = b;
                payloads_.push_back(pipe::encode(sm));
            }
        } else if (o_.rank == 0U) {
            const auto t0 = std::chrono::steady_clock::now();
            payloads_.clear();
            for (const auto& g : segs) {
                for (;;) {  // the acknowledgements of earlier resets / imports first
                    if (pending_.empty()) throw std::logic_error("hetero: a step whose rows were not submitted");
                    const auto want = pending_.front();
                    auto m = conn_->receive();
                    bytes_in_ += pipe::kHeaderBytes + m.payload.size();
                    if (m.header.type == pipe::MsgType::error) throw std::runtime_error("hetero: FRONT error: " + pipe::decode_text(m.payload));
                    if (m.header.request != want.request) throw pipe::WireError("hetero: FRONT answer out of order");
                    pending_.pop_front();
                    if (want.type != pipe::MsgType::seq_step) {
                        if (m.header.type != pipe::MsgType::ack) throw pipe::WireError("hetero: expected ACK from the FRONT");
                        continue;
                    }
                    if (m.header.type != pipe::MsgType::seq_boundary || m.header.flags != g.slot || want.slot != g.slot || want.position != g.position ||
                        want.rows != g.tokens.size())
                        throw pipe::WireError("hetero: the FRONT's boundary is not the step's segment of slot " + std::to_string(g.slot));
                    payloads_.push_back(std::move(m.payload));
                    break;
                }
            }
            stats_.wait_seconds += seconds_since(t0);
        }
        if (!o_.local && o_.world == 2U) forward();
        boundaries_.clear();
        for (const auto& p : payloads_) {
            auto sm = pipe::decode_step(p, geo_, BatchForward::max_rows, true);
            boundaries_.push_back(std::move(sm.boundary));
        }
        const auto t1 = std::chrono::steady_clock::now();
        const auto& tokens = fwd_.step_boundary(rows, o_.cut, boundaries_);
        stats_.tail_seconds += seconds_since(t1);
        ++stats_.steps;
        stats_.segments += segs.size();
        stats_.rows += rows.size();
        stats_.bytes_in = bytes_in_;
        return tokens;
    }

    // Rank 0: ends the FRONT session (CLOSE, its ACK).
    void close() {
        if (o_.local || o_.rank != 0U || !conn_) return;
        try {
            conn_->send(pipe::MsgType::close, next_request_++, {});
            for (;;) {
                const auto m = conn_->receive();
                if (m.header.type == pipe::MsgType::ack && pending_.empty()) break;
                if (!pending_.empty()) pending_.pop_front();
            }
        } catch (const std::exception&) {
        }
        conn_.reset();
        listener_.reset();  // a FRONT reconnecting for its next session must not reach this ending process
    }
    [[nodiscard]] const StageStats& stats() const noexcept { return stats_; }
    [[nodiscard]] std::uint32_t cut() const noexcept { return o_.cut; }
    [[nodiscard]] bool local() const noexcept { return o_.local; }

private:
    struct Segment {
        std::uint32_t slot = 0, position = 0;
        std::vector<std::uint32_t> tokens;
    };
    struct Pending {
        pipe::MsgType type;
        std::uint64_t request = 0;
        std::uint32_t slot = 0, position = 0, rows = 0;
    };
    static double seconds_since(std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    }
    // the runs of equal slots of `rows` (a slot's rows are consecutive, from its position)
    std::vector<Segment> segments(std::span<const BatchRow> rows) const {
        std::vector<Segment> out;
        for (const auto& r : rows) {
            if (out.empty() || out.back().slot != r.slot) out.push_back({r.slot, fwd_.position(r.slot), {}});
            out.back().tokens.push_back(r.token);
        }
        return out;
    }
    // rank 0 -> rank 1: the step's boundary payloads (header: magic, count, total bytes, every payload's size)
    void forward() {
        const auto t0 = std::chrono::steady_clock::now();
        struct Head {
            std::uint32_t magic = 0x31464248U;  // "HBF1"
            std::uint32_t count = 0;
            std::uint64_t bytes = 0;
            std::uint32_t sizes[12] = {};
        } head;
        static_assert(sizeof(Head) == 64U);
        if (o_.rank == 0U) {
            if (payloads_.size() > 12U) throw std::logic_error("hetero: more than 12 segments in a step");
            head.count = static_cast<std::uint32_t>(payloads_.size());
            for (std::size_t i = 0; i < payloads_.size(); ++i) {
                head.sizes[i] = static_cast<std::uint32_t>(payloads_[i].size());
                head.bytes += payloads_[i].size();
            }
        }
        std::array<std::byte, 64> out{}, in{};
        std::memcpy(out.data(), &head, 64);
        exchange_(out, in);
        if (o_.rank == 1U) {
            std::memcpy(&head, in.data(), 64);
            if (head.magic != 0x31464248U || head.count > 12U) throw std::runtime_error("hetero: bad forward header from rank 0");
        }
        std::vector<std::byte> send(o_.rank == 0U ? head.bytes : head.bytes), recv(o_.rank == 1U ? head.bytes : 0U);
        if (o_.rank == 0U) {
            std::size_t at = 0;
            for (const auto& p : payloads_) {
                std::memcpy(send.data() + at, p.data(), p.size());
                at += p.size();
            }
        }
        if (head.bytes != 0U) exchange_(send, o_.rank == 1U ? std::span<std::byte>(recv) : std::span<std::byte>{});
        if (o_.rank == 1U) {
            payloads_.clear();
            std::size_t at = 0;
            for (std::uint32_t i = 0; i < head.count; ++i) {
                const auto* b = reinterpret_cast<const std::uint8_t*>(recv.data()) + at;
                payloads_.emplace_back(b, b + head.sizes[i]);
                at += head.sizes[i];
            }
        }
        stats_.bytes_forwarded += head.bytes;
        stats_.forward_seconds += seconds_since(t0);
    }

    BatchForward& fwd_;
    StageOptions o_;
    Exchange exchange_;
    pipe::BoundaryGeometry geo_;
    std::unique_ptr<pipe::Listener> listener_;
    std::unique_ptr<pipe::Connection> conn_;
    std::deque<Pending> pending_;
    std::map<std::uint32_t, std::uint32_t> commits_;
    std::uint64_t next_request_ = 2;
    std::uint32_t front_rows_ = 0, front_positions_ = 0;
    std::uint64_t bytes_in_ = 0;
    std::vector<std::vector<std::uint8_t>> payloads_;
    std::vector<pipe::Boundary> boundaries_;
    StageStats stats_;
};

}  // namespace strix::ds41::hetero
