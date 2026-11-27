#pragma once
// N-rank layer pipeline (model-agnostic): the stages of a contiguous layer split on a line of ranks, stage s runs
// layers [cuts[s], cuts[s+1]). Only the residual rows cross a cut; every layer's sequence state (KV cache, recurrent
// state) stays with the rank that owns the layer, during prefill AND decode.
// - Prefill: windows in prompt order, a flow shop (stage s+1 runs window w while stage s runs window w+1) over ONE
//   shared two-party seam pair::Pipe PER LINK; on a link the lower stage produces (role 0) and the upper consumes
//   (role 1), so a middle stage holds a consumer pipe upstream and a producer pipe downstream (pipe.hpp unchanged).
// - Decode (one row per token): one hidden row per cut per token downstream, the next token relayed back upstream
//   hop by hop, so every stage knows every generated id.
// The cuts come from the shared planner (strix/pair/chain.hpp via the family binding) and the ranks/links from the
// topology declaration (strix/device/topology.hpp); one stage (no links) is the single-rank path.
// Home: shared (host shared). Bound by: qwen35 dense HIP (apps/hip_qwen35_pair_main.hip). Families with the same
// layer-range engine shape (one residual stream, layer-owned state) bind it through StageOps / DecodeOps.
// Frame header words owned here (engine words of pair::Pipe): [0] magic, [2] start position, [3] rows, [4] payload
// bytes, [5] identity (model + plan + prompt), [6] FNV-1a of the window's token ids, [7] cut layer.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/pair/pipe.hpp"
#include "strix/pair/stage_map.hpp"

namespace strix::pair::layers {
inline constexpr std::uint64_t seam_magic = 0x31454e4950594c53ULL;   // "SLYPINE1"
inline constexpr std::uint64_t hidden_magic = 0x31444e4459414c53ULL; // decode hidden row
inline constexpr std::uint64_t token_magic = 0x314e4b5459414c53ULL;  // decode token relay
inline constexpr std::size_t decode_header_bytes = 4U * sizeof(std::uint64_t);
inline constexpr std::uint64_t state_magic = 0x31544154535941ULL;    // end-of-prefill state relay to the decoder
enum Word : std::size_t { w_magic = 0, w_start = 2, w_rows = 3, w_bytes = 4, w_identity = 5, w_tokens = 6, w_cut = 7 };

inline std::uint64_t fnv(std::uint64_t hash, std::span<const std::uint32_t> values) {
    for (const auto v : values) { hash ^= v; hash *= 1099511628211ULL; }
    return hash;
}
inline std::uint64_t fnv(std::uint64_t hash, std::string_view text) {
    for (const auto c : text) { hash ^= static_cast<unsigned char>(c); hash *= 1099511628211ULL; }
    return hash;
}

// This rank's place in the split: stage index and the stage boundaries (cuts.front() == 0, cuts.back() == layers).
struct Placement {
    std::vector<std::uint32_t> cuts;
    std::uint32_t stage = 0;
    std::uint32_t stages() const { return cuts.size() < 2U ? 0U : static_cast<std::uint32_t>(cuts.size() - 1U); }
    std::uint32_t first() const { return cuts.at(stage); }
    std::uint32_t last() const { return cuts.at(stage + 1U); }
    bool head() const { return stage + 1U == stages(); }   // runs the final layers, the head and the sampler
    bool embeds() const { return stage == 0U; }
    void check() const {
        if (stages() == 0U || stage >= stages() || cuts.front() != 0U)
            throw std::invalid_argument("layer pipeline: cuts must start at 0 and name this rank's stage");
        for (std::size_t i = 1; i < cuts.size(); ++i)
            if (cuts[i] <= cuts[i - 1U]) throw std::invalid_argument("layer pipeline: cuts must increase");
    }
};

// Prompt windows (same windows on every stage and on the single-rank path: the bits depend on them).
struct Window { std::uint32_t start = 0, rows = 0; };
inline std::vector<Window> windows(std::size_t prompt, std::uint32_t width) {
    if (width == 0U) throw std::invalid_argument("layer pipeline: window width must be positive");
    std::vector<Window> out;
    for (std::size_t at = 0; at < prompt; at += width)
        out.push_back({static_cast<std::uint32_t>(at), static_cast<std::uint32_t>(std::min<std::size_t>(width, prompt - at))});
    return out;
}

// Window width of a two-stage pipe from a calibration of the half-window time per width (stage time of one window).
// The width changes no bit when the binding is window-invariant; only the flow shop changes: wall = (windows + 1) x
// half-window time, so a narrower window that costs a little more per row can win by filling the pipe sooner. Costs are
// an input measured on the deployment, never compiled in; no usable cost -> 0 (the caller keeps its width). Ties keep
// the first entry.
struct WindowCost {
    std::uint32_t rows = 0;
    double half_ms = 0;
};
[[nodiscard]] inline std::uint32_t two_stage_window_rows(std::size_t prompt, std::span<const WindowCost> costs) {
    std::uint32_t best = 0;
    double best_ms = 0;
    for (const auto& w : costs) {
        if (w.rows == 0U || !(w.half_ms > 0)) continue;
        const double ms = static_cast<double>((prompt + w.rows - 1U) / w.rows + 1U) * w.half_ms;
        if (best_ms == 0 || ms < best_ms) best = w.rows, best_ms = ms;
    }
    return best;
}

// Consumer-side header check (PipeOptions::validate) of one link: returns the payload bytes.
inline std::function<std::size_t(const PipeHeader&, std::uint64_t)> validator(
    std::uint64_t identity, std::uint64_t count, std::uint32_t cut, std::uint32_t max_rows, std::size_t row_bytes) {
    return [=](const PipeHeader& h, std::uint64_t sequence) {
        if (h[w_magic] != seam_magic || h[1] != sequence || h[41] != count || h[w_cut] != cut || h[w_identity] != identity ||
            h[w_rows] == 0U || h[w_rows] > max_rows || h[w_bytes] != h[w_rows] * row_bytes)
            throw std::runtime_error("layer pipeline seam header mismatch (model, plan, prompt, cut or window)");
        return static_cast<std::size_t>(h[w_bytes]);
    };
}

// What a model binds for prefill. All engine work is stream-ordered; `fence` waits for it.
struct StageOps {
    // Run this rank's layers for one window. tokens: the window's ids (only the first stage embeds them; others find
    // their input rows already loaded by `seam`). head: compute the logits of the window's last row (last stage, last window).
    std::function<void(std::span<const std::uint32_t> tokens, const Window& window, bool head)> forward;
    // save: residual rows of the window -> frame (snapshot ordered after `forward`); load: frame -> residual rows.
    std::function<void(PipeFrame& frame, bool save, std::size_t bytes)> seam;
    std::function<void()> fence;
    std::size_t row_bytes = 0;
    // Interleaved stages (prefill over a StageMap, below): run layers [first, last) for one window; first == 0 embeds
    // `tokens`, a later range finds its rows loaded by `seam`. Not used by the Placement form.
    std::function<void(std::span<const std::uint32_t> tokens, const Window& window, std::uint32_t first, std::uint32_t last,
                       bool head)> range;
};
struct PrefillStats {
    double wait_ms = 0, seal_ms = 0, total_ms = 0;
    std::vector<double> window_ms;   // forward enqueue + its seam work, per window
    // Per-window timeline (host clock). wait: acquire of the input frame. sync: the consumer's load + fence (it drains the
    // previous window's GPU work) or the producer's seal (it drains this window's). enqueue: the forward call.
    // The GPU work of a rank's windows is hidden in sync (+ enqueue, which overlaps it); wait is the rank idling on its input.
    std::vector<double> wait_window_ms, sync_window_ms, enqueue_window_ms;
};

// One prefill on this rank. `in`: consumer pipe from the previous stage (null on the first), `out`: producer pipe to the
// next stage (null on the last). Both pipes must have been built with count == windows().size().
inline PrefillStats prefill(const Placement& p, std::span<const std::uint32_t> prompt, std::uint32_t width, std::uint64_t identity,
                            Pipe* in, Pipe* out, const StageOps& ops) {
    using Clock = std::chrono::steady_clock;
    const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
    p.check();
    if ((in != nullptr) == p.embeds() || (out != nullptr) == p.head())
        throw std::invalid_argument("layer pipeline: pipes do not match the stage (in: not first, out: not last)");
    const auto ws = windows(prompt.size(), width);
    if ((in && in->count() != ws.size()) || (out && out->count() != ws.size()))
        throw std::invalid_argument("layer pipeline: pipe count differs from the window count");
    PrefillStats stats;
    const auto begin = Clock::now();
    for (std::uint64_t w = 0; w < ws.size(); ++w) {
        const auto& win = ws[w];
        const auto tokens = prompt.subspan(win.start, win.rows);
        const auto bytes = std::size_t(win.rows) * ops.row_bytes;
        const auto t0 = Clock::now();
        double wait = 0, sync = 0;
        if (in) {
            auto& frame = in->acquire(w);
            wait = ms(Clock::now() - t0);
            stats.wait_ms += wait;
            const auto l0 = Clock::now();
            const auto& h = frame.header;
            if (h[w_start] != win.start || h[w_rows] != win.rows || h[w_tokens] != fnv(14695981039346656037ULL, tokens) ||
                h[w_cut] != p.first())
                throw std::runtime_error("layer pipeline seam window does not match this stage's window (position, rows, tokens)");
            ops.seam(frame, false, bytes);
            ops.fence();            // the slot is reused only after its rows are in the engine
            in->release(w);
            sync += ms(Clock::now() - l0);
        }
        const auto e0 = Clock::now();
        ops.forward(tokens, win, p.head() && w + 1U == ws.size());
        stats.enqueue_window_ms.push_back(ms(Clock::now() - e0));
        if (out) {
            const auto s0 = Clock::now();
            auto& frame = out->acquire(w);
            frame.header = {};
            auto& h = frame.header;
            h[w_magic] = seam_magic; h[1] = w; h[41] = out->count();
            h[w_start] = win.start; h[w_rows] = win.rows; h[w_bytes] = bytes; h[w_identity] = identity;
            h[w_tokens] = fnv(14695981039346656037ULL, tokens); h[w_cut] = p.last();
            frame.resize(bytes);
            ops.seam(frame, true, bytes);
            if (frame.staging) frame.staging->seal(bytes);
            out->release(w);
            const auto seal = ms(Clock::now() - s0);
            stats.seal_ms += seal;
            sync += seal;
        }
        stats.wait_window_ms.push_back(wait);
        stats.sync_window_ms.push_back(sync);
        stats.window_ms.push_back(ms(Clock::now() - t0));
    }
    if (in) in->finish(ws.size());
    if (out) out->finish(ws.size());
    stats.total_ms = ms(Clock::now() - begin);
    return stats;
}

// Ordered byte link to a neighbour stage for the decode messages (the same transport the pipe ran on, after finish()).
struct Link {
    std::function<void(std::span<const std::byte>)> send;
    std::function<void(std::span<std::byte>)> receive;
};
struct DecodeOps {
    // One row at the next position: the first stage embeds `token`; the others find the row loaded by import_row.
    // head: the last stage computes the logits; top() returns their argmax (synchronizes).
    std::function<void(std::uint32_t token, bool head)> forward;
    std::function<std::uint32_t()> top;
    std::function<void(std::span<std::byte>)> export_row;        // synchronizes
    std::function<void(std::span<const std::byte>)> import_row;  // host bytes stay valid until the next forward returns
    std::size_t row_bytes = 0;
    // Interleaved stages (decode over a StageMap, below): one row through layers [first, last); first == 0 embeds `token`.
    std::function<void(std::uint32_t token, std::uint32_t first, std::uint32_t last, bool head)> range;
    // Pipelined speculation (optional, first stage of two or more; both set = on). The first stage is idle while the
    // later stages finish row n, so it runs row n+1 on a GUESSED token and sends it only when the relayed token equals
    // the guess; on a miss it undoes the guessed row and runs the true one. The wire carries the same messages as
    // without guessing and every row is the one-row forward of the true token at its position: bitwise the plain loop.
    // guess(last, next): `last` = the confirmed token this stage just ran (each confirmed id is passed once, in order);
    //   return true with a proposal in `next`, false = no guess this step.
    // rewind(): undo the last forward completely (position and host state back; the redo at the same position must
    //   rewrite every device row the guessed forward wrote). A family whose state cannot be undone leaves it empty.
    std::function<bool(std::uint32_t last, std::uint32_t& next)> guess;
    std::function<void()> rewind;
};
struct DecodeStats {
    std::vector<std::uint32_t> ids;
    double total_ms = 0;
    std::uint64_t guesses = 0, hits = 0;  // first stage with guess/rewind: guessed rows run / rows sent without a redo
};

// Greedy decode of `count` ids, the first one being `first_token` (known on the last stage only: the head of the
// prefill). `up` / `down`: links to the previous / next stage (null at the ends). Returns the same ids on every stage.
inline DecodeStats decode(const Placement& p, std::uint32_t first_token, std::uint32_t count, const Link* up, const Link* down,
                          const DecodeOps& ops) {
    p.check();
    if ((up != nullptr) == p.embeds() || (down != nullptr) == p.head())
        throw std::invalid_argument("layer pipeline: links do not match the stage (up: not first, down: not last)");
    std::vector<std::byte> message(decode_header_bytes + ops.row_bytes);
    const auto relay = [&](std::uint32_t token, std::uint64_t step) {   // last -> ... -> first
        std::uint64_t words[4] = {token_magic, step, token, 0};
        if (down) {
            down->receive(std::as_writable_bytes(std::span(words)));
            if (words[0] != token_magic || words[1] != step || words[3] != 0U) throw std::runtime_error("layer pipeline token relay out of order");
            token = static_cast<std::uint32_t>(words[2]);
        }
        if (up) { words[0] = token_magic; words[1] = step; words[2] = token; words[3] = 0; up->send(std::as_bytes(std::span(words))); }
        return token;
    };
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    DecodeStats out;
    if (count == 0U) return out;
    std::uint32_t token = relay(p.head() ? first_token : 0U, 0U);
    out.ids.push_back(token);
    if (!up && down && ops.guess && ops.rewind) {  // first stage, pipelined speculation (see DecodeOps::guess)
        std::vector<std::byte> spare(message.size());
        bool ready = false;  // `message` already holds the row of this step (a guess that hit)
        for (std::uint64_t step = 1; step < count; ++step) {
            if (!ready) {
                ops.forward(token, false);
                ops.export_row(std::span(message).subspan(decode_header_bytes));
            }
            SeamCrc crc; crc.update(std::span<const std::byte>(message).subspan(decode_header_bytes));
            const std::uint64_t h[4] = {hidden_magic, step, ops.row_bytes, crc.value()};
            std::memcpy(message.data(), h, sizeof h);
            down->send(message);
            std::uint32_t g = 0;
            const bool guessed = step + 1U < count && ops.guess(token, g);
            if (guessed) {
                ops.forward(g, false);
                ops.export_row(std::span(spare).subspan(decode_header_bytes));
                ++out.guesses;
            }
            token = relay(0U, step);
            out.ids.push_back(token);
            ready = guessed && g == token;
            if (ready) { message.swap(spare); ++out.hits; }
            else if (guessed) ops.rewind();
        }
        out.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        return out;
    }
    for (std::uint64_t step = 1; step < count; ++step) {
        if (up) {
            up->receive(message);
            std::uint64_t h[4];
            std::memcpy(h, message.data(), sizeof h);
            SeamCrc crc; crc.update(std::span(message).subspan(decode_header_bytes));
            if (h[0] != hidden_magic || h[1] != step || h[2] != ops.row_bytes || h[3] != crc.value())
                throw std::runtime_error("layer pipeline decode row mismatch (order, size or CRC)");
            ops.import_row(std::span<const std::byte>(message).subspan(decode_header_bytes));
        }
        ops.forward(token, p.head());
        if (down) {
            ops.export_row(std::span(message).subspan(decode_header_bytes));
            SeamCrc crc; crc.update(std::span<const std::byte>(message).subspan(decode_header_bytes));
            const std::uint64_t h[4] = {hidden_magic, step, ops.row_bytes, crc.value()};
            std::memcpy(message.data(), h, sizeof h);
            down->send(message);
        }
        token = relay(p.head() ? ops.top() : 0U, step);
        out.ids.push_back(token);
    }
    out.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return out;
}

// A bulk payload crosses a link in pieces of at most `chunk` bytes (the byte stream is the same; only the transport
// calls are bounded). The chunk keeps every call of the layer-state and tail-row messages within what the pipe frames
// already exercise (8 MiB); it is not a proven fix: a checksum mismatch also occurred with 8 MiB calls.
inline constexpr std::size_t link_chunk_bytes = std::size_t{8} << 20;
inline void send_chunked(const Link& link, std::span<const std::byte> bytes, std::size_t chunk = link_chunk_bytes) {
    if (chunk == 0) throw std::invalid_argument("link chunk must be positive");
    for (std::size_t at = 0; at < bytes.size(); at += chunk) link.send(bytes.subspan(at, std::min(chunk, bytes.size() - at)));
}
inline void receive_chunked(const Link& link, std::span<std::byte> bytes, std::size_t chunk = link_chunk_bytes) {
    if (chunk == 0) throw std::invalid_argument("link chunk must be positive");
    for (std::size_t at = 0; at < bytes.size(); at += chunk) link.receive(bytes.subspan(at, std::min(chunk, bytes.size() - at)));
}

// One bulk message with end-to-end integrity: {magic, id, bytes, checksum, extra} + payload. The receiver answers every
// attempt with one word (1 accepted, 0 resend); a payload that fails its checksum is requested again, at most
// `message_attempts` times, and the two copies are compared on stderr (differing bytes, first offset, 4 KiB pages) so a
// transport that corrupts intermittently is measured instead of guessed. Measured on the two-cable host transport (8K
// prompt, KV of one attention layer = 32 MiB): a single run of the hand-off failed its checksum without any fault in the
// sender, in sizes the 10 MiB pipe frames never showed.
inline constexpr unsigned message_attempts = 3;
inline unsigned send_verified(const Link& to, const std::uint64_t (&words)[5], std::span<const std::byte> bytes, std::size_t chunk) {
    for (unsigned attempt = 0; attempt < message_attempts; ++attempt) {
        to.send(std::as_bytes(std::span(words)));
        send_chunked(to, bytes, chunk);
        std::uint64_t ack = 0;
        to.receive(std::as_writable_bytes(std::span(&ack, 1)));
        if (ack == 1U) return attempt;
        if (ack != 0U) throw std::runtime_error("bulk message: bad acknowledgement word (protocol error)");
    }
    throw std::runtime_error("bulk message refused " + std::to_string(message_attempts) + " times by the receiver (checksum)");
}
// `admit` checks the header words (id, length, extra) against what the receiver expects and throws a named protocol error
// otherwise; it runs on every attempt BEFORE the payload is allocated or read, so a damaged length never sizes a buffer.
inline unsigned receive_verified(const Link& from, std::uint64_t (&words)[5], std::vector<std::byte>& bytes, std::uint64_t magic,
                                 const std::function<std::uint32_t(std::span<const std::byte>)>& check, std::size_t chunk,
                                 const std::string& what, const std::function<void(const std::uint64_t (&)[5])>& admit) {
    std::vector<std::byte> bad;
    for (unsigned attempt = 0; attempt < message_attempts; ++attempt) {
        from.receive(std::as_writable_bytes(std::span(words)));
        if (words[0] != magic) throw std::runtime_error(what + ": bad message header");
        admit(words);
        bytes.resize(static_cast<std::size_t>(words[2]));
        receive_chunked(from, bytes, chunk);
        const bool ok = check(bytes) == words[3];
        const std::uint64_t ack = ok ? 1U : 0U;
        from.send(std::as_bytes(std::span(&ack, 1)));
        if (ok) {
            if (!bad.empty() && bad.size() == bytes.size()) {
                std::size_t differing = 0, first = bytes.size(), pages = 0;
                for (std::size_t i = 0; i < bytes.size(); ++i)
                    if (bad[i] != bytes[i]) { ++differing; if (first == bytes.size()) first = i; }
                for (std::size_t i = 0; i < bytes.size(); i += 4096U)
                    if (std::memcmp(bad.data() + i, bytes.data() + i, std::min<std::size_t>(4096U, bytes.size() - i)) != 0) ++pages;
                std::fprintf(stderr, "%s: checksum mismatch repaired by resend (message %llu, %zu bytes): %zu differing bytes in %zu pages, first at %zu\n",
                             what.c_str(), static_cast<unsigned long long>(words[1]), bytes.size(), differing, pages, first);
            }
            return attempt;
        }
        if (bad.empty()) bad = bytes;
    }
    throw std::runtime_error(what + ": checksum mismatch (message " + std::to_string(words[1]) + ", " + std::to_string(bytes.size()) +
                             " bytes) after " + std::to_string(message_attempts) + " attempts");
}

// End-of-prefill relay of every stage's layer state to ONE decoder stage (the planner's handoff plan: the decoder owns
// the whole model, decode then runs there alone at single-rank cost instead of crossing every cut per token).
// Messages travel along the line towards `decoder`: a stage below it forwards what arrives from below plus its own
// layers downwards (to stage + 1), a stage above it forwards upwards (to stage - 1). One message per layer, so the
// sender's export of layer l + 1 overlaps the receiver's check and import of layer l. The head stage attaches the
// prefill's first token. The decoder calls `load` once per layer it does not run and returns the first token.
struct RelayOps {
    std::function<std::vector<std::byte>(std::uint32_t layer)> save;          // one layer's state on this stage
    std::function<void(std::uint32_t layer, std::span<const std::byte>)> load; // decoder only
    // Checksum of one message's bytes; empty = SeamCrc (bytewise table). A binder with a hardware CRC32C passes it:
    // the relay moves the whole KV + recurrent state once, a few hundred MB at long contexts.
    std::function<std::uint32_t(std::span<const std::byte>)> checksum;
    std::size_t chunk_bytes = link_chunk_bytes;   // largest single transport call of a message's payload
    // Receiver-side length check of each incoming message, before its payload is allocated: the exact state bytes of one
    // layer on this engine (empty: only the bound below applies). A header outside it is a named protocol error.
    std::function<std::uint64_t(std::uint32_t layer)> layer_bytes;
    std::uint64_t max_message_bytes = std::uint64_t{1} << 32;
};
struct RelayStats { std::uint32_t token = 0; std::uint64_t bytes = 0, messages = 0, resends = 0; double ms = 0; };
inline RelayStats relay_state(const Placement& p, std::uint32_t decoder, std::uint32_t first_token, const Link* up,
                              const Link* down, const RelayOps& ops) {
    using Clock = std::chrono::steady_clock;
    p.check();
    if (decoder >= p.stages()) throw std::invalid_argument("state relay: decoder stage outside the plan");
    if ((up != nullptr) == p.embeds() || (down != nullptr) == p.head())
        throw std::invalid_argument("layer pipeline: links do not match the stage (up: not first, down: not last)");
    const auto start = Clock::now();
    const auto check_sum = [&](std::span<const std::byte> b) {
        if (ops.checksum) return ops.checksum(b);
        SeamCrc crc; crc.update(b); return crc.value();
    };
    RelayStats out;
    // One message per layer, streamed: {state_magic, layer, bytes, checksum, token + 1 (0: none)} + bytes. A link
    // carries a count first; a forwarding stage passes each message on as it arrives, then appends its own layers.
    struct Message { std::uint64_t words[5]{}; std::vector<std::byte> bytes; };
    const std::uint64_t layers_total = p.cuts.back();
    const auto admit = [&](const std::uint64_t (&w)[5]) {
        if (w[1] >= layers_total)
            throw std::runtime_error("state relay: message names layer " + std::to_string(w[1]) + " of a " +
                                     std::to_string(layers_total) + "-layer plan (protocol error)");
        const std::uint64_t expected = ops.layer_bytes ? ops.layer_bytes(static_cast<std::uint32_t>(w[1])) : 0U;
        if ((ops.layer_bytes && w[2] != expected) || w[2] > ops.max_message_bytes)
            throw std::runtime_error("state relay: message length " + std::to_string(w[2]) + " bytes for layer " +
                                     std::to_string(w[1]) + (ops.layer_bytes ? ", expected " + std::to_string(expected)
                                                                             : ", bound " + std::to_string(ops.max_message_bytes)) +
                                     " (protocol error)");
    };
    const auto get_msg = [&](const Link* from) {
        Message m;
        out.resends += receive_verified(*from, m.words, m.bytes, state_magic, check_sum, ops.chunk_bytes, "state relay", admit);
        return m;
    };
    const auto put_msg = [&](const Link* to, const Message& m) { out.resends += send_verified(*to, m.words, m.bytes, ops.chunk_bytes); };
    const auto own = [&](std::uint32_t layer) {
        Message m;
        m.bytes = ops.save(layer);
        m.words[0] = state_magic; m.words[1] = layer; m.words[2] = m.bytes.size(); m.words[3] = check_sum(m.bytes);
        m.words[4] = p.head() ? std::uint64_t(first_token) + 1U : 0U;
        return m;
    };
    const auto count_from = [&](const Link* from) {
        std::uint64_t count = 0;
        if (from) from->receive(std::as_writable_bytes(std::span(&count, 1)));
        return count;
    };
    const std::uint64_t mine = p.last() - p.first();
    if (p.stage != decoder) {
        const bool below = p.stage < decoder;
        const Link* from = below ? up : down;   // the far side of the line (null at the end)
        const Link* to = below ? down : up;
        const auto incoming = count_from(from);
        const std::uint64_t count = incoming + mine;
        to->send(std::as_bytes(std::span(&count, 1)));
        for (std::uint64_t i = 0; i < incoming; ++i) { const auto m = get_msg(from); out.bytes += m.bytes.size(); put_msg(to, m); }
        for (auto layer = p.first(); layer < p.last(); ++layer) { const auto m = own(layer); out.bytes += m.bytes.size(); put_msg(to, m); }
        out.messages = count;
    } else {
        bool token = p.head();
        out.token = first_token;
        std::uint64_t received = 0;
        for (const Link* from : {p.stage > 0U ? up : nullptr, down}) {
            const auto count = count_from(from);
            for (std::uint64_t i = 0; i < count; ++i) {
                const auto m = get_msg(from);
                if (m.words[1] >= p.first() && m.words[1] < p.last()) throw std::runtime_error("state relay: a layer of the decoder's own stage");
                ops.load(static_cast<std::uint32_t>(m.words[1]), m.bytes);
                out.bytes += m.bytes.size();
                if (m.words[4]) { out.token = static_cast<std::uint32_t>(m.words[4] - 1U); token = true; }
                ++received;
            }
        }
        out.messages = received;
        if (received + mine != p.cuts.back() || !token) throw std::runtime_error("state relay: decoder did not receive every layer");
    }
    out.ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return out;
}

// When the layer-split prefill beats every rank prefilling the whole prompt alone (the replicated prefill a tensor-parallel
// decode otherwise starts from). Equal ranks, one half-window time t / 2 per stage: the flow shop wall is (W + 1) t / 2
// against W t, a saving of (W - 1) t / 2 for W windows; the hand-off moves the state of the layers each rank lacks once
// (about 0.12 s per 8K positions of the 27B model on the USB4 link). From three windows the saving is at least one full
// window time, longer than that hand-off for every window width the engine runs (>= 64 rows). Below, the replicated
// prefill stays.
[[nodiscard]] constexpr bool pipeline_prefill_wins(std::size_t windows) noexcept { return windows >= 3U; }

// Line barrier of a stage chain over its links: returns on a stage only after every stage has entered it. The head stage
// speaks first (up), each stage forwards to the next one down the line only after hearing from below; so no rank sends or
// reads anything but these words until all stages are here. Run it after the prefill pipes are gone (a stage that finished
// first must not start the state hand-off while another stage's pipe progress thread still reads the shared transport;
// the barrier only orders the hand-off after the pipes: intermittent corruption was seen with it in place) and before the first timed phase.
inline void line_barrier(const Link* up, const Link* down) {
    std::uint64_t word = 0x52524142ULL;
    const auto out = [](const Link* l, std::uint64_t& w) { l->send(std::as_bytes(std::span(&w, 1))); };
    const auto in = [](const Link* l, std::uint64_t& w) { l->receive(std::as_writable_bytes(std::span(&w, 1))); };
    if (down) in(down, word);            // everything below has arrived
    if (up) { out(up, word); in(up, word); }   // report up, wait for the release
    if (down) out(down, word);           // release below
}

// Wire version of the layer-pipeline line: 2 = every bulk message of the state relay and the tail rows is answered with
// one acknowledgement word (version 1 had none, and its builds send no hello). A stage chain checks it once, before
// anything else crosses the line, in the line barrier's order: the request {hello_magic, version} travels up from the
// head stage, the answer {hello_answer_magic, version} travels back down, and each word is checked before the next is
// read. A peer of another build is refused by name instead of waiting for words it never sends: a version-1 peer's first
// word is its line barrier word, or the echo of our request (the barrier sends back the word it received), never an
// answer.
inline constexpr std::uint64_t hello_magic = 0x314f4c4c45485953ULL;         // "SYHELLO1" (request, upwards)
inline constexpr std::uint64_t hello_answer_magic = 0x414f4c4c45485953ULL;  // "SYHELLOA" (answer, downwards)
inline constexpr std::uint64_t wire_version = 2;
inline void line_hello(const Link* up, const Link* down, std::uint64_t version = wire_version) {
    const auto out = [&](const Link* l, std::uint64_t magic) {
        const std::uint64_t words[2] = {magic, version};
        l->send(std::as_bytes(std::span(words)));
    };
    const auto in = [&](const Link* l, std::uint64_t magic, const char* side) {
        std::uint64_t word = 0;
        l->receive(std::as_writable_bytes(std::span(&word, 1)));
        if (word != magic)
            throw std::runtime_error(std::string("layer pipeline: the ") + side +
                                     " peer sent no wire hello (a build before layer wire version 2); this build speaks version " +
                                     std::to_string(version) + ": run every rank from the same build");
        l->receive(std::as_writable_bytes(std::span(&word, 1)));
        return word;
    };
    const auto refuse_if = [&](std::uint64_t peer, const char* side) {
        if (peer != version)
            throw std::runtime_error("layer pipeline: wire version mismatch: this build speaks version " + std::to_string(version) +
                                     ", the " + side + " peer version " + std::to_string(peer) + ": run every rank from the same build");
    };
    std::uint64_t below = version;
    if (down) below = in(down, hello_magic, "downstream");
    if (up) { out(up, hello_magic); refuse_if(in(up, hello_answer_magic, "upstream"), "upstream"); }
    if (down) { out(down, hello_answer_magic); refuse_if(below, "downstream"); }   // answer first: the peer names it too
}

// End-of-prefill hand-off of a layer-split prefill to a decode that runs on EVERY rank (tensor-parallel decode: each rank
// holds the whole model and the whole sequence state). Every stage is the decoder once, in stage order, over the same
// per-layer messages as relay_state(): afterwards every rank holds the state of every layer, bit for bit what a whole
// prefill on that rank leaves (the layer ranges run the same launches as the whole range, so the state bytes are equal).
// The caller runs a line barrier after the prefill pipes are gone (a stage that finished first must not send while the
// other stage's pipe progress thread still reads the shared transport; the barrier orders the phases, it does not explain the
// intermittent corruption seen on the host transport with it in place, whose cause is not established).
// The first token (known on the head stage) reaches the other stages with the relay to them. One round per stage, so the
// two directions of a link do not overlap: the wall is the sum of the rounds.
inline RelayStats share_state(const Placement& p, std::uint32_t first_token, const Link* up, const Link* down,
                              const RelayOps& ops) {
    RelayStats total;
    total.token = first_token;
    for (std::uint32_t decoder = 0; decoder < p.stages(); ++decoder) {
        const auto round = relay_state(p, decoder, first_token, up, down, ops);
        total.bytes += round.bytes;
        total.messages += round.messages;
        total.resends += round.resends;
        total.ms += round.ms;
        if (p.stage == decoder) total.token = round.token;
    }
    return total;
}

// Per-window rows from the head stage back to the first stage after a two-stage prefill (the rows only the head computes:
// the final trunk hidden that a speculation drafter reads on every rank). One message per window, in window order:
// {tail_magic, window, rows, bytes, checksum} + bytes. `save` fills the head's rows of one window (its own record of them),
// `load` hands them to the first stage. Two stages only; returns the bytes moved on this rank.
inline constexpr std::uint64_t tail_magic = 0x314c494154594c53ULL;  // "SLYTAIL1"
struct TailOps {
    std::size_t row_bytes = 0;
    std::function<void(std::uint32_t window, std::span<std::byte> out)> save;                   // head stage
    std::function<void(std::uint32_t window, const Window&, std::span<const std::byte>)> load;  // first stage
    std::function<std::uint32_t(std::span<const std::byte>)> checksum;                          // empty: SeamCrc
    std::size_t chunk_bytes = link_chunk_bytes;
};
inline std::uint64_t return_rows(const Placement& p, std::size_t prompt_tokens, std::uint32_t width, const Link* up,
                                 const Link* down, const TailOps& ops) {
    p.check();
    if (p.stages() != 2U) throw std::invalid_argument("tail rows: two stages only");
    if (ops.row_bytes == 0) throw std::invalid_argument("tail rows: row_bytes unset");
    const auto check_sum = [&](std::span<const std::byte> b) {
        if (ops.checksum) return ops.checksum(b);
        SeamCrc crc; crc.update(b); return crc.value();
    };
    const auto ws = windows(prompt_tokens, width);
    std::uint64_t moved = 0;
    std::vector<std::byte> bytes;
    for (std::uint32_t w = 0; w < ws.size(); ++w) {
        std::uint64_t words[5]{};
        if (p.head()) {
            if (!up) throw std::invalid_argument("tail rows: the head stage needs its upstream link");
            bytes.resize(std::size_t(ws[w].rows) * ops.row_bytes);
            ops.save(w, bytes);
            words[0] = tail_magic; words[1] = w; words[2] = bytes.size(); words[3] = check_sum(bytes); words[4] = ws[w].rows;
            (void)send_verified(*up, words, bytes, ops.chunk_bytes);
        } else {
            if (!down) throw std::invalid_argument("tail rows: the first stage needs its downstream link");
            const std::uint64_t expected = std::uint64_t(ws[w].rows) * ops.row_bytes;
            (void)receive_verified(*down, words, bytes, tail_magic, check_sum, ops.chunk_bytes, "tail rows",
                                   [&](const std::uint64_t (&h)[5]) {
                                       if (h[1] != w || h[4] != ws[w].rows || h[2] != expected)
                                           throw std::runtime_error("tail rows: message does not match the window (window " +
                                                                    std::to_string(h[1]) + ", length " + std::to_string(h[2]) +
                                                                    " bytes, expected " + std::to_string(expected) + ")");
                                   });
            ops.load(w, ws[w], bytes);
        }
        moved += bytes.size();
    }
    return moved;
}

// Interleaved virtual stages on two ranks (strix/pair/stage_map.hpp): the 2v stages of a StageMap alternate between the
// ranks (stage s on rank s % 2) and every layer's sequence state stays with the rank that owns its stage, in prefill AND
// decode. Each rank runs its (stage, chunk) steps in the shared StagePipes order (groups of chunks, depth-first); every
// stage still takes the windows in prompt order, so each layer sees the same windows in the same order as the
// single-rank path: the bits equal the one-node prefill at the same width. Fill + drain cost one stage (P / 2v) instead
// of one half-pass (P / 2): wall ~ chunks x P / 2 + P / 2v + hops (regime.hpp interleaved_wall_ms).
// Whole-layer cuts only: a half cut splits a layer inside the engine and is not a layer range.
inline void check_stage_map(const StageMap& map, unsigned rank, bool range_bound) {
    map.validate(map.layers());
    if (rank > 1U) throw std::invalid_argument("layer pipeline: a stage map runs on ranks 0 and 1");
    if (!range_bound) throw std::invalid_argument("layer pipeline: a stage map needs the engine's range hook");
    for (const auto& cut : map.cuts)
        if (cut.half) throw std::invalid_argument("layer pipeline: stage maps run whole-layer cuts (a half cut is not a layer range)");
}

// Consumer-side header check of a StageMap seam (StagePipes checks [42]-[44] first): the cut word names the boundary's cut.
inline std::function<std::size_t(const PipeHeader&, std::uint64_t)> stage_validator(
    std::uint64_t identity, const StageMap& map, std::uint32_t max_rows, std::size_t row_bytes) {
    return [=](const PipeHeader& h, std::uint64_t sequence) {
        const std::uint64_t boundary = map.stages() > 2U ? h[43] : 0U;   // wire v3 (v = 1) has the one boundary
        if (h[w_magic] != seam_magic || h[1] != sequence || boundary + 1U >= map.stages() ||
            h[w_cut] != map.cuts[static_cast<std::size_t>(boundary) + 1U].wire() || h[w_identity] != identity ||
            h[w_rows] == 0U || h[w_rows] > max_rows || h[w_bytes] != h[w_rows] * row_bytes)
            throw std::runtime_error("layer pipeline stage seam header mismatch (model, plan, prompt, cut or window)");
        return static_cast<std::size_t>(h[w_bytes]);
    };
}

// One prefill on this rank over `pipes` (built for the same map with chunks == windows(prompt, width).size() and
// options.validate = stage_validator). Engine hooks: ops.range (layers of one stage), ops.seam, ops.fence.
inline PrefillStats prefill(const StageMap& map, unsigned rank, std::span<const std::uint32_t> prompt, std::uint32_t width,
                            std::uint64_t identity, StagePipes& pipes, const StageOps& ops) {
    using Clock = std::chrono::steady_clock;
    const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
    check_stage_map(map, rank, static_cast<bool>(ops.range));
    const auto ws = windows(prompt.size(), width);
    if (pipes.chunks() != ws.size() || pipes.map().fnv() != map.fnv())
        throw std::invalid_argument("layer pipeline: stage pipes were built for another map or window count");
    const std::uint32_t last_stage = map.stages() - 1U;
    PrefillStats stats;
    stats.window_ms.assign(ws.size(), 0.0);
    const auto begin = Clock::now();
    for (const auto step : pipes.order()) {
        const auto& win = ws[static_cast<std::size_t>(step.chunk)];
        const auto tokens = prompt.subspan(win.start, win.rows);
        const auto bytes = std::size_t(win.rows) * ops.row_bytes;
        const auto t0 = Clock::now();
        if (step.stage > 0U) {
            const std::uint32_t boundary = step.stage - 1U;
            auto& frame = pipes.acquire_in(step.chunk, boundary);
            stats.wait_ms += ms(Clock::now() - t0);
            const auto& h = frame.header;
            if (h[w_start] != win.start || h[w_rows] != win.rows || h[w_tokens] != fnv(14695981039346656037ULL, tokens))
                throw std::runtime_error("layer pipeline stage seam window does not match this stage's window (position, rows, tokens)");
            ops.seam(frame, false, bytes);
            ops.fence();            // the slot is reused only after its rows are in the engine
            pipes.release_in(step.chunk, boundary);
        }
        ops.range(tokens, win, map.cuts[step.stage].layer, map.cuts[step.stage + 1U].layer,
                  step.stage == last_stage && step.chunk + 1U == ws.size());
        if (step.stage < last_stage) {
            const auto s0 = Clock::now();
            auto& frame = pipes.acquire_out(step.chunk, step.stage);   // StagePipes owns [1], [41], [43], [44]
            auto& h = frame.header;
            h[w_magic] = seam_magic; h[w_start] = win.start; h[w_rows] = win.rows; h[w_bytes] = bytes; h[w_identity] = identity;
            h[w_tokens] = fnv(14695981039346656037ULL, tokens); h[w_cut] = map.cuts[step.stage + 1U].wire();
            frame.resize(bytes);
            ops.seam(frame, true, bytes);
            if (frame.staging) frame.staging->seal(bytes);
            pipes.release_out(step.chunk, step.stage);
            stats.seal_ms += ms(Clock::now() - s0);
        }
        stats.window_ms[static_cast<std::size_t>(step.chunk)] += ms(Clock::now() - t0);
    }
    pipes.finish();
    stats.total_ms = ms(Clock::now() - begin);
    return stats;
}

// Greedy decode over the same map: per token the row visits the 2v stages in order (one hop per boundary: 4 header words
// + one row, CRC), then the head rank (the last stage is odd: rank 1) relays the token to rank 0. `peer`: ordered duplex
// link to the other rank (after StagePipes::finish()). 2v hops per token instead of 2: the planner charges decode tokens.
inline DecodeStats decode(const StageMap& map, unsigned rank, std::uint32_t first_token, std::uint32_t count, const Link& peer,
                          const DecodeOps& ops) {
    check_stage_map(map, rank, static_cast<bool>(ops.range));
    const std::uint32_t stages = map.stages();
    const bool head_rank = map.owner(stages - 1U) == rank;
    std::vector<std::byte> message(decode_header_bytes + ops.row_bytes);
    const auto relay = [&](std::uint32_t token, std::uint64_t step) {
        std::uint64_t words[4] = {token_magic, step, token, 0};
        if (head_rank) {
            peer.send(std::as_bytes(std::span(words)));
            return token;
        }
        peer.receive(std::as_writable_bytes(std::span(words)));
        if (words[0] != token_magic || words[1] != step || words[3] != 0U) throw std::runtime_error("layer pipeline token relay out of order");
        return static_cast<std::uint32_t>(words[2]);
    };
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    DecodeStats out;
    if (count == 0U) return out;
    std::uint32_t token = relay(head_rank ? first_token : 0U, 0U);
    out.ids.push_back(token);
    for (std::uint64_t step = 1; step < count; ++step) {
        for (std::uint32_t s = 0; s < stages; ++s) {
            if (map.owner(s) != rank) continue;
            if (s > 0U) {
                peer.receive(message);
                std::uint64_t h[4];
                std::memcpy(h, message.data(), sizeof h);
                SeamCrc crc; crc.update(std::span(message).subspan(decode_header_bytes));
                if (h[0] != hidden_magic || h[1] != step * stages + s - 1U || h[2] != ops.row_bytes || h[3] != crc.value())
                    throw std::runtime_error("layer pipeline decode row mismatch (order, size or CRC)");
                ops.import_row(std::span<const std::byte>(message).subspan(decode_header_bytes));
            }
            ops.range(token, map.cuts[s].layer, map.cuts[s + 1U].layer, s + 1U == stages);
            if (s + 1U < stages) {
                ops.export_row(std::span(message).subspan(decode_header_bytes));
                SeamCrc crc; crc.update(std::span<const std::byte>(message).subspan(decode_header_bytes));
                const std::uint64_t h[4] = {hidden_magic, step * stages + s, ops.row_bytes, crc.value()};
                std::memcpy(message.data(), h, sizeof h);
                peer.send(message);
            }
        }
        token = relay(head_rank ? ops.top() : 0U, step);
        out.ids.push_back(token);
    }
    out.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return out;
}
} // namespace strix::pair::layers
