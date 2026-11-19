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
#include <cstring>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/pair/pipe.hpp"

namespace strix::pair::layers {
inline constexpr std::uint64_t seam_magic = 0x31454e4950594c53ULL;   // "SLYPINE1"
inline constexpr std::uint64_t hidden_magic = 0x31444e4459414c53ULL; // decode hidden row
inline constexpr std::uint64_t token_magic = 0x314e4b5459414c53ULL;  // decode token relay
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
};
struct PrefillStats {
    double wait_ms = 0, seal_ms = 0, total_ms = 0;
    std::vector<double> window_ms;   // forward enqueue + its seam work, per window
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
        if (in) {
            auto& frame = in->acquire(w);
            stats.wait_ms += ms(Clock::now() - t0);
            const auto& h = frame.header;
            if (h[w_start] != win.start || h[w_rows] != win.rows || h[w_tokens] != fnv(14695981039346656037ULL, tokens) ||
                h[w_cut] != p.first())
                throw std::runtime_error("layer pipeline seam window does not match this stage's window (position, rows, tokens)");
            ops.seam(frame, false, bytes);
            ops.fence();            // the slot is reused only after its rows are in the engine
            in->release(w);
        }
        ops.forward(tokens, win, p.head() && w + 1U == ws.size());
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
            stats.seal_ms += ms(Clock::now() - s0);
        }
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
};
struct DecodeStats { std::vector<std::uint32_t> ids; double total_ms = 0; };

// Greedy decode of `count` ids, the first one being `first_token` (known on the last stage only: the head of the
// prefill). `up` / `down`: links to the previous / next stage (null at the ends). Returns the same ids on every stage.
inline DecodeStats decode(const Placement& p, std::uint32_t first_token, std::uint32_t count, const Link* up, const Link* down,
                          const DecodeOps& ops) {
    p.check();
    if ((up != nullptr) == p.embeds() || (down != nullptr) == p.head())
        throw std::invalid_argument("layer pipeline: links do not match the stage (up: not first, down: not last)");
    std::vector<std::byte> message(32U + ops.row_bytes);   // 4 header words + one row
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
    for (std::uint64_t step = 1; step < count; ++step) {
        if (up) {
            up->receive(message);
            std::uint64_t h[4];
            std::memcpy(h, message.data(), sizeof h);
            SeamCrc crc; crc.update(std::span(message).subspan(32U));
            if (h[0] != hidden_magic || h[1] != step || h[2] != ops.row_bytes || h[3] != crc.value())
                throw std::runtime_error("layer pipeline decode row mismatch (order, size or CRC)");
            ops.import_row(std::span<const std::byte>(message).subspan(32U));
        }
        ops.forward(token, p.head());
        if (down) {
            ops.export_row(std::span(message).subspan(32U));
            SeamCrc crc; crc.update(std::span<const std::byte>(message).subspan(32U));
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
} // namespace strix::pair::layers
