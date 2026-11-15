#pragma once
// Host-only layer-cut policy and wire descriptors. Backends own device copies, not resolution/framing.
#include "strix/env_flags.hpp"
#include "strix/pair/chain.hpp"
#include "strix/handoff.hpp"
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace strix::pair::layercut {
namespace chain = strix::pair::chain;
// Covers startup, including the first connection and its HELLO. The cancellation
// callback must be noexcept and wake any pending accept/read; it runs at most once.
class StartupScope {
    std::function<void()> cancel_;
    std::mutex mutex_;
    std::condition_variable cv_;
    enum class State { pending, ready, failed };
    State state_ = State::pending;
    std::thread timer_;
    void fail_locked() {
        if (state_ != State::pending) return;
        state_ = State::failed;
        cancel_();
        cv_.notify_all();
    }
public:
    explicit StartupScope(std::function<void()> cancel,
                          std::chrono::milliseconds timeout = std::chrono::seconds(120))
        : cancel_(std::move(cancel)), timer_([this, timeout] {
            std::unique_lock lock(mutex_);
            if (!cv_.wait_for(lock, timeout, [this] { return state_ != State::pending; })) fail_locked();
        }) {}
    StartupScope(const StartupScope&) = delete;
    StartupScope& operator=(const StartupScope&) = delete;
    ~StartupScope() { fail(); timer_.join(); }
    void fail() {
        std::lock_guard lock(mutex_);
        fail_locked();
    }
    void complete() {
        std::lock_guard lock(mutex_);
        if (state_ == State::failed) throw std::runtime_error("layercut startup cancelled or timed out");
        state_ = State::ready;
        cv_.notify_all();
    }
};
inline constexpr std::uint32_t kAct = 0x4C43'4101U, kKv = 0x4C43'4B02U;
enum class Act { relay, direct };
inline const char* act_name(Act a) { return a == Act::relay ? "relay" : "direct"; }
struct Cli { std::optional<std::string> split, cut, block, act; };
struct Inputs {
    chain::ModelShape model;
    std::uint64_t rows = 0;
    double rtx_rows_s = 0, pair_rows_s = 0, link_mbps = 0;
};
struct Resolved {
    chain::Split split = chain::Split::device_only;
    std::uint32_t cut = 0, block = 2048;
    Act act = Act::relay;
    bool planned = false;
    std::uint32_t first = 0;
    std::string reason, lines;
};
using Lookup = std::function<const char*(const char*)>;
inline std::uint32_t number(std::string_view value, const char* key) {
    std::uint32_t n = 0;
    const auto r = std::from_chars(value.data(), value.data() + value.size(), n);
    if (r.ec != std::errc{} || r.ptr != value.data() + value.size() || value.empty())
        throw std::invalid_argument(std::string(key) + ": expected an unsigned decimal integer");
    return n;
}
inline double link_class(std::string_view s) {
    if (s == "1gbe") return 118;
    if (s == "2.5gbe") return 290;
    if (s == "10gbe") return 1180;
    if (s == "usb4") return 4600;
    throw std::invalid_argument("STRIX_HANDOFF_LINK: expected 1gbe|2.5gbe|10gbe|usb4");
}
inline Resolved resolve(const Cli& cli, const Inputs& input, Lookup env = [](const char* k) { return envcfg::text(k); }) {
    struct Value { std::string text, origin; };
    const auto read = [&](const std::optional<std::string>& c, const char* key, const char* fallback) -> Value {
        if (c) return {*c, "cli"}; // an overridden environment value is never parsed
        const char* e = env(key);
        return e && *e ? Value{e, "env"} : Value{fallback, "default"};
    };
    auto s = read(cli.split, "STRIX_SPLIT", "device-only");
    auto c = read(cli.cut, "STRIX_SPLIT_CUT", "auto");
    auto b = read(cli.block, "STRIX_SPLIT_BLOCK", "2048");
    auto a = read(cli.act, "STRIX_SPLIT_ACT", "relay");
    Resolved out;
    if (s.text == "device-only") out.split = chain::Split::device_only;
    else if (s.text == "handoff-full") out.split = chain::Split::handoff;
    else if (s.text == "layer-cut") out.split = chain::Split::layer_cut;
    else if (s.text == "auto") out.split = chain::Split::auto_plan;
    else throw std::invalid_argument("STRIX_SPLIT: expected device-only|handoff-full|layer-cut|auto");
    if (a.text == "relay") out.act = Act::relay;
    else if (a.text == "direct") out.act = Act::direct;
    else throw std::invalid_argument("STRIX_SPLIT_ACT: expected relay|direct");
    if (c.text != "auto") {
        out.cut = number(c.text, "STRIX_SPLIT_CUT");
        if (out.cut == 0 || out.cut >= input.model.layers) throw std::invalid_argument("STRIX_SPLIT_CUT: outside [1,L-1]");
    }
    if (b.text != "auto") {
        out.block = number(b.text, "STRIX_SPLIT_BLOCK");
        if (out.block == 0 || out.block % 16 != 0) throw std::invalid_argument("STRIX_SPLIT_BLOCK: expected positive multiple of 16");
    }
    const bool plan = s.text == "auto" || b.text == "auto" ||
        (c.text == "auto" && (c.origin != "default" || out.split == chain::Split::layer_cut));
    if (plan) {
        double mbps = input.link_mbps;
        if (mbps == 0) { const char* e = env("STRIX_HANDOFF_LINK"); if (e && *e) mbps = link_class(e); }
        if (!(input.rtx_rows_s > 0) || !std::isfinite(input.rtx_rows_s) || !(input.pair_rows_s > 0) ||
            !std::isfinite(input.pair_rows_s) || !(mbps > 0) || !std::isfinite(mbps) || input.rows == 0)
            throw std::invalid_argument("STRIX_SPLIT: auto needs declared RTX/pair rows/s, link MB/s and rows");
        std::vector<chain::Device> devices{{"rtx", chain::scaled_costs(input.rtx_rows_s, input.model.layers)},
                                          {"pair", chain::scaled_costs(input.pair_rows_s, input.model.layers)}};
        chain::Request req;
        req.split = out.split; req.rows = input.rows; req.decoders = {1};
        req.blocks = b.text == "auto" ? std::vector<std::uint32_t>{256, 512, 1024, 2048, 4096} : std::vector<std::uint32_t>{out.block};
        if (c.text != "auto") req.legal_cut = [cut = out.cut](std::uint32_t k) { return k == cut; };
        if (out.split == chain::Split::layer_cut) req.cuts = {out.cut};
        const auto best = chain::choose(devices, {{mbps * 1000, mbps * 1000, 0}}, input.model, req).best;
        if (!best.valid) throw std::invalid_argument("STRIX_SPLIT: " + best.reason);
        out.planned = true; out.first = best.first;
        if (s.text == "auto") {
            out.split = best.stages() == 2 ? chain::Split::layer_cut : (best.first == best.decoder ? chain::Split::device_only : chain::Split::handoff);
            s.origin = "planner";
        }
        if (c.text == "auto") { out.cut = best.stages() == 2 ? best.cuts[1] : 0; c.origin = "planner"; }
        if (b.text == "auto") { out.block = best.block; b.origin = "planner"; }
        out.reason = best.reason + "; declared rates, qualification pending";
    } else out.reason = "explicit policy; default route unchanged; layer-cut qualification pending";
    if (out.split == chain::Split::layer_cut && (out.cut == 0 || out.cut >= input.model.layers))
        throw std::invalid_argument("STRIX_SPLIT_CUT: layer-cut requires [1,L-1]");
    const auto line = [&](const char* key, const std::string& v, const std::string& origin) {
        out.lines += std::string("split ") + key + " " + v + " (" + origin + ")\n";
    };
    line("split", chain::split_name(out.split), s.origin);
    line("cut", std::to_string(out.cut), c.origin);
    line("block", std::to_string(out.block), b.origin);
    line("act", act_name(out.act), a.origin);
    return out;
}
struct Block { std::uint64_t row0; std::uint32_t rows; };
inline std::vector<Block> blocks(std::uint64_t n, std::uint32_t c) {
    if (c == 0 || c % 16 != 0) throw std::invalid_argument("STRIX_SPLIT_BLOCK: expected positive multiple of 16");
    std::vector<Block> v;
    for (std::uint64_t r = 0; r < n;) {
        const auto rows = static_cast<std::uint32_t>(std::min<std::uint64_t>(c, n - r));
        v.push_back({r, rows}); r += rows;
    }
    return v;
}
struct Meta {
    std::uint32_t version = 1, cut = 0, block = 2048, hidden = 0;
    std::uint64_t rows = 0, token_fnv64 = 0;
    std::string identity;
};
// Little endian integers, followed by u32 byte length and UTF-8 identity (no NUL).
inline std::vector<std::uint8_t> pack(const Meta& m) {
    std::vector<std::uint8_t> v;
    const auto put = [&](std::uint64_t x, unsigned n) { for (unsigned i = 0; i < n; ++i) v.push_back(static_cast<std::uint8_t>(x >> (8 * i))); };
    if (m.identity.size() > 65500) throw std::invalid_argument("layercut meta identity too long");
    put(m.version, 4); put(m.cut, 4); put(m.block, 4); put(m.hidden, 4);
    put(m.rows, 8); put(m.token_fnv64, 8); put(m.identity.size(), 4);
    v.insert(v.end(), m.identity.begin(), m.identity.end());
    return v;
}
inline Meta unpack(std::span<const std::uint8_t> v) {
    std::size_t at = 0;
    const auto get = [&](unsigned n) {
        if (n > v.size() - at) throw std::invalid_argument("layercut meta truncated");
        std::uint64_t x = 0;
        for (unsigned i = 0; i < n; ++i) x |= std::uint64_t{v[at++]} << (8 * i);
        return x;
    };
    Meta m;
    m.version = static_cast<std::uint32_t>(get(4)); m.cut = static_cast<std::uint32_t>(get(4));
    m.block = static_cast<std::uint32_t>(get(4)); m.hidden = static_cast<std::uint32_t>(get(4));
    m.rows = get(8); m.token_fnv64 = get(8);
    const auto n = get(4);
    if (n != v.size() - at) throw std::invalid_argument("layercut meta identity length mismatch");
    m.identity.assign(reinterpret_cast<const char*>(v.data() + at), static_cast<std::size_t>(n));
    return m;
}
inline void validate(const Meta& m, const Meta& expected, std::uint32_t layers) {
    const auto bad = [](const char* s) { throw std::invalid_argument(std::string("layercut meta mismatch: ") + s); };
    if (m.version != 1 || m.version != expected.version) bad("version");
    if (m.cut == 0 || m.cut >= layers || m.cut != expected.cut) bad("cut");
    if (m.block == 0 || m.block % 16 != 0 || m.block != expected.block) bad("block");
    if (m.hidden == 0 || m.hidden != expected.hidden) bad("hidden");
    if (m.rows == 0 || m.rows != expected.rows) bad("rows");
    if (m.token_fnv64 != expected.token_fnv64) bad("token_fnv64");
    if (m.identity.empty() || m.identity != expected.identity) bad("identity");
}
inline std::uint64_t token_fnv(std::span<const std::uint32_t> toks) {
    std::uint64_t h = 14695981039346656037ULL;
    for (auto t : toks) for (unsigned i = 0; i < 4; ++i) { h = (h ^ static_cast<std::uint8_t>(t >> (i * 8))) * 1099511628211ULL; }
    return h;
}
// Checks order/coverage even when a valid handoff manifest describes incomplete or duplicated model pieces.
class Pieces {
    Meta m_;
    bool act_, kv_, relay_;
    std::uint64_t row_ = 0;
    std::uint32_t block_ = 0, layer_ = 0;
    bool want_act_;
public:
    Pieces(Meta m, bool act, bool kv, bool relay) : m_(std::move(m)), act_(act), kv_(kv), relay_(relay), want_act_(act) {}
    void accept(const handoff::PieceHeader& p, std::size_t bytes, std::uint64_t state_row_bytes) {
        const auto rows = std::min<std::uint64_t>(m_.block, m_.rows - row_);
        const bool act = want_act_;
        if (row_ >= m_.rows || p.row0 != row_ || p.rows != rows ||
            p.kind != (act ? kAct : kKv) || p.index != (act ? block_ : layer_) ||
            p.flags != (act && relay_ ? handoff::kRelay : 0U) ||
            bytes != rows * (act ? std::uint64_t{m_.hidden} * 4 : state_row_bytes))
            throw std::invalid_argument("layercut piece: kind/index/flags/range/size/order mismatch");
        if (act && kv_) { want_act_ = false; return; }
        if (!act && ++layer_ < m_.cut) return;
        row_ += rows; ++block_; layer_ = 0; want_act_ = act_;
    }
    void finish() const { if (row_ != m_.rows) throw std::invalid_argument("layercut END: incomplete coverage"); }
};
// Component-major block -> component-major full layer; widths are bytes/row of K, V, IK.
inline void scatter_state(std::span<std::uint8_t> dst, std::span<const std::uint8_t> src,
                          std::uint64_t n, std::uint64_t row0, std::uint64_t rows,
                          std::span<const std::uint32_t> widths) {
    std::uint64_t off = 0, in = 0, width = 0;
    for (auto w : widths) width += w;
    if (row0 > n || rows > n - row0 || dst.size() != n * width || src.size() != rows * width)
        throw std::invalid_argument("layercut state range/size mismatch");
    for (auto w : widths) {
        std::memcpy(dst.data() + off + row0 * w, src.data() + in, static_cast<std::size_t>(rows * w));
        off += n * w; in += rows * w;
    }
}
} // namespace strix::pair::layercut
