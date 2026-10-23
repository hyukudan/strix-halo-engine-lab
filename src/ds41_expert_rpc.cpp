#include "strix/ds41_expert_rpc.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace strix::ds41::xrpc {

namespace {

// ---- little-endian encoding ----
class Writer {
public:
    void u8(std::uint8_t v) { b_.push_back(v); }
    void u16(std::uint16_t v) { put(&v, 2); }
    void u32(std::uint32_t v) { put(&v, 4); }
    void u64(std::uint64_t v) { put(&v, 8); }
    void f32(float v) { put(&v, 4); }
    void bytes(const void* p, std::size_t n) { put(p, n); }
    void text(const std::string& s) {
        u32(static_cast<std::uint32_t>(s.size()));
        put(s.data(), s.size());
    }
    template <class T>
    void array(const std::vector<T>& v) {
        put(v.data(), v.size() * sizeof(T));
    }
    std::vector<std::uint8_t> take() { return std::move(b_); }
    void reserve(std::size_t n) { b_.reserve(n); }

private:
    void put(const void* p, std::size_t n) {
        const auto* c = static_cast<const std::uint8_t*>(p);
        b_.insert(b_.end(), c, c + n);  // the supported platforms are little-endian
    }
    std::vector<std::uint8_t> b_;
};

class Reader {
public:
    explicit Reader(std::span<const std::uint8_t> p) : p_(p) {}
    std::uint8_t u8() { return get<std::uint8_t>(); }
    std::uint16_t u16() { return get<std::uint16_t>(); }
    std::uint32_t u32() { return get<std::uint32_t>(); }
    std::uint64_t u64() { return get<std::uint64_t>(); }
    float f32() { return get<float>(); }
    std::string text(std::size_t max = 1U << 16U) {
        const auto n = u32();
        if (n > max) throw WireError("ds41 expert rpc: text field too long");
        need(n);
        std::string s(reinterpret_cast<const char*>(p_.data() + at_), n);
        at_ += n;
        return s;
    }
    template <class T>
    void array(std::vector<T>& v, std::size_t count) {
        need(count * sizeof(T));
        v.resize(count);
        if (count != 0U) std::memcpy(v.data(), p_.data() + at_, count * sizeof(T));
        at_ += count * sizeof(T);
    }
    void end() const {
        if (at_ != p_.size()) throw WireError("ds41 expert rpc: trailing bytes in a message");
    }
    [[nodiscard]] std::size_t at() const noexcept { return at_; }

private:
    void need(std::size_t n) const {
        if (n > p_.size() - at_) throw WireError("ds41 expert rpc: message too short");
    }
    template <class T>
    T get() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p_.data() + at_, sizeof(T));
        at_ += sizeof(T);
        return v;
    }
    std::span<const std::uint8_t> p_;
    std::size_t at_ = 0;
};

std::uint32_t fnv32(const std::uint8_t* p, std::size_t n) {
    std::uint32_t h = 2166136261U;
    for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 16777619U;
    return h;
}

void put_geometry(Writer& w, const Geometry& g) {
    w.u32(g.layers); w.u32(g.hidden); w.u32(g.inter); w.u32(g.experts); w.u32(g.top_k); w.u32(g.split);
}
Geometry get_geometry(Reader& r) {
    Geometry g;
    g.layers = r.u32(); g.hidden = r.u32(); g.inter = r.u32(); g.experts = r.u32(); g.top_k = r.u32(); g.split = r.u32();
    if (g.hidden == 0U || g.hidden % 32U != 0U || g.top_k == 0U || g.top_k > kMaxTopK || g.experts == 0U || g.experts > (1U << 20U) ||
        g.split > g.experts || g.layers == 0U || g.layers > 64U)
        throw WireError("ds41 expert rpc: bad geometry");
    return g;
}

// The row's slots in ascending expert id and how many have id < split.
void row_order(std::span<const std::uint32_t> ids, std::uint32_t r, std::uint32_t top_k, std::uint32_t split, std::uint32_t experts,
               std::uint32_t* order, std::uint32_t& low) {
    std::array<std::pair<std::uint32_t, std::uint32_t>, kMaxTopK> row{};
    for (std::uint32_t k = 0; k < top_k; ++k) {
        const auto id = ids[std::size_t{r} * top_k + k];
        if (id >= experts) throw WireError("ds41 expert rpc: expert id out of range");
        for (std::uint32_t j = 0; j < k; ++j)
            if (row[j].first == id) throw WireError("ds41 expert rpc: an expert repeats within a row");
        row[k] = {id, r * top_k + k};
    }
    std::sort(row.begin(), row.begin() + top_k);
    low = 0;
    for (std::uint32_t k = 0; k < top_k; ++k) {
        order[k] = row[k].second;
        low += row[k].first < split ? 1U : 0U;
    }
}

#if defined(_WIN32)
using sock_t = SOCKET;
constexpr sock_t kBad = INVALID_SOCKET;
int last_error() { return WSAGetLastError(); }
bool would_block(int e) { return e == WSAEWOULDBLOCK; }
bool interrupted(int e) { return e == WSAEINTR; }
void close_socket(std::intptr_t fd) { ::closesocket(static_cast<sock_t>(fd)); }
struct WinsockInit {
    WinsockInit() {
        WSADATA d;
        if (WSAStartup(MAKEWORD(2, 2), &d) != 0) throw std::runtime_error("WSAStartup failed");
    }
};
void ensure_init() { static WinsockInit init; }
#else
using sock_t = int;
constexpr sock_t kBad = -1;
int last_error() { return errno; }
bool would_block(int e) { return e == EAGAIN || e == EWOULDBLOCK; }
bool interrupted(int e) { return e == EINTR; }
void close_socket(std::intptr_t fd) { ::close(static_cast<int>(fd)); }
void ensure_init() {}
#endif

[[noreturn]] void sys_fail(const std::string& what) {
    throw WireError("ds41 expert rpc: " + what + " failed (error " + std::to_string(last_error()) + ")");
}

void set_nonblocking(sock_t s, bool on) {
#if defined(_WIN32)
    u_long mode = on ? 1UL : 0UL;
    if (::ioctlsocket(s, FIONBIO, &mode) != 0) sys_fail("FIONBIO");
#else
    const int flags = ::fcntl(s, F_GETFL, 0);
    if (flags < 0 || ::fcntl(s, F_SETFL, on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK)) != 0) sys_fail("O_NONBLOCK");
#endif
}

void tune_socket(sock_t s, std::chrono::milliseconds timeout) {
    int one = 1;
    if (::setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one)) != 0) sys_fail("TCP_NODELAY");
    int buf = 4 << 20;
    (void)::setsockopt(s, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&buf), sizeof(buf));
    (void)::setsockopt(s, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char*>(&buf), sizeof(buf));
    if (timeout.count() > 0) {
#if defined(_WIN32)
        DWORD ms = static_cast<DWORD>(timeout.count());
        (void)::setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
        (void)::setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
#else
        timeval tv{};
        tv.tv_sec = static_cast<time_t>(timeout.count() / 1000);
        tv.tv_usec = static_cast<suseconds_t>((timeout.count() % 1000) * 1000);
        if (::setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0 || ::setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0)
            sys_fail("socket timeouts");
#endif
    }
}

std::string peer_name(sock_t s) {
    sockaddr_storage a{};
    socklen_t n = sizeof(a);
    if (::getpeername(s, reinterpret_cast<sockaddr*>(&a), &n) != 0) return "?";
    char host[NI_MAXHOST] = {}, port[NI_MAXSERV] = {};
    if (::getnameinfo(reinterpret_cast<sockaddr*>(&a), n, host, sizeof(host), port, sizeof(port), NI_NUMERICHOST | NI_NUMERICSERV) != 0) return "?";
    return std::string(host) + ":" + port;
}

}  // namespace

const char* to_string(MsgType type) noexcept {
    switch (type) {
    case MsgType::hello: return "HELLO";
    case MsgType::hello_ack: return "HELLO_ACK";
    case MsgType::experts: return "EXPERTS";
    case MsgType::result: return "RESULT";
    case MsgType::ping: return "PING";
    case MsgType::pong: return "PONG";
    case MsgType::close: return "CLOSE";
    case MsgType::ack: return "ACK";
    case MsgType::error: return "ERROR";
    }
    return "?";
}

std::array<std::uint8_t, kHeaderBytes> encode_header(const Header& h) {
    std::array<std::uint8_t, kHeaderBytes> b{};
    const auto type = static_cast<std::uint16_t>(h.type);
    std::memcpy(b.data(), &h.magic, 4);
    std::memcpy(b.data() + 4, &h.version, 2);
    std::memcpy(b.data() + 6, &type, 2);
    std::memcpy(b.data() + 8, &h.request, 8);
    std::memcpy(b.data() + 16, &h.bytes, 8);
    std::memcpy(b.data() + 24, &h.flags, 4);
    const auto check = fnv32(b.data(), 28);
    std::memcpy(b.data() + 28, &check, 4);
    return b;
}

Header decode_header(std::span<const std::uint8_t> b, std::uint64_t max_bytes) {
    if (b.size() < kHeaderBytes) throw WireError("ds41 expert rpc: short header");
    Header h;
    std::uint16_t type = 0;
    std::uint32_t check = 0;
    std::memcpy(&h.magic, b.data(), 4);
    std::memcpy(&h.version, b.data() + 4, 2);
    std::memcpy(&type, b.data() + 6, 2);
    std::memcpy(&h.request, b.data() + 8, 8);
    std::memcpy(&h.bytes, b.data() + 16, 8);
    std::memcpy(&h.flags, b.data() + 24, 4);
    std::memcpy(&check, b.data() + 28, 4);
    if (h.magic != kMagic) throw WireError("ds41 expert rpc: bad magic (not a strix/ds41-expert-rpc peer)");
    if (h.version != kVersion) throw WireError("ds41 expert rpc: unsupported version " + std::to_string(h.version));
    if (check != fnv32(b.data(), 28)) throw WireError("ds41 expert rpc: header check mismatch");
    if (type < 1U || type > 9U) throw WireError("ds41 expert rpc: unknown message type " + std::to_string(type));
    if (h.bytes > max_bytes) throw WireError("ds41 expert rpc: message of " + std::to_string(h.bytes) + " bytes exceeds the limit");
    h.type = static_cast<MsgType>(type);
    return h;
}

// ---- messages ----

std::vector<std::uint8_t> encode(const Hello& m) {
    Writer w;
    w.bytes(m.token.data(), 32);
    w.u64(m.model_identity);
    put_geometry(w, m.geometry);
    w.text(m.client);
    w.text(m.numeric_identity);
    return w.take();
}
Hello decode_hello(std::span<const std::uint8_t> p) {
    Reader r(p);
    Hello m;
    for (auto& b : m.token) b = r.u8();
    m.model_identity = r.u64();
    m.geometry = get_geometry(r);
    m.client = r.text();
    m.numeric_identity = r.text();
    r.end();
    return m;
}

std::vector<std::uint8_t> encode(const HelloAck& m) {
    Writer w;
    w.u32(m.status);
    w.u64(m.model_identity);
    put_geometry(w, m.geometry);
    w.u32(m.residency.expert_first);
    w.u32(m.residency.expert_end);
    w.u64(m.residency.layer_mask);
    w.u32(m.world);
    w.u32(m.max_rows);
    w.text(m.engine);
    w.text(m.numeric_identity);
    w.text(m.message);
    return w.take();
}
HelloAck decode_hello_ack(std::span<const std::uint8_t> p) {
    Reader r(p);
    HelloAck m;
    m.status = r.u32();
    m.model_identity = r.u64();
    m.geometry = get_geometry(r);
    m.residency.expert_first = r.u32();
    m.residency.expert_end = r.u32();
    m.residency.layer_mask = r.u64();
    m.world = r.u32();
    m.max_rows = r.u32();
    m.engine = r.text();
    m.numeric_identity = r.text();
    m.message = r.text();
    r.end();
    return m;
}

std::uint64_t request_bytes(std::uint32_t rows, std::uint32_t top_k, const Geometry& g) noexcept {
    return kHeaderBytes + 16U + std::uint64_t{rows} * (8U * top_k + 4U + g.a8_row_bytes());
}
std::uint64_t result_bytes(std::uint32_t items, const Geometry& g) noexcept {
    return kHeaderBytes + kResultHeadBytes + std::uint64_t{items} * g.vector_bytes();
}

std::vector<std::uint8_t> encode(const ExpertsRequest& m, const Geometry& g) {
    const std::size_t slots = std::size_t{m.rows} * m.top_k;
    if (m.top_k != g.top_k || m.ids.size() != slots || m.weights.size() != slots || m.remote.size() != m.rows ||
        m.codes.size() != std::size_t{m.rows} * g.hidden || m.deltas.size() != std::size_t{m.rows} * g.groups())
        throw std::invalid_argument("ds41 expert rpc: inconsistent EXPERTS request");
    Writer w;
    w.reserve(static_cast<std::size_t>(request_bytes(m.rows, m.top_k, g) - kHeaderBytes));
    w.u32(m.layer); w.u32(m.rows); w.u32(m.top_k); w.u32(m.flags);
    w.array(m.ids); w.array(m.weights); w.array(m.remote); w.array(m.codes); w.array(m.deltas);
    return w.take();
}
ExpertsRequest decode_experts(std::span<const std::uint8_t> p, const Geometry& g) {
    Reader r(p);
    ExpertsRequest m;
    m.layer = r.u32(); m.rows = r.u32(); m.top_k = r.u32(); m.flags = r.u32();
    if (m.top_k != g.top_k || m.rows == 0U || m.rows > 65536U || m.layer >= g.layers)
        throw WireError("ds41 expert rpc: EXPERTS request shape does not match the session");
    const std::size_t slots = std::size_t{m.rows} * m.top_k;
    r.array(m.ids, slots); r.array(m.weights, slots); r.array(m.remote, m.rows);
    r.array(m.codes, std::size_t{m.rows} * g.hidden); r.array(m.deltas, std::size_t{m.rows} * g.groups());
    r.end();
    for (const auto mask : m.remote)
        if (m.top_k < 32U && (mask >> m.top_k) != 0U) throw WireError("ds41 expert rpc: remote mask names a slot beyond top_k");
    return m;
}

std::vector<std::uint8_t> encode_result_head(std::uint32_t layer, std::uint32_t items, std::uint32_t hidden, const ResultTiming& t) {
    Writer w;
    w.u32(layer); w.u32(items); w.u32(hidden); w.u32(0);
    w.u64(t.server_ns); w.u64(t.compute_ns); w.u64(t.relay_ns);
    return w.take();
}
ResultView view_result(std::span<const std::uint8_t> p) {
    Reader r(p);
    ResultView v;
    v.layer = r.u32(); v.items = r.u32(); v.hidden = r.u32(); (void)r.u32();
    v.timing.server_ns = r.u64(); v.timing.compute_ns = r.u64(); v.timing.relay_ns = r.u64();
    if (p.size() != kResultHeadBytes + std::size_t{v.items} * v.hidden * 4U) throw WireError("ds41 expert rpc: RESULT size mismatch");
    v.vectors = reinterpret_cast<const float*>(p.data() + kResultHeadBytes);
    return v;
}
ExpertsResult decode_result(std::span<const std::uint8_t> p) {
    const auto v = view_result(p);
    ExpertsResult m;
    m.layer = v.layer; m.items = v.items; m.hidden = v.hidden; m.timing = v.timing;
    m.vectors.resize(std::size_t{v.items} * v.hidden);
    if (!m.vectors.empty()) std::memcpy(m.vectors.data(), p.data() + kResultHeadBytes, m.vectors.size() * 4U);
    return m;
}

// ---- plans ----

ResponsePlan plan_response(const ExpertsRequest& req, const Geometry& g, const Residency& residency) {
    const auto K = req.top_k;
    if (K == 0U || K > kMaxTopK || req.ids.size() != std::size_t{req.rows} * K || req.remote.size() != req.rows)
        throw WireError("ds41 expert rpc: inconsistent request for a plan");
    ResponsePlan plan;
    plan.order.resize(std::size_t{req.rows} * K);
    plan.low.resize(req.rows);
    for (std::uint32_t r = 0; r < req.rows; ++r) {
        row_order(req.ids, r, K, g.split, g.experts, plan.order.data() + std::size_t{r} * K, plan.low[r]);
        for (std::uint32_t k = 0; k < K; ++k)
            if (((req.remote[r] >> k) & 1U) != 0U && !residency.serves(req.layer, req.ids[std::size_t{r} * K + k]))
                throw WireError("ds41 expert rpc: a remote slot names expert " + std::to_string(req.ids[std::size_t{r} * K + k]) + " of layer " +
                                std::to_string(req.layer) + ", which this server does not keep");
    }
    const bool fold = (req.flags & kNoFold) == 0U;
    const auto is_remote = [&](std::uint32_t slot) { return ((req.remote[slot / K] >> (slot % K)) & 1U) != 0U; };
    for (std::uint8_t h = 0; h < 2U; ++h)
        for (std::uint32_t r = 0; r < req.rows; ++r) {
            const std::uint32_t first = h == 0U ? 0U : plan.low[r], end = h == 0U ? plan.low[r] : K;
            const auto* seq = plan.order.data() + std::size_t{r} * K;
            std::uint32_t p = first;
            if (fold)
                while (p < end && is_remote(seq[p])) ++p;
            if (p > first) {
                PlanItem it;
                it.kind = PlanItem::Kind::fold;
                it.half = h;
                it.row = r;
                it.first = 0;
                it.count = p - first;
                it.slot = seq[first];
                plan.fold_begin.push_back(static_cast<std::uint32_t>(plan.fold_slots.size()));
                for (auto j = first; j < p; ++j) plan.fold_slots.push_back(seq[j]);
                plan.items.push_back(it);
            }
            for (auto j = p; j < end; ++j) {
                if (!is_remote(seq[j])) continue;
                PlanItem it;
                it.kind = PlanItem::Kind::raw;
                it.half = h;
                it.row = r;
                it.first = j - first;
                it.count = 1;
                it.slot = seq[j];
                plan.fold_begin.push_back(0);
                plan.items.push_back(it);
            }
        }
    std::vector<std::pair<std::uint32_t, std::uint32_t>> entries;  // (expert, slot)
    for (std::uint32_t s = 0; s < req.rows * K; ++s)
        if (is_remote(s)) entries.emplace_back(req.ids[s], s);
    std::sort(entries.begin(), entries.end());
    plan.entry_slots.reserve(entries.size());
    for (const auto& e : entries) plan.entry_slots.push_back(e.second);
    return plan;
}

CombinePlan plan_combine(std::uint32_t rows, std::uint32_t top_k, std::span<const std::uint32_t> ids, std::span<const std::uint32_t> remote,
                         std::uint32_t split, std::span<const Answer> answers) {
    if (top_k == 0U || top_k > kMaxTopK || ids.size() != std::size_t{rows} * top_k || remote.size() != rows)
        throw WireError("ds41 expert rpc: inconsistent window for a combine plan");
    CombinePlan cp;
    cp.rows = rows;
    cp.top_k = top_k;
    cp.order.resize(std::size_t{rows} * top_k);
    cp.low.resize(rows);
    for (std::uint32_t r = 0; r < rows; ++r)
        row_order(ids, r, top_k, split, 0xffffffffU, cp.order.data() + std::size_t{r} * top_k, cp.low[r]);
    cp.seed.assign(std::size_t{rows} * 2U, -1);
    cp.start.assign(std::size_t{rows} * 2U, 0U);
    cp.source.assign(std::size_t{rows} * top_k, -1);
    std::vector<std::uint8_t> covered(std::size_t{rows} * top_k, 0U);
    for (std::size_t a = 0; a < answers.size(); ++a) {
        const auto& plan = *answers[a].plan;
        if (plan.items.size() >= (1U << 24U) || a >= 256U) throw WireError("ds41 expert rpc: too many answers or items");
        const auto map_row = [&](std::uint32_t r) {
            if (answers[a].row_map.empty()) return r;
            if (r >= answers[a].row_map.size()) throw WireError("ds41 expert rpc: answer row outside its row map");
            return answers[a].row_map[r];
        };
        for (std::size_t i = 0; i < plan.items.size(); ++i) {
            const auto& it = plan.items[i];
            const auto wr = map_row(it.row);
            if (wr >= rows) throw WireError("ds41 expert rpc: answer row outside the window");
            const auto src = static_cast<std::int64_t>((a << 24U) | i);
            if (it.kind == PlanItem::Kind::fold) {
                auto& seed = cp.seed[std::size_t{wr} * 2U + it.half];
                if (seed >= 0) throw WireError("ds41 expert rpc: two folds for one (row, half)");
                seed = src;
                cp.start[std::size_t{wr} * 2U + it.half] = it.count;
                for (std::uint32_t j = 0; j < it.count; ++j) {
                    const auto s = plan.fold_slots[plan.fold_begin[i] + j];
                    const auto ws = wr * top_k + s % top_k;
                    if (covered[ws]++ != 0U) throw WireError("ds41 expert rpc: a slot answered twice");
                }
            } else {
                const auto ws = wr * top_k + it.slot % top_k;
                if (covered[ws]++ != 0U) throw WireError("ds41 expert rpc: a slot answered twice");
                cp.source[ws] = src;
            }
        }
    }
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t k = 0; k < top_k; ++k) {
            const bool rem = ((remote[r] >> k) & 1U) != 0U;
            if (rem != (covered[std::size_t{r} * top_k + k] != 0U))
                throw WireError(rem ? "ds41 expert rpc: a remote slot has no answer" : "ds41 expert rpc: an answer covers a local slot");
        }
        // a seed must cover exactly the leading slots of its half (in this window's order)
        for (std::uint32_t h = 0; h < 2U; ++h) {
            const auto s = cp.seed[std::size_t{r} * 2U + h];
            if (s < 0) continue;
            const auto& plan = *answers[static_cast<std::size_t>(s >> 24)].plan;
            const auto i = static_cast<std::size_t>(s & 0xffffff);
            const auto first = h == 0U ? 0U : cp.low[r];
            for (std::uint32_t j = 0; j < plan.items[i].count; ++j)
                if (plan.fold_slots[plan.fold_begin[i] + j] % top_k != cp.order[std::size_t{r} * top_k + first + j] % top_k)
                    throw WireError("ds41 expert rpc: a fold does not cover the leading slots of its half");
        }
    }
    return cp;
}

// ---- token ----

std::array<std::uint8_t, 32> read_token_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("ds41 expert rpc: cannot read the token file " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string hex;
    for (const char c : ss.str())
        if (!std::isspace(static_cast<unsigned char>(c))) hex.push_back(c);
    if (hex.size() != 64U) throw std::runtime_error("ds41 expert rpc: the token file must hold 64 hex characters");
    std::array<std::uint8_t, 32> t{};
    for (std::size_t i = 0; i < 32U; ++i) {
        const auto nib = [&](char c) -> unsigned {
            if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
            if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
            if (c >= 'A' && c <= 'F') return static_cast<unsigned>(c - 'A' + 10);
            throw std::runtime_error("ds41 expert rpc: the token file holds a non-hex character");
        };
        t[i] = static_cast<std::uint8_t>(nib(hex[2U * i]) << 4U | nib(hex[2U * i + 1U]));
    }
    return t;
}
bool same_token(const std::array<std::uint8_t, 32>& a, const std::array<std::uint8_t, 32>& b) noexcept {
    std::uint8_t d = 0;
    for (std::size_t i = 0; i < 32U; ++i) d = static_cast<std::uint8_t>(d | (a[i] ^ b[i]));
    return d == 0U;
}
std::uint64_t fnv_floats(std::span<const float> v) noexcept {
    std::uint64_t h = 14695981039346656037ULL;
    for (const float f : v) {
        std::uint32_t b;
        std::memcpy(&b, &f, 4);
        for (int k = 0; k < 4; ++k, b >>= 8U) h = (h ^ (b & 0xffU)) * 1099511628211ULL;
    }
    return h;
}

// ---- TCP ----

Connection::Connection(std::intptr_t fd, std::chrono::milliseconds timeout) : fd_(fd), timeout_(timeout) {
    tune_socket(static_cast<sock_t>(fd_), timeout);
    peer_ = peer_name(static_cast<sock_t>(fd_));
}
void Connection::set_busy_poll(bool on) {
    set_nonblocking(static_cast<sock_t>(fd_), on);
    busy_poll_ = on;
}
Connection::~Connection() {
    if (fd_ != -1) close_socket(fd_);
}

void Connection::write_gather(std::span<const std::span<const std::uint8_t>> parts) {
    // one system call per round over every remaining part (one segment for small messages with TCP_NODELAY)
    std::vector<std::span<const std::uint8_t>> left;
    for (const auto& p : parts)
        if (!p.empty()) left.push_back(p);
    const auto start = std::chrono::steady_clock::now();
    std::size_t at = 0;
    while (at < left.size()) {
        const std::size_t count = std::min<std::size_t>(left.size() - at, 16U);
#if defined(_WIN32)
        WSABUF bufs[16];
        for (std::size_t i = 0; i < count; ++i) {
            bufs[i].buf = reinterpret_cast<CHAR*>(const_cast<std::uint8_t*>(left[at + i].data()));
            bufs[i].len = static_cast<ULONG>(std::min<std::size_t>(left[at + i].size(), 1U << 30U));
        }
        DWORD sent = 0;
        const int rc = ::WSASend(static_cast<sock_t>(fd_), bufs, static_cast<DWORD>(count), &sent, 0, nullptr, nullptr);
        long long k = rc == 0 ? static_cast<long long>(sent) : -1;
#else
        iovec iov[16];
        for (std::size_t i = 0; i < count; ++i) {
            iov[i].iov_base = const_cast<std::uint8_t*>(left[at + i].data());
            iov[i].iov_len = left[at + i].size();
        }
        msghdr mh{};
        mh.msg_iov = iov;
        mh.msg_iovlen = count;
        const long long k = ::sendmsg(static_cast<sock_t>(fd_), &mh, MSG_NOSIGNAL);
#endif
        if (k > 0) {
            auto n = static_cast<std::size_t>(k);
            while (n != 0U) {
                auto& p = left[at];
                const auto take = std::min(n, p.size());
                p = p.subspan(take);
                n -= take;
                if (p.empty()) ++at;
            }
            continue;
        }
        const int e = last_error();
        if (k < 0 && interrupted(e)) continue;
        if (k < 0 && would_block(e)) {
            if (busy_poll_ && (timeout_.count() == 0 || std::chrono::steady_clock::now() - start < timeout_)) continue;
            throw WireError("ds41 expert rpc: send timed out to " + peer_);
        }
        sys_fail("send to " + peer_);
    }
}

void Connection::read_all(std::uint8_t* p, std::size_t n) {
    const auto start = std::chrono::steady_clock::now();
    while (n != 0U) {
        const int chunk = static_cast<int>(std::min<std::size_t>(n, 1U << 30U));
#if defined(_WIN32)
        const int k = ::recv(static_cast<sock_t>(fd_), reinterpret_cast<char*>(p), chunk, 0);
#else
        const auto k = ::recv(static_cast<sock_t>(fd_), p, static_cast<std::size_t>(chunk), 0);
#endif
        if (k > 0) {
            p += k;
            n -= static_cast<std::size_t>(k);
            continue;
        }
        if (k == 0) throw WireError("ds41 expert rpc: connection closed by " + peer_);
        const int e = last_error();
        if (interrupted(e)) continue;
        if (would_block(e)) {
            if (busy_poll_ && (timeout_.count() == 0 || std::chrono::steady_clock::now() - start < timeout_)) continue;
            throw WireError("ds41 expert rpc: receive timed out from " + peer_);
        }
        sys_fail("receive from " + peer_);
    }
#if defined(__linux__)
    int one = 1;  // answer promptly: no delayed ACK on a request/answer link
    (void)::setsockopt(static_cast<sock_t>(fd_), IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));
#endif
}

void Connection::send(MsgType type, std::uint64_t request, std::span<const std::uint8_t> payload, std::uint32_t flags) {
    const std::span<const std::uint8_t> parts[1] = {payload};
    send_parts(type, request, parts, flags);
}

void Connection::send_parts(MsgType type, std::uint64_t request, std::span<const std::span<const std::uint8_t>> parts, std::uint32_t flags) {
    Header h;
    h.type = type;
    h.request = request;
    h.flags = flags;
    std::uint64_t total = 0;
    for (const auto& p : parts) total += p.size();
    h.bytes = total;
    const auto head = encode_header(h);
    std::vector<std::span<const std::uint8_t>> all;
    all.reserve(parts.size() + 1U);
    all.emplace_back(head.data(), kHeaderBytes);
    for (const auto& p : parts) all.push_back(p);
    write_gather(all);
    sent_ += kHeaderBytes + total;
}

void Connection::send_begin(MsgType type, std::uint64_t request, std::uint64_t bytes, std::span<const std::span<const std::uint8_t>> first,
                            std::uint32_t flags) {
    Header h;
    h.type = type;
    h.request = request;
    h.flags = flags;
    h.bytes = bytes;
    const auto head = encode_header(h);
    std::vector<std::span<const std::uint8_t>> all;
    all.emplace_back(head.data(), kHeaderBytes);
    std::uint64_t n = 0;
    for (const auto& p : first) { all.push_back(p); n += p.size(); }
    if (n > bytes) throw std::invalid_argument("ds41 expert rpc: more first bytes than the message holds");
    write_gather(all);
    sent_ += kHeaderBytes + n;
}
void Connection::send_raw(std::span<const std::uint8_t> bytes) {
    const std::span<const std::uint8_t> one[1] = {bytes};
    write_gather(one);
    sent_ += bytes.size();
}

Header Connection::receive_into(std::vector<std::uint8_t>& payload, std::uint64_t max_bytes) {
    std::array<std::uint8_t, kHeaderBytes> head{};
    read_all(head.data(), kHeaderBytes);
    const auto h = decode_header(head, max_bytes);
    payload.resize(static_cast<std::size_t>(h.bytes));
    if (h.bytes != 0U) read_all(payload.data(), payload.size());
    received_ += kHeaderBytes + h.bytes;
    return h;
}

Message Connection::receive(std::uint64_t max_bytes) {
    Message m;
    m.header = receive_into(m.payload, max_bytes);
    return m;
}

Listener::Listener(const std::string& host, std::uint16_t port) {
    ensure_init();
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;
    const auto service = std::to_string(port);
    if (::getaddrinfo(host.empty() ? nullptr : host.c_str(), service.c_str(), &hints, &res) != 0 || res == nullptr)
        throw WireError("ds41 expert rpc: cannot resolve the listen address " + host);
    const sock_t s = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (s == kBad) {
        ::freeaddrinfo(res);
        sys_fail("socket");
    }
    fd_ = static_cast<std::intptr_t>(s);
    int one = 1;
    (void)::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one), sizeof(one));
    if (::bind(s, res->ai_addr, static_cast<socklen_t>(res->ai_addrlen)) != 0) {
        ::freeaddrinfo(res);
        sys_fail("bind " + host + ":" + service);
    }
    ::freeaddrinfo(res);
    if (::listen(s, 4) != 0) sys_fail("listen");
    sockaddr_storage a{};
    socklen_t n = sizeof(a);
    if (::getsockname(s, reinterpret_cast<sockaddr*>(&a), &n) != 0) sys_fail("getsockname");
    port_ = a.ss_family == AF_INET6 ? ntohs(reinterpret_cast<sockaddr_in6*>(&a)->sin6_port) : ntohs(reinterpret_cast<sockaddr_in*>(&a)->sin_port);
}
Listener::~Listener() {
    if (fd_ != -1) close_socket(fd_);
}

std::unique_ptr<Connection> Listener::accept(std::chrono::milliseconds wait, std::chrono::milliseconds io_timeout) {
    const auto s = static_cast<sock_t>(fd_);
#if defined(_WIN32)
    WSAPOLLFD p{};
    p.fd = s;
    p.events = POLLRDNORM;
    const int r = ::WSAPoll(&p, 1, wait.count() > 0 ? static_cast<int>(wait.count()) : -1);
#else
    pollfd p{};
    p.fd = s;
    p.events = POLLIN;
    int r;
    do r = ::poll(&p, 1, wait.count() > 0 ? static_cast<int>(wait.count()) : -1);
    while (r < 0 && errno == EINTR);
#endif
    if (r == 0) return nullptr;
    if (r < 0) sys_fail("poll on the listener");
    const sock_t c = ::accept(s, nullptr, nullptr);
    if (c == kBad) sys_fail("accept");
    return std::make_unique<Connection>(static_cast<std::intptr_t>(c), io_timeout);
}

std::pair<std::string, std::uint16_t> split_host_port(const std::string& s) {
    const auto c = s.rfind(':');
    if (c == std::string::npos || c == 0U || c + 1U == s.size()) throw std::invalid_argument("expected HOST:PORT, got " + s);
    const auto port = std::stoul(s.substr(c + 1));
    if (port > 65535U) throw std::invalid_argument("port out of range in " + s);
    return {s.substr(0, c), static_cast<std::uint16_t>(port)};
}

std::unique_ptr<Connection> connect(const std::string& host_port, std::chrono::milliseconds wait, std::chrono::milliseconds io_timeout) {
    ensure_init();
    const auto [host, port] = split_host_port(host_port);
    const auto deadline = std::chrono::steady_clock::now() + wait;
    for (;;) {
        addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_NUMERICSERV;
        const auto service = std::to_string(port);
        if (::getaddrinfo(host.c_str(), service.c_str(), &hints, &res) == 0 && res != nullptr) {
            for (auto* a = res; a != nullptr; a = a->ai_next) {
                const sock_t s = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
                if (s == kBad) continue;
                if (::connect(s, a->ai_addr, static_cast<socklen_t>(a->ai_addrlen)) == 0) {
                    ::freeaddrinfo(res);
                    return std::make_unique<Connection>(static_cast<std::intptr_t>(s), io_timeout);
                }
                close_socket(static_cast<std::intptr_t>(s));
            }
            ::freeaddrinfo(res);
        }
        if (std::chrono::steady_clock::now() >= deadline) throw WireError("ds41 expert rpc: cannot connect to " + host_port);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

// ---- client ----

ExpertClient::ExpertClient(const std::string& host_port, const Hello& hello, std::chrono::milliseconds wait, std::chrono::milliseconds io_timeout,
                           bool busy_poll)
    : conn_(connect(host_port, wait, io_timeout)), geometry_(hello.geometry) {
    conn_->send(MsgType::hello, 0, encode(hello));
    const auto m = conn_->receive(1U << 20U);
    if (m.header.type == MsgType::error)
        throw WireError("ds41 expert rpc: server refused the session: " + std::string(m.payload.begin(), m.payload.end()));
    if (m.header.type != MsgType::hello_ack) throw WireError("ds41 expert rpc: expected HELLO_ACK");
    ack_ = decode_hello_ack(m.payload);
    if (ack_.status != 0U) throw WireError("ds41 expert rpc: server refused the session (status " + std::to_string(ack_.status) + "): " + ack_.message);
    if (!(ack_.geometry == geometry_)) throw WireError("ds41 expert rpc: server geometry differs");
    if (busy_poll) conn_->set_busy_poll(true);
}

ExpertClient::~ExpertClient() = default;

std::uint64_t ExpertClient::submit(const ExpertsRequest& req) {
    scratch_ = encode(req, geometry_);
    const auto id = next_++;
    conn_->send(MsgType::experts, id, scratch_);
    return id;
}

ResultView ExpertClient::collect(std::vector<std::uint8_t>& payload) {
    const auto h = conn_->receive_into(payload);
    if (h.type == MsgType::error) throw WireError("ds41 expert rpc: server error: " + std::string(payload.begin(), payload.end()));
    if (h.type != MsgType::result) throw WireError("ds41 expert rpc: expected RESULT");
    if (h.request != expected_) throw WireError("ds41 expert rpc: RESULT out of order");
    ++expected_;
    return view_result(payload);
}

std::string ExpertClient::ping() {
    conn_->send(MsgType::ping, 0, {});
    const auto m = conn_->receive(1U << 20U);
    if (m.header.type != MsgType::pong) throw WireError("ds41 expert rpc: expected PONG");
    return std::string(m.payload.begin(), m.payload.end());
}

void ExpertClient::close() {
    conn_->send(MsgType::close, 0, {});
    const auto m = conn_->receive(1U << 20U);
    if (m.header.type != MsgType::ack) throw WireError("ds41 expert rpc: expected ACK to CLOSE");
}

}  // namespace strix::ds41::xrpc
