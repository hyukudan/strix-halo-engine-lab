#include "strix/ds41_pipeline_wire.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iterator>
#include <thread>

#include <cctype>

// The messages and the cut are portable; the TCP classes below are POSIX sockets (a Windows FRONT brings its own socket
// code and uses the messages only).
#ifndef _WIN32
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace strix::ds41::pipe {
namespace {

// ---- little-endian writer / bounds-checked reader ----

class Writer {
public:
    template <typename T>
    void put(T v) {
        static_assert(std::is_integral_v<T> || std::is_enum_v<T>);
        auto u = static_cast<std::uint64_t>(v);
        for (std::size_t i = 0; i < sizeof(T); ++i, u >>= 8U) out_.push_back(static_cast<std::uint8_t>(u & 0xffU));
    }
    void bytes(std::span<const std::uint8_t> b) { out_.insert(out_.end(), b.begin(), b.end()); }
    void floats(std::span<const float> v) {
        const auto* b = reinterpret_cast<const std::uint8_t*>(v.data());
        out_.insert(out_.end(), b, b + v.size() * 4U);
    }
    void u32s(std::span<const std::uint32_t> v) {
        for (const auto x : v) put(x);
    }
    void text(const std::string& s) {
        put(static_cast<std::uint32_t>(s.size()));
        out_.insert(out_.end(), s.begin(), s.end());
    }
    std::vector<std::uint8_t> take() { return std::move(out_); }

private:
    std::vector<std::uint8_t> out_;
};

class Reader {
public:
    explicit Reader(std::span<const std::uint8_t> p, const char* what) : p_(p), what_(what) {}
    template <typename T>
    T get() {
        need(sizeof(T));
        std::uint64_t u = 0;
        for (std::size_t i = 0; i < sizeof(T); ++i) u |= std::uint64_t{p_[at_ + i]} << (8U * i);
        at_ += sizeof(T);
        return static_cast<T>(u);
    }
    std::span<const std::uint8_t> bytes(std::uint64_t n) {
        need(n);
        const auto s = p_.subspan(at_, static_cast<std::size_t>(n));
        at_ += static_cast<std::size_t>(n);
        return s;
    }
    std::vector<float> floats(std::uint64_t n) {
        if (n > (std::uint64_t{1} << 40U)) fail("float count");
        const auto b = bytes(n * 4U);
        std::vector<float> v(static_cast<std::size_t>(n));
        std::memcpy(v.data(), b.data(), b.size());
        return v;
    }
    std::vector<std::uint32_t> u32s(std::uint64_t n) {
        need(n * 4U);
        std::vector<std::uint32_t> v(static_cast<std::size_t>(n));
        for (auto& x : v) x = get<std::uint32_t>();
        return v;
    }
    std::string text(std::size_t max_bytes = 1U << 20U) {
        const auto n = get<std::uint32_t>();
        if (n > max_bytes) fail("text length");
        const auto b = bytes(n);
        return {b.begin(), b.end()};
    }
    void end() const {
        if (at_ != p_.size()) throw WireError(std::string(what_) + ": " + std::to_string(p_.size() - at_) + " trailing bytes");
    }
    [[noreturn]] void fail(const std::string& why) const { throw WireError(std::string(what_) + ": bad " + why); }

private:
    void need(std::uint64_t n) const {
        if (n > p_.size() - at_) throw WireError(std::string(what_) + ": truncated (" + std::to_string(n) + " bytes wanted at " + std::to_string(at_) + ")");
    }
    std::span<const std::uint8_t> p_;
    std::size_t at_ = 0;
    const char* what_;
};

std::uint32_t header_check(std::span<const std::uint8_t> first28) {
    std::uint32_t h = 2166136261U;
    for (const auto b : first28) h = (h ^ b) * 16777619U;
    return h;
}

#ifndef _WIN32
[[noreturn]] void sys_fail(const std::string& what) { throw WireError(what + ": " + std::strerror(errno)); }
#endif

}  // namespace

const char* to_string(MsgType type) noexcept {
    switch (type) {
    case MsgType::hello: return "HELLO";
    case MsgType::hello_ack: return "HELLO_ACK";
    case MsgType::step: return "STEP";
    case MsgType::result: return "RESULT";
    case MsgType::commit: return "COMMIT";
    case MsgType::reset: return "RESET";
    case MsgType::export_state: return "EXPORT_STATE";
    case MsgType::state: return "STATE";
    case MsgType::import_state: return "IMPORT_STATE";
    case MsgType::ack: return "ACK";
    case MsgType::ping: return "PING";
    case MsgType::pong: return "PONG";
    case MsgType::close: return "CLOSE";
    case MsgType::error: return "ERROR";
    case MsgType::spec_begin: return "SPEC_BEGIN";
    case MsgType::spec_accept: return "SPEC_ACCEPT";
    case MsgType::spec_commit: return "SPEC_COMMIT";
    }
    return "UNKNOWN";
}

const char* to_string(SectionKind kind) noexcept {
    switch (kind) {
    case SectionKind::mix_ffn: return "mix_ffn";
    case SectionKind::kv_entries: return "kv";
    case SectionKind::topk: return "topk";
    case SectionKind::candidates: return "cand";
    }
    return "unknown";
}

std::array<std::uint8_t, kHeaderBytes> encode_header(const Header& h) {
    Writer w;
    w.put(h.magic);
    w.put(h.version);
    w.put(static_cast<std::uint16_t>(h.type));
    w.put(h.request);
    w.put(h.bytes);
    w.put(h.flags);
    auto v = w.take();
    const auto check = header_check(v);
    for (int i = 0; i < 4; ++i) v.push_back(static_cast<std::uint8_t>((check >> (8U * static_cast<unsigned>(i))) & 0xffU));
    std::array<std::uint8_t, kHeaderBytes> out{};
    std::copy(v.begin(), v.end(), out.begin());
    return out;
}

Header decode_header(std::span<const std::uint8_t> bytes, std::uint64_t max_bytes) {
    if (bytes.size() != kHeaderBytes) throw WireError("header: expected 32 bytes");
    Reader r(bytes, "header");
    Header h;
    h.magic = r.get<std::uint32_t>();
    if (h.magic != kMagic) throw WireError("header: bad magic (not strix/ds41-pipeline)");
    h.version = r.get<std::uint16_t>();
    if (h.version != kVersion) throw WireError("header: protocol version " + std::to_string(h.version) + ", this engine speaks " + std::to_string(kVersion));
    const auto type = r.get<std::uint16_t>();
    if (type < 1U || type > 17U) throw WireError("header: unknown message type " + std::to_string(type));
    h.type = static_cast<MsgType>(type);
    h.request = r.get<std::uint64_t>();
    h.bytes = r.get<std::uint64_t>();
    h.flags = r.get<std::uint32_t>();
    if (r.get<std::uint32_t>() != header_check(bytes.first(28))) throw WireError("header: check mismatch");
    if (h.bytes > max_bytes) throw WireError("header: payload of " + std::to_string(h.bytes) + " bytes above the bound");
    return h;
}

BoundaryGeometry boundary_geometry(const deepseek41::Config& c) {
    BoundaryGeometry g;
    g.layers = c.num_hidden_layers;
    g.hidden = c.hidden_size;
    g.hc = c.hc_mult;
    g.vocab = c.vocab_size;
    g.topk = c.index_topk;
    g.candidate_block = c.candidate_block_size;
    g.kv_entry_bytes = c.head_dim / 2U + c.head_dim / 16U;
    g.idx_entry_bytes = c.index_head_dim / 2U + c.index_head_dim / 32U;
    return g;
}

BoundaryPlan plan_boundary(const std::vector<deepseek41::LayerAttributes>& layers, std::uint32_t cut) {
    if (cut == 0U || cut >= layers.size()) throw std::invalid_argument("pipeline cut must be in [1, layers)");
    BoundaryPlan p;
    p.cut = cut;
    const auto add = [&](std::vector<SourceLayer>& v, std::int32_t s) {
        const auto u = static_cast<std::uint32_t>(s);
        if (std::none_of(v.begin(), v.end(), [&](const SourceLayer& x) { return x.layer == u; }))
            v.push_back({u, layers[u].compress_ratio});
    };
    std::int32_t candidate_source = -1;
    for (std::size_t l = 0; l < layers.size(); ++l)
        if (layers[l].candidate_source) candidate_source = static_cast<std::int32_t>(l);
    for (std::size_t l = cut; l < layers.size(); ++l) {
        const auto& a = layers[l];
        if (a.compress_ratio == 0U) continue;
        if (a.kv_source_layer >= 0 && static_cast<std::uint32_t>(a.kv_source_layer) < cut) add(p.kv_sources, a.kv_source_layer);
        if (a.index_source_layer >= 0 && static_cast<std::uint32_t>(a.index_source_layer) < cut) add(p.topk_sources, a.index_source_layer);
        if (a.uses_candidates && a.index_source && candidate_source >= 0 && static_cast<std::uint32_t>(candidate_source) < cut) {
            p.candidate_source = candidate_source;
            const auto& src = layers[static_cast<std::size_t>(candidate_source)];
            p.candidate_ratio = layers[static_cast<std::size_t>(src.kv_source_layer)].compress_ratio;
        }
    }
    const auto by_layer = [](const SourceLayer& a, const SourceLayer& b) { return a.layer < b.layer; };
    std::sort(p.kv_sources.begin(), p.kv_sources.end(), by_layer);
    std::sort(p.topk_sources.begin(), p.topk_sources.end(), by_layer);
    return p;
}

StepBytes step_bytes(const BoundaryPlan& plan, const BoundaryGeometry& g, std::uint32_t position, std::uint32_t rows) {
    StepBytes b;
    b.tokens = std::uint64_t{rows} * 4U;
    b.stream = g.stream_bytes_per_row() * rows;
    b.mix = std::uint64_t{rows} * g.mix_floats() * 4U;
    constexpr std::uint64_t section_header = 16U;
    b.framing = kHeaderBytes + 24U + section_header;  // step fields + the mix section
    for (const auto& s : plan.kv_sources) {
        const std::uint64_t n = (position + rows) / s.ratio - position / s.ratio;
        b.kv += n * (g.kv_entry_bytes + g.idx_entry_bytes);
        b.framing += section_header;
    }
    for (std::size_t i = 0; i < plan.topk_sources.size(); ++i) {
        b.topk_max += std::uint64_t{rows} * (1U + g.topk) * 4U;
        b.framing += section_header;
    }
    if (plan.candidate_source >= 0) {
        const std::uint64_t count = (position + rows) / plan.candidate_ratio;
        b.candidates = std::uint64_t{rows} * ((count + g.candidate_block - 1U) / g.candidate_block);
        b.framing += section_header;
    }
    return b;
}

const Section* Boundary::find(SectionKind kind, std::uint32_t layer) const noexcept {
    for (const auto& s : sections)
        if (s.kind == kind && s.layer == layer) return &s;
    return nullptr;
}

void validate_boundary(const Boundary& b, const BoundaryPlan& plan, const BoundaryGeometry& g) {
    const auto fail = [](const std::string& why) { throw WireError("pipeline boundary: " + why); };
    if (b.rows == 0U) fail("no rows");
    if (b.tokens.size() != b.rows) fail("token count != rows");
    for (const auto t : b.tokens)
        if (t >= g.vocab) fail("token outside the vocabulary");
    if (b.stream_floats() != std::size_t{b.rows} * g.hc * g.hidden) fail("stream size");
    std::size_t expected = 1U + plan.kv_sources.size() + plan.topk_sources.size() + (plan.candidate_source >= 0 ? 1U : 0U);
    if (b.sections.size() != expected) fail(std::to_string(b.sections.size()) + " sections, the cut carries " + std::to_string(expected));
    const auto* mix = b.find(SectionKind::mix_ffn, plan.cut - 1U);
    if (mix == nullptr || mix->count != b.rows || mix->bytes.size() != std::size_t{b.rows} * g.mix_floats() * 4U)
        fail("mix_ffn@" + std::to_string(plan.cut - 1U) + " missing or of the wrong size");
    for (const auto& s : plan.kv_sources) {
        const auto* k = b.find(SectionKind::kv_entries, s.layer);
        const std::uint32_t first = b.position / s.ratio, end = (b.position + b.rows) / s.ratio;
        if (k == nullptr) fail("kv@" + std::to_string(s.layer) + " missing");
        if (k->first != first || k->count != end - first || k->bytes.size() != std::size_t{k->count} * (g.kv_entry_bytes + g.idx_entry_bytes))
            fail("kv@" + std::to_string(s.layer) + " entries [" + std::to_string(k->first) + ", +" + std::to_string(k->count) + ") instead of [" +
                 std::to_string(first) + ", " + std::to_string(end) + ")");
    }
    for (const auto& s : plan.topk_sources) {
        const auto* k = b.find(SectionKind::topk, s.layer);
        if (k == nullptr || k->count != b.rows) fail("topk@" + std::to_string(s.layer) + " missing or of the wrong row count");
        Reader r(k->bytes, "topk section");
        for (std::uint32_t row = 0; row < b.rows; ++row) {
            const auto n = r.get<std::uint32_t>();
            if (n > g.topk) fail("topk@" + std::to_string(s.layer) + " row with " + std::to_string(n) + " ids");
            (void)r.bytes(std::uint64_t{n} * 4U);
        }
        r.end();
    }
    if (plan.candidate_source >= 0) {
        const auto c = static_cast<std::uint32_t>(plan.candidate_source);
        const auto* k = b.find(SectionKind::candidates, c);
        const std::uint64_t count = (b.position + b.rows) / plan.candidate_ratio;
        const auto stride = static_cast<std::uint32_t>((count + g.candidate_block - 1U) / g.candidate_block);
        if (k == nullptr || k->count != b.rows || k->first != stride || k->bytes.size() != std::size_t{b.rows} * stride)
            fail("cand@" + std::to_string(c) + " missing or of the wrong size");
    }
}

// ---- messages ----

std::vector<std::uint8_t> encode(const Hello& m) {
    Writer w;
    w.bytes(m.token);
    w.put(m.cut);
    w.put(m.layers);
    w.put(m.hidden);
    w.put(m.hc);
    w.put(m.vocab);
    w.put(m.max_positions);
    w.put(m.model_identity);
    w.text(m.backend);
    w.text(m.numeric_identity);
    return w.take();
}

Hello decode_hello(std::span<const std::uint8_t> p) {
    Reader r(p, "HELLO");
    Hello m;
    const auto t = r.bytes(32);
    std::copy(t.begin(), t.end(), m.token.begin());
    m.cut = r.get<std::uint32_t>();
    m.layers = r.get<std::uint32_t>();
    m.hidden = r.get<std::uint32_t>();
    m.hc = r.get<std::uint32_t>();
    m.vocab = r.get<std::uint32_t>();
    m.max_positions = r.get<std::uint32_t>();
    m.model_identity = r.get<std::uint64_t>();
    m.backend = r.text(256);
    m.numeric_identity = r.text(1024);
    r.end();
    return m;
}

std::vector<std::uint8_t> encode(const HelloAck& m) {
    Writer w;
    w.put(m.status);
    w.put(m.first_layer);
    w.put(m.end_layer);
    w.put(m.world);
    w.put(m.max_positions);
    w.put(m.model_identity);
    w.text(m.engine);
    w.text(m.numeric_identity);
    w.text(m.message);
    return w.take();
}

HelloAck decode_hello_ack(std::span<const std::uint8_t> p) {
    Reader r(p, "HELLO_ACK");
    HelloAck m;
    m.status = r.get<std::uint32_t>();
    m.first_layer = r.get<std::uint32_t>();
    m.end_layer = r.get<std::uint32_t>();
    m.world = r.get<std::uint32_t>();
    m.max_positions = r.get<std::uint32_t>();
    m.model_identity = r.get<std::uint64_t>();
    m.engine = r.text(256);
    m.numeric_identity = r.text(1024);
    m.message = r.text(4096);
    r.end();
    return m;
}

std::vector<std::uint8_t> encode(const StepMsg& m) {
    const auto& b = m.boundary;
    Writer w;
    w.put(m.flags);
    w.put(b.position);
    w.put(b.rows);
    w.put(static_cast<std::uint32_t>(b.sections.size()));
    w.put(static_cast<std::uint64_t>(b.stream.size()));
    w.u32s(b.tokens);
    w.floats(b.stream);
    for (const auto& s : b.sections) {
        w.put(static_cast<std::uint16_t>(s.kind));
        w.put(static_cast<std::uint16_t>(s.layer));
        w.put(s.first);
        w.put(s.count);
        w.put(static_cast<std::uint32_t>(s.bytes.size()));
        w.bytes(s.bytes);
    }
    return w.take();
}

StepMsg decode_step(std::span<const std::uint8_t> p, const BoundaryGeometry& g, std::uint32_t max_rows, bool view_stream) {
    Reader r(p, "STEP");
    StepMsg m;
    auto& b = m.boundary;
    m.flags = r.get<std::uint32_t>();
    b.position = r.get<std::uint32_t>();
    b.rows = r.get<std::uint32_t>();
    if (b.rows == 0U || b.rows > max_rows) r.fail("row count " + std::to_string(b.rows));
    const auto sections = r.get<std::uint32_t>();
    if (sections > 4U * g.layers + 4U) r.fail("section count");
    const auto stream = r.get<std::uint64_t>();
    if (stream != std::uint64_t{b.rows} * g.hc * g.hidden) r.fail("stream size");
    b.tokens = r.u32s(b.rows);
    if (view_stream) {
        const auto bytes = r.bytes(stream * 4U);
        b.stream_view = reinterpret_cast<const float*>(bytes.data());  // the payload's FP32 bytes (little-endian host)
        b.stream_view_floats = static_cast<std::size_t>(stream);
    } else {
        b.stream = r.floats(stream);
    }
    for (std::uint32_t i = 0; i < sections; ++i) {
        Section s;
        const auto kind = r.get<std::uint16_t>();
        if (kind < 1U || kind > 4U) r.fail("section kind " + std::to_string(kind));
        s.kind = static_cast<SectionKind>(kind);
        s.layer = r.get<std::uint16_t>();
        if (s.layer >= g.layers) r.fail("section layer");
        s.first = r.get<std::uint32_t>();
        s.count = r.get<std::uint32_t>();
        const auto n = r.get<std::uint32_t>();
        const auto bytes = r.bytes(n);
        s.bytes.assign(bytes.begin(), bytes.end());
        b.sections.push_back(std::move(s));
    }
    r.end();
    return m;
}

std::vector<std::uint8_t> encode(const Result& m) {
    Writer w;
    w.put(m.position);
    w.put(static_cast<std::uint32_t>(m.tokens.size()));
    w.put(static_cast<std::uint32_t>(m.logits.size()));
    w.u32s(m.tokens);
    for (const auto h : m.logits_fnv) w.put(h);
    w.floats(m.logits);
    return w.take();
}

Result decode_result(std::span<const std::uint8_t> p, std::uint32_t vocab, std::uint32_t max_rows) {
    Reader r(p, "RESULT");
    Result m;
    m.position = r.get<std::uint32_t>();
    const auto rows = r.get<std::uint32_t>();
    if (rows == 0U || rows > max_rows) r.fail("row count");
    const auto logits = r.get<std::uint32_t>();
    if (logits != 0U && logits != vocab) r.fail("logits count");
    m.tokens = r.u32s(rows);
    for (std::uint32_t i = 0; i < rows; ++i) m.logits_fnv.push_back(r.get<std::uint64_t>());
    m.logits = r.floats(logits);
    r.end();
    for (const auto t : m.tokens)
        if (t >= vocab) r.fail("token");
    return m;
}

std::vector<std::uint8_t> encode(const StateMsg& m) {
    Writer w;
    w.text(m.manifest);
    w.put(static_cast<std::uint64_t>(m.state.size()));
    w.bytes(m.state);
    return w.take();
}

StateMsg decode_state(std::span<const std::uint8_t> p) {
    Reader r(p, "STATE");
    StateMsg m;
    m.manifest = r.text(1U << 20U);
    const auto n = r.get<std::uint64_t>();
    const auto b = r.bytes(n);
    m.state.assign(b.begin(), b.end());
    r.end();
    return m;
}

std::vector<std::uint8_t> encode_u32(std::uint32_t v) {
    Writer w;
    w.put(v);
    return w.take();
}
std::uint32_t decode_u32(std::span<const std::uint8_t> p) {
    Reader r(p, "u32 payload");
    const auto v = r.get<std::uint32_t>();
    r.end();
    return v;
}
std::vector<std::uint8_t> encode_text(const std::string& s) {
    Writer w;
    w.text(s);
    return w.take();
}
std::string decode_text(std::span<const std::uint8_t> p, std::size_t max_bytes) {
    Reader r(p, "text payload");
    auto s = r.text(max_bytes);
    r.end();
    return s;
}

std::array<std::uint8_t, 32> read_token_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::invalid_argument("cannot read the session token file " + path.string());
    std::string hex;
    for (char c; in.get(c);)
        if (!std::isspace(static_cast<unsigned char>(c))) hex.push_back(c);
    if (hex.size() != 64U) throw std::invalid_argument("session token file: expected 64 hex characters (32 bytes)");
    std::array<std::uint8_t, 32> t{};
    const auto nibble = [&](char c) -> std::uint8_t {
        if (c >= '0' && c <= '9') return static_cast<std::uint8_t>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<std::uint8_t>(c - 'a' + 10);
        if (c >= 'A' && c <= 'F') return static_cast<std::uint8_t>(c - 'A' + 10);
        throw std::invalid_argument("session token file: not hexadecimal");
    };
    for (std::size_t i = 0; i < 32U; ++i) t[i] = static_cast<std::uint8_t>(nibble(hex[2U * i]) << 4U | nibble(hex[2U * i + 1U]));
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

#ifndef _WIN32
// ---- TCP ----

std::pair<std::string, std::uint16_t> split_host_port(const std::string& s) {
    const auto colon = s.rfind(':');
    if (colon == std::string::npos || colon == 0U || colon + 1U == s.size()) throw std::invalid_argument("expected HOST:PORT, got '" + s + "'");
    const auto port = std::stoul(s.substr(colon + 1U));
    if (port > 65535U) throw std::invalid_argument("port out of range in '" + s + "'");
    return {s.substr(0, colon), static_cast<std::uint16_t>(port)};
}

namespace {
void set_timeouts(int fd, std::chrono::milliseconds timeout) {
    const int one = 1;
    if (::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0) sys_fail("TCP_NODELAY");
    if (timeout.count() > 0) {
        timeval tv{};
        tv.tv_sec = static_cast<time_t>(timeout.count() / 1000);
        tv.tv_usec = static_cast<suseconds_t>((timeout.count() % 1000) * 1000);
        if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0 || ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0)
            sys_fail("socket timeouts");
    }
}
std::string peer_name(int fd) {
    sockaddr_storage a{};
    socklen_t n = sizeof(a);
    if (::getpeername(fd, reinterpret_cast<sockaddr*>(&a), &n) != 0) return "?";
    char host[NI_MAXHOST] = {}, port[NI_MAXSERV] = {};
    if (::getnameinfo(reinterpret_cast<sockaddr*>(&a), n, host, sizeof(host), port, sizeof(port), NI_NUMERICHOST | NI_NUMERICSERV) != 0) return "?";
    return std::string(host) + ":" + port;
}
}  // namespace

Connection::Connection(int fd, std::chrono::milliseconds timeout) : fd_(fd), timeout_(timeout) {
    set_timeouts(fd_, timeout);
    peer_ = peer_name(fd_);
}

Connection::~Connection() {
    if (fd_ >= 0) ::close(fd_);
}

void Connection::send_truncated(MsgType type, std::uint64_t request, std::span<const std::uint8_t> payload, std::size_t keep, std::uint32_t flags) {
    Header h;
    h.type = type;
    h.request = request;
    h.bytes = payload.size();
    h.flags = flags;
    const auto head = encode_header(h);
    std::vector<std::uint8_t> out(head.begin(), head.end());
    out.insert(out.end(), payload.begin(), payload.begin() + static_cast<std::ptrdiff_t>(std::min(keep, payload.size())));
    for (std::size_t done = 0; done < out.size();) {
        const auto n = ::send(fd_, out.data() + done, out.size() - done, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            sys_fail("pipeline send");
        }
        done += static_cast<std::size_t>(n);
    }
    sent_ += out.size();
}

void Connection::send(MsgType type, std::uint64_t request, std::span<const std::uint8_t> payload, std::uint32_t flags) {
    Header h;
    h.type = type;
    h.request = request;
    h.bytes = payload.size();
    h.flags = flags;
    const auto head = encode_header(h);
    iovec v[2] = {{const_cast<std::uint8_t*>(head.data()), head.size()}, {const_cast<std::uint8_t*>(payload.data()), payload.size()}};
    std::size_t total = head.size() + payload.size(), done = 0;
    int first = 0;
    while (done < total) {
        msghdr msg{};
        msg.msg_iov = v + first;
        msg.msg_iovlen = static_cast<decltype(msg.msg_iovlen)>(2 - first);
        const auto n = ::sendmsg(fd_, &msg, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            sys_fail("pipeline send");
        }
        done += static_cast<std::size_t>(n);
        auto left = static_cast<std::size_t>(n);
        while (left > 0U && first < 2) {
            const auto take = std::min(left, v[first].iov_len);
            v[first].iov_base = static_cast<std::uint8_t*>(v[first].iov_base) + take;
            v[first].iov_len -= take;
            left -= take;
            if (v[first].iov_len == 0U) ++first;
        }
        if (first < 2 && v[first].iov_len == 0U) ++first;
    }
    sent_ += total;
}

void Connection::set_busy_poll(bool on) {
    busy_ = on;
#ifdef SO_BUSY_POLL
    if (on) {
        const int us = 50;  // best effort (may need CAP_NET_ADMIN above net.core.busy_read); the spin below works without it
        (void)::setsockopt(fd_, SOL_SOCKET, SO_BUSY_POLL, &us, sizeof(us));
    }
#endif
}

void Connection::read_all(std::uint8_t* p, std::size_t n, std::chrono::steady_clock::time_point* first_byte) {
    std::size_t got = 0;
    auto deadline = std::chrono::steady_clock::time_point::max();
    if (busy_ && timeout_.count() > 0) deadline = std::chrono::steady_clock::now() + timeout_;
    std::uint32_t spins = 0;
    while (got < n) {
        const auto r = ::recv(fd_, p + got, n - got, busy_ ? MSG_DONTWAIT : 0);
        if (r == 0) throw WireError("pipeline peer closed the connection");
        if (r < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                if (!busy_) throw WireError("pipeline receive timed out");
                if ((++spins & 0x3ffU) == 0U && std::chrono::steady_clock::now() > deadline) throw WireError("pipeline receive timed out");
                continue;  // spin: the wake-up of a blocking read is what busy polling saves
            }
            sys_fail("pipeline receive");
        }
        if (got == 0U && first_byte != nullptr) *first_byte = std::chrono::steady_clock::now();
        got += static_cast<std::size_t>(r);
        if (busy_ && timeout_.count() > 0) deadline = std::chrono::steady_clock::now() + timeout_;
    }
}

Header Connection::receive_header(std::uint64_t max_bytes, std::chrono::steady_clock::time_point* first_byte) {
    std::array<std::uint8_t, kHeaderBytes> head{};
    read_all(head.data(), head.size(), first_byte);
    const auto h = decode_header(head, max_bytes);
    received_ += kHeaderBytes;
    return h;
}

void Connection::receive_payload(std::span<std::uint8_t> out) {
    read_all(out.data(), out.size(), nullptr);
    received_ += out.size();
}

Message Connection::receive(std::uint64_t max_bytes) {
    Message m;
    m.header = receive_header(max_bytes);
    m.payload.resize(static_cast<std::size_t>(m.header.bytes));
    receive_payload(m.payload);
    return m;
}

Listener::Listener(const std::string& host, std::uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;
    addrinfo* res = nullptr;
    const auto service = std::to_string(port);
    if (const int e = ::getaddrinfo(host.empty() ? nullptr : host.c_str(), service.c_str(), &hints, &res); e != 0)
        throw WireError("pipeline listen " + host + ": " + ::gai_strerror(e));
    std::unique_ptr<addrinfo, decltype(&::freeaddrinfo)> guard(res, &::freeaddrinfo);
    fd_ = ::socket(res->ai_family, res->ai_socktype | SOCK_CLOEXEC, res->ai_protocol);
    if (fd_ < 0) sys_fail("socket");
    const int one = 1;
    (void)::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (::bind(fd_, res->ai_addr, res->ai_addrlen) != 0) sys_fail("pipeline bind " + host + ":" + service);
    if (::listen(fd_, 4) != 0) sys_fail("listen");
    sockaddr_storage a{};
    socklen_t n = sizeof(a);
    if (::getsockname(fd_, reinterpret_cast<sockaddr*>(&a), &n) != 0) sys_fail("getsockname");
    port_ = ntohs(a.ss_family == AF_INET6 ? reinterpret_cast<sockaddr_in6*>(&a)->sin6_port : reinterpret_cast<sockaddr_in*>(&a)->sin_port);
}

Listener::~Listener() {
    if (fd_ >= 0) ::close(fd_);
}

std::unique_ptr<Connection> Listener::accept(std::chrono::milliseconds wait, std::chrono::milliseconds io_timeout) {
    pollfd p{fd_, POLLIN, 0};
    for (;;) {
        const int r = ::poll(&p, 1, wait.count() > 0 ? static_cast<int>(wait.count()) : -1);
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) sys_fail("poll");
        if (r == 0) return nullptr;
        break;
    }
    const int fd = ::accept4(fd_, nullptr, nullptr, SOCK_CLOEXEC);
    if (fd < 0) sys_fail("accept");
    return std::make_unique<Connection>(fd, io_timeout);
}

std::unique_ptr<Connection> connect(const std::string& host_port, std::chrono::milliseconds wait, std::chrono::milliseconds io_timeout) {
    const auto [host, port] = split_host_port(host_port);
    const auto deadline = std::chrono::steady_clock::now() + wait;
    std::string last;
    for (;;) {
        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_NUMERICSERV;
        addrinfo* res = nullptr;
        if (const int e = ::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res); e != 0) {
            last = ::gai_strerror(e);
        } else {
            std::unique_ptr<addrinfo, decltype(&::freeaddrinfo)> guard(res, &::freeaddrinfo);
            for (auto* a = res; a != nullptr; a = a->ai_next) {
                const int fd = ::socket(a->ai_family, a->ai_socktype | SOCK_CLOEXEC, a->ai_protocol);
                if (fd < 0) continue;
                if (::connect(fd, a->ai_addr, a->ai_addrlen) == 0) {
                    // a loopback connect to a port of the ephemeral range with no listener can connect the socket to
                    // itself (TCP simultaneous open): treat it as refused and retry
                    sockaddr_storage local{}, remote{};
                    socklen_t ln = sizeof(local), rn = sizeof(remote);
                    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&local), &ln) == 0 && ::getpeername(fd, reinterpret_cast<sockaddr*>(&remote), &rn) == 0 &&
                        ln == rn && std::memcmp(&local, &remote, ln) == 0) {
                        last = "connected to itself (no listener yet)";
                        ::close(fd);
                        continue;
                    }
                    return std::make_unique<Connection>(fd, io_timeout);
                }
                last = std::strerror(errno);
                ::close(fd);
            }
        }
        if (std::chrono::steady_clock::now() >= deadline) throw WireError("pipeline connect " + host_port + ": " + last);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

#endif  // _WIN32

}  // namespace strix::ds41::pipe
