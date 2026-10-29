#include "strix/model_input.hpp"
#include "strix/ds41_pipeline_dspark.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

namespace strix::ds41::pipe {
namespace {

constexpr std::uint32_t kSpecVersion = 1;

struct Writer {
    std::vector<std::uint8_t> b;
    void u32(std::uint32_t v) {
        for (int i = 0; i < 4; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
    }
    void u64(std::uint64_t v) {
        for (int i = 0; i < 8; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
    }
    void f32(float v) {
        std::uint32_t u;
        std::memcpy(&u, &v, 4);
        u32(u);
    }
    void list(const std::vector<std::uint32_t>& v) {
        u32(static_cast<std::uint32_t>(v.size()));
        for (const auto x : v) u32(x);
    }
};

struct Reader {
    std::span<const std::uint8_t> p;
    std::size_t at = 0;
    void need(std::size_t n) const {
        if (at + n > p.size()) throw WireError("spec message: truncated");
    }
    std::uint32_t u32() {
        need(4);
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= std::uint32_t{p[at + i]} << (8 * i);
        at += 4;
        return v;
    }
    std::uint64_t u64() {
        need(8);
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= std::uint64_t{p[at + i]} << (8 * i);
        at += 8;
        return v;
    }
    float f32() {
        const std::uint32_t u = u32();
        float v;
        std::memcpy(&v, &u, 4);
        return v;
    }
    std::vector<std::uint32_t> list(std::uint32_t max) {
        const std::uint32_t n = u32();
        if (n > max) throw WireError("spec message: list of " + std::to_string(n) + " entries (at most " + std::to_string(max) + ")");
        std::vector<std::uint32_t> v(n);
        for (auto& x : v) x = u32();
        return v;
    }
    void end() const {
        if (at != p.size()) throw WireError("spec message: trailing bytes");
    }
};

}  // namespace

std::vector<std::uint8_t> encode(const SpecBegin& m) {
    Writer w;
    w.u32(kSpecVersion);
    w.u32(m.epoch);
    w.u32(m.position);
    w.u32(m.next_token);
    w.u32(m.max_new);
    w.u32(m.flags);
    w.list(m.eos);
    return std::move(w.b);
}

SpecBegin decode_spec_begin(std::span<const std::uint8_t> p) {
    Reader r{p};
    if (r.u32() != kSpecVersion) throw WireError("SPEC_BEGIN: version");
    SpecBegin m;
    m.epoch = r.u32();
    m.position = r.u32();
    m.next_token = r.u32();
    m.max_new = r.u32();
    m.flags = r.u32();
    m.eos = r.list(64);
    r.end();
    return m;
}

std::vector<std::uint8_t> encode(const SpecAccept& m) {
    Writer w;
    w.u32(kSpecVersion);
    w.u32(m.epoch);
    w.u32(m.window);
    w.u32(static_cast<std::uint32_t>(m.status));
    w.u32(m.position);
    w.u32(m.rows);
    w.u32(m.kept);
    w.list(m.argmax);
    w.u32(static_cast<std::uint32_t>(m.row_fnv.size()));
    for (const auto f : m.row_fnv) w.u64(f);
    w.u32(m.committed);
    w.u32(m.next_token);
    w.list(m.published);
    w.u32(static_cast<std::uint32_t>(m.done));
    w.list(m.plan);
    w.f32(m.draft_ms);
    w.f32(m.control_ms);
    return std::move(w.b);
}

SpecAccept decode_spec_accept(std::span<const std::uint8_t> p, std::uint32_t max_rows) {
    Reader r{p};
    if (r.u32() != kSpecVersion) throw WireError("SPEC_ACCEPT: version");
    SpecAccept m;
    m.epoch = r.u32();
    m.window = r.u32();
    const std::uint32_t st = r.u32();
    if (st > 3U) throw WireError("SPEC_ACCEPT: status");
    m.status = static_cast<SpecStatus>(st);
    m.position = r.u32();
    m.rows = r.u32();
    m.kept = r.u32();
    m.argmax = r.list(max_rows);
    const std::uint32_t nf = r.u32();
    if (nf > max_rows) throw WireError("SPEC_ACCEPT: row FNVs");
    m.row_fnv.resize(nf);
    for (auto& f : m.row_fnv) f = r.u64();
    m.committed = r.u32();
    m.next_token = r.u32();
    m.published = r.list(max_rows);
    const std::uint32_t d = r.u32();
    if (d > 2U) throw WireError("SPEC_ACCEPT: done");
    m.done = static_cast<SpecDone>(d);
    m.plan = r.list(max_rows);
    m.draft_ms = r.f32();
    m.control_ms = r.f32();
    r.end();
    if (m.status == SpecStatus::ok) {
        if (m.window != 0U && (m.rows == 0U || m.argmax.size() != m.rows || m.kept == 0U || m.kept > m.rows || m.published.size() > m.kept))
            throw WireError("SPEC_ACCEPT: inconsistent window");
        if (m.done == SpecDone::more && m.plan.empty()) throw WireError("SPEC_ACCEPT: no plan");
        if (!m.plan.empty() && m.plan[0] != m.next_token) throw WireError("SPEC_ACCEPT: the plan does not start at the next token");
    }
    return m;
}

std::vector<std::uint8_t> encode_spec_commit(std::uint32_t epoch, std::uint32_t window) {
    Writer w;
    w.u32(kSpecVersion);
    w.u32(epoch);
    w.u32(window);
    return std::move(w.b);
}

std::pair<std::uint32_t, std::uint32_t> decode_spec_commit(std::span<const std::uint8_t> p) {
    Reader r{p};
    if (r.u32() != kSpecVersion) throw WireError("SPEC_COMMIT: version");
    const std::uint32_t e = r.u32(), w = r.u32();
    r.end();
    return {e, w};
}

std::uint32_t accepted_prefix(std::span<const std::uint32_t> inputs, std::span<const std::uint32_t> argmax) {
    if (inputs.empty() || argmax.size() != inputs.size()) throw std::invalid_argument("accepted_prefix: one argmax per input row");
    std::uint32_t a = 0;
    while (a + 1U < inputs.size() && inputs[a + 1U] == argmax[a]) ++a;
    return a;
}

SpecOutcome spec_outcome(std::span<const std::uint32_t> inputs, std::span<const std::uint32_t> argmax, std::uint32_t remaining,
                         std::span<const std::uint32_t> eos) {
    if (remaining == 0U) throw std::invalid_argument("spec_outcome: nothing left to publish");
    SpecOutcome o;
    o.accepted = accepted_prefix(inputs, argmax);
    for (std::uint32_t j = 0; j <= o.accepted; ++j) {
        o.published.push_back(argmax[j]);
        if (std::find(eos.begin(), eos.end(), argmax[j]) != eos.end()) {
            o.done = SpecDone::eos;
            break;
        }
        if (o.published.size() == remaining) {
            o.done = SpecDone::max_new;
            break;
        }
    }
    o.kept = static_cast<std::uint32_t>(o.published.size());
    return o;
}

SpecDecisionLog::Kind SpecDecisionLog::classify(std::uint64_t request) const noexcept {
    if (!active_ || spec_epoch(request) != epoch_) return Kind::stale;
    const std::uint32_t w = spec_window(request);
    if (w == window_ + 1U && !finished_) return Kind::fresh;
    if (w == window_ && !last_.empty()) return Kind::duplicate;
    return Kind::stale;
}

SpecCostTable read_spec_cost_table(const std::string& path) {
    strix::ModelInput f(path);
    if (!f) throw std::invalid_argument("cannot read the cost table " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    SpecCostTable t;
    t.verify.assign(9, 0.0);
    // a small reader for {"V": {"R": ms, ..}, "D": ms, "A": ms, "label": ".."}: numbers after quoted keys
    const auto number_after = [&](std::size_t at, double& out) {
        at = s.find(':', at);
        if (at == std::string::npos) return false;
        ++at;
        while (at < s.size() && (s[at] == ' ' || s[at] == '\t' || s[at] == '\n' || s[at] == '\r')) ++at;
        std::size_t used = 0;
        try {
            out = std::stod(s.substr(at), &used);
        } catch (const std::exception&) {
            return false;
        }
        return used != 0U;
    };
    const auto vpos = s.find("\"V\"");
    if (vpos == std::string::npos) throw std::invalid_argument("cost table without \"V\": " + path);
    const auto open = s.find('{', vpos), close = s.find('}', vpos);
    for (std::uint32_t r = 1; r <= 8U; ++r) {
        const auto k = s.find("\"" + std::to_string(r) + "\"", open);
        if (k == std::string::npos || k > close) continue;
        double ms = 0.0;
        if (number_after(k, ms)) t.verify[r] = ms * 1e-3;
    }
    for (const auto& [key, dst] : {std::pair<const char*, double*>{"\"D\"", &t.draft}, {"\"A\"", &t.extra}}) {
        const auto k = s.find(key, close);
        double ms = 0.0;
        if (k != std::string::npos && number_after(k, ms)) *dst = ms * 1e-3;
    }
    const auto lk = s.find("\"label\"");
    if (lk != std::string::npos) {
        const auto q0 = s.find('"', s.find(':', lk) + 1), q1 = s.find('"', q0 + 1);
        if (q0 != std::string::npos && q1 != std::string::npos) t.label = s.substr(q0 + 1, q1 - q0 - 1);
    }
    if (t.verify[1] == 0.0) throw std::invalid_argument("cost table without V[1]: " + path);
    return t;
}

}  // namespace strix::ds41::pipe
