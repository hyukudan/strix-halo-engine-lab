#include "strix/ds41_pipeline_multi.hpp"

#include <cstring>
#include <string>

namespace strix::ds41::pipe {
namespace {

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}

// bounds-checked little-endian reads of one message
class In {
public:
    In(std::span<const std::uint8_t> p, const char* what) : p_(p), what_(what) {}
    std::uint32_t u32() {
        need(4);
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= std::uint32_t{p_[at_ + static_cast<std::size_t>(i)]} << (8 * i);
        at_ += 4U;
        return v;
    }
    std::span<const std::uint8_t> bytes(std::size_t n) {
        need(n);
        const auto s = p_.subspan(at_, n);
        at_ += n;
        return s;
    }
    std::span<const std::uint8_t> rest() { return bytes(p_.size() - at_); }
    void end() const {
        if (at_ != p_.size()) throw WireError(std::string(what_) + ": " + std::to_string(p_.size() - at_) + " trailing bytes");
    }

private:
    void need(std::size_t n) const {
        if (p_.size() - at_ < n) throw WireError(std::string(what_) + ": truncated");
    }
    std::span<const std::uint8_t> p_;
    const char* what_;
    std::size_t at_ = 0;
};

}  // namespace

std::vector<std::uint8_t> encode(const MultiHello& m) {
    const auto hello = encode(m.hello);
    std::vector<std::uint8_t> out;
    put32(out, static_cast<std::uint32_t>(hello.size()));
    out.insert(out.end(), hello.begin(), hello.end());
    put32(out, m.slots);
    put32(out, m.max_rows);
    return out;
}

MultiHello decode_multi_hello(std::span<const std::uint8_t> p) {
    In r(p, "MULTI_HELLO");
    MultiHello m;
    const auto n = r.u32();
    m.hello = decode_hello(r.bytes(n));
    m.slots = r.u32();
    m.max_rows = r.u32();
    r.end();
    if (m.slots == 0U || m.max_rows == 0U) throw WireError("MULTI_HELLO: no slots or no rows");
    return m;
}

std::vector<std::uint8_t> encode(const SeqStep& m) {
    std::vector<std::uint8_t> out;
    out.reserve(16U + 4U * m.tokens.size());
    put32(out, m.slot);
    put32(out, m.position);
    put32(out, static_cast<std::uint32_t>(m.tokens.size()));
    put32(out, m.commit);
    for (const auto t : m.tokens) put32(out, t);
    return out;
}

SeqStep decode_seq_step(std::span<const std::uint8_t> p, std::uint32_t max_rows) {
    In r(p, "SEQ_STEP");
    SeqStep m;
    m.slot = r.u32();
    m.position = r.u32();
    const auto rows = r.u32();
    m.commit = r.u32();
    if (rows == 0U || rows > max_rows) throw WireError("SEQ_STEP: " + std::to_string(rows) + " rows (1.." + std::to_string(max_rows) + ")");
    m.tokens.resize(rows);
    for (auto& t : m.tokens) t = r.u32();
    r.end();
    return m;
}

std::vector<std::uint8_t> encode_seq_import(std::uint32_t slot, const StateMsg& state) {
    std::vector<std::uint8_t> out;
    put32(out, slot);
    const auto body = encode(state);
    out.insert(out.end(), body.begin(), body.end());
    return out;
}

std::pair<std::uint32_t, StateMsg> decode_seq_import(std::span<const std::uint8_t> p) {
    In r(p, "SEQ_IMPORT");
    const auto slot = r.u32();
    return {slot, decode_state(r.rest())};
}

}  // namespace strix::ds41::pipe
