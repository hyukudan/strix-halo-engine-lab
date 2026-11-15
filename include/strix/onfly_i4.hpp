#pragma once
// TURBO+ onfly INT4 weights for Qwen (decode dense projections): the request, its tensor-group switches, the named
// identity that enters every cache key, and the tensor classifier. One resolver (resolve_onfly_i4) serves the CLI and
// the HTTP server (same binary, same RequestedOptions); the effective values are printed by --print-effective-config.
//
//   --turbo-weights native|onfly|converted   STRIX_TURBO_WEIGHTS         native (default): the packs as they are
//   --turbo-weight-bits 4                    STRIX_TURBO_WEIGHT_BITS     onfly / converted: 4 only so far
//   --turbo-onfly-groups LIST                STRIX_TURBO_ONFLY_GROUPS    gdn_in,gdn_out,qsa_io,shared_expert (default: all)
//   --turbo-onfly-scale-rule legacy|full     STRIX_TURBO_ONFLY_SCALE     default full (core/iu4.hpp fast full range)
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::onfly_i4 {

enum Group : unsigned { gdn_in = 1U, gdn_out = 2U, qsa_io = 4U, shared_expert = 8U, all_groups = 15U };
inline constexpr std::string_view kIdentity = "qwen-turbo-plus-onfly-i4g64-v1";
inline constexpr std::string_view kScopeNote = "decode dense projections (rows <= 8) i4g64; prefill (rows >= 16) q8_0";
inline constexpr int kScaleLegacy = 0, kScaleFullRange = 2001;

enum class Source : unsigned { native = 0, onfly = 1, converted = 2 };

struct Request {  // as typed (empty = not given)
    std::string weights, bits, groups, scale_rule;
};
struct Resolved {
    Source source = Source::native;
    unsigned bits = 4, groups = all_groups;
    int scale_rule = kScaleFullRange;
    [[nodiscard]] bool active() const { return source == Source::onfly; }
    // Identity for cache keys / receipts; empty for native (byte-identical to the unmodified engine).
    [[nodiscard]] std::string identity() const {
        if (!active()) return {};
        return std::string(kIdentity) + "-g" + std::to_string(groups) + (scale_rule == kScaleLegacy ? "-legacy" : "-full");
    }
};

inline std::string_view group_name(unsigned bit) {
    switch (bit) {
    case gdn_in: return "gdn_in";
    case gdn_out: return "gdn_out";
    case qsa_io: return "qsa_io";
    case shared_expert: return "shared_expert";
    default: return "?";
    }
}
inline unsigned parse_groups(std::string_view list) {
    if (list.empty() || list == "all") return all_groups;
    unsigned mask = 0;
    while (!list.empty()) {
        const auto comma = list.find(',');
        const auto item = list.substr(0, comma);
        unsigned bit = 0;
        for (unsigned b : {gdn_in, gdn_out, qsa_io, shared_expert}) if (group_name(b) == item) bit = b;
        if (!bit) throw std::invalid_argument("--turbo-onfly-groups: unknown group '" + std::string(item) + "'");
        mask |= bit;
        list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1U);
    }
    return mask;
}
inline std::string groups_text(unsigned mask) {
    std::string out;
    for (unsigned b : {gdn_in, gdn_out, qsa_io, shared_expert})
        if (mask & b) out += (out.empty() ? "" : ",") + std::string(group_name(b));
    return out.empty() ? "none" : out;
}

// CLI > environment > default; `env(name)` returns nullptr when unset. `converted` is refused (not implemented).
inline Resolved resolve(const Request& cli, const std::function<const char*(const char*)>& env) {
    const auto pick = [&](const std::string& typed, const char* name) {
        if (!typed.empty()) return typed;
        const char* v = env(name);
        return v ? std::string(v) : std::string{};
    };
    Resolved r;
    const auto weights = pick(cli.weights, "STRIX_TURBO_WEIGHTS");
    if (weights.empty() || weights == "native") r.source = Source::native;
    else if (weights == "onfly") r.source = Source::onfly;
    else if (weights == "converted")
        throw std::invalid_argument("--turbo-weights converted: not implemented for Qwen (no pre-converted low-bit pack format yet)");
    else throw std::invalid_argument("--turbo-weights: expected native, onfly or converted");
    const auto bits = pick(cli.bits, "STRIX_TURBO_WEIGHT_BITS");
    if (!bits.empty() && bits != "4")
        throw std::invalid_argument("--turbo-weight-bits: only 4 is implemented (3 and 2 later)");
    r.groups = parse_groups(pick(cli.groups, "STRIX_TURBO_ONFLY_GROUPS"));
    const auto rule = pick(cli.scale_rule, "STRIX_TURBO_ONFLY_SCALE");
    if (rule.empty() || rule == "full") r.scale_rule = kScaleFullRange;
    else if (rule == "legacy") r.scale_rule = kScaleLegacy;
    else throw std::invalid_argument("--turbo-onfly-scale-rule: expected legacy or full");
    return r;
}

// Group of a Q8_0 dense projection by tensor name (0 = not an onfly candidate).
inline unsigned classify(std::string_view name) {
    const auto ends = [&](std::string_view s) { return name.size() >= s.size() && name.substr(name.size() - s.size()) == s; };
    if (ends("attn_qkv.weight") || ends("attn_gate.weight")) return gdn_in;
    if (ends("ssm_out.weight")) return gdn_out;
    if (ends("attn_q.weight") || ends("attn_k.weight") || ends("attn_v.weight") || ends("attn_output.weight")) return qsa_io;
    if (ends("ffn_gate_shexp.weight") || ends("ffn_up_shexp.weight") || ends("ffn_down_shexp.weight")) return shared_expert;
    return 0;
}

}  // namespace strix::onfly_i4
