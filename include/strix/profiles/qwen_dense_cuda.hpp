#pragma once
// CUDA Qwen dense (qwen35, 27B) product profile: ONE tier word '--profile exact|default|fast|quality' resolved by the
// shared resolver (strix::profile::resolve, strix/engine_profile.hpp) for the decode and the speculative CLI.
// This header is data: the lever rows and the regimes in which each FAST cell is qualified (gate receipt). Choosing
// `fast` turns ON every lever qualified for the run's regime; an explicit per-lever option overrides the profile and
// is recorded as explicit; `exact` / `default` touches no option (today's bits, byte-identical effective config).
// Promoting a lever after its gate is ONE row in `qualified` below, never code.
// TURBO is a registry slot only (quality bar pending the owner's decision): this engine declares no turbo version, so
// `--profile turbo` is refused by the shared resolver. Host-only: no CUDA types (options are templated).
#include "strix/engine_profile.hpp"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace strix::qwen35::profile {
namespace sp = ::strix::profile;
using sp::Kind;
using sp::Lever;
using sp::Name;

// Regime of a run, known before the engine loads.
enum class Weights : std::uint8_t { exact_pack, nvfp4_native, nvfp4_rtn };  // --turbo-weights off | native | onfly/converted
enum class Route : std::uint8_t { large_prefill, windows };  // decode CLI large-M prefill | spec CLI forward_rows windows
struct Regime {
    Weights weights = Weights::exact_pack;
    Route route = Route::large_prefill;
    std::uint64_t prompt_tokens = 0;
};
inline const char* weights_name(Weights w) {
    return w == Weights::nvfp4_rtn ? "nvfp4-rtn (onfly/converted)" : w == Weights::nvfp4_native ? "nvfp4-native" : "exact-pack";
}
inline const char* route_name(Route r) { return r == Route::windows ? "windows (spec CLI)" : "large-prefill (decode CLI)"; }

// Columns: default, fast, quality, turbo. The fast column is admitted per regime from `qualified` (ready is computed);
// every row is runnable in this build. builtin = today's default; off = the explicit off switch.
inline constexpr Lever table[] = {
    {.name = "prefill_arith", .env = "", .option = "--prefill-arith", .value = {"", "fast4", "", ""}, .builtin = "exact",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "fast4 (nvfp4-tile-mma-f32-v1, identity -pf-fast4)", .off = "exact",
     .source = "W4A4 FAST4 large-M prefill; gate with chunk16 PASS 8K KL 0.01275 top-1 0.9648, 32K KL 0.00626 top-1 0.9668 (52dd7d1f0)"},
    {.name = "prefill_gdn", .env = "", .option = "--prefill-gdn", .value = {"", "chunk16", "", ""}, .builtin = "serial",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "chunk16 (gdn-chunk16-f32-v1, identity -gdnc16)", .off = "serial",
     .source = "chunked GDN prefill; gated together with fast4 (52dd7d1f0); ignored under prefill_arith exact"},
    {.name = "prefill_hid", .env = "", .option = "--prefill-hid", .value = {"", "bf16", "", ""}, .builtin = "f32",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "bf16 (nvfp4-tile-h16-v1: BF16 FFN hidden rows, identity -h16)", .off = "f32",
     .source = "FAST4 glu epilogue stores BF16 H, down operand quantised from it (half the H bytes); no gate receipt yet"},
    {.name = "prefill_attn8", .env = "", .option = "--prefill-attn8", .value = {"", "on", "", ""}, .builtin = "off",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "on (e4m3 flash over an e4m3 KV shadow, identity -a8)", .off = "off",
     .source = "with fast4+chunk16: 32K 2/2 windows PASS KL 0.00589 top-1 0.9746-0.9805, warm 12,354 vs 10,946 tok/s; no 8K gate"},
    {.name = "split_k", .env = "", .option = "--split-k", .value = {"", "fast", "", ""}, .builtin = "exact",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "fast (nvfp4-splitk-fast-v1, identity -skf1)", .off = "exact",
     .source = "NVFP4 GEMV split-K FAST targets (strix/split_k_policy.hpp, 188-SM cc 12.0 rows); gate EXACT -> FAST pooled "
               "KL 0.00000 top-1 0.9990 dNLL -0.00000; spec verify -0.08 .. -0.12 ms/step (2 ABAB pairs)"},
    {.name = "decode_attn8", .env = "", .option = "--decode-attn8", .value = {"", "on", "", ""}, .builtin = "off",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "on (e4m3 decode / verify attention over the e4m3 K/V shadow, tensor-core S and PV, identity -d8)", .off = "off",
     .source = "cuda_core attention_decode8 (attention_flash8.cu): split-key items (2048-key chunk, kv head, 128 row x head "
               "pairs) read each K/V tile once per kv head; verify rows == one-row calls bitwise; gate 8K / 28K / 64K pending"},
};

// Levers that act on the windowed forward_rows path (the spec CLI prefills through it: cuda_spec.hpp prefill). The
// prefill_* rows act only on the large-M prefill and are not offered there.
// decode_attn8 acts on every forward_rows call (decode, verify and the spec CLI's prefill windows): offered on both routes.
inline constexpr std::string_view kWindowsLevers[] = {"split_k", "decode_attn8"};

// Where a FAST cell is qualified: one row per (lever, weights, route, minimum prompt tokens) with its receipt.
struct Qualified {
    std::string_view lever;
    Weights weights;
    Route route;
    std::uint64_t min_prompt_tokens;
    std::string_view receipt;
};
inline constexpr Qualified qualified[] = {
    {"prefill_arith", Weights::nvfp4_rtn, Route::large_prefill, 0,
     "52dd7d1f0: FAST4 + chunk16 gate 8K / 32K PASS (converted NVFP4 pack); 8K 13,437-13,714 tok/s"},
    {"prefill_gdn", Weights::nvfp4_rtn, Route::large_prefill, 0, "52dd7d1f0: gated with fast4 (same windows)"},
    {"prefill_attn8", Weights::nvfp4_rtn, Route::large_prefill, 32768,
     "with fast4 + chunk16: 32K gate 2/2 windows PASS (KL 0.00589); 8K gate not run"},
    // nvfp4-splitk-fast-v1 on the spec CLI (converted NVFP4 pack, RTX PRO 6000 at 600 W, greedy, sub15d): quality gate
    // EXACT -> FAST 4 windows x 512 rows pooled KL 0.00000 (q99 9.5e-06), top-1 0.9990, dNLL -0.00000; spec ids == FAST
    // serial x1200 on code / think / en / ja; two strict A/B pairs (exact, fast, exact, fast; 3 runs each): verify
    // 14.068 / 14.079 -> 13.993 / 13.980 ms per step (think), 14.936 / 14.937 -> 14.849 / 14.812 (code); tok/s think
    // 262.02 / 261.66 -> 263.83 / 263.90, code 334.32 / 333.94 -> 335.53 / 336.18. No other lever is offered on this
    // route, so FAST-before == EXACT.
    {"split_k", Weights::nvfp4_rtn, Route::windows, 0,
     "gate EXACT -> FAST pooled KL 0.00000 top-1 0.9990; spec ids == FAST serial x1200 (4 contents); "
     "verify -0.6 % think, -0.7 % code (RTX PRO 6000, 600 W)"},
    // Decode CLI (large prefill) row: NOT added. There `fast` already applies fast4 + chunk16 and the composition
    // fast4 + chunk16 + skf1 has no joint gate receipt (G1 measured split-K alone, FAST-before == EXACT). Condition to add
    // {"split_k", Weights::nvfp4_rtn, Route::large_prefill, 0, ...}: a composition gate of that bundle against fast4 + chunk16.
};

inline bool offered(std::string_view lever, Route route) {
    if (route == Route::large_prefill) return true;
    for (const auto windowed : kWindowsLevers) if (windowed == lever) return true;
    return false;
}
inline const Qualified* qualification(std::string_view lever, const Regime& regime, std::span<const Qualified> rows) {
    for (const auto& q : rows)
        if (q.lever == lever && q.weights == regime.weights && q.route == regime.route && regime.prompt_tokens >= q.min_prompt_tokens)
            return &q;
    return nullptr;
}

// Owns the per-regime rows the resolution points into (non-movable: entries hold pointers).
struct Resolved {
    Regime regime;
    std::vector<Lever> rows;
    sp::Engine engine;
    sp::Resolution resolution;
    Resolved() = default;
    Resolved(const Resolved&) = delete;
    Resolved& operator=(const Resolved&) = delete;
    [[nodiscard]] const sp::Entry* find(std::string_view lever) const { return resolution.find(lever); }
};

// cli: {lever name, value} for each per-lever option given on the command line. Profile names: exact = default.
inline std::unique_ptr<Resolved> resolve(std::optional<Name> profile, const Regime& regime, const sp::Pairs& cli,
                                         std::span<const Qualified> rows = qualified) {
    auto out = std::make_unique<Resolved>();
    out->regime = regime;
    for (const auto& lever : table) {
        if (!offered(lever.name, regime.route)) continue;
        Lever row = lever;
        row.ready[static_cast<unsigned>(Name::fast)] = qualification(lever.name, regime, rows) != nullptr;
        out->rows.push_back(row);
    }
    for (const auto& [key, value] : cli) {
        bool known = false;
        for (const auto& row : out->rows) known |= row.name == key;
        if (!known) throw std::invalid_argument("qwen35-cuda-dense: " + key + " is not offered on route " + route_name(regime.route));
    }
    out->engine = sp::Engine{"qwen35-cuda-dense", "", std::span<const Lever>(out->rows), false, {}, {}};
    sp::Inputs inputs;
    inputs.profile = profile;
    inputs.cli = cli;
    inputs.env = [](const char*) -> const char* { return nullptr; };  // CLI-only levers: the environment selects nothing
    out->resolution = sp::resolve(out->engine, inputs);
    for (auto& entry : out->resolution.entries) {
        if (entry.status != sp::Status::pending) continue;
        std::string where;
        for (const auto& q : rows)
            if (q.lever == entry.lever->name)
                where += std::string(where.empty() ? "" : "; ") + weights_name(q.weights) + " / " + route_name(q.route) +
                         (q.min_prompt_tokens ? " / prompt >= " + std::to_string(q.min_prompt_tokens) : std::string());
        entry.reason = where.empty() ? "no gate receipt yet" : "qualified only for " + where;
    }
    return out;
}

// Parses "exact|default|fast|quality|turbo" (exact is the default column).
inline Name parse(std::string_view word) { return word == "exact" ? Name::standard : sp::parse(word); }

// Regime of the options before load (turbo_weights is the weight source of this engine, not the TURBO tier).
template <class Options>
Regime regime_of(const Options& o, Route route, std::uint64_t prompt_tokens) {
    using T = std::remove_cvref_t<decltype(o.turbo_weights)>;
    const Weights w = o.turbo_weights == T::off ? Weights::exact_pack : o.turbo_weights == T::native ? Weights::nvfp4_native : Weights::nvfp4_rtn;
    return Regime{w, route, prompt_tokens};
}

// Writes the resolved levers into the options. Builtin and pending rows touch nothing (EXACT leaves every option as
// built). Every value is validated before any option changes; a bad value throws with the option's message.
template <class Options>
void apply(Options& o, const Resolved& r) {
    using A = std::remove_cvref_t<decltype(o.prefill_arith)>;
    using G = std::remove_cvref_t<decltype(o.prefill_gdn)>;
    using H = std::remove_cvref_t<decltype(o.prefill_hid)>;
    using S = std::remove_cvref_t<decltype(o.nvfp4_split)>;
    const auto take = [&r](std::string_view lever) -> const std::string* {
        const auto* e = r.find(lever);
        if (e == nullptr || e->status == sp::Status::builtin || e->status == sp::Status::pending) return nullptr;
        return &e->value;
    };
    std::optional<A> arith;
    std::optional<bool> big, attn8;
    std::optional<G> gdn;
    std::optional<H> hid;
    std::optional<S> split;
    std::optional<bool> dec8;
    if (const auto* v = take("prefill_arith")) {
        if (*v == "windows") big = false;  // reference: forward_rows windows of 32 + commit, arithmetic exact
        else if (*v == "exact") arith = A::exact;
        else if (*v == "flash") arith = A::flash;
        else if (*v == "fast") arith = A::fast;
        else if (*v == "fast2") arith = A::fast2;
        else if (*v == "fast8") arith = A::fast8;
        else if (*v == "fast4") arith = A::fast4;
        else throw std::runtime_error("--prefill-arith expects windows|exact|flash|fast|fast2|fast8|fast4");
    }
    // enum levers: converted by the engine's own spelling table (parse_lever_value, found by ADL on the option type), never a
    // second list here (a second list rejected --prefill-gdn chunk16tc on the CLI while the engine parser accepted it)
    if (const auto* v = take("prefill_gdn")) gdn = parse_lever_value(G{}, *v);
    if (const auto* v = take("prefill_hid")) hid = parse_lever_value(H{}, *v);
    if (const auto* v = take("prefill_attn8")) {
        if (*v != "on" && *v != "off") throw std::runtime_error("--prefill-attn8 expects on|off");
        attn8 = *v == "on";
    }
    if (const auto* v = take("split_k")) {
        if (*v != "exact" && *v != "fast") throw std::runtime_error("--split-k expects exact|fast");
        split = *v == "fast" ? S::fast : S::exact;
    }
    if (const auto* v = take("decode_attn8")) {
        if (*v != "on" && *v != "off") throw std::runtime_error("--decode-attn8 expects on|off");
        dec8 = *v == "on";
    }
    if (arith) o.prefill_arith = *arith;
    if (big) o.prefill_big = *big;
    if (gdn) o.prefill_gdn = *gdn;
    if (hid) o.prefill_hid = *hid;
    if (attn8) o.prefill_attn8 = *attn8;
    if (split) o.nvfp4_split = *split;
    if (dec8) o.decode_attn8 = *dec8;
}

// The resolved lever set for the effective config. Empty under the default (exact) profile: its effective config stays
// byte-identical to the build without a profile word.
inline std::string describe(const Resolved& r) {
    const auto& res = r.resolution;
    if (res.profile == Name::standard) return {};
    // a tier word that turned nothing on is not called "qualified" (its levers are builtin, pending, off or explicit)
    const std::string qual = res.profile_applied() ? res.qualification() : "nothing applied by the profile";
    std::string out = "  profile: " + std::string(sp::name(res.profile)) + " (" + std::string(res.engine->tag) + "; " +
                      qual + "; regime weights " + weights_name(r.regime.weights) + ", route " +
                      route_name(r.regime.route) + ", prompt " + std::to_string(r.regime.prompt_tokens) + " tokens)\n";
    if (res.quality_fallback()) out += "    no quality rows for this engine: default levers\n";
    for (const auto& e : res.entries) {
        out += "    " + std::string(e.lever->name) + " = " + e.value + " [" + sp::status_name(e.status) + "]";
        if (e.status == sp::Status::set) out += " " + std::string(e.lever->target);  // the identity the profile turned on
        if (e.status == sp::Status::pending) out += " pending " + e.target + ": " + e.reason;
        out += "\n";
    }
    out += "  profile-fingerprint: " + sp::hex_identity(res.fingerprint()) + "\n";
    return out;
}
}  // namespace strix::qwen35::profile
