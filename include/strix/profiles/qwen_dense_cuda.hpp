#pragma once
// CUDA Qwen dense (qwen35, 27B) product profile: ONE tier word '--profile exact|default|fast|quality' resolved by the
// shared resolver (strix::profile::resolve, strix/engine_profile.hpp) for the decode and the speculative CLI.
// This header is data: the lever rows and the regimes in which each FAST cell is qualified (gate receipt). Choosing
// `fast` turns ON every lever qualified for the run's regime; an explicit per-lever option overrides the profile and
// is recorded as explicit; `exact` / `default` touches no option (today's bits, byte-identical effective config).
// Promoting a lever after its gate is ONE row in `qualified` below, never code.
// TURBO inherits these FAST rows and considers the model's additional defaults against TURBO evidence.
// Host-only: no CUDA types (options are templated).
#include "strix/model_data/qwen3_8_27b.hpp"
#include "strix/profile/gate_records.hpp"
#include "strix/session/kv_precision.hpp"

#include <algorithm>
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
namespace model = ::strix::model_data::qwen3_8_27b;
inline constexpr std::string_view kTurboVersion = "qwen35-cuda-turbo-v1";
inline constexpr std::string_view kKvIdentity = "kv-e4m3-shadow-v1";
inline constexpr std::string_view kAttnIdentity = "qk-pv-e4m3-v1";
inline constexpr std::string_view kDenseIdentity = model::profile::dense_activation::identity;
inline constexpr std::string_view kDraftIdentity = model::profile::drafter_window::identity;
inline constexpr std::size_t kTurboColumn = static_cast<std::size_t>(Name::turbo);

// Regime of a run, known before the engine loads.
enum class Weights : std::uint8_t { exact_pack, nvfp4_native, nvfp4_rtn };  // --turbo-weights off | native | onfly/converted
enum class Route : std::uint8_t { large_prefill, windows };  // decode CLI large-M prefill | spec CLI forward_rows windows
struct Regime {
    Weights weights = Weights::exact_pack;
    Route route = Route::large_prefill;
    std::uint64_t prompt_tokens = 0;
    std::string_view device{}, arch{};
    bool drafter = false;
    bool linear_session = true;
    bool uniform_prompt_length = true;
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
    {.name = "prefill_gdn", .env = "", .option = "--prefill-gdn", .value = {"", "chunk16tc", "", ""}, .builtin = "serial",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "chunk16tc (gdn-chunk16-tf32-v2: split-TF32 tensor-core chunk scan, identity -gdnc16t2)", .off = "serial",
     .source = "chunked GDN prefill, K2 scan in split TF32 (FP32-class); with fast4: 8K 2/2 windows PASS KL 0.01243 top-1 0.9727, "
               "with fast4 + attn8: 32K 2/2 windows PASS KL 0.00761 top-1 0.9727; chunk16 (f32 scan) stays selectable; "
               "ignored under prefill_arith exact"},
    {.name = "prefill_hid", .env = "", .option = "--prefill-hid", .value = {"", "bf16", "", ""}, .builtin = "f32",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "bf16 (nvfp4-tile-h16-v1: BF16 FFN hidden rows, identity -h16)", .off = "f32",
     .source = "FAST4 glu epilogue stores BF16 H, down operand quantised from it (half the H bytes); no gate receipt yet"},
    {.name = "prefill_attn8", .env = "", .option = "--prefill-attn8", .value = {"", "on", "", ""}, .builtin = "off",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "on (e4m3 flash over an e4m3 KV shadow, identity -a8)", .off = "off",
     .source = "with fast4+chunk16: 8K 2/2 windows PASS KL 0.01498 top-1 0.9688 per window, warm 14,361 tok/s (600 W); 32K 2/2 windows PASS KL 0.00589 top-1 "
                      "0.9746-0.9805, warm 12,354 vs 10,946 tok/s"},
    {.name = "split_k", .env = "", .option = "--split-k", .value = {"", "fast", "", ""}, .builtin = "exact",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "fast (nvfp4-splitk-fast-v1, identity -skf1)", .off = "exact",
     .source = "NVFP4 GEMV split-K FAST targets (strix/split_k_policy.hpp, 188-SM cc 12.0 rows); gate EXACT -> FAST pooled "
               "KL 0.00000 top-1 0.9990 dNLL -0.00000; spec verify -0.08 .. -0.12 ms/step (2 ABAB pairs)"},
    {.name = "decode_attn8", .env = "", .option = "--decode-attn8",
     .value = {"", "on", "", model::profile::decode_attn8::turbo}, .builtin = "off",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, true},
     .target = "on (e4m3 decode / verify attention over the e4m3 K/V shadow, tensor-core S and PV, identity -d8)", .off = "off",
     .source = "cuda_core attention_decode8 (attention_flash8.cu): split-key items (2048-key chunk, kv head, 128 row x head "
               "pairs) read each K/V tile once per kv head; verify rows == one-row calls bitwise; gate 8K / 28K / 64K pending"},
    {.name = "kv_fp8", .env = "", .option = "--kv", .value = {"", "", "", model::profile::kv_fp8::turbo},
     .builtin = "native", .kind = Kind::bit_changing, .runnable = {false, false, false, true},
     .target = kKvIdentity, .off = "native", .source = "TURBO gate and workload receipts",
     .omit_builtin_fingerprint = true},
    {.name = "dense_activation", .env = "", .option = "--dense-activation",
     .value = {"", "", "", model::profile::dense_activation::turbo}, .builtin = "off",
     .kind = Kind::bit_changing, .runnable = {false, false, false, true},
     .target = kDenseIdentity, .off = "off", .source = "TURBO gate and workload receipts",
     .omit_builtin_fingerprint = true},
    {.name = "drafter_window", .env = "", .option = "--chain-k",
     .value = {"", "", "", model::profile::drafter_window::turbo}, .builtin = "7",
     .kind = Kind::exact, .runnable = {false, false, false, true},
     .target = kDraftIdentity, .off = "7", .source = "TURBO gate, token identity and workload receipts",
     .omit_builtin_fingerprint = true},
};

// Levers that act on the windowed forward_rows path (the spec CLI prefills through it: cuda_spec.hpp prefill). The
// prefill_* rows act only on the large-M prefill and are not offered there.
// decode_attn8 acts on every forward_rows call (decode, verify and the spec CLI's prefill windows): offered on both routes.
inline constexpr std::string_view kWindowsLevers[] = {"split_k", "decode_attn8", "kv_fp8", "dense_activation"};

// Where a FAST cell is qualified: per (weights, route), one shared regime ledger row (sp::RegimeGate: lever, value ""
// = the cell's value, prompt range, receipt). Lower bound 0 also covers a prompt length unknown before load.
struct Qualified {
    Weights weights;
    Route route;
    sp::RegimeGate gate;
};
inline constexpr Qualified qualified[] = {
    {Weights::nvfp4_rtn, Route::large_prefill, {"prefill_arith", "", 0, sp::kNoPromptBound,
     "52dd7d1f0: FAST4 + chunk16 gate 8K / 32K PASS (converted NVFP4 pack); 8K 13,437-13,714 tok/s"}},
    // chunk16tc (gdn-chunk16-tf32-v2) gated in the FAST compositions on the converted NVFP4 pack, RTX PRO 6000 at 600 W:
    // 8K (fast4) 2/2 windows PASS pooled KL 0.01243 top-1 0.9727 dNLL -0.0108; 32K (fast4 + attn8) 2/2 windows PASS pooled
    // KL 0.00761 top-1 0.9727 dNLL -0.0047. The earlier chunk16 receipt (52dd7d1f0) stays valid for --prefill-gdn chunk16.
    {Weights::nvfp4_rtn, Route::large_prefill, {"prefill_gdn", "", 0, sp::kNoPromptBound,
     "gdn-chunk16-tf32-v2 with fast4: 8K gate 2/2 windows PASS (KL 0.01243); with fast4 + attn8: 32K gate 2/2 windows PASS (KL 0.00761)"}},
    {Weights::nvfp4_rtn, Route::large_prefill, {"prefill_attn8", "", 8192, sp::kNoPromptBound,
     "with fast4 + chunk16: 8K gate 2/2 windows PASS (KL 0.01498); 32K gate 2/2 windows PASS (KL 0.00589); "
     "with fast4 + chunk16tc: 32K gate 2/2 windows PASS (KL 0.00761)"}},
    // nvfp4-splitk-fast-v1 on the spec CLI (converted NVFP4 pack, RTX PRO 6000 at 600 W, greedy, sub15d): quality gate
    // EXACT -> FAST 4 windows x 512 rows pooled KL 0.00000 (q99 9.5e-06), top-1 0.9990, dNLL -0.00000; spec ids == FAST
    // serial x1200 on code / think / en / ja; two strict A/B pairs (exact, fast, exact, fast; 3 runs each): verify
    // 14.068 / 14.079 -> 13.993 / 13.980 ms per step (think), 14.936 / 14.937 -> 14.849 / 14.812 (code); tok/s think
    // 262.02 / 261.66 -> 263.83 / 263.90, code 334.32 / 333.94 -> 335.53 / 336.18. No other lever is offered on this
    // route, so FAST-before == EXACT.
    {Weights::nvfp4_rtn, Route::windows, {"split_k", "", 0, sp::kNoPromptBound,
     "gate EXACT -> FAST pooled KL 0.00000 top-1 0.9990; spec ids == FAST serial x1200 (4 contents); "
     "verify -0.6 % think, -0.7 % code (RTX PRO 6000, 600 W)"}},
    // Decode CLI (large prefill) row: NOT added. There `fast` already applies fast4 + chunk16tc and the composition
    // fast4 + chunk16tc + skf1 has no joint gate receipt (G1 measured split-K alone, FAST-before == EXACT). Condition to add
    // {Weights::nvfp4_rtn, Route::large_prefill, {"split_k", "", 0, sp::kNoPromptBound, ...}}: a composition gate of that bundle against fast4 + chunk16tc.
};

inline bool offered(std::string_view lever, Route route, bool drafter = false) {
    if (lever == "drafter_window") return drafter;
    if (route == Route::large_prefill) return true;
    for (const auto windowed : kWindowsLevers) if (windowed == lever) return true;
    return false;
}
inline const Qualified* qualification(std::string_view lever, const Regime& regime, std::span<const Qualified> rows) {
    for (const auto& q : rows)
        if (q.weights == regime.weights && q.route == regime.route && sp::regime_covers(q.gate, lever, "", regime.prompt_tokens))
            return &q;
    return nullptr;
}

// Qualification records are scoped to the exact weight form, prefill route and linear session.
struct TurboEvidence {
    Weights weights;
    Route route;
    std::string_view gate_set;
    std::span<const sp::GateRecord> gates;
    std::span<const sp::RegimeRecord> regimes;
};
inline constexpr std::span<const TurboEvidence> turbo_evidence{};

struct TurboAdmission {
    bool ready = false;
    std::string reason;
};

inline TurboAdmission turbo_admission(std::string_view identity, const Regime& run,
                                      std::span<const TurboEvidence> evidence) {
    if (!run.linear_session) return {false, "requires a single linear session"};
    if (!run.uniform_prompt_length) return {false, "requires a single prompt length per run"};
    for (const auto& record : evidence) {
        if (record.weights != run.weights || record.route != run.route) continue;
        const auto gate = sp::gate_verdict(identity, {model::name, record.gate_set, {}, {}}, Name::turbo, record.gates);
        if (!gate) continue;
        if (*gate == sp::GateVerdict::fail) return {false, "TURBO gate FAIL"};
        if (*gate == sp::GateVerdict::hold) return {false, "TURBO gate HOLD"};
        if (*gate == sp::GateVerdict::unresolved) return {false, "TURBO gate UNRESOLVED"};
        const auto* receipt = sp::measured_regime(identity, run.device, {.context = run.prompt_tokens}, record.regimes);
        if (receipt) return {true, "TURBO gate PASS (" + std::string(record.gate_set) + "), gain over FAST: " +
                                    std::string(receipt->receipt)};
        return {false, "TURBO gate PASS; no gain over FAST recorded for this regime"};
    }
    return {false, "TURBO gate not run (qual --tier turbo)"};
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
                                         std::span<const Qualified> rows = qualified,
                                         std::span<const TurboEvidence> evidence = turbo_evidence) {
    auto out = std::make_unique<Resolved>();
    out->regime = regime;
    const bool turbo = profile == Name::turbo;
    const auto cuda_admission = [&](std::string_view identity) -> TurboAdmission {
        if (regime.arch != "sm_120") return {false, "requires CUDA sm_120"};
        return turbo_admission(identity, regime, evidence);
    };
    const auto attn_admit = turbo ? cuda_admission(kAttnIdentity) : TurboAdmission{};
    const auto kv_admit = turbo ? cuda_admission(kKvIdentity) : TurboAdmission{};
    const auto admission = [&](const Lever& row) -> TurboAdmission {
        if (row.name == "decode_attn8") return attn_admit;
        if (row.name == "kv_fp8") return kv_admit;
        if (row.name == "dense_activation") {
            if (regime.weights == Weights::exact_pack) return {false, "requires NVFP4 projection weights"};
            return cuda_admission(kDenseIdentity);
        }
        return turbo_admission(row.target, regime, evidence);
    };
    const bool kv_override = std::ranges::any_of(cli, [](const auto& opt) {
        return opt.first == "kv_fp8" || opt.first == "decode_attn8";
    });
    std::vector<TurboAdmission> admissions;
    for (const auto& lever : table) {
        if (!offered(lever.name, regime.route, regime.drafter)) continue;
        Lever row = lever;
        row.ready[static_cast<unsigned>(Name::fast)] = qualification(lever.name, regime, rows) != nullptr;
        TurboAdmission result;
        if (turbo && !row.value[kTurboColumn].empty()) {
            result = admission(row);
            if (row.name == "decode_attn8" || row.name == "kv_fp8") {
                // The current CUDA body reads the shadow and runs both E4M3 products as one operation.
                const auto& peer = row.name == "kv_fp8" ? attn_admit : kv_admit;
                if (result.ready && !peer.ready) result = {false, "paired arm: " + peer.reason};
                if (kv_override) result = {false, "controlled explicitly by --kv or --decode-attn8"};
            }
            if (row.name == "decode_attn8" && !result.ready && row.ready[static_cast<unsigned>(Name::fast)])
                row.value[kTurboColumn] = {};
            row.ready[kTurboColumn] = result.ready;
        }
        admissions.push_back(std::move(result));
        out->rows.push_back(row);
    }
    for (const auto& [key, value] : cli) {
        bool known = false;
        for (const auto& row : out->rows) known |= row.name == key;
        if (!known) throw std::invalid_argument("qwen35-cuda-dense: " + key + " is not offered on route " + route_name(regime.route));
    }
    out->engine = sp::Engine{"qwen35-cuda-dense", "", std::span<const Lever>(out->rows), false, kTurboVersion, {}};
    sp::Inputs inputs;
    inputs.profile = profile;
    inputs.cli = cli;
    inputs.env = [](const char*) -> const char* { return nullptr; };  // CLI-only levers: the environment selects nothing
    out->resolution = sp::resolve(out->engine, inputs);
    for (std::size_t i = 0; i < out->resolution.entries.size(); ++i) {
        auto& entry = out->resolution.entries[i];
        if (!admissions[i].reason.empty() &&
            (entry.status == sp::Status::pending || entry.status == sp::Status::explicit_cli ||
             (entry.status == sp::Status::set && admissions[i].ready))) {
            entry.reason = admissions[i].reason;
            continue;
        }
        if (entry.status != sp::Status::pending) continue;
        std::string where;
        for (const auto& q : rows)
            if (q.gate.lever == entry.lever->name)
                where += std::string(where.empty() ? "" : "; ") + weights_name(q.weights) + " / " + route_name(q.route) +
                         sp::regime_range_text(q.gate);
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
    std::optional<bool> dense;
    if (const auto* v = take("dense_activation")) {
        if (*v != "on" && *v != "off") throw std::runtime_error("--dense-activation expects on|off");
        dense = *v == "on";
    }
    if (arith) o.prefill_arith = *arith;
    if (big) o.prefill_big = *big;
    if (gdn) o.prefill_gdn = *gdn;
    if (hid) o.prefill_hid = *hid;
    if (attn8) o.prefill_attn8 = *attn8;
    if (split) o.nvfp4_split = *split;
    if (dec8) o.decode_attn8 = *dec8;
    if (dense) o.dense_activation = *dense;
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
    if (res.is_turbo()) out += "    " + res.versioned() + ": quality may be lower than FAST\n";
    if (res.quality_fallback()) out += "    no quality rows for this engine: default levers\n";
    for (const auto& e : res.entries) {
        out += "    " + std::string(e.lever->name) + " = " + e.value + " [" + sp::status_name(e.status) + "]";
        if (e.status == sp::Status::set) out += " " + std::string(e.lever->target);  // the identity the profile turned on
        if (e.status == sp::Status::pending) out += " pending " + e.target;
        if (!e.reason.empty()) out += ": " + e.reason;
        out += "\n";
    }
    out += "  profile-fingerprint: " + sp::hex_identity(res.fingerprint()) + "\n";
    return out;
}
// KV precision (--kv, strix/session/kv_precision.hpp): native BF16; fp8 is the e4m3 K/V shadow read by the attn8 levers
// (decode_attn8 on every route, prefill_attn8 on the large-prefill route). --kv folds into those lever options; a lever
// option given together with --kv is refused (two spellings of one setting). Without --kv the levers keep their own
// options and profile cells, and observe_kv() reports what they actually turned on.
namespace kvp = ::strix::session::kv_precision;
inline constexpr kvp::Caps kKvCaps{"qwen35-cuda-dense", kvp::Precision::bf16, kvp::bit(kvp::Precision::fp8), {},
                                   "kv-e4m3-tok-v1 (e4m3 shadow, attn8)", {}};
inline constexpr std::string_view kKvLevers[] = {"decode_attn8", "prefill_attn8"};

inline kvp::Resolution resolve_kv(std::optional<Name> profile, const kvp::Request& request, Route route, sp::Pairs& cli) {
    auto kv = kvp::resolve(profile.value_or(Name::standard), request, kKvCaps, kvp::TierDefault::resolved_by_levers);
    if (!request.given()) return kv;
    for (const auto lever : kKvLevers) {
        for (const auto& [key, value] : cli)
            if (key == lever)
                throw std::invalid_argument("qwen35-cuda-dense: --kv sets " + std::string(lever) + "; drop the --" +
                                            std::string(lever.substr(0, lever.find('_'))) + "-attn8 option");
    }
    for (const auto lever : kKvLevers)
        if (offered(lever, route)) cli.emplace_back(std::string(lever), kv.precision == kvp::Precision::fp8 ? "on" : "off");
    if (profile == Name::turbo)
        cli.emplace_back("kv_fp8", kv.precision == kvp::Precision::fp8 ? "fp8" : "native");
    return kv;
}

// Records the attention's KV read precision as the resolved levers left it (profile cells or per-lever options).
inline void observe_kv(kvp::Resolution& kv, const Resolved& r) {
    std::string on;
    unsigned offered_levers = 0, on_levers = 0;
    for (const auto lever : kKvLevers) {
        const auto* e = r.find(lever);
        if (e == nullptr) continue;
        ++offered_levers;
        if (e->value == "on" && e->status != sp::Status::builtin && e->status != sp::Status::pending) {
            ++on_levers;
            on += (on.empty() ? "" : ",") + std::string(lever);
        }
    }
    if (on_levers == 0 || kv.precision == kvp::Precision::fp8) return;
    if (const auto* e = r.find("kv_fp8"); e && e->status == sp::Status::set) {
        kv.precision = kvp::Precision::fp8;
        kv.source = kvp::Source::tier;
        kv.form = std::string(kKvCaps.form(kv.precision));
        kv.note = on + ": E4M3 KV shadow read for these phases; BF16 backing retained";
        return;
    }
    if (on_levers == offered_levers) {
        kv.precision = kvp::Precision::fp8;
        kv.source = kvp::Source::lever;
        kv.form = std::string(kKvCaps.form(kv.precision));
    }
    kv.note += std::string(kv.note.empty() ? "" : "; ") + on + " on: attention reads the e4m3 KV shadow" +
               (on_levers == offered_levers ? "" : " in that phase only");
}
}  // namespace strix::qwen35::profile
