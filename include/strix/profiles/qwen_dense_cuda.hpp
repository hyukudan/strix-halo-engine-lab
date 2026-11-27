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
    std::string_view cost_device{};
    sp::JointWorkload workload{};
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
     .source = "W4A4 FAST4 large-M prefill; gate with chunk16 PASS 8K KL 0.01275 top-1 0.9648, 32K KL 0.00626 top-1 0.9668 (52dd7d1f0)",
     .identity = {{{"exact", "v1"}, {"fast4", "nvfp4-tile-mma-f32-v1"}}}},
    {.name = "prefill_gdn", .env = "", .option = "--prefill-gdn", .value = {"", "chunk16tc", "", ""}, .builtin = "serial",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "chunk16tc (gdn-chunk16-tf32-v2: split-TF32 tensor-core chunk scan, identity -gdnc16t2)", .off = "serial",
     .source = "chunked GDN prefill, K2 scan in split TF32 (FP32-class); with fast4: 8K 2/2 windows PASS KL 0.01243 top-1 0.9727, "
               "with fast4 + attn8: 32K 2/2 windows PASS KL 0.00761 top-1 0.9727; chunk16 (f32 scan) stays selectable; "
               "ignored under prefill_arith exact",
     .identity = {{{"serial", "v1"}, {"chunk16tc", "gdn-chunk16-tf32-v2"}, {"chunk16", "gdn-chunk16-f32-v1"}}}},
    {.name = "prefill_hid", .env = "", .option = "--prefill-hid", .value = {"", "bf16", "", ""}, .builtin = "f32",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "bf16 (nvfp4-tile-h16-v1: BF16 FFN hidden rows, identity -h16)", .off = "f32",
     .source = "FAST4 glu epilogue stores BF16 H, down operand quantised from it (half the H bytes); no gate receipt yet",
     .identity = {{{"f32", "v1"}, {"bf16", "nvfp4-tile-h16-v1"}}}},
    {.name = "prefill_attn8", .env = "", .option = "--prefill-attn8", .value = {"", "on", "", ""}, .builtin = "off",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "on (e4m3 flash over an e4m3 KV shadow, identity -a8)", .off = "off",
     .source = "with fast4+chunk16: 8K 2/2 windows PASS KL 0.01498 top-1 0.9688 per window, warm 14,361 tok/s (600 W); 32K 2/2 windows PASS KL 0.00589 top-1 "
                      "0.9746-0.9805, warm 12,354 vs 10,946 tok/s",
     .identity = {{{"off", "v1"}, {"on", "prefill-attn8-e4m3-v1"}}}},
    {.name = "split_k", .env = "", .option = "--split-k", .value = {"", "fast", "", ""}, .builtin = "exact",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "fast (nvfp4-splitk-fast-v1, identity -skf1)", .off = "exact",
     .source = "NVFP4 GEMV split-K FAST targets (strix/split_k_policy.hpp, 188-SM cc 12.0 rows); gate EXACT -> FAST pooled "
               "KL 0.00000 top-1 0.9990 dNLL -0.00000; spec verify -0.08 .. -0.12 ms/step (2 ABAB pairs)",
     .identity = {{{"exact", "v1"}, {"fast", "nvfp4-splitk-fast-v1"}}}},
    {.name = "decode_attn8", .env = "", .option = "--decode-attn8",
     .value = {"", "on", "", model::profile::decode_attn8::turbo}, .builtin = "off",
     .kind = Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, true},
     .target = "on (e4m3 decode / verify attention over the e4m3 K/V shadow, tensor-core S and PV, identity -d8)", .off = "off",
     .source = "cuda_core attention_decode8 (attention_flash8.cu): split-key items (2048-key chunk, kv head, 128 row x head "
               "pairs) read each K/V tile once per kv head; verify rows == one-row calls bitwise; gate 8K / 28K / 64K pending",
     .identity = {{{"off", "v1"}, {"on", kAttnIdentity}, {model::profile::decode_attn8::turbo, kAttnIdentity}}}},
    {.name = "kv_fp8", .env = "", .option = "--kv", .value = {"", "", "", model::profile::kv_fp8::turbo},
     .builtin = "native", .kind = Kind::bit_changing, .runnable = {false, false, false, true},
     .target = kKvIdentity, .off = "native", .source = "TURBO gate and workload receipts",
     .omit_builtin_fingerprint = true,
     .identity = {{{"native", "v1"}, {model::profile::kv_fp8::turbo, kKvIdentity}}}},
    {.name = "dense_activation", .env = "", .option = "--dense-activation",
     .value = {"", "", "", model::profile::dense_activation::turbo}, .builtin = "off",
     .kind = Kind::bit_changing, .runnable = {false, false, false, true},
     .target = kDenseIdentity, .off = "off", .source = "TURBO gate and workload receipts",
     .omit_builtin_fingerprint = true,
     .identity = {{{"off", "v1"}, {model::profile::dense_activation::turbo, kDenseIdentity}}}},
    {.name = "drafter_window", .env = "", .option = "--chain-k",
     .value = {"", "", "", model::profile::drafter_window::turbo}, .builtin = "7",
     .kind = Kind::exact, .runnable = {false, false, false, true},
     .target = kDraftIdentity, .off = "7", .source = "TURBO gate, token identity and workload receipts",
     .omit_builtin_fingerprint = true,
     .identity = {{{"7", "chain-window-7-v1"}, {model::profile::drafter_window::turbo, kDraftIdentity}}}, .timed = true},
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
    {Weights::nvfp4_rtn, Route::large_prefill, {"prefill_arith", "fast4", 0, sp::kNoPromptBound,
     "52dd7d1f0: FAST4 + chunk16 gate 8K / 32K PASS (converted NVFP4 pack); 8K 13,437-13,714 tok/s"}},
    // chunk16tc (gdn-chunk16-tf32-v2) gated in the FAST compositions on the converted NVFP4 pack, RTX PRO 6000 at 600 W:
    // 8K (fast4) 2/2 windows PASS pooled KL 0.01243 top-1 0.9727 dNLL -0.0108; 32K (fast4 + attn8) 2/2 windows PASS pooled
    // KL 0.00761 top-1 0.9727 dNLL -0.0047. The earlier chunk16 receipt (52dd7d1f0) stays valid for --prefill-gdn chunk16.
    {Weights::nvfp4_rtn, Route::large_prefill, {"prefill_gdn", "chunk16tc", 0, sp::kNoPromptBound,
     "gdn-chunk16-tf32-v2 with fast4: 8K gate 2/2 windows PASS (KL 0.01243); with fast4 + attn8: 32K gate 2/2 windows PASS (KL 0.00761)"}},
    {Weights::nvfp4_rtn, Route::large_prefill, {"prefill_attn8", "on", 8192, sp::kNoPromptBound,
     "with fast4 + chunk16: 8K gate 2/2 windows PASS (KL 0.01498); 32K gate 2/2 windows PASS (KL 0.00589); "
     "with fast4 + chunk16tc: 32K gate 2/2 windows PASS (KL 0.00761)"}},
    // nvfp4-splitk-fast-v1 on the spec CLI (converted NVFP4 pack, RTX PRO 6000 at 600 W, greedy, sub15d): quality gate
    // EXACT -> FAST 4 windows x 512 rows pooled KL 0.00000 (q99 9.5e-06), top-1 0.9990, dNLL -0.00000; spec ids == FAST
    // serial x1200 on code / think / en / ja; two strict A/B pairs (exact, fast, exact, fast; 3 runs each): verify
    // 14.068 / 14.079 -> 13.993 / 13.980 ms per step (think), 14.936 / 14.937 -> 14.849 / 14.812 (code); tok/s think
    // 262.02 / 261.66 -> 263.83 / 263.90, code 334.32 / 333.94 -> 335.53 / 336.18. No other lever is offered on this
    // route, so FAST-before == EXACT.
    {Weights::nvfp4_rtn, Route::windows, {"split_k", "fast", 0, sp::kNoPromptBound,
     "gate EXACT -> FAST pooled KL 0.00000 top-1 0.9990; spec ids == FAST serial x1200 (4 contents); "
     "verify -0.6 % think, -0.7 % code (RTX PRO 6000, 600 W)"}},
    // Decode CLI (large prefill) row: NOT added. There `fast` already applies fast4 + chunk16tc and the composition
    // fast4 + chunk16tc + skf1 has no joint gate receipt (G1 measured split-K alone, FAST-before == EXACT). Condition to add
    // {Weights::nvfp4_rtn, Route::large_prefill, {"split_k", "", 0, sp::kNoPromptBound, ...}}: a composition gate of that bundle against fast4 + chunk16tc.
};

// Incumbent grants: the FAST cells that main selected from a quality row before the ladder rule, frozen as data at the
// identities and prompt bands they ran in (same shape as `qualified`). A grant keeps the cell applying where no timing
// row covers the regime; it is NOT derived from `qualified`, so a new quality row (a pending cell such as prefill_hid, or
// a wider band) leaves its cell at the tier below until a matching timing row is registered. Do not add rows here.
struct IncumbentGrant {
    Qualified qualified;
    std::string_view identity;  // the declared implementation identity granted (Lever::identity of the FAST value)
};
inline constexpr IncumbentGrant incumbent_grants[] = {
    {{Weights::nvfp4_rtn, Route::large_prefill, {"prefill_arith", "fast4", 0, sp::kNoPromptBound,
     "frozen incumbent: 52dd7d1f0 FAST4 + chunk16 gate 8K / 32K PASS (converted NVFP4 pack)"}},
     "nvfp4-tile-mma-f32-v1"},
    {{Weights::nvfp4_rtn, Route::large_prefill, {"prefill_gdn", "chunk16tc", 0, sp::kNoPromptBound,
     "frozen incumbent: gdn-chunk16-tf32-v2 with fast4 gate 8K / 32K PASS"}},
     "gdn-chunk16-tf32-v2"},
    {{Weights::nvfp4_rtn, Route::large_prefill, {"prefill_attn8", "on", 8192, sp::kNoPromptBound,
     "frozen incumbent: with fast4 + chunk16 / chunk16tc gate 8K / 32K 2/2 windows PASS"}},
     "prefill-attn8-e4m3-v1"},
    {{Weights::nvfp4_rtn, Route::windows, {"split_k", "fast", 0, sp::kNoPromptBound,
     "frozen incumbent: nvfp4-splitk-fast-v1 gate EXACT -> FAST pooled KL 0.00000 top-1 0.9990, spec ids == FAST serial x1200"}},
     "nvfp4-splitk-fast-v1"},
};

inline bool offered(std::string_view lever, Route route, bool drafter = false) {
    if (lever == "drafter_window") return drafter;
    if (route == Route::large_prefill) return true;
    for (const auto windowed : kWindowsLevers) if (windowed == lever) return true;
    return false;
}
inline const Qualified* qualification(std::string_view lever, std::string_view value, const Regime& regime, std::span<const Qualified> rows) {
    for (const auto& q : rows)
        if (q.weights == regime.weights && q.route == regime.route && sp::regime_covers(q.gate, lever, value, regime.prompt_tokens))
            return &q;
    return nullptr;
}

inline const IncumbentGrant* incumbent_grant(const Lever& lever, const Regime& regime,
                                             std::span<const IncumbentGrant> grants) {
    // Bound to the declared implementation identity of the FAST value (Lever::identity, not the display target): a
    // changed implementation under the same value is not granted.
    const auto fast_value = lever.value[static_cast<unsigned>(Name::fast)];
    for (const auto& grant : grants)
        if (!grant.identity.empty() && sp::declared_identity(lever.identity, fast_value) == grant.identity &&
            qualification(lever.name, fast_value, regime, std::span(&grant.qualified, 1)))
            return &grant;
    return nullptr;
}

// Version of the builtin CUDA execution bound by joint receipts. The resolved options below include all per-lever
// overrides of this execution; the caller supplies the device and complete workload of the supported launch form.
// The complete key of one execution for a paired receipt: the execution outside the table (Regime::execution) and
// every lever at the value it runs with its declared implementation identity (Lever::identity). Empty (no paired
// receipt matches) when the execution is not known or a value has no declared identity.
inline std::string joint_execution(const sp::Resolution& resolution, std::string_view execution) {
    if (execution.empty()) return {};
    std::string identity = "qwen35-cuda-dense-run@2 " + std::string(execution);
    for (const auto& entry : resolution.entries) {
        const auto declared = sp::declared_identity(entry.lever->identity, entry.value);
        if (!sp::identity_token(declared)) return {};
        identity += " " + std::string(entry.lever->name) + "=" + entry.value + "{" + std::string(declared) + "}";
    }
    return identity;
}

// Qualification records are scoped to the exact weight form, prefill route and linear session.
struct TurboEvidence {
    Weights weights;
    Route route;
    std::string_view gate_set;
    std::span<const sp::GateRecord> gates;
    std::span<const sp::LadderRow> ladder;
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
        return {true, "TURBO gate PASS (" + std::string(record.gate_set) + ")"};
    }
    return {false, "TURBO gate not run (qual --tier turbo)"};
}

// Owns the per-regime rows the resolution points into (non-movable: entries hold pointers).
struct Resolved {
    Regime regime;
    std::vector<Lever> rows;
    std::vector<sp::LadderRow> ladder;
    sp::Engine engine;
    sp::Resolution resolution;
    std::string execution;  // sp::Regime::execution of the timing lookup (owned here; empty: no ordinary row decides)
    Resolved() = default;
    Resolved(const Resolved&) = delete;
    Resolved& operator=(const Resolved&) = delete;
    [[nodiscard]] const sp::Entry* find(std::string_view lever) const { return resolution.find(lever); }
};
// The paired key of a resolved run (joint_execution over its own execution).
inline std::string joint_execution(const Resolved& resolved) {
    return joint_execution(resolved.resolution, resolved.execution);
}

// cli: {lever name, value} for each per-lever option given on the command line. Profile names: exact = default.
inline std::unique_ptr<Resolved> resolve(std::optional<Name> profile, const Regime& regime, const sp::Pairs& cli,
                                         std::span<const Qualified> rows = qualified,
                                         std::span<const TurboEvidence> evidence = turbo_evidence,
                                         std::span<const IncumbentGrant> grants = incumbent_grants) {
    auto out = std::make_unique<Resolved>();
    out->regime = regime;
    const bool turbo = profile == Name::turbo;
    // The shadow and its reader are one operation: single-cell timing rows of either half never admit it (they time
    // something else), only a joint row (LadderRow::joint) measured on the combined operation does.
    const auto paired_half = [](const sp::LadderRow& row) {
        return row.joint.empty() && !row.incumbent && row.tier == Name::turbo &&
               (row.lever == "kv_fp8" || row.lever == "decode_attn8");
    };
    for (const auto& record : evidence)
        if (record.weights == regime.weights && record.route == regime.route)
            for (const auto& row : record.ladder)
                if (!paired_half(row)) out->ladder.push_back(row);
    // Ordinary timing rows bind the device product, the priced workload and the weight / route settings, not only the
    // architecture: a run whose product or workload is not known (no cost identity) is decided by no ordinary row.
    const bool priced = regime.uniform_prompt_length && regime.linear_session &&
                        regime.workload.prompt_tokens == regime.prompt_tokens;
    if (priced) {
        const std::string settings = std::string("weights=") + weights_name(regime.weights) + "|route=" +
                                     route_name(regime.route) + "|arch=" + std::string(regime.arch) +
                                     (regime.drafter ? "|drafter" : "");
        out->execution = sp::ladder_execution({"cuda", regime.cost_device, regime.workload, settings});
    }
    const sp::Regime timing_regime{sp::Phase::prefill,
                                   regime.uniform_prompt_length && regime.linear_session ? regime.prompt_tokens : 0, 1,
                                   regime.arch, out->execution};
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
        // A timed TURBO schedule (drafter_window) is admitted here by its quality gate only; whether it runs is the
        // shared chain's timing judgement over the complete key of the execution it extends (Lever::timed).
        return turbo_admission(row.target, regime, evidence);
    };
    const bool kv_override = std::ranges::any_of(cli, [](const auto& opt) {
        return opt.first == "kv_fp8" || opt.first == "decode_attn8";
    });
    std::vector<TurboAdmission> admissions;
    std::size_t kv_index = 0, reader_index = 0;
    std::string_view kv_value, reader_value;
    for (const auto& lever : table) {
        if (!offered(lever.name, regime.route, regime.drafter)) continue;
        Lever row = lever;
        const auto* q = qualification(lever.name, lever.value[static_cast<unsigned>(Name::fast)], regime, rows);
        row.ready[static_cast<unsigned>(Name::fast)] = q != nullptr;
        // A frozen incumbent grant (not the quality row) keeps a cell that already ran before the ladder rule.
        if (const auto* grant = incumbent_grant(lever, regime, grants); grant != nullptr && lever.kind != Kind::exact)
            out->ladder.push_back(sp::incumbent_cell(lever.name, lever.value[static_cast<unsigned>(Name::fast)], Name::fast,
                                                     grant->qualified.gate.receipt, grant->identity));
        TurboAdmission result;
        if (turbo && !row.value[kTurboColumn].empty()) {
            result = admission(row);
            if (row.name == "decode_attn8" || row.name == "kv_fp8") {
                // Start from the ordinary FAST cell in both halves. An admitted pair is installed atomically below.
                if (row.name == "kv_fp8") {
                    kv_index = out->rows.size();
                    kv_value = row.value[kTurboColumn];
                } else {
                    reader_index = out->rows.size();
                    reader_value = row.value[kTurboColumn];
                }
                row.ready[kTurboColumn] = false;
            } else {
                row.ready[kTurboColumn] = result.ready;
            }
        }
        admissions.push_back(std::move(result));
        out->rows.push_back(row);
    }
    for (const auto& [key, value] : cli) {
        bool known = false;
        for (const auto& row : out->rows) known |= row.name == key;
        if (!known) throw std::invalid_argument("qwen35-cuda-dense: " + key + " is not offered on route " + route_name(regime.route));
    }
    out->engine = sp::Engine{"qwen35-cuda-dense", "", out->rows, false, kTurboVersion, {}, {}, out->ladder};
    sp::Inputs inputs;
    inputs.profile = profile;
    inputs.cli = cli;
    // This startup policy applies to a complete single-prompt run. Multi-prompt
    // and nonlinear workloads stay outside the timing bands.
    inputs.regime = timing_regime;
    inputs.env = [](const char*) -> const char* { return nullptr; };  // CLI-only levers: the environment selects nothing
    out->resolution = sp::resolve(out->engine, inputs);
    if (turbo) {
        inputs.profile = Name::fast;
        auto fast = sp::resolve(out->engine, inputs);
        const auto reference = joint_execution(fast, out->execution);
        auto& kv = out->resolution.entries[kv_index];
        auto& reader = out->resolution.entries[reader_index];
        kv = std::move(fast.entries[kv_index]);
        reader = std::move(fast.entries[reader_index]);
        if (!kv_override && kv_admit.ready && attn_admit.ready) {
            const auto candidate_base = joint_execution(out->resolution, out->execution);
            const auto work = regime.workload.prompt_tokens == regime.prompt_tokens
                ? regime.workload : sp::JointWorkload{};
            const sp::JointLadderScope scope{regime.cost_device, reference, candidate_base, kKvIdentity, kAttnIdentity, work, reader_value};
            const auto joint = sp::ladder_judge_pair(out->ladder, "kv_fp8", kv_value, Name::turbo, timing_regime,
                                                     "decode_attn8", scope);
            if (joint.rung == sp::Rung::applies) {
                const auto apply_pair = [&](sp::Entry& entry, std::string_view value) {
                    entry.requested = entry.value = value;
                    entry.status = sp::Status::set;
                    entry.target.clear();
                    entry.ladder = sp::ladder_route(entry.lever->name, timing_regime,
                        {value, Name::turbo, " joint timing (" + std::string(joint.row->receipt) + ")"});
                    entry.ladder_lowered = false;
                };
                out->rows[kv_index].ready[kTurboColumn] = true;
                out->rows[reader_index].ready[kTurboColumn] = true;
                apply_pair(kv, kv_value);
                apply_pair(reader, reader_value);
            } else {
                kv.reason = reader.reason = "paired arm: no winning joint receipt for this FAST execution and workload";
            }
        } else {
            const auto why = kv_override ? "controlled explicitly by --kv or --decode-attn8"
                                         : !kv_admit.ready ? kv_admit.reason : attn_admit.reason;
            kv.reason = reader.reason = "paired arm: " + why;
        }
    }
    for (std::size_t i = 0; i < out->resolution.entries.size(); ++i) {
        auto& entry = out->resolution.entries[i];
        if (entry.reason.empty() && !admissions[i].reason.empty() &&
            (entry.status == sp::Status::pending || entry.status == sp::Status::explicit_cli ||
             (entry.status == sp::Status::set && admissions[i].ready))) {
            entry.reason = admissions[i].reason;
            continue;
        }
        if (entry.status != sp::Status::pending || !entry.reason.empty()) continue;
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

// ---- The decode CLI's prompt route: ONE host-side planning entry (plan_prompt, before load) + the executed route after the
// engine's admission (execute_route). The CLI runs exactly what execute_route returns and prints its line.
//
// An explicit or profile prefill_arith takes the large-M route (CudaEngine::prefill), as before this entry existed. The
// BUILTIN 'exact' (no tier word, --profile exact) takes it only where it reproduces, bit for bit, what the same invocation ran
// before: the CLI windows of --rows (forward_rows + commit). That needs (a) every other lever row at its builtin (a
// non-builtin decode_attn8 / split_k / kv_fp8 / dense_activation acts on forward_rows, i.e. on the old prompt, and the large
// route does not run them; TURBO resolves prefill_arith builtin too), no engine override outside the table, no option the
// identity evidence does not cover; (b) the prompt length and weights inside kExactLargeIdentity; (c) no per-row output and no
// slot bench (both read forward_rows windows). Otherwise the windows run and the reason is printed.
//
// Identity of the builtin EXACT large route with those windows: RTX receipts q31 (2,048 and 8,192 tokens) and q33 (32,771 =
// 16 chunks of 2,048 + a 3-row tail on 2 slots; 2,051 at 128 Ki context; --decode-attn8 on, --dump-logits and a declined
// admission run the windows bit for bit), synthetic tests/qwen35/cuda_engine_test.cu (13 .. 387 rows, chunk 128, 1 and 2
// slots, bf16 / q8_0 / onfly nvfp4 packs). The native-nvfp4 weight toggle is not covered.
inline constexpr sp::RegimeGate kExactLargeIdentity{"prefill_arith", "exact", 1, 32771, "rtx q31 + q33; cuda_engine_test"};

struct PromptRequest {
    std::uint64_t prompt_tokens = 0;    // the prompt this invocation prefills (--prompt-ids, or --prompt-len + --ctx-fill)
    std::uint32_t rows = 16;            // --rows: the CLI windows width
    bool per_row_output = false;        // --dump-logits / --dump-mtp: every prompt row, only the windows produce them
    bool slot_bench = false;            // --slots N: slot 0 is filled by forward_slots windows of --rows (no prompt route)
    std::vector<std::string> unshown;   // options outside the lever table that change forward_rows arithmetic, no identity
};
enum class PromptPath : std::uint8_t {
    large,           // CudaEngine::prefill on the large-M route (admission at load)
    engine_windows,  // CudaEngine::prefill's own windows (--prefill-arith windows: the engine reference)
    windows,         // the CLI windows of --rows + commit
};
struct PromptRoute {
    PromptPath path = PromptPath::windows;
    bool by_default = false;  // large only because the builtin exact resolved: a declined admission keeps the CLI windows
    std::string arith;        // the resolved prefill_arith value ("" when the route does not offer the lever)
    std::string reason;       // why the windows run (empty when large)
    std::uint32_t rows = 16;  // the CLI windows width
    [[nodiscard]] bool large() const { return path == PromptPath::large; }
};
inline PromptRoute plan_prompt(const Resolved& r, const PromptRequest& q) {
    PromptRoute out;
    out.rows = q.rows;
    const auto* pa = r.find("prefill_arith");
    if (pa) out.arith = pa->value;
    const bool by_default = pa && (pa->status == sp::Status::builtin || pa->status == sp::Status::pending);
    std::string keep;  // the first reason the builtin keeps the windows
    if (by_default) {
        for (const auto& e : r.resolution.entries)
            if (keep.empty() && e.lever && e.lever->name != "prefill_arith" && e.value != e.lever->builtin)
                keep = std::string(e.lever->name) + " " + e.value + " acts on the windows the builtin exact ran";
        for (const auto& o : r.resolution.engine_overrides)
            if (keep.empty()) keep = "engine override " + o;
        for (const auto& o : q.unshown)
            if (keep.empty()) keep = o + ": no identity of the large route with the windows";
        if (keep.empty() && r.regime.weights == Weights::nvfp4_native) keep = "native nvfp4 weights: no identity of the large route";
        if (keep.empty() && !sp::regime_covers(kExactLargeIdentity, "prefill_arith", "exact", q.prompt_tokens))
            keep = "prompt " + std::to_string(q.prompt_tokens) + " tokens outside the large-route identity range " +
                   std::to_string(kExactLargeIdentity.min_prompt_tokens) + ".." + std::to_string(kExactLargeIdentity.max_prompt_tokens);
    }
    if (out.arith.empty()) {
        out.reason = "prefill_arith not offered on this route";
    } else if (q.slot_bench) {
        out.reason = "--slots: slot 0 is prefilled by forward_slots windows" +
                     (out.arith == "exact" || out.arith == "windows" ? std::string() : "; prefill_arith " + out.arith + " not applied");
    } else if (q.per_row_output) {
        if (out.arith != "exact" && out.arith != "windows")
            throw std::invalid_argument("qwen35-cuda-dense: --dump-logits / --dump-mtp read every prompt row on the windows route "
                                        "(exact arithmetic); prefill_arith " + out.arith + " cannot run there: drop the profile "
                                        "word or set --prefill-arith exact");
        out.reason = "--dump-logits / --dump-mtp read every prompt row";
    } else if (!keep.empty()) {
        out.reason = "builtin exact keeps the windows: " + keep;
    } else if (out.arith == "windows") {
        out.path = PromptPath::engine_windows;
        out.reason = "--prefill-arith windows";
    } else {
        out.path = PromptPath::large;
        out.by_default = by_default;
    }
    return out;
}
// The intent line (effective config, before load): the route asked for; the large route waits for the admission.
inline std::string describe_route(const PromptRoute& p) {
    switch (p.path) {
    case PromptPath::large:
        return "  prompt-route: large-M prefill, prefill_arith " + p.arith + (p.by_default ? " (builtin)" : "") +
               "; workspace admitted at load\n";
    case PromptPath::engine_windows: return "  prompt-route: engine windows (" + p.reason + ")\n";
    case PromptPath::windows: break;
    }
    return "  prompt-route: windows of " + std::to_string(p.rows) + " rows (" + p.reason + ")\n";
}
// What runs after the engine's admission (a: CudaEngine::PrefillAdmission or the same fields; nullptr: no large route asked).
// A declined builtin route keeps the CLI windows it replaced; a declined explicit / profile route runs the engine's windows
// fallback (what CudaEngine::prefill ran before the admission existed). `line` is the receipt of the executed route.
struct ExecutedRoute {
    PromptPath path = PromptPath::windows;
    std::string line;
};
template <class Admission>
ExecutedRoute execute_route(const PromptRoute& p, const Admission* a) {
    ExecutedRoute x;
    const auto windows = [&](const std::string& why) {
        x.path = PromptPath::windows;
        x.line = "  prompt-route executed: windows of " + std::to_string(p.rows) + " rows (" + why + ")\n";
        return x;
    };
    if (p.path == PromptPath::windows || a == nullptr) return windows(p.reason.empty() ? "no admission" : p.reason);
    if (p.path == PromptPath::large && a->admitted) {
        x.path = PromptPath::large;
        x.line = "  prompt-route executed: large-M prefill, prefill_arith " + p.arith + "; slots " + std::to_string(a->slots) + " of " +
                 std::to_string(a->slots_wanted) + "; workspace " + std::to_string(a->workspace_bytes >> 20U) + " MiB admitted, " +
                 std::to_string(a->free_bytes >> 20U) + " MiB free after\n";
        return x;
    }
    const std::string why = p.path == PromptPath::engine_windows ? p.reason : "large route not admitted: " + a->reason;
    if (p.by_default) return windows("builtin exact, " + why);
    x.path = PromptPath::engine_windows;
    x.line = "  prompt-route executed: engine windows of " + std::to_string(a->fallback_rows) + " rows (" + why +
             (p.arith == "exact" || p.arith == "windows" ? "" : "; prefill_arith " + p.arith + " not applied") + ")\n";
    return x;
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
        if (!e.ladder.empty()) out += e.ladder + "\n";
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
    kv.tier = r.resolution.profile;
    std::string on;
    unsigned offered_levers = 0, on_levers = 0;
    for (const auto lever : kKvLevers) {
        const auto* e = r.find(lever);
        if (e == nullptr) continue;
        ++offered_levers;
        if (e->value == "on" && e->status != sp::Status::builtin && e->status != sp::Status::pending) {
            if (kv.tier == Name::standard) {
                kvp::Request request;
                kvp::set(request, "fp8", e->lever->option);
                (void)kvp::resolve(kv.tier, request, kKvCaps);
            }
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
