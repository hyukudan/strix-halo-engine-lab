#pragma once

#include "strix/env_flags.hpp"
#include "strix/fnv.hpp"
#include "strix/turbo.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Product profiles of an engine: one switch '--profile default|fast|quality|turbo' selects every lever's value.
//   default  today's exact defaults (exports nothing: the plain invocation, byte for byte)
//   fast     default + every QUALIFIED bit-changing lever of the engine (and any marked provisional, stamped so)
//   quality  a quality pack where the engine has one, else default plus one notice
//   turbo    opt-in by name, never a default: every speed lever under the relative TURBO bar (strix/turbo.hpp). An
//            empty turbo cell takes the lever's fast cell (value and admission); a cell equal to `off` keeps the lever
//            out. Its versioned name (Engine::turbo_version) and the weight toggle enter the rank agreement and the
//            cache namespace even with every lever off. An engine without a versioned name refuses turbo.
// Each lever is one table row (the engine's *_profile_table.hpp). A profile column that is not qualified yet
// (`ready = false`) is pending: it is never applied and never makes the profile refuse; the builtin keeps
// running, the receipt lists the row with its declared target and one stderr line says so. '--allow-pending'
// (or the engine's ..._PROFILE_ALLOW_PENDING=1) applies a pending column whose implementation exists in this
// build (`runnable`), for benchmarks only, and stamps the receipt "unqualified: <rows>". A `provisional` column
// runs its runnable code ahead of qualification without --allow-pending, stamped "qualification in progress: <rows>"
// with one stderr line per row; it is never "qualified". Precedence per lever:
// CLI option > environment variable > profile > builtin; an explicit setting is recorded as explicit (or `off`).
// Header-only and host-only (no HIP/CUDA, no JSON library): shared by the two-node engines' apps.
namespace strix::profile {

enum class Name : unsigned { standard = 0, fast = 1, quality = 2, turbo = 3 };  // `standard` prints as "default"
inline constexpr std::size_t kProfileColumns = 4;  // default, fast, quality, turbo
enum class Kind { exact, bit_changing, pack };

// One declared implementation identity: the value a row can run and the versioned implementation it runs
// ("fast4" -> "nvfp4-tile-mma-f32-v1"). Unused entries are empty.
struct ValueIdentity {
    std::string_view value, identity;
};
inline constexpr std::size_t kIdentityValues = 6;
using Identities = std::array<ValueIdentity, kIdentityValues>;

struct Lever {
    std::string_view name;                  // receipt key
    std::string_view env;                   // switch the profile exports (empty: CLI-only lever)
    std::string_view option;                // CLI option that overrides the lever (empty: none)
    std::array<std::string_view, kProfileColumns> value;  // per profile (default, fast, quality, turbo); empty: left alone
    std::string_view builtin;               // behaviour of this build with the switch unset
    Kind kind = Kind::exact;
    std::array<bool, kProfileColumns> ready{};     // qualified (qualification receipt) and present; false: pending
    std::array<bool, kProfileColumns> runnable{};  // a pending column's implementation exists in this build
    std::string_view target;                // declared target of a pending column; empty: the value
    std::string_view off;                   // explicit value that switches the lever off
    std::string_view source;                // qualification receipt, or what a pending column waits for
    bool native_option = false;            // an adapter's native CLI parser owns validation/export
    std::array<bool, kProfileColumns> provisional{};  // selected ahead of qualification; still requires `runnable`; wins over `ready` until the row is flipped
    bool omit_builtin_fingerprint = false;  // additive governed rows preserve the legacy identity at builtin
    // Comma list of values measured to compute the same bits as each other (receipt in `source`), e.g. window widths of
    // a window-invariant prefill: choosing among them is scheduling, so an explicit member shares the qualification of a
    // ready column whose value is in the list. Empty: every value is its own class. Not part of the fingerprint.
    std::string_view same_bits{};
    // Declared implementation identity of every value this row can run: the builtin, every non-empty column value and
    // `off` at least (identity_defect()); an explicit alternative the list does not name is not identified. Timing
    // receipts bind these identities (LadderArms::over()): a changed implementation under the same value is a new
    // identity here, and no receipt of the old one decides it. Not part of the fingerprint.
    Identities identity{};
    // An exact schedule (kind exact) whose non-builtin FAST / TURBO cell applies only where a matching timing receipt
    // shows a gain, judged in the shared chain like a lossy cell (ladder_judged()).
    bool timed = false;
    // For 0/1 environment switches, match the runtime flag parser with `builtin` as its default.
    std::optional<EnvFlagMatch> env_flag_match{};
};

// True when `value` is one whole item of the comma list `list`.
[[nodiscard]] inline bool bit_class_member(std::string_view list, std::string_view value) {
    while (!value.empty() && !list.empty()) {
        const auto comma = list.find(',');
        if (list.substr(0, comma) == value) return true;
        if (comma == std::string_view::npos) break;
        list.remove_prefix(comma + 1U);
    }
    return false;
}

// Column of `lever` that `profile` reads: an empty turbo cell takes the fast cell (TURBO includes every FAST lever
// unless its turbo cell says otherwise); a direct query for an absent column is rejected. Shared by
// every adapter whose rows carry `value` per profile.
template <class Row>
inline unsigned value_column(const Row& lever, Name profile) {
    const auto column = static_cast<unsigned>(profile);
    // Legacy adapters have only default/fast/quality columns and refuse TURBO.
    if (column >= std::size(lever.value)) throw std::invalid_argument("profile column is not available on this adapter");
    if (profile == Name::turbo && lever.value[column].empty())
        return static_cast<unsigned>(Name::fast);
    return column;
}

// Ladder rule: TURBO is never slower than FAST, FAST never slower than EXACT, in any regime. A lossy cell applies in a
// regime only where a measured ladder row shows its tier faster than the tier below beyond the row's stated noise
// margin; in every other regime (no row, a row within noise, a slower row) it resolves to the tier below, and the route
// line says why. A regime is the phase, the context (prefill: prompt tokens; decode: resident context of the step), the
// lane count and the device class. Rows are data next to the engine's lever table (Engine::ladder); the same predicate
// gates weight-load decision rows (strix/weight/levers.hpp WeightLoadEvidence).
enum class Phase : std::uint8_t { prefill, decode };
inline constexpr std::uint64_t kNoContextBound = ~std::uint64_t{0};
inline constexpr std::uint32_t kNoLaneBound = ~std::uint32_t{0};

struct Regime {
    Phase phase = Phase::decode;
    std::uint64_t context = 0;  // 0: not known (never inside a row's band)
    std::uint32_t lanes = 1;
    std::string_view device{};  // device class of the run ("sm_120", "gfx1151")
    // The effective execution outside the lever table that a timing receipt depends on, spelled by the adapter with
    // ladder_execution(): backend, device product identity, priced workload and the effective timing settings. Part of
    // every ordinary timing reference (LadderArms::over); empty: not known, and no ordinary timing row decides.
    std::string_view execution{};
};

// At startup a server often knows its capacity but cannot resolve per request before a real prompt arrives.
[[nodiscard]] constexpr Regime startup_prefill_regime(std::string_view device = {}) {
    return {Phase::prefill, 0, 1, device};
}

// Joint receipts bind the complete workload and both executions, beyond a timing band's coarse bounds.
struct JointWorkload {
    std::uint64_t prompt_tokens = 0, context_capacity = 0;
    std::uint32_t lanes = 0, prefill_chunk = 0, prefill_rows = 0, decode_rows = 0,
                  verify_rows = 0, generated_tokens = 0;
    bool operator==(const JointWorkload&) const = default;
};

// What an ordinary timing receipt was measured on besides the lever values (Regime::execution): the backend, the device
// product (device/{hip,cuda}_cost_identity.hpp: architecture, product, compute units, memory, clocks, runtime; a device
// class alone is not enough), the priced workload and the adapter's effective timing settings outside the table
// (weight format, route, schedules), each spelled once after the dispatch precedence resolved them.
struct LadderExecution {
    std::string_view backend, device;
    JointWorkload workload{};
    std::string_view settings{};
};
// The spelling of `execution` (empty when the backend, the device product, the prompt tokens, the context capacity,
// the lanes or the prefill chunk is not known: no ordinary timing row decides such a run).
[[nodiscard]] inline std::string ladder_execution(const LadderExecution& execution) {
    const auto& w = execution.workload;
    if (execution.backend.empty() || execution.device.empty() || w.prompt_tokens == 0 || w.context_capacity == 0 ||
        w.lanes == 0 || w.prefill_chunk == 0)
        return {};
    std::string out = "exec/v1 backend=" + std::string(execution.backend) + " device=" + std::string(execution.device);
    out += " prompt=" + std::to_string(w.prompt_tokens) + " context=" + std::to_string(w.context_capacity) +
           " lanes=" + std::to_string(w.lanes) + " chunk=" + std::to_string(w.prefill_chunk) +
           " prefill_rows=" + std::to_string(w.prefill_rows) + " decode_rows=" + std::to_string(w.decode_rows) +
           " verify_rows=" + std::to_string(w.verify_rows) + " generated=" + std::to_string(w.generated_tokens);
    if (!execution.settings.empty()) out += " settings=" + std::string(execution.settings);
    return out;
}

// The last value of `name` among repeated `key=value` settings: the dispatch precedence of every CLI parser that
// appends repeatable options (the reference of a timing receipt and the execution read the same value).
template <class Pairs>
[[nodiscard]] inline const std::string* last_setting(const Pairs& pairs, std::string_view name) {
    const std::string* out = nullptr;
    for (const auto& [key, value] : pairs) if (key == name) out = &value;
    return out;
}

// The implementation identity `list` (Lever::identity) declares for `value`: exact equality of the value, no prefix rule;
// empty when the list does not declare it.
// The empty value (a switch left unset: an empty builtin) is a value too and is declared like any other.
[[nodiscard]] constexpr std::string_view declared_identity(const Identities& list, std::string_view value) noexcept {
    for (const auto& item : list)
        if (item.value == value && !item.identity.empty()) return item.identity;
    return {};
}
// An identity token is non-empty and free of the reference spelling's separators.
[[nodiscard]] constexpr bool identity_token(std::string_view identity) noexcept {
    return !identity.empty() && identity.find_first_of("{},;|= ") == std::string_view::npos;
}
// First value of `lever` (its builtin, even an empty one, every non-empty column value and a non-empty `off`) without a
// well-formed declared identity; empty:
// the row is completely identified. A row type without an `identity` member is a compile-time error.
template <class Row>
[[nodiscard]] std::string identity_defect(const Row& lever) {
    const auto missing = [&](std::string_view value) { return !identity_token(declared_identity(lever.identity, value)); };
    // One identity per value: a second entry for the same value would be unreachable (declared_identity() reads the first).
    for (std::size_t i = 0; i < lever.identity.size(); ++i)
        for (std::size_t j = i + 1U; j < lever.identity.size(); ++j)
            if (!lever.identity[i].value.empty() && lever.identity[i].value == lever.identity[j].value &&
                lever.identity[i].identity != lever.identity[j].identity)
                return std::string(lever.name) + "=" + std::string(lever.identity[i].value) + " (two identities)";
    if (missing(lever.builtin)) return std::string(lever.name) + "=" + std::string(lever.builtin);
    for (const auto value : lever.value)
        if (!value.empty() && missing(value)) return std::string(lever.name) + "=" + std::string(value);
    if constexpr (requires { lever.off; })
        if (!lever.off.empty() && missing(lever.off)) return std::string(lever.name) + "=" + std::string(lever.off);
    return {};
}
// A cell the ladder judges: a lossy cell, or an exact schedule declared as timed (Lever::timed).
template <class Row>
[[nodiscard]] constexpr bool ladder_judged(const Row& lever) noexcept {
    if constexpr (requires { lever.timed; })
        if (lever.timed) return true;
    return lever.kind != decltype(lever.kind)::exact;
}

struct JointLadderScope {
    std::string_view device{}, reference{}, candidate_base{}, first_identity{}, second_identity{};
    JointWorkload workload{};
    std::string_view partner_value{};
    bool operator==(const JointLadderScope&) const = default;
    [[nodiscard]] constexpr bool complete() const {
        return !device.empty() && !reference.empty() && !candidate_base.empty() &&
               !first_identity.empty() && !second_identity.empty() && !partner_value.empty() &&
               workload.prompt_tokens != 0 && workload.context_capacity != 0 && workload.lanes != 0 &&
               workload.prefill_chunk != 0 && workload.prefill_rows != 0 && workload.decode_rows != 0;
    }
};

// One measured regime of one lossy cell: `speedup` = throughput of `tier` / throughput of the tier below on the same
// build, device and regime (an ABAB receipt), `noise` its stated margin.
struct BandEvidence {
    // Zero entries are unused. Timing points can just miss the nominal edge when CLI and serving count the
    // same fixture differently. Gate lengths record held-out workloads, independently of timing.
    std::array<std::uint64_t, 4> timed{};
    std::array<std::uint64_t, 4> gated{};
    std::string_view gate_scope{};
    // The lower edge has no timing within kBandEdgeTolerance tokens (EDGE-TIMING PENDING in the band inventory).
    bool lower_edge_unproven = false;
    // Edge state, data: true narrows an unproven band to its first in-band timing point. False (every row until the
    // device edge timings report) keeps the nominal band, so a pending edge never removes a tier default unmeasured.
    bool narrow_to_floor = false;
};

struct LadderRow {
    std::string_view lever;  // profile row name (or weight-load policy key)
    std::string_view value;  // the measured value of the cell (empty: every value)
    Name tier = Name::fast;  // fast | turbo: the column the value runs in
    Phase phase = Phase::decode;
    std::string_view device;  // measured device class (never empty)
    std::uint64_t min_context = 1, max_context = kNoContextBound;
    std::uint32_t min_lanes = 1, max_lanes = kNoLaneBound;
    double speedup = 0.0;
    double noise = 0.0;
    std::string_view receipt;
    // Incumbent grant: a cell a resolver already applied before the ladder rule, kept with the receipt that qualified
    // it (or made it provisional). It covers every regime, so the cell keeps running exactly where its own readiness
    // (unchanged by the ladder) admits it; where a timing row covers the regime, the timing row decides. Phase, device,
    // bands, speedup and noise are unused.
    bool incumbent = false;
    // Paired decision: names the partner lever. The row was measured on the combined operation of this cell and the
    // partner's cell (one complete workload against the effective arm of the tier below) and decides only that pair
    // (ladder_judge with the same `joint`). A plain judge never reads it, so two separately timed cells cannot admit
    // a pair. Empty: an ordinary single-cell row.
    std::string_view joint{};
    JointLadderScope joint_scope{};
    // The complete key of an ordinary row: the reference execution it was measured over (its denominator) and its
    // candidate, spelled as LadderArms::over() spells them: kLadderArmVersion; the execution outside the table
    // (Regime::execution: backend, device product, priced workload, effective settings); "lever=value{identity}" for
    // EVERY lever of the table in table order with its declared implementation identity (Lever::identity; builtin
    // values, exact schedules and explicit settings included): the levers before this one at the row's tier, this one
    // at the tier below, the later ones at their incumbent value of the row's tier; then "candidate=" this lever at
    // `value` with its identity; then ";tier=" and the row's tier. The row decides a cell only on exact equality of
    // that whole key (with lever, value,
    // tier, phase, device class and band: ladder_receipt_matches()); any other run is outside the row, which then has
    // no effect at all there. Paired rows bind through `joint_scope` instead.
    std::string_view over{};
    // The implementation identity the incumbent grant or band receipt covers (Lever::identity of the candidate value).
    // A changed implementation under the same value needs its own receipt. Required for grants and band receipts.
    std::string_view identity{};
    // Band receipt: the row was timed at representative workloads of its band (min_context..max_context) on its device
    // class and carries no literal `over`. An adapter that spells Regime::execution binds it per resolution with
    // bind_band_receipts(): `over` becomes that run's own reference (LadderArms::over) for the row's lever, value and
    // tier. The adapter owns the measured execution scope (product, pack and route). The binder requires a known
    // execution and the measured candidate identity above before matching the phase, device class and band.
    bool band_bound = false;
    BandEvidence band_evidence{};
};

// Bound the rounding allowance to three tokens (the same fixture can differ by a few tokens by entry point).
inline constexpr std::uint64_t kBandEdgeTolerance = 3;

[[nodiscard]] constexpr bool band_edge_timed(const LadderRow& row) noexcept {
    return std::any_of(row.band_evidence.timed.begin(), row.band_evidence.timed.end(), [&](std::uint64_t tokens) {
        return tokens != 0 && tokens <= row.max_context &&
               (tokens >= row.min_context ? tokens - row.min_context : row.min_context - tokens) <= kBandEdgeTolerance;
    });
}

// The configured band is retained for the device campaign. An unproven edge keeps the nominal band until its row sets
// narrow_to_floor; then it starts at the first winning measurement inside it. A band with missing or inconsistent
// evidence (no timing at a proven edge, no in-band timing for an unproven one) cannot match any request.
[[nodiscard]] constexpr std::uint64_t ladder_min_context(const LadderRow& row) noexcept {
    if (!row.band_bound) return row.min_context;
    if (!row.band_evidence.lower_edge_unproven) return band_edge_timed(row) ? row.min_context : 0;
    std::uint64_t first = 0;
    for (const auto tokens : row.band_evidence.timed)
        if (tokens != 0 && tokens >= row.min_context && tokens <= row.max_context && (first == 0 || tokens < first))
            first = tokens;
    return first == 0 || row.band_evidence.narrow_to_floor ? first : row.min_context;
}

// Receipts of incumbent grants for cells that ran before the ladder rule without a quality gate of their own.
inline constexpr std::string_view kProvisionalBeforeLadder =
    "provisional cell selected before the ladder rule; the row's source names the qualification in progress";
// An incumbent grant row (LadderRow::incumbent). `value` empty: the tier's cell. `identity`: the declared implementation
// identity of the granted value (LadderRow::identity).
[[nodiscard]] constexpr LadderRow incumbent_cell(std::string_view lever, std::string_view value, Name tier,
                                                 std::string_view receipt, std::string_view identity) noexcept {
    LadderRow row{};
    row.lever = lever;
    row.value = value;
    row.tier = tier;
    row.receipt = receipt;
    row.incumbent = true;
    row.identity = identity;
    return row;
}

[[nodiscard]] constexpr Name tier_below(Name tier) noexcept {
    return tier == Name::turbo ? Name::fast : Name::standard;
}
[[nodiscard]] constexpr bool valid_ladder_measurement(double speedup, double noise) noexcept {
    return speedup > 0.0 && speedup <= std::numeric_limits<double>::max() && noise >= 0.0 && noise < 1.0;
}
template <class Context, class Lanes>
[[nodiscard]] constexpr bool ladder_band(Context context, Context min_context, Context max_context,
                                         Lanes lanes, Lanes min_lanes, Lanes max_lanes) noexcept {
    return min_context > 0 && min_lanes > 0 && context >= min_context && context <= max_context &&
           lanes >= min_lanes && lanes <= max_lanes;
}
// Strictly faster than the tier below beyond the noise margin (a zero or unmeasured speedup never is).
[[nodiscard]] constexpr bool ladder_gain(double speedup, double noise) noexcept {
    return valid_ladder_measurement(speedup, noise) && speedup > 1.0 + noise;
}
[[nodiscard]] constexpr bool ladder_names(const LadderRow& row, std::string_view lever, std::string_view value,
                                          Name tier) noexcept {
    return row.lever == lever && (row.value.empty() || row.value == value) && row.tier == tier;
}
// The key a timing receipt must equal to decide one cell in one regime. `reference` is the complete ordinary key
// (LadderArms::over()); `joint` / `joint_scope` the complete paired key. Empty reference or incomplete scope: no
// receipt matches. `identity` is the declared implementation identity of `value` (Lever::identity): an incumbent grant
// keeps the cell only for exactly that identity (LadderRow::identity); empty: no grant applies.
struct ReceiptKey {
    std::string_view lever, value;
    Name tier = Name::fast;
    const Regime* regime = nullptr;
    std::string_view reference{};
    std::string_view joint{};
    const JointLadderScope* joint_scope = nullptr;
    std::string_view identity{};
};
// The one matching rule of every timing receipt (ordinary and paired): equality of the lever, the measured value, the
// tier, the phase and the device class, the regime inside the row's band, and EXACT equality of the complete key (the
// ordinary reference spelling, or the paired scope). No prefix, no partial key: a receipt that does not match has no
// effect on the cell (it neither admits, nor lowers, nor revokes an incumbent grant). Incumbent grants are not timing
// receipts and never match.
[[nodiscard]] constexpr bool ladder_receipt_matches(const LadderRow& row, const ReceiptKey& key) noexcept {
    if (row.incumbent || key.regime == nullptr || row.value.empty() || row.lever != key.lever ||
        row.value != key.value || row.tier != key.tier || row.joint != key.joint)
        return false;
    const auto& regime = *key.regime;
    if (row.phase != regime.phase || row.device.empty() || row.device != regime.device ||
        !ladder_band(regime.context, ladder_min_context(row), row.max_context, regime.lanes, row.min_lanes, row.max_lanes))
        return false;
    if (key.joint.empty())
        return !key.reference.empty() && !regime.execution.empty() && row.over == key.reference;
    return key.joint_scope != nullptr && key.joint_scope->complete() && row.joint_scope == *key.joint_scope;
}

enum class Rung : std::uint8_t { applies, unmeasured, within_noise, slower, incumbent };
[[nodiscard]] constexpr std::string_view rung_name(Rung rung) noexcept {
    return rung == Rung::applies ? "applies" : rung == Rung::unmeasured ? "unmeasured regime"
        : rung == Rung::within_noise ? "within noise" : rung == Rung::slower ? "slower" : "incumbent";
}
// No measurement decides the regime: the only rung --allow-pending may run as a benchmark arm. A measured slower or
// within-noise verdict is never overridden.
[[nodiscard]] constexpr bool rung_unmeasured(Rung rung) noexcept { return rung == Rung::unmeasured; }
// The cell runs in its own tier (a measured gain or an incumbent grant).
[[nodiscard]] constexpr bool rung_runs(Rung rung) noexcept { return rung == Rung::applies || rung == Rung::incumbent; }
struct LadderVerdict {
    Rung rung = Rung::unmeasured;
    const LadderRow* row = nullptr;  // the deciding row: the least favourable matching receipt, or the grant
    Name resolved = Name::standard;  // the tier whose cell the regime runs
};

// The verdict over the receipts matching `key` (ladder_receipt_matches()): applies only when every matching receipt
// shows a gain; with no matching receipt, an incumbent grant with a receipt keeps the cell (Rung::incumbent), and
// otherwise the cell resolves to the tier below (unmeasured). A grant is matched on lever, value, tier, receipt AND the
// declared implementation identity of the value (ReceiptKey::identity). Receipts that do not match are ignored entirely.
[[nodiscard]] constexpr LadderVerdict ladder_verdict(std::span<const LadderRow> rows, const ReceiptKey& key) noexcept {
    const LadderRow* worst = nullptr;
    const LadderRow* grant = nullptr;
    for (const auto& row : rows) {
        if (row.incumbent) {
            if (!grant && key.joint.empty() && !row.receipt.empty() && !row.identity.empty() &&
                row.identity == key.identity && ladder_names(row, key.lever, key.value, key.tier))
                grant = &row;
            continue;
        }
        if (!ladder_receipt_matches(row, key)) continue;
        if (row.receipt.empty() || !valid_ladder_measurement(row.speedup, row.noise))
            return {Rung::unmeasured, nullptr, tier_below(key.tier)};
        if (!worst || row.speedup - row.noise < worst->speedup - worst->noise) worst = &row;
    }
    if (!worst)
        return grant ? LadderVerdict{Rung::incumbent, grant, key.tier}
                     : LadderVerdict{Rung::unmeasured, nullptr, tier_below(key.tier)};
    if (ladder_gain(worst->speedup, worst->noise)) return {Rung::applies, worst, key.tier};
    const bool slower = worst->speedup < 1.0 - worst->noise;
    return {slower ? Rung::slower : Rung::within_noise, worst, tier_below(key.tier)};
}
// The verdict of `value` of `lever` in `tier` for `regime` over `arm`, the complete reference of the execution that runs
// (LadderArms::over(); required: an empty arm matches no timing receipt, so only an incumbent grant can run the cell)
// and `identity` the declared implementation identity of `value` (Lever::identity; only a grant of that identity runs).
[[nodiscard]] constexpr LadderVerdict ladder_judge(std::span<const LadderRow> rows, std::string_view lever,
                                                   std::string_view value, Name tier, const Regime& regime,
                                                   std::string_view arm, std::string_view identity) noexcept {
    return ladder_verdict(rows, {lever, value, tier, &regime, arm, {}, nullptr, identity});
}
// The verdict of a paired cell (LadderRow::joint): only receipts of the same pair with the same complete scope match;
// grants never admit a pair.
[[nodiscard]] constexpr LadderVerdict ladder_judge_pair(std::span<const LadderRow> rows, std::string_view lever,
                                                        std::string_view value, Name tier, const Regime& regime,
                                                        std::string_view joint,
                                                        const JointLadderScope& joint_scope) noexcept {
    if (joint.empty()) return {Rung::unmeasured, nullptr, tier_below(tier)};
    return ladder_verdict(rows, {lever, value, tier, &regime, {}, joint, &joint_scope});
}

// Legacy identity-variant API. These selections are explicit arithmetic levers, not permission for a separate
// serving identity. CLI and HTTP must resolve the same default and selected values for each model/backend/tier.
// A load, slot or cohort-width decision may change the schedule only with a bitwise identity proof.
// The legacy API below still admits an explicit named route (another reduction/grouping order or packed MoE pass): it
// runs only when requested by name ('--identity-variant ID[@VERSION]' or the variant's own option, a model TOML key, a
// serve config) THROUGH resolve(), which admits it on a declared backend under a tier its row admits. EXACT admits only
// a variant that is itself bitwise-defined (deterministic and batch invariant against its named reference) and declared
// as an EXACT identity variant (`exact_identity`). An admitted variant prints "identity <id>@<version>" in the receipt,
// the effective config and the startup line, and enters the rank fingerprint; with none admitted every output stays the
// existing default's, byte for byte. This compatibility rule does not qualify CLI/server equality; divergent defaults
// must be migrated with served A/B evidence. Engines read the decision only through Resolution::selected(id).
struct IdentityVariant {
    std::string_view id;                        // "inkling2-mini-moe-packed"
    unsigned version = 0;                       // bumped when the arithmetic changes; a request may pin "id@version"
    std::string_view changes;                   // what differs from the incumbent identity
    std::string_view reference;                 // what the variant is qualified against
    std::string_view backends;                  // comma list of the backends it is declared for ("hip", "hip,cuda")
    std::array<bool, kProfileColumns> tiers{};  // tiers that may admit it (default, fast, quality, turbo)
    bool exact_identity = false;                // bitwise-defined identity of its own: the only kind EXACT admits
    std::string_view option{};                  // CLI alias of '--identity-variant <id>' (empty: none)
    std::string_view toml_key{};                // model TOML / serve config key that requests it (empty: none)
    std::string_view source{};                  // required qualification receipt
};
inline constexpr std::string_view kIdentityVariantOption = "--identity-variant";

// Declaration rules, checked at compile time by the engine table and again by resolve().
[[nodiscard]] constexpr bool valid_variants(std::span<const IdentityVariant> variants,
                                            std::span<const Lever> levers = {}) {
    for (std::size_t i = 0; i < variants.size(); ++i) {
        const auto& v = variants[i];
        if (v.id.empty() || v.version == 0U || v.changes.empty() || v.reference.empty() || v.backends.empty() ||
            v.source.empty())
            return false;
        if (v.id.find_first_of("@, =") != std::string_view::npos) return false;
        // CLI aliases must not consume existing profile/lever options before their own parser.
        if (!v.option.empty()) {
            if (!v.option.starts_with("--") || v.option.size() == 2U ||
                v.option.find_first_of("@, =\t\r\n") != std::string_view::npos ||
                v.option == "--profile" || v.option == "--allow-pending" || v.option == kIdentityVariantOption ||
                v.option == turbo::kWeightsOption || v.option == turbo::kBitsOption || v.option == turbo::kPackOption)
                return false;
            for (const auto& lever : levers) if (v.option == lever.option) return false;
        }
        if (v.toml_key.find_first_of("@, =\t\r\n") != std::string_view::npos) return false;
        if (v.tiers[0] && !v.exact_identity) return false;  // EXACT admits only an exact identity variant
        bool admitted = false;
        for (const bool tier : v.tiers) admitted = admitted || tier;
        if (!admitted) return false;
        for (std::size_t j = 0; j < i; ++j) {
            const auto& w = variants[j];
            if (w.id == v.id || (!v.option.empty() && w.option == v.option) ||
                (!v.toml_key.empty() && w.toml_key == v.toml_key) ||
                v.id == w.option || w.id == v.option || v.id == w.toml_key || w.id == v.toml_key)
                return false;
        }
    }
    return true;
}

// One engine: its receipt tag, the allow-pending variable and the table.
struct Engine {
    std::string_view tag;                   // "deepseek41-pair", "glm-pair", "minimax-pair"
    std::string_view allow_env;             // e.g. STRIX_DS41_PROFILE_ALLOW_PENDING
    std::span<const Lever> table;
    bool quality_rows = false;              // false: 'quality' runs default with one notice
    std::string_view turbo_version{};       // versioned TURBO name ("ds41-turbo-v1"); empty: turbo refused
    turbo::WeightCaps weight_caps{};        // weight sources / widths this build can load under turbo
    std::span<const IdentityVariant> variants{};  // declared identity variants (empty: the incumbent identity only)
    std::span<const LadderRow> ladder{};    // measured regimes of the lossy cells (ladder rule above)
};

// Tier words: exact | fast | turbo; `exact` is the default column (spelled "default" in receipts, kept as an alias).
inline Name parse(std::string_view value) {
    if (value == "default" || value == "exact") return Name::standard;
    if (value == "fast") return Name::fast;
    if (value == "quality") return Name::quality;
    if (value == "turbo") return Name::turbo;
    throw std::invalid_argument("--profile must be exact (default), fast, quality or turbo (got '" + std::string(value) + "')");
}
inline bool is_name(std::string_view value) {
    return value == "default" || value == "exact" || value == "fast" || value == "quality" || value == "turbo";
}
inline const char* name(Name value) {
    switch (value) {
    case Name::standard: return "default";
    case Name::fast: return "fast";
    case Name::quality: return "quality";
    case Name::turbo: return "turbo";
    }
    return "unknown";
}

enum class Status { set, explicit_env, explicit_cli, builtin, pending, unqualified, off, provisional };
// Shared column admission; adapters retain asset, route and baseline policies.
inline Status admit_column(bool ready, bool runnable, bool allow_pending, bool provisional = false) {
    if (!ready && !((provisional || allow_pending) && runnable)) return Status::pending;
    return provisional ? Status::provisional : ready ? Status::set : Status::unqualified;
}
inline const char* status_name(Status value) {
    switch (value) {
    case Status::set: return "profile";
    case Status::explicit_env: return "explicit-env";
    case Status::explicit_cli: return "explicit-cli";
    case Status::builtin: return "builtin";
    case Status::pending: return "pending";
    case Status::unqualified: return "unqualified";
    case Status::off: return "off";
    case Status::provisional: return "provisional";
    }
    return "unknown";
}
inline const char* kind_name(Kind value) {
    switch (value) {
    case Kind::exact: return "exact";
    case Kind::bit_changing: return "bit-changing";
    case Kind::pack: return "pack";
    }
    return "unknown";
}

inline std::string json_string(std::string_view s) {
    std::string out = "\"";
    for (const char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20U) {
                char buf[8];
                std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
                out += buf;
            } else out += c;
        }
    }
    return out + "\"";
}

struct Entry {
    const Lever* lever = nullptr;
    std::string value;   // effective value (explicit, profile or builtin)
    Status status = Status::builtin;
    std::string target;  // pending: the declared target that is not applied
    bool cli_override = false;  // adapter provenance; does not enter the rank fingerprint
    std::string reason;         // adapter admission explanation
    std::string requested{};    // what the profile column or the explicit setting asked for (empty: builtin)
    std::string ladder{};       // effective tier decision for this regime
    bool ladder_lowered = false;  // the ladder ran a lower cell than the column's (its route line is a notice)
    std::string ladder_reference{};  // the reference the deciding timing rows must name (LadderRow::over)
};

// Adapters with their own app/option schemas bind the same qualification rule to their value columns.
template <class Row>
[[nodiscard]] bool explicit_value_qualified(const Row& lever, std::string_view value, std::string_view off,
                                             std::span<const bool> provisional = {}, std::string_view same_bits = {}) {
    if (lever.kind != Kind::bit_changing) return true;
    if (value == lever.builtin || (!off.empty() && value == off)) return true;
    const bool in_class = bit_class_member(same_bits, value);
    for (std::size_t c = 0; c < lever.value.size(); ++c)
        if (lever.ready[c] && (provisional.empty() || !provisional[c]) && !lever.value[c].empty() &&
            (value == lever.value[c] || (in_class && bit_class_member(same_bits, lever.value[c]))))
            return true;
    return false;
}

template <class ResolvedEntry>
[[nodiscard]] Status qualification_status(const ResolvedEntry& entry, std::string_view off,
                                           std::span<const bool> provisional = {}, std::string_view same_bits = {}) {
    using EntryStatus = decltype(entry.status);
    if constexpr (requires { EntryStatus::provisional; }) {
        if (entry.status == EntryStatus::provisional) return Status::provisional;
    }
    if (entry.status == EntryStatus::unqualified ||
        ((entry.status == EntryStatus::explicit_cli || entry.status == EntryStatus::explicit_env) &&
         !explicit_value_qualified(*entry.lever, entry.value, off, provisional, same_bits)))
        return Status::unqualified;
    return Status::builtin;
}

[[nodiscard]] inline Status qualification_status(const Entry& entry) {
    return qualification_status(entry, entry.lever->off, entry.lever->provisional, entry.lever->same_bits);
}

template <class ResolvedEntry, class StatusOf>
[[nodiscard]] std::string qualification_summary(const std::vector<ResolvedEntry>& entries, StatusOf status_of) {
    std::string provisional, unqualified;
    for (const auto& entry : entries) {
        const auto status = status_of(entry);
        auto* rows = status == Status::provisional ? &provisional : status == Status::unqualified ? &unqualified : nullptr;
        if (rows != nullptr) *rows += (rows->empty() ? "" : ",") + std::string(entry.lever->name);
    }
    std::string out;
    if (!provisional.empty()) out = "qualification in progress: " + provisional;
    if (!unqualified.empty()) out += (out.empty() ? "" : "; ") + std::string("unqualified: ") + unqualified;
    return out.empty() ? "qualified" : out;
}

[[nodiscard]] inline std::string qualification_summary(const std::vector<Entry>& entries) {
    return qualification_summary(entries, [](const Entry& entry) { return qualification_status(entry); });
}

struct Inputs;
struct Resolution;
inline Resolution resolve(const Engine&, const Inputs&);

struct Resolution {
    const Engine* engine = nullptr;
    Name profile = Name::standard;
    bool allow_pending = false;
    std::vector<Entry> entries;
    turbo::Weights weights{};             // weight toggle (native outside turbo)
    // Bit-changing switches outside the lever table that the engine applied and can attest ("name=value"). The tier
    // notice names them and then never claims the exact column.
    std::vector<std::string> engine_overrides;
    // Identity variants resolve() admitted, in declaration order; each points into engine->variants.
    [[nodiscard]] const std::vector<const IdentityVariant*>& identity_variants() const {
        if (!admitted_.empty() && (engine != admitted_engine_ || profile != admitted_profile_))
            throw std::logic_error("the resolution was changed after admission (admitted for " +
                                   std::string(admitted_engine_->tag) + " --profile " +
                                   std::string(admitted_profile_ == Name::standard ? "exact" : name(admitted_profile_)) +
                                   ")");
        return admitted_;
    }

    [[nodiscard]] bool incumbent_identity() const { return identity_variants().empty(); }
    // True when the declared variant `id` was requested and admitted. An id the engine does not declare is a defect of
    // the caller (a route switch with no declaration), never a quiet false; so is a resolution whose engine or tier
    // was changed after admission (the admitted set belongs to the engine and the tier resolve() checked).
    [[nodiscard]] bool selected(std::string_view id) const {
        const auto& variants = identity_variants();
        const IdentityVariant* declared = nullptr;
        if (engine != nullptr)
            for (const auto& variant : engine->variants) if (variant.id == id) declared = &variant;
        if (declared == nullptr)
            throw std::logic_error("identity variant '" + std::string(id) + "' is not declared by " +
                                   (engine != nullptr ? std::string(engine->tag) : std::string("an unresolved profile")));
        for (const auto* variant : variants) if (variant == declared) return true;
        return false;
    }
    // "identity incumbent" or "identity <id>@<version>[ <id>@<version>...]": the startup line's identity field.
    [[nodiscard]] std::string identity() const {
        const auto& variants = identity_variants();
        if (variants.empty()) return "identity incumbent";
        std::string out = "identity";
        for (const auto* variant : variants)
            out += ' ' + std::string(variant->id) + '@' + std::to_string(variant->version);
        return out;
    }

    [[nodiscard]] bool is_turbo() const { return profile == Name::turbo; }
    // Versioned product name: the turbo name under turbo, else the profile name.
    [[nodiscard]] std::string versioned() const {
        return is_turbo() ? std::string(engine->turbo_version) : std::string(name(profile));
    }
    // Namespace the cache-key builders append (empty outside turbo: legacy namespaces unchanged).
    [[nodiscard]] std::string cache_namespace() const {
        return is_turbo() ? turbo::cache_namespace(engine->turbo_version, fingerprint(), weights.identity()) : std::string();
    }

    [[nodiscard]] const Entry* find(std::string_view key) const {
        for (const auto& entry : entries) if (entry.lever->name == key) return &entry;
        return nullptr;
    }
    [[nodiscard]] bool quality_fallback() const { return profile == Name::quality && !engine->quality_rows; }
    // True when the profile column changed at least one lever (set, provisional or unqualified); false when every lever
    // is builtin, pending, off or explicit. Display only: not part of the fingerprint.
    [[nodiscard]] bool profile_applied() const {
        for (const auto& entry : entries)
            if (entry.status == Status::set || entry.status == Status::provisional || entry.status == Status::unqualified)
                return true;
        return false;
    }
    // An explicit setting (CLI or environment) of a bit-changing lever keeps the run qualified only when its value is
    // the builtin, the lever's off value, or the value of a column the table marks ready and not provisional (or a
    // member of that value's same_bits class): the table's qualification data, one rule for every engine.
    [[nodiscard]] static bool explicit_value_qualified(const Entry& entry) {
        const Lever& lever = *entry.lever;
        return ::strix::profile::explicit_value_qualified(lever, entry.value, lever.off, lever.provisional, lever.same_bits);
    }
    // "qualified"; else "qualification in progress: <rows>" for provisional columns and/or "unqualified: <rows>"
    // when --allow-pending ran pending columns or an explicit setting chose a bit-changing value no qualified column
    // has, separated by "; ".
    [[nodiscard]] std::string qualification() const {
        return qualification_summary(entries);
    }
    // One stderr line per pending, provisional or unqualified row (and the quality fallback note).
    [[nodiscard]] std::vector<std::string> notices() const {
        std::vector<std::string> out;
        const std::string head = std::string(engine->tag) + " profile " + name(profile) + ": ";
        if (quality_fallback()) out.push_back(head + "no quality rows for this model, using default");
        if (!weights.honoured()) out.push_back(head + weights.describe().substr(0, weights.describe().size() - 1U));
        for (const auto& entry : entries) if (entry.ladder_lowered) out.push_back(entry.ladder);
        for (const auto& entry : entries) if (entry.status == Status::provisional)
            out.push_back(head + std::string(entry.lever->name) + "=" + entry.value + " is bit-changing; qualification in progress");
        for (const auto& entry : entries) if (entry.status == Status::pending)
            out.push_back(head + std::string(entry.lever->name) + " pending " + entry.target + " (" + entry.value + " runs)");
        for (const auto& entry : entries) if (entry.status == Status::unqualified)
            out.push_back(head + std::string(entry.lever->name) + "=" + entry.value + " UNQUALIFIED (--allow-pending, benchmarks only)");
        return out;
    }
    // {"profile":...,"engine":...,"qualification":...,"levers":{...},"pending":[...],"unqualified":[...],"provisional":[...]}
    [[nodiscard]] std::string json() const {
        const auto& variants = identity_variants();
        std::string levers, pending, unqualified, provisional;
        for (const auto& entry : entries) {
            const auto& lever = *entry.lever;
            if (!levers.empty()) levers += ',';
            levers += json_string(lever.name) + ":{\"value\":" + json_string(entry.value) +
                ",\"status\":" + json_string(status_name(entry.status)) +
                ",\"switch\":" + json_string(lever.env.empty() ? lever.option : lever.env) +
                ",\"kind\":" + json_string(kind_name(lever.kind)) +
                ",\"source\":" + json_string(lever.source);
            if (!entry.ladder.empty()) levers += ",\"ladder\":" + json_string(entry.ladder);
            levers += "}";
            if (entry.status == Status::pending) {
                if (!pending.empty()) pending += ',';
                pending += "{\"lever\":" + json_string(lever.name) + ",\"target\":" + json_string(entry.target) +
                    ",\"runs\":" + json_string(entry.value) + "}";
            }
            if (qualification_status(entry) == Status::unqualified) {
                if (!unqualified.empty()) unqualified += ',';
                unqualified += json_string(lever.name);
            }
            if (entry.status == Status::provisional) {
                if (!provisional.empty()) provisional += ',';
                provisional += json_string(lever.name);
            }
        }
        std::string out = "{\"profile\":" + json_string(name(profile)) + ",\"engine\":" + json_string(engine->tag) +
            ",\"qualification\":" + json_string(qualification());
        if (is_turbo())
            out += ",\"turbo\":{\"version\":" + json_string(engine->turbo_version) +
                ",\"weights_requested\":" + json_string(weights.requested_text()) +
                ",\"weights\":" + json_string(weights.identity()) + ",\"weights_origin\":" + json_string(weights.origin) +
                ",\"weights_reason\":" + json_string(weights.reason) +
                ",\"cache_namespace\":" + json_string(cache_namespace()) + "}";
        if (quality_fallback()) out += ",\"note\":\"quality: no quality rows for this model, using default\"";
        if (!variants.empty()) {  // absent for the incumbent identity: legacy JSON unchanged
            out += ",\"identity\":[";
            for (std::size_t i = 0; i < variants.size(); ++i) {
                const auto& variant = *variants[i];
                out += std::string(i == 0U ? "" : ",") + "{\"id\":" + json_string(variant.id) +
                    ",\"version\":" + std::to_string(variant.version) +
                    ",\"exact_identity\":" + (variant.exact_identity ? "true" : "false") +
                    ",\"reference\":" + json_string(variant.reference) + ",\"changes\":" + json_string(variant.changes) +
                    ",\"source\":" + json_string(variant.source) + "}";
            }
            out += "]";
        }
        return out + ",\"levers\":{" + levers + "},\"pending\":[" + pending + "],\"unqualified\":[" + unqualified +
            "],\"provisional\":[" + provisional + "]}";
    }
    // Tier word as the CLI spells it (exact | fast | quality | turbo).
    [[nodiscard]] const char* word() const { return profile == Name::standard ? "exact" : name(profile); }
    // The one summary line every engine prints: "profile <word> levers <name=value ...|none>". It lists each lever
    // whose effective value differs from this build's builtin (profile column or explicit setting), in table order.
    [[nodiscard]] std::string summary() const {
        std::string out = std::string("profile ") + word() + " levers";
        bool any = false;
        for (const auto& entry : entries) {
            if (entry.value == entry.lever->builtin) continue;
            out += ' ' + std::string(entry.lever->name) + '=' + entry.value;
            any = true;
        }
        return any ? out : out + " none";
    }
    // A requested FAST or TURBO tier that applied no QUALIFIED lever is never left silent: one line states the count
    // (0 levers qualified on this device) and what runs instead. Levers that ran anyway (an unqualified or provisional
    // column row under --allow-pending, or an explicit CLI / environment value that differs from the builtin) are named
    // as overrides and the line then does not claim the exact column. Empty for exact/quality or when at least one
    // profile row is qualified. Kept out of notices() so the per-row notice contract is unchanged.
    [[nodiscard]] std::string tier_notice() const {
        const auto& variants = identity_variants();
        if (profile != Name::fast && profile != Name::turbo) return {};
        std::size_t pending = 0, qualified = 0;
        std::string overrides;
        for (const auto& entry : entries) {
            pending += entry.status == Status::pending ? 1U : 0U;
            qualified += entry.status == Status::set ? 1U : 0U;
            const bool ran_anyway = entry.status == Status::provisional || entry.status == Status::unqualified;
            const bool explicit_value = (entry.status == Status::explicit_cli || entry.status == Status::explicit_env) &&
                                        entry.value != entry.lever->builtin;
            if (ran_anyway || explicit_value)
                overrides += (overrides.empty() ? "" : ",") + std::string(entry.lever->name) + '=' + entry.value;
        }
        if (qualified != 0U) return {};
        for (const auto& extra : engine_overrides) overrides += (overrides.empty() ? "" : ",") + extra;
        for (const auto* variant : variants)  // an admitted variant never runs "the exact column"
            overrides += (overrides.empty() ? "identity " : ",identity ") + std::string(variant->id) + '@' +
                std::to_string(variant->version);
        // TURBO may change weight representation even when its speed-lever column is empty.
        // Never describe a low-bit run as identical to the native-weight EXACT column.
        const bool nonnative_weights = is_turbo() && weights.effective != turbo::Source::native;
        std::string run_description;
        if (!overrides.empty()) {
            run_description = "runs with unqualified or explicit overrides (" + overrides + "), not the exact column";
            if (nonnative_weights) run_description += "; uses " + weights.identity() + " weights";
        } else {
            run_description = nonnative_weights ? "uses " + weights.identity() + " weights (not the exact column)"
                                                : "runs the exact column";
        }
        std::string out = std::string(engine->tag) + " profile " + word() + ": " +
            (profile == Name::fast ? "FAST" : "TURBO") + " requested, 0 levers qualified on this device; " + run_description;
        if (pending != 0U) out += " (" + std::to_string(pending) + " pending)";
        return out;
    }
    // Receipt lines in print order: summary, then the tier notice when there is one.
    [[nodiscard]] std::vector<std::string> receipt_lines() const {
        std::vector<std::string> out{summary()};
        if (auto notice = tier_notice(); !notice.empty()) out.push_back(std::move(notice));
        if (!incumbent_identity()) out.push_back(identity());  // the incumbent's receipt lines are unchanged
        return out;
    }
    // Line form for line-oriented --print-effective-config outputs (appended after the engine's own lines).
    [[nodiscard]] std::string describe() const {
        const auto& variants = identity_variants();
        std::string out = std::string("profile ") + name(profile) + "\nprofile qualification " + qualification() + "\n";
        if (quality_fallback()) out += "profile note quality: no quality rows for this model, using default\n";
        if (is_turbo())
            out += "profile version " + versioned() + "\n" + weights.describe() + "turbo cache namespace " + cache_namespace() + "\n";
        for (const auto& entry : entries) {
            const auto& lever = *entry.lever;
            out += "profile lever " + std::string(lever.name) + " " + entry.value + " (" + status_name(entry.status) + ", " +
                kind_name(lever.kind) + ", " + std::string(lever.env.empty() ? lever.option : lever.env) + ")";
            if (entry.status == Status::pending) out += " pending " + entry.target;
            if (is_turbo() && !entry.requested.empty() && entry.requested != entry.value)
                out += " requested " + entry.requested;
            if (is_turbo() && !entry.reason.empty()) out += " reason " + entry.reason;
            out += "\n";
            if (!entry.ladder.empty()) out += entry.ladder + "\n";
        }
        for (const auto* variant : variants)
            out += "identity " + std::string(variant->id) + '@' + std::to_string(variant->version) + " (" +
                (variant->exact_identity ? "exact identity" : "bit-changing") + "; reference " +
                std::string(variant->reference) + "; changes " + std::string(variant->changes) + ")\n";
        return out;
    }
    // FNV-1a 64 of what the ranks of a pair must share: the profile name, the qualification and every lever's
    // effective value (not how it was set: an explicit value equal to the profile's is the same run).
    [[nodiscard]] std::uint64_t fingerprint() const {
        const auto& variants = identity_variants();
        std::uint64_t h = 1469598103934665603ULL;
        const auto mix = [&h](std::string_view s) {
            h = host_fnv::byte(host_fnv::text(s, h), 0xFFU);  // preserve legacy seed and separator
        };
        mix(name(profile));
        mix(qualification());
        if (is_turbo()) {  // the versioned name and the weight identity: never equal to a non-turbo run
            mix(engine->turbo_version);
            mix(weights.identity());
        }
        for (const auto& entry : entries) {
            if (entry.lever->omit_builtin_fingerprint && entry.value == entry.lever->builtin) continue;
            mix(entry.lever->name);
            mix(entry.value);
        }
        for (const auto* variant : variants) {  // nothing mixed for the incumbent: legacy fingerprints unchanged
            mix("identity");
            mix(variant->id);
            mix(std::to_string(variant->version));
        }
        return h;
    }

private:
    friend Resolution resolve(const Engine&, const Inputs&);
    static std::vector<const IdentityVariant*> resolve_variants(const Engine&, const Inputs&, const Resolution&);
    std::vector<const IdentityVariant*> admitted_;  // written only by resolve()
    const Engine* admitted_engine_ = nullptr;       // the engine and tier the set was admitted for
    Name admitted_profile_ = Name::standard;
};

// Rank agreement: both ranks of a pair must run the same resolved profile (a bit-changing row on one rank only
// desynchronizes the exchanges). `exchange` swaps 32 host bytes with the peer (the app's control channel, both ranks
// at the same point of the startup); a mismatch throws, naming both profiles.
using HostExchange = std::function<void(std::span<const std::byte>, std::span<std::byte>)>;
inline void check_peer(const Resolution& resolution, const HostExchange& exchange) {
    struct Hello {
        std::uint32_t magic = 0x31465250U;  // "PRF1"
        std::uint32_t reserved = 0;
        std::uint64_t fingerprint = 0;
        char profile[16] = {};
    };
    static_assert(sizeof(Hello) == 32U);
    Hello mine, peer;
    mine.fingerprint = resolution.fingerprint();
    std::snprintf(mine.profile, sizeof mine.profile, "%s", name(resolution.profile));
    std::array<std::byte, sizeof(Hello)> out{}, in{};
    std::memcpy(out.data(), &mine, sizeof mine);
    exchange(out, in);
    std::memcpy(&peer, in.data(), sizeof peer);
    peer.profile[sizeof peer.profile - 1U] = '\0';
    if (peer.magic != mine.magic)
        throw std::runtime_error("product profile check: the peer did not send its profile (different engine build?)");
    if (peer.fingerprint != mine.fingerprint) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%016llx vs %016llx", static_cast<unsigned long long>(mine.fingerprint),
                      static_cast<unsigned long long>(peer.fingerprint));
        throw std::runtime_error(std::string("product profile differs between the ranks: this rank '") + name(resolution.profile) +
            "' (" + resolution.qualification() + "), peer '" + peer.profile + "' (fingerprint " + buf +
            "); run both ranks with the same --profile, --allow-pending and lever switches");
    }
}

using Lookup = std::function<const char*(const char*)>;
inline const char* environment(const Lookup& env, std::string_view key);

// Shared defaults row; adapters retain application-specific admission and exports.
struct DefaultItem {
    const char* name;
    const char* env;
    const char* on;
    const char* off;
    const char* receipt;
    bool app_sets = true;
    bool in_baseline = true;
};
enum class DefaultSource { on, baseline, env, cli };
struct DefaultValue {
    std::string value;
    DefaultSource source = DefaultSource::on;
};
inline DefaultValue resolve_default(const DefaultItem& row, bool baseline, const Lookup& env,
                                    std::optional<std::string_view> cli = std::nullopt) {
    if (cli) return {std::string(*cli), DefaultSource::cli};
    if (const char* value = environment(env, row.env))
        return {value, DefaultSource::env};
    if (baseline && row.in_baseline) return {row.off, DefaultSource::baseline};
    return {row.on, DefaultSource::on};
}
using Pairs = std::vector<std::pair<std::string, std::string>>;
// Where an identity-variant request came from (receipt provenance only: every origin is admitted by the same rule).
enum class VariantOrigin { cli, toml, serve };
struct VariantRequest {
    std::string id;        // declared id (request_variant canonicalizes aliases)
    unsigned version = 0;  // 0: any; else must equal the declared version
    VariantOrigin origin = VariantOrigin::cli;
};

struct Inputs {
    std::optional<Name> profile;  // --profile; none: default
    Pairs cli;                    // levers set by their own CLI option: {lever name, value}
    Lookup env;                   // environment lookup; empty: the process environment
    bool allow_pending = false;   // --allow-pending
    turbo::WeightsRequest weights{};  // --turbo-weights / --turbo-weight-bits / --turbo-weights-pack
    Regime regime{};
    std::vector<VariantRequest> variants{};  // explicit identity-variant requests (request_variant)
    std::string backend{};        // backend the run resolves to ("hip", "cuda"); required when a variant is requested
};

// Records an explicit identity-variant request: `token` is "id" or "id@version" from any origin, the variant's option
// (cli) or its TOML key (toml). Every entry point (app CLI, take_options, server launcher, model TOML) goes through
// this one function, so the same request selects the same variant everywhere. An undeclared token is refused here.
inline void request_variant(const Engine& engine, Inputs& inputs, std::string_view token, VariantOrigin origin) {
    if (!valid_variants(engine.variants, engine.table))
        throw std::logic_error(std::string(engine.tag) + ": malformed identity variant declaration");
    std::string_view id = token;
    unsigned version = 0;
    if (const auto at = token.find('@'); at != std::string_view::npos) {
        id = token.substr(0, at);
        const auto digits = token.substr(at + 1U);
        for (const char c : digits) {
            if (c < '0' || c > '9' || version > 100000U)
                throw std::invalid_argument(std::string(engine.tag) + ": bad identity variant version in '" +
                                            std::string(token) + "'");
            version = version * 10U + static_cast<unsigned>(c - '0');
        }
        if (digits.empty() || version == 0U)
            throw std::invalid_argument(std::string(engine.tag) + ": bad identity variant version in '" +
                                        std::string(token) + "'");
    }
    const IdentityVariant* match = nullptr;
    for (const auto& variant : engine.variants)
        if (variant.id == id || (origin == VariantOrigin::cli && !variant.option.empty() && variant.option == id) ||
            (origin == VariantOrigin::toml && !variant.toml_key.empty() && variant.toml_key == id))
            match = &variant;
    if (match == nullptr) {
        std::string declared;
        for (const auto& variant : engine.variants)
            declared += (declared.empty() ? "" : ", ") + std::string(variant.id) + '@' + std::to_string(variant.version);
        throw std::invalid_argument(std::string(engine.tag) + ": unknown identity variant '" + std::string(token) +
                                    "' (declared: " + (declared.empty() ? "none" : declared) + ")");
    }
    inputs.variants.push_back({std::string(match->id), version, origin});
}

inline const char* environment(const Lookup& env, std::string_view key) {
    const std::string k(key);
    if (env) return env(k.c_str());
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv: read once at startup, before any thread
#endif
    return std::getenv(k.c_str());
#ifdef _MSC_VER
#pragma warning(pop)
#endif
}

inline constexpr char kPresetEnv[] = "STRIX_INTERNAL_MODEL_PRESET";

// Model presets automatically admit qualified cells. Provisional experiments require an explicit opt-in.
inline Status admit_profile_column(bool ready, bool runnable, bool allow_pending, bool provisional,
                                   bool preset) {
    if (preset && !allow_pending && provisional) return Status::pending;
    return admit_column(ready, runnable, allow_pending, provisional);
}

inline bool model_preset() {
    const char* value = environment({}, kPresetEnv);
    return value && std::string_view(value) == "1";
}

// The statuses the profile itself applies (an explicit setting is the app's own).
inline bool applied(Status status) {
    return status == Status::set || status == Status::unqualified || status == Status::provisional;
}

// Column readiness is supplied by the adapter (including asset and quality qualification).
template <class Row>
[[nodiscard]] inline Status column_admission(const Row& lever, unsigned column, bool allow_pending, bool preset) {
    bool provisional = false;
    if constexpr (requires { lever.provisional[column]; }) provisional = lever.provisional[column];
    return admit_profile_column(lever.ready[column], lever.runnable[column], allow_pending, provisional, preset);
}

struct LadderCell {
    std::string_view value;
    Name tier = Name::standard;
    std::string route;
    bool benchmark = false;
    std::string reference{};  // LadderArms::over() of the highest judged step (what its timing rows must name)
    bool measured = false;    // a timing row decided the cell (applies, slower or within noise)
};

// Version of the reference spelling (LadderRow::over). Bumped when the spelling rule changes, so a receipt written
// under another rule never matches.
inline constexpr std::string_view kLadderArmVersion = "ladder-arm/v4";

// The effective executions of a regime (ladder_arms()): the cell every lever of the table runs in EXACT, FAST and
// TURBO, in table order (builtin values, exact schedule levers and explicit settings included), the incumbent value
// of every lever per tier, and the execution outside the table (Regime::execution).
struct LadderArms {
    std::string execution;
    std::vector<std::string> levers;                    // table order
    std::vector<Identities> identities;                 // Lever::identity per lever (table order)
    std::array<std::vector<std::string>, 3> values;     // standard, fast, turbo; values[t][i] = what lever i runs in t
    std::array<std::vector<std::string>, 3> incumbent;  // what lever i runs in t without any ordinary timing row
    std::array<std::vector<LadderCell>, 3> cells;       // the resolved cell of lever i in t

    [[nodiscard]] static constexpr std::size_t slot(Name tier) noexcept {
        return tier == Name::turbo ? 2U : tier == Name::fast ? 1U : 0U;
    }
    [[nodiscard]] std::size_t index(std::string_view lever) const {
        for (std::size_t i = 0; i < levers.size(); ++i)
            if (levers[i] == lever) return i;
        return levers.size();
    }
    // "lever=value{identity}" with the identity the row declares for that value; false (nothing appended) when the
    // row declares none (an explicit alternative the table does not identify): the arm is not completely identified.
    [[nodiscard]] bool spell(std::string& out, std::size_t i, std::string_view value) const {
        const auto identity = declared_identity(identities[i], value);
        if (!identity_token(identity)) return false;
        out.append(levers[i]).append("=").append(value).append("{").append(identity).append("}");
        return true;
    }
    // The complete key of an ordinary receipt of `candidate` for `lever` in `tier` (LadderRow::over):
    // kLadderArmVersion; the execution outside the table (Regime::execution); then every lever of the table in table
    // order with its declared identity: the levers before `lever` at the value they run in `tier`, `lever` at its
    // resolved cell of the tier below, every later lever at its incumbent value in `tier` (incumbent grants, exact
    // schedules and explicit settings included, no timing receipt); then "candidate=" and the candidate with its
    // identity. A timed change therefore extends the complete execution that runs under it, and a later timed change
    // extends that one in turn: the cells a tier exports are one measured chain from its incumbent execution. Empty
    // (no receipt matches) when the execution is not known, `lever` is not in the table these arms were built from, or
    // any spelled value has no declared identity. Ends with ";tier=<tier>": a key compares one pair of executions for
    // one tier only.
    [[nodiscard]] std::string over(Name tier, std::string_view lever, std::string_view candidate) const {
        const auto at = index(lever);
        const auto& upper = values[slot(tier)];
        const auto& lower = values[slot(tier_below(tier))];
        const auto& later = incumbent[slot(tier)];
        if (execution.empty() || at == levers.size() || upper.size() < at || lower.size() != levers.size() ||
            later.size() != levers.size())
            return {};
        std::string out(kLadderArmVersion);
        out.append(";").append(execution).append(";");
        for (std::size_t i = 0; i < levers.size(); ++i) {
            if (i != 0) out.append(",");
            if (!spell(out, i, i < at ? upper[i] : i == at ? lower[i] : later[i])) return {};
        }
        out.append(";candidate=");
        if (!spell(out, at, candidate)) return {};
        out.append(";tier=").append(name(tier));
        return out;
    }
    // The whole execution of `tier` (receipts and fixtures): the execution outside the table, every lever at `tier`
    // with its identity; empty when a value has no declared identity.
    [[nodiscard]] std::string arm(Name tier) const {
        std::string out(kLadderArmVersion);
        out.append(";").append(execution).append(";");
        const auto& at = values[slot(tier)];
        for (std::size_t i = 0; i < levers.size() && i < at.size(); ++i) {
            if (i != 0) out.append(",");
            if (!spell(out, i, at[i])) return {};
        }
        return out;
    }
};
// The execution segment of a reference spelling (LadderArms::over()); empty when `over` is not a current, complete
// spelling (version, execution, levers, candidate).
[[nodiscard]] inline std::string_view ladder_reference_execution(std::string_view over) {
    if (!over.starts_with(kLadderArmVersion) || over.size() <= kLadderArmVersion.size() ||
        over[kLadderArmVersion.size()] != ';')
        return {};
    over.remove_prefix(kLadderArmVersion.size() + 1U);
    const auto end = over.find(';');
    if (end == 0 || end == std::string_view::npos || over.find(";candidate=", end) == std::string_view::npos) return {};
    return over.substr(0, end);
}
// No explicit setting (adapters without CLI / env overrides of lossy levers).
struct NoExplicitSetting {
    std::optional<std::string_view> operator()(const auto&) const noexcept { return std::nullopt; }
};

// The cell `lever` runs in `tier` over `lower`, its resolved cell of the tier below (the same cell the tier below
// exports, an unqualified benchmark cell included: a fallback never drops what the reference was spelled over).
template <class Row, class Admission>
[[nodiscard]] inline LadderCell ladder_cell(const Row& lever, std::span<const LadderRow> rows, Name tier,
                                            const Regime& regime, const Admission& admit, bool allow_pending,
                                            const LadderArms& arms, LadderCell lower) {
    const auto column = value_column(lever, tier);
    const auto wanted = lever.value[column];
    // A cell spelling the builtin runs without a qualification of its own (resolve() treats it as builtin).
    const bool eligible = !wanted.empty() && (wanted == lever.builtin || applied(admit(column)));
    if (!ladder_judged(lever) || (tier != Name::fast && tier != Name::turbo))
        return {eligible ? wanted : lever.builtin, tier, {}};

    lower.measured = false;  // the lower step's own verdict is not this tier's
    if (!wanted.empty() && !eligible) {
        // TURBO nests FAST: a TURBO proposal its column does not admit (pending, or not runnable in this build) leaves
        // the cell FAST resolved, whatever value the proposal spells (a companion setting of the proposal stays
        // pending in its own row); it never drops an admitted FAST member. FAST falls back to the builtin.
        if (tier == Name::turbo) {
            if (lower.value != lever.builtin)
                lower.route = " turbo=" + std::string(wanted) + " refused by column admission" + lower.route;
            return lower;
        }
        return {lever.builtin, Name::standard, {}};
    }
    if (!eligible || (tier == Name::turbo && column == static_cast<unsigned>(Name::fast)) ||
        wanted == lower.value) return lower;
    std::string step = " " + std::string(name(tier)) + "=" + std::string(wanted) + " ";
    // The complete key of the execution that runs under this cell (the lower cell of this lever included): only a
    // receipt of exactly that key decides; any other receipt has no effect here.
    const std::string reference = arms.over(tier, lever.name, wanted);
    std::string_view identity{};
    if constexpr (requires { lever.identity; }) identity = declared_identity(lever.identity, wanted);
    const auto verdict = ladder_judge(rows, lever.name, wanted, tier, regime, reference, identity);
    step += rung_name(verdict.rung);
    if (verdict.row) {
        if (!verdict.row->incumbent) {
            char ratio[64];
            std::snprintf(ratio, sizeof ratio, " x%.3f noise %.3f", verdict.row->speedup, verdict.row->noise);
            step += ratio;
        }
        step += " (" + std::string(verdict.row->receipt) + ")";
    }
    const bool measured = verdict.row != nullptr && !verdict.row->incumbent;
    if (rung_runs(verdict.rung)) return {wanted, tier, step, false, reference, measured};
    // The explicit benchmark arm can collect the evidence the default needs where no measurement decides the regime.
    // It never claims qualification, and never runs a cell measured slower or within noise.
    if (tier == Name::fast && allow_pending && lever.runnable[column] && rung_unmeasured(verdict.rung))
        return {wanted, tier, step + " (unqualified benchmark opt-in)", true, reference};
    lower.route = step + lower.route;
    lower.reference = reference;
    lower.measured = measured;
    return lower;
}

// The effective EXACT, FAST and TURBO executions in `regime`: the cell of every lever of `table` (in table order,
// exact schedule levers and builtin values included) per tier. `admit_of(lever, column)` is the adapter's column
// admission and `explicit_of(lever)` its explicit CLI / env setting of the lever (std::nullopt: none; a repeated
// option is its last value, the dispatch precedence: last_setting()), which runs in every tier; the "off" alias is
// spelled as the lever's off value. Tiers resolve bottom up; a tier falls back to the cells the tier below resolved.
// Within a tier the incumbent execution comes first (no timing receipt: grants, exact schedules, explicit settings);
// then the cells in table order, each judged over the complete key of the execution it would extend
// (LadderArms::over()). A receipt that does not match a key has no effect, so a cell without a matching receipt runs
// exactly its incumbent value and the chain is final after one judged pass. `allow_pending` is the run's
// --allow-pending: an unqualified benchmark cell it runs is part of the reference, and the tier above falls back to it.
// A table read with timing receipts must declare every row's identities (identity_defect(); std::logic_error at load).
template <class Row, class AdmitOf, class ExplicitOf = NoExplicitSetting>
[[nodiscard]] inline LadderArms ladder_arms(std::span<const Row> table, std::span<const LadderRow> rows,
                                            const Regime& regime, const AdmitOf& admit_of,
                                            const ExplicitOf& explicit_of = {}, bool allow_pending = false) {
    LadderArms arms;
    arms.execution = std::string(regime.execution);
    // A timing row of a lever this table does not have decides nothing and asks no identities of it.
    const bool timed = std::any_of(rows.begin(), rows.end(), [&](const LadderRow& row) {
        return !row.incumbent && std::any_of(table.begin(), table.end(), [&](const auto& lever) { return lever.name == row.lever; });
    });
    for (const auto& lever : table) {
        if (timed)
            if (const auto defect = identity_defect(lever); !defect.empty())
                throw std::logic_error("profile ladder: " + defect + " has no declared implementation identity");
        arms.levers.emplace_back(lever.name);
        arms.identities.emplace_back(lever.identity);
    }
    std::vector<LadderRow> grants;
    for (const auto& row : rows) if (row.incumbent) grants.push_back(row);
    const auto n = arms.levers.size();
    for (const Name tier : {Name::standard, Name::fast, Name::turbo}) {
        const auto t = LadderArms::slot(tier);
        auto& values = arms.values[t];
        auto& cells = arms.cells[t];
        const auto cell_of = [&](std::size_t i, std::span<const LadderRow> judged) -> LadderCell {
            const auto& lever = table[i];
            if (const auto set = explicit_of(lever)) {
                std::string_view value = *set;
                // The "off" alias switches the lever off: it runs the lever's own off value.
                if constexpr (requires { lever.off; })
                    if (value == "off" && !lever.off.empty()) value = lever.off;
                return {value, tier, {}};
            }
            const auto lower = tier == Name::standard ? LadderCell{} : arms.cells[LadderArms::slot(tier_below(tier))][i];
            // Legacy adapters still build the common arms to resolve FAST. They cannot select TURBO.
            if (tier == Name::turbo && static_cast<unsigned>(tier) >= std::size(lever.value)) return lower;
            return ladder_cell(lever, judged, tier, regime, [&](unsigned c) { return admit_of(lever, c); },
                               allow_pending, arms, lower);
        };
        const auto run = [&](std::span<const LadderRow> judged) {
            values.clear();
            cells.clear();
            for (std::size_t i = 0; i < n; ++i) {
                auto cell = cell_of(i, judged);
                values.emplace_back(cell.value);
                cells.push_back(std::move(cell));
            }
        };
        run(grants);  // the incumbent execution: no timing receipt decides any cell
        arms.incumbent[t] = values;
        run(rows);
    }
    return arms;
}

// A synthetic execution for probing the band and candidate identity checks without an adapter.
inline constexpr std::string_view kBandProbeExecution = "exec/v1 band-probe";
// Binds the band receipts of `rows` to the run's own reference (LadderRow::band_bound). `storage` owns the spellings
// (a deque: views stay valid) and must outlive the rows. Rows with an unknown execution, unidentified reference or
// changed candidate identity stay unbound and decide nothing. A row already bound is re-bound when its reference
// changed: the reference of a later lever extends the cells the earlier rows admitted (the ladder chain), so a caller
// repeats bind + arms until nothing changes (resolve() does). Returns true when a row's reference changed.
inline bool bind_band_receipts(std::vector<LadderRow>& rows, const LadderArms& arms, std::deque<std::string>& storage) {
    bool changed = false;
    for (auto& row : rows) {
        if (!row.band_bound || row.incumbent || !row.joint.empty()) continue;
        const auto at = arms.index(row.lever);
        const bool measured_candidate = identity_token(row.identity) && at < arms.identities.size() &&
            declared_identity(arms.identities[at], row.value) == row.identity;
        auto over = measured_candidate ? arms.over(row.tier, row.lever, row.value) : std::string{};
        if (over == row.over) continue;
        if (over.empty()) row.over = {};
        else {
            storage.push_back(std::move(over));
            row.over = storage.back();
        }
        changed = true;
    }
    return changed;
}

// Binds the band receipts until the chain settles: a chain of k band-bound levers settles in at most k + 1 passes (each
// pass binds the next lever over the cells the previous pass admitted); the bound keeps a defective table from looping.
// Every resolver that builds ladder arms calls this one loop (`build_arms` rebuilds the arms over the bound rows).
template <class BuildArms>
void settle_band_receipts(std::vector<LadderRow>& rows, LadderArms& arms, std::deque<std::string>& storage,
                          BuildArms&& build_arms) {
    for (std::size_t pass = 0; pass <= rows.size() && bind_band_receipts(rows, arms, storage); ++pass)
        arms = build_arms();
}

// Route line of one lever in one regime: 'strix-ladder/v1 <lever> <phase> ctx=<n> lanes=<n> dev=<device>:<steps>
// -> <tier>=<value>'. Says which cell runs and why every higher cell does not.
[[nodiscard]] inline std::string ladder_route(std::string_view lever, const Regime& regime, const LadderCell& cell) {
    std::string out = "strix-ladder/v1 " + std::string(lever) +
                      (regime.phase == Phase::prefill ? " prefill ctx=" : " decode ctx=") + std::to_string(regime.context) +
                      " lanes=" + std::to_string(regime.lanes) + " dev=" + std::string(regime.device) + ":";
    out += cell.route;
    out += " -> " + std::string(name(cell.tier)) + "=" + std::string(cell.value);
    return out;
}

// Resolve a profile-owned FAST/TURBO cell before any adapter can export it.
template <class Row, class Admission>
inline LadderCell select_profile_cell(const Row& lever, std::span<const LadderRow> rows, Name tier,
                                      const Regime& regime, const Admission& admit, bool allow_pending,
                                      const LadderArms& arms) {
    const auto column = value_column(lever, tier);
    if (!ladder_judged(lever) || (tier != Name::fast && tier != Name::turbo))
        return {lever.value[column], tier, {}};
    // The cell the arms resolved (the one their references were spelled over). A lever outside those arms is judged
    // over its own lower cells, and no ordinary row matches it (LadderArms::over()).
    const auto at = arms.index(lever.name);
    const auto& resolved = arms.cells[LadderArms::slot(tier)];
    auto cell = at < resolved.size() ? resolved[at] : [&] {
        auto lower = ladder_cell(lever, rows, Name::standard, regime, admit, allow_pending, arms, {});
        if (tier == Name::turbo) lower = ladder_cell(lever, rows, Name::fast, regime, admit, allow_pending, arms, lower);
        return ladder_cell(lever, rows, tier, regime, admit, allow_pending, arms, lower);
    }();
    if (!cell.route.empty()) cell.route = ladder_route(lever.name, regime, cell);
    return cell;
}

inline void apply_ladder(Entry& entry, std::span<const LadderRow> rows, Name tier, const Regime& regime,
                         const auto& admit, bool allow_pending, const LadderArms& arms) {
    const auto& lever = *entry.lever;
    if (!ladder_judged(lever) || (tier != Name::fast && tier != Name::turbo)) return;
    auto cell = select_profile_cell(lever, rows, tier, regime, admit, allow_pending, arms);
    entry.ladder = std::move(cell.route);
    entry.ladder_reference = std::move(cell.reference);
    entry.ladder_lowered = !entry.ladder.empty() && cell.tier != tier;
    if (cell.benchmark) {
        entry.value = cell.value;
        entry.status = Status::unqualified;
        entry.target.clear();
        return;
    }
    // The ladder kept the value the profile column resolved: its admission status stands as resolved.
    if (cell.value == entry.value) return;
    entry.value = cell.value;
    if (cell.value != lever.builtin || (cell.tier == Name::turbo && !entry.ladder.empty())) {
        entry.status = admit(value_column(lever, cell.tier));
        entry.target.clear();
    } else if (!entry.ladder.empty()) {
        entry.status = Status::pending;
        entry.target = entry.requested;
    } else if (applied(entry.status)) {
        entry.status = Status::builtin;
    }
}

// Identity-variant admission, the one rule for every engine and entry point: each request names a declared variant
// (and its version when pinned), the run's backend is one the variant declares, and the tier column admits it (EXACT
// only for an exact identity variant). Anything else refuses the run before any work; nothing falls back silently.
inline std::vector<const IdentityVariant*> Resolution::resolve_variants(const Engine& engine, const Inputs& inputs,
                                                                         const Resolution& out) {
    if (!valid_variants(engine.variants, engine.table))
        throw std::logic_error(std::string(engine.tag) + ": malformed identity variant declaration");
    for (const auto& request : inputs.variants) {  // a hand-built request must still name a declared variant
        bool declared = false;
        for (const auto& variant : engine.variants) declared = declared || request.id == variant.id;
        if (!declared)
            throw std::invalid_argument(std::string(engine.tag) + ": unknown identity variant '" + request.id + "'");
    }
    std::vector<const IdentityVariant*> admitted;
    for (const auto& variant : engine.variants) {
        bool requested = false;
        for (const auto& request : inputs.variants) {
            if (request.id != variant.id) continue;
            if (request.version != 0U && request.version != variant.version)
                throw std::invalid_argument(std::string(engine.tag) + ": identity variant " + request.id + '@' +
                                            std::to_string(request.version) + " is not built (this build declares @" +
                                            std::to_string(variant.version) + ")");
            requested = true;
        }
        if (!requested) continue;
        const std::string label = std::string(engine.tag) + ": identity variant " + std::string(variant.id) + '@' +
                                  std::to_string(variant.version);
        if (inputs.backend.empty()) throw std::invalid_argument(label + " requires a resolved backend");
        if (!bit_class_member(variant.backends, inputs.backend))
            throw std::invalid_argument(label + " is not declared for backend " + inputs.backend + " (declared: " +
                                        std::string(variant.backends) + ")");
        const auto column = static_cast<std::size_t>(out.profile);
        if (!variant.tiers[column] || (out.profile == Name::standard && !variant.exact_identity)) {
            std::string admits;
            for (std::size_t c = 0; c < kProfileColumns; ++c)
                if (variant.tiers[c])
                    admits += (admits.empty() ? "" : ",") + std::string(c == 0U ? "exact" : name(static_cast<Name>(c)));
            throw std::invalid_argument(label + " is not admitted under --profile " + out.word() + " (admitted: " +
                                        admits + ")");
        }
        admitted.push_back(&variant);
    }
    return admitted;
}

inline Resolution resolve(const Engine& engine, const Inputs& inputs) {
    Resolution out;
    out.engine = &engine;
    out.profile = inputs.profile.value_or(Name::standard);
    const char* allow = engine.allow_env.empty() ? nullptr : environment(inputs.env, engine.allow_env);
    out.allow_pending = inputs.allow_pending || (allow != nullptr && std::string_view(allow) == "1");
    if (out.is_turbo() && engine.turbo_version.empty())
        throw std::invalid_argument(std::string(engine.tag) + ": --profile turbo is not declared for this engine");
    out.weights = turbo::resolve_weights(out.is_turbo(), inputs.weights,
        [&inputs](const char* key) { return environment(inputs.env, key); }, engine.weight_caps);
    // 'quality' without quality rows is the default column; an empty turbo cell is the fast cell.
    const auto profile_column = out.quality_fallback() ? Name::standard : out.profile;
    const bool preset = model_preset();
    const auto effective_env_value = [](const Lever& lever, const char* raw) -> std::string_view {
        if (!lever.env_flag_match) return raw;
        return envcfg::parse_flag(raw, lever.builtin == "1", *lever.env_flag_match) ? "1" : "0";
    };
    const auto admit = [&](const Lever& lever, unsigned column) {
        return column_admission(lever, column, out.allow_pending, preset);
    };
    std::vector<LadderRow> ladder(engine.ladder.begin(), engine.ladder.end());
    const auto build_arms = [&] { return ladder_arms(engine.table, std::span<const LadderRow>(ladder), inputs.regime,
        admit,
        [&](const Lever& lever) -> std::optional<std::string_view> {
            if (const auto* cli = last_setting(inputs.cli, lever.name)) return std::string_view(*cli);
            if (const char* env = lever.env.empty() ? nullptr : environment(inputs.env, lever.env))
                return effective_env_value(lever, env);
            return std::nullopt;
        }, out.allow_pending); };
    auto arms = build_arms();
    std::deque<std::string> band_references;  // band receipts bind to this run's own reference
    settle_band_receipts(ladder, arms, band_references, build_arms);
    for (const auto& lever : engine.table) {
        const auto column = value_column(lever, profile_column);
        const std::string* cli = last_setting(inputs.cli, lever.name);  // the value the arms spelled (dispatch precedence)
        const char* env = lever.env.empty() ? nullptr : environment(inputs.env, lever.env);
        const auto wanted = lever.value[column];
        Entry entry{&lever, std::string(lever.builtin), Status::builtin, {}, false, {}, std::string(wanted)};
        if (cli != nullptr || env != nullptr) {  // a set variable is explicit even when empty (the app judges it)
            entry.requested = cli != nullptr ? *cli : std::string(env);
            if (cli != nullptr) entry.value = *cli;
            else entry.value = effective_env_value(lever, env);
            entry.status = cli != nullptr ? Status::explicit_cli : Status::explicit_env;
            if ((!lever.off.empty() && entry.value == lever.off) || entry.value == "off") entry.status = Status::off;
        } else if (wanted.empty() || wanted == lever.builtin) {
            // builtin (the default column always is: default exports nothing)
        } else {
            entry.status = admit(lever, column);
            if (entry.status == Status::pending) entry.target = std::string(lever.target.empty() ? wanted : lever.target);
            else entry.value = std::string(wanted);
        }
        if (cli == nullptr && env == nullptr)
            apply_ladder(entry, ladder, profile_column, inputs.regime, [&](unsigned c) {
                return admit(lever, c);
            }, out.allow_pending, arms);
        out.entries.push_back(std::move(entry));
    }
    out.admitted_ = Resolution::resolve_variants(engine, inputs, out);
    out.admitted_engine_ = &engine;
    out.admitted_profile_ = out.profile;
    return out;
}

// First defect of an engine's ladder rows (empty: none). With any timing receipt, every row of the table declares its
// implementation identities (identity_defect()). Every row names a judged lever of the table (lossy, or an exact
// schedule declared timed), a fast or turbo
// column whose cell is the row's value, a device, a receipt, a finite positive speedup, a noise margin in
// [0, 1) and ordered bands starting at context >= 1.
[[nodiscard]] inline std::string ladder_defect(const Engine& engine) {
    const std::string tag(engine.tag);
    // A table read with timing receipts declares the implementation identity of every value of every row.
    if (std::any_of(engine.ladder.begin(), engine.ladder.end(), [](const LadderRow& row) { return !row.incumbent; }))
        for (const auto& lever : engine.table)
            if (const auto defect = identity_defect(lever); !defect.empty())
                return tag + "/" + defect + ": no declared implementation identity";
    for (const auto& row : engine.ladder) {
        const std::string where = tag + "/" + std::string(row.lever) + "=" + std::string(row.value) + " " +
                                  std::string(name(row.tier)) + " (" + std::string(row.receipt) + ")";
        const Lever* lever = nullptr;
        for (const auto& candidate : engine.table)
            if (candidate.name == row.lever) lever = &candidate;
        if (lever == nullptr) return where + ": ladder row names no lever of the table";
        if (!ladder_judged(*lever)) return where + ": ladder row on an exact lever (not a timed schedule)";
        if (row.tier != Name::fast && row.tier != Name::turbo) return where + ": ladder rows are fast or turbo";
        const auto cell = lever->value[value_column(*lever, row.tier)];
        if (cell.empty() || (!row.value.empty() && row.value != cell))
            return where + ": ladder row value is not the tier's cell";
        if (row.incumbent) {
            if (row.receipt.empty()) return where + ": incumbent grant without a receipt";
            if (row.identity.empty()) return where + ": incumbent grant without the implementation identity it granted";
            if (!row.joint.empty()) return where + ": incumbent grant cannot be a paired row";
            continue;
        }
        if (!row.joint.empty()) {
            if (row.value.empty() || !row.joint_scope.complete()) return where + ": paired row requires a complete execution scope";
            const auto same = [&](const Lever& candidate) { return candidate.name == row.joint; };
            const auto partner = std::find_if(engine.table.begin(), engine.table.end(), same);
            if (partner == engine.table.end() || partner->kind == Kind::exact || row.joint == row.lever)
                return where + ": paired row names no other lossy lever of the table";
        }
        if (row.joint.empty() && row.value.empty()) return where + ": timing row names no measured value";
        if (row.band_bound && (!row.joint.empty() || !identity_token(row.identity)))
            return where + ": band receipt requires a single measured candidate and its implementation identity";
        if (row.joint.empty() && !row.band_bound && ladder_reference_execution(row.over).empty())
            return where + ": timing row without a versioned reference execution (LadderArms::over spelling)";
        if (row.joint.empty() && !row.band_bound && !row.over.ends_with(std::string(";tier=") + name(row.tier)))
            return where + ": timing row key names another tier";
        if (row.device.empty() || row.receipt.empty()) return where + ": ladder row without device or receipt";
        if (!valid_ladder_measurement(row.speedup, row.noise)) return where + ": invalid speedup or noise";
        if (row.min_context < 1 || row.min_context > row.max_context || row.min_lanes < 1 || row.min_lanes > row.max_lanes)
            return where + ": ladder row band empty or unordered";
        if (row.band_bound && (ladder_min_context(row) == 0 ||
                               row.band_evidence.lower_edge_unproven == band_edge_timed(row)))
            return where + ": band has no timing receipt at its lower edge or an explicitly narrowed measured floor";
    }
    return {};
}

// Probe regimes of an engine's ladder: every row's band corners and the regime just outside each edge, per device and
// phase, on the execution the row was measured on. A walker resolves every lossy lever of every tier there (strix ladder host test). Incumbent grants cover every
// regime: they add a fixed grid (both phases, unknown / short / long contexts, one and eight lanes, unknown device and
// the two device classes of this repository).
[[nodiscard]] inline std::vector<Regime> ladder_probes(const Engine& engine) {
    std::vector<Regime> out;
    if (std::any_of(engine.ladder.begin(), engine.ladder.end(), [](const LadderRow& row) { return row.incumbent; }))
        for (const auto phase : {Phase::prefill, Phase::decode})
            for (const std::uint64_t context : {0ULL, 1ULL, 4096ULL, 8192ULL, 8193ULL, 32768ULL, 131072ULL})
                for (const std::uint32_t lanes : {1U, 8U})
                    for (const std::string_view device : {std::string_view{}, std::string_view{"gfx1151"},
                                                          std::string_view{"sm_120"}})
                        out.push_back({phase, context, lanes, device});
    for (const auto& row : engine.ladder) {
        if (row.incumbent) continue;
        const auto floor = ladder_min_context(row);
        if (floor == 0) continue;
        const std::uint64_t contexts[] = {row.min_context, floor, floor - 1, row.max_context, row.min_context - 1,
                                          row.max_context == kNoContextBound ? row.max_context : row.max_context + 1};
        const std::uint32_t lanes[] = {row.min_lanes, row.max_lanes, row.min_lanes - 1,
                                       row.max_lanes == kNoLaneBound ? row.max_lanes : row.max_lanes + 1};
        for (const auto context : contexts)
            for (const auto lane : lanes)
                if (lane >= 1) out.push_back({row.phase, context, lane, row.device, (row.band_bound ? kBandProbeExecution : ladder_reference_execution(row.over))});
    }
    return out;
}

// FAST gate ledger: one passing FAST quality-gate receipt for one lever value of one engine, kept as data next to the
// engine's profile table. The ledger, not the row's cell, is what says a value passed its gate, so a checker can see
// "gated but not selected": the resolved `fast` set must apply every ledger value, and every bit-changing fast cell
// marked qualified must have a ledger receipt (or belong to a word-qualified FAST bundle, strix/fast_profile.hpp
// Bundle::profile_receipt). Engines qualified per regime keep a regime ledger of RegimeGate rows (below).
struct FastGate {
    std::string_view lever;    // row name
    std::string_view value;    // the fast value the gate passed
    std::string_view receipt;  // gate receipt (public docs anchor)
};

// Regime ledger row: a FastGate receipt that holds only for prompts of [min_prompt_tokens, max_prompt_tokens] tokens
// (the longest prefill of the run). Inside a row's range the lever's fast cell reads as qualified; outside every row the
// engine table's own ready / provisional flags apply. The default lower bound 1 keeps "prompt length not known before
// load" (0) outside the row; a row with lower bound 0 also covers it. An empty value covers every value of the cell.
// Engines that qualify per weights / route / device / topology hold this row beside those keys
// (strix/profiles/qwen_dense_cuda.hpp, strix/profiles/hy4.hpp); the range and value test is regime_covers, shared.
inline constexpr std::uint64_t kNoPromptBound = ~std::uint64_t{0};
struct RegimeGate {
    std::string_view lever;
    std::string_view value;
    std::uint64_t min_prompt_tokens = 1;
    std::uint64_t max_prompt_tokens = kNoPromptBound;
    std::string_view receipt;
};
[[nodiscard]] constexpr bool regime_covers(const RegimeGate& row, std::string_view lever, std::string_view value,
                                           std::uint64_t prompt_tokens) {
    return row.lever == lever && (row.value.empty() || row.value == value) && prompt_tokens >= row.min_prompt_tokens &&
           prompt_tokens <= row.max_prompt_tokens;
}
[[nodiscard]] inline const RegimeGate* regime_gate(std::span<const RegimeGate> rows, std::string_view lever,
                                                   std::string_view value, std::uint64_t prompt_tokens) {
    for (const auto& row : rows)
        if (regime_covers(row, lever, value, prompt_tokens)) return &row;
    return nullptr;
}
// "" or " / prompt >= N" / " / prompt <= N" / " / prompt N..M": the range part of a receipt line.
[[nodiscard]] inline std::string regime_range_text(const RegimeGate& row) {
    const bool lo = row.min_prompt_tokens > 1, hi = row.max_prompt_tokens != kNoPromptBound;
    if (lo && hi) return " / prompt " + std::to_string(row.min_prompt_tokens) + ".." + std::to_string(row.max_prompt_tokens);
    if (lo) return " / prompt >= " + std::to_string(row.min_prompt_tokens);
    if (hi) return " / prompt <= " + std::to_string(row.max_prompt_tokens);
    return {};
}

// First ledger defect of `engine` (empty: none). `covered`: row names qualified by a word-qualified bundle receipt.
inline std::string fast_gate_defect(const Engine& engine, std::span<const FastGate> ledger,
                                    std::span<const std::string_view> covered = {}) {
    const std::string tag(engine.tag);
    for (const auto& gate : ledger) {
        const std::string where = tag + "/" + std::string(gate.lever) + "=" + std::string(gate.value);
        if (gate.receipt.empty()) return where + ": FAST gate ledger row without a receipt";
        const auto fast = static_cast<unsigned>(Name::fast);
        const Lever* lever = nullptr;
        for (const auto& row : engine.table)
            if (row.name == gate.lever) lever = &row;
        if (!lever) return where + ": FAST gate receipt names no profile row";
        if (lever->value[fast] != gate.value || !applied(column_admission(*lever, fast, false, false)))
            return where + ": gated but not registered for --profile fast (" + std::string(gate.receipt) + ")";
    }
    constexpr auto fast = static_cast<unsigned>(Name::fast);
    for (const auto& row : engine.table) {
        if (row.kind == Kind::exact || !row.ready[fast] || row.provisional[fast] || row.value[fast].empty() ||
            row.value[fast] == row.builtin)
            continue;
        bool receipted = false;
        for (const auto& gate : ledger) receipted = receipted || (gate.lever == row.name && gate.value == row.value[fast]);
        for (const auto name : covered) receipted = receipted || name == row.name;
        if (!receipted)
            return tag + "/" + std::string(row.name) + ": bit-changing fast cell marked qualified without a FAST gate receipt";
    }
    return {};
}

// Variables an adapter exports after resolution: every applied profile column, plus every row whose value the
// resolver canonicalized (`canonical`: e.g. a FAST bundle's expanded members, strix/fast_profile.hpp), so the
// execution readers see the same values as the receipt, the rank agreement and the cache identity.
template <class Keys>
inline std::vector<std::pair<std::string, std::string>> exports(const Resolution& resolution, const Keys& canonical) {
    std::vector<std::pair<std::string, std::string>> out;
    // Pure proposal: inspection may describe a pending weight request. Execution uses apply().
    for (const auto& e : resolution.entries)
        if (!e.lever->env.empty() && (applied(e.status) || canonical.contains(std::string(e.lever->env))))
            out.emplace_back(std::string(e.lever->env), e.value);
    // Under turbo: the cache namespace every cache-key builder appends (strix/turbo.hpp).
    if (resolution.is_turbo()) out.emplace_back(std::string(turbo::kCacheNamespaceEnv), resolution.cache_namespace());
    return out;
}

using Export = std::function<bool(const char*, const char*)>;
inline bool export_environment(const char* key, const char* value) {
#ifdef _WIN32
    return _putenv_s(key, value) == 0;
#else
    return ::setenv(key, value, 1) == 0;
#endif
}
// Exports the switches the profile sets (status set / unqualified / provisional, env levers only; CLI-only levers:
// profile_args).
// Call right after argument parsing, before anything reads (and caches) those switches. Default exports nothing.
// Under turbo it also exports the cache namespace (turbo::kCacheNamespaceEnv) that every cache-key builder of the
// process appends, and refuses a weight request this build cannot honour (after --print-effective-config printed it).
inline void apply(const Resolution& resolution, const Export& put = export_environment) {
    resolution.weights.require_honoured();
    for (const auto& entry : resolution.entries) {
        const auto& lever = *entry.lever;
        if (lever.env.empty() || !applied(entry.status)) continue;
        if (!put(std::string(lever.env).c_str(), entry.value.c_str()))
            throw std::runtime_error("cannot export profile switch " + std::string(lever.env));
    }
    if (resolution.is_turbo() &&
        !put(std::string(turbo::kCacheNamespaceEnv).c_str(), resolution.cache_namespace().c_str()))
        throw std::runtime_error("cannot export " + std::string(turbo::kCacheNamespaceEnv));
}

// Options the profile adds to the app's arguments for CLI-only levers it sets ({option, value} each; put them
// in front of the user's arguments). Default adds none.
inline std::vector<std::string> profile_args(const Resolution& resolution) {
    std::vector<std::string> out;
    for (const auto& entry : resolution.entries) {
        const auto& lever = *entry.lever;
        if (!lever.env.empty() || lever.option.empty() || !applied(entry.status)) continue;
        out.emplace_back(lever.option);
        out.push_back(entry.value);
    }
    return out;
}

// Removes '--profile default|fast|quality|turbo' ('--profile=NAME'), '--allow-pending' and the turbo weight options
// (--turbo-weights, --turbo-weight-bits, --turbo-weights-pack), '--identity-variant ID[@V]' and each declared variant's
// own option (request_variant) from `args` into `inputs` and records
// (without removing) the table's lever options as explicit CLI values. Any other '--profile' token ('--profile' followed
// by something that is not a profile name, '--profile=OTHER') stays in `args` untouched, for the app's own parser: an
// older meaning (GLM kernel timing, MiniMax timed session) or a value of another option keeps working. Only a second
// profile name is an error.
inline void take_options(const Engine& engine, std::vector<std::string>& args, Inputs& inputs) {
    if (!valid_variants(engine.variants, engine.table))
        throw std::logic_error(std::string(engine.tag) + ": malformed identity variant declaration");
    std::vector<std::string> kept;
    kept.reserve(args.size());
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        const auto equal = a.find('=');
        const std::string key = a.substr(0, equal);
        if (a == "--allow-pending") { inputs.allow_pending = true; continue; }
        if (key == kIdentityVariantOption) {  // consumed: the app never sees a variant switch, only the resolution
            std::string value;
            if (equal != std::string::npos) value = a.substr(equal + 1U);
            else if (i + 1 < args.size()) value = args[++i];
            if (value.empty()) throw std::invalid_argument(key + " requires a value");
            request_variant(engine, inputs, value, VariantOrigin::cli);
            continue;
        }
        bool alias = false;
        for (const auto& variant : engine.variants) alias = alias || (!variant.option.empty() && a == variant.option);
        if (alias) {
            request_variant(engine, inputs, a, VariantOrigin::cli);
            continue;
        }
        if (key == turbo::kWeightsOption || key == turbo::kBitsOption || key == turbo::kPackOption) {
            std::string value;
            if (equal != std::string::npos) value = a.substr(equal + 1U);
            else if (i + 1 < args.size()) value = args[++i];
            if (value.empty()) throw std::invalid_argument(key + " requires a value");
            auto& slot = key == turbo::kWeightsOption ? inputs.weights.source :
                key == turbo::kBitsOption ? inputs.weights.bits : inputs.weights.pack;
            slot = value;
            continue;
        }
        if (key == "--profile") {
            std::string value;
            if (equal != std::string::npos) value = a.substr(equal + 1U);
            else if (i + 1 < args.size()) value = args[i + 1];
            // '--profile WORD' with a plain word that is not a profile name is a typo ('--profile fastt'); a path, a
            // NAME:... session, an option ('--profile --x') or a '--profile=...' token belongs to the app.
            if (equal == std::string::npos && !value.empty() && !is_name(value) && value.find_first_of("-:/.=") == std::string::npos)
                throw std::invalid_argument("--profile must be exact (default), fast, quality or turbo (got '" + value + "')");
            if (is_name(value)) {
                if (inputs.profile) throw std::invalid_argument("duplicate --profile");
                inputs.profile = parse(value);
                if (equal == std::string::npos) ++i;
                continue;
            }
        }
        kept.push_back(a);
        const Lever* lever = nullptr;
        for (const auto& row : engine.table) if (!row.option.empty() && key == row.option) lever = &row;
        if (lever == nullptr) continue;
        // The app parses its own option too (it stays in args); the last spelling wins, as in the app's parser.
        std::string value;
        if (equal != std::string::npos) value = a.substr(equal + 1U);
        else if (i + 1 < args.size()) kept.push_back(value = args[++i]);
        inputs.cli.emplace_back(std::string(lever->name), value);
    }
    args = std::move(kept);
}

// Compatibility adapter for a catalogue projection. Native resolution still
// owns app masks, assets, explicit override precedence and receipt formats.
struct EffectiveCell {
    std::string_view row, value;
    bool provisional = false;
};

template <class Row, class Resolve>
[[nodiscard]] auto resolve_effective_profile(Name tier, std::span<const Row> legacy,
                                             std::span<const EffectiveCell> cells,
                                             std::vector<Row>& storage, Resolve&& native) {
    // Publish the rows only after resolution succeeds. The result borrows
    // their addresses until storage is changed or destroyed.
    std::vector<Row> projected(legacy.begin(), legacy.end());
    const auto column = static_cast<std::size_t>(tier);
    for (const auto& cell : cells) {
        Row* match = nullptr;
        for (auto& row : projected) {
            if (row.name == cell.row) {
                match = &row;
                break;
            }
        }
        if (match == nullptr || column >= match->value.size())
            throw std::invalid_argument("unbound profile cell: " + std::string(cell.row));
        if constexpr (requires { match->provisional[column]; }) {
            match->provisional[column] = cell.provisional;
        } else if (cell.provisional) {
            throw std::invalid_argument("native profile cannot represent provisional cell: " + std::string(cell.row));
        }
        match->value[column] = cell.value;
        match->ready[column] = !cell.provisional;
        match->runnable[column] = true;
    }
    auto result = std::forward<Resolve>(native)(std::span<const Row>(projected));
    storage.swap(projected);
    return result;
}

inline std::string hex_identity(std::uint64_t value) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(value));
    return buf;
}

// Numerical compatibility and execution provenance are deliberately independent.
// Empty fields mean unknown, never a proof that two executions are compatible.
struct NumericalIdentity {
    std::string arithmetic_contract;
    std::string model_sha256;
    std::string pack_hash;
    std::string table_hash;
    std::string state_format;
    std::string prefill_lineage;
    std::string reduction_plan;

    [[nodiscard]] std::uint64_t id() const {
        auto h = host_fnv::field(host_fnv::offset64, "strix-numerical-identity/v1");
        for (const auto* value : {&arithmetic_contract, &model_sha256, &pack_hash, &table_hash,
                                  &state_format, &prefill_lineage, &reduction_plan})
            h = host_fnv::field(h, *value);
        return h;
    }
    [[nodiscard]] std::string json() const {
        return "{\"id\":" + json_string(hex_identity(id())) +
            ",\"arithmetic_contract\":" + json_string(arithmetic_contract) +
            ",\"model_sha256\":" + json_string(model_sha256) +
            ",\"pack_hash\":" + json_string(pack_hash) +
            ",\"table_hash\":" + json_string(table_hash) +
            ",\"state_format\":" + json_string(state_format) +
            ",\"prefill_lineage\":" + json_string(prefill_lineage) +
            ",\"reduction_plan\":" + json_string(reduction_plan) + "}";
    }
};

struct ExecutionReceipt {
    std::string engine;
    NumericalIdentity identity;
    std::string implementation;
    std::string build;
    std::string kernel;
    std::string device;
    std::string topology;
    std::string command;  // canonical invocation when supplied by the harness
    // Ordered named digests; duplicate names are rejected rather than ambiguous JSON.
    std::vector<std::pair<std::string, std::string>> digests;

    [[nodiscard]] std::string json() const {
        std::string hashes;
        for (std::size_t i = 0; i < digests.size(); ++i) {
            for (std::size_t j = 0; j < i; ++j)
                if (digests[j].first == digests[i].first)
                    throw std::invalid_argument("duplicate receipt digest: " + digests[i].first);
            if (!hashes.empty()) hashes += ',';
            hashes += json_string(digests[i].first) + ':' + json_string(digests[i].second);
        }
        return "{\"schema\":\"strix-engine-receipt/v1\",\"engine\":" + json_string(engine) +
            ",\"identity\":" + identity.json() +
            ",\"implementation\":" + json_string(implementation) +
            ",\"build\":" + json_string(build) + ",\"kernel\":" + json_string(kernel) +
            ",\"device\":" + json_string(device) + ",\"topology\":" + json_string(topology) +
            ",\"command\":" + json_string(command) +
            ",\"digests\":{" + hashes + "}}";
    }
    [[nodiscard]] std::string witness() const { return "strix-engine-witness " + json() + "\n"; }
};

}  // namespace strix::profile
