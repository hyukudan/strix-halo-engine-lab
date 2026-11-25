#pragma once

#include "strix/fnv.hpp"
#include "strix/turbo.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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
// unless its turbo cell says otherwise). Shared by every adapter whose rows carry `value` per profile.
template <class Row>
inline unsigned value_column(const Row& lever, Name profile) {
    const auto column = static_cast<unsigned>(profile);
    if (profile == Name::turbo && lever.value[column].empty()) return static_cast<unsigned>(Name::fast);
    return column;
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
};

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
        if (lever.kind != Kind::bit_changing) return true;
        if (entry.value == lever.builtin || (!lever.off.empty() && entry.value == lever.off)) return true;
        const bool in_class = bit_class_member(lever.same_bits, entry.value);
        for (std::size_t c = 0; c < kProfileColumns; ++c)
            if (lever.ready[c] && !lever.provisional[c] && !lever.value[c].empty() &&
                (entry.value == lever.value[c] || (in_class && bit_class_member(lever.same_bits, lever.value[c]))))
                return true;
        return false;
    }
    // "qualified"; else "qualification in progress: <rows>" for provisional columns and/or "unqualified: <rows>"
    // when --allow-pending ran pending columns or an explicit setting chose a bit-changing value no qualified column
    // has, separated by "; ".
    [[nodiscard]] std::string qualification() const {
        std::string provisional, unqualified;
        for (const auto& entry : entries) {
            const bool explicit_unqualified = (entry.status == Status::explicit_cli || entry.status == Status::explicit_env) &&
                                              !explicit_value_qualified(entry);
            auto* rows = entry.status == Status::provisional ? &provisional :
                entry.status == Status::unqualified || explicit_unqualified ? &unqualified : nullptr;
            if (rows != nullptr) *rows += (rows->empty() ? "" : ",") + std::string(entry.lever->name);
        }
        std::string out;
        if (!provisional.empty()) out = "qualification in progress: " + provisional;
        if (!unqualified.empty()) out += (out.empty() ? "" : "; ") + std::string("unqualified: ") + unqualified;
        return out.empty() ? "qualified" : out;
    }
    // One stderr line per pending, provisional or unqualified row (and the quality fallback note).
    [[nodiscard]] std::vector<std::string> notices() const {
        std::vector<std::string> out;
        const std::string head = std::string(engine->tag) + " profile " + name(profile) + ": ";
        if (quality_fallback()) out.push_back(head + "no quality rows for this model, using default");
        if (!weights.honoured()) out.push_back(head + weights.describe().substr(0, weights.describe().size() - 1U));
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
                ",\"source\":" + json_string(lever.source) + "}";
            if (entry.status == Status::pending) {
                if (!pending.empty()) pending += ',';
                pending += "{\"lever\":" + json_string(lever.name) + ",\"target\":" + json_string(entry.target) +
                    ",\"runs\":" + json_string(entry.value) + "}";
            }
            if (entry.status == Status::unqualified) {
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

// Native provisional profiles remain available for experiments. Model presets
// select the same rows, but only admit qualified production defaults.
inline Status admit_profile_column(bool ready, bool runnable, bool allow_pending, bool provisional,
                                   bool preset) {
    if (preset && !allow_pending && provisional) return Status::pending;
    return admit_column(ready, runnable, allow_pending, provisional);
}

inline bool model_preset() {
    const char* value = environment({}, kPresetEnv);
    return value && std::string_view(value) == "1";
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
    for (const auto& lever : engine.table) {
        const auto column = value_column(lever, profile_column);
        const std::string* cli = nullptr;
        for (const auto& [key, value] : inputs.cli) if (key == lever.name) cli = &value;
        const char* env = lever.env.empty() ? nullptr : environment(inputs.env, lever.env);
        const auto wanted = lever.value[column];
        Entry entry{&lever, std::string(lever.builtin), Status::builtin, {}, false, {}, std::string(wanted)};
        if (cli != nullptr || env != nullptr) {  // a set variable is explicit even when empty (the app judges it)
            entry.requested = cli != nullptr ? *cli : std::string(env);
            entry.value = cli != nullptr ? *cli : std::string(env);
            entry.status = cli != nullptr ? Status::explicit_cli : Status::explicit_env;
            if ((!lever.off.empty() && entry.value == lever.off) || entry.value == "off") entry.status = Status::off;
        } else if (wanted.empty() || wanted == lever.builtin) {
            // builtin (the default column always is: default exports nothing)
        } else {
            entry.status = admit_profile_column(lever.ready[column], lever.runnable[column], out.allow_pending,
                                                lever.provisional[column], preset);
            if (entry.status == Status::pending) entry.target = std::string(lever.target.empty() ? wanted : lever.target);
            else entry.value = std::string(wanted);
        }
        out.entries.push_back(std::move(entry));
    }
    out.admitted_ = Resolution::resolve_variants(engine, inputs, out);
    out.admitted_engine_ = &engine;
    out.admitted_profile_ = out.profile;
    return out;
}

// The statuses the profile itself applies (an explicit setting is the app's own).
inline bool applied(Status status) {
    return status == Status::set || status == Status::unqualified || status == Status::provisional;
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
    Inputs request;
    request.profile = Name::fast;
    request.env = [](const char*) -> const char* { return nullptr; };
    const auto resolved = resolve(engine, request);
    const std::string tag(engine.tag);
    for (const auto& gate : ledger) {
        const std::string where = tag + "/" + std::string(gate.lever) + "=" + std::string(gate.value);
        if (gate.receipt.empty()) return where + ": FAST gate ledger row without a receipt";
        const Entry* entry = resolved.find(gate.lever);
        if (entry == nullptr) return where + ": FAST gate receipt names no profile row";
        if (!applied(entry->status) || entry->value != gate.value)
            return where + ": gated but not selected by --profile fast (" + std::string(gate.receipt) + ")";
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
