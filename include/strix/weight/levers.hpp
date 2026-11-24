#pragma once
#include "registry.hpp"
#include "resolve.hpp"
#include <strix/engine_profile.hpp>
#include <strix/lever.hpp>

#include <array>
#include <optional>
#include <string>
#include <string_view>
namespace strix::weight {
inline constexpr lever::Decl measured_decisions{
    "weight-measured-decisions", "STRIX_WEIGHT_MEASURED_DECISIONS", lever::Exactness::bitwise,
    true, {}, {}, "per-device/compiler/regime byte-equal faster decision receipts"};
static_assert(lever::valid(std::span(&measured_decisions, 1)));
// The family passes its existing environment reader; this never introduces a second getenv path.
inline void apply_decision_override(ResolveRequest& r, const char* raw) {
    r.use_qualified_decisions = lever::select(measured_decisions, {}, raw).on;
}

// These decisions cover a whole family image. Changing the set of tagged roles,
// transcode rule or numerical bundle requires a new policy revision.
// `id` is backend- and model-neutral: 'weights.<roles>.<format>@<revision>'. A model binds one of the shared
// policies below (with_legacy when an earlier record used another key) and never mints a model-named id.
// `legacy` is the id an earlier receipt or decision row stored for the same policy (empty name: none). It only
// resolves old records; receipts print `id`.
struct WeightLoadPolicy {
    Id id;
    FormatId target;
    Id legacy{};
    // The output head's storage when it differs from the other tagged roles' (empty name: `target`). The per-role
    // format map of a mix is data of the shared policy row, never a model code path.
    FormatId head{};
    [[nodiscard]] constexpr FormatId head_target() const noexcept { return head.name.empty() ? target : head; }
};

// Attention projections and the output head stored as Q8_0, converted from BF16 at load.
inline constexpr WeightLoadPolicy kAttentionHeadQ8Policy{{"weights.attention_head.q8_0", 1}, kQ8WeightStorage};
// The output head alone stored as Q8_0, converted from BF16 at load; every other tagged role kept BF16 (per-role).
inline constexpr WeightLoadPolicy kHeadQ8Policy{{"weights.head.q8_0", 1}, kBf16WeightStorage, {}, kQ8WeightStorage};
// Low-bit mixes (strix/weight/host_lowbit.hpp, converted from BF16 at load; HIP consumers): attention projections and
// head q6_g64; attention q4_g64 with a q6_g64 head; both q4_g64.
inline constexpr WeightLoadPolicy kAttentionHeadQ6Policy{{"weights.attention_head.q6_g64", 1}, kQ6G64WeightStorage};
inline constexpr WeightLoadPolicy kAttentionQ4HeadQ6Policy{
    {"weights.attention_q4_g64.head_q6_g64", 1}, kQ4G64WeightStorage, {}, kQ6G64WeightStorage};
inline constexpr WeightLoadPolicy kAttentionHeadQ4Policy{{"weights.attention_head.q4_g64", 1}, kQ4G64WeightStorage};
// The output head alone in a low-bit storage, every other tagged role kept BF16. The head feeds no later layer, so
// these rows leave the routes of a mixture-of-experts model untouched.
inline constexpr WeightLoadPolicy kHeadQ6Policy{
    {"weights.head.q6_g64", 1}, kBf16WeightStorage, {}, kQ6G64WeightStorage};
inline constexpr WeightLoadPolicy kHeadQ4Policy{
    {"weights.head.q4_g64", 1}, kBf16WeightStorage, {}, kQ4G64WeightStorage};
// Every attention projection matrix (query, key, value, output and any auxiliary projection of the attention input,
// such as an output gate, relative or sink projection) stored as Q8_0, converted from BF16 at load; the output head
// stays BF16 (per-role: `head`), so an exact head (full or screened) keeps its bits. Router and normalisation weights
// are not covered.
inline constexpr WeightLoadPolicy kAttentionQ8Policy{{"weights.attention.q8_0", 1}, kQ8WeightStorage, {}, kBf16WeightStorage};
inline constexpr std::array<const WeightLoadPolicy*, 8> kWeightLoadPolicies{
    &kAttentionHeadQ8Policy, &kHeadQ8Policy, &kAttentionQ8Policy, &kAttentionHeadQ6Policy, &kAttentionQ4HeadQ6Policy,
    &kAttentionHeadQ4Policy, &kHeadQ6Policy, &kHeadQ4Policy};

// The binding of a shared policy by a model whose earlier records stored it under `legacy`.
[[nodiscard]] constexpr WeightLoadPolicy with_legacy(const WeightLoadPolicy& shared, Id legacy) noexcept {
    return {shared.id, shared.target, legacy, shared.head};
}

// True when `key` is the 'name@revision' text of `id` (decimal revision, no sign, no leading zero).
[[nodiscard]] constexpr bool id_has_key(Id id, std::string_view key) noexcept {
    if (id.name.empty() || !id.revision || key.size() <= id.name.size() + 1 ||
        key.substr(0, id.name.size()) != id.name || key[id.name.size()] != '@') return false;
    const auto digits = key.substr(id.name.size() + 1);
    if (digits.front() == '0') return false;
    std::uint64_t value = 0;
    for (const char c : digits) {
        if (c < '0' || c > '9') return false;
        value = value * 10U + static_cast<std::uint64_t>(c - '0');
        if (value > id.revision) return false;
    }
    return value == id.revision;
}

[[nodiscard]] constexpr bool same_id(Id a, Id b) noexcept { return a.name == b.name && a.revision == b.revision; }

// True when a record stored under `stored` belongs to `policy`: its id, or its legacy id.
[[nodiscard]] constexpr bool policy_has_id(const WeightLoadPolicy& policy, Id stored) noexcept {
    return same_id(policy.id, stored) || (!policy.legacy.name.empty() && same_id(policy.legacy, stored));
}
// The same for a stored 'name@revision' key (a receipt line).
[[nodiscard]] constexpr bool policy_has_key(const WeightLoadPolicy& policy, std::string_view key) noexcept {
    return id_has_key(policy.id, key) || (!policy.legacy.name.empty() && id_has_key(policy.legacy, key));
}

// The shared policy a stored key names, or one of `bound` (a model's bindings, legacy ids included). The
// result is the shared catalogue row of that id; nullptr: unknown key.
[[nodiscard]] constexpr const WeightLoadPolicy* find_weight_load_policy(
    std::string_view key, std::span<const WeightLoadPolicy> bound = {}) noexcept {
    for (const auto& binding : bound)
        if (policy_has_key(binding, key))
            for (const auto* policy : kWeightLoadPolicies)
                if (same_id(policy->id, binding.id)) return policy;
    for (const auto* policy : kWeightLoadPolicies)
        if (id_has_key(policy->id, key)) return policy;
    return nullptr;
}

// Shared ids are unique, versioned and generic ('weights.' prefix); shared rows carry no legacy id.
[[nodiscard]] constexpr bool valid_weight_load_policies() noexcept {
    for (std::size_t i = 0; i < kWeightLoadPolicies.size(); ++i) {
        const auto& p = *kWeightLoadPolicies[i];
        if (p.id.name.substr(0, 8) != "weights." || !p.id.revision || p.target.name.empty() || !p.legacy.name.empty())
            return false;
        for (std::size_t j = 0; j < i; ++j)
            if (kWeightLoadPolicies[j]->id.name == p.id.name) return false;
    }
    return true;
}
static_assert(valid_weight_load_policies());

// `build` is the implementation identity of the admitted consumers (implementation_identity below), not the
// executable digest: a table compiled into an executable cannot name the digest of the executable that holds it.
// `checkpoint` is the raw SHA-256 of the checkpoint content the gate and timing receipts ran on.
struct WeightLoadScope {
    Backend backend;
    std::string_view device_class, compiler_id, regime;
    Digest256 build, checkpoint;
};

// Regimes a decision row is qualified for (WeightLoadScope::regime): the gate v2 regime of its quality receipt.
// Prompt rows of a prefill chunk are the prefill regime; one-row steps (and their verify rows) the decode regime.
inline constexpr std::string_view kPrefillRegime = "prefill";
inline constexpr std::string_view kDecodeRegime = "decode";

// Where a row's evidence lives. `receipt` is a stable repository-relative id of the gate report (or receipt set),
// `date` its ISO date, `rows` the scored rows of the regime. `max_context` is the largest context the gate covered
// when it was banded by context (0: not banded); a banded row applies only to an engine whose resident context is
// known and inside the band. `max_rows` is the widest step of the row's regime its timing receipt measured a gain at
// (decode: lanes, or lanes x verified tokens of a verify step; 0: not banded); an engine that keeps the regime's
// incumbent storage resident beside the row's runs the row's storage only for steps of at most that many rows.
struct WeightLoadEvidence {
    std::string_view receipt, date;
    std::uint32_t rows = 0, max_context = 0, max_rows = 0;
};

// A row with a zero measured_gain records a gate v2 PASS without a speed gain in its regime: it never selects the
// storage by itself, but admits its regime onto an image another regime's full row selects (see WeightLoadRequest).
// A row with a gate v2 PASS qualifies FAST and TURBO. A row without one (zero gate_v2_pass) qualifies TURBO only, and
// only with a measured gain and its measured quality against BF16 stated in `turbo_quality` (receipt text such as
// 'kl=... top1=... dnll=... flip_ratio=... receipt=...'); FAST never takes it.
struct WeightLoadDecision {
    Id policy;
    WeightLoadScope scope;
    Digest256 gate_v2_pass, measured_gain;
    WeightLoadEvidence evidence{};
    std::string_view turbo_quality{};
};

struct WeightLoadRequest {
    profile::Name tier;
    WeightLoadScope scope;
    // Intersection of the formats consumed by every selected projection and
    // head site, including any exact fallbacks that can run under this regime.
    std::span<const FormatId> consumer_formats;
    bool enabled = true;
    bool allow_pending = false;  // Explicit, stamped qualification-arm opt-in.
    // Every regime that runs on the one resident image (empty: scope.regime alone). The image takes the policy storage
    // only when some regime has a full row (gate v2 PASS and measured gain) and every listed regime has a gate v2 PASS
    // row: a regime the gate did not pass would need a second resident image, which this resolver never selects.
    std::span<const std::string_view> regimes{};
    std::uint32_t context = 0;  // resident context capacity in tokens (0: unknown; banded rows then never apply)
};

// Resolved storage of the two tagged role classes (the output head, every other tagged role). One FormatId converts to
// the same storage for both.
struct RoleStorage {
    FormatId attention = kBf16WeightStorage, head = kBf16WeightStorage;
    constexpr RoleStorage() noexcept = default;
    constexpr RoleStorage(FormatId all) noexcept : attention(all), head(all) {}  // NOLINT(google-explicit-constructor)
    constexpr RoleStorage(FormatId attention_storage, FormatId head_storage) noexcept
        : attention(attention_storage), head(head_storage) {}
};

// What one regime contributed to a plan: a full row (gate v2 PASS + measured gain), a gate-only row, or nothing.
enum class RegimeQualification : std::uint8_t { none, gate_only, full };

[[nodiscard]] constexpr std::string_view regime_qualification_name(RegimeQualification q) noexcept {
    return q == RegimeQualification::full ? "full" : q == RegimeQualification::gate_only ? "gate-only" : "none";
}

struct RegimeWeightLoad {
    std::string_view regime;
    RegimeQualification qualification = RegimeQualification::none;
    const WeightLoadDecision* row = nullptr;          // matched row (nullptr: none)
    const WeightLoadDecision* outside_band = nullptr;  // a row of this regime whose context band misses the request
};

inline constexpr std::size_t kMaxWeightLoadRegimes = 4;

struct WeightLoadPlan {
    Id policy;
    FormatId storage;  // every tagged role but the head
    profile::Status status;
    std::string_view reason;
    Digest256 gate_v2_pass{}, measured_gain{};
    FormatId head_storage = kBf16WeightStorage;
    std::string_view turbo_quality{};  // non-empty: admitted by a TURBO-only row (quality below the FAST gate, stated)
    std::array<RegimeWeightLoad, kMaxWeightLoadRegimes> regimes{};  // per regime, in request order
    std::size_t regime_count = 0;

    [[nodiscard]] FormatId storage_for(std::optional<Role> role) const {
        return !role ? kBf16WeightStorage : *role == Role::head ? head_storage : storage;
    }
    [[nodiscard]] RoleStorage roles() const { return {storage, head_storage}; }
    [[nodiscard]] std::span<const RegimeWeightLoad> regime_rows() const { return std::span(regimes).first(regime_count); }
    // The step band of `regime` (its matched row's evidence.max_rows; 0: no row, or not banded).
    [[nodiscard]] std::uint32_t row_band(std::string_view regime) const {
        for (const auto& r : regime_rows())
            if (r.regime == regime && r.row != nullptr) return r.row->evidence.max_rows;
        return 0U;
    }
};

// A 64-hex content digest literal for receipt rows compiled into a decision table. A malformed literal yields the
// zero digest, which never matches a decision row.
[[nodiscard]] constexpr Digest256 digest_from_hex(std::string_view hex) noexcept {
    const auto nibble = [](char c) -> int {
        return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
    };
    Digest256 out{};
    if (hex.size() != 2 * out.bytes.size()) return {};
    for (std::size_t i = 0; i < out.bytes.size(); ++i) {
        const int high = nibble(hex[2 * i]), low = nibble(hex[2 * i + 1]);
        if (high < 0 || low < 0) return {};
        out.bytes[i] = static_cast<std::uint8_t>(high * 16 + low);
    }
    return out;
}

// Digest of the versioned arithmetic of every consumer route the policy admits (projections, head, prefill tile,
// fallbacks). Order-insensitive; a change of any consumer's arithmetic revision changes it. An empty list, an
// unnamed entry, a zero revision or a duplicate yields the zero digest, which never matches a decision row.
[[nodiscard]] Digest256 implementation_identity(std::span<const ArithmeticId> consumers);

// Only speed evidence with a v2 gate PASS, or with a stated measured TURBO quality (a TURBO row, which the resolver
// gives the TURBO tier only), enters this table. Its receipts must come from the same consumer implementation
// identity, checkpoint, backend, device, compiler and regime.
const WeightLoadDecision* find_weight_load_decision(Id, const WeightLoadScope&,
                                                  std::span<const WeightLoadDecision>);
// As above, also matching rows recorded under the policy's legacy id.
const WeightLoadDecision* find_weight_load_decision(const WeightLoadPolicy&, const WeightLoadScope&,
                                                  std::span<const WeightLoadDecision>);
// The gate v2 PASS row of `scope.regime` (full rows first, then gate-only), matched like the above and inside its
// context band for `context` (0: unknown). `outside_band` (optional) receives a row that matched but whose band
// misses `context`. With `turbo` a TURBO row (no gate v2 PASS, measured gain, stated quality) also matches, after
// every gate v2 PASS row of the regime.
const WeightLoadDecision* find_weight_load_gate(const WeightLoadPolicy&, const WeightLoadScope&,
                                                std::span<const WeightLoadDecision>, std::uint32_t context,
                                                const WeightLoadDecision** outside_band = nullptr, bool turbo = false);
WeightLoadPlan resolve_weight_load(const Registry&, const WeightLoadPolicy&,
                                   const WeightLoadRequest&, std::span<const WeightLoadDecision>);

// Receipt line of the scope a decision row must match: 'strix-weight-load-scope/v1 device <class> compiler <id> regime
// <regime> build <hex> checkpoint <hex>' (64-digit digests, the literals weight::digest_from_hex reads).
[[nodiscard]] std::string weight_load_scope_receipt(const WeightLoadScope& scope);

// Per-regime receipt line of a plan: 'strix-weight-load-regimes/v1 <policy id> <storage id> <status> <regime>=<q>'
// per regime, <q> = full | gate-only | none, then ':<receipt id>' of the matched row, or 'none:outside-band<=N' when
// a row's context band misses the engine's context. Says which regime rows are active and why the others are not.
[[nodiscard]] std::string weight_load_regime_receipt(const WeightLoadPlan& plan);

// Before a device is known (a route description): the committed rows of `policy` for `backend`, per regime,
// '<regime>=<full|gate-only|none>[@<device>]' joined by ','. Says what a device could match, not what it matched.
[[nodiscard]] inline std::string registered_regime_rows(const WeightLoadPolicy& policy, Backend backend,
                                                        std::span<const std::string_view> regimes,
                                                        std::span<const WeightLoadDecision> decisions) {
    std::string out;
    for (const auto regime : regimes) {
        if (!out.empty()) out += ',';
        out += regime;
        out += '=';
        const WeightLoadDecision* found = nullptr;
        for (const auto& row : decisions)
            if (policy_has_id(policy, row.policy) && row.scope.backend == backend && row.scope.regime == regime &&
                !(row.gate_v2_pass == Digest256{}) && (!found || !(row.measured_gain == Digest256{}))) found = &row;
        out += regime_qualification_name(!found ? RegimeQualification::none
            : found->measured_gain == Digest256{} ? RegimeQualification::gate_only : RegimeQualification::full);
        if (found) {
            out += '@';
            out += found->scope.device_class;
        }
    }
    return out;
}

// Receipt line of a resolved plan: 'strix-weight-load/v1 <policy id> <storage id> [head <head storage id>] <status>'
// (the head part when the head storage differs). The policy is always printed under its generic id; a reader
// resolves older lines through find_weight_load_policy.
[[nodiscard]] inline std::string weight_load_receipt(const WeightLoadPlan& plan) {
    const auto put = [](std::string& out, Id id) {
        out += id.name;
        out += '@';
        out += std::to_string(id.revision);
    };
    std::string out = "strix-weight-load/v1 ";
    put(out, plan.policy);
    out += ' ';
    put(out, plan.storage);
    if (!(plan.head_storage == plan.storage)) {
        out += " head ";
        put(out, plan.head_storage);
    }
    out += ' ';
    out += profile::status_name(plan.status);
    return out;
}
}
