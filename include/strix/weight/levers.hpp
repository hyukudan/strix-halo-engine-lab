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
// The output head alone stored as Q8_0, converted from BF16 at load.
inline constexpr WeightLoadPolicy kHeadQ8Policy{{"weights.head.q8_0", 1}, kQ8WeightStorage};
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
inline constexpr std::array<const WeightLoadPolicy*, 7> kWeightLoadPolicies{
    &kAttentionHeadQ8Policy, &kHeadQ8Policy, &kAttentionHeadQ6Policy, &kAttentionQ4HeadQ6Policy, &kAttentionHeadQ4Policy,
    &kHeadQ6Policy, &kHeadQ4Policy};

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

// A row with a gate v2 PASS qualifies FAST and TURBO. A row without one (zero gate_v2_pass) qualifies TURBO only, and
// only with a measured gain and its measured quality against BF16 stated in `turbo_quality` (receipt text such as
// 'kl=... top1=... dnll=... flip_ratio=... receipt=...'); FAST never takes it.
struct WeightLoadDecision {
    Id policy;
    WeightLoadScope scope;
    Digest256 gate_v2_pass, measured_gain;
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

struct WeightLoadPlan {
    Id policy;
    FormatId storage;  // every tagged role but the head
    profile::Status status;
    std::string_view reason;
    Digest256 gate_v2_pass{}, measured_gain{};
    FormatId head_storage = kBf16WeightStorage;
    std::string_view turbo_quality{};  // non-empty: admitted by a TURBO-only row (quality below the FAST gate, stated)

    [[nodiscard]] FormatId storage_for(std::optional<Role> role) const {
        return !role ? kBf16WeightStorage : *role == Role::head ? head_storage : storage;
    }
    [[nodiscard]] RoleStorage roles() const { return {storage, head_storage}; }
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
WeightLoadPlan resolve_weight_load(const Registry&, const WeightLoadPolicy&,
                                   const WeightLoadRequest&, std::span<const WeightLoadDecision>);

// Receipt line of a resolved plan: 'strix-weight-load/v1 <policy id> <storage id> <status>'. The policy is
// always printed under its generic id; a reader resolves older lines through find_weight_load_policy.
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
    out += ' ';
    out += profile::status_name(plan.status);
    return out;
}
}
