#pragma once
#include "registry.hpp"
#include "resolve.hpp"
#include <strix/engine_profile.hpp>
#include <strix/lever.hpp>

#include <optional>
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
struct WeightLoadPolicy {
    Id id;
    FormatId target;
};

struct WeightLoadScope {
    Backend backend;
    std::string_view device_class, compiler_id, regime;
    Digest256 build, checkpoint;
};

struct WeightLoadDecision {
    Id policy;
    WeightLoadScope scope;
    Digest256 gate_v2_pass, measured_gain;
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

struct WeightLoadPlan {
    Id policy;
    FormatId storage;
    profile::Status status;
    std::string_view reason;
    Digest256 gate_v2_pass{}, measured_gain{};

    [[nodiscard]] FormatId storage_for(std::optional<Role> role) const {
        return role ? storage : kBf16WeightStorage;
    }
};

// Only passing v2 gate and speed evidence enters this table. Its receipts must
// come from the same executable, checkpoint, backend, compiler and regime.
const WeightLoadDecision* find_weight_load_decision(Id, const WeightLoadScope&,
                                                  std::span<const WeightLoadDecision>);
WeightLoadPlan resolve_weight_load(const Registry&, const WeightLoadPolicy&,
                                   const WeightLoadRequest&, std::span<const WeightLoadDecision>);
}
