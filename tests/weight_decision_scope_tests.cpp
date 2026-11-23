// Load-time weight decisions are scoped to the consumers' versioned arithmetic (weight::implementation_identity)
// and to the checkpoint content digest, never to the executable that holds the decision table.
#include <strix/profiles/hy4.hpp>
#include <strix/weight/levers.hpp>

#include <array>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>

namespace {
namespace weight = strix::weight;

bool check(bool ok, const char* what) {
    if (!ok) std::fprintf(stderr, "FAIL: %s\n", what);
    return ok;
}
}  // namespace

int main() {
    using weight::ArithmeticId;
    using weight::Digest256;
    bool ok = true;
    const std::array<ArithmeticId, 2> consumers{ArithmeticId{"rows-sweep", 1}, ArithmeticId{"f16-tile", 2}};
    const std::array<ArithmeticId, 2> reordered{consumers[1], consumers[0]};
    const std::array<ArithmeticId, 2> bumped{consumers[0], ArithmeticId{"f16-tile", 3}};
    const auto impl = weight::implementation_identity(consumers);
    ok &= check(!(impl == Digest256{}), "identity of a valid consumer set is non-zero");
    ok &= check(weight::implementation_identity(reordered) == impl, "identity is order-insensitive");
    ok &= check(!(weight::implementation_identity(bumped) == impl), "a bumped revision changes the identity");
    ok &= check(!(weight::implementation_identity(std::span(consumers).first(1)) == impl), "a dropped consumer changes the identity");
    const std::array<ArithmeticId, 2> duplicate{consumers[0], ArithmeticId{"rows-sweep", 2}};
    const std::array<ArithmeticId, 1> unnamed{ArithmeticId{"", 1}}, zero{ArithmeticId{"rows-sweep", 0}};
    ok &= check(weight::implementation_identity({}) == Digest256{}, "empty set refused");
    ok &= check(weight::implementation_identity(duplicate) == Digest256{}, "duplicate consumer refused");
    ok &= check(weight::implementation_identity(unnamed) == Digest256{}, "unnamed consumer refused");
    ok &= check(weight::implementation_identity(zero) == Digest256{}, "zero revision refused");

    constexpr auto parsed = weight::digest_from_hex("07000000000000000000000000000000000000000000000000000000000000ff");
    static_assert(parsed.bytes[0] == 7 && parsed.bytes[1] == 0 && parsed.bytes[31] == 0xff);
    ok &= check(weight::digest_from_hex("07") == Digest256{}, "short digest literal refused");
    ok &= check(weight::digest_from_hex("g7000000000000000000000000000000000000000000000000000000000000ff") == Digest256{},
                "non-hex digest literal refused");

    const auto checkpoint = weight::digest_from_hex("0700000000000000000000000000000000000000000000000000000000000001");
    const auto gate = weight::digest_from_hex("0800000000000000000000000000000000000000000000000000000000000001");
    const auto gain = weight::digest_from_hex("0900000000000000000000000000000000000000000000000000000000000001");
    const weight::Id policy{"policy", 1};
    weight::WeightLoadScope scope{weight::Backend::hip, "gfx1151", "hip-7.16", "resident", impl, checkpoint};
    const std::array rows{weight::WeightLoadDecision{policy, scope, gate, gain}};
    ok &= check(weight::find_weight_load_decision(policy, scope, rows) == rows.data(), "matching scope selects the row");
    auto other = scope;
    other.build = weight::implementation_identity(bumped);
    ok &= check(!weight::find_weight_load_decision(policy, other, rows), "row retired by a consumer revision bump");
    other.build = weight::implementation_identity({});
    ok &= check(!weight::find_weight_load_decision(policy, other, rows), "zero implementation identity never matches");
    other = scope;
    other.checkpoint = gate;
    ok &= check(!weight::find_weight_load_decision(policy, other, rows), "another checkpoint never matches");

    // Generic weight-load lever ids: one shared policy per role set and format; a model binds it and keeps the
    // id its earlier records used only as an alias.
    const auto& head = weight::kHeadQ8Policy;
    constexpr weight::Id legacy{"family-attention-head-bf16-q8_0", 1};
    constexpr auto attention_head = weight::with_legacy(weight::kAttentionHeadQ8Policy, legacy);
    constexpr std::array bound{attention_head};
    ok &= check(weight::serialize_id(attention_head.id) == "weights.attention_head.q8_0@1", "attention+head id");
    ok &= check(weight::serialize_id(head.id) == "weights.head.q8_0@1", "head id");
    static_assert(weight::find_weight_load_policy("weights.attention_head.q8_0@1") == &weight::kAttentionHeadQ8Policy);
    static_assert(weight::find_weight_load_policy("family-attention-head-bf16-q8_0@1") == nullptr);
    static_assert(weight::find_weight_load_policy("family-attention-head-bf16-q8_0@1", bound) ==
                  &weight::kAttentionHeadQ8Policy);
    static_assert(weight::find_weight_load_policy("weights.attention_head.q8_0@1", bound) ==
                  &weight::kAttentionHeadQ8Policy);
    static_assert(weight::find_weight_load_policy("family-attention-head-bf16-q8_0@2", bound) == nullptr);
    static_assert(weight::find_weight_load_policy("weights.head.q8_0@1", bound) == &weight::kHeadQ8Policy);
    static_assert(weight::find_weight_load_policy("weights.head.q8_0@2") == nullptr);
    static_assert(weight::find_weight_load_policy("weights.head.q8_0@01") == nullptr);
    static_assert(weight::find_weight_load_policy("weights.head.q8_0@") == nullptr);
    static_assert(weight::find_weight_load_policy("weights.head.q8_0") == nullptr);
    static_assert(weight::find_weight_load_policy("weights.head.q8_0@1x") == nullptr);
    static_assert(weight::find_weight_load_policy("") == nullptr);
    static_assert(weight::id_has_key({"a", 12}, "a@12") && !weight::id_has_key({"a", 12}, "a@1") &&
                  !weight::id_has_key({"a", 1}, "a@12") && !weight::id_has_key({"a", 0}, "a@0") &&
                  !weight::id_has_key({"a", 1}, "ab@1") && !weight::id_has_key({"", 1}, "@1"));
    bool hy4_head = false;
    for (const auto& row : strix::hy4::profile::table)
        hy4_head |= row.name == "head_q8" && row.target.find(weight::serialize_id(head.id)) != std::string_view::npos;
    ok &= check(hy4_head, "Hy4 head row names the generic head policy");
    // A decision row recorded under the former model-named key still selects the generic policy; another
    // policy, another revision or another key does not.
    const std::array renamed{weight::WeightLoadDecision{legacy, scope, gate, gain},
                             weight::WeightLoadDecision{attention_head.id, scope, gate, gain}};
    ok &= check(weight::find_weight_load_decision(attention_head, scope, renamed) == renamed.data(),
                "legacy decision row matches the generic policy");
    ok &= check(weight::find_weight_load_decision(attention_head, scope, std::span(renamed).last(1)) == &renamed[1],
                "generic decision row matches");
    ok &= check(!weight::find_weight_load_decision(head, scope, renamed), "another policy never matches");
    ok &= check(!weight::find_weight_load_decision(attention_head.id, scope, std::span(renamed).first(1)),
                "the plain id lookup accepts no alias");
    const std::array bumped_legacy{weight::WeightLoadDecision{{legacy.name, 2}, scope, gate, gain}};
    ok &= check(!weight::find_weight_load_decision(attention_head, scope, bumped_legacy),
                "another legacy revision never matches");

    // The resolver and its receipt line use the generic id, whatever key the decision row was stored under.
    constexpr std::array formats{weight::kBf16WeightStorage, weight::kQ8WeightStorage};
    weight::WeightLoadRequest request{strix::profile::Name::fast, scope, formats, true, false};
    const auto plan = weight::resolve_weight_load(weight::builtin_registry(), attention_head, request,
                                                  std::span(renamed).first(1));
    ok &= check(plan.status == strix::profile::Status::set && plan.storage == weight::kQ8WeightStorage,
                "FAST selects the qualified storage through a legacy row");
    ok &= check(weight::weight_load_receipt(plan) == "strix-weight-load/v1 weights.attention_head.q8_0@1 gguf.q8_0@1 profile",
                "receipt prints the generic id");
    request.tier = strix::profile::Name::standard;
    ok &= check(weight::weight_load_receipt(weight::resolve_weight_load(weight::builtin_registry(), attention_head, request,
                                                                        renamed)) ==
                    "strix-weight-load/v1 weights.attention_head.q8_0@1 plain.bf16@1 builtin",
                "EXACT receipt keeps BF16 storage");
    return ok ? 0 : 1;
}
