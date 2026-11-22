// Load-time weight decisions are scoped to the consumers' versioned arithmetic (weight::implementation_identity)
// and to the checkpoint content digest, never to the executable that holds the decision table.
#include <strix/weight/levers.hpp>

#include <array>
#include <cstdio>
#include <span>

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
    return ok ? 0 : 1;
}
