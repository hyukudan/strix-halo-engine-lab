// Load-time weight decisions are scoped to the consumers' versioned arithmetic (weight::implementation_identity)
// and to the checkpoint content digest, never to the executable that holds the decision table.
#include <strix/profiles/hy4.hpp>
#include <strix/weight/levers.hpp>
#include <strix/weight/runtime.hpp>

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
    const auto toolkit = weight::hip_clang_compiler_id("24.0.0git (https://github.com/ROCm/llvm-project.git "
                                                       "d6f6cb691863e9258034ab48c1dfce21689d5606)");
    ok &= check(toolkit == "hip-clang-24.0.0git-d6f6cb691863", "HIP toolkit id from the device compiler text");
    ok &= check(weight::hip_clang_compiler_id("20.0.0") == "hip-clang-20.0.0", "HIP toolkit id without a commit");
    weight::WeightLoadScope scope{weight::Backend::hip, "gfx1151", toolkit, "resident", impl, checkpoint};
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
    const auto runtime_compiler = weight::current_compiler_id();
    other = scope;
    other.compiler_id = runtime_compiler;
    ok &= check(!weight::find_weight_load_decision(policy, other, rows),
                "the runtime compiler id cannot match a HIP weight-load row");

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
    // The attention-only row is its own policy: no alias resolves to it and it never resolves to the head rows.
    ok &= check(weight::serialize_id(weight::kAttentionQ8Policy.id) == "weights.attention.q8_0@1", "attention id");
    static_assert(weight::find_weight_load_policy("weights.attention.q8_0@1") == &weight::kAttentionQ8Policy);
    static_assert(weight::find_weight_load_policy("weights.attention.q8_0@1", bound) == &weight::kAttentionQ8Policy);
    static_assert(weight::find_weight_load_policy("weights.attention.q8_0@2") == nullptr);
    static_assert(weight::same_id(weight::kAttentionQ8Policy.target, weight::kQ8WeightStorage));
    static_assert(!weight::same_id(weight::kAttentionQ8Policy.id, weight::kAttentionHeadQ8Policy.id));
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
    weight::WeightLoadPlan head_only{{"weights.head.q6_g64", 1}, weight::kBf16WeightStorage, strix::profile::Status::set, ""};
    head_only.head_storage = weight::kQ6G64WeightStorage;
    ok &= check(weight::weight_load_receipt(head_only) ==
                    "strix-weight-load/v1 weights.head.q6_g64@1 plain.bf16@1 head strix.q6_g64@1 profile",
                "receipt names a head storage that differs");
    // The role rows in the per-role form: the head-only Q8_0 row keeps the other roles BF16, the attention row keeps
    // the head BF16, and the plan of an arm names both role classes.
    static_assert(weight::same_id(weight::kHeadQ8Policy.target, weight::kBf16WeightStorage) &&
                  weight::same_id(weight::kHeadQ8Policy.head_target(), weight::kQ8WeightStorage));
    static_assert(weight::same_id(weight::kAttentionQ8Policy.target, weight::kQ8WeightStorage) &&
                  weight::same_id(weight::kAttentionQ8Policy.head_target(), weight::kBf16WeightStorage));
    {
        const weight::WeightLoadRequest arm{strix::profile::Name::fast, scope, formats, true, true};
        const auto head_arm = weight::resolve_weight_load(weight::builtin_registry(), weight::kHeadQ8Policy, arm, {});
        const auto attention_arm =
            weight::resolve_weight_load(weight::builtin_registry(), weight::kAttentionQ8Policy, arm, {});
        ok &= check(head_arm.roles().attention == weight::kBf16WeightStorage &&
                        head_arm.roles().head == weight::kQ8WeightStorage &&
                        attention_arm.roles().attention == weight::kQ8WeightStorage &&
                        attention_arm.roles().head == weight::kBf16WeightStorage,
                    "head-only and attention-only Q8_0 arms name their role storages");
    }

    // Per-regime rows: one resident image serves prefill and decode. A gate-only row (gate v2 PASS, no measured
    // gain) never selects the image; a full row selects it only when every regime of the image has a gate row.
    {
        const std::array<std::string_view, 2> regimes{weight::kPrefillRegime, weight::kDecodeRegime};
        auto prefill = scope;
        prefill.regime = weight::kPrefillRegime;
        auto decode = scope;
        decode.regime = weight::kDecodeRegime;
        const weight::WeightLoadEvidence pf_receipt{"receipts/prefill", "2026-11-23", 2457, 0, 0};
        const weight::WeightLoadEvidence banded{"receipts/decode", "2026-11-23", 2457, 65536, 0};
        const auto& q8 = weight::kAttentionHeadQ8Policy;
        weight::WeightLoadRequest r{strix::profile::Name::fast, scope, formats, true, false, regimes, {32768, 0}};
        const auto resolve = [&](std::span<const weight::WeightLoadDecision> table) {
            return weight::resolve_weight_load(weight::builtin_registry(), q8, r, table);
        };
        const std::array gate_only{weight::WeightLoadDecision{q8.id, prefill, gate, {}, pf_receipt}};
        auto p = resolve(gate_only);
        ok &= check(p.storage == weight::kBf16WeightStorage && p.status == strix::profile::Status::pending &&
                        p.regime_count == 2 && p.regimes[0].qualification == weight::RegimeQualification::gate_only &&
                        p.regimes[1].qualification == weight::RegimeQualification::none,
                    "a gate-only prefill row keeps the image BF16 and is reported");
        ok &= check(weight::weight_load_regime_receipt(p) == "strix-weight-load-regimes/v1 weights.attention_head.q8_0@1 "
                        "plain.bf16@1 pending prefill=gate-only:receipts/prefill decode=none", "per-regime receipt line");
        ok &= check(weight::find_weight_load_decision(q8, prefill, gate_only) == nullptr,
                    "a gate-only row is not a full decision row");
        const std::array full_decode_only{weight::WeightLoadDecision{q8.id, decode, gate, gain, banded}};
        p = resolve(full_decode_only);
        ok &= check(p.storage == weight::kBf16WeightStorage && p.status == strix::profile::Status::pending,
                    "a full decode row cannot select the image while prefill has no gate row (no second image)");
        const std::array both{weight::WeightLoadDecision{q8.id, prefill, gate, {}, pf_receipt},
                              weight::WeightLoadDecision{q8.id, decode, gate, gain, banded}};
        p = resolve(both);
        ok &= check(p.storage == weight::kQ8WeightStorage && p.status == strix::profile::Status::set &&
                        p.measured_gain == gain && p.regimes[1].qualification == weight::RegimeQualification::full,
                    "gate rows in every regime plus one full row select the image");
        r.extent.context = 131072;
        p = resolve(both);
        ok &= check(p.storage == weight::kBf16WeightStorage && p.regimes[1].row == nullptr &&
                        p.regimes[1].outside_band == &both[1], "a context outside the gate band is refused");
        ok &= check(weight::weight_load_regime_receipt(p).ends_with("decode=none:outside-band<=65536"),
                    "outside-band receipt");
        r.extent.context = 0;
        ok &= check(resolve(both).storage == weight::kBf16WeightStorage, "unknown context never meets a banded row");
        r.extent.context = 32768;
        // A decode gain banded by lanes: inside the band (and only with known lanes) the row is full; outside it
        // keeps its gate v2 PASS (admits decode) but never selects; the receipt says why.
        const weight::WeightLoadEvidence one_lane{"receipts/decode", "2026-11-24", 2649, 65536, 1};
        const std::array lane_banded{weight::WeightLoadDecision{q8.id, prefill, gate, {}, pf_receipt},
                                     weight::WeightLoadDecision{q8.id, decode, gate, gain, one_lane}};
        ok &= check(weight::find_weight_load_decision(q8.id, decode, lane_banded) == nullptr,
                    "a direct id lookup with unknown extent cannot select a banded decision");
        ok &= check(weight::find_weight_load_decision(q8, decode, lane_banded) == nullptr,
                    "a policy lookup with unknown extent cannot select a banded decision");
        ok &= check(weight::find_weight_load_decision(q8.id, decode, lane_banded, {65536, 1}) == &lane_banded[1] &&
                        weight::find_weight_load_decision(q8, decode, lane_banded, {65536, 1}) == &lane_banded[1],
                    "both decision lookups select at the upper bounds of both bands");
        ok &= check(!weight::find_weight_load_decision(q8.id, decode, lane_banded, {65537, 1}) &&
                        !weight::find_weight_load_decision(q8, decode, lane_banded, {65536, 2}) &&
                        !weight::find_weight_load_decision(q8, decode, lane_banded, {0, 1}) &&
                        !weight::find_weight_load_decision(q8, decode, lane_banded, {65536, 0}),
                    "full decision lookups reject each band independently, including unknown bounds");
        p = resolve(lane_banded);
        ok &= check(p.storage == weight::kBf16WeightStorage && p.regimes[1].row == &lane_banded[1] &&
                        p.regimes[1].qualification == weight::RegimeQualification::gate_only &&
                        p.regimes[1].gain_outside_lanes, "unknown lanes never meet a lane-banded gain");
        r.extent.lanes = 1;
        p = resolve(lane_banded);
        ok &= check(p.storage == weight::kQ8WeightStorage && p.measured_gain == gain &&
                        p.regimes[1].qualification == weight::RegimeQualification::full && !p.regimes[1].gain_outside_lanes,
                    "one lane inside the lane band selects the image");
        ok &= check(weight::weight_load_regime_receipt(p) == "strix-weight-load-regimes/v1 weights.attention_head.q8_0@1 "
                        "gguf.q8_0@1 profile prefill=gate-only:receipts/prefill decode=full:receipts/decode",
                    "lane band receipt inside the band");
        r.extent.lanes = 8;
        p = resolve(lane_banded);
        ok &= check(p.storage == weight::kBf16WeightStorage && p.status == strix::profile::Status::pending &&
                        p.regimes[1].qualification == weight::RegimeQualification::gate_only,
                    "eight lanes outside the lane band keep BF16");
        ok &= check(weight::weight_load_regime_receipt(p).ends_with("decode=gate-only:receipts/decode:gain-lanes<=1"),
                    "lane band receipt outside the band");
        ok &= check(weight::find_weight_load_gate(q8, decode, lane_banded, {32768, 8}) == &lane_banded[1] &&
                        weight::find_weight_load_gate(q8, decode, lane_banded, {131072, 1}) == nullptr,
                    "the gate lookup keeps a lane-banded PASS row and still applies the context band");
        ok &= check(weight::registered_regime_rows(q8, weight::Backend::hip, regimes, lane_banded) ==
                        "prefill=gate-only@" + std::string(scope.device_class) + ",decode=full@" +
                        std::string(scope.device_class) + "/lanes<=1", "committed rows show the lane band");
        ok &= check(weight::weight_load_gain_selects(lane_banded[1], {32768, 1}) &&
                        !weight::weight_load_gain_selects(lane_banded[1], {32768, 2}) &&
                        !weight::weight_load_gain_selects(lane_banded[1], {131072, 1}) &&
                        !weight::weight_load_gain_selects(lane_banded[0], {32768, 1}),
                    "a gain selects only inside both bands; a gate-only row never");
        r.extent.lanes = 0;
        r.tier = strix::profile::Name::standard;
        ok &= check(resolve(both).storage == weight::kBf16WeightStorage, "EXACT never takes a regime row");
        r.tier = strix::profile::Name::fast;
        r.enabled = false;
        ok &= check(resolve(both).status == strix::profile::Status::off, "off switch");
        r.enabled = true;
        r.regimes = {};
        r.scope = decode;
        ok &= check(resolve(both).storage == weight::kQ8WeightStorage, "single-regime request keeps its old meaning");
        r.scope = prefill;
        ok &= check(resolve(both).storage == weight::kBf16WeightStorage, "single gate-only regime never selects");
    }

    // A row without a gate v2 PASS qualifies TURBO only, with a measured gain and a stated quality; FAST never takes it.
    const auto& q4 = weight::kAttentionHeadQ4Policy;
    constexpr std::array lowbit{weight::kBf16WeightStorage, weight::kQ4G64WeightStorage};
    const std::array turbo{weight::WeightLoadDecision{q4.id, scope, {}, gain, {}, "kl=0.01 top1=0.94"}};
    const std::array unstated{weight::WeightLoadDecision{q4.id, scope, {}, gain}};
    const std::array unmeasured{weight::WeightLoadDecision{q4.id, scope, {}, {}, {}, "kl=0.01 top1=0.94"}};
    weight::WeightLoadRequest lowbit_request{strix::profile::Name::turbo, scope, lowbit, true, false};
    const auto turbo_plan = weight::resolve_weight_load(weight::builtin_registry(), q4, lowbit_request, turbo);
    ok &= check(turbo_plan.status == strix::profile::Status::set && turbo_plan.storage == weight::kQ4G64WeightStorage &&
                    turbo_plan.turbo_quality == "kl=0.01 top1=0.94" && turbo_plan.gate_v2_pass == Digest256{},
                "TURBO takes a TURBO row and states its quality");
    ok &= check(weight::resolve_weight_load(weight::builtin_registry(), q4, lowbit_request, unstated).storage ==
                    weight::kBf16WeightStorage,
                "a TURBO row without a stated quality is refused");
    ok &= check(weight::resolve_weight_load(weight::builtin_registry(), q4, lowbit_request, unmeasured).storage ==
                    weight::kBf16WeightStorage,
                "a TURBO row without a measured gain is refused");
    lowbit_request.tier = strix::profile::Name::fast;
    const auto fast_plan = weight::resolve_weight_load(weight::builtin_registry(), q4, lowbit_request, turbo);
    ok &= check(fast_plan.storage == weight::kBf16WeightStorage && fast_plan.turbo_quality.empty(),
                "FAST never takes a TURBO row");
    {
        // Per-regime TURBO rows: TURBO takes the image when every regime has a row; a gate v2 PASS row of the same
        // regime is preferred to the TURBO row; FAST still refuses.
        const std::array<std::string_view, 2> regimes{weight::kPrefillRegime, weight::kDecodeRegime};
        auto prefill = scope;
        prefill.regime = weight::kPrefillRegime;
        auto decode = scope;
        decode.regime = weight::kDecodeRegime;
        weight::WeightLoadRequest r{strix::profile::Name::turbo, scope, lowbit, true, false, regimes, {16384, 0}};
        const std::array turbo_rows{weight::WeightLoadDecision{q4.id, prefill, {}, gain, {}, "kl=0.02"},
                              weight::WeightLoadDecision{q4.id, decode, {}, gain, {}, "kl=0.03"}};
        auto p = weight::resolve_weight_load(weight::builtin_registry(), q4, r, turbo_rows);
        ok &= check(p.storage == weight::kQ4G64WeightStorage && p.turbo_quality == "kl=0.02" && p.regime_count == 2 &&
                        p.regimes[1].row == &turbo_rows[1],
                    "TURBO rows in every regime select the image and state the quality");
        const std::array prefill_only{turbo_rows[0]};
        ok &= check(weight::resolve_weight_load(weight::builtin_registry(), q4, r, prefill_only).storage ==
                        weight::kBf16WeightStorage,
                    "a regime without a TURBO row keeps the image BF16");
        const std::array mixed{turbo_rows[0], weight::WeightLoadDecision{q4.id, decode, gate, gain, {}, {}}, turbo_rows[1]};
        p = weight::resolve_weight_load(weight::builtin_registry(), q4, r, mixed);
        ok &= check(p.regimes[1].row == &mixed[1] && p.turbo_quality == "kl=0.02",
                    "a gate v2 PASS with a gain is preferred to its TURBO row");
        const std::array without_prefill_gain{weight::WeightLoadDecision{q4.id, prefill, gate, {}}, turbo_rows[1]};
        ok &= check(weight::resolve_weight_load(weight::builtin_registry(), q4, r, without_prefill_gain).storage ==
                        weight::kBf16WeightStorage,
                    "the TURBO image cannot replace BF16 in a regime without a measured gain");
        r.tier = strix::profile::Name::fast;
        ok &= check(weight::resolve_weight_load(weight::builtin_registry(), q4, r, turbo_rows).storage ==
                        weight::kBf16WeightStorage,
                    "FAST refuses per-regime TURBO rows");
        // A decode row banded by step rows: the plan reports the band of the regime whose row matched.
        r.tier = strix::profile::Name::turbo;
        weight::WeightLoadRequest d{strix::profile::Name::turbo, decode, lowbit, true, false, {}, {16384, 0}};
        const std::array banded{weight::WeightLoadDecision{q4.id, decode, {}, gain, {"r", "2026-11-24", 1008, 0, 0, 16}, "kl=0.03"}};
        const auto bp = weight::resolve_weight_load(weight::builtin_registry(), q4, d, banded);
        ok &= check(bp.storage == weight::kQ4G64WeightStorage && bp.row_band(weight::kDecodeRegime) == 16U &&
                        bp.row_band(weight::kPrefillRegime) == 0U && p.row_band(weight::kDecodeRegime) == 0U,
                    "decode step band of the matched row (unbanded rows and other regimes: 0)");
    }
    {
        weight::WeightLoadScope s{weight::Backend::hip, "gfx1151", "hip-7.1", "resident", gate, gain};
        ok &= check(weight::weight_load_scope_receipt(s) ==
                        "strix-weight-load-scope/v1 device gfx1151 compiler hip-7.1 regime resident build "
                        "0800000000000000000000000000000000000000000000000000000000000001 checkpoint "
                        "0900000000000000000000000000000000000000000000000000000000000001",
                    "scope receipt prints the digest literals");
    }
    return ok ? 0 : 1;
}
