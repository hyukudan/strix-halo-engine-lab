// CUDA Qwen dense product profile (strix/profiles/qwen_dense_cuda.hpp): one tier word over the shared resolver.
// Invariants: EXACT (no word, `exact`, `default`) changes no option and prints nothing (effective config byte-identical);
// FAST turns on exactly the levers qualified for the regime (split_k = nvfp4-splitk-fast-v1 on the spec windows route) and
// prints their identities; `--split-k exact` under fast gives back the EXACT options; explicit options override; the
// decode and the spec CLI resolve split_k identically except the one named fast-tier case; turbo is refused.
#include "strix/profiles/qwen_dense_cuda.hpp"
#include "strix/core/dense_activation.hpp"
#include "strix/spec/block_drafter.hpp"

#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_fail;
    }
}
// Same field names and enumerators as qwen35::EngineOptions (cuda_engine.hpp), host-only.
enum class PrefillArith : unsigned char { exact, flash, fast, fast2, fast8, fast4 };
enum class PrefillGdn : unsigned char { serial, chunk16, chunk16tc };
enum class PrefillHid : unsigned char { f32, bf16 };
enum class TurboWeights : unsigned char { off, native, onfly, converted };
enum class Split : unsigned char { exact, fast };
// the engine's spelling hooks (cuda_engine.hpp parse_lever_value), found by apply() through ADL; the real EngineOptions path is
// covered by tests/qwen35_host_guards (every spelling of parse_prefill_gdn / parse_prefill_hid through resolve + apply)
PrefillGdn parse_lever_value(PrefillGdn, const std::string& s) {
    if (s == "serial") return PrefillGdn::serial;
    if (s == "chunk16") return PrefillGdn::chunk16;
    if (s == "chunk16tc") return PrefillGdn::chunk16tc;
    throw std::invalid_argument("--prefill-gdn expects serial|chunk16|chunk16tc, got '" + s + "'");
}
PrefillHid parse_lever_value(PrefillHid, const std::string& s) {
    if (s == "f32") return PrefillHid::f32;
    if (s == "bf16") return PrefillHid::bf16;
    throw std::invalid_argument("--prefill-hid expects f32|bf16, got '" + s + "'");
}
struct Options {
    PrefillArith prefill_arith = PrefillArith::exact;
    PrefillGdn prefill_gdn = PrefillGdn::serial;
    PrefillHid prefill_hid = PrefillHid::f32;
    bool prefill_big = true;
    bool prefill_attn8 = false;
    Split nvfp4_split = Split::exact;
    bool decode_attn8 = false;
    bool dense_activation = false;
    TurboWeights turbo_weights = TurboWeights::off;
    bool operator==(const Options&) const = default;
};
namespace qp = strix::qwen35::profile;
using strix::profile::Name;
using strix::profile::Pairs;

Options run(std::optional<Name> profile, TurboWeights w, qp::Route route, std::uint64_t tokens, const Pairs& cli = {},
            std::string* text = nullptr, std::span<const qp::Qualified> rows = qp::qualified,
            qp::PromptRoute* route_out = nullptr, bool per_row = false) {
    Options o;
    o.turbo_weights = w;
    auto regime = qp::regime_of(o, route, tokens);
    regime.arch = "fixture";
    // Ordinary timing rows bind the device product and the priced workload (sp::Regime::execution).
    regime.cost_device = "fixture-cost-device";
    regime.workload = {tokens, tokens + 1024, 1, 2048, 32, 1, 0, 100};
    std::vector<strix::profile::LadderRow> ladder;
    for (const auto& row : rows)
        if (row.weights == regime.weights && row.route == route)
            ladder.push_back({row.gate.lever, row.gate.value, Name::fast, strix::profile::Phase::prefill, regime.arch,
                              std::max<std::uint64_t>(1, row.gate.min_prompt_tokens), row.gate.max_prompt_tokens,
                              1, 1, 1.1, 0.02, "fixture timing"});
    // Each timing row names its measured FAST value and the execution it was measured over (LadderArms::over): the
    // default FAST execution without explicit settings (an explicit override makes the row stale). Cells of one tier
    // extend its incumbent execution in table order, so the references settle within one pass per row.
    for (auto& row : ladder)
        if (row.value.empty())
            for (const auto& lever : qp::table)
                if (lever.name == row.lever) row.value = lever.value[static_cast<unsigned>(Name::fast)];
    std::vector<std::string> references(ladder.size());
    const bool lossy = profile && (*profile == Name::fast || *profile == Name::turbo);
    // The first pass reads the references with no timing row (grants, schedules and lower cells only): every row is
    // then measured over the execution its cell extends, and the later passes settle the timed cells in table order.
    for (std::size_t pass = 0; lossy && pass <= ladder.size() + 1U; ++pass) {
        const qp::TurboEvidence measured[] = {
            {regime.weights, route, {}, {}, pass == 0 ? std::span<const strix::profile::LadderRow>{} : std::span<const strix::profile::LadderRow>(ladder)}};
        const auto fast = qp::resolve(Name::fast, regime, {}, rows, measured);
        bool settled = pass != 0;
        for (std::size_t i = 0; i < ladder.size(); ++i) {
            const auto* entry = fast->find(ladder[i].lever);
            if (entry == nullptr || entry->ladder_reference == references[i]) continue;
            references[i] = entry->ladder_reference;
            ladder[i].over = references[i];
            settled = false;
        }
        if (settled) break;
    }
    const qp::TurboEvidence evidence[] = {{regime.weights, route, {}, {}, ladder}};
    const auto r = qp::resolve(profile, regime, cli, rows, evidence);
    qp::apply(o, *r);
    if (text) *text = qp::describe(*r);
    if (route_out) *route_out = qp::prompt_route(*r, per_row);
    return o;
}
// The decode CLI's prompt route for one resolution (qp::prompt_route on the same resolve as run()).
qp::PromptRoute route_of(std::optional<Name> profile, TurboWeights w, qp::Route route, std::uint64_t tokens,
                         const Pairs& cli = {}, bool per_row = false) {
    qp::PromptRoute pr;
    (void)run(profile, w, route, tokens, cli, nullptr, qp::qualified, &pr, per_row);
    return pr;
}
bool throws(const auto& fn, const char* needle) {
    try {
        fn();
    } catch (const std::exception& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}
}  // namespace

int main() {
    const TurboWeights all_w[] = {TurboWeights::off, TurboWeights::native, TurboWeights::onfly, TurboWeights::converted};
    const qp::Route all_r[] = {qp::Route::large_prefill, qp::Route::windows};
    const std::uint64_t all_t[] = {0, 8192, 32768, 131072};

    // 1. EXACT: no word, `exact`, `default` -> options exactly as built, no profile text, in every regime.
    for (const auto profile : {std::optional<Name>{}, std::optional<Name>{qp::parse("exact")}, std::optional<Name>{qp::parse("default")}})
        for (const auto w : all_w)
            for (const auto route : all_r)
                for (const auto t : all_t) {
                    std::string text = "x";
                    Options want;
                    want.turbo_weights = w;
                    check(run(profile, w, route, t, {}, &text) == want, "EXACT changed an option");
                    check(text.empty(), "EXACT printed profile text (effective config not byte-identical)");
                }

    // 2. FAST, decode CLI (large prefill), NVFP4 onfly/converted: fast4 + chunk16tc + attn8 from 8K; split_k pending
    //    there (qualified on the spec windows route only: no composition receipt with fast4 + chunk16).
    for (const auto w : {TurboWeights::onfly, TurboWeights::converted}) {
        std::string text;
        const Options o8 = run(Name::fast, w, qp::Route::large_prefill, 8192, {}, &text);
        check(o8.prefill_arith == PrefillArith::fast4 && o8.prefill_gdn == PrefillGdn::chunk16tc, "FAST 8K bundle");
        check(o8.prefill_attn8 && o8.nvfp4_split == Split::exact && o8.prefill_big, "FAST 8K: attn8 on, split_k off");
        const Options o4 = run(Name::fast, w, qp::Route::large_prefill, 4096);
        check(!o4.prefill_attn8 && o4.prefill_arith == PrefillArith::fast4, "FAST below 8K: attn8 stays off");
        check(text.find("prefill_arith = fast4 [profile]") != std::string::npos, "FAST text lists prefill_arith");
        check(text.find("prefill_attn8 = on [profile]") != std::string::npos, "FAST 8K text lists prefill_attn8");
        check(text.find("split_k = exact [pending]") != std::string::npos &&
                  text.find("qualified only for nvfp4-rtn (onfly/converted) / windows (spec CLI)") != std::string::npos,
              "FAST text explains split_k");
        check(text.find("prefill_arith = fast4 [profile] fast4 (nvfp4-tile-mma-f32-v1") != std::string::npos,
              "FAST text prints the identity of an applied lever");
        check(text.find("profile-fingerprint: ") != std::string::npos, "FAST fingerprint line");
        const Options o32 = run(Name::fast, w, qp::Route::large_prefill, 32768);
        check(o32.prefill_arith == PrefillArith::fast4 && o32.prefill_gdn == PrefillGdn::chunk16tc && o32.prefill_attn8, "FAST 32K adds attn8");
    }
    // FAST in unqualified weight regimes: nothing changes on either route.
    for (const auto t : all_t)
        for (const auto w : {TurboWeights::off, TurboWeights::native})
            for (const auto route : all_r) {
                Options want;
                want.turbo_weights = w;
                check(run(Name::fast, w, route, t) == want, "FAST applied a lever in an unqualified weight regime");
            }
    // 2b. FAST on the spec windows route, NVFP4 onfly/converted: exactly split_k fast (nvfp4-splitk-fast-v1), at every
    //     prompt length; the off switch --split-k exact gives back the EXACT options bit for bit.
    for (const auto t : all_t)
        for (const auto w : {TurboWeights::onfly, TurboWeights::converted}) {
            std::string text;
            Options want;
            want.turbo_weights = w;
            Options fast_want = want;
            fast_want.nvfp4_split = Split::fast;
            check(run(Name::fast, w, qp::Route::windows, t, {}, &text) == fast_want, "FAST windows: split_k fast only");
            check(text.find("split_k = fast [profile] fast (nvfp4-splitk-fast-v1, identity -skf1)") != std::string::npos,
                  "FAST windows text names nvfp4-splitk-fast-v1 (incumbent grant where no timing row covers)");
            check(text.find("prefill_") == std::string::npos, "FAST windows text lists a lever not offered there");
            std::string off_text;
            check(run(Name::fast, w, qp::Route::windows, t, {{"split_k", "exact"}}, &off_text) == want, "--split-k exact under fast != EXACT");
            // the off switch prints as [off] (shared resolver: explicit value == the lever's off value) and turns no identity on
            check(off_text.find("split_k = exact [off]") != std::string::npos && off_text.find("-skf1") == std::string::npos,
                  "FAST windows --split-k exact text: split_k = exact [off], no -skf1");
        }

    // 2c. Registry: every qualified row names a lever of the table that is offered on its route, carries a receipt, and
    //     split_k is qualified exactly once (NVFP4 onfly/converted, spec windows route).
    {
        int split_rows = 0;
        for (const auto& q : qp::qualified) {
            bool in_table = false;
            for (const auto& lever : qp::table) in_table |= q.gate.lever == lever.name;
            check(in_table && qp::offered(q.gate.lever, q.route) && !q.gate.receipt.empty(), "qualified row without lever / route / receipt");
            if (q.gate.lever == "split_k") {
                ++split_rows;
                check(q.weights == qp::Weights::nvfp4_rtn && q.route == qp::Route::windows && q.gate.min_prompt_tokens == 0,
                      "split_k qualified outside its receipt regime");
            }
            check(q.weights != qp::Weights::exact_pack && q.weights != qp::Weights::nvfp4_native, "row for an ungated weight source");
        }
        check(split_rows == 1, "split_k must have exactly one qualified row");
    }
    // quality: no quality rows -> default levers.
    {
        Options want;
        want.turbo_weights = TurboWeights::converted;
        check(run(Name::quality, TurboWeights::converted, qp::Route::large_prefill, 32768) == want, "quality is default");
    }

    // 3. Explicit per-lever options override the profile (both directions) and keep today's spellings.
    {
        const Options a = run(Name::fast, TurboWeights::converted, qp::Route::large_prefill, 8192, {{"prefill_gdn", "serial"}});
        // The explicit value runs; the fast4 timing receipt was measured with the FAST chunk16tc scan, so it does not
        // match this run and has no effect: the frozen FAST grant (fast4 ran before the ladder rule) resolves.
        check(a.prefill_arith == PrefillArith::fast4 && a.prefill_gdn == PrefillGdn::serial, "explicit --prefill-gdn serial under fast");
        const Options b = run(std::nullopt, TurboWeights::off, qp::Route::large_prefill, 0, {{"prefill_arith", "fast8"}, {"prefill_attn8", "on"}});
        check(b.prefill_arith == PrefillArith::fast8 && b.prefill_attn8, "explicit levers without a profile word");
        const Options c = run(std::nullopt, TurboWeights::off, qp::Route::large_prefill, 0, {{"prefill_arith", "windows"}});
        check(!c.prefill_big && c.prefill_arith == PrefillArith::exact, "--prefill-arith windows");
        const Options d = run(Name::fast, TurboWeights::converted, qp::Route::large_prefill, 32768, {{"prefill_attn8", "off"}, {"prefill_arith", "exact"}});
        check(d.prefill_arith == PrefillArith::exact && !d.prefill_attn8, "explicit off switches under fast");
        // every engine spelling passes the resolver (the CLI path once kept a second spelling list that refused chunk16tc);
        // chunk16 (the f32 scan) stays reachable under fast as an explicit choice; the explicit setting is part of the key,
        // so FAST timing rows measured over the chunk16tc arm do not match and have no effect: the frozen grant resolves
        const Options e = run(Name::fast, TurboWeights::converted, qp::Route::large_prefill, 8192, {{"prefill_gdn", "chunk16"}});
        check(e.prefill_arith == PrefillArith::fast4 && e.prefill_gdn == PrefillGdn::chunk16, "explicit --prefill-gdn chunk16 under fast");
        const Options f = run(std::nullopt, TurboWeights::converted, qp::Route::large_prefill, 0,
                              {{"prefill_arith", "fast4"}, {"prefill_gdn", "chunk16tc"}, {"prefill_hid", "bf16"}});
        check(f.prefill_arith == PrefillArith::fast4 && f.prefill_gdn == PrefillGdn::chunk16tc && f.prefill_hid == PrefillHid::bf16,
              "explicit --prefill-hid bf16 without a profile word");
        // prefill_hid has no gate receipt: FAST leaves the FP32 hidden rows (pending), at 8K and at 32K
        check(run(Name::fast, TurboWeights::converted, qp::Route::large_prefill, 8192).prefill_hid == PrefillHid::f32 &&
                  run(Name::fast, TurboWeights::converted, qp::Route::large_prefill, 32768).prefill_hid == PrefillHid::f32,
              "FAST keeps prefill_hid f32 until gated");
    }
    // Bad values throw and leave the options unchanged; a prefill lever is not offered on the windows route.
    {
        Options o;
        o.turbo_weights = TurboWeights::converted;
        const Options before = o;
        const auto bad = [&o](const char* lever, const char* value) {
            const auto r = qp::resolve(Name::fast, qp::regime_of(o, qp::Route::large_prefill, 32768), Pairs{{lever, value}});
            qp::apply(o, *r);
        };
        check(throws([&] { bad("prefill_arith", "fast3"); }, "--prefill-arith expects") && o == before, "bad --prefill-arith");
        check(throws([&] { bad("prefill_gdn", "chunk8"); }, "--prefill-gdn expects") && o == before, "bad --prefill-gdn");
        check(throws([&] { bad("prefill_hid", "fp16"); }, "--prefill-hid expects") && o == before, "bad --prefill-hid");
        check(throws([&] { bad("prefill_attn8", "1"); }, "--prefill-attn8 expects") && o == before, "bad --prefill-attn8");
        check(throws([&] { bad("split_k", "fast2"); }, "--split-k expects") && o == before, "bad --split-k");
        check(throws([] { (void)run(std::nullopt, TurboWeights::off, qp::Route::windows, 0, {{"prefill_arith", "fast4"}}); }, "not offered"),
              "prefill lever on the windows route");
        check(throws([&] { bad("dense_activation", "1"); }, "--dense-activation expects") && o == before, "bad dense activation");
        check(throws([] { (void)qp::parse("fastest"); }, "--profile"), "bad profile word");
    }

    // 4. Parity: the decode CLI (large prefill) and the spec CLI (windows) resolve split_k identically for every explicit
    //    word and for every tier except fast, in every weight regime. The one named difference: under fast the spec route
    //    is qualified and the decode route is pending (composition with fast4 + chunk16 not gated). Adding the decode row
    //    (the one-row flip) makes the two equal again.
    std::vector<qp::Qualified> flipped(std::begin(qp::qualified), std::end(qp::qualified));
    flipped.push_back({qp::Weights::nvfp4_rtn, qp::Route::large_prefill, {"split_k", "", 0, strix::profile::kNoPromptBound, "test"}});
    for (const auto& rows : {std::span<const qp::Qualified>(qp::qualified), std::span<const qp::Qualified>(flipped)})
        for (const auto profile : {std::optional<Name>{}, std::optional<Name>{Name::fast}, std::optional<Name>{Name::quality}})
            for (const char* word : {"", "exact", "fast"})
                for (const auto w : all_w) {
                    const Pairs cli = *word ? Pairs{{"split_k", word}} : Pairs{};
                    const Split dec = run(profile, w, qp::Route::large_prefill, 8192, cli, nullptr, rows).nvfp4_split;
                    const Split spec = run(profile, w, qp::Route::windows, 8192, cli, nullptr, rows).nvfp4_split;
                    const bool flipped_rows = rows.size() != std::size(qp::qualified);
                    const bool rtn = w == TurboWeights::onfly || w == TurboWeights::converted;
                    const bool fast_tier = !*word && profile == Name::fast && rtn;
                    const bool want_spec = *word ? std::string(word) == "fast" : fast_tier;
                    const bool want_dec = *word ? std::string(word) == "fast" : fast_tier && flipped_rows;
                    check(spec == (want_spec ? Split::fast : Split::exact), "spec CLI split_k value per tier");
                    check(dec == (want_dec ? Split::fast : Split::exact), "decode CLI split_k value per tier / flip row");
                    check(dec == spec || (fast_tier && !flipped_rows), "decode / spec CLI differ outside the named case");
                }

    // 4. A tier word that turned nothing on is not called "qualified" (exact pack: every lever pending; review finding 7).
    {
        std::string text;
        const Options o = run(Name::fast, TurboWeights::off, qp::Route::windows, 300, {}, &text);
        check(o == Options{}, "FAST on the exact pack changed an option");
        check(text.find("nothing applied by the profile") != std::string::npos && text.find("; qualified;") == std::string::npos,
              "FAST that applied nothing must not print 'qualified'");
        run(Name::fast, TurboWeights::onfly, qp::Route::windows, 300, {}, &text);
        check(text.find("; qualified;") != std::string::npos && text.find("nothing applied") == std::string::npos,
              "FAST that applied split_k prints 'qualified'");
    }

    // 2d. decode_attn8 (FAST e4m3 decode / verify attention): offered on both routes; pending in the shipped registry (fast
    //     leaves it off everywhere); a qualified row promotes it exactly from its min_prompt_tokens on its own route, and
    //     the explicit --decode-attn8 off under fast keeps it off.
    {
        check(qp::offered("decode_attn8", qp::Route::windows) && qp::offered("decode_attn8", qp::Route::large_prefill),
              "decode_attn8 must be offered on both routes");
        for (const auto route : all_r)
            for (const auto t : all_t) {
                std::string text;
                check(!run(Name::fast, TurboWeights::converted, route, t, {}, &text).decode_attn8, "shipped fast turned decode_attn8 on");
                check(text.find("decode_attn8 = off [pending]") != std::string::npos, "fast text must show decode_attn8 pending");
            }
        const qp::Qualified rows[] = {{qp::Weights::nvfp4_rtn, qp::Route::windows, {"decode_attn8", "", 8192, strix::profile::kNoPromptBound, "synthetic test row"}}};
        check(!run(Name::fast, TurboWeights::converted, qp::Route::windows, 4096, {}, nullptr, rows).decode_attn8, "below the row's context");
        check(run(Name::fast, TurboWeights::converted, qp::Route::windows, 8192, {}, nullptr, rows).decode_attn8, "at the row's context");
        check(!run(Name::fast, TurboWeights::converted, qp::Route::large_prefill, 131072, {}, nullptr, rows).decode_attn8,
              "a windows row must not turn the decode CLI on");
        check(!run(Name::fast, TurboWeights::converted, qp::Route::windows, 131072, {{"decode_attn8", "off"}}, nullptr, rows).decode_attn8,
              "--decode-attn8 off under fast");
        check(run(std::nullopt, TurboWeights::off, qp::Route::large_prefill, 0, {{"decode_attn8", "on"}}).decode_attn8,
              "--decode-attn8 on without a profile word");
        check(throws([] { (void)run(std::nullopt, TurboWeights::converted, qp::Route::windows, 0, {{"decode_attn8", "auto"}}); },
                     "--decode-attn8 expects on|off"),
              "bad --decode-attn8 value must throw with the option's message");
    }

    // The app's built-in registry has no device receipts. Candidate defaults inherit FAST and report what is pending.
    {
        namespace p = strix::profile;
        using V = p::GateVerdict;
        qp::Regime regime{qp::Weights::nvfp4_rtn, qp::Route::windows, 8192, "fixture GPU", "sm_120", true};
        regime.cost_device = "fixture-cost-device";
        regime.workload = {8192, 16384, 1, 2048, 32, 1, 4, 100};
        const auto fast = qp::resolve(Name::fast, regime, {});
        const auto pending = qp::resolve(Name::turbo, regime, {});
        check(pending->find("split_k")->value == fast->find("split_k")->value, "TURBO keeps split_k FAST");
        check(pending->find("kv_fp8")->value == "native" && pending->find("decode_attn8")->value == "off" &&
                  pending->find("dense_activation")->value == "off" && pending->find("drafter_window")->value == "7",
              "missing evidence keeps all new defaults on the reference path");
        const auto text = qp::describe(*pending);
        check(text.find("quality may be lower than FAST") != std::string::npos &&
                  text.find("TURBO gate not run") != std::string::npos, "TURBO startup announces quality and qualification");

        Pairs explicit_arms{{"dense_activation", "on"}, {"drafter_window", "15"}};
        auto kv = qp::resolve_kv(Name::turbo, qp::kvp::Request{qp::kvp::Precision::fp8, false, "--kv"},
                                 regime.route, explicit_arms);
        const auto explicit_run = qp::resolve(Name::turbo, regime, explicit_arms);
        Options options;
        qp::apply(options, *explicit_run);
        check(options.decode_attn8 && options.dense_activation && kv.precision == qp::kvp::Precision::fp8 &&
                  strix::spec::parse_chain_depth(explicit_run->find("drafter_window")->value) == 15U,
              "explicit qualification switches reach each candidate");
        check(throws([] { (void)strix::spec::parse_chain_depth("17"); }, "--chain-k"), "chain depth bounded by the shared planner");

        std::array classes{p::ClassVerdict{"code", "8k", V::pass, 800}, p::ClassVerdict{"prose", "8k", V::pass, 800}};
        constexpr std::array identities{qp::kKvIdentity, qp::kAttnIdentity,
                                       qp::kDenseIdentity, qp::kDraftIdentity};
        std::array<p::GateRecord, identities.size()> gates{};
        constexpr std::array keys{"kv_fp8", "decode_attn8", "dense_activation", "drafter_window"};
        std::array<p::LadderRow, identities.size()> gains{};
        for (std::size_t i = 0; i < identities.size(); ++i) {
            gates[i] = {identities[i], qp::model::name, "fixture", V::pass, classes, Name::turbo};
            gains[i] = {keys[i], "", Name::turbo, p::Phase::prefill, regime.arch, 8192, 8192, 1, 1,
                        1.1, 0.01, "fixture gain"};
        }
        gains[0].value = "fp8";
        gains[0].joint = "decode_attn8";  // the pair is admitted by one joint receipt on the shadow lever
        const qp::TurboEvidence evidence[] = {{regime.weights, regime.route, "fixture", gates, gains}};
        const auto resolve = [&](const qp::Regime& r, const Pairs& opts = {}) {
            return qp::resolve(Name::turbo, r, opts, qp::qualified, evidence);
        };
        const auto reference = qp::joint_execution(*fast);
        std::string candidate_base = qp::joint_execution(*resolve(regime));
        gains[0].joint_scope = {regime.cost_device, reference, candidate_base, qp::kKvIdentity, qp::kAttnIdentity,
                                regime.workload, "on"};
        // Single-cell rows name their TURBO value and the reference that runs under them (LadderArms::over: cells of
        // the tier compose in table order, so a later cell is measured over the earlier ones); the joint receipt names
        // the TURBO execution it was measured in, which carries those cells. Both settle together.
        std::array<std::string, identities.size()> references{};
        for (std::size_t i = 1; i < gains.size(); ++i)
            for (const auto& lever : qp::table)
                if (lever.name == keys[i]) gains[i].value = lever.value[static_cast<unsigned>(Name::turbo)];
        // The TURBO schedule is a timed exact lever (Lever::timed): judged in the shared chain over the complete key of
        // the execution it extends, like every other timed cell. Review r63 P1 #2: a receipt naming only the schedule
        // (the old truncated spelling) never decides it.
        const std::string execution = resolve(regime)->execution;
        check(!execution.empty(), "the priced dense CLI spells its execution");
        const std::string truncated = std::string(p::kLadderArmVersion) + ";" + execution +
                                      ";drafter_window=7{chain-window-7-v1};candidate=drafter_window=" +
                                      std::string(gains[3].value) + "{" + std::string(qp::kDraftIdentity) + "}";
        gains[3].over = truncated;
        check(resolve(regime)->find("drafter_window")->value == "7", "a truncated schedule receipt does not decide");
        const auto settle = [&] {
            for (std::size_t pass = 0; pass < gains.size(); ++pass) {
                candidate_base = qp::joint_execution(*resolve(regime));
                gains[0].joint_scope.candidate_base = candidate_base;
                for (std::size_t i = 1; i < gains.size(); ++i) {
                    const auto resolved = resolve(regime);  // the entry lives in this resolution
                    const auto* entry = resolved->find(keys[i]);
                    if (entry == nullptr || entry->ladder_reference.empty()) continue;
                    references[i] = entry->ladder_reference;
                    gains[i].over = references[i];
                }
            }
        };
        for (std::size_t pass = 0; pass < gains.size(); ++pass) {
            candidate_base = qp::joint_execution(*resolve(regime));
            gains[0].joint_scope.candidate_base = candidate_base;
            for (std::size_t i = 1; i < gains.size(); ++i) {
                const auto resolved = resolve(regime);  // the entry lives in this resolution
                const auto* entry = resolved->find(keys[i]);
                if (entry == nullptr || entry->ladder_reference.empty()) continue;
                references[i] = entry->ladder_reference;
                gains[i].over = references[i];
            }
        }
        check(references[2].starts_with("ladder-arm/v4;"), ("Qwen dense reports the TURBO dense reference: " + references[2]).c_str());
        const auto admitted = resolve(regime);
        qp::apply(options, *admitted);
        kv = qp::resolve_kv(Name::turbo, {}, regime.route, explicit_arms);
        qp::observe_kv(kv, *admitted);
        check(admitted->find("split_k")->value == "fast" && admitted->find("drafter_window")->value == "15" &&
                  options.decode_attn8 && options.dense_activation && kv.source == qp::kvp::Source::tier,
              ("matching receipts enable all four TURBO cells on top of FAST: " + qp::describe(*admitted)).c_str());
        check(resolve(regime, {{"decode_attn8", "off"}})->find("kv_fp8")->value == "native", "off switch disables the paired default");
        // Review r63 P1 #2: the schedule's receipt binds the whole lower execution. A companion setting changed
        // explicitly (split_k exact instead of the FAST fast) is another composition: the schedule falls back.
        check(resolve(regime, {{"split_k", "exact"}})->find("drafter_window")->value == "7",
              "an explicit companion setting retires the schedule receipt");
        // Measured with dense_activation off (its own receipt slower), the schedule applies over that execution only:
        // an explicit dense_activation=on is a composition nobody timed.
        gains[2].speedup = 0.5;
        settle();
        check(resolve(regime)->find("dense_activation")->value == "off" && resolve(regime)->find("drafter_window")->value == "15",
              "the schedule applies over the execution it was measured on");
        check(resolve(regime, {{"dense_activation", "on"}})->find("drafter_window")->value == "7",
              "an explicit dense_activation=on retires the schedule receipt measured with it off");
        gains[2].speedup = 1.1;
        settle();
        check(resolve(regime)->find("drafter_window")->value == "15", "receipts restored");
        auto different_work = regime;
        different_work.cost_device = "another product with the same sm_120";
        check(resolve(different_work)->find("kv_fp8")->value == "native", "joint receipt binds the measured product");
        different_work = regime;
        ++different_work.workload.decode_rows;
        check(resolve(different_work)->find("kv_fp8")->value == "native", "joint receipt binds decode width");
        check(resolve(different_work)->find("dense_activation")->value == "off" &&
                  resolve(different_work)->find("drafter_window")->value == "7",
              "ordinary and schedule receipts bind the priced workload too");
        different_work = regime;
        ++different_work.workload.verify_rows;
        check(resolve(different_work)->find("kv_fp8")->value == "native", "joint receipt binds verify width");
        gains[0].joint_scope.second_identity = "different-reader@2";
        check(resolve(regime)->find("kv_fp8")->value == "native", "joint receipt binds both versioned candidates");
        gains[0].joint_scope.second_identity = qp::kAttnIdentity;
        gains[0].max_lanes = 0;
        check(resolve(regime)->find("kv_fp8")->value == "native", "paired default needs the joint receipt in the run shape");
        gains[0].max_lanes = 1;
        gains[0].joint = {};
        check(resolve(regime)->find("kv_fp8")->value == "native" && resolve(regime)->find("decode_attn8")->value == "off",
              "two individually winning single-cell rows without a joint receipt do not admit the pair");
        gains[0].joint = "decode_attn8";
        gains[0].device = "sm_89";
        check(resolve(regime)->find("kv_fp8")->value == "native", "a joint receipt of another device does not admit the pair");
        gains[0].device = regime.arch;
        gains[0].speedup = 1.005;
        check(resolve(regime)->find("kv_fp8")->value == "native", "a joint receipt within noise does not admit the pair");
        gains[0].speedup = 1.1;
        gates[1].tier = Name::fast;
        check(resolve(regime)->find("kv_fp8")->value == "native", "paired default needs both TURBO records");
        gates[1].tier = Name::turbo;
        check(resolve(regime)->find("kv_fp8")->value == "fp8",
              ("a valid joint receipt admits the pair again: " + resolve(regime)->find("kv_fp8")->ladder).c_str());
        {   // r32 P1-a: a refused pair leaves both halves exactly as the standalone FAST resolution (qualified FAST reader).
            std::vector<qp::Qualified> reader(std::begin(qp::qualified), std::end(qp::qualified));
            reader.push_back({regime.weights, regime.route,
                              {"decode_attn8", "", 0, p::kNoPromptBound, "fixture FAST reader receipt"}});
            const std::string fast_reader_reference = qp::resolve(Name::fast, regime, {}, reader)->find("decode_attn8")->ladder_reference;
            check(fast_reader_reference.starts_with("ladder-arm/v4;"), "Qwen dense reports the FAST reader reference");
            const auto fast_reader = [&](Name tier) {
                std::vector<p::LadderRow> ladder(gains.begin(), gains.end());  // the evidence as it stands now
                ladder.push_back({"decode_attn8", "on", Name::fast, p::Phase::prefill, regime.arch, 8192, 8192, 1, 1,
                                  1.05, 0.01, "fixture FAST reader gain"});
                ladder.back().over = fast_reader_reference;
                const qp::TurboEvidence reader_evidence[] = {{regime.weights, regime.route, "fixture", gates, ladder}};
                return qp::resolve(tier, regime, {}, reader, reader_evidence);
            };
            const auto turbo = [&] { return fast_reader(Name::turbo); };
            const auto standalone = fast_reader(Name::fast);
            check(standalone->find("decode_attn8")->value == "on" && p::applied(standalone->find("decode_attn8")->status),
                  "fixture: the qualified FAST reader runs standalone");
            Options expected;
            qp::apply(expected, *standalone);
            const auto same_as_fast = [&](const char* what) {
                const auto run = turbo();
                for (const char* lever : {"kv_fp8", "decode_attn8"}) {
                    const auto* got = run->find(lever);
                    const auto* want = standalone->find(lever);
                    check(got->value == want->value && p::applied(got->status) == p::applied(want->status), what);
                    check(got->status == want->status, what);
                }
                Options pair_options;
                qp::apply(pair_options, *run);
                // the pair's applied option: the reader runs as the standalone FAST arm (the unrelated TURBO cells are not compared)
                check(pair_options.decode_attn8 && pair_options.decode_attn8 == expected.decode_attn8 &&
                          pair_options.nvfp4_split == expected.nvfp4_split, what);
            };
            same_as_fast("receipt measured over FAST without the reader cannot promote the pair");
            const auto reader_reference = qp::joint_execution(*standalone);
            const auto reader_candidate = qp::joint_execution(*turbo());
            gains[0].joint_scope.reference = reader_reference;
            gains[0].joint_scope.candidate_base = reader_candidate;
            const auto with_pair = turbo();
            check(with_pair->find("kv_fp8")->value == "fp8" && with_pair->find("decode_attn8")->value == "on",
                  "joint receipt: the pair runs on TURBO over the qualified FAST reader");
            gains[0].joint = {};
            same_as_fast("no joint receipt: both halves keep the standalone FAST resolution");
            gains[0].joint = "decode_attn8";
            gains[0].max_lanes = 0;
            same_as_fast("joint receipt outside the run shape: standalone FAST");
            gains[0].max_lanes = 1;
            gates[0].tier = Name::fast;
            same_as_fast("shadow gate missing: standalone FAST");
            gates[0].tier = Name::turbo;
            gates[1].tier = Name::fast;
            same_as_fast("reader gate missing: standalone FAST");
            gates[1].tier = Name::turbo;
            gains[0].joint_scope.reference = reference;
            gains[0].joint_scope.candidate_base = candidate_base;
        }
        {
            auto large = regime;
            large.route = qp::Route::large_prefill;
            const qp::TurboEvidence ledger[] = {{large.weights, large.route, "fixture", gates, gains}};
            const auto run = [&](Name tier, const Pairs& opts = {}) {
                return qp::resolve(tier, large, opts, qp::qualified, ledger);
            };
            const auto large_reference = qp::joint_execution(*run(Name::fast));
            const auto large_candidate = qp::joint_execution(*run(Name::turbo));
            gains[0].joint_scope.reference = large_reference;
            gains[0].joint_scope.candidate_base = large_candidate;
            check(run(Name::turbo)->find("kv_fp8")->value == "fp8", "joint large-prefill fixture wins over its FAST arm");
            const auto changed = run(Name::turbo, {{"prefill_attn8", "off"}});
            check(changed->find("kv_fp8")->value == "native" && changed->find("decode_attn8")->value == "off",
                  "switching off FAST prefill attention refuses both TURBO halves");
            gains[0].joint_scope.reference = reference;
            gains[0].joint_scope.candidate_base = candidate_base;
        }
        {   // r32 P2-e: the launcher's native route `--profile fast --kv fp8` forces both FP8 readers, with missing and with
            // valid pair evidence alike (the TURBO ledger neither admits nor removes it).
            auto large = regime;
            large.route = qp::Route::large_prefill;
            const qp::TurboEvidence valid[] = {{large.weights, large.route, "fixture", gates, gains}};
            for (const auto ledger : {std::span<const qp::TurboEvidence>{}, std::span<const qp::TurboEvidence>(valid)}) {
                Pairs forced;
                const auto kv_fp8 = qp::resolve_kv(Name::fast, qp::kvp::Request{qp::kvp::Precision::fp8, false, "--kv"},
                                                   large.route, forced);
                const auto run = qp::resolve(Name::fast, large, forced, qp::qualified, ledger);
                Options forced_options;
                qp::apply(forced_options, *run);
                check(kv_fp8.precision == qp::kvp::Precision::fp8 && forced_options.decode_attn8 && forced_options.prefill_attn8 &&
                          run->find("decode_attn8")->status == p::Status::explicit_cli &&
                          run->find("prefill_attn8")->status == p::Status::explicit_cli,
                      "--profile fast --kv fp8 forces the FP8 readers whatever the TURBO ledger holds");
            }
        }
        classes[1].rows = 799;
        check(resolve(regime)->find("dense_activation")->status == p::Status::pending, "all classes need final evidence");
        classes[1].rows = 800;
        auto other = regime;
        other.prompt_tokens = 8191;
        check(resolve(other)->find("dense_activation")->value == "off", "uses the measured regime");
        other = regime;
        other.arch = "sm_89";
        check(resolve(other)->find("decode_attn8")->reason.find("sm_120") != std::string::npos, "unsupported architecture is named");
        other = regime;
        other.linear_session = false;
        check(resolve(other)->find("decode_attn8")->value == "off", "FP8 shadow declines non-linear sessions");

        using strix::core::dense::Activation;
        using strix::core::dense::select;
        static_assert(select(Activation::fp4, true, 128, 16) == Activation::fp4);
        static_assert(select(Activation::fp8, true, 128, 16) == Activation::fp8);
        static_assert(select(Activation::fp8, false, 128, 16) == Activation::native);
        static_assert(select(Activation::fp4, true, 127, 16) == Activation::native);
    }

    // r32 P1-b: incumbent grants are frozen data, not derived from quality rows. A new quality PASS (a pending cell, or a wider
    // band) leaves its cell at the tier below until a matching timing row is registered.
    {
        namespace p = strix::profile;
        qp::Regime regime{qp::Weights::nvfp4_rtn, qp::Route::large_prefill, 8192, "fixture GPU", "sm_120", true};
        regime.cost_device = "fixture-cost-device";  // a timing row binds the device product and the priced workload
        regime.workload = {8192, 16384, 1, 2048, 32, 1, 0, 100};
        const auto lever_value = [](const qp::Resolved& r, const char* lever) { return r.find(lever)->value; };
        auto changed = qp::table[0];
        check(qp::incumbent_grant(changed, regime, qp::incumbent_grants) != nullptr, "fixture incumbent grant exists");
        changed.value[static_cast<unsigned>(Name::fast)] = "fast8";
        check(qp::incumbent_grant(changed, regime, qp::incumbent_grants) == nullptr,
              "a replacement FAST value cannot inherit the incumbent grant");
        changed = qp::table[0];
        for (auto& item : changed.identity)
            if (item.value == "fast4") item.identity = "nvfp4-tile-mma-f32-v2";
        check(qp::incumbent_grant(changed, regime, qp::incumbent_grants) == nullptr,
              "a revised FAST identity cannot inherit the incumbent grant (review r75 P1-b: the declared identity binds)");
        changed = qp::table[0];
        changed.target = "fast4 (display text revised)";
        check(qp::incumbent_grant(changed, regime, qp::incumbent_grants) != nullptr,
              "the display target is not the identity the grant binds");
        const auto shipped = qp::resolve(Name::fast, regime, {});
        check(lever_value(*shipped, "prefill_arith") == "fast4" && lever_value(*shipped, "prefill_gdn") == "chunk16tc" &&
                  lever_value(*shipped, "prefill_attn8") == "on" && lever_value(*shipped, "prefill_hid") == "f32",
              "frozen grants keep the shipped FAST cells (prefill_hid stays pending)");
        std::vector<qp::Qualified> rows(std::begin(qp::qualified), std::end(qp::qualified));
        rows.push_back({regime.weights, regime.route, {"prefill_hid", "", 0, p::kNoPromptBound, "fixture new quality row"}});
        const auto extended = qp::resolve(Name::fast, regime, {}, rows);
        check(lever_value(*extended, "prefill_hid") == "f32" && !p::applied(extended->find("prefill_hid")->status),
              "a new quality PASS without a timing row leaves prefill_hid at the lower tier");
        check(lever_value(*extended, "prefill_arith") == "fast4", "the new row leaves the incumbent cells alone");
        p::LadderRow timing[] = {{"prefill_hid", "bf16", Name::fast, p::Phase::prefill, regime.arch, 8192, 8192, 1, 1,
                                  1.04, 0.01, "fixture timing"}};
        // Measured over the reference that runs under the cell (LadderArms::over).
        const std::string hid_reference = extended->find("prefill_hid")->ladder_reference;
        check(hid_reference.starts_with("ladder-arm/v4;"), ("Qwen dense reports the prefill_hid reference: " + hid_reference).c_str());
        timing[0].over = hid_reference;
        const qp::TurboEvidence evidence[] = {{regime.weights, regime.route, {}, {}, timing}};
        check(lever_value(*qp::resolve(Name::fast, regime, {}, rows, evidence), "prefill_hid") == "bf16",
              "a matching timing row admits the newly qualified cell");
        auto outside = regime;
        outside.prompt_tokens = 8193;
        check(lever_value(*qp::resolve(Name::fast, outside, {}, rows, evidence), "prefill_hid") == "f32",
              "a timing row covers only its measured band");
        // a wider quality band than the frozen grant: prefill_attn8 was granted from 8,192 tokens
        rows.push_back({regime.weights, regime.route, {"prefill_attn8", "", 4096, p::kNoPromptBound, "fixture wider band"}});
        auto small = regime;
        small.prompt_tokens = 4096;
        check(lever_value(*qp::resolve(Name::fast, small, {}, rows), "prefill_attn8") == "off",
              "a wider quality band is not a wider grant");
    }

    {  // the decode CLI's prompt route follows the RESOLVED prefill_arith, builtin or explicit alike
        const auto pr_w = TurboWeights::converted;
        const auto pr_lp = qp::Route::large_prefill;
        const auto pr_none = route_of(std::nullopt, pr_w, pr_lp, 32768);
        check(pr_none.large && pr_none.arith == "exact" && pr_none.reason.empty(), "no tier word: builtin exact takes the large route");
        check(run(std::nullopt, pr_w, pr_lp, 32768) == run(std::nullopt, pr_w, pr_lp, 32768, {{"prefill_arith", "exact"}}),
              "builtin exact and explicit exact leave the same options");
        const auto pr_ex = route_of(Name::standard, pr_w, pr_lp, 32768);
        check(pr_ex.large && pr_ex.arith == "exact", "--profile exact: large route, exact arithmetic");
        const auto pr_exp = route_of(Name::standard, pr_w, pr_lp, 32768, {{"prefill_arith", "exact"}});
        check(pr_exp.large == pr_ex.large && pr_exp.arith == pr_ex.arith, "explicit --prefill-arith exact = builtin exact");
        const auto pr_fast = route_of(Name::fast, pr_w, pr_lp, 8192);
        check(pr_fast.large && pr_fast.arith == "fast4", "FAST 8K: large route, fast4");
        const auto pr_win = route_of(std::nullopt, pr_w, pr_lp, 2048, {{"prefill_arith", "windows"}});
        check(pr_win.large && pr_win.arith == "windows", "explicit windows keeps the engine's windows inside prefill()");
        const auto pr_dump = route_of(Name::standard, pr_w, pr_lp, 2048, {}, true);
        check(!pr_dump.large && pr_dump.arith == "exact" && pr_dump.reason.find("--dump-mtp") != std::string::npos,
              "per-row dumps run the windows route and say why");
        check(qp::describe_route(pr_dump, 16).find("windows of 16 rows") != std::string::npos &&
                  qp::describe_route(pr_ex, 16) == "  prompt-route: large-M prefill, prefill_arith exact\n",
              "the receipt line names the route that runs");
        check(throws([&] { (void)route_of(Name::fast, pr_w, pr_lp, 8192, {}, true); }, "cannot run there"),
              "per-row dumps refuse a lossy prefill arithmetic instead of dropping it silently");
        const auto pr_spec = route_of(Name::standard, pr_w, qp::Route::windows, 2048);
        check(!pr_spec.large && pr_spec.arith.empty(), "windows route (spec CLI): prefill_arith not offered, no large route");
    }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
