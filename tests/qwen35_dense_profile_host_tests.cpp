// CUDA Qwen dense product profile (strix/profiles/qwen_dense_cuda.hpp): one tier word over the shared resolver.
// Invariants: EXACT (no word, `exact`, `default`) changes no option and prints nothing (effective config byte-identical);
// FAST turns on exactly the levers qualified for the regime (split_k = nvfp4-splitk-fast-v1 on the spec windows route) and
// prints their identities; `--split-k exact` under fast gives back the EXACT options; explicit options override; the
// decode and the spec CLI resolve split_k identically except the one named fast-tier case; turbo is refused.
#include "strix/profiles/qwen_dense_cuda.hpp"

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
    TurboWeights turbo_weights = TurboWeights::off;
    bool operator==(const Options&) const = default;
};
namespace qp = strix::qwen35::profile;
using strix::profile::Name;
using strix::profile::Pairs;

Options run(std::optional<Name> profile, TurboWeights w, qp::Route route, std::uint64_t tokens, const Pairs& cli = {},
            std::string* text = nullptr, std::span<const qp::Qualified> rows = qp::qualified) {
    Options o;
    o.turbo_weights = w;
    const auto r = qp::resolve(profile, qp::regime_of(o, route, tokens), cli, rows);
    qp::apply(o, *r);
    if (text) *text = qp::describe(*r);
    return o;
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
                  "FAST windows text names nvfp4-splitk-fast-v1");
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
        check(a.prefill_arith == PrefillArith::fast4 && a.prefill_gdn == PrefillGdn::serial, "explicit --prefill-gdn serial under fast");
        const Options b = run(std::nullopt, TurboWeights::off, qp::Route::large_prefill, 0, {{"prefill_arith", "fast8"}, {"prefill_attn8", "on"}});
        check(b.prefill_arith == PrefillArith::fast8 && b.prefill_attn8, "explicit levers without a profile word");
        const Options c = run(std::nullopt, TurboWeights::off, qp::Route::large_prefill, 0, {{"prefill_arith", "windows"}});
        check(!c.prefill_big && c.prefill_arith == PrefillArith::exact, "--prefill-arith windows");
        const Options d = run(Name::fast, TurboWeights::converted, qp::Route::large_prefill, 32768, {{"prefill_attn8", "off"}, {"prefill_arith", "exact"}});
        check(d.prefill_arith == PrefillArith::exact && !d.prefill_attn8, "explicit off switches under fast");
        // every engine spelling passes the resolver (the CLI path once kept a second spelling list that refused chunk16tc);
        // chunk16 (the f32 scan) stays reachable under fast as an explicit choice
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
    // Bad values throw and leave the options unchanged; a prefill lever is not offered on the windows route; turbo refused.
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
        check(throws([] { (void)run(Name::turbo, TurboWeights::converted, qp::Route::large_prefill, 0); }, "turbo"), "turbo refused");
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

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
