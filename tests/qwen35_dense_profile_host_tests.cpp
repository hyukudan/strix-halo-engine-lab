// CUDA Qwen dense product profile (strix/profiles/qwen_dense_cuda.hpp): one tier word over the shared resolver.
// Invariants: EXACT (no word, `exact`, `default`) changes no option and prints nothing (effective config byte-identical);
// FAST turns on exactly the levers qualified for the regime; explicit options override; the decode and the spec CLI
// resolve split_k identically for every tier; promoting a lever is one `qualified` row; turbo is refused.
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
enum class PrefillGdn : unsigned char { serial, chunk16 };
enum class TurboWeights : unsigned char { off, native, onfly, converted };
enum class Split : unsigned char { exact, fast };
struct Options {
    PrefillArith prefill_arith = PrefillArith::exact;
    PrefillGdn prefill_gdn = PrefillGdn::serial;
    bool prefill_big = true;
    bool prefill_attn8 = false;
    Split nvfp4_split = Split::exact;
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

    // 2. FAST, decode CLI (large prefill), NVFP4 onfly/converted: fast4 + chunk16; attn8 only from 32K; split_k pending.
    for (const auto w : {TurboWeights::onfly, TurboWeights::converted}) {
        std::string text;
        const Options o8 = run(Name::fast, w, qp::Route::large_prefill, 8192, {}, &text);
        check(o8.prefill_arith == PrefillArith::fast4 && o8.prefill_gdn == PrefillGdn::chunk16, "FAST 8K bundle");
        check(!o8.prefill_attn8 && o8.nvfp4_split == Split::exact && o8.prefill_big, "FAST 8K: attn8 / split_k must stay off");
        check(text.find("prefill_arith = fast4 [profile]") != std::string::npos, "FAST text lists prefill_arith");
        check(text.find("prefill_attn8 = off [pending]") != std::string::npos && text.find("prompt >= 32768") != std::string::npos,
              "FAST text explains attn8 regime");
        check(text.find("split_k = exact [pending]") != std::string::npos && text.find("no gate receipt yet") != std::string::npos,
              "FAST text explains split_k");
        check(text.find("profile-fingerprint: ") != std::string::npos, "FAST fingerprint line");
        const Options o32 = run(Name::fast, w, qp::Route::large_prefill, 32768);
        check(o32.prefill_arith == PrefillArith::fast4 && o32.prefill_gdn == PrefillGdn::chunk16 && o32.prefill_attn8, "FAST 32K adds attn8");
    }
    // FAST in unqualified weight regimes and on the spec windows route: nothing changes.
    for (const auto t : all_t) {
        for (const auto w : {TurboWeights::off, TurboWeights::native}) {
            Options want;
            want.turbo_weights = w;
            check(run(Name::fast, w, qp::Route::large_prefill, t) == want, "FAST applied in an unqualified weight regime");
        }
        for (const auto w : all_w) {
            Options want;
            want.turbo_weights = w;
            check(run(Name::fast, w, qp::Route::windows, t) == want, "FAST applied a lever on the spec windows route");
        }
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
        check(throws([&] { bad("prefill_attn8", "1"); }, "--prefill-attn8 expects") && o == before, "bad --prefill-attn8");
        check(throws([&] { bad("split_k", "fast2"); }, "--split-k expects") && o == before, "bad --split-k");
        check(throws([] { (void)run(std::nullopt, TurboWeights::off, qp::Route::windows, 0, {{"prefill_arith", "fast4"}}); }, "not offered"),
              "prefill lever on the windows route");
        check(throws([] { (void)run(Name::turbo, TurboWeights::converted, qp::Route::large_prefill, 0); }, "turbo"), "turbo refused");
        check(throws([] { (void)qp::parse("fastest"); }, "--profile"), "bad profile word");
    }

    // 4. Parity: the decode CLI (large prefill) and the spec CLI (windows) resolve split_k identically for every tier and
    //    every explicit word, in every weight regime; and after the one-row flip both turn it on under fast.
    std::vector<qp::Qualified> flipped(std::begin(qp::qualified), std::end(qp::qualified));
    flipped.push_back({"split_k", qp::Weights::nvfp4_rtn, qp::Route::large_prefill, 0, "test"});
    flipped.push_back({"split_k", qp::Weights::nvfp4_rtn, qp::Route::windows, 0, "test"});
    for (const auto& rows : {std::span<const qp::Qualified>(qp::qualified), std::span<const qp::Qualified>(flipped)})
        for (const auto profile : {std::optional<Name>{}, std::optional<Name>{Name::fast}, std::optional<Name>{Name::quality}})
            for (const char* word : {"", "exact", "fast"})
                for (const auto w : all_w) {
                    const Pairs cli = *word ? Pairs{{"split_k", word}} : Pairs{};
                    const Split dec = run(profile, w, qp::Route::large_prefill, 8192, cli, nullptr, rows).nvfp4_split;
                    const Split spec = run(profile, w, qp::Route::windows, 8192, cli, nullptr, rows).nvfp4_split;
                    check(dec == spec, "decode / spec CLI resolve split_k differently");
                    const bool flipped_rows = rows.size() != std::size(qp::qualified);
                    const bool want_fast = *word ? std::string(word) == "fast"
                                                 : flipped_rows && profile == Name::fast &&
                                                       (w == TurboWeights::onfly || w == TurboWeights::converted);
                    check(dec == (want_fast ? Split::fast : Split::exact), "split_k value per tier / flip row");
                }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
