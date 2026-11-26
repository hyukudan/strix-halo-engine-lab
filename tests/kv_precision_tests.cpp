// Host tests of the shared KV precision toggle (strix/session/kv_precision.hpp): words, the --kv option and its
// serve-config key, per-tier resolution (exact / fast = native, turbo = fp8 where the engine has it), named refusals per
// engine capability, receipts, and the CUDA Qwen dense adapter that folds --kv into its attn8 levers.
#include "strix/session/kv_precision.hpp"

#include "strix/deepseek41_pair_defaults.hpp"
#include "strix/profiles/ds41_rtx.hpp"
#include "strix/profiles/qwen_dense_cuda.hpp"
#include "strix/serve/serve_config.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
template <class F>
std::string refusal(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return {};
}

namespace kvp = strix::session::kv_precision;
using kvp::Precision;
using strix::profile::Name;

// Engine capabilities used by the fixtures (the bound engines' own constants where the header is host-only).
constexpr kvp::Caps kHip{"qwen-hip", Precision::bf16, kvp::bit(Precision::int8), {}, {},
                         "the HIP 8-bit KV form is --kv int8 (kv-int8-tok-v1)"};
constexpr kvp::Caps kBf16Only{"glm-pair", Precision::bf16, 0U, "bf16 (MLA latent)", {}, {}};
constexpr kvp::Caps kF32Native{"f32-engine", Precision::fp32, kvp::bit(Precision::fp16), {}, {}, {}};
const kvp::Caps& kCuda = strix::qwen35::profile::kKvCaps;
const kvp::Caps& kDs41 = strix::deepseek41::pair_defaults::kKvCaps;

kvp::Request request(std::string_view word) {
    kvp::Request r;
    kvp::set(r, word);
    return r;
}

void words() {
    for (const auto w : kvp::kWords) check(kvp::parse(w) && kvp::word(*kvp::parse(w)) == w, "every word round-trips");
    check(!kvp::parse("fp4").has_value() && !kvp::parse("kv8").has_value() && !kvp::parse("native").has_value(),
          "unknown words and `native` are not precisions");
    check(kvp::default_form(Precision::fp8) == "kv-e4m3-tok-v1" && kvp::default_form(Precision::int8) == "kv-int8-tok-v1" &&
              kvp::default_form(Precision::bf16) == "bf16",
          "default forms are strix/kv_storage.hpp's identities");
    check(kHip.supported_text() == "bf16|int8" && kBf16Only.supported_text() == "bf16" && kCuda.supported_text() == "bf16|fp8",
          "supported list");
}

void options() {
    const char* a1[] = {"bin", "--kv", "fp8", "x"};
    kvp::Request r;
    int i = 1;
    check(kvp::take(4, a1, i, r) && i == 2 && r.value == Precision::fp8 && r.spelling == "--kv", "--kv VALUE");
    const char* a2[] = {"bin", "--kv=bf16"};
    kvp::Request r2;
    i = 1;
    check(kvp::take(2, a2, i, r2) && i == 1 && r2.value == Precision::bf16, "--kv=VALUE");
    const char* a3[] = {"bin", "--kvx", "--kv-heads"};
    kvp::Request r3;
    i = 1;
    check(!kvp::take(3, a3, i, r3) && (i = 2, !kvp::take(3, a3, i, r3)) && !r3.given(), "other options are not --kv");
    const char* a4[] = {"bin", "--kv"};
    i = 1;
    check(!refusal([&] { kvp::Request x; (void)kvp::take(2, a4, i, x); }).empty(), "--kv without a value is refused");
    check(refusal([] { (void)request("fp4"); }).find("expects native|fp32|fp16|bf16|fp8|int8") != std::string::npos,
          "unknown word refused with the list");
    check(refusal([] { auto x = request("fp8"); kvp::set(x, "bf16", "--legacy"); }) ==
              "--legacy: the KV precision is already set by --kv",
          "a second setting (any spelling) is refused");
    check(request("native").native && !request("native").value, "--kv native");

    // extract(): compacts argv in place and stops at "--".
    std::vector<std::string> storage{"bin", "--a", "--kv", "fp8", "--b", "--", "--kv", "bf16"};
    std::vector<char*> argv;
    for (auto& s : storage) argv.push_back(s.data());
    argv.push_back(nullptr);
    int argc = static_cast<int>(storage.size());
    const auto got = kvp::extract(argc, argv.data());
    check(got.value == Precision::fp8 && argc == 6 && std::string_view(argv[1]) == "--a" && std::string_view(argv[2]) == "--b" &&
              std::string_view(argv[3]) == "--" && std::string_view(argv[4]) == "--kv" && argv[6] == nullptr,
          "extract removes --kv before -- and keeps the rest");

    const char* t1[] = {"bin", "--profile", "turbo"};
    const char* t2[] = {"bin", "--profile", "--x"};
    const char* t3[] = {"bin", "--profile=fast"};
    const char* t4[] = {"bin", "--", "--profile", "turbo"};
    check(kvp::scan_tier(3, t1) == Name::turbo && kvp::scan_tier(3, t2) == Name::standard && kvp::scan_tier(2, t3) == Name::fast &&
              kvp::scan_tier(4, t4) == Name::standard,
          "scan_tier reads only a tier word (a bare --profile is timing in some engines)");
    const kvp::Caps dual_cache{"dual-cache", Precision::bf16, kvp::bit(Precision::fp8)};
    kvp::Request request;
    kvp::set(request, "fp8");
    const auto strict = kvp::ExactOverride::require_native;
    const auto tier_default = kvp::TierDefault::resolved_by_levers;
    check(refusal([&] { (void)kvp::resolve(Name::standard, request, dual_cache, tier_default, strict); }).find("requires native KV")
              != std::string::npos,
          "explicit strict policy refuses bit-changing KV under exact");
    check(kvp::resolve(Name::fast, request, dual_cache, tier_default, strict).precision == Precision::fp8,
          "strict exact policy admits explicit FP8 under fast");
}

void tiers() {
    const kvp::Request none;
    for (const auto* caps : {&kHip, &kBf16Only, &kCuda, &kDs41, &kF32Native})
        for (const auto tier : {Name::standard, Name::fast, Name::quality}) {
            const auto r = kvp::resolve(tier, none, *caps);
            check(r.precision == caps->native && r.source == kvp::Source::native && !r.changes_output() && r.note.empty(),
                  "exact / fast / quality: the native KV");
        }
    auto t = kvp::resolve(Name::turbo, none, kCuda);
    check(t.precision == Precision::fp8 && t.source == kvp::Source::tier && t.changes_output() &&
              t.form == "kv-e4m3-tok-v1 (e4m3 shadow, attn8)",
          "turbo defaults to fp8 where the engine has it");
    t = kvp::resolve(Name::turbo, none, kHip);
    check(t.precision == Precision::bf16 && t.source == kvp::Source::native &&
              t.note.find("turbo default fp8 unavailable on this engine: native kept") != std::string::npos &&
              t.note.find("--kv int8") != std::string::npos,
          "turbo without fp8 keeps native with a note (no silent switch to int8)");
    t = kvp::resolve(Name::turbo, none, kDs41);
    check(t.precision == Precision::fp8 && t.source == kvp::Source::native && !t.changes_output() &&
              t.form == "ds41-fp8-ring+fp4-compressed (model-defined)",
          "a native fp8 model stays native under turbo");
    t = kvp::resolve(Name::turbo, request("native"), kCuda);
    check(t.precision == Precision::bf16 && t.note.empty(), "--kv native opts turbo out of fp8");
    t = kvp::resolve(Name::turbo, request("bf16"), kCuda);
    check(t.precision == Precision::bf16 && t.source == kvp::Source::explicit_option, "explicit wins over the tier default");
    t = kvp::resolve(Name::fast, request("fp8"), kCuda);
    check(t.precision == Precision::fp8 && t.source == kvp::Source::explicit_option && t.note.empty(), "explicit fp8 under fast");
    t = kvp::resolve(Name::standard, request("fp8"), kCuda);
    check(t.precision == Precision::fp8 && t.note == "explicit non-native KV under exact: not the exact reference",
          "explicit fp8 under exact runs, stamped as not the reference");
    t = kvp::resolve(Name::standard, request("bf16"), kCuda);
    check(t.source == kvp::Source::explicit_option && !t.changes_output() && t.note.empty(), "explicit native: no note");
    // The same inputs give the same resolution: nothing else (slots, memory, context) is an input.
    const auto a = kvp::resolve(Name::turbo, none, kCuda), b = kvp::resolve(Name::turbo, none, kCuda);
    check(a.json() == b.json(), "deterministic");
}

void refusals() {
    check(refusal([] { (void)kvp::resolve(Name::turbo, request("fp8"), kHip); }) ==
              "qwen-hip: --kv fp8 refused: this engine has no fp8 KV path (supported: bf16|int8; native bf16); "
              "the HIP 8-bit KV form is --kv int8 (kv-int8-tok-v1)",
          "HIP refuses fp8 by name and points at int8");
    check(refusal([] { (void)kvp::resolve(Name::fast, request("fp8"), kBf16Only); }) ==
              "glm-pair: --kv fp8 refused: this engine has no fp8 KV path (supported: bf16; native bf16)",
          "a BF16-only engine refuses fp8");
    check(!refusal([] { (void)kvp::resolve(Name::fast, request("int8"), kBf16Only); }).empty(), "and int8");
    check(refusal([] { (void)kvp::resolve(Name::standard, request("bf16"), kDs41); }).find(
              "deepseek41-pair: --kv bf16 refused: this engine has no bf16 KV path (supported: fp8; native fp8); "
              "the KV format is defined by the model") == 0,
          "a model-defined FP8 KV refuses bf16");
    check(!refusal([] { (void)kvp::resolve(Name::standard, request("fp16"), kCuda); }).empty(), "fp16 without a path");
    check(refusal([] { (void)kvp::resolve(Name::standard, request("fp16"), kF32Native); }).empty(), "fp16 with a path");
    check(strix::ds41::rtx::profile::kKvCaps.native == Precision::fp8, "DS41 RTX: native fp8");
}

void receipts() {
    const auto r = kvp::resolve(Name::turbo, {}, kCuda);
    check(r.identity() == "kv=fp8", "identity field");
    check(r.receipt() == "kv=fp8 native=bf16 source=turbo-default form=kv-e4m3-tok-v1 (e4m3 shadow, attn8) tier=turbo",
          "receipt text");
    check(r.json() == "{\"precision\":\"fp8\",\"native\":\"bf16\",\"source\":\"turbo-default\",\"form\":\"kv-e4m3-tok-v1 (e4m3 "
                      "shadow, attn8)\",\"tier\":\"turbo\",\"changes_output\":true}",
          "receipt json");
    const auto n = kvp::resolve(Name::turbo, {}, kHip);
    check(n.json().find("\"note\":\"turbo default fp8 unavailable") != std::string::npos &&
              kvp::receipt_line(n).starts_with("strix-kv: {\"precision\":\"bf16\""),
          "the note and the receipt line");
}

void serve_config() {
    namespace cf = strix::serve::config_file;
    const cf::Option table[] = {kvp::kServeConfigOption, {"attention_kv_storage", "--attention-kv-storage", cf::Kind::value, 0, {}}};
    auto e = cf::expand_text("kv = \"fp8\"\n", "/cfg/serve.toml", {"bin"}, table);
    check(e.argv.size() == 3 && e.argv[1] == "--kv" && e.argv[2] == "fp8", "kv = \"fp8\" expands to --kv fp8");
    e = cf::expand_text("kv = \"fp8\"\n", "/cfg/serve.toml", {"bin", "--kv", "bf16"}, table);
    check(e.argv.size() == 3 && e.argv[2] == "bf16", "the CLI wins over the file");
}

void cuda_adapter() {
    namespace qp = strix::qwen35::profile;
    // --kv fp8 folds into both attn8 levers on the large-prefill route, decode_attn8 only on the windows route.
    strix::profile::Pairs cli;
    auto kv = qp::resolve_kv(Name::fast, request("fp8"), qp::Route::large_prefill, cli);
    check(cli.size() == 2 && cli[0] == std::pair<std::string, std::string>{"decode_attn8", "on"} &&
              cli[1] == std::pair<std::string, std::string>{"prefill_attn8", "on"} && kv.precision == Precision::fp8,
          "--kv fp8 = attn8 levers on (large prefill)");
    cli.clear();
    (void)qp::resolve_kv(std::nullopt, request("bf16"), qp::Route::windows, cli);
    check(cli.size() == 1 && cli[0] == std::pair<std::string, std::string>{"decode_attn8", "off"}, "--kv bf16 = off (windows)");
    cli.clear();
    (void)qp::resolve_kv(Name::fast, {}, qp::Route::large_prefill, cli);
    check(cli.empty(), "no --kv and no tier default: the levers stay with their own options and cells");
    cli = {{"decode_attn8", "off"}};
    check(refusal([&] { (void)qp::resolve_kv(Name::fast, request("fp8"), qp::Route::large_prefill, cli); }) ==
              "qwen35-cuda-dense: --kv sets decode_attn8; drop the --decode-attn8 option",
          "a lever option together with --kv is refused");
    // observe_kv(): the receipt reports what the resolved levers do.
    cli = {{"decode_attn8", "on"}};
    const auto resolved = qp::resolve(std::nullopt, qp::Regime{qp::Weights::nvfp4_rtn, qp::Route::windows, 64}, cli);
    kv = kvp::resolve(Name::standard, {}, qp::kKvCaps);
    qp::observe_kv(kv, *resolved);
    check(kv.precision == Precision::fp8 && kv.source == kvp::Source::lever && kv.note.find("decode_attn8 on") != std::string::npos,
          "a per-lever option that reads the e4m3 shadow is reported as fp8");
    cli = {{"prefill_attn8", "on"}};
    const auto partial = qp::resolve(std::nullopt, qp::Regime{qp::Weights::nvfp4_rtn, qp::Route::large_prefill, 64}, cli);
    kv = kvp::resolve(Name::standard, {}, qp::kKvCaps);
    qp::observe_kv(kv, *partial);
    check(kv.precision == Precision::bf16 && kv.note.find("prefill_attn8 on: attention reads the e4m3 KV shadow in that phase only") !=
                                                 std::string::npos,
          "one phase on: native precision with a note naming the phase");
}

}  // namespace

int main() {
    try {
        words();
        options();
        tiers();
        refusals();
        receipts();
        serve_config();
        cuda_adapter();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: unexpected exception: %s\n", e.what());
        ++failures;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d kv precision test(s) failed\n", failures);
        return 1;
    }
    std::printf("kv precision tests: ok\n");
    return 0;
}
