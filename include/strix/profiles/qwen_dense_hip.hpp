#pragma once
// Qwen 27B dense HIP product profile: the tier word '--profile exact|default|fast' of the decode and speculative apps,
// resolved by the shared strix::profile::resolve (strix/engine_profile.hpp). This header is data plus one
// argument-vector adapter; the apps call take() once, first thing in main().
//
// One lever, two spellings of the same prompt-window size: `--wide N` (decode app) and `--prefill-rows N`
// (spec app). Window sizes above 16 rows are the FAST wide class (IU4 pack: dense-iu4a16-f16-v1). The FAST cell is
// 512 rows, QUALIFIED by the composite FAST gate v2 (kFastGates): 512-row wide windows with the A4W4 layer table below
// against the real EXACT reference (`--wide-exact 512`), held-out window set, KL 0.0069, top-1 0.971, dNLL -0.0104
// (the wide IU4 class alone: KL 0.00003). It is the window of the published FAST figures. `exact` (spelled `default`
// in receipts) is the default column: no option is added, the engine's 64-row EXACT windows run byte for byte. An
// explicit `--wide` / `--prefill-rows` always wins over the word; `--wide-exact` / `--prefill-exact` keep their own
// meaning (EXACT wide windows) and suppress the profile's window size. Off switch of the FAST cell: `--wide 16`.
// Second lever, both apps: the A4W4 layer table (STRIX_DENSE_A4W4_LAYERS, an environment lever exported by take()). Its
// fast cell is QUALIFIED (kFastGates): the wide IU4 GEMMs of layers 48-63 run the shared A4W4 class
// (dense-iu4a4-rot-v1), the other layers keep their path. It acts on wide windows (rows > 16, not the EXACT wide
// windows): the FAST window cell above is such a window.
// TURBO: no HIP TURBO row is qualified (the kv_fp8 / decode_attn8 cells are CUDA sm_120 bindings), so every TURBO cell
// is empty and takes its FAST cell: `--profile turbo` runs exactly the FAST levers (TURBO nests FAST).
// Pair app: NOT bound. Its --wide default (512) is already the FAST wide class, so `exact` could not mean today's
// behaviour there; it needs its own table.
// Host-only: no HIP types.
#include "strix/engine_profile.hpp"

#include <algorithm>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::qwen35::hip_profile {
namespace sp = ::strix::profile;

enum class App { decode, spec };

// Columns: default, fast, quality, turbo. builtin = today's behaviour with the option absent.
inline constexpr sp::Lever table[] = {
    {.name = "wide_rows", .env = "", .option = "--wide", .value = {"", "512", "", ""}, .builtin = "engine default (64-row EXACT windows, IU4)",
     .kind = sp::Kind::bit_changing, .ready = {true, true, true, false}, .runnable = {true, true, true, false},
     .target = "", .off = "16",
     .source = "FAST gate v2 PASS, 512-row wide windows + A4W4 layers 48-63 vs EXACT --wide-exact 512, held-out set (docs/models/qwen3.8-27b/README.md#a4w4-layer-table-hip-fast)", .identity = {{{"engine default (64-row EXACT windows, IU4)", "v1"}, {"512", "v1"}, {"16", "v1"}}}},
    {.name = "a4w4_layers", .env = "STRIX_DENSE_A4W4_LAYERS", .option = "", .value = {"", "48-63", "", ""},
     .builtin = "no A4W4 layer (STRIX_DENSE_A4W4=1: every layer, unqualified)", .kind = sp::Kind::bit_changing,
     .ready = {true, true, true, false}, .runnable = {true, true, true, false}, .target = "", .off = "none",
     .source = "FAST gate v2 PASS, layers 48-63 (docs/models/qwen3.8-27b/README.md#a4w4-layer-table-hip-fast)", .identity = {{{"no A4W4 layer (STRIX_DENSE_A4W4=1: every layer, unqualified)", "v1"}, {"48-63", "v1"}, {"none", "v1"}}}},
    {.name = "prefill_rows", .env = "", .option = "--prefill-rows", .value = {"", "512", "", ""},
     .builtin = "engine default (64-row EXACT windows, IU4)",
     .kind = sp::Kind::bit_changing, .ready = {true, true, true, false}, .runnable = {true, true, true, false},
     .target = "", .off = "16",
     .source = "FAST gate v2 PASS, 512-row wide windows + A4W4 layers 48-63 vs EXACT --wide-exact 512, held-out set (docs/models/qwen3.8-27b/README.md#a4w4-layer-table-hip-fast)", .identity = {{{"engine default (64-row EXACT windows, IU4)", "v1"}, {"512", "v1"}, {"16", "v1"}}}},
};
inline constexpr std::string_view kAllowEnv = "STRIX_QWEN35_HIP_PROFILE_ALLOW_PENDING";
// FAST gate ledger (strix::profile::FastGate): the qualified fast cells and their receipts.
inline constexpr sp::FastGate kFastGates[] = {
    {"wide_rows", "512", "docs/models/qwen3.8-27b/README.md#a4w4-layer-table-hip-fast"},
    {"a4w4_layers", "48-63", "docs/models/qwen3.8-27b/README.md#a4w4-layer-table-hip-fast"},
    {"prefill_rows", "512", "docs/models/qwen3.8-27b/README.md#a4w4-layer-table-hip-fast"},
};
// Ladder rows: grants of the qualified fast cells, kept with their quality and timing receipts. The window cell is
// faster than the EXACT 64-row windows at every measured length (gfx1151, same binary, same ids: 2K 614 vs 193 tok/s,
// 8K 660 vs 116, 32K finishes in ~65 s where the 64-row EXACT windows exceed the 470 s turn); the receipts are in
// docs/models/qwen3.8-27b/README.md#a4w4-layer-table-hip-fast.
inline constexpr std::string_view kWideGrant =
    "FAST gate v2 PASS, 512-row wide windows + A4W4 48-63 vs EXACT --wide-exact 512, faster than EXACT at 2K/8K/32K "
    "(docs/models/qwen3.8-27b/README.md#a4w4-layer-table-hip-fast)";
inline constexpr sp::LadderRow kLadderRows[] = {
    sp::incumbent_cell("wide_rows", "512", sp::Name::fast, kWideGrant, "v1"),
    sp::incumbent_cell("a4w4_layers", "48-63", sp::Name::fast,
                       "FAST gate v2 PASS, layers 48-63 (docs/models/qwen3.8-27b/README.md#a4w4-layer-table-hip-fast)", "v1"),
    sp::incumbent_cell("prefill_rows", "512", sp::Name::fast, kWideGrant, "v1"),
};
// Versioned TURBO name: every TURBO cell is empty (the FAST cell runs).
inline constexpr std::string_view kTurboVersion = "qwen35-hip-turbo-v1";

// One engine per app: each app owns only its own spelling of the window lever; both take the A4W4 layer table.
inline sp::Engine engine(App app) {
    const std::span<const sp::Lever> rows(table);
    const std::span<const sp::LadderRow> ladder(kLadderRows);  // same order as the table: each app's two rows
    return {app == App::spec ? "qwen35-hip-spec" : "qwen35-hip-decode", kAllowEnv,
            app == App::spec ? rows.subspan(1U, 2U) : rows.subspan(0U, 2U), false, kTurboVersion, {}, {},
            app == App::spec ? ladder.subspan(1U, 2U) : ladder.subspan(0U, 2U)};
}

struct Result {
    sp::Resolution resolution;
    std::vector<std::string> args;  // user arguments after the word is consumed, profile options in front
};

// Pure form (host-testable): args without argv[0]. The word and --allow-pending are consumed; the options the
// profile applies are put in front, so the app's own parser sees the user's explicit option last.
inline Result resolve_args(const sp::Engine& eng, std::vector<std::string> args, const sp::Lookup& env = {}, const sp::Regime& regime = {}) {
    sp::Inputs in;
    in.env = env;
    in.regime = regime;
    sp::take_options(eng, args, in);
    Result out{sp::resolve(eng, in), {}};
    bool exact_window = false;
    for (const auto& a : args) exact_window = exact_window || a == "--wide-exact" || a == "--prefill-exact";
    if (!exact_window) out.args = sp::profile_args(out.resolution);
    out.args.insert(out.args.end(), args.begin(), args.end());
    return out;
}

// App entry: rewrites argc/argv in place (storage lives for the process), exports the environment levers the profile
// applies (strix::profile::apply), prints the pending/unqualified notices.
inline bool take(App app, int& argc, char**& argv) {
    static std::vector<std::string> storage;
    static std::vector<char*> pointers;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    const auto eng = engine(app);
    const auto result = resolve_args(eng, std::move(args));
    if (std::ranges::find(result.args, "--print-effective-config") != result.args.end()) {
        std::puts(result.resolution.json().c_str());
        return true;
    }
    sp::apply(result.resolution);
    for (const auto& entry : result.resolution.entries) {
        if (!entry.ladder.empty()) std::fprintf(stderr, "%s\n", entry.ladder.c_str());
        if (entry.status == sp::Status::pending)
            std::fprintf(stderr, "profile %s: %s pending %s (%s runs)\n", sp::name(result.resolution.profile),
                         std::string(entry.lever->name).c_str(), entry.target.c_str(), entry.value.c_str());
    }
    storage.assign(1U, argc > 0 ? argv[0] : "");
    storage.insert(storage.end(), result.args.begin(), result.args.end());
    pointers.clear();
    for (auto& s : storage) pointers.push_back(s.data());
    pointers.push_back(nullptr);
    argc = static_cast<int>(storage.size());
    argv = pointers.data();
    return false;
}
}  // namespace strix::qwen35::hip_profile
