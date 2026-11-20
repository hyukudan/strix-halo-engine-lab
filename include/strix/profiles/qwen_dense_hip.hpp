#pragma once
// Qwen 27B dense HIP product profile: the tier word '--profile exact|default|fast' of the decode and speculative apps,
// resolved by the shared strix::profile::resolve (strix/engine_profile.hpp). This header is data plus one
// argument-vector adapter; the apps call take() once, first thing in main().
//
// One lever, two spellings of the same prompt-window size: `--wide N` (decode app) and `--prefill-rows N`
// (spec app). Window sizes above 16 rows are the FAST wide class (dense-q8a8-f16-v1). The FAST cell is PENDING: no
// composite quality gate receipt exists, so `--profile fast` alone changes nothing today (receipt status pending,
// one notice) unless `--allow-pending` (or STRIX_QWEN35_HIP_PROFILE_ALLOW_PENDING=1) applies it for benchmarking.
// `exact` and `default` are the default column: no option is added, today's behaviour byte for byte. An explicit
// `--wide` / `--prefill-rows` always wins over the word; `--wide-exact` / `--prefill-exact` keep their own meaning
// (EXACT wide windows) and suppress the profile's window size. Promoting the cell = flipping ready[fast], never code.
// Second lever, both apps: the A4W4 layer table (STRIX_DENSE_A4W4_LAYERS, an environment lever exported by take()). Its
// fast cell is QUALIFIED (kFastGates): the wide IU4 GEMMs of layers 48-63 run the shared A4W4 class
// (dense-iu4a4-rot-v1), the other layers keep their path. It only acts on wide windows (rows > 16, not the EXACT
// wide windows), so with today's pending wide cell it needs an explicit --wide / --prefill-rows above 16.
// Pair app: NOT bound. Its --wide default (512) is already the FAST wide class, so `exact` could not mean today's
// behaviour there and the 64-row fast cell would shrink the window; it needs its own table once gated.
// Host-only: no HIP types.
#include "strix/engine_profile.hpp"

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
    {.name = "wide_rows", .env = "", .option = "--wide", .value = {"", "64", "", ""}, .builtin = "16-row windows",
     .kind = sp::Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "64-row FAST wide windows (dense-q8a8-f16-v1)", .off = "16",
     .source = "composite FAST gate (wide prefill rows > 16) pending"},
    {.name = "a4w4_layers", .env = "STRIX_DENSE_A4W4_LAYERS", .option = "", .value = {"", "48-63", "", ""},
     .builtin = "no A4W4 layer (STRIX_DENSE_A4W4=1: every layer, unqualified)", .kind = sp::Kind::bit_changing,
     .ready = {true, true, true, false}, .runnable = {true, true, true, false}, .target = "", .off = "none",
     .source = "FAST gate v2 PASS, layers 48-63 (docs/models/qwen3.8-27b/README.md#a4w4-layer-table-hip-fast)"},
    {.name = "prefill_rows", .env = "", .option = "--prefill-rows", .value = {"", "64", "", ""},
     .builtin = "engine default (64-row EXACT windows, IU4)",
     .kind = sp::Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "64-row FAST wide windows (dense-q8a8-f16-v1)", .off = "16",
     .source = "composite FAST gate (wide prefill rows > 16) pending"},
};
inline constexpr std::string_view kAllowEnv = "STRIX_QWEN35_HIP_PROFILE_ALLOW_PENDING";
// FAST gate ledger (strix::profile::FastGate): the qualified fast cells and their receipts.
inline constexpr sp::FastGate kFastGates[] = {
    {"a4w4_layers", "48-63", "docs/models/qwen3.8-27b/README.md#a4w4-layer-table-hip-fast"},
};

// One engine per app: each app owns only its own spelling of the window lever; both take the A4W4 layer table.
inline sp::Engine engine(App app) {
    const std::span<const sp::Lever> rows(table);
    return {app == App::spec ? "qwen35-hip-spec" : "qwen35-hip-decode", kAllowEnv,
            app == App::spec ? rows.subspan(1U, 2U) : rows.subspan(0U, 2U)};
}

struct Result {
    sp::Resolution resolution;
    std::vector<std::string> args;  // user arguments after the word is consumed, profile options in front
};

// Pure form (host-testable): args without argv[0]. The word and --allow-pending are consumed; the options the
// profile applies are put in front, so the app's own parser sees the user's explicit option last.
inline Result resolve_args(const sp::Engine& eng, std::vector<std::string> args, const sp::Lookup& env = {}) {
    sp::Inputs in;
    in.env = env;
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
inline void take(App app, int& argc, char**& argv) {
    static std::vector<std::string> storage;
    static std::vector<char*> pointers;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    const auto eng = engine(app);
    const auto result = resolve_args(eng, std::move(args));
    sp::apply(result.resolution);
    for (const auto& entry : result.resolution.entries)
        if (entry.status == sp::Status::pending)
            std::fprintf(stderr, "profile %s: %s pending %s (%s runs)\n", sp::name(result.resolution.profile),
                         std::string(entry.lever->name).c_str(), entry.target.c_str(), entry.value.c_str());
    storage.assign(1U, argc > 0 ? argv[0] : "");
    storage.insert(storage.end(), result.args.begin(), result.args.end());
    pointers.clear();
    for (auto& s : storage) pointers.push_back(s.data());
    pointers.push_back(nullptr);
    argc = static_cast<int>(storage.size());
    argv = pointers.data();
}
}  // namespace strix::qwen35::hip_profile
