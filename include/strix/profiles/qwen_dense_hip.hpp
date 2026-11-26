#pragma once
// Qwen 27B dense HIP profile binding for the decode and speculative CLIs.
// Both use the shared profile resolver with the route their native parser selected.
#include "strix/engine_profile.hpp"
#include "strix/qwen35/prefill_windows.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::qwen35::hip_profile {
namespace sp = ::strix::profile;

enum class App { decode, spec };

inline constexpr std::string_view kAllowEnv = "STRIX_QWEN35_HIP_PROFILE_ALLOW_PENDING";
inline constexpr std::string_view kExactWindow = "engine default (64-row EXACT windows, IU4)";
inline constexpr std::string_view kLayersEnv = "STRIX_DENSE_A4W4_LAYERS";
inline constexpr std::string_view kPending = "final gate v2 content-class verdict and matching timing required";
inline constexpr std::string_view kEvidence = "docs/models/qwen3.8-27b/README.md#a4w4-layer-table-hip-fast";

// The historical 8192-token gate covers one held-out text. The two settings form one
// composite candidate, available through explicit controls. Neither cell has a production grant,
// including with --allow-pending. An empty TURBO column inherits the resolved FAST route.
inline constexpr sp::Lever table[] = {
    {.name = "wide_rows", .env = "", .option = "--wide", .value = {"", "512", "", ""}, .builtin = kExactWindow,
     .kind = sp::Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, false, true, false},
     .target = kPending, .off = "16", .source = kEvidence,
     .identity = {{{kExactWindow, "v1"}, {"512", "v1"}, {"16", "v1"}}}},
    {.name = "a4w4_layers", .env = kLayersEnv, .option = "", .value = {"", "48-63", "", ""}, .builtin = "none",
     .kind = sp::Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, false, true, false},
     .target = kPending, .off = "none", .source = kEvidence,
     .identity = {{{"none", "v1"}, {"48-63", "v1"}}}},
    {.name = "prefill_rows", .env = "", .option = "--prefill-rows", .value = {"", "512", "", ""}, .builtin = kExactWindow,
     .kind = sp::Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, false, true, false},
     .target = kPending, .off = "16", .source = kEvidence,
     .identity = {{{kExactWindow, "v1"}, {"512", "v1"}, {"16", "v1"}}}},
};

inline constexpr sp::Engine kDecode{"qwen35-hip-decode", kAllowEnv, std::span(table).first<2>(), false, "qwen35-hip-turbo-v1"};
inline constexpr sp::Engine kSpec{"qwen35-hip-spec", kAllowEnv, std::span(table).last<2>(), false, "qwen35-hip-turbo-v1"};

[[nodiscard]] inline constexpr const sp::Engine& engine(App app) {
    return app == App::decode ? kDecode : kSpec;
}

[[nodiscard]] inline constexpr std::string_view window_lever(App app) {
    return app == App::decode ? "wide_rows" : "prefill_rows";
}

// ---- Where the FAST cells run (decode app) -------------------------------------------------------------------------
// The run outside the lever table, spelled by the app before load. Default: not known -> no FAST cell qualifies.
struct Scope {
    bool one_rank = false;  // no tensor-parallel split (--tp-rank absent): each TP rank prefills alone, not gated
    bool iu4_pack = false;  // the dense projections are the IU4 g64 pack (gguf type 200) the gate and timing ran on
};

// Quality per prompt band (shared regime ledger rows, sp::RegimeGate): gate v2, per content class (prose, code,
// structured; >= 1,024 rows each), reference EXACT --wide-exact 512 (bitwise = the 64-row EXACT windows), one-ulp null.
// Device class gfx1151, one rank, one lane, IU4 pack; the prompt logits do not depend on MTP (the drafter only observes
// the prefill windows), so the MTP mode is not part of the key. Outside every row the cell stays pending.
struct Qualified {
    std::string_view device;
    sp::RegimeGate gate;
};
inline constexpr std::string_view kGate2k =
    "gate v2 2,048 tokens: 12 held-out windows (4 prose, 4 code, 4 structured) x 512 rows, EXACT --wide-exact 512 ref, one-ulp null";
inline constexpr std::string_view kGate8k =
    "gate v2 8,192 tokens: 12 held-out windows (4 prose, 4 code, 4 structured) x 512 rows, EXACT --wide-exact 512 ref, one-ulp null";
inline constexpr std::string_view kGate24k =
    "gate v2 24,576 tokens: 3 held-out texts (1 prose, 1 code, 1 structured) x cuts 24,576 + 24,064 x 512 rows, EXACT --wide-exact 512 ref, one-ulp null";
// wide_rows 512 = the 512-row window alone (arm wide): PASS in every class of every band below. The composite
// (+ a4w4_layers 48-63, arm composite) FAILED at 2,048 (prose KL 0.210, code 0.067, structured 0.071) and at 8,192
// (prose KL 0.179, code 0.065): no A4W4 row, so its FAST cell stays pending. Above 24,576 tokens (the 32,768 EXACT
// reference exceeds a night turn) no row: the previous route.
inline constexpr Qualified kQualified[] = {
    {"gfx1151", {"wide_rows", "512", 1, 2048, kGate2k}},
    {"gfx1151", {"wide_rows", "512", 2049, 8192, kGate8k}},
    {"gfx1151", {"wide_rows", "512", 8193, 24576, kGate24k}},
};

// Timing per band (shared band receipts, LadderRow::band_bound): bound per run to its own reference by the shared
// resolver, so a row decides only on a known execution, gfx1151 prefill, one lane and its band: wide_rows 512 over
// the 64-row EXACT windows (an A4W4 row would chain over the 512-row window once one passes the gate).
// Timing receipts (gfx1151, IU4 pack, one node, --no-mtp --gen 1, one binary and one id set per band, mirrored
// order, kfd evicted 0; prompt ms, EXACT one cold prefill per round, the window warm of --repeat 2, mean of 2 rounds):
inline constexpr double kWide2k = 10271.0 / 2993.0;    // 512-row window over the 64-row EXACT windows, 2,048 tokens
inline constexpr double kWide8k = 67597.0 / 12925.0;   // 8,192 tokens
inline constexpr double kWide24k = 534392.0 / 45431.0;  // 24,576 tokens (EXACT one cold run, its own turn)
inline constexpr double kNoise = 0.02;
inline constexpr std::string_view kTime2k = "qwen_night-tim-c1-2k8k 2,048: EXACT-64 10.27 s, wide-512 2.99 s";
inline constexpr std::string_view kTime8k = "qwen_night-tim-c1-2k8k 8,192: EXACT-64 67.60 s, wide-512 12.93 s";
inline constexpr std::string_view kTime24k = "qwen_night-tim-c2-24k(-exact) 24,576: EXACT-64 534.39 s, wide-512 45.43 s";
[[nodiscard]] inline constexpr sp::LadderRow band_row(std::string_view lever, std::string_view value, std::uint64_t lo,
                                                     std::uint64_t hi, double speedup, std::string_view receipt) {
    return {.lever = lever, .value = value, .tier = sp::Name::fast, .phase = sp::Phase::prefill, .device = "gfx1151",
            .min_context = lo, .max_context = hi, .min_lanes = 1, .max_lanes = 1, .speedup = speedup, .noise = kNoise,
            .receipt = receipt, .band_bound = true};
}
inline constexpr sp::LadderRow kBandRows[] = {
    band_row("wide_rows", "512", 1, 2048, kWide2k, kTime2k),
    band_row("wide_rows", "512", 2049, 8192, kWide8k, kTime8k),
    band_row("wide_rows", "512", 8193, 24576, kWide24k, kTime24k),
};

[[nodiscard]] inline const Qualified* qualification(std::string_view lever, std::string_view value,
                                                    const sp::Regime& regime) {
    for (const auto& q : kQualified)
        if (q.device == regime.device && sp::regime_covers(q.gate, lever, value, regime.context)) return &q;
    return nullptr;
}

// Bit 0: the window cell is qualified in this regime; bit 1: the A4W4 layer cell (only on top of the window cell).
[[nodiscard]] inline unsigned qualified_mask(App app, const sp::Regime& regime, const Scope& scope) {
    if (app != App::decode || !scope.one_rank || !scope.iu4_pack || regime.phase != sp::Phase::prefill ||
        regime.lanes != 1 || regime.execution.empty())
        return 0U;
    const auto fast = static_cast<unsigned>(sp::Name::fast);
    const bool window = qualification(table[0].name, table[0].value[fast], regime) != nullptr;
    const bool layers = window && qualification(table[1].name, table[1].value[fast], regime) != nullptr;
    return (window ? 1U : 0U) | (layers ? 2U : 0U);
}

// The decode engine with the FAST readiness of this regime. Static storage: a Resolution keeps pointers to its
// engine and levers.
[[nodiscard]] inline const sp::Engine& scoped_engine(App app, unsigned mask) {
    if (app != App::decode || mask == 0U) return engine(app);
    static const auto rows = [] {
        std::array<std::array<sp::Lever, 2>, 4> out{};
        const auto fast = static_cast<unsigned>(sp::Name::fast);
        for (unsigned m = 0; m < 4U; ++m) {
            out[m] = {table[0], table[1]};
            out[m][0].ready[fast] = out[m][0].runnable[fast] = (m & 1U) != 0U;
            out[m][1].ready[fast] = out[m][1].runnable[fast] = (m & 2U) != 0U;
        }
        return out;
    }();
    static const auto make = [](unsigned m) {
        return sp::Engine{kDecode.tag, kAllowEnv, std::span<const sp::Lever>(rows[m]), false, kDecode.turbo_version,
                          {}, {}, std::span<const sp::LadderRow>(kBandRows)};
    };
    static const std::array<sp::Engine, 4> engines{make(0U), make(1U), make(2U), make(3U)};
    return engines[mask & 3U];
}

template <class NextRows>
bool take_window_flag(App app, std::string_view arg, const NextRows& next_rows, PrefillFlags& flags) {
    if (app == App::decode && (arg == "--wide" || arg == "--wide-exact")) {
        flags.width = next_rows();
        flags.exact = arg == "--wide-exact";
        return true;
    }
    if (app == App::spec && arg == "--prefill-rows") {
        flags.width = next_rows();
        return true;
    }
    if (app == App::spec && arg == "--prefill-exact") {
        flags.exact = true;
        return true;
    }
    return false;
}

// Feed the native parser's final width, exact modifier and diagnostic route to the resolver.
// This also handles ordered --wide/--wide-exact overrides and the spec app's valueless --prefill-exact.
// The window width a resolution runs: the explicit width, else the window cell the profile applied, else 0 (engine
// default windows).
[[nodiscard]] inline std::uint32_t resolved_width(App app, const sp::Resolution& resolution, const PrefillFlags& flags) {
    if (flags.width || flags.row_path()) return flags.width;
    const auto* window = resolution.find(window_lever(app));
    if (!window || !sp::applied(window->status)) return 0U;
    std::uint32_t width = 0;
    for (const char c : window->value) {
        if (c < '0' || c > '9') return 0U;
        width = width * 10U + static_cast<std::uint32_t>(c - '0');
    }
    return width;
}

// Hand the window the profile applied to the engine flags (an explicit width or row path is already there).
inline void apply_window(App app, const sp::Resolution& resolution, PrefillFlags& flags) {
    if (flags.width || flags.row_path()) return;
    if (const auto width = resolved_width(app, resolution, flags)) {
        flags.width = width;
        flags.exact = false;
    }
}

[[nodiscard]] inline sp::Resolution resolve_prefill(App app, sp::Inputs inputs, const PrefillFlags& flags,
                                                     std::uint32_t max_rows, const Scope& scope = {}) {
    const auto window = window_lever(app);
    const auto diagnostic_rows = flags.row_path() ? (flags.rows ? flags.rows : 1U) : 0U;
    inputs.cli.clear();  // the parsed width supersedes the shared option scanner's spelling
    if (flags.width || diagnostic_rows)
        inputs.cli.emplace_back(window, std::to_string(flags.width ? flags.width : diagnostic_rows));

    auto resolution = sp::resolve(scoped_engine(app, qualified_mask(app, inputs.regime, scope)), inputs);
    const char* master = sp::environment(inputs.env, "STRIX_DENSE_A4W4");
    const auto width = diagnostic_rows ? 0U : resolved_width(app, resolution, flags);
    const auto fast = static_cast<unsigned>(sp::Name::fast);
    for (auto& entry : resolution.entries) {
        if (entry.lever->name == window) {
            if (diagnostic_rows) entry.reason = "explicit decode row path";
            else if (flags.exact) {
                entry.status = sp::Status::explicit_cli;
                entry.target.clear();
                entry.reason = "explicit exact window";
            } else if (flags.width) entry.reason = "explicit window";
        } else {
            // The profile's A4W4 cell was gated and timed only on its own window cell (the 512-row wide window).
            const auto gated = std::find_if(std::begin(table), std::end(table),
                                            [&](const sp::Lever& l) { return l.name == window; })->value[fast];
            if (entry.status == sp::Status::set || entry.status == sp::Status::unqualified ||
                entry.status == sp::Status::provisional) {
                if (flags.exact || width == 0U || std::to_string(width) != gated ||
                    !::strix::envcfg::parse_flag(master, true)) {
                    entry.value = entry.lever->builtin;
                    entry.status = sp::Status::off;
                    entry.reason = "A4W4 layers run only with the gated 512-row window (STRIX_DENSE_A4W4=0: off)";
                }
                continue;
            }
            const bool listed = entry.status == sp::Status::explicit_env && !entry.value.empty();
            const bool wide = !flags.exact && width > max_rows;
            if (!wide || !::strix::envcfg::parse_flag(master, listed)) {
                if (entry.status == sp::Status::explicit_env) {
                    entry.value = entry.lever->builtin;
                    entry.status = sp::Status::off;
                }
            } else if (!listed && entry.status != sp::Status::off) {
                entry.value = "all";
                entry.requested = entry.value;
                entry.status = sp::Status::explicit_env;
                entry.target.clear();
            }
        }
    }
    return resolution;
}

struct Command {
    sp::Inputs inputs;
    std::vector<std::string> args;

    Command(App app, int argc, char** argv) {
        for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
        sp::take_options(engine(app), args, inputs);
    }
};

[[nodiscard]] inline std::string effective_config(App app, const sp::Resolution& resolution) {
    auto json = resolution.json();
    const auto* window = resolution.find(window_lever(app));
    if (window && !window->reason.empty()) {
        json.pop_back();
        json += ",\"prefill_route\":" + sp::json_string(window->reason) + "}";
    }
    return json;
}

// Complete resolution before loading weights or reading cached arithmetic switches.
inline bool finish(App app, const sp::Resolution& resolution, bool inspect) {
    if (inspect) {
        std::puts(effective_config(app, resolution).c_str());
        return true;
    }
    sp::apply(resolution);
    for (const auto& entry : resolution.entries) {
        if (entry.status == sp::Status::pending)
            std::fprintf(stderr, "profile %s: %s pending (%s); %s runs\n", resolution.word(),
                         std::string(entry.lever->name).c_str(), entry.target.c_str(), entry.value.c_str());
    }
    return false;
}
}  // namespace strix::qwen35::hip_profile
