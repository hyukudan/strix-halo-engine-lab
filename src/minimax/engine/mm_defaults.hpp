#pragma once
// MiniMax M3.1 defaults-on resolver (optimizations are ON by default in code and are
// switched OFF, never the other way round). Host only, no HIP: the apps and the engine call apply() before they read
// any switch, and the CPU tests diff its output against the measured production receipt.
//
// Optimizations and admission policies resolve ON unless their env names the off value.
// Unqualified admission cells select plain; the item receipt distinguishes policy from measured kernels.
// STRIX_MM_BASELINE=1 is the master switch: every item resolves to its off value (the path before the item); an
// explicit per-item value still wins in both modes. apply() exports each unset item's resolved value into the process
// environment, so the readers in the engine, the exchange modules and the dist layer see exactly what
// --print-effective-config prints.
#include <cstdint>
#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace strix::minimax::defaults {

struct Item {
    const char* name;     // short id (docs/optimizations.json)
    const char* env;      // the switch
    const char* on;       // default value (the measured production value)
    const char* off;      // the value that restores the path before the item
    const char* receipt;  // evidence of the measurement that adopted it
};

inline constexpr const char* kBaselineEnv = "STRIX_MM_BASELINE";

std::span<const Item> items();

// Startup parameter, independent of the baseline switch (async start itself is
// switched off by baseline). Zero restores unconditional grouped async start.
std::size_t pf_async_min_rows(const char* value);
inline bool pf_async_group(bool enabled, std::span<const std::size_t> rows, std::size_t minimum) {
    if (!enabled || rows.empty()) return false;
    for (const auto n : rows) if (n < minimum) return false;
    return true;
}

enum class Source { on, baseline, env };
struct Resolved {
    const Item* item = nullptr;
    std::string value;
    Source source = Source::on;
};

using Lookup = std::function<const char*(const char*)>;

bool baseline(const Lookup& env);                 // STRIX_MM_BASELINE set and not 0
std::vector<Resolved> resolve(const Lookup& env);  // pure: same order as items()

// Process environment, once per process: exports every unset item's resolved value (no overwrite) and prints one line
// when the baseline switch is on.
void apply();
// True when apply() exported `env` (the user did not set it): a default that cannot hold falls back with a notice
// instead of failing.
bool from_defaults(const char* env);

// The engine-shape options of the plain invocation (other options do not change the configuration).
struct Cli {
    std::string app = "rank";  // rank | serve
    std::string dspark_prior;  // explicit --dspark-prior, both apps
    std::string dspark_mode;  // explicit serve CLI, empty = resolver
    std::string transport = "usb4-auto";
    std::uint32_t slots = 8;
    std::uint64_t capacity = 8256;
    std::uint32_t max_rows = 4096;
};
Cli parse_cli(const std::string& app, const std::vector<std::string>& args);  // args without argv[0]

// Canonical text, one key=value per line: CLI shape, route/build switches, every item and resolved serve policy.
// Serve configuration validates scheduler and DSpark policy, independently of process environment export.
std::string effective_config(const Cli& cli, const Lookup& env);
inline constexpr const char* kPrintFlag = "--print-effective-config";
// App entry: when argv holds kPrintFlag, apply() and print effective_config() of the invocation (no model, no GPU)
// and return true (the app exits 0); otherwise apply() and return false. Serve policy is validated
// on both paths before any transport/device/model work; rank invocations do not validate serve policy.
bool print_or_apply(const std::string& app, int argc, char** argv);

}  // namespace strix::minimax::defaults
