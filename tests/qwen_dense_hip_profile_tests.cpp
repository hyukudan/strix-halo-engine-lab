// Host coverage of the dense HIP profile binding and prefill routes.
#include "strix/profiles/qwen_dense_hip.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {
namespace q = strix::qwen35;
namespace hp = q::hip_profile;
namespace sp = strix::profile;
using Args = std::vector<std::string>;
constexpr std::uint32_t max_rows = 16;
constexpr std::uint32_t default_rows = 64;
int failures = 0;

void require(bool ok, const char* check) {
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "FAIL %s\n", check);
}

const sp::Lookup no_env = [](const char*) -> const char* { return nullptr; };
const sp::Lookup layers_env = [](const char* key) -> const char* {
    return key == hp::kLayersEnv ? "48-63" : nullptr;
};
const sp::Lookup layers_off = [](const char* key) -> const char* {
    return key == hp::kLayersEnv ? "none" : nullptr;
};

const sp::Entry& window(const sp::Resolution& resolution, hp::App app) {
    return *resolution.find(hp::window_lever(app));
}
const sp::Entry& layers(const sp::Resolution& resolution) {
    return *resolution.find("a4w4_layers");
}

struct Parsed {
    q::PrefillFlags flags;
    sp::Resolution receipt;
};

Parsed run(hp::App app, Args args, const sp::Lookup& env = no_env) {
    args.insert(args.begin(), "strix");
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    hp::Command command(app, static_cast<int>(argv.size()), argv.data());
    command.inputs.env = env;
    Parsed result;
    const auto& remaining = command.args;
    for (std::size_t i = 0; i < remaining.size(); ++i) {
        const auto& arg = remaining[i];
        const auto next_rows = [&] { return static_cast<std::uint32_t>(std::stoul(remaining.at(++i))); };
        if (hp::take_window_flag(app, arg, next_rows, result.flags)) continue;
        if (app == hp::App::decode && arg == "--rows") result.flags.rows = next_rows();
        else if (app == hp::App::decode && arg == "--dump-logits") result.flags.dump = !remaining.at(++i).empty();
        else if (app == hp::App::decode && arg == "--dump-window") result.flags.dump_window = true;
        else if (app == hp::App::decode && arg == "--gen") (void)next_rows();
        else require(false, "unexpected fixture argument");
    }
    result.receipt = hp::resolve_prefill(app, command.inputs, result.flags, max_rows);
    return result;
}

std::uint32_t dumped_prompt_rows(const q::PrefillPlan& plan) {
    std::uint32_t rows = 0;
    for (const auto& w : plan.windows) rows += w.tops();
    return rows;
}

void pending_composite_and_exact_identity() {
    const sp::Regime regimes[] = {
        {}, {sp::Phase::prefill, 2048, 1, "gfx1151", "solo-iu4-plain"},
        {sp::Phase::prefill, 8192, 1, "gfx1151", "solo-iu4-plain"},
        {sp::Phase::prefill, 8193, 1, "gfx1151", "solo-iu4-plain"},
        {sp::Phase::prefill, 32500, 1, "gfx1151", "solo-iu4-plain"},
        {sp::Phase::prefill, 8192, 1, "gfx1151", "tp2-iu4-plain"},
        {sp::Phase::prefill, 8192, 2, "gfx1151", "solo-iu4-plain"},
        {sp::Phase::prefill, 8192, 1, "gfx1100", "solo-iu4-plain"},
        {sp::Phase::prefill, 8192, 1, "gfx1151", "solo-q8-plain"},
        {sp::Phase::prefill, 8192, 1, "gfx1151", "solo-iu4-mtp"},
    };
    for (const auto app : {hp::App::decode, hp::App::spec}) {
        const auto bare = run(app, {});
        const auto exact = run(app, {"--profile", "exact"});
        const auto dflt = run(app, {"--profile=default"});
        require(bare.receipt.fingerprint() == exact.receipt.fingerprint() &&
                    exact.receipt.fingerprint() == dflt.receipt.fingerprint(),
                "EXACT and its aliases retain the implicit engine route");
        require(window(exact.receipt, app).value == hp::kExactWindow &&
                    sp::profile_args(exact.receipt).empty(), "EXACT adds no width option");
        for (const auto& regime : regimes) {
            for (const auto tier : {sp::Name::fast, sp::Name::turbo}) {
                sp::Inputs inputs;
                inputs.profile = tier;
                inputs.regime = regime;
                inputs.env = no_env;
                inputs.allow_pending = true;
                const auto r = hp::resolve_prefill(app, inputs, {}, max_rows);
                require(r.profile == tier && window(r, app).status == sp::Status::pending &&
                            layers(r).status == sp::Status::pending && sp::profile_args(r).empty() &&
                            window(r, app).value == window(exact.receipt, app).value &&
                            layers(r).value == layers(exact.receipt).value,
                        "FAST and TURBO fall back together until the composite has a final scoped gate");
                require(hp::effective_config(app, r).find(hp::kPending) != std::string::npos,
                        "fallback receipt names the missing evidence");
            }
        }

        const char* width = app == hp::App::decode ? "--wide" : "--prefill-rows";
        for (const auto* value : {"16", "128", "512"}) {
            const auto overridden = run(app, {"--profile", "turbo", width, value});
            require(window(overridden.receipt, app).value == value &&
                        !sp::applied(layers(overridden.receipt).status),
                    "an explicit width cannot inherit A4W4");
        }
        const auto explicit_composite = run(app, {"--profile", "fast", width, "512"}, layers_env);
        require(layers(explicit_composite.receipt).status == sp::Status::explicit_env &&
                    layers(explicit_composite.receipt).value == "48-63",
                "explicit benchmark can request the two settings");
        const sp::Lookup master_off = [](const char* key) -> const char* {
            return std::string_view(key) == "STRIX_DENSE_A4W4" ? "0" : layers_env(key);
        };
        const auto master_disabled = run(app, {"--profile", "fast", width, "512"}, master_off);
        require(layers(master_disabled.receipt).status == sp::Status::off &&
                    layers(master_disabled.receipt).value == "none",
                "global A4W4 off switch wins with an explicit layer list");
        const auto layers_disabled = run(app, {"--profile", "fast"}, layers_off);
        require(window(layers_disabled.receipt, app).value == hp::kExactWindow &&
                    layers(layers_disabled.receipt).status == sp::Status::off,
                "layer off switch cannot retain an automatic 512-row window");
    }
}

void diagnostic_windows() {
    const auto app = hp::App::decode;
    const auto plain = run(app, {"--profile", "fast", "--dump-logits", "out", "--gen", "1"});
    const auto row_plan = q::plan_prefill(plain.flags, 513, default_rows, max_rows);
    require(window(plain.receipt, app).value == "1" && row_plan.row_path &&
                dumped_prompt_rows(row_plan) == 513, "plain dump preserves all 513 prompt rows");
    const auto rows = run(app, {"--profile", "turbo", "--rows", "4"});
    const auto rows_plan = q::plan_prefill(rows.flags, 513, default_rows, max_rows);
    require(window(rows.receipt, app).value == "4" && rows_plan.row_path && rows_plan.width == 4,
            "explicit row width keeps the diagnostic route");
    const auto wide = run(app, {"--profile", "fast", "--rows", "4", "--wide", "128", "--dump-logits", "out"});
    const auto wide_plan = q::plan_prefill(wide.flags, 640, 128, max_rows);
    require(!wide_plan.row_path && dumped_prompt_rows(wide_plan) == 1, "explicit wide dump keeps the last row");
    const auto full = run(app, {"--profile", "fast", "--wide", "128", "--dump-logits", "out", "--dump-window"});
    require(dumped_prompt_rows(q::plan_prefill(full.flags, 640, 128, max_rows)) == 128,
            "explicit wide dump-window keeps the last window");
}

void exact_window_receipt() {
    const auto check = [](hp::App app, const Args& args) {
        const auto parsed = run(app, args, layers_env);
        const auto& receipt = parsed.receipt;
        require(parsed.flags.exact && parsed.flags.width == 64 && window(receipt, app).value == "64" &&
                    window(receipt, app).status == sp::Status::explicit_cli &&
                    layers(receipt).status == sp::Status::off && layers(receipt).value == "none" &&
                    hp::effective_config(app, receipt).find("\"prefill_route\":\"explicit exact window\"") != std::string::npos,
                "effective config reports the exact width and inactive A4W4");
    };
    check(hp::App::decode, {"--profile", "fast", "--wide", "512", "--wide-exact", "64"});
    check(hp::App::spec, {"--profile", "fast", "--prefill-rows", "64", "--prefill-exact"});
    check(hp::App::spec, {"--profile", "turbo", "--prefill-exact", "--prefill-rows", "64"});
    const auto last = run(hp::App::decode, {"--profile", "fast", "--wide-exact", "64", "--wide", "128"}, layers_env);
    require(!last.flags.exact && window(last.receipt, hp::App::decode).value == "128" &&
                layers(last.receipt).status == sp::Status::explicit_env,
            "last decode width wins in both the receipt and execution");
}

void documentation_receipt(const char* path) {
    std::ifstream input(path);
    require(input.good(), "open the documented timing receipt");
    const std::string text{std::istreambuf_iterator<char>(input), {}};
    const auto start = text.find("Window timing on");
    const auto end = text.find("FAST gate v2", start);
    const auto timing = start == std::string::npos ? std::string() : text.substr(start, end - start);
    require(timing.find("ca2de6ae8") != std::string::npos && timing.find("gfx1151") != std::string::npos &&
                timing.find("MTP enabled") != std::string::npos && timing.find("MHz") != std::string::npos,
            "timing receipt identifies the revision, device, command mode and measured GPU clocks");
    require(timing.find("| 8192 | 115.7 | 547.8 |") != std::string::npos &&
                timing.find("| 32500 | timeout at 470 s | 436.8 |") != std::string::npos,
            "timing receipt uses the matching 8K arms and records 32.5K EXACT as a timeout");
}
}  // namespace

int main(int argc, char** argv) {
    pending_composite_and_exact_identity();
    diagnostic_windows();
    exact_window_receipt();
    if (argc == 2) documentation_receipt(argv[1]);
    else require(false, "timing receipt path required");
    if (failures == 0) std::printf("qwen dense hip profile: PASS\n");
    return failures == 0 ? 0 : 1;
}
