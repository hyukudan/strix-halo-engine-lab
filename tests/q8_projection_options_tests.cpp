#include "../apps/hip_decode_options.hpp"

#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix;
using namespace strix::decode_cli;
namespace {
int failures = 0;
void expect(bool good, const char* message) {
    if (!good) { ++failures; std::cerr << message << '\n'; }
}
Options parse(std::initializer_list<const char*> args) {
    std::vector<std::string> storage;
    for (const auto* arg : args) storage.emplace_back(arg);
    std::vector<char*> argv;
    for (auto& arg : storage) argv.push_back(arg.data());
    return parse_options(static_cast<int>(argv.size()), argv.data());
}
template<class F> void rejects(F&& action, const char* message) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, message);
}
}
int main() {
    const auto model = ModelContract::qwen38_flash_next();
    gguf::FileInfo info;
    ModelManifest manifest; manifest.architecture = "qwen3next";
    manifest.layout_identity = model_manifest_layout_identity(manifest);
    auto defaults = parse({"decoder", "m", "1"});
    expect(!defaults.prefetch_q8_lm_head && !defaults.wave8_q8_lm_head && !defaults.wave8_q8_gdn_input &&
        !defaults.wave8_q8_gdn_output && !defaults.wave8_q8_qsa_output,
        "default never selects a new organization");
    const auto resolve = [&](Options& options, bool has_a8) {
        return resolve_optimization_profile(options, model, info, manifest,
            validated_model_content_identity, has_a8);
    };
    expect(!defaults.prefetch_hc_projection_next_tile, "HC next-tile overlay defaults off");
    auto hc_next = parse({"decoder", "m", "1", "--prefetch-hc-projection-next-tile"});
    expect(!hc_next.any_low_level_flag && resolve(hc_next, true).automatic_applied,
        "HC next-tile overlays auto without disabling its prerequisites");
    validate_resolved_hc_projection_next_tile(hc_next);
    for (auto member : {&Options::parallel_hyper_mix, &Options::staged_hyper_projections,
                        &Options::fused_staged_hyper_mix, &Options::typed_hyper_projection,
                        &Options::prefetch_typed_hyper_rows}) {
        auto missing = hc_next; missing.*member = false;
        rejects([&] { validate_resolved_hc_projection_next_tile(missing); },
            "HC next-tile rejects each unresolved prerequisite");
    }
    auto hc_fallback = parse({"decoder", "m", "1", "--prefetch-hc-projection-next-tile"});
    expect(resolve(hc_fallback, false).reason == "a8-runtime-not-built", "HC next-tile preserves automatic fallback reason");
    rejects([&] { validate_resolved_hc_projection_next_tile(hc_fallback); }, "fallback cannot invent HC prerequisites");
    auto hc_manual_missing = parse({"decoder", "m", "1", "--optimization-profile", "manual", "--prefetch-hc-projection-next-tile"});
    (void)resolve(hc_manual_missing, false);
    rejects([&] { validate_resolved_hc_projection_next_tile(hc_manual_missing); }, "manual HC overlay needs explicit prerequisites");
    auto hc_manual = parse({"decoder", "m", "1", "--optimization-profile", "manual",
        "--parallel-hyper-mix", "--staged-hyper-projections", "--fused-staged-hyper-mix",
        "--typed-fused-hyper-projections", "--prefetch-typed-hyper-rows", "--prefetch-hc-projection-next-tile"});
    expect(!resolve(hc_manual, false).automatic_applied, "manual HC overlay preserves manual mode");
    validate_resolved_hc_projection_next_tile(hc_manual);
    rejects([] { (void)parse({"decoder", "m", "1", "--prefetch-hc-projection-next-tile", "--prefetch-hc-projection-next-tile"}); }, "duplicate HC overlay rejects");
    rejects([] { (void)parse({"decoder", "m", "1", "--prefetch-hc-projection-next-tile", "--execution-mode", "graph"}); }, "HC overlay rejects graph before model IO");
    rejects([] { (void)parse({"decoder", "m", "1", "--prefetch-hc-projection-next-tile", "--integer-lm-head", "a8-iu8"}); }, "HC overlay rejects integer head before model IO");
    for (const bool wave8 : {false, true}) {
        auto candidate = wave8 ? parse({"decoder", "m", "1", "--prefetch-q8-lm-head", "--wave8-q8-lm-head", "--wave8-q8-gdn-input"})
            : parse({"decoder", "m", "1", "--prefetch-q8-lm-head"});
        expect(!candidate.any_low_level_flag, "new overlays preserve automatic base selection");
        const auto selected = resolve(candidate, true);
        validate_resolved_q8_projection_organization(candidate);
        expect(selected.automatic_applied && candidate.virtual_q8_lm_head &&
            candidate.prefetch_q8_lm_head && candidate.wave8_q8_lm_head == wave8,
            "explicit overlay composes with automatic base without claiming its qualification");
    }
    auto no_a8 = parse({"decoder", "m", "1", "--prefetch-q8-lm-head"});
    expect(resolve(no_a8, false).reason == "a8-runtime-not-built", "missing capability retains original automatic fallback");
    rejects([&] { validate_resolved_q8_projection_organization(no_a8); }, "fallback cannot silently supply missing virtual LM prerequisite");
    auto manual = parse({"decoder", "m", "1", "--optimization-profile", "manual", "--virtual-q8-lm-head",
        "--prefetch-q8-lm-head", "--wave8-q8-lm-head", "--virtual-q8-gdn-input", "--prefetch-q8-gdn-input", "--wave8-q8-gdn-input"});
    expect(!resolve(manual, false).automatic_applied, "manual Q8 candidates need no A8 build");
    validate_resolved_q8_projection_organization(manual);
    rejects([] { (void)parse({"decoder", "m", "1", "--wave8-q8-lm-head"}); }, "wave8 LM requires explicit prefetch option");
    for (const auto* flag : {"--prefetch-q8-lm-head", "--wave8-q8-lm-head", "--wave8-q8-gdn-input",
                             "--wave8-q8-gdn-output", "--wave8-q8-qsa-output"})
        rejects([&] { (void)parse({"decoder", "m", "1", flag, flag}); }, "duplicate organization option rejects");
    rejects([] { (void)parse({"decoder", "m", "1", "--prefetch-q8-lm-head", "--execution-mode", "graph"}); }, "graph rejects before model IO");
    rejects([] { (void)parse({"decoder", "m", "1", "--wave8-q8-gdn-input", "--integer-lm-head", "a8-iu8"}); }, "integer head excluded before model IO");
    auto missing_gdn = parse({"decoder", "m", "1", "--optimization-profile", "manual", "--wave8-q8-gdn-input"});
    (void)resolve(missing_gdn, false);
    rejects([&] { validate_resolved_q8_projection_organization(missing_gdn); }, "manual GDN requires both existing input prerequisites");
    auto output_auto = parse({"decoder", "m", "1", "--wave8-q8-gdn-output", "--wave8-q8-qsa-output"});
    expect(!output_auto.any_low_level_flag, "output organizations overlay the automatic base");
    expect(resolve(output_auto, true).automatic_applied, "output organizations retain qualified automatic base");
    validate_resolved_q8_projection_organization(output_auto);
    expect(output_auto.wave8_q8_gdn_output && output_auto.wave8_q8_qsa_output &&
        output_auto.virtual_q8_gdn_output, "automatic GDN prerequisite resolves before output validation");
    auto output_manual = parse({"decoder", "m", "1", "--optimization-profile", "manual",
        "--virtual-q8-gdn-output", "--wave8-q8-gdn-output", "--wave8-q8-qsa-output"});
    expect(!resolve(output_manual, false).automatic_applied, "manual output modes preserve manual selection");
    validate_resolved_q8_projection_organization(output_manual);
    auto missing_output = parse({"decoder", "m", "1", "--optimization-profile", "manual",
        "--wave8-q8-gdn-output"});
    (void)resolve(missing_output, false);
    rejects([&] { validate_resolved_q8_projection_organization(missing_output); },
        "manual output organization requires explicit virtual output");
    auto output_no_a8 = parse({"decoder", "m", "1", "--wave8-q8-gdn-output"});
    expect(resolve(output_no_a8, false).reason == "a8-runtime-not-built", "output overlay preserves base fallback");
    rejects([&] { validate_resolved_q8_projection_organization(output_no_a8); },
        "fallback cannot manufacture GDN output prerequisite");
    auto qsa_manual = parse({"decoder", "m", "1", "--optimization-profile", "manual",
        "--wave8-q8-qsa-output"});
    (void)resolve(qsa_manual, false);
    validate_resolved_q8_projection_organization(qsa_manual);
    for (const auto* flag : {"--wave8-q8-gdn-output", "--wave8-q8-qsa-output"}) {
        rejects([&] { (void)parse({"decoder", "m", "1", flag, "--execution-mode", "graph"}); },
            "output organizations exclude graph before model IO");
        rejects([&] { (void)parse({"decoder", "m", "1", flag, "--integer-lm-head", "a8-iu8"}); },
            "output organizations exclude integer head before model IO");
    }
    return failures ? 1 : 0;
}
