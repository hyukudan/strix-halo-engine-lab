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
    expect(!defaults.prefetch_q8_lm_head && !defaults.wave8_q8_lm_head && !defaults.wave8_q8_gdn_input,
        "default never selects a new organization");
    const auto resolve = [&](Options& options, bool has_a8) {
        return resolve_optimization_profile(options, model, info, manifest,
            validated_model_content_identity, has_a8);
    };
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
    for (const auto* flag : {"--prefetch-q8-lm-head", "--wave8-q8-lm-head", "--wave8-q8-gdn-input"})
        rejects([&] { (void)parse({"decoder", "m", "1", flag, flag}); }, "duplicate organization option rejects");
    rejects([] { (void)parse({"decoder", "m", "1", "--prefetch-q8-lm-head", "--execution-mode", "graph"}); }, "graph rejects before model IO");
    rejects([] { (void)parse({"decoder", "m", "1", "--wave8-q8-gdn-input", "--integer-lm-head", "a8-iu8"}); }, "integer head excluded before model IO");
    auto missing_gdn = parse({"decoder", "m", "1", "--optimization-profile", "manual", "--wave8-q8-gdn-input"});
    (void)resolve(missing_gdn, false);
    rejects([&] { validate_resolved_q8_projection_organization(missing_gdn); }, "manual GDN requires both existing input prerequisites");
    return failures ? 1 : 0;
}
