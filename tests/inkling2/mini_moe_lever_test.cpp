#include "inkling2/inkling2_mini_resident_route.hpp"
#include "inkling2/inkling2_mini_weight_options.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <string>

namespace {

namespace mini = strix::inkling2;
namespace profile = strix::profile;

void require(bool passed, const char* detail) {
    if (!passed) throw std::runtime_error(detail);
}

// Hermetic environment: only the Inkling switches under test, when given.
profile::Inputs inputs(profile::Name tier, const char* value = nullptr, const char* image = nullptr,
                       const char* fold = nullptr, const char* narrow = nullptr,
                       const char* controller = nullptr) {
    profile::Inputs in;
    in.profile = tier;
    in.env = [value, image, fold, narrow, controller](const char* key) -> const char* {
        if (std::strcmp(key, "STRIX_INK_MOE_W4A4") == 0) return value;
        if (std::strcmp(key, "STRIX_INK_COMBINE_IMAGE") == 0) return image;
        if (std::strcmp(key, "STRIX_INK_PREPARED_ROUND_FOLD") == 0) return fold;
        if (std::strcmp(key, "STRIX_MOE_EXPERTS_NVFP4") == 0) return narrow;
        return std::strcmp(key, "STRIX_INK_CONTROLLER_PLANES") == 0 ? controller : nullptr;
    };
    return in;
}

void test_controller_planes() {
    constexpr std::string_view id = "inkling2-mini-controller-planes@1";
    mini::MiniWeightOptions options;
    for (const auto tier : {profile::Name::standard, profile::Name::fast, profile::Name::turbo}) {
        options.tier = inputs(tier);
        const auto incumbent = mini::resolve_mini_profile(options, true);
        const auto* entry = incumbent.find(id);
        require(entry && entry->value == "3", "controller must keep three planes by default");
        options.tier = inputs(tier, nullptr, nullptr, nullptr, nullptr, "3");
        require(mini::resolve_mini_profile(options, true).fingerprint() == incumbent.fingerprint(),
                "explicit three planes preserves the incumbent identity");
        options.tier = inputs(tier, nullptr, nullptr, nullptr, nullptr, "1");
        bool refused = false;
        try {
            const auto candidate = mini::resolve_mini_profile(options, true);
            require(candidate.find(id)->value == "1" && candidate.fingerprint() != incumbent.fingerprint(),
                    "one plane carries its own candidate identity");
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        require(refused == (tier == profile::Name::standard), "EXACT refuses the unqualified one-plane route");
    }
}

// The engine resolves the W4A4 row: an unbound table would ignore STRIX_INK_MOE_W4A4 at every prompt length.
void test_row() {
    require(mini::kMiniProfile.table.size() == mini::kMiniLevers.size(), "Inkling lever table bound to the engine");
    require(mini::kMiniProfile.table.size() == 8U,
            "W4A4, down image, prepared round fold, narrow experts, controller planes, projected-bias, served route and "
            "the block-candidate row");
    const auto& lever = mini::kMiniProfile.table.front();
    require(lever.name == mini::kMiniMoeW4a4 && lever.name == "inkling2-mini-moe-w4a4@1", "w4a4 identity");
    require(lever.env == "STRIX_INK_MOE_W4A4" && lever.off == "0" && lever.builtin == "0", "w4a4 off switch");
    require(lever.kind == profile::Kind::bit_changing, "bit-changing lever");
    require(lever.value[0].empty() && lever.value[1] == "1" && lever.value[3].empty(), "exact off, fast/turbo on");
    for (std::size_t column = 0; column < profile::kProfileColumns; ++column)
        require(!lever.ready[column] && !lever.provisional[column], "row stays opt-in until its FAST gate");
    require(lever.runnable[1] && lever.runnable[3] && !lever.runnable[0], "fast/turbo columns runnable");
}

// Exact and default fast/turbo keep the BF16 tile; an explicit STRIX_INK_MOE_W4A4=1 opts in (receipt unqualified);
// 0 is the off switch; malformed values are refused by the engine resolver.
void test_resolution() {
    const auto exact = profile::resolve(mini::kMiniProfile, inputs(profile::Name::standard));
    require(!mini::mini_moe_w4a4(exact), "exact keeps the BF16 expert tile");
    require(exact.find(mini::kMiniMoeW4a4) != nullptr, "exact resolution carries the row");
    require(exact.qualification() == "qualified", "exact receipt");
    const profile::Engine legacy{"inkling2-mini", "", {}, false, "inkling2-mini-turbo-v1"};
    require(exact.fingerprint() == profile::resolve(legacy, inputs(profile::Name::standard)).fingerprint(),
            "exact identity unchanged by the additive rows by default");
    const auto exact_image = profile::resolve(mini::kMiniProfile, inputs(profile::Name::standard, nullptr, "1"));
    const auto exact_fold = profile::resolve(mini::kMiniProfile, inputs(profile::Name::standard, nullptr, nullptr, "1"));
    require(exact.fingerprint() != exact_image.fingerprint(), "the selected down image is part of the run identity");
    require(exact.fingerprint() != exact_fold.fingerprint(), "the selected round fold is part of the run identity");
    require(exact_image.fingerprint() != exact_fold.fingerprint(), "the two levers are distinct identities");

    mini::MiniWeightOptions options;
    for (const auto tier : {profile::Name::fast, profile::Name::turbo}) {
        const auto pending = profile::resolve(mini::kMiniProfile, inputs(tier));
        require(pending.find(mini::kMiniMoeW4a4)->status == profile::Status::pending, "pending fast/turbo row");
        require(!mini::mini_moe_w4a4(pending), "pending row keeps the BF16 expert tile by default");
        options.tier = inputs(tier, "1");
        const auto opted = mini::resolve_mini_profile(options, true);
        require(mini::mini_moe_w4a4(opted), "explicit opt-in runs W4A4");
        require(opted.find(mini::kMiniMoeW4a4)->status == profile::Status::explicit_env, "explicit status");
        require(opted.qualification() == "unqualified: " + std::string(mini::kMiniMoeW4a4),
                "explicit opt-in is unqualified");
        options.tier = inputs(tier, "0");
        require(!mini::mini_moe_w4a4(mini::resolve_mini_profile(options, true)), "off switch");
        using A = strix::moe_core::grouped_w4a4::Activations;
        for (const auto& [value, expected] : {std::pair<const char*, A>{"1", A::nvfp4},
                                              {"mse", A::nvfp4_mse},
                                              {"mse-gate-up", A::nvfp4_mse_gate_up},
                                              {"mse-r2-gate-up", A::nvfp4_mse_r2_gate_up},
                                              {"mse-r2", A::nvfp4_mse_r2},
                                              {"r2", A::nvfp4_r2}}) {
            options.tier = inputs(tier, value);
            require(mini::mini_moe_w4a4_class(mini::resolve_mini_profile(options, true)) == expected,
                    "lever value selects the shared activation class");
            require(mini::mini_moe_w4a4(mini::resolve_mini_profile(options, true)), "candidate values run W4A4");
            // Backend admission (CLI, gate, step program, server): CUDA runs the class; HIP refuses it instead of
            // running the BF16 tile under the class's name.
            const auto resolved = mini::resolve_mini_profile(options, true);
            require(mini::admit_mini_moe_w4a4(resolved, "cuda") == expected, "CUDA admits the selected class");
            bool hip_refused = false;
            try {
                static_cast<void>(mini::admit_mini_moe_w4a4(resolved, "hip"));
            } catch (const std::invalid_argument& error) {
                hip_refused = std::string(error.what()).find(strix::moe_core::grouped_w4a4::identity(expected)) !=
                              std::string::npos;
            }
            require(hip_refused, "HIP refuses a selected W4A4 class and names it");
        }
        options.tier = inputs(tier, "0");
        require(mini::admit_mini_moe_w4a4(mini::resolve_mini_profile(options, true), "hip") == A::bf16,
                "HIP admits the off switch");
        options.tier = inputs(tier);
        require(mini::admit_mini_moe_w4a4(mini::resolve_mini_profile(options, true), "hip") == A::bf16,
                "HIP admits the pending default (BF16 tile)");
    }
    bool refused = false;
    try {
        options.tier = inputs(profile::Name::fast, "yes");
        static_cast<void>(mini::resolve_mini_profile(options, true));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "malformed W4A4 switch must be refused");
}

// The down-image lever: off in every profile unless selected explicitly (STRIX_INK_COMBINE_IMAGE=1), one value for every
// entry point (the CLI, the gate and the server resolve through resolve_mini_profile), a malformed value is refused, and
// only gfx1151 admits the image route.
void test_combine_image() {
    const auto* lever = &mini::kMiniProfile.table[1];
    require(lever->name == mini::kMiniCombineImage && lever->name == "inkling2-mini-combine-down-image@1", "image identity");
    require(lever->env == "STRIX_INK_COMBINE_IMAGE" && lever->off == "0" && lever->builtin == "0", "image off switch");
    require(lever->kind == profile::Kind::exact, "image lever is exact");
    for (const auto& cell : lever->value) require(cell.empty(), "no profile column selects the image");
    const profile::Engine bare{"inkling2-mini", "", {}, false, "inkling2-mini-turbo-v1"};
    require(profile::resolve(bare, inputs(profile::Name::standard)).find(mini::kMiniCombineImage) == nullptr &&
                !mini::mini_combine_image(profile::resolve(bare, inputs(profile::Name::standard))),
            "a table without the row keeps the FP32 down rows");
    mini::MiniWeightOptions options;
    for (const auto tier : {profile::Name::standard, profile::Name::fast, profile::Name::turbo}) {
        options.tier = inputs(tier);
        const auto resolved = mini::resolve_mini_profile(options, true);
        require(!mini::mini_combine_image(resolved), "no profile selects the image by default");
        require(resolved.find(mini::kMiniCombineImage)->value == "0", "default value is the builtin 0");
        if (tier == profile::Name::standard) require(resolved.qualification() == "qualified", "exact stays qualified");
        options.tier = inputs(tier, nullptr, "0");
        require(!mini::mini_combine_image(mini::resolve_mini_profile(options, true)), "explicit 0 switches the image off");
        options.tier = inputs(tier, nullptr, "1");
        require(mini::mini_combine_image(mini::resolve_mini_profile(options, true)), "explicit 1 selects the image");
    }
    bool refused = false;
    try {
        options.tier = inputs(profile::Name::standard, nullptr, "yes");
        static_cast<void>(mini::resolve_mini_profile(options, true));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "malformed image switch must be refused");
    require(mini::mini_combine_image_arch("gfx1151"), "gfx1151 is the qualified target");
    for (const char* arch : {"gfx1150", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx115", "gfx11510", ""})
        require(!mini::mini_combine_image_arch(arch), "the image is declined outside gfx1151");
}

// A decoder that records what it was configured with: every entry point configures through configure_mini_decoder.
// `runs` / `folds` are what the target's backend admits of each selected lever.
struct FakeDecoder {
    strix::moe_core::grouped_w4a4::Activations w4a4 = strix::moe_core::grouped_w4a4::Activations::bf16;
    int image_calls = 0;
    bool image = false, runs = true, fold = false, folds = true, projected = false;
    void set_attention_projected_bias(bool enabled) { projected = enabled; }
    bool set_prepared_round_fold(bool selected) { fold = selected; return selected && folds; }
    void set_moe_w4a4(strix::moe_core::grouped_w4a4::Activations a) { w4a4 = a; }
    bool set_combine_image(bool selected) { ++image_calls; image = selected; return selected && runs; }
};

void test_configure() {
    mini::MiniWeightOptions options;
    options.tier = inputs(profile::Name::standard, nullptr, "1");
    const auto resolved = mini::resolve_mini_profile(options, true);
    FakeDecoder decoder;
    require(mini::configure_mini_decoder(decoder, resolved).down_image && decoder.image_calls == 1 && decoder.image,
            "the resolved image lever reaches the decoder");
    require(!decoder.fold, "the round fold stays off while only the image is selected");
    // The projected-bias lever reaches every decoder through the same helper the CLI, the gate and the server call.
    {
        profile::Inputs on = inputs(profile::Name::standard);
        on.env = [](const char* key) -> const char* {
            return std::strcmp(key, "STRIX_INK_ATTN_PROJECTED_BIAS") == 0 ? "1" : nullptr;
        };
        options.tier = on;
        FakeDecoder projected;
        mini::configure_mini_decoder(projected, mini::resolve_mini_profile(options, true));
        require(projected.projected, "the resolved projected-bias lever reaches the decoder through configure_mini_decoder");
    }
    options.tier = inputs(profile::Name::standard);
    FakeDecoder by_default;
    {
        const auto levers = mini::configure_mini_decoder(by_default, mini::resolve_mini_profile(options, true));
        require(!levers.down_image && !levers.prepared_round_fold && !by_default.fold && !by_default.image,
                "no other lever is applied by default");
        require(!by_default.projected, "the projected-bias route is opt-in: off by default");
    }
    options.tier = inputs(profile::Name::standard, nullptr, nullptr, "1");
    FakeDecoder fold_on;
    {
        const auto levers = mini::configure_mini_decoder(fold_on, mini::resolve_mini_profile(options, true));
        require(fold_on.fold && !fold_on.image && levers.prepared_round_fold && !levers.down_image,
                "an explicit round fold reaches the decoder through the same helper and is reported as admitted");
    }
    // A target whose backend does not run the fold (CUDA, any HIP target but the qualified one): the request reaches the
    // decoder, the effective state reported by every entry point is off, and the startup name is "declined".
    FakeDecoder fold_declined;
    fold_declined.folds = false;
    {
        const auto levers = mini::configure_mini_decoder(fold_declined, mini::resolve_mini_profile(options, true));
        require(fold_declined.fold && !levers.prepared_round_fold, "a declined round fold is reported off, not on");
        const auto requested = mini::mini_prepared_round_fold(mini::resolve_mini_profile(options, true));
        require(requested && std::string(mini::mini_round_fold_name(levers.prepared_round_fold, requested)) == "declined",
                "the startup line names a declined round fold");
    }
    FakeDecoder declined;
    declined.runs = false;
    require(!mini::configure_mini_decoder(declined, resolved).down_image && declined.image,
            "a target that does not run the image declines");
    require(std::string(mini::mini_down_image_name(true, true)) == "on" &&
                std::string(mini::mini_down_image_name(false, true)) == "declined" &&
                std::string(mini::mini_down_image_name(false, false)) == "off", "startup field names");
    require(std::string(mini::mini_round_fold_name(true, true)) == "on" &&
                std::string(mini::mini_round_fold_name(false, true)) == "declined" &&
                std::string(mini::mini_round_fold_name(false, false)) == "off", "round fold startup field names");
    options.tier = inputs(profile::Name::standard, nullptr, "0");
    FakeDecoder off;
    require(!mini::configure_mini_decoder(off, mini::resolve_mini_profile(options, true)).down_image && !off.image &&
                off.image_calls == 1, "an off switch is passed through");
    options.tier = inputs(profile::Name::standard, nullptr, nullptr, "0");
    FakeDecoder fold_off;
    require(!mini::configure_mini_decoder(fold_off, mini::resolve_mini_profile(options, true)).prepared_round_fold &&
                !fold_off.fold, "a round fold off switch is passed through");
}

// Where the round fold runs: the qualified HIP target only. The CUDA preparation has no fold, so a selection is declined
// for every architecture string the CUDA backend could see.
void test_round_fold_admission() {
    require(mini::mini_prepared_round_fold_admitted(true, "gfx1151"), "gfx1151 is the qualified target");
    for (const char* arch : {"gfx1100", "gfx1200", "gfx942", "gfx90a", ""})
        require(!mini::mini_prepared_round_fold_admitted(true, arch), "the fold is declined outside gfx1151");
    for (const char* arch : {"gfx1151", "sm_89", "sm_120", ""})
        require(!mini::mini_prepared_round_fold_admitted(false, arch), "the CUDA backend declines the fold");
}

// The projected-bias row: an exact lever, opt-in (off by default in every tier, builtin the image route), bound to the
// engine, switchable by STRIX_INK_ATTN_PROJECTED_BIAS, and refused on a malformed value. Native execution does not
// change the resolution.
void test_projected_bias() {
    const auto& row = mini::kMiniLevers[5];
    require(row.name == mini::kMiniAttnProjectedBias && row.name == "inkling2-mini-attn-projected-bias@1", "row identity");
    require(row.env == "STRIX_INK_ATTN_PROJECTED_BIAS" && row.off == "0" && row.builtin == "0" &&
                row.kind == profile::Kind::exact, "row is an exact lever, off by default");
    for (std::size_t column = 0; column < profile::kProfileColumns; ++column)
        require(row.value[column].empty() && !row.ready[column] && !row.provisional[column] && row.runnable[column],
                "no profile column selects the projected route before its landing-head identity");
    const auto in = [](profile::Name tier, const char* value) {
        profile::Inputs inputs;
        inputs.profile = tier;
        inputs.env = [value](const char* key) -> const char* {
            return value != nullptr && std::strcmp(key, "STRIX_INK_ATTN_PROJECTED_BIAS") == 0 ? value : nullptr;
        };
        return inputs;
    };
    mini::MiniWeightOptions options;
    for (const auto tier : {profile::Name::standard, profile::Name::fast, profile::Name::turbo}) {
        for (const auto& [value, expected] : {std::pair<const char*, bool>{nullptr, false}, {"0", false}, {"1", true}}) {
            options.tier = in(tier, value);
            for (const bool native : {false, true})
                require(mini::mini_attn_projected_bias(mini::resolve_mini_profile(options, native)) == expected,
                        "projected-bias lever resolves one value per tier for every entry point");
        }
    }
    const profile::Engine legacy{"inkling2-mini", "", {}, false, "inkling2-mini-turbo-v1"};
    require(profile::resolve(mini::kMiniProfile, in(profile::Name::standard, nullptr)).fingerprint() ==
                profile::resolve(legacy, in(profile::Name::standard, nullptr)).fingerprint(),
            "default identity unchanged: the default is the row's builtin, which the fingerprint omits");
    require(profile::resolve(mini::kMiniProfile, in(profile::Name::standard, "1")).fingerprint() !=
                profile::resolve(legacy, in(profile::Name::standard, nullptr)).fingerprint(),
            "an explicit opt-in is a distinct run identity");
    options.tier = in(profile::Name::fast, "yes");
    bool refused = false;
    try {
        static_cast<void>(mini::resolve_mini_profile(options, true));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "malformed projected-bias switch must be refused");
}

// The narrow expert image row: bit-changing, off in EXACT, pending (off) in FAST and TURBO until its gate and served
// receipt; an explicit STRIX_MOE_EXPERTS_NVFP4=1 opts in (unqualified), 0 is the off switch, other values are refused.
void test_narrow_experts() {
    const auto& lever = mini::kMiniProfile.table[3];
    require(lever.name == mini::kMiniExpertsNvfp4 && lever.name == "moe.experts.mxfp8-nvfp4@1", "narrow identity");
    require(lever.env == "STRIX_MOE_EXPERTS_NVFP4" && lever.off == "0" && lever.builtin == "0", "narrow off switch");
    require(lever.kind == profile::Kind::bit_changing && lever.value[0].empty() && lever.value[3] == "1",
            "narrow: exact off, turbo target on");
    for (std::size_t column = 0; column < profile::kProfileColumns; ++column)
        require(!lever.ready[column] && !lever.provisional[column], "narrow row stays opt-in until measured");
    require(!mini::mini_experts_nvfp4(profile::resolve(mini::kMiniProfile, inputs(profile::Name::standard))),
            "exact keeps the checkpoint expert image");
    mini::MiniWeightOptions options;
    options.tier = inputs(profile::Name::standard, nullptr, nullptr, nullptr, "1");
    bool exact_refused = false;
    try {
        static_cast<void>(mini::resolve_mini_profile(options, true));
    } catch (const std::invalid_argument& error) {
        exact_refused = std::string_view(error.what()).find("forbidden under --profile exact") != std::string_view::npos;
    }
    require(exact_refused, "EXACT must refuse the explicit narrow image");
    options.tier = inputs(profile::Name::standard, nullptr, nullptr, nullptr, "0");
    require(!mini::mini_experts_nvfp4(mini::resolve_mini_profile(options, true)), "EXACT admits explicit off");
    for (const auto tier : {profile::Name::fast, profile::Name::turbo}) {
        require(!mini::mini_experts_nvfp4(profile::resolve(mini::kMiniProfile, inputs(tier))),
                "pending row keeps the checkpoint expert image by default");
        options.tier = inputs(tier, nullptr, nullptr, nullptr, "1");
        const auto opted = mini::resolve_mini_profile(options, true);
        require(mini::mini_experts_nvfp4(opted), "explicit opt-in loads the narrow image");
        require(opted.qualification() == "unqualified: " + std::string(mini::kMiniExpertsNvfp4),
                "explicit narrow opt-in is unqualified");
        require(!mini::mini_moe_w4a4(opted), "the narrow switch leaves the W4A4 lever alone");
        options.tier = inputs(tier, nullptr, nullptr, nullptr, "0");
        require(!mini::mini_experts_nvfp4(mini::resolve_mini_profile(options, true)), "narrow off switch");
    }
    bool refused = false;
    try {
        options.tier = inputs(profile::Name::turbo, nullptr, nullptr, nullptr, "yes");
        static_cast<void>(mini::resolve_mini_profile(options, true));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "malformed narrow switch must be refused");
}
}  // namespace

// The served route row: opt-in in every profile column (EXACT, FAST and TURBO keep the incumbent served route), an
// explicit 1 selects it and labels the run unqualified until its migration is qualified, 0 keeps it off.
void test_served_route() {
    const profile::Lever* lever = nullptr;
    for (const auto& row : mini::kMiniProfile.table)
        if (row.name == mini::kMiniServedCliRoute) lever = &row;
    require(lever != nullptr && lever->name == "inkling2-mini-served-cli-route@1" &&
                lever->env == "STRIX_INK_SERVED_CLI_ROUTE" && lever->builtin == "0" && lever->off == "0" &&
                lever->kind == profile::Kind::bit_changing && lever->omit_builtin_fingerprint,
            "served route row identity");
    for (const auto tier : {profile::Name::standard, profile::Name::fast, profile::Name::turbo}) {
        require(lever->value[static_cast<unsigned>(tier)].empty() && !lever->ready[static_cast<unsigned>(tier)],
                "served route: no profile column selects it");
        for (const char* value : {static_cast<const char*>(nullptr), "0", "1"}) {
            mini::MiniWeightOptions options;
            options.tier.profile = tier;
            options.tier.env = [value](const char* key) -> const char* {
                return std::strcmp(key, "STRIX_INK_SERVED_CLI_ROUTE") == 0 ? value : nullptr;
            };
            const auto resolution = mini::resolve_mini_profile(options, true);
            const bool on = value != nullptr && std::strcmp(value, "1") == 0;
            require(mini::mini_served_cli_route(resolution) == on, "served route: explicit opt-in only");
            require((resolution.qualification() == "qualified") == !on, "served route: an opt-in run is unqualified");
            // The command line's resolution: the request stays visible on its own line, never as an effective lever.
            auto cli = resolution;
            mini::mini_cli_decline_served_route(cli);
            require(!mini::mini_served_cli_route(cli) && cli.qualification() == "qualified" &&
                        cli.summary().find("served-cli-route") == std::string::npos,
                    "served route: the command line's summary and qualification carry no server-only setting");
            require(mini::mini_served_route_cli_line(cli) ==
                        std::string("strix-served-route/v1 requested ") + (on ? "1" : "0") +
                            " effective n/a (strix-serve only; the command line runs its own route)",
                    "served route: the command line reports the lever as not applicable to its route");
        }
        // A malformed setting is the server's refusal only: the shared resolution (the command line's) does not abort.
        mini::MiniWeightOptions options;
        options.tier.profile = tier;
        options.tier.env = [](const char* key) -> const char* {
            return std::strcmp(key, "STRIX_INK_SERVED_CLI_ROUTE") == 0 ? "2" : nullptr;
        };
        const auto resolution = mini::resolve_mini_profile(options, true);
        bool refused = false;
        try {
            (void)mini::mini_served_cli_route(resolution);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        auto cli = resolution;
        mini::mini_cli_decline_served_route(cli);
        require(refused && mini::mini_served_route_cli_line(cli).find("requested 2 effective n/a") != std::string::npos,
                "served route: a malformed setting is refused by the server and reported by the command line");
    }
}

int main() {
    try {
        test_row();
        test_served_route();
        test_resolution();
        test_combine_image();
        test_controller_planes();
        test_configure();
        test_round_fold_admission();
        test_projected_bias();
        test_narrow_experts();
    } catch (const std::exception& error) {
        std::cerr << "inkling2-mini-moe-lever: " << error.what() << '\n';
        return 1;
    }
    std::cout << "inkling2-mini-moe-lever: ok\n";
    return 0;
}
