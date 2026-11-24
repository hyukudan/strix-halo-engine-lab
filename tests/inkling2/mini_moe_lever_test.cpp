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

// Hermetic environment: only the W4A4 expert switch, the down-image switch and the round fold switch, when given.
profile::Inputs inputs(profile::Name tier, const char* value = nullptr, const char* image = nullptr,
                       const char* fold = nullptr) {
    profile::Inputs in;
    in.profile = tier;
    in.env = [value, image, fold](const char* key) -> const char* {
        if (std::strcmp(key, "STRIX_INK_MOE_W4A4") == 0) return value;
        if (std::strcmp(key, "STRIX_INK_COMBINE_IMAGE") == 0) return image;
        return std::strcmp(key, "STRIX_INK_PREPARED_ROUND_FOLD") == 0 ? fold : nullptr;
    };
    return in;
}

// The engine resolves the W4A4 row: an unbound table would ignore STRIX_INK_MOE_W4A4 at every prompt length.
void test_row() {
    require(mini::kMiniProfile.table.size() == mini::kMiniLevers.size(), "Inkling lever table bound to the engine");
    require(mini::kMiniProfile.table.size() == 3U, "the W4A4 row, the down image row and the prepared round fold row");
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
    const auto exact_off = profile::resolve(mini::kMiniProfile, inputs(profile::Name::standard, nullptr, "0", "0"));
    require(exact_off.fingerprint() == profile::resolve(legacy, inputs(profile::Name::standard)).fingerprint(),
            "exact identity unchanged by the additive rows while the down image and the round fold are off");
    require(exact.fingerprint() != exact_off.fingerprint(), "the selected levers are part of the run identity");
    require(profile::resolve(mini::kMiniProfile, inputs(profile::Name::standard, nullptr, "0")).fingerprint() !=
                exact_off.fingerprint(), "the round fold alone is part of the run identity");
    require(profile::resolve(mini::kMiniProfile, inputs(profile::Name::standard, nullptr, nullptr, "0")).fingerprint() !=
                exact_off.fingerprint(), "the down image alone is part of the run identity");

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
                                              {"mse-gate-up", A::nvfp4_mse_gate_up}}) {
            options.tier = inputs(tier, value);
            require(mini::mini_moe_w4a4_class(mini::resolve_mini_profile(options, true)) == expected,
                    "lever value selects the shared activation class");
            require(mini::mini_moe_w4a4(mini::resolve_mini_profile(options, true)), "candidate values run W4A4");
        }
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

// The down-image lever: builtin off, on by the exact and fast rows (turbo takes the fast cell), one value for every entry
// point (the CLI and the server resolve through resolve_mini_profile), an explicit 0 switches it off, a malformed value is
// refused, and only gfx1151 admits the image route.
void test_combine_image() {
    const auto* lever = &mini::kMiniProfile.table[1];
    require(lever->name == mini::kMiniCombineImage && lever->name == "inkling2-mini-combine-down-image@1", "image identity");
    require(lever->env == "STRIX_INK_COMBINE_IMAGE" && lever->off == "0" && lever->builtin == "0", "image off switch");
    require(lever->kind == profile::Kind::exact, "image lever is exact");
    require(lever->value[0] == "1" && lever->value[1] == "1" && lever->value[3].empty(), "exact and fast on, turbo cell empty");
    const profile::Engine bare{"inkling2-mini", "", {}, false, "inkling2-mini-turbo-v1"};
    require(profile::resolve(bare, inputs(profile::Name::standard)).find(mini::kMiniCombineImage) == nullptr &&
                !mini::mini_combine_image(profile::resolve(bare, inputs(profile::Name::standard))),
            "a table without the row keeps the FP32 down rows");
    mini::MiniWeightOptions options;
    for (const auto tier : {profile::Name::standard, profile::Name::fast, profile::Name::turbo}) {
        options.tier = inputs(tier);
        const auto resolved = mini::resolve_mini_profile(options, true);
        require(mini::mini_combine_image(resolved), "exact, fast and turbo rows select the image");
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
struct FakeDecoder {
    strix::moe_core::grouped_w4a4::Activations w4a4 = strix::moe_core::grouped_w4a4::Activations::bf16;
    int image_calls = 0;
    bool image = false, runs = true, fold = false;
    void set_prepared_round_fold(bool selected) { fold = selected; }
    void set_moe_w4a4(strix::moe_core::grouped_w4a4::Activations a) { w4a4 = a; }
    bool set_combine_image(bool selected) { ++image_calls; image = selected; return selected && runs; }
};

void test_configure() {
    mini::MiniWeightOptions options;
    options.tier = inputs(profile::Name::standard);
    const auto resolved = mini::resolve_mini_profile(options, true);
    FakeDecoder decoder;
    require(mini::configure_mini_decoder(decoder, resolved) && decoder.image_calls == 1 && decoder.image,
            "the resolved image lever reaches the decoder");
    require(decoder.fold, "the resolved round fold lever reaches the decoder through the same helper");
    FakeDecoder declined;
    declined.runs = false;
    require(!mini::configure_mini_decoder(declined, resolved) && declined.image,
            "a target that does not run the image declines");
    require(std::string(mini::mini_down_image_name(true, true)) == "on" &&
                std::string(mini::mini_down_image_name(false, true)) == "declined" &&
                std::string(mini::mini_down_image_name(false, false)) == "off", "startup field names");
    options.tier = inputs(profile::Name::standard, nullptr, "0");
    FakeDecoder off;
    require(!mini::configure_mini_decoder(off, mini::resolve_mini_profile(options, true)) && !off.image && off.image_calls == 1,
            "an off switch is passed through");
    options.tier = inputs(profile::Name::standard, nullptr, nullptr, "0");
    FakeDecoder fold_off;
    static_cast<void>(mini::configure_mini_decoder(fold_off, mini::resolve_mini_profile(options, true)));
    require(!fold_off.fold, "a round fold off switch is passed through");
}

}  // namespace

int main() {
    try {
        test_row();
        test_resolution();
        test_combine_image();
        test_configure();
    } catch (const std::exception& error) {
        std::cerr << "inkling2-mini-moe-lever: " << error.what() << '\n';
        return 1;
    }
    std::cout << "inkling2-mini-moe-lever: ok\n";
    return 0;
}
