#include "inkling2/inkling2_mini_resident_route.hpp"
#include "inkling2/inkling2_mini_weight_options.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <utility>
#include <string>

namespace {

namespace mini = strix::inkling2;
namespace profile = strix::profile;

void require(bool passed, const char* detail) {
    if (!passed) throw std::runtime_error(detail);
}

// The process-wide model preset switch strix-serve exports around its engine (restored on scope exit).
struct PresetEnv {
    PresetEnv() { set("1"); }
    ~PresetEnv() { set(nullptr); }
    PresetEnv(const PresetEnv&) = delete;
    PresetEnv& operator=(const PresetEnv&) = delete;
    static void set(const char* value) {
#ifdef _MSC_VER
        (void)_putenv_s(profile::kPresetEnv, value ? value : "");
#else
        if (value) (void)::setenv(profile::kPresetEnv, value, 1);
        else (void)::unsetenv(profile::kPresetEnv);
#endif
    }
};

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
    require(mini::kMiniProfile.table.size() == 19U,
            "W4A4, down image, prepared round fold, narrow experts, controller planes, projected-bias, served route, "
            "screened head, calibrated low-bit, verify rows, speculation, image recipe, block-candidate, three StrixPack rows, folded scale, rows-select and the SP4 fixed-table head");
    require(mini::kMiniLevers[11].name == mini::kMiniVisionRecipe, "image recipe sits before the block-candidate row");
    require(mini::kMiniLevers[12].name == strix::spec::kBlockDraftLever.name &&
                mini::kMiniLevers[13].name == strix::weight::kAttentionHeadSP4Lever.name &&
                mini::kMiniLevers[14].name == strix::weight::kAttentionHeadSP3Lever.name &&
                mini::kMiniLevers[15].name == strix::weight::kAttentionHeadSP2Lever.name &&
                mini::kMiniLevers[16].name == strix::weight::kLowBitFoldedScaleLever.name &&
                mini::kMiniLevers[17].name == strix::weight::kLowBitRowsSelectLever.name,
            "existing block row retains its slot and StrixPack and both low-bit candidates are bound");
    require(mini::kMiniLevers[18].name == mini::kMiniLowBitHead, "the SP4 fixed-table head row closes the table");
    // The speculation row: exact, builtin auto (the default of every tier, legacy identity), off explicit.
    const auto speculation = std::ranges::find_if(mini::kMiniLevers, [](const profile::Lever& lever) {
        return lever.name == mini::kMiniSpeculation;
    });
    require(speculation != mini::kMiniLevers.end() && speculation->kind == profile::Kind::exact &&
                speculation->builtin == "auto" && speculation->off == "off" &&
                std::ranges::all_of(speculation->value, [](std::string_view v) { return v.empty(); }),
            "speculation row: exact, auto by default (builtin), off explicit");
    for (const auto tier : {profile::Name::standard, profile::Name::fast, profile::Name::turbo}) {
        mini::MiniWeightOptions options;
        options.tier = inputs(tier);
        const auto incumbent = mini::resolve_mini_profile(options, true);
        require(mini::mini_speculation(incumbent), "speculation resolves auto (the builtin)");
        options.tier = inputs(tier);
        options.tier.env = [](const char* key) -> const char* {
            return std::strcmp(key, "STRIX_INK_SPECULATION") == 0 ? "off" : nullptr;
        };
        const auto selected = mini::resolve_mini_profile(options, true);
        require(!mini::mini_speculation(selected) && selected.fingerprint() != incumbent.fingerprint(),
                "STRIX_INK_SPECULATION=off switches speculation off under its own identity, in every tier");
    }
    const auto& lever = mini::kMiniProfile.table.front();
    require(lever.name == mini::kMiniMoeW4a4 && lever.name == "inkling2-mini-moe-w4a4@1", "w4a4 identity");
    require(lever.env == "STRIX_INK_MOE_W4A4" && lever.off == "0" && lever.builtin == "0", "w4a4 off switch");
    require(lever.kind == profile::Kind::bit_changing, "bit-changing lever");
    require(lever.value[0].empty() && lever.value[1] == "1" && lever.value[3] == "iu4", "exact off, fast 1, turbo iu4");
    for (std::size_t column = 0; column < profile::kProfileColumns; ++column)
        require(lever.ready[column] == (column == 3U) && !lever.provisional[column],
                "fast stays opt-in until its FAST gate; turbo iu4 is ready with its declared quality");
    require(profile::ladder_defect(mini::kMiniProfile).empty(), "Inkling ladder rows are well formed");
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
    // The CUDA EXACT defaults (projected bias, screened head) are non-builtin effective values, so the default
    // fingerprint mixes them; with both explicitly 0 (their builtins, which the fingerprint omits) it is the legacy one.
    auto incumbent_inputs = inputs(profile::Name::standard);
    incumbent_inputs.env = [](const char* key) -> const char* {
        return std::strcmp(key, "STRIX_INK_ATTN_PROJECTED_BIAS") == 0 || std::strcmp(key, "STRIX_INK_SCREENED_HEAD") == 0
                   ? "0" : nullptr;
    };
    require(profile::resolve(mini::kMiniProfile, incumbent_inputs).fingerprint() ==
                profile::resolve(legacy, inputs(profile::Name::standard)).fingerprint(),
            "exact identity unchanged by the additive rows with the EXACT defaults switched off");
    require(exact.fingerprint() != profile::resolve(legacy, inputs(profile::Name::standard)).fingerprint(),
            "the EXACT defaults are a distinct run identity from the legacy one");
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
                                              {"r2", A::nvfp4_r2},
                                              {"iu4", A::iu4_rot}}) {
            options.tier = inputs(tier, value);
            require(mini::mini_moe_w4a4_class(mini::resolve_mini_profile(options, true)) == expected,
                    "lever value selects the shared activation class");
            require(mini::mini_moe_w4a4(mini::resolve_mini_profile(options, true)), "candidate values select a class");
            // Backend admission (CLI, gate, step program, server): CUDA runs the class; HIP refuses it instead of
            // running the BF16 tile under the class's name.
            const auto resolved = mini::resolve_mini_profile(options, true);
            // The rotated integer class is the HIP class, the NVFP4 classes are CUDA classes.
            const bool rotated = expected == A::iu4_rot;
            const char* runs = rotated ? "hip" : "cuda";
            const char* refuses = rotated ? "cuda" : "hip";
            require(mini::admit_mini_moe_w4a4(resolved, runs) == expected, "the class's backend admits it");
            bool refused_here = false;
            try {
                static_cast<void>(mini::admit_mini_moe_w4a4(resolved, refuses));
            } catch (const std::invalid_argument& error) {
                refused_here = std::string(error.what()).find(strix::moe_core::grouped_w4a4::identity(expected)) !=
                               std::string::npos;
            }
            require(refused_here, "the other backend refuses a selected class and names it");
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

// TURBO iu4 runs only in the measured prefill regimes of its ladder rows (gfx1151, prompts of 4,307..32,760 tokens);
// every other regime, device and tier keeps the BF16 tile; explicit switches run as set.
void test_ladder() {
    using A = strix::moe_core::grouped_w4a4::Activations;
    mini::MiniWeightOptions options;
    // The execution every entry point spells for the timing rows (mini_ladder_execution): device product, resident
    // context capacity, prefill chunk. The band receipts bind to it per run (LadderRow::band_bound).
    // The host test names the HIP backend: the class is admitted per backend (the build's own backend is CUDA here).
    const mini::MiniPromptExecution known{"gfx1151|product|cu40|mem", 65536U, 128U, "hip"};
    const auto prompt = [&](profile::Name tier, const char* device, std::uint64_t tokens, const char* value = nullptr,
                            const mini::MiniPromptExecution& run = {"gfx1151|product|cu40|mem", 65536U, 128U, "hip"}) {
        options.tier = inputs(tier, value);
        return mini::mini_prompt_moe(options, device, tokens, false, nullptr, run);
    };
    for (const std::uint64_t tokens : {4307ULL, 4309ULL, 8192ULL, 8236ULL, 8237ULL, 32758ULL, 32760ULL})
        require(prompt(profile::Name::turbo, "gfx1151", tokens) == A::iu4_rot, "TURBO prefill inside the rows runs iu4");
    for (const std::uint64_t tokens : {0ULL, 1ULL, 4306ULL, 32761ULL, 131072ULL})
        require(prompt(profile::Name::turbo, "gfx1151", tokens) == A::bf16, "unmeasured prompt sizes keep the BF16 tile");
    require(prompt(profile::Name::turbo, "sm_120", 8236U) == A::bf16, "other devices keep the BF16 tile");
    require(prompt(profile::Name::turbo, "", 8236U) == A::bf16, "an unknown device keeps the BF16 tile");
    require(prompt(profile::Name::fast, "gfx1151", 8236U) == A::bf16, "FAST keeps the BF16 tile");
    require(prompt(profile::Name::standard, "gfx1151", 8236U) == A::bf16, "EXACT keeps the BF16 tile");
    require(prompt(profile::Name::turbo, "gfx1151", 8236U, "0") == A::bf16, "off switch wins in TURBO");
    require(prompt(profile::Name::fast, "gfx1151", 2048U, "iu4") == A::iu4_rot, "explicit iu4 runs as set");
    // The rows are band receipts bound to the run's own execution: an execution that is not known decides nothing.
    for (const auto& unknown : {mini::MiniPromptExecution{"", 0U, 0U, "hip"},
                                mini::MiniPromptExecution{"gfx1151|product", 65536U, 0U, "hip"},
                                mini::MiniPromptExecution{"", 65536U, 128U, "hip"}})
        require(prompt(profile::Name::turbo, "gfx1151", 8236U, nullptr, unknown) == A::bf16,
                "an unknown execution keeps the BF16 tile");
    require(!mini::mini_ladder_execution(known, 8236U).empty() && mini::mini_ladder_execution({"", 0U, 0U, "hip"}, 8236U).empty(),
            "the execution spelling needs the device product and the chunk");
    require(mini::mini_ladder_execution(known, 8236U) != mini::mini_ladder_execution(known, 8237U),
            "the execution spells the prompt");
    for (const auto& row : mini::kMiniProfile.ladder)
        require(row.band_bound && row.over.empty() && !row.incumbent && row.joint.empty(),
                "Inkling timing rows are band receipts");

    profile::Resolution receipt;
    options.tier = inputs(profile::Name::turbo);
    static_cast<void>(mini::mini_prompt_moe(options, "gfx1151", 8236U, false, &receipt, known));
    const auto* entry = receipt.find(mini::kMiniMoeW4a4);
    require(entry != nullptr && entry->status == profile::Status::set, "TURBO iu4 applies as a ready cell");
    require(entry->ladder.find("turbo=iu4 applies x1.091") != std::string::npos, "route line names the measured row");
    auto decode = inputs(profile::Name::turbo);
    decode.regime = {profile::Phase::decode, 8236U, 1, "gfx1151"};
    require(mini::mini_moe_w4a4_class(profile::resolve(mini::kMiniProfile, decode)) == A::bf16, "decode keeps the BF16 tile");

    options.tier = inputs(profile::Name::turbo);
    require(mini::mini_prompt_moe_may_select(options, "gfx1151", A::iu4_rot, known), "a TURBO gfx1151 load builds iu4 images");
    require(!mini::mini_prompt_moe_may_select(options, "gfx1151", A::iu4_rot), "an unknown execution builds none");
    require(!mini::mini_prompt_moe_may_select(options, "sm_120", A::iu4_rot, known), "a TURBO load elsewhere builds none");
    options.tier = inputs(profile::Name::fast);
    require(!mini::mini_prompt_moe_may_select(options, "gfx1151", A::iu4_rot, known), "a FAST load builds none");
    options.tier = inputs(profile::Name::fast, "iu4");
    require(mini::mini_prompt_moe_may_select(options, "sm_120", A::iu4_rot, known), "an explicit iu4 load builds them");

    // strix-serve resolves under a model preset, which admits only ready cells: the server selects what the CLI
    // selects for the same prompt (served prompts of the receipt: 4,307 / 8,234 / 32,758 tokens).
    {
        const PresetEnv preset;
        for (const std::uint64_t tokens : {4307ULL, 8234ULL, 32758ULL})
            require(prompt(profile::Name::turbo, "gfx1151", tokens) == A::iu4_rot, "served TURBO selects the CLI cell");
        require(prompt(profile::Name::turbo, "gfx1151", 2048U) == A::bf16, "served TURBO keeps BF16 outside the rows");
        require(prompt(profile::Name::fast, "gfx1151", 8234U) == A::bf16, "served FAST keeps the BF16 tile");
        options.tier = inputs(profile::Name::turbo);
        require(mini::mini_prompt_moe_may_select(options, "gfx1151", A::iu4_rot, known), "a served TURBO load builds images");
    }
    require(!profile::model_preset(), "preset scope restored");
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
// `runs` / `folds` are what the target's backend admits of each selected lever; `builds` whether the screened plane can
// be built (the decoder's head decision is the shared mini_select_head, as in MiniResidentDecoder::set_screened_head).
struct FakeDecoder {
    strix::moe_core::grouped_w4a4::Activations w4a4 = strix::moe_core::grouped_w4a4::Activations::bf16;
    int image_calls = 0, screen_builds = 0;
    bool image = false, runs = true, fold = false, folds = true, projected = false, builds = true;
    unsigned verify = 0U, verifies = 2U;  // selected verify rows level; highest level this backend admits (0: declines)
    void set_attention_projected_bias(bool enabled) { projected = enabled; }
    bool set_prepared_round_fold(bool selected) { fold = selected; return selected && folds; }
    void set_moe_w4a4(strix::moe_core::grouped_w4a4::Activations a) { w4a4 = a; }
    bool set_combine_image(bool selected) { ++image_calls; image = selected; return selected && runs; }
    mini::MiniHead set_screened_head(bool selected) {
        return mini::mini_select_head(selected, [this] { ++screen_builds; return builds; });
    }
    unsigned set_verify_rows(unsigned selected) { verify = selected; return selected < verifies ? selected : verifies; }
};

void test_configure() {
    mini::MiniWeightOptions options;
    options.tier = inputs(profile::Name::standard, nullptr, "1");
    const auto resolved = mini::resolve_mini_profile(options, true);
    FakeDecoder decoder;
    require(mini::configure_mini_decoder(decoder, resolved).down_image && decoder.image_calls == 1 && decoder.image,
            "the resolved image lever reaches the decoder");
    require(!decoder.fold, "the round fold stays off while only the image is selected");
    // The projected-bias lever reaches every decoder through the same helper the CLI, the gate and the server call; an
    // explicit 0 keeps the incumbent image route.
    for (const char* value : {"1", "0"}) {
        profile::Inputs sel = inputs(profile::Name::standard);
        sel.env = [value](const char* key) -> const char* {
            return std::strcmp(key, "STRIX_INK_ATTN_PROJECTED_BIAS") == 0 ? value : nullptr;
        };
        options.tier = sel;
        FakeDecoder projected;
        mini::configure_mini_decoder(projected, mini::resolve_mini_profile(options, true));
        require(projected.projected == (value[0] == '1'),
                "the resolved projected-bias lever reaches the decoder through configure_mini_decoder");
    }
    options.tier = inputs(profile::Name::standard);
    FakeDecoder by_default;
    {
        const auto levers = mini::configure_mini_decoder(by_default, mini::resolve_mini_profile(options, true));
        require(!levers.down_image && !levers.prepared_round_fold && !by_default.fold && !by_default.image,
                "no other lever is applied by default");
        require(by_default.projected, "the projected-bias route is the default (RTX identity receipts)");
        require(by_default.verify == 0U && levers.verify_rows == 0U, "the verify rows lever is opt-in: off by default");
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
                row.kind == profile::Kind::exact, "row is an exact lever, builtin the image route, off switch 0");
    for (std::size_t column = 0; column < profile::kProfileColumns; ++column) {
        // Default, fast and quality select the projected route (RTX identity receipts); the empty turbo cell inherits fast.
        require(row.value[column] == (column == 3U ? "" : "1") && row.ready[column] && !row.provisional[column] &&
                    row.runnable[column],
                "every profile column selects the projected route after its RTX identity receipts");
    }
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
        for (const auto& [value, expected] : {std::pair<const char*, bool>{nullptr, true}, {"0", false}, {"1", true}}) {
            options.tier = in(tier, value);
            for (const bool native : {false, true})
                require(mini::mini_attn_projected_bias(mini::resolve_mini_profile(options, native)) == expected,
                        "projected-bias lever resolves one value per tier for every entry point");
        }
    }
    // The default is the lever ON (a non-builtin effective value, so the fingerprint mixes it); the row declares no
    // engine identity variant, so the resolution stays the incumbent identity. An explicit 0 is the builtin, which the
    // fingerprint omits: the legacy run identity.
    const profile::Engine legacy{"inkling2-mini", "", {}, false, "inkling2-mini-turbo-v1"};
    const auto legacy_print = profile::resolve(legacy, in(profile::Name::standard, nullptr)).fingerprint();
    // The other CUDA EXACT default (the screened head) is switched off too: its default is a non-builtin value as well.
    profile::Inputs both_off = in(profile::Name::standard, nullptr);
    both_off.env = [](const char* key) -> const char* {
        return std::strcmp(key, "STRIX_INK_ATTN_PROJECTED_BIAS") == 0 || std::strcmp(key, "STRIX_INK_SCREENED_HEAD") == 0
                   ? "0" : nullptr;
    };
    require(profile::resolve(mini::kMiniProfile, both_off).fingerprint() == legacy_print,
            "an explicit 0 keeps the legacy identity: the builtin, which the fingerprint omits");
    require(profile::resolve(mini::kMiniProfile, in(profile::Name::standard, nullptr)).fingerprint() != legacy_print,
            "the default is the lever on, a distinct run identity from the image route");
    require(profile::resolve(mini::kMiniProfile, in(profile::Name::standard, nullptr)).fingerprint() ==
                profile::resolve(mini::kMiniProfile, in(profile::Name::standard, "1")).fingerprint(),
            "an explicit 1 is the default's own run (the fingerprint holds effective values)");
    require(profile::resolve(mini::kMiniProfile, in(profile::Name::standard, nullptr)).incumbent_identity(),
            "the lever is a row, not an identity variant");
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
// The verify rows row: an exact lever, opt-in (off in every tier, builtin the per-row incumbent), bound to the engine,
// switchable by STRIX_INK_VERIFY_ROWS, refused on a malformed value, and reaching the decoder through the one helper.
// The block-candidate row follows the image recipe.
void test_verify_rows() {
    const auto& row = *std::find_if(mini::kMiniLevers.begin(), mini::kMiniLevers.end(),
                                    [](const auto& lever) { return lever.name == mini::kMiniVerifyRows; });
    require(row.name == mini::kMiniVerifyRows && row.name == "inkling2-mini-verify-rows@1", "verify rows row identity");
    require(mini::kMiniLevers[12].name == strix::spec::kBlockDraftLever.name, "the block-candidate row follows the image recipe");
    require(row.env == "STRIX_INK_VERIFY_ROWS" && row.off == "0" && row.builtin == "0" &&
                row.kind == profile::Kind::exact, "verify rows row is an exact lever, off by default");
    for (std::size_t column = 0; column < profile::kProfileColumns; ++column)
        require(row.value[column].empty() && !row.ready[column] && !row.provisional[column] && row.runnable[column],
                "no profile column selects the verify rows route before its device identity");
    const auto in = [](profile::Name tier, const char* value) {
        profile::Inputs inputs;
        inputs.profile = tier;
        inputs.env = [value](const char* key) -> const char* {
            return value != nullptr && std::strcmp(key, "STRIX_INK_VERIFY_ROWS") == 0 ? value : nullptr;
        };
        return inputs;
    };
    mini::MiniWeightOptions options;
    for (const auto tier : {profile::Name::standard, profile::Name::fast, profile::Name::turbo}) {
        for (const auto& [value, expected] :
             {std::pair<const char*, unsigned>{nullptr, 0U}, {"0", 0U}, {"1", 1U}, {"2", mini::kMiniVerifyLanesPick}}) {
            options.tier = in(tier, value);
            for (const bool native : {false, true}) {
                const auto resolved = mini::resolve_mini_profile(options, native);
                require(mini::mini_verify_rows(resolved) == expected,
                        "verify rows lever resolves one value per tier for every entry point");
                FakeDecoder decoder;
                require(mini::configure_mini_decoder(decoder, resolved).verify_rows == expected &&
                            decoder.verify == expected, "the resolved verify rows lever reaches the decoder");
            }
        }
    }
    const profile::Engine legacy{"inkling2-mini", "", {}, false, "inkling2-mini-turbo-v1"};
    profile::Inputs exact_off = in(profile::Name::standard, nullptr);  // the two CUDA EXACT defaults off (their builtins)
    exact_off.env = [](const char* key) -> const char* {
        return std::strcmp(key, "STRIX_INK_ATTN_PROJECTED_BIAS") == 0 || std::strcmp(key, "STRIX_INK_SCREENED_HEAD") == 0
                   ? "0" : nullptr;
    };
    require(profile::resolve(mini::kMiniProfile, exact_off).fingerprint() ==
                profile::resolve(legacy, in(profile::Name::standard, nullptr)).fingerprint(),
            "identity unchanged by the verify rows row (builtin omitted from the fingerprint)");
    require(profile::resolve(mini::kMiniProfile, in(profile::Name::standard, "1")).fingerprint() !=
                profile::resolve(mini::kMiniProfile, in(profile::Name::standard, nullptr)).fingerprint(),
            "an explicit verify rows selection is a distinct identity");
    require(profile::resolve(mini::kMiniProfile, in(profile::Name::standard, "2")).fingerprint() !=
                profile::resolve(mini::kMiniProfile, in(profile::Name::standard, "1")).fingerprint(),
            "the lanes pick level is a distinct identity from level 1");
    // A backend that does not run it (CUDA): the request reaches the decoder, the reported state is declined.
    options.tier = in(profile::Name::standard, "1");
    FakeDecoder cuda;
    cuda.verifies = 0U;
    const auto levers = mini::configure_mini_decoder(cuda, mini::resolve_mini_profile(options, true));
    require(cuda.verify == 1U && levers.verify_rows == 0U &&
                std::string(mini::mini_verify_rows_name(levers.verify_rows, 1U)) == "declined",
            "a declined verify rows lever is reported declined, not on");
    // A decoder without the BF16 screened head runs level 1 of a level-2 selection and says so.
    options.tier = in(profile::Name::standard, "2");
    FakeDecoder rows_only;
    rows_only.verifies = 1U;
    const auto partial = mini::configure_mini_decoder(rows_only, mini::resolve_mini_profile(options, true));
    require(rows_only.verify == 2U && partial.verify_rows == 1U &&
                std::string(mini::mini_verify_rows_name(partial.verify_rows, 2U)) == "on+lanes-pick-declined" &&
                std::string(mini::mini_verify_rows_name(2U, 2U)) == "on+lanes-pick" &&
                std::string(mini::mini_verify_rows_name(1U, 1U)) == "on" &&
                std::string(mini::mini_verify_rows_name(0U, 0U)) == "off",
            "the admitted verify rows level is reported against the selected one");
    // A server reports the lanes pick only where a cohort can run it: two or more admitted slots on the incumbent
    // served route; the served CLI route and a one-slot server run level 1, lower levels pass through.
    require(mini::mini_served_verify_rows(2U, false, 4U) == 2U && mini::mini_served_verify_rows(2U, false, 2U) == 2U &&
                mini::mini_served_verify_rows(2U, false, 1U) == 1U && mini::mini_served_verify_rows(2U, true, 4U) == 1U &&
                mini::mini_served_verify_rows(1U, false, 1U) == 1U && mini::mini_served_verify_rows(0U, true, 1U) == 0U &&
                std::string(mini::mini_verify_rows_name(mini::mini_served_verify_rows(2U, true, 4U), 2U)) ==
                    "on+lanes-pick-declined" &&
                std::string(mini::mini_verify_rows_name(mini::mini_served_verify_rows(2U, false, 1U), 2U)) ==
                    "on+lanes-pick-declined",
            "a server with one admitted slot or on the served CLI route does not report the lanes pick");
    options.tier = in(profile::Name::standard, "3");
    bool refused = false;
    try {
        static_cast<void>(mini::resolve_mini_profile(options, true));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "a malformed verify rows value is refused before any load");
}

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
// r65 #1162.1 and the text-only property: with the image recipe unset, every profile resolves to exactly what the
// table without the row resolves (receipt, notice, qualification, fingerprint), so a text run's identity is unchanged,
// and so does any explicit setting of the row (r72 #1162 P2-1); an image request re-admits it: axial@1 is refused
// there under exact and runs, named and stamped unqualified, under fast and turbo.
void test_vision_recipe_row() {
    std::vector<profile::Lever> before;
    for (const auto& lever : mini::kMiniLevers)
        if (lever.name != mini::kMiniVisionRecipe) before.push_back(lever);
    require(before.size() + 1U == mini::kMiniLevers.size(), "one image recipe row");
    profile::Engine without_row = mini::kMiniProfile;
    without_row.table = before;
    mini::MiniWeightOptions options;
    for (const auto tier : {profile::Name::standard, profile::Name::fast, profile::Name::turbo}) {
        options.tier = inputs(tier);
        const auto with = mini::resolve_mini_profile(options, true);
        auto old = profile::resolve(without_row, inputs(tier));
        // Compare after the same weight admission step: unavailable StrixPack rows
        // are filtered from the effective receipt, independently of the vision row.
        static_cast<void>(mini::admit_mini_strixpack_receipt(old));
        require(with.summary() == old.summary() && with.tier_notice() == old.tier_notice() &&
                    with.qualification() == old.qualification() && with.fingerprint() == old.fingerprint(),
                "unset image recipe: the text-only resolution is the table's without the row");
        for (const char* recipe : {"axial@1", "legacy", "axial"}) {
            options.tier = inputs(tier);
            options.tier.env = [recipe](const char* key) -> const char* {
                return std::strcmp(key, "STRIX_INKLING2_VISION_RECIPE") == 0 ? recipe : nullptr;
            };
            // The image-only row never reaches a text-only run: startup is not refused and the summary, notice,
            // qualification and fingerprint are those of the unset row, under every profile and for any value.
            const auto resolved = mini::resolve_mini_profile(options, true);
            require(resolved.summary() == with.summary() && resolved.tier_notice() == with.tier_notice() &&
                        resolved.qualification() == with.qualification() &&
                        resolved.fingerprint() == with.fingerprint() && !mini::mini_vision_axial(resolved),
                    "an image recipe setting leaves the text-only resolution unchanged");
            // An image request re-admits the setting: axial@1 runs (stamped unqualified) only under fast / turbo.
            bool refused = false;
            try {
                const auto media = mini::mini_vision_media_tier(resolved);
                require(mini::mini_vision_axial(media) == (std::string_view(recipe) == "axial@1"), "recipe value");
                if (std::string_view(recipe) == "legacy")
                    require(media.fingerprint() == with.fingerprint(), "explicit legacy keeps the identity");
                else
                    require(media.qualification().find("unqualified: inkling2-vision-recipe") != std::string::npos,
                            "axial is stamped unqualified");
            } catch (const std::invalid_argument&) {
                refused = true;
            }
            require(refused == (std::string_view(recipe) == "axial" ||
                                (tier == profile::Name::standard && std::string_view(recipe) == "axial@1")),
                    "an image request under EXACT refuses only the unqualified axial recipe; a malformed value is refused");
        }
    }
}
// The screened-head row: an exact lever, the default in every tier after its RTX identity receipts (builtin the full
// head, which an explicit 0 keeps), bound to the engine after main's rows (the block-candidate row retains its original
// slot), switchable by STRIX_INK_SCREENED_HEAD, refused on a malformed value; the default and the explicit 1 are one
// run identity, the explicit 0 keeps the legacy one.
void test_screened_head_row() {
    const profile::Lever* lever = nullptr;
    std::size_t index = 0;
    for (std::size_t i = 0; i < mini::kMiniProfile.table.size(); ++i)
        if (mini::kMiniProfile.table[i].name == mini::kMiniScreenedHead) {
            lever = &mini::kMiniProfile.table[i];
            index = i;
        }
    require(lever != nullptr && lever->name == "inkling2-mini-screened-head@1" && index == 7U &&
                mini::kMiniProfile.table[6].name == mini::kMiniServedCliRoute &&
                mini::kMiniProfile.table[12].name == strix::spec::kBlockDraftLever.name,
            "screened head row identity and position (the block-candidate row follows the image recipe)");
    require(lever->env == "STRIX_INK_SCREENED_HEAD" && lever->off == "0" && lever->builtin == "0" &&
                lever->kind == profile::Kind::exact && lever->omit_builtin_fingerprint && lever->option.empty(),
            "screened head row is an exact lever, builtin the full head, off switch 0");
    for (std::size_t column = 0; column < profile::kProfileColumns; ++column)
        require(lever->value[column] == (column == 3U ? "" : "1") && lever->ready[column] &&
                    !lever->provisional[column] && lever->runnable[column],
                "every profile column selects the screened head after its RTX identity receipts (turbo inherits fast)");
    const auto in = [](profile::Name tier, const char* value) {
        profile::Inputs inputs;
        inputs.profile = tier;
        inputs.env = [value](const char* key) -> const char* {
            return value != nullptr && std::strcmp(key, "STRIX_INK_SCREENED_HEAD") == 0 ? value : nullptr;
        };
        return inputs;
    };
    mini::MiniWeightOptions options;
    for (const auto tier : {profile::Name::standard, profile::Name::fast, profile::Name::turbo}) {
        for (const auto& [value, expected] : {std::pair<const char*, bool>{nullptr, true}, {"0", false}, {"1", true}}) {
            options.tier = in(tier, value);
            for (const bool native : {false, true}) {
                const auto resolution = mini::resolve_mini_profile(options, native);
                require(mini::mini_screened_head(resolution) == expected,
                        "screened head lever resolves one value per tier for every entry point");
                require(resolution.incumbent_identity(), "the lever is a row, not an identity variant");
            }
        }
        // The default is on: it shares the explicit 1's identity and differs from the explicit 0 (the builtin, which the
        // fingerprint omits).
        const auto unset = profile::resolve(mini::kMiniProfile, in(tier, nullptr)).fingerprint();
        require(profile::resolve(mini::kMiniProfile, in(tier, "1")).fingerprint() == unset,
                "an explicit 1 is the default's own run identity");
        require(profile::resolve(mini::kMiniProfile, in(tier, "0")).fingerprint() != unset,
                "an explicit 0 keeps the incumbent full head, a distinct run identity from the default");
    }
    const profile::Engine legacy{"inkling2-mini", "", {}, false, "inkling2-mini-turbo-v1"};
    profile::Inputs both_off = in(profile::Name::standard, nullptr);  // the other CUDA EXACT default off too
    both_off.env = [](const char* key) -> const char* {
        return std::strcmp(key, "STRIX_INK_ATTN_PROJECTED_BIAS") == 0 || std::strcmp(key, "STRIX_INK_SCREENED_HEAD") == 0
                   ? "0" : nullptr;
    };
    require(profile::resolve(mini::kMiniProfile, both_off).fingerprint() ==
                profile::resolve(legacy, in(profile::Name::standard, nullptr)).fingerprint(),
            "an explicit 0 keeps the legacy identity: the builtin, which the fingerprint omits");
    for (const char* bad : {"yes", "2"}) {
        options.tier = in(profile::Name::fast, bad);
        bool refused = false;
        try {
            static_cast<void>(mini::resolve_mini_profile(options, true));
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        require(refused, "malformed screened head switch must be refused");
    }
}

// The head decision (MiniResidentDecoder::set_screened_head through configure_mini_decoder): lever off (explicit 0) -> the full head
// without building the plane, even where it could be built; on -> screened; a plane that cannot be built -> the full
// head with the reason; the startup field names each state.
void test_screened_head_decision() {
    int builds = 0;
    const auto buildable = [&] { ++builds; return true; };
    require(mini::mini_select_head(false, buildable) == mini::MiniHead{} && builds == 0,
            "lever off: full head, the plane is never built");
    require(mini::mini_select_head(true, buildable) == mini::MiniHead{true, mini::MiniHeadReason::lever} && builds == 1,
            "lever on: screened head");
    require(mini::mini_select_head(true, [] { return false; }) ==
                mini::MiniHead{false, mini::MiniHeadReason::build_refused},
            "lever on, plane refused: full head and why");
    require(mini::mini_head_field({true, mini::MiniHeadReason::lever}) == "screened lever" &&
                mini::mini_head_field({}) == "full lever-off" &&
                mini::mini_head_field({false, mini::MiniHeadReason::build_refused}) == "full build-refused" &&
                mini::mini_head_field({false, mini::MiniHeadReason::kv_slot}) == "full kv-slot" &&
                mini::mini_head_field({true, mini::MiniHeadReason::hip_binding}) == "screened hip-binding" &&
                mini::mini_head_field({false, mini::MiniHeadReason::hip_binding}) == "full hip-binding",
            "head startup field names");
    // The per-append pick (MiniResidentDecoder::append): under the lever a one-row decode / prefill pick without host
    // scores runs screened; a one-row verify window (verify_top of one row) and the partial-commit replay of one kept
    // row (commit -> append(keep == 1, no scores, verify)) keep the full head; HIP's own binding is unchanged.
    using mini::MiniAppendKind;
    constexpr auto lever = mini::MiniHeadReason::lever;
    constexpr auto hip = mini::MiniHeadReason::hip_binding;
    require(mini::mini_screened_pick(lever, true, MiniAppendKind::decode, true, false, false) &&
                mini::mini_screened_pick(lever, true, MiniAppendKind::prefill, true, false, false),
            "lever: decode / prefill greedy pick runs screened");
    require(!mini::mini_screened_pick(lever, true, MiniAppendKind::verify, true, false, false),
            "lever: one-row verify window and the keep == 1 commit replay keep the full head");
    require(!mini::mini_screened_pick(lever, true, MiniAppendKind::verify, false, false, false) &&
                !mini::mini_screened_pick(lever, true, MiniAppendKind::verify, true, true, false),
            "lever: replay without a pick and multi-row verify keep the full head");
    require(!mini::mini_screened_pick(lever, true, MiniAppendKind::decode, true, false, true) &&
                !mini::mini_screened_pick(lever, false, MiniAppendKind::decode, true, false, false),
            "lever: host scores or an unbuilt plane keep the full head");
    require(mini::mini_screened_pick(hip, true, MiniAppendKind::verify, true, false, false) &&
                !mini::mini_screened_pick(hip, false, MiniAppendKind::verify, true, false, false),
            "HIP binding keeps its rule");
    const auto with = [](const char* value) {
        mini::MiniWeightOptions options;
        options.tier = inputs(profile::Name::standard);
        options.tier.env = [value](const char* key) -> const char* {
            return value != nullptr && std::strcmp(key, "STRIX_INK_SCREENED_HEAD") == 0 ? value : nullptr;
        };
        return mini::resolve_mini_profile(options, true);
    };
    {
        FakeDecoder decoder;
        const auto levers = mini::configure_mini_decoder(decoder, with("0"));
        require(levers.head == mini::MiniHead{} && decoder.screen_builds == 0,
                "0: every entry point keeps the full head and builds no plane");
    }
    for (const char* value : {static_cast<const char*>(nullptr), "1"}) {
        FakeDecoder on;
        require(mini::configure_mini_decoder(on, with(value)).head ==
                        mini::MiniHead{true, mini::MiniHeadReason::lever} &&
                    on.screen_builds == 1,
                "unset or 1: the resolved lever reaches the decoder through configure_mini_decoder");
    }
    for (const char* value : {static_cast<const char*>(nullptr), "1"}) {
        FakeDecoder refused;
        refused.builds = false;
        require(mini::configure_mini_decoder(refused, with(value)).head ==
                    mini::MiniHead{false, mini::MiniHeadReason::build_refused},
                "unset or 1 with a plane that cannot be built: the full head, reported build-refused");
    }
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
        test_vision_recipe_row();
        test_served_route();
        test_resolution();
        test_ladder();
        test_combine_image();
        test_controller_planes();
        test_configure();
        test_round_fold_admission();
        test_projected_bias();
        test_verify_rows();
        test_narrow_experts();
        test_screened_head_row();
        test_screened_head_decision();
    } catch (const std::exception& error) {
        std::cerr << "inkling2-mini-moe-lever: " << error.what() << '\n';
        return 1;
    }
    std::cout << "inkling2-mini-moe-lever: ok\n";
    return 0;
}
