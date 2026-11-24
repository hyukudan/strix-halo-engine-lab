#include "inkling2/inkling2_mini_weight_options.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

// The prepared-attention round fold is resolved once, by the shared profile resolver both launchers call
// (resolve_mini_profile: the serial CLI and the server read the same resolution): off in the lever's builtin, on in
// Inkling 2 Mini's EXACT and FAST rows (TURBO takes the FAST cell), off by STRIX_INK_PREPARED_ROUND_FOLD=0.
namespace {

namespace mini = strix::inkling2;
namespace profile = strix::profile;

void require(bool passed, const char* detail) {
    if (!passed) throw std::runtime_error(detail);
}

profile::Inputs inputs(profile::Name tier, const char* value = nullptr) {
    profile::Inputs in;
    in.profile = tier;
    in.env = [value](const char* key) -> const char* {
        return value != nullptr && std::strcmp(key, "STRIX_INK_PREPARED_ROUND_FOLD") == 0 ? value : nullptr;
    };
    return in;
}

bool fold(profile::Name tier, const char* value = nullptr) {
    mini::MiniWeightOptions options;
    options.tier = inputs(tier, value);
    return mini::mini_prepared_round_fold(mini::resolve_mini_profile(options, true));
}

const profile::Lever& row() {
    for (const auto& lever : mini::kMiniProfile.table)
        if (lever.name == mini::kMiniPreparedRoundFold) return lever;
    throw std::runtime_error("round fold row missing");
}

void test_row() {
    const auto& lever = row();
    require(lever.name == "inkling2-mini-prepared-round-fold@1", "identity");
    require(lever.env == "STRIX_INK_PREPARED_ROUND_FOLD" && lever.off == "0" && lever.builtin == "0",
            "default-off builtin, 0 switches it off");
    require(lever.kind == profile::Kind::exact, "exact schedule trait");
    // The model's data: EXACT and FAST on (quality, with no rows of its own, reads the EXACT column); the TURBO cell stays empty (it takes the FAST cell).
    require(lever.value[0] == "1" && lever.value[1] == "1" && lever.value[2].empty() && lever.value[3].empty(),
            "row cells");
}

void test_resolution() {
    // The lever's builtin (a resolution without the model's row) is off.
    const profile::Engine bare{"inkling2-mini", "", {}, false, "inkling2-mini-turbo-v1"};
    require(bare.table.empty() && !mini::mini_prepared_round_fold(profile::resolve(bare, inputs(profile::Name::standard))),
            "no row: off");
    // The model's rows turn it on.
    require(fold(profile::Name::standard), "EXACT row on");
    require(fold(profile::Name::fast), "FAST row on");
    require(fold(profile::Name::turbo), "TURBO takes the FAST cell");
    require(fold(profile::Name::quality), "an engine without quality rows reads the default column: on");
    // The off switch, and an explicit on, in every column.
    for (const auto tier : {profile::Name::standard, profile::Name::fast, profile::Name::turbo}) {
        require(!fold(tier, "0"), "off switch");
        require(fold(tier, "1"), "explicit on");
    }
    require(!fold(profile::Name::quality, "0"), "off switch in the quality column");
    bool refused = false;
    try {
        static_cast<void>(fold(profile::Name::standard, "yes"));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "malformed switch refused");
    // The fold is a trait of its own: the W4A4 row stays pending and off whatever it says.
    mini::MiniWeightOptions options;
    options.tier = inputs(profile::Name::fast);
    require(!mini::mini_moe_w4a4(mini::resolve_mini_profile(options, true)), "W4A4 unaffected");
    // CLI and server share the resolver: both resolve the same options to the same value and the same receipt.
    for (const auto tier : {profile::Name::standard, profile::Name::fast, profile::Name::turbo}) {
        for (const char* value : {static_cast<const char*>(nullptr), "0", "1"}) {
            mini::MiniWeightOptions cli, serve;
            cli.tier = inputs(tier, value);
            serve.tier = inputs(tier, value);
            const auto a = mini::resolve_mini_profile(cli, true), b = mini::resolve_mini_profile(serve, true);
            require(mini::mini_prepared_round_fold(a) == mini::mini_prepared_round_fold(b) &&
                        a.summary() == b.summary() && a.fingerprint() == b.fingerprint(),
                    "CLI and server resolve alike");
        }
    }
}

// The receipt shows the row: the startup profile line names the lever the EXACT row sets, off or explicit.
void test_receipt() {
    mini::MiniWeightOptions options;
    options.tier = inputs(profile::Name::standard);
    const auto on = mini::resolve_mini_profile(options, true).summary();
    options.tier = inputs(profile::Name::standard, "0");
    const auto off = mini::resolve_mini_profile(options, true).summary();
    require(on.find("prepared-round-fold") != std::string::npos, "on: the profile line names the row");
    require(off.find("prepared-round-fold") == std::string::npos, "off: nothing to report");
    std::cout << "exact (fold on): " << on << "\nexact (STRIX_INK_PREPARED_ROUND_FOLD=0): " << off << '\n';
}

}  // namespace

int main() {
    try {
        test_row();
        test_resolution();
        test_receipt();
    } catch (const std::exception& error) {
        std::cerr << "inkling2-mini-round-fold-lever: " << error.what() << '\n';
        return 1;
    }
    std::cout << "inkling2-mini-round-fold-lever: ok\n";
    return 0;
}
