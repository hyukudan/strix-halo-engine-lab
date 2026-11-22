#include "inkling2/inkling2_mini_resident_route.hpp"
#include "inkling2/inkling2_mini_weight_options.hpp"

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

namespace mini = strix::inkling2;
namespace profile = strix::profile;
using Class = mini::MiniDenseClass;
using Kind = mini::MiniAppendKind;

void require(bool passed, const char* detail) {
    if (!passed) throw std::runtime_error(detail);
}

// Hermetic environment: only the dense lever switch, when given.
profile::Lookup environment(const char* dense) {
    return [dense](const char* key) -> const char* {
        return dense != nullptr && std::strcmp(key, "STRIX_INK_DENSE_MMA1") == 0 ? dense : nullptr;
    };
}

profile::Inputs inputs(profile::Name tier, const char* dense = nullptr) {
    profile::Inputs out;
    out.profile = tier;
    out.env = environment(dense);
    return out;
}

// Class of a large prefill chunk, a one-row prefill call, a decode row and a verify block.
void require_regimes(bool prefill_mma1, const char* detail) {
    const auto expected = prefill_mma1 ? Class::bf16_mma1 : Class::exact_mma3;
    require(mini::mini_dense_class(prefill_mma1, Kind::prefill, 2048U) == expected, detail);
    require(mini::mini_dense_class(prefill_mma1, Kind::prefill, 2U) == expected, detail);
    require(mini::mini_dense_class(prefill_mma1, Kind::prefill, 1U) == Class::exact_mma3, "one-row prefill left mma3");
    require(mini::mini_dense_class(prefill_mma1, Kind::decode, 1U) == Class::exact_mma3, "decode left mma3");
    require(mini::mini_dense_class(prefill_mma1, Kind::decode, 4U) == Class::exact_mma3, "packed decode left mma3");
    require(mini::mini_dense_class(prefill_mma1, Kind::verify, mini::kMiniMaxVerifyRows) == Class::exact_mma3,
            "verify left mma3");
}

void test_row() {
    require(mini::kMiniProfile.table.size() == 3U, "Inkling lever table");
    const auto& lever = mini::kMiniProfile.table.front();
    require(lever.name == mini::kMiniDensePrefillMma1, "lever identity");
    require(lever.name == "inkling2-mini-dense-prefill-bf16-mma1@1", "versioned lever name");
    require(lever.env == "STRIX_INK_DENSE_MMA1" && lever.off == "0" && lever.builtin == "0", "off switch");
    require(lever.kind == profile::Kind::bit_changing, "bit-changing lever");
    require(lever.value[0].empty() && lever.value[1] == "1" && lever.value[3].empty(), "exact off, fast/turbo on");
}

// The shipped row: exact keeps mma3; fast and turbo request mma1 but the row is runnable and not ready (no FAST gate
// receipt), so it is pending, every regime still runs mma3 and the receipt says pending. STRIX_INK_DENSE_MMA1=1 opts in.
void test_shipped() {
    const auto exact = profile::resolve(mini::kMiniProfile, inputs(profile::Name::standard));
    require(!mini::mini_dense_prefill_mma1(exact), "exact must keep mma3");
    require(exact.find(mini::kMiniDensePrefillMma1)->status == profile::Status::builtin, "exact status");
    require(exact.notices().empty() && exact.qualification() == "qualified", "exact receipt");
    const profile::Engine legacy{"inkling2-mini", "", {}, false, "inkling2-mini-turbo-v1"};
    require(exact.fingerprint() == profile::resolve(legacy, inputs(profile::Name::standard)).fingerprint(),
            "exact identity unchanged by the additive row");
    require_regimes(mini::mini_dense_prefill_mma1(exact), "exact regimes");

    for (const auto tier : {profile::Name::fast, profile::Name::turbo}) {
        const auto resolution = profile::resolve(mini::kMiniProfile, inputs(tier));
        const auto* entry = resolution.find(mini::kMiniDensePrefillMma1);
        require(entry->status == profile::Status::pending && entry->target == "1", "pending fast/turbo row");
        require(!mini::mini_dense_prefill_mma1(resolution), "pending row must not run mma1");
        require(resolution.notices().size() == 2U, "one pending notice per pending row (mma1, W4A4)");
        require(mini::mini_decode_split(resolution) == mini::MiniDecodeSplit::automatic, "fast/turbo default: split auto");
        require_regimes(false, "pending regimes");
    }
    const auto& row = mini::kMiniProfile.table.front();
    for (std::size_t column = 0; column < profile::kProfileColumns; ++column)
        require(!row.ready[column] && !row.provisional[column], "row must stay opt-in until its FAST gate");
    require(row.runnable[1] && row.runnable[3] && !row.runnable[0], "fast/turbo columns runnable");
    // The engine resolver accepts the shipped row and the explicit opt-in, stamped unqualified.
    mini::MiniWeightOptions options;
    options.tier = inputs(profile::Name::fast);
    require(!mini::mini_dense_prefill_mma1(mini::resolve_mini_profile(options, true)), "resolver fast");
    for (const auto tier : {profile::Name::standard, profile::Name::fast, profile::Name::turbo}) {
        options.tier = inputs(tier, "1");
        const auto opted = mini::resolve_mini_profile(options, true);
        require(mini::mini_dense_prefill_mma1(opted), "explicit 1 must run mma1 in prefill");
        require(opted.find(mini::kMiniDensePrefillMma1)->status == profile::Status::explicit_env, "explicit status");
        require(opted.qualification() == "unqualified: " + std::string(mini::kMiniDensePrefillMma1),
                "explicit opt-in is unqualified");
        require_regimes(true, "explicit regimes");
        options.tier = inputs(tier, "0");
        require(!mini::mini_dense_prefill_mma1(mini::resolve_mini_profile(options, true)), "explicit 0 keeps mma3");
    }
    // A build without a runnable route refuses the explicit opt-in.
    std::array<profile::Lever, 1> unrunnable{row};
    unrunnable[0].runnable = {};
    const profile::Engine without{"inkling2-mini", "", unrunnable, false, "inkling2-mini-turbo-v1"};
    bool refused = false;
    try {
        static_cast<void>(mini::mini_dense_prefill_mma1(profile::resolve(without, inputs(profile::Name::fast, "1"))));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "explicit 1 without a runnable route must be refused");
    refused = false;
    try {
        options.tier = inputs(profile::Name::fast, "yes");
        static_cast<void>(mini::resolve_mini_profile(options, true));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "malformed switch must be refused");
}

// The same row once it is selected by default (provisional or, after its gate, ready in fast; turbo inherits fast): fast and turbo run mma1
// in multi-row prefill only, exact and the off switch keep mma3.
void test_runnable() {
    std::array<profile::Lever, 1> rows{mini::kMiniProfile.table.front()};
    rows[0].provisional = {false, true, false, true};
    const profile::Engine engine{"inkling2-mini", "", rows, false, "inkling2-mini-turbo-v1"};

    require(!mini::mini_dense_prefill_mma1(profile::resolve(engine, inputs(profile::Name::standard))), "exact mma3");
    require_regimes(false, "exact regimes");
    for (const auto tier : {profile::Name::fast, profile::Name::turbo}) {
        const auto on = profile::resolve(engine, inputs(tier));
        require(on.find(mini::kMiniDensePrefillMma1)->status == profile::Status::provisional, "provisional row");
        require(mini::mini_dense_prefill_mma1(on), "fast/turbo must run mma1 in prefill");
        require_regimes(true, "fast/turbo regimes");
        const auto off = profile::resolve(engine, inputs(tier, "0"));
        require(off.find(mini::kMiniDensePrefillMma1)->status == profile::Status::off, "off status");
        require(!mini::mini_dense_prefill_mma1(off), "off switch must restore mma3");
        require(off.qualification() == "qualified", "off switch stays qualified");
        require(off.fingerprint() != on.fingerprint(), "mma1 changes the profile identity");
        require_regimes(false, "off regimes");
    }
}

}  // namespace

// W4A4 expert row: runnable, not ready. Exact and default fast/turbo keep the BF16 tile; an explicit
// STRIX_INK_MOE_W4A4=1 opts in (fast/turbo); 0 is the off switch; malformed values are refused.
void test_w4a4() {
    const auto& lever = mini::kMiniProfile.table[1];
    require(lever.name == mini::kMiniMoeW4a4 && lever.name == "inkling2-mini-moe-w4a4@1", "w4a4 identity");
    require(lever.env == "STRIX_INK_MOE_W4A4" && lever.off == "0" && lever.builtin == "0", "w4a4 off switch");
    require(lever.kind == profile::Kind::bit_changing && lever.runnable[1] && !lever.ready[1], "w4a4 runnable, not ready");
    const auto with = [](profile::Name tier, const char* value) {
        profile::Inputs in;
        in.profile = tier;
        in.env = [value](const char* key) -> const char* {
            return value != nullptr && std::strcmp(key, "STRIX_INK_MOE_W4A4") == 0 ? value : nullptr;
        };
        return in;
    };
    require(!mini::mini_moe_w4a4(profile::resolve(mini::kMiniProfile, with(profile::Name::standard, nullptr))),
            "exact keeps the BF16 expert tile");
    for (const auto tier : {profile::Name::fast, profile::Name::turbo}) {
        require(!mini::mini_moe_w4a4(profile::resolve(mini::kMiniProfile, with(tier, nullptr))),
                "pending row keeps the BF16 expert tile by default");
        require(mini::mini_moe_w4a4(profile::resolve(mini::kMiniProfile, with(tier, "1"))), "explicit opt-in runs W4A4");
        require(!mini::mini_moe_w4a4(profile::resolve(mini::kMiniProfile, with(tier, "0"))), "off switch");
    }
    bool refused = false;
    try {
        static_cast<void>(mini::mini_moe_w4a4(profile::resolve(mini::kMiniProfile, with(profile::Name::fast, "yes"))));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "malformed W4A4 switch must be refused");
}

// Decode split row (generic id attn.decode.split@1): exact keeps the one-pass flash; fast/turbo run auto (ready after
// gate v2); the generic switch and the legacy STRIX_INKLING_DECODE_SPLIT both set it (generic wins);
// 0 is the off switch; auto / 1 select the crossover / every visit; malformed values are refused. The crossover mask
// follows the shared plan (two chunks): no split below 257 visible keys on either window class.
void test_decode_split() {
    const auto& lever = mini::kMiniProfile.table[2];
    require(lever.name == mini::kMiniDecodeSplit && lever.name == "attn.decode.split@1", "split generic identity");
    require(lever.env == "STRIX_ATTN_DECODE_SPLIT" && lever.off == "0" && lever.builtin == "0", "split off switch");
    require(lever.kind == profile::Kind::bit_changing && lever.value[1] == "auto" && lever.value[0].empty() &&
                lever.ready[1] && lever.ready[3] && !lever.ready[0] && !lever.provisional[1], "split fast cell auto, ready");
    const auto with = [](profile::Name tier, const char* generic, const char* legacy) {
        mini::MiniWeightOptions options;
        options.tier.profile = tier;
        options.tier.env = [generic, legacy](const char* key) -> const char* {
            if (generic != nullptr && std::strcmp(key, "STRIX_ATTN_DECODE_SPLIT") == 0) return generic;
            if (legacy != nullptr && std::strcmp(key, "STRIX_INKLING_DECODE_SPLIT") == 0) return legacy;
            return nullptr;
        };
        return mini::mini_decode_split(mini::resolve_mini_profile(options, true));
    };
    using Split = mini::MiniDecodeSplit;
    require(with(profile::Name::standard, nullptr, nullptr) == Split::off, "exact keeps the one-pass flash");
    for (const auto tier : {profile::Name::fast, profile::Name::turbo})
        require(with(tier, nullptr, nullptr) == Split::automatic, "fast/turbo default: split from the crossover on");
    require(with(profile::Name::fast, "0", nullptr) == Split::off, "off switch restores the one-pass flash");
    require(with(profile::Name::standard, nullptr, "1") == Split::every_visit, "legacy switch opts in");
    require(with(profile::Name::fast, "auto", nullptr) == Split::automatic, "generic switch selects auto");
    require(with(profile::Name::fast, "0", "1") == Split::off, "generic switch wins over the legacy one");
    require(with(profile::Name::fast, nullptr, "0") == Split::off, "legacy off switch");
    bool refused = false;
    try {
        static_cast<void>(with(profile::Name::fast, "yes", nullptr));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "malformed split switch must be refused");
}

int main() {
    try {
        test_row();
        test_w4a4();
        test_decode_split();
        test_shipped();
        test_runnable();
    } catch (const std::exception& error) {
        std::cerr << "inkling2-mini-dense-lever: " << error.what() << '\n';
        return 1;
    }
    std::cout << "inkling2-mini-dense-lever: ok\n";
    return 0;
}
