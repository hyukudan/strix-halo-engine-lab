// Identity variants of the shared profile resolver (include/strix/engine_profile.hpp): a bit-changing route runs only
// when requested by name through the resolver and admitted by the tier and backend rule; the incumbent identity keeps
// every receipt, JSON and fingerprint byte for byte; CLI and server/TOML requests resolve to the same run.
#include "strix/engine_profile.hpp"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace strix::profile;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <class Fn>
std::string refusal(Fn&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return {};
}

constexpr std::array<Lever, 1> kLevers{{
    {"t-lever@1", "STRIX_T_IDV_LEVER", "", {"", "1", "", ""}, "0", Kind::bit_changing,
     {false, true, false, true}, {false, true, false, true}, "", "0", "receipt t-lever"},
}};

// packed: bitwise-defined against its reference, admitted by every tier on HIP only.
// screen: lossy-free but batch-dependent on CUDA, so FAST / TURBO only.
constexpr std::array<IdentityVariant, 2> kVariants{{
    {"t-packed", 1, "MoE experts in one packed pass", "lone decode, bitwise", "hip", {true, true, true, true}, true,
     "--t-packed-decode", "t_packed_decode", "receipt packed"},
    {"t-screen", 2, "screened head picks", "full head scores", "cuda,hip", {false, true, false, true}, false, "", "",
     "receipt screen"},
}};
static_assert(valid_variants(kVariants));

constexpr IdentityVariant kExactWithoutIdentity{"x", 1, "c", "r", "hip", {true, false, false, false}, false};
static_assert(!valid_variants(std::span(&kExactWithoutIdentity, 1)), "EXACT admits only an exact identity variant");
constexpr IdentityVariant kNoTier{"x", 1, "c", "r", "hip", {}, true};
static_assert(!valid_variants(std::span(&kNoTier, 1)), "a variant no tier admits is dead");
constexpr IdentityVariant kNoReference{"x", 1, "c", "", "hip", {false, true, false, false}, false};
static_assert(!valid_variants(std::span(&kNoReference, 1)), "every variant names its reference");
constexpr IdentityVariant kNoVersion{"x", 0, "c", "r", "hip", {false, true, false, false}, false};
static_assert(!valid_variants(std::span(&kNoVersion, 1)), "every variant is versioned");
constexpr std::array<IdentityVariant, 2> kDuplicate{{
    {"x", 1, "c", "r", "hip", {false, true, false, false}, false},
    {"x", 2, "c", "r", "cuda", {false, true, false, false}, false},
}};
static_assert(!valid_variants(kDuplicate), "ids are unique");

constexpr Engine kEngine{"t-engine", "", kLevers, false, "t-turbo-v1", {}, kVariants};
constexpr Engine kLegacy{"t-engine", "", kLevers, false, "t-turbo-v1"};  // the same engine before variants existed
constexpr Engine kBare{"t-bare", "", {}, false, "t-bare-turbo-v1", {}, kVariants};

Inputs inputs(Name tier, std::string backend) {
    Inputs in;
    in.profile = tier;
    in.env = [](const char*) -> const char* { return nullptr; };  // never the process environment
    in.backend = std::move(backend);
    return in;
}

void incumbent_unchanged() {
    for (const Name tier : {Name::standard, Name::fast, Name::quality, Name::turbo}) {
        const auto now = resolve(kEngine, inputs(tier, "hip"));
        const auto before = resolve(kLegacy, inputs(tier, "hip"));
        require(now.incumbent_identity() && now.identity() == "identity incumbent", "incumbent identity");
        require(!now.selected("t-packed") && !now.selected("t-screen"), "nothing selected by default");
        require(now.receipt_lines() == before.receipt_lines(), "incumbent receipt lines unchanged");
        require(now.describe() == before.describe(), "incumbent effective config unchanged");
        require(now.json() == before.json(), "incumbent JSON unchanged");
        require(now.fingerprint() == before.fingerprint(), "incumbent fingerprint unchanged");
        require(now.cache_namespace() == before.cache_namespace(), "incumbent cache namespace unchanged");
    }
}

// A defect of the caller (std::logic_error that is not an admission refusal): its message, or "" when none was thrown.
template <class Fn>
std::string logic_refusal(Fn&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return {};
    } catch (const std::logic_error& e) {
        return e.what();
    }
    return {};
}

// Compile-time view of what a binding can do with a resolution (access checking is part of satisfaction).
template <class R>
concept can_add_variant = requires(R r, const IdentityVariant* v) { r.identity_variants().push_back(v); };
template <class R>
concept sees_admitted_set = requires(R r) { r.admitted_; };

void only_declared_requests_select() {
    // A route Boolean has no entry point: the engine asks the resolution, which knows only declared rows.
    const auto run = resolve(kEngine, inputs(Name::standard, "hip"));
    bool threw = false;
    try {
        static_cast<void>(run.selected("t-undeclared"));
    } catch (const std::logic_error&) {
        threw = true;
    }
    require(threw, "selected() refuses an undeclared id");
    auto in = inputs(Name::fast, "hip");
    require(refusal([&] { request_variant(kEngine, in, "t-undeclared", VariantOrigin::cli); })
                .find("unknown identity variant 't-undeclared' (declared: t-packed@1, t-screen@2)") != std::string::npos,
            "request of an undeclared id refused with the declared list");
    in.variants.push_back({"t-undeclared", 0, VariantOrigin::serve});  // hand-built Inputs
    require(refusal([&] { static_cast<void>(resolve(kEngine, in)); }).find("unknown identity variant") !=
                std::string::npos, "resolve() refuses a hand-built undeclared request");
    // The admitted set is read-only to a binding: only resolve() writes it (compile-time property).
    static_assert(!can_add_variant<Resolution>, "a binding cannot add a variant to the resolution");
    static_assert(!sees_admitted_set<Resolution>, "the admitted set is private");
    // A resolution changed after admission (tier or engine) never selects: the set belongs to the tier and the
    // engine resolve() checked. t-screen is FAST-only; moving the resolution to EXACT afterwards must not run it.
    auto admitted = inputs(Name::fast, "hip");
    request_variant(kEngine, admitted, "t-screen", VariantOrigin::serve);
    auto moved = resolve(kEngine, admitted);
    require(moved.selected("t-screen"), "t-screen admitted under fast");
    moved.profile = Name::standard;
    require(logic_refusal([&] { static_cast<void>(moved.selected("t-screen")); })
                .find("changed after admission (admitted for t-engine --profile fast)") != std::string::npos,
            "a tier changed after admission refuses the selection");
    auto rebound = resolve(kEngine, admitted);
    rebound.engine = &kBare;
    require(!logic_refusal([&] { static_cast<void>(rebound.selected("t-screen")); }).empty(),
            "an engine changed after admission refuses the selection");
    const auto copied = resolve(kEngine, admitted);
    require(copied.selected("t-screen") && copied.identity() == "identity t-screen@2", "a copy keeps the admission");
    // A default-built resolution has no admission at all.
    Resolution blank;
    blank.engine = &kEngine;
    blank.profile = Name::fast;
    require(!blank.selected("t-screen") && blank.incumbent_identity(), "only resolve() admits a variant");
    // An option alias is CLI only; a TOML key is TOML only.
    auto alias = inputs(Name::fast, "hip");
    require(!refusal([&] { request_variant(kEngine, alias, "--t-packed-decode", VariantOrigin::toml); }).empty(),
            "a CLI alias is not a TOML key");
    require(!refusal([&] { request_variant(kEngine, alias, "t_packed_decode", VariantOrigin::cli); }).empty(),
            "a TOML key is not a CLI alias");
}

void tier_and_backend_admission() {
    auto screen = inputs(Name::standard, "cuda");
    request_variant(kEngine, screen, "t-screen", VariantOrigin::cli);
    require(refusal([&] { static_cast<void>(resolve(kEngine, screen)); }) ==
                "t-engine: identity variant t-screen@2 is not admitted under --profile exact (admitted: fast,turbo)",
            "a non-exact variant is refused under EXACT");
    screen.profile = Name::quality;
    require(!refusal([&] { static_cast<void>(resolve(kEngine, screen)); }).empty(), "refused under quality");
    screen.profile = Name::fast;
    require(resolve(kEngine, screen).selected("t-screen"), "admitted under FAST");
    screen.profile = Name::turbo;
    require(resolve(kEngine, screen).selected("t-screen"), "admitted under TURBO");

    auto packed = inputs(Name::standard, "hip");
    request_variant(kEngine, packed, "t-packed", VariantOrigin::cli);
    const auto exact = resolve(kEngine, packed);
    require(exact.selected("t-packed") && !exact.selected("t-screen"), "exact identity variant admitted under EXACT");
    packed.backend = "cuda";
    require(refusal([&] { static_cast<void>(resolve(kEngine, packed)); }) ==
                "t-engine: identity variant t-packed@1 is not declared for backend cuda (declared: hip)",
            "an undeclared backend is refused");
    packed.backend.clear();
    require(refusal([&] { static_cast<void>(resolve(kEngine, packed)); }) ==
                "t-engine: identity variant t-packed@1 requires a resolved backend",
            "a variant request needs the backend");

    auto pinned = inputs(Name::fast, "hip");
    request_variant(kEngine, pinned, "t-screen@2", VariantOrigin::cli);
    require(resolve(kEngine, pinned).selected("t-screen"), "pinned current version admitted");
    auto stale = inputs(Name::fast, "hip");
    request_variant(kEngine, stale, "t-screen@1", VariantOrigin::cli);
    require(refusal([&] { static_cast<void>(resolve(kEngine, stale)); }) ==
                "t-engine: identity variant t-screen@1 is not built (this build declares @2)",
            "a stale pinned version is refused");
    for (const char* bad : {"t-screen@", "t-screen@x", "t-screen@0"})
        require(!refusal([&] { request_variant(kEngine, stale, bad, VariantOrigin::cli); }).empty(), "bad version");
}

void printed_identity() {
    auto in = inputs(Name::standard, "hip");
    request_variant(kEngine, in, "t-packed", VariantOrigin::cli);
    const auto run = resolve(kEngine, in);
    const auto incumbent = resolve(kEngine, inputs(Name::standard, "hip"));
    require(run.identity() == "identity t-packed@1", "startup identity field");
    require(run.receipt_lines().back() == "identity t-packed@1", "receipt prints the identity");
    require(run.receipt_lines().front() == incumbent.receipt_lines().front(), "summary line unchanged");
    require(run.describe().find("identity t-packed@1 (exact identity; reference lone decode, bitwise; changes MoE "
                                "experts in one packed pass)\n") != std::string::npos,
            "effective config prints the identity and its reference");
    require(run.json().find(",\"identity\":[{\"id\":\"t-packed\",\"version\":1,\"exact_identity\":true,") !=
                std::string::npos, "JSON carries the identity");
    require(run.fingerprint() != incumbent.fingerprint(), "ranks with different identities never agree");
    require(run.qualification() == "qualified", "a declared variant admitted by its tier is qualified");

    // Declaration order, whatever the request order; one identity per variant however often it is requested.
    auto both = inputs(Name::fast, "hip");
    request_variant(kEngine, both, "t-screen", VariantOrigin::serve);
    request_variant(kEngine, both, "t-packed", VariantOrigin::cli);
    request_variant(kEngine, both, "--t-packed-decode", VariantOrigin::cli);
    require(resolve(kEngine, both).identity() == "identity t-packed@1 t-screen@2", "declaration order, deduplicated");

    // FAST with no qualified lever: the notice never claims the exact column when a variant runs.
    auto bare = inputs(Name::fast, "cuda");
    request_variant(kBare, bare, "t-screen", VariantOrigin::cli);
    const auto notice = resolve(kBare, bare).tier_notice();
    require(notice.find("(identity t-screen@2), not the exact column") != std::string::npos,
            "tier notice names the admitted variant");
    require(resolve(kBare, inputs(Name::fast, "cuda")).tier_notice().find("runs the exact column") !=
                std::string::npos, "incumbent tier notice unchanged");
}

void cli_server_parity() {
    std::vector<std::vector<std::string>> cli{
        {"--model", "m", "--identity-variant", "t-packed", "--x"},
        {"--model", "m", "--identity-variant=t-packed@1", "--x"},
        {"--model", "m", "--t-packed-decode", "--x"},
    };
    auto server = inputs(Name::standard, "hip");
    request_variant(kEngine, server, "t-packed", VariantOrigin::serve);
    auto toml = inputs(Name::standard, "hip");
    request_variant(kEngine, toml, "t_packed_decode", VariantOrigin::toml);
    const auto served = resolve(kEngine, server);
    require(resolve(kEngine, toml).fingerprint() == served.fingerprint(), "TOML key == server request");
    for (auto& args : cli) {
        auto in = inputs(Name::standard, "hip");
        in.profile.reset();
        take_options(kEngine, args, in);
        require(args == std::vector<std::string>{"--model", "m", "--x"}, "variant tokens consumed, others kept");
        const auto run = resolve(kEngine, in);
        require(run.identity() == served.identity() && run.json() == served.json() &&
                    run.describe() == served.describe() && run.fingerprint() == served.fingerprint(),
                "CLI request == server request");
    }
    std::vector<std::string> missing{"--identity-variant"};
    auto in = inputs(Name::standard, "hip");
    require(!refusal([&] { take_options(kEngine, missing, in); }).empty(), "--identity-variant needs a value");
    std::vector<std::string> unknown{"--identity-variant", "nope"};
    require(!refusal([&] { take_options(kEngine, unknown, in); }).empty(), "unknown variant refused at parse");
    std::vector<std::string> legacy{"--t-packed-decode"};  // an engine without the declaration leaves the token alone
    take_options(kLegacy, legacy, in);
    require(legacy.size() == 1U && in.variants.empty(), "undeclared option is not a variant");
}

}  // namespace

int main() {
    try {
        incumbent_unchanged();
        only_declared_requests_select();
        tier_and_backend_admission();
        printed_identity();
        cli_server_parity();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    std::puts("profile identity variant tests: OK");
    return 0;
}
