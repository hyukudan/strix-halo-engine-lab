#include "glm_profile_overrides.hpp"
#include "strix/engine_profile.hpp"
#include "strix/deepseek41_pair_profile_table.hpp"

#include <iostream>
#include <map>

// A new host hash namespace must not collide with existing local digest helpers.
namespace strix {
namespace {
constexpr std::uint64_t fnv(std::string_view) { return 7; }
}
static_assert(fnv("legacy digest") == 7);
}

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}

int main() {
    using namespace strix::profile;
    static_assert(strix::host_fnv::text("") == 0xcbf29ce484222325ULL);
    static_assert(strix::host_fnv::text("hello") == 0xa430d84680aabd0bULL);
    const DefaultItem row{"item", "SWITCH", "1", "0", "receipt"};
    std::map<std::string, std::string> vars;
    const Lookup env = [&vars](const char* key) -> const char* {
        const auto it = vars.find(key);
        return it == vars.end() ? nullptr : it->second.c_str();
    };
    require(resolve_default(row, false, env).value == "1", "default");
    require(resolve_default(row, true, env).value == "0", "baseline");
    vars["SWITCH"] = "";
    require(resolve_default(row, true, env).value.empty(), "explicit empty");
    require(resolve_default(row, true, env, "cli").value == "cli", "CLI precedence");
    auto exempt = row;
    exempt.in_baseline = false;
    vars.clear();
    require(resolve_default(exempt, true, env).value == "1", "baseline exemption");
    for (unsigned bits = 0; bits < 16; ++bits) {
        const bool ready = (bits & 1U) != 0, runnable = (bits & 2U) != 0;
        const bool allow = (bits & 4U) != 0, provisional = (bits & 8U) != 0;
        const auto expected = !ready && !((provisional || allow) && runnable) ? Status::pending :
            provisional ? Status::provisional : ready ? Status::set : Status::unqualified;
        require(admit_column(ready, runnable, allow, provisional) == expected, "column admission matrix");
    }

    const Lever pending{"item", "SWITCH", "--item", {"0", "1", ""}, "0",
        Kind::bit_changing, {}, {false, true, false}, "candidate", "0", "pending"};
    const std::array table{pending};
    const Engine engine{"test", "ALLOW", table};
    Inputs inputs;
    inputs.env = env;
    inputs.profile = Name::fast;
    const auto refused = resolve(engine, inputs);
    require(refused.entries[0].status == Status::pending, "FAST pending refusal");
    require(profile_args(refused).empty(), "pending args");
    inputs.allow_pending = true;
    require(resolve(engine, inputs).entries[0].status == Status::unqualified, "FAST admission stamp");
    vars["SWITCH"] = "0";
    require(resolve(engine, inputs).entries[0].status == Status::off, "OFF wins");
    inputs.cli.emplace_back("item", "2");
    require(resolve(engine, inputs).entries[0].value == "2", "CLI beats env");
    vars.clear();
    inputs.cli.clear();
    inputs.profile = Name::standard;
    require(resolve(engine, inputs).entries[0].value == "0", "DEFAULT unchanged");

    // Explicit settings of a bit-changing lever: qualified only at the builtin, the off value or a ready, not provisional
    // column value (table data); a provisional or pending column value set by hand stamps the run unqualified.
    {
        const Lever ready_fast{.name = "rows", .env = "", .option = "--rows", .value = {"", "64", "", ""}, .builtin = "0",
            .kind = Kind::bit_changing, .ready = {false, true, false, false}, .runnable = {true, true, false, false},
            .target = "", .off = "0", .source = "gate receipt"};
        const Lever provisional_fast{.name = "window", .env = "", .option = "--window", .value = {"", "2048", "", ""},
            .builtin = "0", .kind = Kind::bit_changing, .ready = {}, .runnable = {true, true, false, false}, .target = "",
            .off = "0", .source = "gate failed", .provisional = {false, true, false, false}};
        const Lever exact{.name = "order", .env = "", .option = "--order", .value = {"", "1", "", ""}, .builtin = "0",
            .kind = Kind::exact, .ready = {true, true, false, false}, .runnable = {true, true, false, false}, .target = "",
            .off = "0", .source = "bitwise"};
        const std::array rows{ready_fast, provisional_fast, exact};
        const Engine e{"explicit", "ALLOW", rows};
        const Lookup none = [](const char*) -> const char* { return nullptr; };
        const auto stamp = [&](Pairs cli) {
            Inputs in;
            in.env = none;
            in.cli = std::move(cli);
            return resolve(e, in).qualification();
        };
        require(stamp({}) == "qualified", "builtin run qualified");
        require(stamp({{"rows", "64"}}) == "qualified", "explicit ready column value qualified");
        require(stamp({{"rows", "0"}, {"window", "0"}}) == "qualified", "explicit builtin / off qualified");
        require(stamp({{"order", "7"}}) == "qualified", "explicit exact lever qualified");
        require(stamp({{"rows", "32"}}) == "unqualified: rows", "explicit unlisted value unqualified");
        require(stamp({{"window", "2048"}}) == "unqualified: window", "explicit provisional column value unqualified");
        require(stamp({{"rows", "32"}, {"window", "1024"}}) == "unqualified: rows,window", "explicit rows listed in order");
    }

    // same_bits: values measured to compute the bits of each other (a scheduling-only choice) share the qualification
    // of a ready, not provisional column whose value is in the class; outside the class, or with no ready member, the
    // explicit value stays unqualified.
    {
        const Lever width{.name = "rows", .env = "", .option = "--rows", .value = {"", "4096", "", ""}, .builtin = "0",
            .kind = Kind::bit_changing, .ready = {false, true, false, false}, .runnable = {true, true, false, false},
            .target = "", .off = "0", .source = "gate receipt; 1024 / 2048 / 4096 bitwise", .same_bits = "1024,2048,4096"};
        const Lever unready{.name = "window", .env = "", .option = "--window", .value = {"", "4096", "", ""}, .builtin = "0",
            .kind = Kind::bit_changing, .ready = {}, .runnable = {true, true, false, false}, .target = "", .off = "0",
            .source = "pending", .same_bits = "2048,4096"};
        const std::array rows{width, unready};
        const Engine e{"same_bits", "ALLOW", rows};
        const Lookup none = [](const char*) -> const char* { return nullptr; };
        const auto stamp = [&](Pairs cli) {
            Inputs in;
            in.env = none;
            in.cli = std::move(cli);
            return resolve(e, in).qualification();
        };
        require(stamp({{"rows", "4096"}}) == "qualified", "same_bits: the ready column value");
        require(stamp({{"rows", "1024"}}) == "qualified" && stamp({{"rows", "2048"}}) == "qualified", "same_bits: class members");
        require(stamp({{"rows", "512"}}) == "unqualified: rows", "same_bits: a value outside the class");
        require(stamp({{"rows", "10240"}}) == "unqualified: rows" && stamp({{"rows", "02048"}}) == "unqualified: rows",
                "same_bits: whole list items only");
        require(stamp({{"window", "2048"}}) == "unqualified: window", "same_bits: no ready member, no qualification");
    }

    // Historical fingerprint must keep its nonstandard seed and FF separators.
    const auto legacy = resolve(engine, inputs);
    std::uint64_t hash = 1469598103934665603ULL;
    for (const std::string_view field : {"default", "qualified", "item", "0"}) {
        for (const char c : field) { hash ^= static_cast<unsigned char>(c); hash *= 1099511628211ULL; }
        hash ^= 255U; hash *= 1099511628211ULL;
    }
    require(legacy.fingerprint() == hash, "legacy fingerprint bytes");
    const Engine shipped{"deepseek41-pair", "ALLOW", strix::deepseek41::pair_defaults::profile_table};
    inputs.profile = Name::fast;
    const auto shipped_fast = resolve(shipped, inputs);
    require(shipped_fast.find("moe_relaxed")->status == Status::provisional, "shipped table positional compatibility");
    check_peer(legacy, [](auto out, auto in) { std::memcpy(in.data(), out.data(), out.size()); });
    bool mismatch = false;
    try {
        check_peer(legacy, [](auto out, auto in) {
            std::memcpy(in.data(), out.data(), out.size());
            in[8] ^= std::byte{1};
        });
    } catch (const std::runtime_error&) { mismatch = true; }
    require(mismatch, "rank mismatch rejected");

    ExecutionReceipt receipt;
    receipt.engine = "test\n\"";
    receipt.identity.arithmetic_contract = "contract";
    const auto numerical = receipt.identity.id();
    receipt.build = "different build";
    receipt.kernel = "different kernel";
    require(receipt.identity.id() == numerical, "execution independent of numerical identity");
    const auto original = receipt.identity;
    for (auto member : {&NumericalIdentity::arithmetic_contract, &NumericalIdentity::model_sha256,
                        &NumericalIdentity::pack_hash, &NumericalIdentity::table_hash,
                        &NumericalIdentity::state_format, &NumericalIdentity::prefill_lineage,
                        &NumericalIdentity::reduction_plan}) {
        receipt.identity = original;
        receipt.identity.*member += "changed";
        require(receipt.identity.id() != numerical, "every numerical field participates");
    }
    NumericalIdentity a, b;
    a.arithmetic_contract = "a"; a.model_sha256 = "bc";
    b.arithmetic_contract = "ab"; b.model_sha256 = "c";
    require(a.id() != b.id(), "field boundaries");
    receipt.identity = original;
    receipt.digests.emplace_back("state-sha256", "unknown");
    const auto witness = receipt.witness();
    require(witness.starts_with("strix-engine-witness {\"schema\":\"strix-engine-receipt/v1\""), "witness schema");
    require(witness.find('\n') == witness.size() - 1, "one line, escaped controls");
    receipt.digests.emplace_back("state-sha256", "duplicate");
    bool duplicate = false;
    try { (void)receipt.json(); } catch (const std::invalid_argument&) { duplicate = true; }
    require(duplicate, "duplicate digest rejected");

    // Shared receipt summary and the loud empty-tier line (one form for every engine).
    {
        const Lookup none = [](const char*) -> const char* { return nullptr; };
        static constexpr std::array<Lever, 0> empty_rows{};
        static constexpr Engine empty{.tag = "empty-engine", .allow_env = "", .table = empty_rows,
                                      .quality_rows = false, .turbo_version = "empty-turbo-v1"};
        Inputs in;
        in.env = none;
        require(resolve(empty, in).summary() == "profile exact levers none", "exact summary");
        require(resolve(empty, in).tier_notice().empty(), "exact has no tier notice");
        in.profile = Name::fast;
        const auto fast = resolve(empty, in);
        require(fast.summary() == "profile fast levers none", "fast summary");
        require(fast.tier_notice() ==
            "empty-engine profile fast: FAST requested, 0 levers qualified on this device; runs the exact column",
            "loud empty FAST");
        require(fast.receipt_lines().size() == 2U && fast.notices().empty(), "notice kept out of notices()");

        static constexpr std::array<Lever, 2> rows{
            Lever{.name = "applied", .env = "STRIX_TEST_APPLIED", .option = "", .value = {"0", "1", "", ""},
                  .builtin = "0", .kind = Kind::bit_changing, .ready = {true, true, true, true}, .target = "",
                  .off = "", .source = "test receipt"},
            Lever{.name = "waiting", .env = "STRIX_TEST_WAITING", .option = "", .value = {"0", "1", "", ""},
                  .builtin = "0", .kind = Kind::bit_changing, .ready = {true, false, true, false}, .target = "x",
                  .off = "", .source = "gate pending"},
        };
        static constexpr Engine two{.tag = "two-engine", .allow_env = "STRIX_TEST_ALLOW", .table = rows,
                                    .quality_rows = false, .turbo_version = "two-turbo-v1"};
        const auto applied = resolve(two, in);
        require(applied.summary() == "profile fast levers applied=1", "applied lever listed, pending one not");
        require(applied.tier_notice().empty() && applied.receipt_lines().size() == 1U, "no loud line once applied");
        static constexpr std::array<Lever, 1> pending_rows{rows[1]};
        static constexpr Engine waiting{.tag = "pending-engine", .allow_env = "STRIX_TEST_ALLOW", .table = pending_rows,
                                        .quality_rows = false, .turbo_version = "p-turbo-v1"};
        require(resolve(waiting, in).tier_notice() ==
            "pending-engine profile fast: FAST requested, 0 levers qualified on this device; runs the exact column (1 pending)",
            "pending count");
        // An explicit setting of a bit-changing lever is an override, not the exact column: the line says so.
        {
            Inputs explicit_in = in;
            explicit_in.env = [](const char* key) -> const char* {
                return std::string_view(key) == "STRIX_TEST_WAITING" ? "1" : nullptr;
            };
            const auto explicit_run = resolve(waiting, explicit_in);
            require(explicit_run.entries.front().status == Status::explicit_env, "explicit env recorded");
            require(explicit_run.tier_notice() ==
                "pending-engine profile fast: FAST requested, 0 levers qualified on this device; "
                "runs with unqualified or explicit overrides (waiting=1), not the exact column",
                "explicit override is not reported as the exact column");
            require(explicit_run.tier_notice().find("runs the exact column") == std::string::npos, "no exact claim");
        }
        // Engine-attested overrides outside the table (GLM standalone MLA switch) also drop the exact claim.
        {
            auto plain = resolve(waiting, in);
            require(plain.tier_notice().find("runs the exact column") != std::string::npos, "plain FAST says exact column");
            const auto glm_env = [](const char* k) -> const char* {
                return std::string_view(k) == "STRIX_GLM_PF_MLA_WMMA" ? "1" : nullptr;
            };
            require(strix::glm::explicit_prefill_overrides(glm_env) ==
                    std::vector<std::string>{"STRIX_GLM_PF_MLA_WMMA=1"}, "standalone MLA override found");
            require(strix::glm::explicit_prefill_overrides([](const char*) -> const char* { return nullptr; }).empty(),
                    "no override when unset");
            plain.engine_overrides = strix::glm::explicit_prefill_overrides(glm_env);
            require(plain.tier_notice().find("STRIX_GLM_PF_MLA_WMMA=1), not the exact column") != std::string::npos,
                    "standalone MLA override named, no exact claim");
            require(strix::glm::explicit_prefill_overrides([](const char* k) -> const char* {
                        return std::string_view(k) == "STRIX_GLM_PF_MLA_WMMA" ? "0" : nullptr;
                    }).empty(),
                    "an explicit opt-out is not an override");
            require(plain.tier_notice().find("runs the exact column") == std::string::npos, "no exact claim with override");
        }
        // An unqualified row run under --allow-pending is not a qualified lever: the zero-qualified line stays.
        {
            static constexpr std::array<Lever, 1> runnable_rows{
                Lever{.name = "runnable", .env = "STRIX_TEST_RUNNABLE", .option = "", .value = {"0", "1", "", ""},
                      .builtin = "0", .kind = Kind::bit_changing, .ready = {true, false, true, false},
                      .runnable = {true, true, true, true}, .target = "x", .off = "", .source = "gate pending"},
            };
            static constexpr Engine unqualified{.tag = "unq-engine", .allow_env = "STRIX_TEST_ALLOW",
                                                .table = runnable_rows, .quality_rows = false, .turbo_version = "u-turbo-v1"};
            Inputs pending_in = in;
            pending_in.allow_pending = true;
            const auto unq = resolve(unqualified, pending_in);
            require(unq.entries.front().status == Status::unqualified, "unqualified row applied under allow-pending");
            require(unq.profile_applied(), "the column was applied");
            require(unq.tier_notice() ==
                "unq-engine profile fast: FAST requested, 0 levers qualified on this device; "
                "runs with unqualified or explicit overrides (runnable=1), not the exact column",
                "unqualified row does not suppress the zero-qualified line");
        }
        in.profile = Name::turbo;
        require(resolve(empty, in).tier_notice().find(": TURBO requested, 0 levers") != std::string::npos, "loud TURBO");
        // A qualified TURBO on-the-fly weight pack changes arithmetic even if no speed lever was selected.
        // The receipt must not claim that it runs the native-weight EXACT column.
        auto weighted_engine = empty;
        weighted_engine.weight_caps.source[1] = true; // onfly
        weighted_engine.weight_caps.bits[4] = true;
        in.weights.source = "onfly";
        in.weights.bits = "4";
        const auto weighted = resolve(weighted_engine, in);
        require(weighted.weights.effective == turbo::Source::onfly, "onfly weights admitted");
        require(weighted.tier_notice().find("uses onfly-int4-rtn-g32 weights (not the exact column)") != std::string::npos,
                "no false EXACT equivalence when TURBO changes weights");
    }
    std::cout << "common profile/receipt tests passed\n";
}
