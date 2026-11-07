// Host-only migration checks. GLM keeps its legacy --profile timing flag;
// product-profile admission below exercises the common resolver independently.
#include "glm_rank_receipt.hpp"

#include <cstdio>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

namespace rd = strix::glm::rank_defaults;
namespace profile = strix::profile;

namespace {
int checks = 0, failures = 0;
#define CHECK(c) do { ++checks; if (!(c)) { ++failures; \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

using Env = std::map<std::string, std::string>;
profile::Lookup lookup(const Env& env) {
    return [&env](const char* key) -> const char* {
        const auto it = env.find(key);
        return it == env.end() ? nullptr : it->second.c_str();
    };
}

void test_common_default_rows() {
    static_assert(std::is_same_v<rd::EnvItem, profile::DefaultItem>);
    const Env absent;
    const auto empty = lookup(absent);
    CHECK(rd::env_items().size() == 36);
    for (const auto& row : rd::env_items()) {
        using Source = profile::DefaultSource;
        auto value = profile::resolve_default(row, false, empty);
        CHECK(value.value == row.on && value.source == Source::on);
        value = profile::resolve_default(row, true, empty);
        CHECK(value.value == (row.in_baseline ? row.off : row.on));
        CHECK(value.source == (row.in_baseline ? Source::baseline : Source::on));
        for (bool baseline : {false, true}) {
            for (const std::string& setting : {std::string(row.off), std::string(row.on), std::string{}}) {
                const Env env{{row.env, setting}};
                value = profile::resolve_default(row, baseline, lookup(env));
                CHECK(value.value == setting && value.source == Source::env);
                value = profile::resolve_default(row, baseline, lookup(env), "cli-value");
                CHECK(value.value == "cli-value" && value.source == Source::cli);
                value = profile::resolve_default(row, baseline, lookup(env), "");
                CHECK(value.value.empty() && value.source == Source::cli);
            }
        }
    }
}

void test_glm_exports_and_precedence() {
    const auto request = rd::scan({"--decode", "64"});
    const Env absent;
    const auto plain = rd::resolve(request, lookup(absent));
    for (const auto& row : rd::env_items()) for (bool baseline : {false, true}) {
        auto r = request;
        r.baseline_flag = baseline;
        for (const std::string& setting : {std::string(row.off), std::string{}}) {
            const Env env{{row.env, setting}};
            const auto e = rd::resolve(r, lookup(env));
            CHECK(e.env.at(row.env).value == setting && e.env.at(row.env).source == "explicit");
            Env exported;
            rd::apply_env(e, [&](const char* key, const char* value) { exported[key] = value; });
            CHECK(exported.count(row.env) == 0);
            for (const auto& other : rd::env_items()) {
                if (std::string(row.env) == other.env) continue;
                const bool off = baseline && other.in_baseline;
                CHECK(e.env.at(other.env).value == (off ? other.off : other.on));
                CHECK(e.env.at(other.env).source == (off ? "baseline" : "default"));
                CHECK(exported.count(other.env) == (off || other.app_sets ? 1U : 0U));
                if (exported.count(other.env)) CHECK(exported.at(other.env) == (off ? other.off : other.on));
            }
        }
    }
    const auto baseline = rd::resolve(rd::scan({"--baseline"}), lookup(absent));
    CHECK(baseline.env.at("STRIX_GLM_MOE").value == "fast");
    CHECK(baseline.env.at("STRIX_GLM_MOE").source == "default");
    const Env overrides{{"STRIX_GLM_IDENTITY", "r1"}, {"STRIX_GLM_DECODE_XCH", "comm"},
                        {"STRIX_GLM_PREFILL", "ref"}, {"STRIX_GLM_MTP_SPEC", "0"}};
    rd::ModelFacts facts;
    facts.mtp_complete = true;
    const auto e = rd::resolve(rd::scan({"--decode", "64", "--identity", "r1b", "--decode-xch", "xfer",
                                       "--prefill", "gemv", "--mtp-spec", "1"}), lookup(overrides), facts);
    CHECK(e.identity.value == "r1");              // legacy env > CLI
    CHECK(e.decode_xch.value == "comm");         // legacy env > CLI
    CHECK(e.prefill_requested.value == "gemv"); // CLI > env
    CHECK(e.mtp_spec.value == "1");             // CLI > env
    CHECK(rd::is_value_free("--profile"));
    const auto timing = rd::scan({"--profile", "--decode", "64"});
    CHECK(timing.decode && !timing.baseline_flag);
    CHECK(rd::print(rd::resolve(timing, lookup(absent))) == rd::print(plain));
}

void test_common_profile_admission() {
    // Synthetic levers, never applied to the GLM process environment or rank CLI.
    const auto levers = [] {
        std::vector<profile::Lever> rows;
        for (const char* key : {"qualified", "pending", "unavailable", "provisional", "provisional-unavailable"}) {
            profile::Lever row;
            row.name = key;
            row.env = key;
            row.option = "--lever";
            row.builtin = "0";
            row.off = "0";
            row.value = {"0", "1", ""};
            row.kind = profile::Kind::bit_changing;
            row.target = "candidate";
            rows.push_back(row);
        }
        rows[0].ready[1] = true;
        rows[1].runnable[1] = true;
        rows[3].runnable[1] = true;
        rows[3].provisional[1] = true;
        rows[4].provisional[1] = true;
        return rows;
    }();
    const profile::Engine engine{"glm-host-test", "ALLOW_PENDING", levers};
    const Env absent;
    profile::Inputs input;
    input.env = lookup(absent);
    const auto standard = profile::resolve(engine, input);
    CHECK(standard.profile == profile::Name::standard);
    CHECK(standard.qualification() == "qualified");
    CHECK(standard.notices().empty());
    unsigned exports = 0;
    profile::apply(standard, [&](const char*, const char*) { ++exports; return true; });
    CHECK(exports == 0);
    for (const auto& entry : standard.entries) {
        CHECK(entry.value == "0" && entry.status == profile::Status::builtin);
    }
    input.profile = profile::Name::fast;
    auto fast = profile::resolve(engine, input);
    CHECK(fast.find("qualified")->status == profile::Status::set);
    CHECK(fast.find("pending")->status == profile::Status::pending);
    CHECK(fast.find("pending")->value == "0" && fast.find("pending")->target == "candidate");
    CHECK(fast.find("unavailable")->status == profile::Status::pending);
    CHECK(fast.find("provisional")->status == profile::Status::provisional);
    CHECK(fast.find("provisional-unavailable")->status == profile::Status::pending);
    CHECK(fast.qualification() == "qualification in progress: provisional");
    Env applied;
    profile::apply(fast, [&](const char* key, const char* value) { applied[key] = value; return true; });
    CHECK((applied == Env{{"qualified", "1"}, {"provisional", "1"}}));
    for (bool env_allow : {false, true}) {
        const Env env = env_allow ? Env{{"ALLOW_PENDING", "1"}} : Env{};
        input.env = lookup(env);
        input.allow_pending = !env_allow;
        const auto allowed = profile::resolve(engine, input);
        CHECK(allowed.allow_pending);
        CHECK(allowed.find("pending")->value == "1");
        CHECK(allowed.find("pending")->status == profile::Status::unqualified);
        CHECK(allowed.find("unavailable")->status == profile::Status::pending);
        CHECK(allowed.find("provisional")->status == profile::Status::provisional);
        CHECK(allowed.find("provisional-unavailable")->status == profile::Status::pending);
        CHECK(allowed.qualification() == "qualification in progress: provisional; unqualified: pending");
    }
    const Env explicit_env{{"qualified", "env-value"}, {"pending", ""}, {"unavailable", "0"}};
    input.env = lookup(explicit_env);
    input.allow_pending = false;
    fast = profile::resolve(engine, input);
    CHECK(fast.find("qualified")->value == "env-value");
    CHECK(fast.find("qualified")->status == profile::Status::explicit_env);
    CHECK(fast.find("pending")->value.empty() && fast.find("pending")->status == profile::Status::explicit_env);
    CHECK(fast.find("unavailable")->status == profile::Status::off);
    input.cli = {{"qualified", "cli-value"}};
    fast = profile::resolve(engine, input);
    CHECK(fast.find("qualified")->value == "cli-value");
    CHECK(fast.find("qualified")->status == profile::Status::explicit_cli);
    input.cli = {{"qualified", "0"}};
    CHECK(profile::resolve(engine, input).find("qualified")->status == profile::Status::off);
    input.cli.clear();
    input.env = lookup(absent);
    input.profile = profile::Name::quality;
    const auto quality = profile::resolve(engine, input);
    CHECK(quality.quality_fallback() && quality.notices().size() == 1);
    for (const auto& entry : quality.entries) CHECK(entry.value == "0");
    bool refused = false;
    try { (void)profile::parse("unsupported"); } catch (const std::invalid_argument&) { refused = true; }
    CHECK(refused);
}

void test_partial_receipt() {
    const Env absent;
    const auto e = rd::resolve(rd::scan({"--decode", "64"}), lookup(absent));
    const auto receipt = rd::host_receipt(e);
    CHECK(receipt.engine == "glm" && receipt.implementation == "strix-glm5-rank");
    CHECK(receipt.identity.arithmetic_contract.find("glm_math.hpp") != std::string::npos);
    CHECK(receipt.identity.arithmetic_contract.find("ascending fmaf chain") != std::string::npos);
    CHECK(receipt.identity.reduction_plan == "r1b" && receipt.identity.prefill_lineage == "gemv");
    CHECK(receipt.identity.model_sha256.empty() && receipt.identity.pack_hash.empty());
    CHECK(receipt.identity.table_hash.empty() && receipt.identity.state_format.empty());
    CHECK(receipt.build.empty() && receipt.kernel.empty() && receipt.device.empty() && receipt.topology.empty());
    CHECK(receipt.digests.empty());
    CHECK(receipt.witness() == "strix-engine-witness " + receipt.json() + "\n");
    CHECK(receipt.json().find("\"model_sha256\":\"\"") != std::string::npos);
    CHECK(receipt.json().find("\"device\":\"\"") != std::string::npos);
    const auto r1 = rd::host_receipt(rd::resolve(rd::scan({"--identity", "r1"}), lookup(absent)));
    const auto all = rd::host_receipt(rd::resolve(rd::scan({"--identity", "r1b-all"}), lookup(absent)));
    CHECK(r1.identity.arithmetic_contract == receipt.identity.arithmetic_contract);
    CHECK(r1.identity.reduction_plan == "r1");
    CHECK(all.identity.reduction_plan == "r1b-all" && all.identity.prefill_lineage == "ref");
    CHECK(r1.identity.id() != receipt.identity.id() && all.identity.id() != receipt.identity.id());
    const Env explicit_identity{{"STRIX_GLM_IDENTITY", "r1"}};
    const auto explicit_r1 = rd::host_receipt(rd::resolve(rd::scan({"--identity", "r1b"}), lookup(explicit_identity)));
    CHECK(explicit_r1.identity.id() == r1.identity.id());
    const auto ref = rd::host_receipt(rd::resolve(rd::scan({"--prefill", "ref"}), lookup(absent)));
    CHECK(ref.identity.id() != receipt.identity.id());
    CHECK(rd::print(e).find("strix-engine-witness") == std::string::npos);
}
}  // namespace

int main() {
    test_common_default_rows();
    test_glm_exports_and_precedence();
    test_common_profile_admission();
    test_partial_receipt();
    std::printf("glm defaults common cpu tests: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
