#include "../apps/qwen_effective_config.hpp"
#include "strix/session_state_cache.hpp"
#include "../apps/prefix_state_cache_identity.hpp"
#include <iostream>
#include <map>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace pr = strix::qwen::profile;
namespace dc = strix::decode_cli;
namespace {
unsigned checks = 0;
void check(bool ok, const char* message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
template<class F> void refuses(F operation, std::string_view needle) {
    try { operation(); }
    catch (const std::invalid_argument& error) {
        check(std::string_view(error.what()).find(needle) != std::string_view::npos, "refusal explains the cause");
        return;
    }
    check(false, "expected refusal");
}
using Env = std::map<std::string, std::string>;
pr::Lookup lookup(const Env& env) {
    return [&env](const char* key) -> const char* {
        const auto it = env.find(key);
        return it == env.end() ? nullptr : it->second.c_str();
    };
}
constexpr pr::Lever synthetic[] = {
    {.name="ready", .env="STRIX_TEST_PROFILE_READY", .option="--ready", .value={"", "1", ""}, .builtin="0",
     .kind=pr::Kind::exact, .ready={true,true,true}, .runnable={true,true,true}, .target="1", .off="0", .source="host fixture"},
    {.name="pending", .env="STRIX_TEST_PROFILE_PENDING", .option="--pending", .value={"", "1", ""}, .builtin="0",
     .kind=pr::Kind::bit_changing, .ready={true,false,true}, .runnable={false,true,false}, .target="candidate", .off="0", .source="host fixture"},
    {.name="absent", .env="STRIX_TEST_PROFILE_ABSENT", .option="--absent", .value={"", "1", ""}, .builtin="0",
     .kind=pr::Kind::bit_changing, .ready={true,false,true}, .runnable={false,false,false}, .target="future", .off="0", .source="host fixture"},
};
dc::Options parse(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    return dc::parse_options(static_cast<int>(argv.size()), argv.data(), true);
}
void product() {
    const Env empty;
    pr::Inputs inputs{{}, lookup(empty), false};
    check(pr::parse("default") == pr::Name::standard && pr::parse("fast") == pr::Name::fast &&
        pr::parse("quality") == pr::Name::quality, "three profile names");
    refuses([] { (void)pr::parse("turbo"); }, "default, fast or quality");
    for (const auto profile : {pr::Name::standard, pr::Name::fast, pr::Name::quality}) {
        for (const bool allow : {false, true}) {
            inputs.allow_pending = allow;
            const auto r = pr::resolve(profile, inputs);
            unsigned exports = 0;
            pr::apply(r, [&](const char*, const char*) { ++exports; return true; });
            check(exports == 0, "current product profiles export no switches, including allow-pending");
            check(r.qualification() == "qualified", "no unqualified implementation executes");
            check(r.entries.size() == (profile == pr::Name::fast ? 4U : 0U), "only fast has product rows");
            const auto json = strix::serve::parse_json_unique(r.json());
            check(json.find("profile")->as_string() == pr::name(profile), "JSON profile name");
            check(json.find("unqualified")->size() == 0U, "JSON no unqualified rows");
            check(json.find("pending")->size() == r.entries.size(), "JSON pending rows");
            for (const auto& entry : r.entries) {
                check(entry.status == pr::Status::pending && entry.value == entry.lever->builtin, "pending keeps builtin");
                check(!entry.target.empty(), "pending records declared target");
            }
            check(r.notices().size() == (profile == pr::Name::quality ? 1U : r.entries.size()), "one notice per pending row");
        }
    }
    check(pr::resolve(pr::Name::quality, inputs).notices()[0] == "quality: no quality rows for this model, using default", "quality notice");
    check(pr::resolve(pr::Name::quality, inputs).json().find("\"note\":\"quality: no quality rows for this model, using default\"") != std::string::npos &&
        pr::resolve(pr::Name::standard, inputs).json().find("\"note\"") == std::string::npos, "quality receipt names the default it runs");
    for (const auto& lever : pr::levers()) {
        check(!lever.name.empty() && !lever.option.empty() && !lever.source.empty(), "editable table has provenance and option");
        check(lever.value[0].empty() && lever.value[2].empty(), "default and quality leave every lever alone");
        check(!lever.ready[1] && !lever.runnable[1], "all current fast code is absent");
        if (lever.env.empty()) continue;
        Env env{{std::string(lever.env), std::string(lever.value[1])}};
        inputs = {{}, lookup(env), true};
        refuses([&] { (void)pr::resolve(pr::Name::standard, inputs); }, "not in this build");
        env[std::string(lever.env)] = std::string(lever.off);
        check(pr::resolve(pr::Name::fast, inputs).find(lever.name)->status == pr::Status::off, "explicit env off wins");
    }
}
void resolver() {
    Env env;
    pr::Inputs inputs{{}, lookup(env), false};
    auto r = pr::resolve(pr::Name::fast, inputs, synthetic);
    check(r.find("ready")->status == pr::Status::set, "ready row applies");
    check(r.find("pending")->status == pr::Status::pending && r.find("absent")->status == pr::Status::pending, "pending never refuses");
    inputs.allow_pending = true;
    r = pr::resolve(pr::Name::fast, inputs, synthetic);
    check(r.find("pending")->status == pr::Status::unqualified && r.find("absent")->status == pr::Status::pending, "only runnable pending applies");
    check(r.qualification() == "unqualified: pending", "unqualified stamp names executed rows");
    auto json = strix::serve::parse_json_unique(r.json());
    check(json.find("unqualified")->as_array()[0].as_string() == "pending", "JSON unqualified list");
    check(json.find("pending")->as_array()[0].find("target")->as_string() == "future", "JSON declared target");
    pr::apply(r, [&](const char* key, const char* value) { env[key] = value; return true; });
    const auto restarted = pr::resolve(pr::Name::fast, inputs, synthetic);
    check(restarted.json() == r.json(), "ready and unqualified export provenance survives re-resolution");
    env["STRIX_TEST_PROFILE_PENDING"] = "0";
    check(pr::resolve(pr::Name::fast, inputs, synthetic).find("pending")->status == pr::Status::off, "changed inherited env is explicit off");
    env["STRIX_TEST_PROFILE_READY"] = "2";
    check(pr::resolve(pr::Name::fast, inputs, synthetic).find("ready")->status == pr::Status::explicit_env, "explicit env beats profile");
    inputs.cli = {{"ready","0"}};
    r = pr::resolve(pr::Name::fast, inputs, synthetic);
    check(r.find("ready")->status == pr::Status::off && r.find("ready")->value == "0", "CLI off beats env");
    pr::apply(r, [&](const char* key, const char* value) { env[key] = value; return true; });
    check(env.at("STRIX_TEST_PROFILE_READY") == "0", "CLI off exports over env");
    inputs.cli = {{"ready","3"}};
    r = pr::resolve(pr::Name::standard, inputs, synthetic);
    check(r.find("ready")->status == pr::Status::explicit_cli, "CLI recorded outside profile");
    pr::apply(r, [&](const char* key, const char* value) { env[key] = value; return true; });
    check(pr::resolve(pr::Name::standard, inputs, synthetic).json() == r.json(), "explicit CLI provenance survives re-resolution");
    inputs.cli.clear(); inputs.allow_pending = false; env.clear();
    env["STRIX_QWEN_PROFILE_ALLOW_PENDING"] = "1";
    check(pr::resolve(pr::Name::fast, inputs, synthetic).qualification() == "unqualified: pending", "allow-pending env alias");
    env["STRIX_TEST_PROFILE_ABSENT"] = "1";
    refuses([&] { (void)pr::resolve(pr::Name::fast, inputs, synthetic); }, "not in this build");
    env.clear(); inputs.cli = {{"absent","1"}};
    refuses([&] { (void)pr::resolve(pr::Name::fast, inputs, synthetic); }, "not in this build");
}
void cli_and_receipts() {
    const auto plain = parse({"decoder","model.gguf","1"});
    for (const auto value : {"default","fast","quality"}) {
        const auto o = parse({"decoder","model.gguf","1","--profile",value,"--allow-pending"});
        check(o.product_profile.profile == pr::parse(value), "CLI profile parsed");
        check(!o.route_owned && !o.any_low_level_flag && o.optimization_profile == plain.optimization_profile,
            "profile selector does not claim explicit route");
        check(o.numerical_profile == plain.numerical_profile && o.prefill_numerical_profile == plain.prefill_numerical_profile &&
            o.prefill_disabled == plain.prefill_disabled && o.early_setup == plain.early_setup && o.eager_code_load == plain.eager_code_load,
            "product profiles preserve existing arithmetic and exact bundle");
        check(dc::effective_prefill_argv(o) == dc::effective_prefill_argv(plain), "prefix effective arguments exclude product metadata");
        const auto effective = dc::initialize_prefill_config({o,{}});
        const auto json = strix::serve::parse_json_unique(dc::effective_config_json(effective));
        check(json.find("profile")->dump() == o.product_profile.json(), "effective config profile receipt");
        auto expected = strix::serve::parse_json_unique(dc::effective_config_json(dc::initialize_prefill_config({plain,{}})));
        expected.set("profile", *json.find("profile"));
        check(expected.dump() == json.dump(), "effective config differs only by profile receipt");
    }
    check(parse({"decoder","model.gguf","1","--profile=fast"}).product_profile.profile == pr::Name::fast, "equals syntax");
    refuses([] { (void)parse({"decoder","model.gguf","1","--profile"}); }, "requires a value");
    refuses([] { (void)parse({"decoder","model.gguf","1","--profile","turbo"}); }, "default, fast or quality");
    refuses([] { (void)parse({"decoder","model.gguf","1","--profile","fast","--profile=default"}); }, "duplicate");
    refuses([] { (void)parse({"decoder","model.gguf","1","--qsa-budget","1536"}); }, "not in this build");
    refuses([] { (void)parse({"decoder","model.gguf","1","--gdn-fp16-chunk","qwen-gdn-fp16-chunk"}); }, "not in this build");
    check(parse({"decoder","model.gguf","1","--profile","fast","--moe-route-bf16=0"}).product_profile.find("moe_route_bf16")->status == pr::Status::off,
        "explicit CLI off recorded");
    check(pr::policy_option("--profile") && pr::policy_option("--allow-pending") && !pr::policy_option("--prefill-disable"), "only policy options excluded from identity");
}
void identity() {
    const auto before = strix::session_numerical_config("fixture");
    const auto prefix = dc::prefix_effective_execution_signature("fixture", false, false);
    check(pr::export_environment("STRIX_QWEN_PROFILE_ALLOW_PENDING", "1"), "set policy metadata");
    check(pr::export_environment("STRIX_QWEN_PROFILE_OWNED_ready", "fast:profile:1"), "set provenance metadata");
    check(strix::session_numerical_config("fixture") == before, "profile metadata has zero session identity bytes");
    check(dc::prefix_effective_execution_signature("fixture", false, false) == prefix, "profile metadata has zero prefix identity bytes");
    check(pr::export_environment("STRIX_TEST_PROFILE_READY", "1"), "set arithmetic fixture");
    check(strix::session_numerical_config("fixture") != before, "applied arithmetic switches remain session identity-bound");
    check(dc::prefix_effective_execution_signature("fixture", false, false) != prefix, "applied arithmetic switches remain prefix identity-bound");
}
#ifndef _WIN32
int restart_child() {
    pr::Inputs inputs; inputs.allow_pending = true;
    const auto r = pr::resolve(pr::Name::fast, inputs, synthetic);
    pr::apply(r);
    const auto get = [](const char* key) { return std::getenv(key); };
    const bool exec = dc::apply_eager_code_loading(true, get, pr::export_environment);
    if (exec) {
        ::execl("/proc/self/exe", "qwen-product-profile-tests", "--restart-child", nullptr);
        return 2;
    }
    check(r.find("ready")->status == pr::Status::set && r.find("pending")->status == pr::Status::unqualified,
        "actual exec retains statuses and unqualified stamp");
    return r.qualification() == "unqualified: pending" ? 0 : 3;
}
void eager_restart() {
    const auto pid = ::fork();
    check(pid >= 0, "fork restart fixture");
    if (pid == 0) {
        ::unsetenv("HIP_ENABLE_DEFERRED_LOADING");
        const int result = restart_child();
        ::_exit(result);
    }
    int status = 0;
    check(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0, "real eager restart is idempotent");
}
#endif
}
int main(int argc, char** argv) try {
#ifndef _WIN32
    if (argc == 2 && std::string_view(argv[1]) == "--restart-child") return restart_child();
    eager_restart();
#else
    (void)argc; (void)argv;
#endif
    product(); resolver(); cli_and_receipts(); identity();
    std::cout << checks << " profile checks passed\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
