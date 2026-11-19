// CPU tests of the DS RTX --profile resolver (strix/profiles/ds41_rtx.hpp): the table, precedence (CLI > environment >
// profile > builtin, --baseline in order), pending refusal and listing, --allow-pending stamping, the selection.
#include "strix/fnv.hpp"
#include "strix/profiles/ds41_rtx.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

namespace pf = strix::ds41::rtx::profile;

namespace {
int failures = 0;
struct Case {
    std::vector<std::string> args;
    std::map<std::string, std::string> env;
};
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

pf::Inputs in(std::vector<std::string> args, std::map<std::string, std::string> env = {}) {
    auto shared = std::make_shared<std::map<std::string, std::string>>(std::move(env));
    return pf::Inputs{std::move(args), [shared](const char* n) -> const char* {
                          const auto it = shared->find(n);
                          return it == shared->end() ? nullptr : it->second.c_str();
                      }};
}
pf::Resolution run(std::vector<std::string> args, std::map<std::string, std::string> env = {}, pf::App app = pf::App::replica) {
    return pf::resolve(app, in(std::move(args), std::move(env)));
}
bool throws(const std::vector<std::string>& args, const std::map<std::string, std::string>& env = {}) {
    try {
        (void)pf::select(in(args, env));
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}
std::string status(const pf::Resolution& r, const char* lever) {
    const auto* e = r.find(lever);
    return e ? pf::status_name(e->status) : "absent";
}
std::string value(const pf::Resolution& r, const char* lever) {
    const auto* e = r.find(lever);
    return e ? e->value : "absent";
}
}  // namespace

int main() {
    // the table: every replica exact row is ON (= builtin) in all three profiles; fast-only rows pending
    const std::vector<std::string> today = {"lean_state", "q8_r1", "dsrtxk", "fuse", "decode_forms", "evict_auto", "controller",
                                            "draft_keep", "draft_vocab_auto", "wb_defer", "ce_pacing", "wb_stream", "file_tier",
                                            "residency_auto", "warmup_graphs", "engram_cache"};
    for (const auto& l : pf::levers()) {
        CHECK(!l.source.empty());
        if (l.kind == pf::Kind::bit_changing) CHECK(!l.ready[1]);  // nothing bit-changing qualified for the RTX replica
        CHECK(l.value[0] == l.value[2]);                             // quality = default for DS on the RTX
    }
    for (const auto& name : {"default", "quality"}) {
        const auto r = run({"--profile", name});
        CHECK(r.entries.size() == today.size() + 1);
        for (const auto& l : today) CHECK(status(r, l.c_str()) == "profile" && value(r, l.c_str()) == "on");
        // the layer-major prompt path: wanted in every profile, not in this build -> pending, the builtin runs
        CHECK(r.pending() == std::vector<std::string>{"prompt_layer_major"} && r.qualification() == "qualified");
        CHECK(value(r, "prompt_layer_major") == "off" && r.find("prompt_layer_major")->wanted == "on");
    }
    const auto none = run({});
    CHECK(none.name == pf::Name::standard);
    CHECK(none.json() == run({"--profile", "default"}).json());  // no --profile = default, the same receipt
    const auto fast = run({"--profile", "fast"});
    for (const auto& l : today) CHECK(status(fast, l.c_str()) == "profile" && value(fast, l.c_str()) == "on");
    CHECK(fast.pending() == std::vector<std::string>{"prompt_layer_major"});
    CHECK(value(fast, "prompt_layer_major") == "off" && fast.find("prompt_layer_major")->wanted == "on");
    CHECK(fast.qualification() == "qualified");
    CHECK(fast.json().find("\"pending\":[\"prompt_layer_major\"],\"unqualified\":[]") != std::string::npos);
    // --allow-pending: the layer-major path is not in this build, so it stays pending and nothing is stamped
    const auto fast_allow = run({"--profile", "fast", "--allow-pending"});
    CHECK(fast_allow.allow_pending && status(fast_allow, "prompt_layer_major") == "pending" && fast_allow.qualification() == "qualified");
    CHECK(fast_allow.json().find("\"allow_pending\":true") != std::string::npos);

    // the prefill service rows (the service does not take --profile yet; the table is ready for it)
    const auto svc = run({"--profile", "fast"}, {}, pf::App::prefill_service);
    CHECK(svc.entries.size() == 3 && status(svc, "batch_long") == "pending" && status(svc, "fp16_dense_prefill") == "pending");
    CHECK(status(svc, "host_promote") == "profile" && value(svc, "host_promote") == "on");
    // host promotion: default ON (the only default-column service row); --no-host-promote / STRIX_DS41_HOST_PROMOTE=0 /
    // --baseline / STRIX_DS41_RTX_BASELINE=1 off, the last of its options and --baseline decides; not a replica lever
    const auto svc_default = run({}, {}, pf::App::prefill_service);
    CHECK(svc_default.entries.size() == 1 && status(svc_default, "host_promote") == "profile" && value(svc_default, "host_promote") == "on");
    CHECK(svc_default.pending().empty() && svc_default.qualification() == "qualified");
    CHECK(run({}).find("host_promote") == nullptr);
    const auto svc_run = [](std::vector<std::string> args, std::map<std::string, std::string> env = {}) {
        return pf::resolve(pf::Name::standard, false, pf::App::prefill_service, in(std::move(args), std::move(env)));
    };
    CHECK(value(svc_run({"--no-host-promote"}), "host_promote") == "off" && status(svc_run({"--no-host-promote"}), "host_promote") == "explicit-cli");
    CHECK(value(svc_run({"--no-host-promote", "--host-promote"}), "host_promote") == "on");
    CHECK(value(svc_run({}, {{"STRIX_DS41_HOST_PROMOTE", "0"}}), "host_promote") == "off");
    CHECK(status(svc_run({}, {{"STRIX_DS41_HOST_PROMOTE", "0"}}), "host_promote") == "explicit-env");
    CHECK(value(svc_run({}, {{"STRIX_DS41_HOST_PROMOTE", "1"}}), "host_promote") == "on");
    CHECK(value(svc_run({"--host-promote"}, {{"STRIX_DS41_HOST_PROMOTE", "0"}}), "host_promote") == "on");
    CHECK(value(svc_run({"--baseline"}), "host_promote") == "off" && status(svc_run({"--baseline"}), "host_promote") == "baseline-cli");
    CHECK(value(svc_run({"--host-promote", "--baseline"}), "host_promote") == "off");
    CHECK(value(svc_run({"--baseline", "--host-promote"}), "host_promote") == "on");
    CHECK(status(svc_run({}, {{"STRIX_DS41_RTX_BASELINE", "1"}, {"STRIX_DS41_HOST_PROMOTE", "1"}}), "host_promote") == "baseline-env");
    CHECK(value(svc_run({}, {{"STRIX_DS41_RTX_BASELINE", "1"}}), "host_promote") == "off");
    CHECK(value(svc_run({"--host-promote"}, {{"STRIX_DS41_RTX_BASELINE", "1"}}), "host_promote") == "on");
    CHECK(value(svc_run({"--batch-max", "--no-host-promote"}), "host_promote") == "on");  // an argument, not the option
    CHECK(value(svc_run({"--host", "127.0.0.1", "--no-host-promote"}), "host_promote") == "off");
    CHECK(svc_run({"--baseline"}).json().find("\"host_promote\":{\"value\":\"off\",\"status\":\"baseline-cli\"") != std::string::npos);
    const auto svc_allow = run({"--profile", "fast", "--allow-pending"}, {}, pf::App::prefill_service);
    CHECK(status(svc_allow, "batch_long") == "unqualified" && value(svc_allow, "batch_long") == "on");
    CHECK(status(svc_allow, "fp16_dense_prefill") == "pending");  // not runnable: never used, even with --allow-pending
    CHECK(svc_allow.qualification() == "unqualified: batch_long");
    CHECK(svc_allow.json().find("\"unqualified\":[\"batch_long\"]") != std::string::npos);
    CHECK(run({"--profile", "fast", "--batch-max", "4"}, {}, pf::App::prefill_service).find("batch_long")->status ==
          pf::Status::explicit_cli);  // an expert override is not the profile's: never refused
    CHECK(run({"--profile", "fast"}, {{"STRIX_DS41_PROFILE_ALLOW_PENDING", "1"}}, pf::App::prefill_service).qualification() ==
          "unqualified: batch_long");

    // precedence: CLI > environment > profile > builtin
    CHECK(status(run({}, {{"STRIX_DS41_WB_STREAM", "0"}}), "wb_stream") == "explicit-env");
    CHECK(value(run({}, {{"STRIX_DS41_WB_STREAM", "0"}}), "wb_stream") == "off");
    CHECK(value(run({}, {{"STRIX_DS41_WB_STREAM", "1"}}), "wb_stream") == "on");
    CHECK(status(run({"--wb-stream"}, {{"STRIX_DS41_WB_STREAM", "0"}}), "wb_stream") == "explicit-cli");
    CHECK(value(run({"--wb-stream"}, {{"STRIX_DS41_WB_STREAM", "0"}}), "wb_stream") == "on");
    CHECK(value(run({"--no-wb-stream"}, {{"STRIX_DS41_WB_STREAM", "1"}}), "wb_stream") == "off");
    CHECK(value(run({"--profile", "fast", "--no-ce-pacing"}), "ce_pacing") == "off");
    CHECK(status(run({"--no-file-tier"}), "wb_stream") == "profile");  // other levers untouched
    // --baseline in order with the lever's own options; STRIX_DS41_RTX_BASELINE=1 over the lever's variable
    CHECK(status(run({"--wb-stream", "--baseline"}), "wb_stream") == "baseline-cli");
    CHECK(value(run({"--wb-stream", "--baseline"}), "wb_stream") == "off");
    CHECK(status(run({"--baseline", "--wb-stream"}), "wb_stream") == "explicit-cli");
    CHECK(status(run({"--baseline"}), "lean_state") == "baseline-cli");
    CHECK(status(run({}, {{"STRIX_DS41_RTX_BASELINE", "1"}, {"STRIX_DS41_WB_STREAM", "1"}}), "wb_stream") == "baseline-env");
    CHECK(status(run({"--wb-stream"}, {{"STRIX_DS41_RTX_BASELINE", "1"}}), "wb_stream") == "explicit-cli");
    CHECK(status(run({"--profile", "fast", "--baseline"}), "prompt_layer_major") == "pending");  // not a baseline lever
    // valued options: the argument is the value and never read as an option
    CHECK(value(run({"--evict", "lru"}), "evict_auto") == "lru");
    CHECK(value(run({"--controller-prior", "none"}), "controller") == "none");
    CHECK(value(run({"--draft-keep", "0"}), "draft_keep") == "0");
    CHECK(value(run({"--draft-vocab", "full", "--no-wb-stream"}), "draft_vocab_auto") == "full");
    CHECK(value(run({"--draft-vocab", "full", "--no-wb-stream"}), "wb_stream") == "off");
    CHECK(status(run({"--residency-seed", "--no-wb-stream"}), "wb_stream") == "profile");  // an argument, not the option
    CHECK(value(run({"--q8-r1-pad"}), "q8_r1") == "pad");
    CHECK(value(run({"--router-pf0"}), "decode_forms") == "off");

    // selection: --profile beats STRIX_DS41_PROFILE; bad names and a missing value refused
    CHECK(pf::select(in({}, {{"STRIX_DS41_PROFILE", "fast"}})).name == pf::Name::fast);
    CHECK(pf::select(in({"--profile", "quality"}, {{"STRIX_DS41_PROFILE", "fast"}})).name == pf::Name::quality);
    CHECK(pf::select(in({}, {{"STRIX_DS41_PROFILE", ""}})).name == pf::Name::standard);
    CHECK(throws({"--profile", "turbo"}));
    CHECK(throws({"--profile"}));
    CHECK(throws({}, {{"STRIX_DS41_PROFILE", "Fast"}}));
    CHECK(!pf::select(in({})).allow_pending);
    CHECK(pf::select(in({"--allow-pending"})).allow_pending);

    // a synthetic table: a runnable pending column is refused unless --allow-pending, then stamped; a ready column applies
    static constexpr pf::Lever synthetic[] = {
        {.name = "ready_lever", .options = {{{"--ready", "on"}}}, .value = {"off", "on", "off"}, .builtin = "off",
         .ready = {true, true, true}, .runnable = {true, true, true}, .apps = 1U, .source = "test"},
        {.name = "runnable_pending", .env = "TEST_RP", .value = {"", "on", ""}, .builtin = "off", .ready = {false, false, false},
         .runnable = {false, true, false}, .kind = pf::Kind::bit_changing, .apps = 1U, .source = "pending: test band"},
    };
    const auto syn = [&](std::vector<std::string> args, bool allow, pf::Name n = pf::Name::fast,
                         std::map<std::string, std::string> env = {}) {
        return pf::resolve(n, allow, pf::App::replica, in(std::move(args), std::move(env)), synthetic);
    };
    CHECK(value(syn({}, false), "ready_lever") == "on" && status(syn({}, false), "ready_lever") == "profile");
    CHECK(value(syn({}, false, pf::Name::standard), "ready_lever") == "off");
    CHECK(status(syn({}, false), "runnable_pending") == "pending" && value(syn({}, false), "runnable_pending") == "off");
    CHECK(syn({}, false).qualification() == "qualified");
    CHECK(status(syn({}, true), "runnable_pending") == "unqualified" && value(syn({}, true), "runnable_pending") == "on");
    CHECK(syn({}, true).qualification() == "unqualified: runnable_pending");
    CHECK(syn({}, true, pf::Name::standard).find("runnable_pending") == nullptr);  // default leaves it alone
    CHECK(status(syn({}, true, pf::Name::fast, {{"TEST_RP", "0"}}), "runnable_pending") == "explicit-env");
    CHECK(syn({}, true, pf::Name::fast, {{"TEST_RP", "0"}}).qualification() == "qualified");

    // the receipt JSON carries the app's effective value when given
    const auto j = none.json([](std::string_view l) { return l == "wb_stream" ? std::string("on") : std::string(); });
    CHECK(j.find("\"wb_stream\":{\"value\":\"on\",\"status\":\"profile\",\"effective\":\"on\",\"switch\":\"--wb-stream --no-wb-stream STRIX_DS41_WB_STREAM\"") !=
          std::string::npos);
    CHECK(j.rfind("{\"profile\":\"default\",\"app\":\"replica\",\"qualification\":\"qualified\",\"allow_pending\":false,\"levers\":{", 0) == 0);

    // Every row, every native option, both apps: baseline is ordered with CLI;
    // own environment booleans and empty values retain the replica's policy.
    for (const auto app : {pf::App::replica, pf::App::prefill_service}) {
        for (const auto& row : pf::levers()) if (row.apps & unsigned(app)) {
            const auto key = std::string(row.name);
            const auto get = [&](std::vector<std::string> args, std::map<std::string, std::string> env = {}) {
                return run(std::move(args), std::move(env), app);
            };
            if (row.baseline) {
                for (const auto& baseline : {get({"--baseline"}), get({}, {{"STRIX_DS41_RTX_BASELINE", "1"}})}) {
                    CHECK(baseline.find(key)->value == "off");
                }
            }
            if (!row.env.empty()) {
                for (const char* v : {"0", "1", "false", ""}) {
                    const auto env = std::map<std::string, std::string>{{std::string(row.env), v}};
                    const auto resolved = get({}, env);
                    CHECK(resolved.find(key)->value == (std::string_view(v) == "0" ? "off" : "on"));
                    CHECK(resolved.find(key)->status == (*v ? pf::Status::explicit_env : pf::Status::profile));
                    auto base_env = env; base_env["STRIX_DS41_RTX_BASELINE"] = "1";
                    if (row.baseline) CHECK(get({}, base_env).find(key)->status == pf::Status::baseline_env);
                }
            }
            for (const auto& option : row.options) if (!option.flag.empty()) {
                std::vector<std::string> args{std::string(option.flag)};
                const std::string wanted = option.value == "*" ? "0" : std::string(option.value);
                if (option.value == "*") args.push_back(wanted);
                const auto own = get(args);
                CHECK(own.find(key)->value == wanted && own.find(key)->status == pf::Status::explicit_cli);
                auto after = args; after.push_back("--baseline");
                const auto last_baseline = get(after);
                CHECK(last_baseline.find(key)->value == (row.baseline ? "off" : wanted));
                CHECK(last_baseline.find(key)->status == (row.baseline ? pf::Status::baseline_cli : pf::Status::explicit_cli));
                auto before = args; before.insert(before.begin(), "--baseline");
                CHECK(get(before).find(key)->value == wanted && get(before).find(key)->status == pf::Status::explicit_cli);
                CHECK(get(args, {{"STRIX_DS41_RTX_BASELINE", "1"}}).find(key)->status == pf::Status::explicit_cli);
                if (!row.env.empty()) CHECK(get(args, {{std::string(row.env), "0"}}).find(key)->value == wanted);
            }
        }
        const auto resolved = run({}, {}, app);
        const auto unknown = resolved.host_receipt();
        CHECK(unknown.engine == "deepseek41-rtx" && unknown.implementation == pf::app_name(app));
        CHECK(unknown.identity.arithmetic_contract.empty() && unknown.identity.model_sha256.empty() && unknown.identity.pack_hash.empty());
        CHECK(unknown.identity.table_hash.empty() && unknown.identity.state_format.empty() && unknown.identity.prefill_lineage.empty());
        CHECK(unknown.identity.reduction_plan.empty() && unknown.build.empty() && unknown.kernel.empty() && unknown.device.empty() && unknown.topology.empty());
        CHECK(unknown.digests.size() == 1 && unknown.digests[0].first == "profile_json_fnv1a64");
        CHECK(unknown.digests[0].second == strix::profile::hex_identity(strix::host_fnv::text(resolved.json())));
        strix::profile::NumericalIdentity identity;
        identity.arithmetic_contract = "caller-admitted-rtx-contract";
        identity.state_format = "caller-known-state-format";
        const auto receipt = resolved.host_receipt(identity);
        CHECK(receipt.identity.id() == identity.id());
        CHECK(receipt.witness() == "strix-engine-witness " + receipt.json() + "\n");
    }
    // Shared admission must not add the common builtin shortcut: RTX reports an
    // unready wanted==builtin column pending, even when both strings match.
    for (const bool ready : {false, true}) for (const bool runnable : {false, true})
    for (const bool allow : {false, true}) for (const auto profile : {pf::Name::standard, pf::Name::fast}) {
        pf::Lever row{.name = "same_builtin", .value = {"off", "off", "off"}, .builtin = "off",
                      .ready = {ready, ready, ready}, .runnable = {runnable, runnable, runnable}, .apps = 1U, .source = {}};
        const auto r = pf::resolve(profile, allow, pf::App::replica, in({}), std::span(&row, 1));
        const auto expected = ready ? pf::Status::profile : allow && runnable ? pf::Status::unqualified : pf::Status::pending;
        CHECK(r.find("same_builtin")->status == expected && r.find("same_builtin")->value == "off");
        CHECK(r.find("same_builtin")->wanted == (ready ? "" : "off"));
    }

    // Identity pins, captured from the resolver before it was shared with the service: no word, `--profile exact`,
    // `--profile default` and STRIX_DS41_PROFILE=exact all give today's receipt JSON byte for byte, for both apps.
    {
        const std::string service_json =
            R"J({"profile":"default","app":"prefill-service","qualification":"qualified","allow_pending":false,"levers":{"host_promote":{"value":"on","status":"profile","switch":"--host-promote --no-host-promote STRIX_DS41_HOST_PROMOTE","kind":"exact","source":"lazy host promotion of streamed file-tier expert layers (exact: the same file bytes; above the host floor)"}},"pending":[],"unqualified":[]})J";
        const std::vector<Case> words = {
            Case{{}, {}}, Case{{"--profile", "exact"}, {}}, Case{{"--profile", "default"}, {}}, Case{{}, {{"STRIX_DS41_PROFILE", "exact"}}}};
        for (const auto& [args, env] : words) {
            const auto rep = pf::resolve(pf::App::replica, in(args, env));
            CHECK(rep.json().size() == 3501U && strix::host_fnv::text(rep.json()) == 0x31f78efb211de9a0ULL);
            CHECK(pf::resolve(pf::App::prefill_service, in(args, env)).json() == service_json);
            CHECK(rep.qualification() == "qualified" && rep.pending().empty() == false);  // prompt_layer_major pending, as today
        }
        // fast: the replica's receipt is today's; the service lists its fast rows as pending and applies nothing new
        CHECK(strix::host_fnv::text(run({"--profile", "fast"}).json()) == 0xfc25babb029d49fdULL);
        const auto sf = run({"--profile", "fast"}, {}, pf::App::prefill_service);
        CHECK(sf.on("host_promote") && !sf.on("batch_long") && !sf.on("fp16_dense_prefill"));
        CHECK(status(sf, "batch_long") == "pending" && status(sf, "fp16_dense_prefill") == "pending");
        CHECK(sf.qualification() == "qualified");
    }
    // Replica and service resolve the same word through the same function: same name / allow-pending, the same
    // --baseline and STRIX_DS41_RTX_BASELINE=1 reading, turbo refused by both.
    {
        const std::vector<Case> cases = {
            Case{{"--profile", "fast"}, {}}, Case{{"--profile", "exact", "--allow-pending"}, {}}, Case{{}, {{"STRIX_DS41_PROFILE", "fast"}}},
            Case{{"--profile", "quality"}, {{"STRIX_DS41_PROFILE_ALLOW_PENDING", "1"}}}, Case{{"--profile", "fast", "--baseline"}, {}}};
        for (const auto& [args, env] : cases) {
            const auto r = pf::resolve(pf::App::replica, in(args, env));
            const auto sv = pf::resolve(pf::App::prefill_service, in(args, env));
            const auto sel = pf::select(in(args, env));
            CHECK(r.name == sv.name && r.name == sel.name && r.allow_pending == sv.allow_pending && r.allow_pending == sel.allow_pending);
        }
        CHECK(!run({"--baseline"}, {}, pf::App::prefill_service).on("host_promote") && !run({"--baseline"}).on("lean_state"));
        CHECK(!run({}, {{"STRIX_DS41_RTX_BASELINE", "1"}}, pf::App::prefill_service).on("host_promote"));
        CHECK(!run({}, {{"STRIX_DS41_RTX_BASELINE", "1"}}).on("lean_state"));
        CHECK(run({"--baseline", "--host-promote"}, {}, pf::App::prefill_service).on("host_promote"));
        CHECK(throws({"--profile", "turbo"}) && throws({"--profile", "bogus"}) && throws({"--profile"}));
        CHECK(pf::parse("exact") == pf::Name::standard && pf::parse("default") == pf::Name::standard);
        CHECK(pf::shared_rows().size() == pf::levers().size());
    }

    if (failures) {
        std::fprintf(stderr, "ds41_rtx_profile_tests: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("PASS DS RTX profile: table (default = quality = today, fast = today + pending), precedence, baseline order, "
                "pending refusal, --allow-pending stamping, selection, receipt JSON\n");
    return 0;
}
