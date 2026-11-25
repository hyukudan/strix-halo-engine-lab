// Host tests of the [multi_device] model data and its one resolver (strix/profile/multi_device.hpp):
// defaults equal today's single-device behaviour, regime selection, tier rules for the wire, explicit placement,
// legacy environment reports, and the registered model descriptions.
#include "strix/model_registry.hpp"
#include "strix/profile/model_preset.hpp"
#include "strix/profile/multi_device.hpp"

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <string>

namespace {
namespace md = strix::profile::multi_device;
using strix::profile::Name;

int failures = 0;
void require(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

void refuses(const std::function<void()>& f, const std::string& needle) {
    try {
        f();
        require(false, "expected refusal containing '" + needle + "'");
    } catch (const std::invalid_argument& e) {
        require(std::string(e.what()).find(needle) != std::string::npos,
                "refusal '" + std::string(e.what()) + "' should contain '" + needle + "'");
    }
}

md::Policy policy(const std::string& text) { return md::parse_policy(strix::model_description::parse(text)); }

md::Query query(Name tier, std::uint32_t devices, md::Phase phase = md::Phase::request,
                std::optional<std::uint64_t> rows = {}, std::uint64_t ready = 1) {
    md::Query q;
    q.tier = tier;
    q.devices = devices;
    q.phase = phase;
    q.prompt_rows = rows;
    q.ready_requests = ready;
    return q;
}

const char* kFull = R"(
[multi_device]
version = 1
default = "replica"
fallback = "single"
order = ["short_concurrent", "long_prefill", "long_prefill_fp8", "parked"]

[multi_device.forms.replica]
ranks = 2
objective = "throughput"

[multi_device.forms.layers]
ranks = 2
axis = "visit"
cut = "auto"
components = ["trunk"]
window_rows = 2048
state = "relay-if-needed"

[multi_device.forms.tensor]
ranks = 2
components = ["attention", "ffn", "moe", "head"]
expert_partition = "intermediate"
expert_shares = [0.5, 0.5]
wire = "bf16-exact"
rows_per_message = "auto"
draft_placement = "leader"
draft_rank = 0

[multi_device.regimes.short_concurrent]
max_prompt_rows = 4095
min_ready_requests = 2
prefer = ["replica", "single"]

[multi_device.regimes.long_prefill]
phase = "prefill"
min_prompt_rows = 4096
prefer = ["layers", "single"]

[multi_device.regimes.long_prefill_fp8]
phase = "prefill"
min_prompt_rows = 16384
min_tier = "fast"
prefer = ["tensor"]
wire = "fp8-e4m3-g64-v1"
lever = "pair-wire-fp8@1"

[multi_device.regimes.parked]
phase = "decode"
prefer = ["tensor"]
enabled = false
)";

void defaults_are_single() {
    const auto absent = policy("name = 'x'\n");
    require(absent.version == 0 && absent.forms.size() == 1, "absent section declares only single");
    for (Name tier : {Name::standard, Name::fast, Name::turbo})
        for (std::uint32_t devices : {1U, 2U, 8U}) {
            const auto d = md::select(absent, query(tier, devices, md::Phase::prefill, 100000, 16));
            require(d.form == md::Form::single && d.ranks == 1 && d.wire == "reference" && d.rule == "absent",
                    "a model without [multi_device] always resolves single");
        }
    refuses([&] { (void)md::select(absent, [] { md::Query q; q.placement = md::Form::replica; q.devices = 2; return q; }()); },
            "replica is not declared");
}

void regimes() {
    const auto p = policy(kFull);
    require(p.regimes.size() == 4 && p.regimes[0].name == "short_concurrent", "regimes follow `order`");
    // One device: the replica default cannot run -> fallback single (today's behaviour on one node).
    auto d = md::select(p, query(Name::standard, 1));
    require(d.form == md::Form::single && d.rule == "fallback" && d.reason.find("needs 2 devices") != std::string::npos,
            "one device falls back to single with the reason");
    // Two devices, one request of unknown length: no regime matches -> default replica over every device.
    d = md::select(p, query(Name::standard, 2));
    require(d.form == md::Form::replica && d.ranks == 2 && d.rule == "default", "two devices default to replicas");
    d = md::select(p, query(Name::standard, 3));
    require(d.form == md::Form::replica && d.ranks == 3, "replicas use every device");
    // Short prompts with concurrency -> replicas through the regime.
    d = md::select(p, query(Name::standard, 2, md::Phase::request, 512, 4));
    require(d.form == md::Form::replica && d.rule == "regime:short_concurrent", "short concurrent -> replica");
    // Long prefill -> staged split, EXACT keeps the reference wire.
    d = md::select(p, query(Name::standard, 2, md::Phase::prefill, 8192));
    require(d.form == md::Form::layers && d.rule == "regime:long_prefill" && d.wire == "reference",
            "long prefill -> layers with the reference wire");
    // A request-phase query never matches a prefill regime.
    d = md::select(p, query(Name::standard, 2, md::Phase::request, 8192));
    require(d.form == md::Form::replica && d.rule == "default", "phase gates a regime");
    // Very long prefill: EXACT skips the FP8 regime (min_tier) and takes long_prefill; FAST takes the FP8 wire.
    d = md::select(p, query(Name::standard, 2, md::Phase::prefill, 32768));
    require(d.form == md::Form::layers && d.rule == "regime:long_prefill", "EXACT never takes the FP8 regime");
    // Order: long_prefill precedes long_prefill_fp8, so FAST also matches long_prefill first.
    d = md::select(p, query(Name::fast, 2, md::Phase::prefill, 32768));
    require(d.rule == "regime:long_prefill", "regimes are evaluated in declared order");
    // Disabled regimes are data only.
    d = md::select(p, query(Name::turbo, 2, md::Phase::decode));
    require(d.form == md::Form::replica && d.rule == "default", "a disabled regime is never selected");
    // The JSON states form, wire, rule and the declared forms.
    const auto json = d.json(p, query(Name::turbo, 2, md::Phase::decode)).dump();
    require(json.find("\"form\":\"replica\"") != std::string::npos && json.find("\"declared\":[") != std::string::npos,
            "effective JSON states the form and the declared forms: " + json);
}

void wire_tiers() {
    const auto p = policy(R"(
[multi_device]
version = 1
[multi_device.forms.tensor]
wire = "fp16"
[multi_device.regimes.long]
min_prompt_rows = 1024
min_tier = "fast"
prefer = ["tensor"]
)");
    auto d = md::select(p, query(Name::fast, 2, md::Phase::request, 2048));
    require(d.form == md::Form::tensor && d.wire == "fp16", "FAST takes a declared narrow wire");
    d = md::select(p, query(Name::standard, 2, md::Phase::request, 2048));
    require(d.form == md::Form::single, "EXACT never resolves a narrow wire");
    md::Query explicit_exact = query(Name::standard, 2);
    explicit_exact.placement = md::Form::tensor;
    refuses([&] { (void)md::select(p, explicit_exact); }, "changes bits and the tier is exact");
    md::Query explicit_fast = query(Name::fast, 2);
    explicit_fast.placement = md::Form::tensor;
    d = md::select(p, explicit_fast);
    require(d.form == md::Form::tensor && d.rule == "explicit", "explicit placement wins when it can run");
    explicit_fast.devices = 1;
    refuses([&] { (void)md::select(p, explicit_fast); }, "needs 2 devices, 1 available");
}

void validation() {
    refuses([] { (void)policy("[multi_device]\ndefault = 'single'\n"); }, "version = 1");
    refuses([] { (void)policy("[multi_device]\nversion = 1\nplace = 1\n"); }, "unknown key multi_device.place");
    refuses([] { (void)policy("[multi_device]\nversion = 1\ndefault = 'replica'\n"); }, "undeclared form");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.forms.replica]\nwire = 'fp16'\n"); },
            "unknown key multi_device.forms.replica.wire");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.forms.replica]\n"
                              "[multi_device]\n"); }, "");
    refuses([] { (void)policy("[multi_device]\nversion = 1\nfallback = 'replica'\n[multi_device.forms.replica]\n"); },
            "fallback must be");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.forms.tensor]\ndraft_rank = 2\n"); },
            "draft_rank");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.forms.speculation]\n"); },
            "leader or split");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.forms.tensor]\nwire = 'int4'\n"); },
            "must be one of");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.regimes.a]\nprefer = ['replica']\n"
                              "wire = 'fp8-e4m3-g64-v1'\n"); }, "needs min_tier");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.regimes.a]\nprefer = ['x']\n"); },
            "unknown form");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.regimes.a]\nprefer = ['single']\n"
                              "[multi_device.regimes.b]\nprefer = ['single']\n"); }, "order must list every regime");
    // Review r9 M2: a repeated name no longer stands in for a missing regime, and a short order is refused.
    refuses([] { (void)policy("[multi_device]\nversion = 1\norder = ['a', 'a']\n[multi_device.regimes.a]\n"
                              "prefer = ['single']\n[multi_device.regimes.b]\nprefer = ['single']\n"); },
            "repeats");
    refuses([] { (void)policy("[multi_device]\nversion = 1\norder = ['a']\n[multi_device.regimes.a]\n"
                              "prefer = ['single']\n[multi_device.regimes.b]\nprefer = ['single']\n"); },
            "order must list every regime");
    refuses([] { (void)policy("[multi_device]\nversion = 1\norder = ['a', 'a']\n[multi_device.regimes.a]\n"
                              "prefer = ['single']\n"); }, "repeats");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.regimes.a]\nprefer = ['single']\n"
                              "min_prompt_rows = 9\nmax_prompt_rows = 8\n"); }, "min_prompt_rows > max_prompt_rows");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.forms.tensor]\nexpert_shares = [0.7, 0.7]\n"); },
            "sum to 1");
    // A regime names only declared forms, enabled or parked: enabling it can never resolve a missing form to single.
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.regimes.a]\nprefer = ['layers', 'single']\n"); },
            "prefer names 'layers', which the model does not declare");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.regimes.a]\nprefer = ['layers']\n"
                              "enabled = false\n"); }, "does not declare");
    // A regime wire binds to forms that carry one; with prefer = ['single'] it would be reported as reference.
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.regimes.a]\nprefer = ['single']\n"
                              "wire = 'fp16'\nmin_tier = 'fast'\n"); }, "cannot apply to 'single'");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.forms.tensor]\nranks = 2\n"
                              "[multi_device.regimes.a]\nprefer = ['tensor', 'single']\nwire = 'fp16'\n"
                              "min_tier = 'fast'\n"); }, "cannot apply to 'single'");
    // A preferred form whose own wire changes bits needs a FAST / TURBO regime.
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.forms.tensor]\nranks = 2\nwire = 'fp16'\n"
                              "[multi_device.regimes.a]\nprefer = ['tensor']\n"); }, "whose wire 'fp16' changes bits");
    const auto bound = policy("[multi_device]\nversion = 1\n[multi_device.forms.tensor]\nranks = 2\n"
                              "[multi_device.regimes.a]\nprefer = ['tensor']\nwire = 'fp16'\nmin_tier = 'fast'\n");
    require(bound.regimes.size() == 1 && bound.regimes[0].wire == "fp16", "a wire bound to a tensor regime parses");
}

void legacy_environment(const std::filesystem::path& root) {
    const std::map<std::string, std::string> env{{"STRIX_QWEN_PF_PIPE_CUT", "20"}, {"STRIX_HY4_TP_DRAFT", "both"},
                                                 {"STRIX_HY4_TP_JOIN_BF16", "1"}, {"STRIX_UNRELATED", "1"},
                                                 {"STRIX_G4_PAIR", ""}};
    const strix::profile::Lookup lookup = [&](const char* key) -> const char* {
        const auto it = env.find(key);
        return it == env.end() ? nullptr : it->second.c_str();
    };
    // The inventory is the selected model's own data ([multi_device.legacy_env]): a model that declares none reports none.
    const char* inventory =
        "name = 'x'\n[multi_device.legacy_env]\nSTRIX_QWEN_PF_PIPE_CUT = 'forms.layers.cut'\n"
        "STRIX_HY4_TP_DRAFT = 'forms.tensor.draft_placement'\nSTRIX_G4_PAIR = 'forms.layers'\n";
    const auto found = md::legacy_overrides(policy(inventory), lookup);
    require(found.size() == 2, "set legacy switches are reported, empty and unrelated ones are not");
    bool cut = false, draft = false;
    for (const auto& o : found) {
        cut |= o.env == "STRIX_QWEN_PF_PIPE_CUT" && o.value == "20" && o.key == "multi_device.forms.layers.cut";
        draft |= o.env == "STRIX_HY4_TP_DRAFT" && o.key == "multi_device.forms.tensor.draft_placement";
    }
    require(cut && draft, "legacy switches map to their [multi_device] keys");
    require(md::legacy_overrides(policy("name = 'x'\n"), lookup).empty(), "a model with no inventory reports none");
    // An inventory-only section declares no placement policy: the model still resolves like one without [multi_device].
    require(policy(inventory).version == 0 &&
            md::select(policy(inventory), query(Name::standard, 2)).rule == "absent", "inventory alone is not a placement policy");
    refuses([] { (void)policy("[multi_device.legacy_env]\nOTHER = 'forms.layers'\n"); }, "must map a STRIX_* switch");
    // Reports never change the decision: engines still read their switches.
    const auto described = strix::model_description::parse(inventory);
    const auto d = md::resolve(described, query(Name::standard, 1), lookup);
    require(d.form == md::Form::single && d.legacy.size() == 2, "legacy reports ride on the decision");
    const auto json = d.json(policy(inventory), query(Name::standard, 1)).dump();
    require(json.find("\"legacy_env\":[{") != std::string::npos, "effective JSON lists legacy switches");
    // No resolver reads the planned keys yet: the report names the env and its reader, never a key or a deprecation.
    require(json.find("\"read_by\":\"engine\"") != std::string::npos && json.find("deprecated") == std::string::npos &&
            json.find("multi_device.forms") == std::string::npos, "legacy report names no unread key");
    // Regression (review r5): another family's switch never appears in this model's report.
    const auto qwen = strix::model_description::load((root / "models/qwen3.8-27b.toml").string());
    const auto hy4 = strix::model_description::load((root / "models/hy4-flash.toml").string());
    require(md::resolve(qwen, query(Name::standard, 1), lookup).legacy.empty(),
            "a Hy4 switch is not engine-read for the Qwen model");
    const auto own = md::resolve(hy4, query(Name::standard, 1), lookup).legacy;
    require(own.size() == 2, "the Hy4 model lists its own two set switches");
    const strix::profile::Lookup qwen_env = [](const char* key) -> const char* {
        return std::string_view(key) == "STRIX_TP_F16_GATHER" ? "1" : nullptr;
    };
    require(md::resolve(qwen, query(Name::standard, 1), qwen_env).legacy.size() == 1 &&
            md::resolve(hy4, query(Name::standard, 1), qwen_env).legacy.empty(), "each model lists only its own switches");
}

// Regression (review r5): the form's declared wire never contradicts the regime's effective wire in the report.
std::string word(const strix::serve::Json* value) { return value ? value->as_string() : "<absent>"; }

void effective_wire_report() {
    const auto p = policy(R"(
[multi_device]
version = 1
[multi_device.forms.tensor]
wire = "fp16"
[multi_device.regimes.exact_long]
min_prompt_rows = 1024
wire = "reference"
prefer = ["tensor"]
)");
    const auto q = query(Name::standard, 2, md::Phase::request, 2048);
    const auto d = md::select(p, q);
    require(d.form == md::Form::tensor && d.wire == "reference", "an EXACT regime picks tensor on the reference wire");
    const auto json = d.json(p, q);
    const auto* settings = json.find("settings");
    require(settings && word(json.find("wire")) == "reference" && word(settings->find("wire")) == "reference",
            "settings.wire agrees with the effective wire");
    require(settings && word(settings->find("declared_wire")) == "fp16", "the form's declared wire stays visible, labelled");
    require(word(json.find("wire_status")) == "exact", "an exact wire is reported exact");
    const auto fast = policy(R"(
[multi_device]
version = 1
[multi_device.forms.tensor]
wire = "fp16"
[multi_device.regimes.long]
min_tier = "fast"
prefer = ["tensor"]
)");
    const auto fq = query(Name::fast, 2);
    const auto fj = md::select(fast, fq).json(fast, fq);
    require(word(fj.find("wire")) == "fp16" && word(fj.find("settings")->find("wire")) == "fp16" &&
            !fj.find("settings")->find("declared_wire") &&
            word(fj.find("wire_status")).find("report-only") == 0, "a bit-changing wire is report-only");
}

// A launch applies single and replica only; any other resolved form is refused with its reason, never printed.
void launch_applies_single_and_replica() {
    const auto description = strix::model_description::parse(
        "name = 'x'\n[multi_device]\nversion = 1\ndefault = 'replica'\n[multi_device.forms.replica]\nranks = 2\n"
        "[multi_device.forms.tensor]\nranks = 2\n");
    md::Query q;
    q.devices = 2;
    // Inspection reports the replica decision and who executes it ...
    const auto replica = strix::profile::resolve_model_preset(description, Name::fast, "hip", q);
    require(replica.placement.form == md::Form::replica, "replica is reported");
    require(replica.config.dump().find("\"executed_by\":\"router\"") != std::string::npos, "the report names the router");
    // ... a standalone engine launch refuses it (it would start one worker), the router admits it.
    refuses([&] { md::admit(replica.placement, md::Executor::engine); }, "runs behind strix-router");
    require(strix::profile::resolve_replica_router(description, {}, 2).placement.form == md::Form::replica,
            "the replica router admits the model");
    md::Query one;
    refuses([&] { md::admit(strix::profile::resolve_model_preset(description, {}, "hip", one).placement,
                            md::Executor::router); }, "strix-router executes only replica");
    // A form no engine executes is reported, refused by both executors.
    q.placement = md::Form::tensor;
    const auto tensor = strix::profile::resolve_model_preset(description, Name::fast, "hip", q);
    require(tensor.placement.form == md::Form::tensor &&
            tensor.config.dump().find("\"executed_by\":\"none\"") != std::string::npos, "tensor is reported, not executed");
    refuses([&] { md::admit(tensor.placement, md::Executor::engine); },
            "'tensor' (explicit: --placement) is not executed by any engine yet");
    refuses([&] { md::admit(tensor.placement, md::Executor::router); }, "executes only replica");
    q.placement = md::Form::single;
    const auto single = strix::profile::resolve_model_preset(description, Name::fast, "hip", q);
    require(single.placement.form == md::Form::single, "explicit single resolves");
    md::admit(single.placement, md::Executor::engine);
}

void registered_models(const std::filesystem::path& root) {
    const auto entries = strix::model_registry::load((root / "models/registry.toml").string());
    require(!entries.empty(), "registry has models");
    for (const auto& entry : entries) {
        const auto description = strix::model_description::load((root / entry.description).string());
        const auto p = md::parse_policy(description);
        // Today's behaviour: one device always resolves single, whatever the model declares.
        for (Name tier : {Name::standard, Name::fast, Name::turbo}) {
            const auto d = md::select(p, query(tier, 1, md::Phase::prefill, 65536, 16));
            require(d.form == md::Form::single, entry.name + ": one device resolves single");
        }
        // The preset resolver states the placement in every effective config.
        const auto preset = strix::profile::resolve_model_preset(description, {}, "hip");
        const auto* member = preset.config.find("multi_device");
        require(member && member->find("form") && member->find("form")->as_string() == "single",
                entry.name + ": effective config states multi_device.form = single on one device");
        if (entry.name == "inkling2-mini") {
            // Two whole-model devices: explicit replicas for every tier and phase, with single as the default.
            require(p.regimes.empty() && p.forms.size() == 2, "inkling2-mini: single + replica, no parked regime");
            for (Name tier : {Name::standard, Name::fast, Name::turbo})
                for (auto phase : {md::Phase::request, md::Phase::prefill, md::Phase::decode}) {
                    auto selected = query(tier, 2, phase, 32768, 1);
                    require(md::select(p, selected).form == md::Form::single, "inkling2-mini: single is the default");
                    selected.placement = md::Form::replica;
                    const auto d = md::select(p, selected);
                    require(d.form == md::Form::replica && d.ranks == 2 && d.wire == "reference",
                            "inkling2-mini: two devices resolve replicas");
                }
            const auto router = strix::profile::resolve_replica_router(description, {}, 2, true);
            require(router.placement.form == md::Form::replica && router.requested == Name::standard,
                    "inkling2-mini: the router admits explicit replicas at the default tier");
            // One endpoint is not a replica set: the router refuses it, for every tier.
            for (Name tier : {Name::standard, Name::fast, Name::turbo})
                refuses([&] { (void)strix::profile::resolve_replica_router(description, tier, 1); },
                        "does not resolve the replica form for 1 replicas");
            // An explicit replica launch reports replicas but is not admitted: it would start one worker.
            md::Query two;
            two.devices = 2;
            two.placement = md::Form::replica;
            const auto standalone = strix::profile::resolve_model_preset(description, {}, "hip", two);
            require(standalone.placement.form == md::Form::replica, "inkling2-mini: two devices report replicas");
            refuses([&] { md::admit(standalone.placement, md::Executor::engine); }, "runs behind strix-router");
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: strix-multi-device-tests SOURCE_ROOT\n";
        return 2;
    }
    defaults_are_single();
    regimes();
    wire_tiers();
    validation();
    legacy_environment(argv[1]);
    effective_wire_report();
    launch_applies_single_and_replica();
    registered_models(argv[1]);
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "multi_device: ok\n";
    return 0;
}
