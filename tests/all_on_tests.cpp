// Host tests of --all-on (strix/profile/all_on.hpp): the tier rules over a fixture catalogue with exact printed lines,
// the invariants of the embedded catalogue for every model description, the shell binding and the CLI option.
//   strix-all-on-tests ROOT           run the tests
//   strix-all-on-tests ROOT --table   print the per engine x tier lever table (markdown) from the real resolver
#include "strix/cli/cli.hpp"
#include "strix/model_description.hpp"
#include "strix/profile/all_on.hpp"
#include "strix/serve/json.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <string>

namespace {
namespace ao = strix::profile::all_on;
using strix::profile::Name;

int failures = 0;
void require(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

void put(const char* key, const char* value) {
#ifdef _WIN32
    (void)_putenv_s(key, value ? value : "");
#else
    if (value) (void)::setenv(key, value, 1);
    else (void)::unsetenv(key);
#endif
}

// One lever of every kind. "m" is the engine under test, "other" a second engine.
constexpr char fixture[] = R"json([
{"row":"r","lever":"e_env","class":"exact","gate":"none","engines":["m"],"backends":["hip"],"phase":"any","env":"X_E_ENV","value":"1"},
{"row":"r","lever":"e_args","class":"exact","gate":"none","engines":["m"],"backends":["hip","cuda"],"phase":"any","args":["--e-args","on"]},
{"row":"r","lever":"f_pass","class":"fast","gate":"pass","engines":["m"],"backends":["hip"],"phase":"any","env":"X_F_PASS","value":"1"},
{"row":"r","lever":"f_pend","class":"fast","gate":"pending","engines":["m"],"backends":["hip"],"phase":"any","env":"X_F_PEND","value":"1","quality":"KL 0.01"},
{"row":"r","lever":"f_fail","class":"fast","gate":"fail","engines":["m"],"backends":["hip"],"phase":"any","env":"X_F_FAIL","value":"1","quality":"turbo limits pass"},
{"row":"r","lever":"t_only","class":"turbo","gate":"none","engines":["m"],"backends":["hip"],"phase":"any","env":"X_T_ONLY","value":"1","quality":"dNLL +0.005"},
{"row":"r","lever":"t_default","class":"turbo","gate":"none","engines":["m"],"backends":["hip"],"phase":"any","env":"X_T_DEF","value":"1","default_tiers":["turbo"],"quality":"q"},
{"row":"r","lever":"kv_fp8","class":"kv","gate":"none","engines":["*"],"backends":["hip","cuda"],"phase":"any","default_tiers":["turbo"]},
{"row":"r","lever":"unk","class":"unclassified","gate":"none","engines":["m"],"backends":[],"phase":"any","reason":"no receipt"},
{"row":"r","lever":"pair_a","class":"exact","gate":"none","engines":["m"],"backends":["hip"],"phase":"any","env":"X_PA","value":"1","incompatible_with":["pair_b"]},
{"row":"r","lever":"pair_b","class":"exact","gate":"none","engines":["m"],"backends":["hip"],"phase":"any","env":"X_PB","value":"1","incompatible_with":["pair_a"]},
{"row":"r","lever":"needs_t","class":"exact","gate":"none","engines":["m"],"backends":["hip"],"phase":"any","env":"X_NT","value":"1","requires":["t_only"]},
{"row":"r","lever":"banded","class":"exact","gate":"none","engines":["m"],"backends":["hip"],"phase":"prefill","tokens":[4096,8192],"env":"X_BAND","value":"1"},
{"row":"r","lever":"cuda_only","class":"exact","gate":"none","engines":["m"],"backends":["cuda"],"phase":"any","env":"X_CUDA","value":"1"},
{"row":"r","lever":"foreign","class":"exact","gate":"none","engines":["other"],"backends":["hip"],"phase":"any","env":"X_FOREIGN","value":"1"}
])json";

using EnvValue = std::function<std::optional<std::string>(std::string_view)>;

// One explicit environment value; every other switch unset.
EnvValue only(std::string key, std::string value) {
    return [key = std::move(key), value = std::move(value)](std::string_view name) -> std::optional<std::string> {
        if (name == key) return value;
        return std::nullopt;
    };
}

std::string lines_of(const std::vector<ao::Lever>& levers, ao::Tier tier, ao::Band band = {}, EnvValue env = {}) {
    ao::Query query;
    query.tier = tier;
    query.backend = "hip";
    query.engines = {"m"};
    query.band = band;
    query.env_value = std::move(env);
    return ao::resolve(levers, query).text();
}

void fixture_tests() {
    const auto levers = ao::parse_catalogue(fixture);
    require(lines_of(levers, ao::Tier::exact) ==
                "lever e_env source=all-on\n"
                "lever e_args source=all-on\n"
                "lever kv_fp8 source=refused reason=KV precision is a separate toggle, off in exact\n"
                "lever unk source=refused reason=unclassified: no receipt\n"
                "lever pair_a source=all-on\n"
                "lever pair_b source=refused reason=incompatible with pair_a\n"
                "lever needs_t source=refused reason=requires t_only\n"
                "lever banded source=refused reason=band 4096..8192 tokens needs --all-on-band prefill:N\n",
            "EXACT all-on adds only exact levers and keeps KV precision: got\n" + lines_of(levers, ao::Tier::exact));
    require(lines_of(levers, ao::Tier::fast) ==
                "lever e_env source=all-on\n"
                "lever e_args source=all-on\n"
                "lever f_pass source=all-on\n"
                "lever f_pend source=refused reason=gate v2 pending for fast\n"
                "lever f_fail source=refused reason=gate v2 FAIL for fast\n"
                "lever kv_fp8 source=refused reason=KV precision is a separate toggle, off in fast\n"
                "lever unk source=refused reason=unclassified: no receipt\n"
                "lever pair_a source=all-on\n"
                "lever pair_b source=refused reason=incompatible with pair_a\n"
                "lever needs_t source=refused reason=requires t_only\n"
                "lever banded source=refused reason=band 4096..8192 tokens needs --all-on-band prefill:N\n",
            "FAST all-on adds the gate-passing lossy levers only: got\n" + lines_of(levers, ao::Tier::fast));
    require(lines_of(levers, ao::Tier::turbo) ==
                "lever e_env source=all-on\n"
                "lever e_args source=all-on\n"
                "lever f_pass source=all-on\n"
                "lever f_pend source=all-on gate=pending reason=quality: KL 0.01\n"
                "lever f_fail source=all-on gate=FAIL reason=quality: turbo limits pass\n"
                "lever t_only source=all-on gate=none reason=quality: dNLL +0.005\n"
                "lever t_default source=default\n"
                "lever kv_fp8 source=default\n"
                "lever unk source=refused reason=unclassified: no receipt\n"
                "lever pair_a source=all-on\n"
                "lever pair_b source=refused reason=incompatible with pair_a\n"
                "lever needs_t source=all-on\n"
                "lever banded source=refused reason=band 4096..8192 tokens needs --all-on-band prefill:N\n",
            "TURBO all-on adds the rest with stated quality: got\n" + lines_of(levers, ao::Tier::turbo));
    // The band is part of the regime: inside it the lever applies, outside it is refused, the other phase skips it.
    const auto in_band = lines_of(levers, ao::Tier::exact, ao::parse_band("prefill:8192"));
    require(in_band.find("lever banded source=all-on\n") != std::string::npos, "in-band lever applies");
    require(lines_of(levers, ao::Tier::exact, ao::parse_band("prefill:9000"))
                .find("lever banded source=refused reason=outside the qualified band 4096..8192 tokens\n") != std::string::npos,
            "out-of-band lever is refused");
    require(lines_of(levers, ao::Tier::exact, ao::parse_band("decode")).find("banded") == std::string::npos,
            "a prefill band lever does not belong to a decode regime");
    // An explicit environment value wins and is kept, with its real state: "0" is off.
    require(lines_of(levers, ao::Tier::exact, {}, only("X_E_ENV", "1"))
                .starts_with("lever e_env source=explicit reason=explicit setting kept\n"),
            "an explicit setting is kept");
    require(lines_of(levers, ao::Tier::exact, {}, only("X_E_ENV", "0"))
                .starts_with("lever e_env source=explicit reason=explicit setting kept: off\n"),
            "an explicit 0 is kept as off");
    // A requirement switched off explicitly refuses its dependant (review P2-2, first case).
    require(lines_of(levers, ao::Tier::turbo, {}, only("X_T_ONLY", "0")).find("lever needs_t source=refused reason=requires t_only\n") !=
                std::string::npos,
            "an explicit 0 does not satisfy a requirement");
    // Incompatibility holds in both directions: an explicit later peer refuses an earlier all-on lever.
    require(lines_of(levers, ao::Tier::exact, {}, only("X_PB", "1")).find(
                "lever pair_a source=refused reason=incompatible with pair_b\nlever pair_b source=explicit reason=explicit setting kept\n") !=
                std::string::npos,
            "an explicit later peer refuses the earlier lever");
    require(lines_of(levers, ao::Tier::exact, {}, only("X_PB", "0")).find("lever pair_a source=all-on\n") != std::string::npos,
            "an explicit 0 peer does not refuse");
    // An exact lever whose device identity is pending says so on its all-on line.
    const auto pending = ao::parse_catalogue(
        R"([{"row":"r","lever":"e_id","class":"exact","gate":"none","engines":["m"],"backends":["hip"],"phase":"any","env":"X_ID","value":"1","identity":"pending"}])");
    require(lines_of(pending, ao::Tier::exact) == "lever e_id source=all-on identity=pending\n", "identity=pending is printed");
    // The plan exports exactly the all-on levers.
    ao::Query query;
    query.tier = ao::Tier::fast;
    query.backend = "hip";
    query.engines = {"m"};
    const auto plan = ao::resolve(levers, query);
    require(plan.env == (std::vector<std::pair<std::string, std::string>>{{"X_E_ENV", "1"}, {"X_F_PASS", "1"}, {"X_PA", "1"}}),
            "plan env lists the all-on levers");
    require(plan.args == (std::vector<std::string>{"--e-args", "on"}), "plan args list the all-on levers");
    // Other backend and other engine.
    query.backend = "cuda";
    require(ao::resolve(levers, query).text().find("cuda_only") != std::string::npos, "cuda lever on cuda");
    require(ao::resolve(levers, query).text().find("foreign") == std::string::npos, "other engine is skipped");
    // Malformed data is refused.
    bool refused = false;
    try {
        (void)ao::parse_catalogue(R"([{"row":"r","lever":"x","class":"exact","gate":"none","engines":["m"],"backends":[],"phase":"any","env":"A","value":"1"}])");
    } catch (const std::invalid_argument&) { refused = true; }
    require(refused, "a classified lever without backends is refused");
    refused = false;
    try {
        (void)ao::parse_catalogue(R"([
{"row":"r","lever":"a","class":"exact","gate":"none","engines":["m"],"backends":["hip"],"phase":"any","env":"A","value":"1","incompatible_with":["b"]},
{"row":"r","lever":"b","class":"exact","gate":"none","engines":["m"],"backends":["hip"],"phase":"any","env":"B","value":"1"}])");
    } catch (const std::invalid_argument&) { refused = true; }
    require(refused, "a one-sided incompatibility is refused");
    refused = false;
    try { (void)ao::parse_band("prefill:x"); } catch (const std::invalid_argument&) { refused = true; }
    require(refused, "a malformed band is refused");
}

strix::model_description::ModelDescription description(const std::filesystem::path& root, const std::string& name) {
    return strix::model_description::load((root / "models" / (name + ".toml")).string());
}

std::vector<std::string> models(const std::filesystem::path& root) {
    std::vector<std::string> out;
    for (const auto& entry : std::filesystem::directory_iterator(root / "models")) {
        const auto name = entry.path().stem().string();
        if (entry.path().extension() == ".toml" && name != "registry") out.push_back(name);
    }
    std::ranges::sort(out);
    return out;
}

std::set<std::string> applied(const ao::Plan& plan) {
    std::set<std::string> out;
    for (const auto& line : plan.lines) if (line.source != "refused") out.insert(line.lever);
    return out;
}

void catalogue_tests(const std::filesystem::path& root) {
    const auto& levers = ao::catalogue();
    require(!levers.empty(), "embedded catalogue is not empty");
    // Every all_on object of docs/optimizations.json is in the embedded export, and the other way round.
    std::ifstream in(root / "docs" / "optimizations.json", std::ios::binary);
    std::stringstream text;
    text << in.rdbuf();
    const auto rows = strix::serve::parse_json(text.str());
    std::set<std::string> from_rows, from_export;
    for (const auto& row : rows.as_array())
        if (const auto* item = row.find("all_on")) from_rows.insert(item->find("lever")->as_string());
    for (const auto& lever : levers) from_export.insert(lever.name);
    require(from_rows == from_export, "data/profile/all_on.json equals the all_on rows of docs/optimizations.json "
                                      "(run tools/ci/export_all_on.py)");
    require(std::ranges::any_of(levers, [](const ao::Lever& lever) { return lever.cls == ao::Class::kv; }), "KV row present");
    for (const auto& row : rows.as_array())
        if (const auto* item = row.find("all_on")) {
            const auto& cls = item->find("class")->as_string();
            require(cls == "unclassified" || cls == "kv" || item->find("needs") != nullptr,
                    item->find("lever")->as_string() + " states its preconditions (\"needs\")");
        }
    for (const auto& name : models(root)) {
        const auto desc = description(root, name);
        for (const char* backend : {"hip", "cuda"}) {
            std::set<std::string> previous;
            for (const auto tier : {ao::Tier::exact, ao::Tier::fast, ao::Tier::turbo}) {
                ao::Query query;
                query.tier = tier;
                query.backend = backend;
                query.engines = ao::engines_of(&desc);
                const auto plan = ao::resolve(levers, query);
                const std::string where = name + "/" + backend + "/" + std::to_string(static_cast<int>(tier));
                for (const auto& line : plan.lines) {
                    const auto it = std::ranges::find(levers, line.lever, &ao::Lever::name);
                    require(it != levers.end(), where + ": unknown lever " + line.lever);
                    if (it == levers.end()) continue;
                    if (line.source == "all-on") {
                        if (tier == ao::Tier::exact) require(it->cls == ao::Class::exact, where + ": " + line.lever + " is not exact");
                        if (tier == ao::Tier::fast)
                            require(it->cls == ao::Class::exact || (it->cls == ao::Class::fast && it->gate == ao::Gate::pass),
                                    where + ": " + line.lever + " is not exact or gate-passing");
                        require(it->cls != ao::Class::unclassified && it->cls != ao::Class::kv, where + ": " + line.lever + " must stay off");
                    }
                    if (it->cls == ao::Class::kv && tier != ao::Tier::turbo)
                        require(line.source == "refused", where + ": all-on must not switch KV precision");
                    if (it->cls == ao::Class::unclassified) require(line.source == "refused", where + ": unclassified stays off");
                }
                const auto now = applied(plan);
                require(std::ranges::includes(now, previous) || tier == ao::Tier::exact, where + ": tiers are nested");
                previous = now;
            }
        }
    }
}

// Preconditions and measured losing regimes (catalogue "needs" / "slower"): an unmet precondition and a band inside a
// losing regime refuse the lever with a printed reason; an explicit setting still wins.
constexpr char needs_fixture[] = R"json([
{"row":"r","lever":"n_env","class":"exact","gate":"none","engines":["m"],"backends":["hip"],"phase":"any","env":"X_N_ENV","value":"1","needs":["env:X_PACK"],"needs_reason":"a pack"},
{"row":"r","lever":"n_alt","class":"turbo","gate":"none","engines":["m"],"backends":["hip"],"phase":"decode","env":"X_N_ALT","value":"1","needs":["env:X_LOW=1|arg:--qualify-weights","model:weights.format.experts=q4_g64"]},
{"row":"r","lever":"s_pre","class":"exact","gate":"none","engines":["m"],"backends":["hip"],"phase":"prefill","env":"X_S_PRE","value":"1","needs":[],"slower":[{"phase":"prefill","tokens":[8192,32768],"receipt":"loo 8K"}]},
{"row":"r","lever":"s_dec","class":"exact","gate":"none","engines":["m"],"backends":["hip"],"phase":"any","env":"X_S_DEC","value":"1","needs":[],"slower":[{"phase":"decode","receipt":"loo dec"}]}
])json";

void needs_slower_tests() {
    const auto levers = ao::parse_catalogue(needs_fixture);
    const auto plan = [&](ao::Tier tier, std::string band, EnvValue env, std::vector<std::string> args,
                          std::vector<std::string> experts) {
        ao::Query query;
        query.tier = tier;
        query.backend = "hip";
        query.engines = {"m"};
        query.band = ao::parse_band(band);
        query.env_value = std::move(env);
        query.args = std::move(args);
        query.model_values = [experts](std::string_view key) {
            return key == "weights.format.experts" ? experts : std::vector<std::string>{};
        };
        return ao::resolve(levers, query).text();
    };
    const std::vector<std::string> bf16{"bf16", "nvfp4"}, q4{"q4_g64"};
    const auto none = plan(ao::Tier::turbo, "any", {}, {}, bf16);
    require(none == "lever n_env source=refused reason=precondition env:X_PACK unmet: a pack\n"
                    "lever n_alt source=refused reason=precondition env:X_LOW=1|arg:--qualify-weights unmet\n"
                    "lever s_pre source=refused reason=measured slower in prefill 8192..32768 tokens (loo 8K); a "
                    "--all-on-band length outside it applies it\n"
                    "lever s_dec source=refused reason=measured slower in decode (loo dec)\n",
            "unmet preconditions and losing regimes refuse with their reasons: got\n" + none);
    const auto met = plan(ao::Tier::turbo, "decode", only("X_PACK", "/p"), {"--qualify-weights=k"}, q4);
    require(met.find("lever n_env source=all-on\n") != std::string::npos &&
                met.find("lever n_alt source=all-on gate=none reason=quality: not stated\n") != std::string::npos,
            "met preconditions (env set, arg=value form, model array element) admit the lever: got\n" + met);
    const auto wrong_value = plan(ao::Tier::turbo, "decode", only("X_LOW", "0"), {}, q4);
    require(wrong_value.find("lever n_alt source=refused reason=precondition env:X_LOW=1|arg:--qualify-weights unmet\n") !=
                std::string::npos,
            "env:NAME=VALUE needs that value: got\n" + wrong_value);
    const auto no_model = plan(ao::Tier::turbo, "decode", only("X_LOW", "1"), {}, bf16);
    require(no_model.find("lever n_alt source=refused reason=precondition model:weights.format.experts=q4_g64 unmet\n") !=
                std::string::npos,
            "every precondition entry must hold: got\n" + no_model);
    const auto short_prompt = plan(ao::Tier::exact, "prefill:2048", {}, {}, bf16);
    require(short_prompt.find("lever s_pre source=all-on\n") != std::string::npos &&
                short_prompt.find("lever s_dec source=all-on\n") != std::string::npos,
            "a band outside every losing regime applies the lever: got\n" + short_prompt);
    const auto long_prompt = plan(ao::Tier::exact, "prefill:8192", {}, {}, bf16);
    require(long_prompt.find("lever s_pre source=refused reason=measured slower in prefill 8192..32768 tokens (loo 8K)\n") !=
                    std::string::npos &&
                long_prompt.find("lever s_dec source=all-on\n") != std::string::npos,
            "a band inside a losing regime refuses only that lever: got\n" + long_prompt);
    const auto decode = plan(ao::Tier::exact, "decode:4096", only("X_S_DEC", "1"), {}, bf16);
    require(decode.find("lever s_dec source=explicit reason=explicit setting kept\n") != std::string::npos,
            "an explicit setting wins over a losing regime: got\n" + decode);
    bool threw = false;
    try {
        (void)ao::parse_catalogue(R"json([{"row":"r","lever":"bad","class":"exact","gate":"none","engines":["m"],"backends":["hip"],"phase":"any","env":"X","value":"1","needs":["file:/x"]}])json");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "an unknown precondition kind is a catalogue error");
}

// The engine side of the one record: a switch exported by --all-on that the engine does not engage prints its line.
void engaged_tests() {
    put(ao::kLeversEnv, "X_A,STRIX_INK_HIP_SERVED_TALL_CHUNK");
    std::ostringstream out;
    require(ao::engaged("X_A", true, "unused", out) && out.str().empty(), "an engaged lever prints nothing");
    require(!ao::engaged("STRIX_INK_HIP_SERVED_TALL_CHUNK", false, "tier fast route", out) &&
                out.str() == "all-on: lever ink_hip_tall_chunk source=all-on applied=no reason=tier fast route\n",
            "an exported lever the engine does not engage says so under its catalogue name: got\n" + out.str());
    std::ostringstream quiet;
    require(!ao::engaged("X_OTHER", false, "r", quiet) && quiet.str().empty(), "a switch all-on did not export prints nothing");
    put(ao::kLeversEnv, nullptr);
    require(!ao::engaged("X_A", false, "r", quiet) && quiet.str().empty(), "without --all-on nothing is printed");
}

// The review's two cases on the shipped catalogue (P2-2).
void explicit_catalogue_tests(const std::filesystem::path& root) {
    const auto& levers = ao::catalogue();
    const auto run = [&](const std::string& model, ao::Tier tier, EnvValue env, std::string band = "any") {
        const auto desc = description(root, model);
        ao::Query query;
        query.tier = tier;
        query.backend = "hip";
        query.engines = ao::engines_of(&desc);
        query.env_value = std::move(env);
        query.band = ao::parse_band(band);
        query.model_values = [desc](std::string_view key) { return ao::model_values(&desc, key); };
        return ao::resolve(levers, query).text();
    };
    // The low-bit arithmetic levers need resident q4 / q6 weights (the shipped model has none): with the TURBO low-bit
    // row requested they are admitted.
    const auto lowbit = [](std::string key, std::string value) -> EnvValue {
        return [key = std::move(key), value = std::move(value)](std::string_view name) -> std::optional<std::string> {
            if (name == key) return value;
            if (name == "STRIX_INK_LOWBIT_TURBO") return std::string("1");
            return std::nullopt;
        };
    };
    // A1: TURBO --all-on never selects a low-bit lever the shipped weights cannot run (the server refused to start).
    const auto shipped = run("inkling2-mini", ao::Tier::turbo, {});
    for (const char* lever : {"ink_lowbit_rows_select", "ink_lowbit_folded_scale", "ink_lowbit_head_sp4t"})
        require(shipped.find("lever " + std::string(lever) + " source=refused reason=precondition ") != std::string::npos,
                std::string(lever) + " is refused with its precondition on the shipped weights: got\n" + shipped);
    // A3: the measured losers are not all-on levers in their losing regimes, and stay all-on outside them.
    const auto qwen_any = run("qwen3.8-flash-next", ao::Tier::turbo, {});
    require(qwen_any.find("lever qwen_moe_direct_b source=refused reason=measured slower in prefill 8192..32768") !=
                    std::string::npos &&
                qwen_any.find("lever qwen_router_bf16 source=refused reason=measured slower in decode") != std::string::npos,
            "the measured losers are refused for a launch of every phase: got\n" + qwen_any);
    const auto qwen_short = run("qwen3.8-flash-next", ao::Tier::turbo, {}, "prefill:2048");
    require(qwen_short.find("lever qwen_moe_direct_b source=all-on") != std::string::npos &&
                qwen_short.find("lever qwen_router_bf16 source=all-on") != std::string::npos,
            "outside their losing regimes the levers stay all-on: got\n" + qwen_short);
    // A4: direct B is not bitwise at 32.5K: never an EXACT all-on lever.
    const auto qwen_exact = run("qwen3.8-flash-next", ao::Tier::exact, {}, "prefill:2048");
    require(qwen_exact.find("qwen_moe_direct_b") == std::string::npos, "EXACT all-on has no lossy direct B: got\n" + qwen_exact);
    // A2: one-byte route words and the tier split / down LDS epilogue are declared alternatives: TURBO all-on keeps the
    // earlier rows and refuses route I8 (the pair ran a non-finite MoE accumulation).
    require(qwen_short.find("lever qwen_moe_route_i8 source=refused reason=incompatible with qwen_moe_tiers\n") !=
                    std::string::npos &&
                qwen_short.find("lever qwen_moe_tiers source=all-on") != std::string::npos,
            "TURBO all-on never runs route I8 with the tier split: got\n" + qwen_short);
    // A5: the served tall chunk is not bitwise: never an EXACT all-on lever; TURBO takes it (gate pending for FAST).
    const auto ink_exact = run("inkling2-mini", ao::Tier::exact, {});
    const auto ink_turbo = run("inkling2-mini", ao::Tier::turbo, {});
    require(ink_exact.find("ink_hip_tall_chunk") == std::string::npos &&
                ink_turbo.find("lever ink_hip_tall_chunk source=all-on gate=pending") != std::string::npos,
            "the tall chunk is a FAST-class lever: got\n" + ink_exact + ink_turbo);
    const auto qwen = run("qwen3.8-flash-next", ao::Tier::exact, only("STRIX_QPF2K_HC_RECOMPUTE_NORM", "0"));
    require(qwen.find("lever qwen_hc_recompute source=explicit reason=explicit setting kept: off\n") != std::string::npos &&
                qwen.find("lever qwen_hc_inverse_reuse source=refused reason=requires qwen_hc_recompute\n") != std::string::npos,
            "an explicitly disabled requirement refuses its dependant: got\n" + qwen);
    const auto ink = run("inkling2-mini", ao::Tier::turbo, lowbit("STRIX_LOWBIT_FOLDED_SCALE", "f16_folded_scale"));
    require(ink.find("lever ink_lowbit_rows_select source=refused reason=incompatible with ink_lowbit_folded_scale\n") !=
                    std::string::npos &&
                ink.find("lever ink_lowbit_folded_scale source=explicit reason=explicit setting kept\n") != std::string::npos,
            "an explicit later peer refuses the earlier incompatible lever: got\n" + ink);
    // An explicit value equal to the switch's incumbent (the default low-bit scale named by its own word) is an
    // explicit off: it neither runs the lever nor refuses its incompatible peer.
    const auto incumbent = run("inkling2-mini", ao::Tier::turbo, lowbit("STRIX_LOWBIT_FOLDED_SCALE", "f16_group_scale"));
    require(incumbent.find("lever ink_lowbit_folded_scale source=explicit reason=explicit setting kept: off\n") !=
                    std::string::npos &&
                incumbent.find("lever ink_lowbit_rows_select source=all-on") != std::string::npos,
            "an explicit incumbent value is off and leaves the incompatible peer on: got\n" + incumbent);
    // An ungated TURBO lever says so on its all-on line.
    const auto turbo = run("inkling2-mini", ao::Tier::turbo, only("STRIX_LOWBIT_PACK", "/packs/strix.sp4t"));
    require(turbo.find("lever ink_lowbit_head_sp4t source=all-on gate=none") != std::string::npos,
            "an ungated TURBO lever prints gate=none: got\n" + turbo);
}

void binding_tests(const std::filesystem::path& root) {
    const auto desc = description(root, "qwen3.8-flash-next");
    put("STRIX_QPF2K_HC_RECOMPUTE_NORM", nullptr);
    put("STRIX_QPF2K_MOE_ROUTE_BF16", nullptr);
    put("STRIX_QWEN_ROUTER_BF16", "0");
    put(ao::kAppliedEnv, nullptr);
    std::vector<std::string> args;
    std::ostringstream out;
    {
        // An exact lever (HC normalized recompute) is exported under FAST; a pending-gate lossy one is not.
        const auto scope = ao::begin(&desc, Name::fast, "hip", "", out, true, args);
        require(std::string(std::getenv("STRIX_QPF2K_HC_RECOMPUTE_NORM") ? std::getenv("STRIX_QPF2K_HC_RECOMPUTE_NORM") : "") == "1",
                "an exact lever is exported under fast");
        require(std::getenv("STRIX_QPF2K_MOE_ROUTE_BF16") == nullptr, "a pending-gate lossy lever is not exported under fast");
        require(std::string(std::getenv("STRIX_QWEN_ROUTER_BF16")) == "0", "an explicit value is not overwritten");
        require(out.str().starts_with("all-on: tier=fast backend=hip levers="), "header line");
        require(out.str().find("lever qwen_hc_recompute source=all-on\n") != std::string::npos, "printed lever line");
        const char* exported = std::getenv(ao::kLeversEnv);
        require(exported != nullptr && std::string(exported).find("STRIX_QPF2K_HC_RECOMPUTE_NORM") != std::string::npos &&
                    std::string(exported).find("STRIX_QWEN_ROUTER_BF16") == std::string::npos,
                "the exported switches are listed for the engines (an explicit one is not)");
        require(out.str().find("lever qwen_router_bf16 source=explicit reason=explicit setting kept: off\n") != std::string::npos,
                "explicit lever line");
        std::ostringstream again;
        const auto second = ao::begin(&desc, Name::fast, "hip", "", again, true, args);
        require(again.str().empty(), "a second shell prints and applies nothing");
    }
    require(std::getenv("STRIX_QPF2K_HC_RECOMPUTE_NORM") == nullptr && std::getenv(ao::kLeversEnv) == nullptr,
            "the scope restores the environment");
    // Dry run: arguments and lines, no exports.
    std::ostringstream dry;
    const auto desc27 = description(root, "qwen3.8-27b");
    std::vector<std::string> dry_args;
    {
        const auto scope = ao::begin(&desc27, Name::fast, "cuda", "", dry, false, dry_args);
        require(std::getenv(ao::kAppliedEnv) == nullptr, "dry run exports nothing");
        require(std::ranges::find(dry_args, "--prelaunch") != dry_args.end(), "dry run lists the lever arguments");
    }
    put("STRIX_QWEN_ROUTER_BF16", nullptr);
    put(ao::kEnv, "2");
    bool refused = false;
    try { (void)ao::env_requested(); } catch (const std::invalid_argument&) { refused = true; }
    require(refused, "STRIX_ALL_ON accepts 0 or 1 only");
    put(ao::kEnv, "1");
    require(ao::env_requested(), "STRIX_ALL_ON=1 requests it");
    put(ao::kEnv, nullptr);
    require(!ao::env_requested(), "unset means off");
}

void cli_tests() {
    const auto request = strix::cli::parse({"run", "qwen3.8-flash-next", "--profile", "fast", "--all-on", "--all-on-band", "prefill:8192"});
    require(request.all_on && request.all_on_band == "prefill:8192", "strix run parses --all-on and --all-on-band");
    require(!strix::cli::parse({"run", "qwen3.8-flash-next"}).all_on, "off unless requested");
}

void table(const std::filesystem::path& root) {
    const auto& levers = ao::catalogue();
    for (const auto& name : models(root)) {
        const auto desc = description(root, name);
        for (const char* backend : {"hip", "cuda"}) {
            std::cout << "\n### " << name << " / " << backend << "\n\n| tier | lever | source | reason |\n|---|---|---|---|\n";
            for (const auto tier : {ao::Tier::exact, ao::Tier::fast, ao::Tier::turbo}) {
                ao::Query query;
                query.tier = tier;
                query.backend = backend;
                query.engines = ao::engines_of(&desc);
                for (const auto& line : ao::resolve(levers, query).lines)
                    std::cout << "| " << (tier == ao::Tier::exact ? "exact" : tier == ao::Tier::fast ? "fast" : "turbo") << " | "
                              << line.lever << " | " << line.source << line.marks << " | " << line.reason << " |\n";
            }
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: strix-all-on-tests ROOT [--table]\n";
        return 2;
    }
    const std::filesystem::path root = argv[1];
    try {
        if (argc > 2 && std::string(argv[2]) == "--table") {
            table(root);
            return 0;
        }
        fixture_tests();
        needs_slower_tests();
        engaged_tests();
        catalogue_tests(root);
        explicit_catalogue_tests(root);
        binding_tests(root);
        cli_tests();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: unexpected exception: " << e.what() << "\n";
        return 1;
    }
    if (failures == 0) std::cout << "all-on tests passed\n";
    return failures == 0 ? 0 : 1;
}
