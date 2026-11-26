// Host checks of the STRIX_QWEN_ALLON composition profile (include/strix/qwen_allon.hpp): no device, no model.
#include "strix/qpf2k_defaults.hpp"
#include "strix/qwen_allon.hpp"

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <string_view>

namespace {
int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}
using Env = std::map<std::string, std::string>;
const char* find(const Env& env, const char* name) {
    const auto it = env.find(name);
    return it == env.end() ? nullptr : it->second.c_str();
}
unsigned run(Env& env) {
    return strix::qwen_allon::apply([&](const char* name) { return find(env, name); },
        [&](const char* name, const char* value) { return env.emplace(name, value).second; });
}
unsigned count(bool fast_only) {
    unsigned n = 0;
    for (const auto& lever : strix::qwen_allon::levers) n += !fast_only || lever.fast ? 1U : 0U;
    return n;
}
// Model the production ordering: ALLON runs before main; readers first run
// after CLI row admission. Separate instantiations model separate processes.
template<unsigned Rows>
struct Startup {
    Env env{{strix::qwen_allon::master_env, "1"}};
    Startup() { run(env); }
    bool router() const {
        static const bool on = std::string_view(find(env, "STRIX_QWEN_ROUTER_BF16")) == "1";
        return on;
    }
    bool direct_b() const {
        static const bool on = [] (const char* v) {
            return v && v[0] == '1' && v[1] == '\0';
        }(find(env, "STRIX_QPF2K_MOE_DIRECT_B"));
        return on;
    }
    bool hc() const {
        static const bool on = [] (const char* v) { return v && v[0] == '1'; }
            (find(env, "STRIX_QPF2K_HC_RECOMPUTE_NORM"));
        return on;
    }
};
Startup<262144U> long_startup;
Startup<32768U> short_startup;
}  // namespace

int main() {
    using namespace strix::qwen_allon;
    const auto get_of = [](const Env& env) {
        return [&env](const char* name) { return find(env, name); };
    };
    {
        const auto saved = injected;
        const auto check_startup = [&](auto& startup, unsigned rows) {
            expect(injected_by_profile("STRIX_QWEN_ROUTER_BF16") &&
                injected_by_profile("STRIX_QPF2K_MOE_DIRECT_B") &&
                injected_by_profile("STRIX_QPF2K_HC_RECOMPUTE_NORM"),
                "static-init apply records all three levers");
            gate_long_prompt(rows, get_of(startup.env),
                [&](const char* name, const char* value) { startup.env[name] = value; });
            // Eager exec inherits values, but static apply cannot reclaim them.
            injected.fill(false);
            expect(run(startup.env) == 0U, "eager restart inherits existing levers");
            const bool expected = rows < long_prompt_rows;
            expect(startup.router() == expected, "router copy selection follows early gate");
            expect(startup.direct_b() == expected, "DIRECT_B first cached read follows early gate");
            expect(startup.hc() == expected, "HC first cached read follows early gate");
            injected = saved;
        };
        check_startup(long_startup, 262144U);
        check_startup(short_startup, 32768U);
    }
    // Gate only profile-injected levers, before resource planning.
    {
        const auto saved = injected;
        for (const auto rows : {32768U, 131072U, 262142U, 262143U, 262144U}) {
            injected.fill(false);
            Env automatic{{master_env, "1"}};
            run(automatic);
            const auto gated = gate_long_prompt(rows, get_of(automatic),
                [&](const char* name, const char* value) { automatic[name] = value; });
            expect(gated == (rows >= long_prompt_rows ? 3U : 0U), "long prompt gate boundary");
            for (const char* name : {"STRIX_QPF2K_MOE_DIRECT_B",
                    "STRIX_QPF2K_HC_RECOMPUTE_NORM", "STRIX_QWEN_ROUTER_BF16"})
                expect(automatic.at(name) == (rows >= long_prompt_rows ? "0" : "1"),
                    "short context retained; measured long regime excluded");
            expect(automatic.at("STRIX_QWEN_DECODE_FUSE") == "1", "unrelated lever retained");
            expect(gate_long_prompt(rows, get_of(automatic),
                [&](const char* name, const char* value) { automatic[name] = value; }) == 0U,
                "gate is idempotent");
        }
        for (const char* value : {"0", "1"}) {
            injected.fill(false);
            Env explicit_env{{master_env, "fast"}, {"STRIX_QPF2K_MOE_DIRECT_B", value},
                {"STRIX_QPF2K_HC_RECOMPUTE_NORM", value}, {"STRIX_QWEN_ROUTER_BF16", value}};
            run(explicit_env);
            expect(gate_long_prompt(262144U, get_of(explicit_env),
                [&](const char* name, const char* v) { explicit_env[name] = v; }) == 0U,
                "explicit ON/OFF overrides ALLON regime selection");
        }
        injected = saved;
    }

    // Table integrity: one row per variable, no row duplicates a shipped default, exact rows come first.
    std::set<std::string_view> seen;
    bool fast_seen = false, order = true;
    for (const auto& lever : levers) {
        expect(seen.insert(lever.env).second, "duplicate variable in the lever table");
        expect(lever.name && *lever.name && lever.on && *lever.on && lever.receipt && *lever.receipt, "empty table cell");
        if (lever.fast) fast_seen = true;
        else if (fast_seen) order = false;
        for (const auto& item : strix::qpf2k::defaults)
            expect(std::string_view(item.name) != lever.env, "an opt-in row is already a shipped default");
    }
    expect(order, "exact levers precede FAST levers");
    expect(count(false) > count(true) && count(true) >= 5U, "table has exact and FAST rows");

    // Modes.
    Env env;
    expect(mode(get_of(env)) == Mode::off, "unset is off");
    for (const char* text : {"", "0"}) { env[master_env] = text; expect(mode(get_of(env)) == Mode::off, "empty and 0 are off"); }
    for (const char* text : {"1", "exact"}) { env[master_env] = text; expect(mode(get_of(env)) == Mode::exact, "1 and exact are exact"); }
    env[master_env] = "fast";
    expect(mode(get_of(env)) == Mode::fast, "fast");
    for (const char* text : {"2", "on", "FAST", "fast "}) { env[master_env] = text; expect(mode(get_of(env)) == Mode::invalid, "other values are invalid"); }

    // Off and invalid change nothing.
    const char* const off_cases[] = {nullptr, "0", "bogus"};
    for (const char* text : off_cases) {
        Env e;
        if (text) e[master_env] = text;
        const auto before = e.size();
        expect(run(e) == 0U && e.size() == before, "off/invalid sets nothing");
        expect(json(get_of(e)).empty() == (!text || std::string_view(text) == "0"), "json only when the switch is set");
    }

    // Exact: every exact lever and no FAST lever.
    Env exact{{master_env, "1"}};
    expect(run(exact) == count(false) - count(true), "exact sets every exact lever");
    for (const auto& lever : levers) {
        const char* value = find(exact, lever.env);
        expect(lever.fast ? value == nullptr : (value && std::string_view(value) == lever.on), "exact set membership");
    }
    // FAST: all levers.
    Env fast{{master_env, "fast"}};
    expect(run(fast) == count(false), "fast sets every lever");
    for (const auto& lever : levers) {
        const char* value = find(fast, lever.env);
        expect(value && std::string_view(value) == lever.on, "fast value");
    }
    expect(run(fast) == 0U, "a second pass sets nothing (idempotent across an image restart)");

    // An explicit value wins, including an explicit 0, and the other levers are still set.
    Env explicit_env{{master_env, "fast"}, {"STRIX_QPF2K_PLE_TILE", "0"}, {"STRIX_QPF2K_QSA_BUDGET", "2048"}};
    expect(run(explicit_env) == count(false) - 2U, "two explicit values are left alone");
    expect(std::string_view(find(explicit_env, "STRIX_QPF2K_PLE_TILE")) == "0", "explicit 0 kept");
    expect(std::string_view(find(explicit_env, "STRIX_QPF2K_QSA_BUDGET")) == "2048", "explicit budget kept");
    Env empty_override{{master_env, "fast"}, {"STRIX_QPF2K_PLE_TILE", ""}};
    unsigned attempted = 0;
    apply(get_of(empty_override), [&](const char* key, const char* value) {
        if (std::string_view(key) == "STRIX_QPF2K_PLE_TILE") ++attempted;
        empty_override[key] = value;
        return true;
    });
    expect(attempted == 0 && empty_override.at("STRIX_QPF2K_PLE_TILE").empty(),
           "an explicit empty value vetoes composition");

    // The baseline vetoes the profile; a baseline of 0 does not.
    Env baseline{{master_env, "fast"}, {"STRIX_BASELINE", "1"}};
    expect(run(baseline) == 0U, "baseline vetoes");
    expect(json(get_of(baseline)).find("\"vetoed\":\"STRIX_BASELINE\"") != std::string::npos, "veto is reported");
    Env not_baseline{{master_env, "1"}, {"STRIX_BASELINE", "0"}};
    expect(run(not_baseline) > 0U, "STRIX_BASELINE=0 is not a veto");

    // Report shape.
    const auto report = json(get_of(fast));
    expect(report.rfind("{\"mode\":\"fast\"", 0) == 0 && report.back() == '}', "json head and tail");
    expect(report.find("\"class\":\"fast\"") != std::string::npos && report.find("\"class\":\"exact\"") != std::string::npos,
        "json lists both classes in fast mode");
    expect(json(get_of(exact)).find("\"class\":\"fast\"") == std::string::npos, "json lists no FAST lever in exact mode");

    std::cout << "qwen allon: " << failures << " failures, " << levers.size() << " levers\n";
    return failures ? 1 : 0;
}
