// Host test of the Qwen 27B dense HIP product word (include/strix/profiles/qwen_dense_hip.hpp): identity of the
// default column, qualified FAST window + layer table, TURBO nests FAST, explicit options win, exact-window options.
#include "strix/profiles/qwen_dense_hip.hpp"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace hp = strix::qwen35::hip_profile;
namespace sp = strix::profile;
using Args = std::vector<std::string>;
int failures = 0;
void require(bool ok, const char* what) {
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "FAIL %s\n", what);
}
const sp::Lookup no_env = [](const char*) -> const char* { return nullptr; };
hp::Result run(hp::App app, const Args& args, const sp::Lookup& env = no_env) {
    return hp::resolve_args(hp::engine(app), args, env);
}
const char* window_lever(hp::App app) { return app == hp::App::decode ? "wide_rows" : "prefill_rows"; }
// The app's window cell and the shared A4W4 layer cell (both engines resolve exactly these two rows).
const sp::Entry& win(const hp::Result& r, hp::App app) { return *r.resolution.find(window_lever(app)); }
const sp::Entry& a4(const hp::Result& r) { return *r.resolution.find("a4w4_layers"); }
}  // namespace

int main() {
    for (const auto app : {hp::App::decode, hp::App::spec}) {
        const char* opt = app == hp::App::decode ? "--wide" : "--prefill-rows";
        const Args user{"--model", "m.gguf", "--gen", "8"};
        Args with_word{"--profile", "exact"};
        with_word.insert(with_word.end(), user.begin(), user.end());
        Args with_default{"--profile=default"};
        with_default.insert(with_default.end(), user.begin(), user.end());
        Args with_fast{"--profile", "fast"};
        with_fast.insert(with_fast.end(), user.begin(), user.end());
        const auto none = run(app, user);
        const auto exact = run(app, with_word);
        const auto dflt = run(app, with_default);
        const auto fast = run(app, with_fast);
        require(none.args == user, "no word: arguments untouched");
        require(exact.args == user && dflt.args == user, "exact / default: arguments untouched (today's behaviour)");
        require(none.resolution.fingerprint() == exact.resolution.fingerprint() &&
                    exact.resolution.fingerprint() == dflt.resolution.fingerprint(),
                "no word == exact == default fingerprint");
        require(exact.resolution.profile == sp::Name::standard && dflt.resolution.profile == sp::Name::standard,
                "exact and default are the default column");
        Args fast_args{opt, "512"};
        fast_args.insert(fast_args.end(), user.begin(), user.end());
        require(fast.args == fast_args, "fast puts the qualified 512-row window option in front of the user's arguments");
        require(fast.resolution.entries.size() == 2U && sp::applied(win(fast, app).status) &&
                    win(fast, app).value == "512" && win(fast, app).ladder.find("incumbent") != std::string::npos,
                "fast applies the gated 512-row wide window (grant with its gate and timing receipts)");
        require(!sp::applied(win(exact, app).status) && win(exact, app).value != "512",
                "exact keeps the engine's 64-row EXACT windows");
        // The qualified layer table ran before the ladder rule: its incumbent grant keeps it in every regime.
        require(sp::applied(a4(fast).status) && a4(fast).value == "48-63" &&
                    a4(fast).ladder.find("incumbent") != std::string::npos,
                "fast keeps the qualified A4W4 layer table (incumbent grant)");
        require(!sp::applied(a4(exact).status) && !sp::applied(a4(dflt).status) && !sp::applied(a4(none).status),
                "exact / default / no word leave the A4W4 layer table alone");
        std::vector<std::pair<std::string, std::string>> put;
        sp::apply(fast.resolution, [&](const char* k, const char* v) { put.emplace_back(k, v); return true; });
        require(put == std::vector<std::pair<std::string, std::string>>{{"STRIX_DENSE_A4W4_LAYERS", "48-63"}},
                "fast exports exactly STRIX_DENSE_A4W4_LAYERS=48-63");
        put.clear();
        sp::LadderRow timing{"a4w4_layers", "48-63", sp::Name::fast, sp::Phase::prefill, "fixture",
                             8192, 8192, 1, 1, 1.04, 0.01, "fixture"};
        auto engine = hp::engine(app);
        // The row is measured over the execution that runs under the cell (LadderArms::over).
        const auto granted = hp::resolve_args(engine, with_fast, no_env, {sp::Phase::prefill, 8192, 1, "fixture", "fixture-exec"});
        const auto* granted_entry = granted.resolution.find("a4w4_layers");
        const std::string reference = granted_entry ? granted_entry->ladder_reference : std::string();
        require(reference.starts_with("ladder-arm/v4;"), "Qwen dense HIP reports the reference of the layer table");
        timing.over = reference;
        // The timing row replaces the layer table's grant only; the window cell keeps its own grant.
        std::vector<sp::LadderRow> rows;
        for (const auto& row : hp::engine(app).ladder)
            if (row.lever != "a4w4_layers") rows.push_back(row);
        rows.push_back(timing);
        engine.ladder = rows;
        const auto measured = hp::resolve_args(engine, with_fast, no_env, {sp::Phase::prefill, 8192, 1, "fixture", "fixture-exec"});
        sp::apply(measured.resolution, [&](const char* k, const char* v) { put.emplace_back(k, v); return true; });
        require(put == std::vector<std::pair<std::string, std::string>>{{"STRIX_DENSE_A4W4_LAYERS", "48-63"}},
                "a matching measured run exports the FAST layer table");
        const sp::Lookup user_env = [](const char* key) -> const char* {
            return std::string(key) == "STRIX_DENSE_A4W4_LAYERS" ? "none" : nullptr;
        };
        const auto off = run(app, with_fast, user_env);
        require(a4(off).value == "none" && !sp::applied(a4(off).status), "an explicit STRIX_DENSE_A4W4_LAYERS wins over fast");
        // allow-pending (flag and env, benchmarks only) applies the runnable pending cell in an unmeasured regime, in front
        // of the user's options, stamped unqualified (main's contract; the route line says it is a benchmark arm).
        Args allow{"--profile", "fast", "--allow-pending"};
        const auto applied = run(app, allow);
        require(applied.args == Args{opt, "512"}, "fast --allow-pending: the same qualified 512-row option");
        require(win(applied, app).status == win(fast, app).status, "allow-pending does not restamp a qualified cell");
        const sp::Lookup allow_env = [](const char* key) -> const char* {
            return std::string(key) == std::string(hp::kAllowEnv) ? "1" : nullptr;
        };
        require(run(app, with_fast, allow_env).args == fast_args, "allow-pending env: the same FAST arguments");
        // explicit option wins over the word and stays in the user's arguments.
        Args explicit_args{"--profile", "fast", "--allow-pending", opt, "16", "--gen", "8"};
        const auto won = run(app, explicit_args);
        require(won.args == Args{opt, "16", "--gen", "8"}, "explicit option wins, no duplicate from the profile");
        require(win(won, app).status == sp::Status::off || win(won, app).status == sp::Status::explicit_cli,
                "explicit option recorded as explicit");
        // exact-window options keep their meaning: the profile adds no window size.
        const char* exact_opt = app == hp::App::decode ? "--wide-exact" : "--prefill-exact";
        const auto ex = run(app, Args{"--profile", "fast", "--allow-pending", exact_opt, "64"});
        require(ex.args == Args{exact_opt, "64"}, "exact-window option suppresses the profile window");
        // TURBO nests FAST on HIP: no HIP TURBO row, every TURBO cell is the FAST cell.
        Args with_turbo{"--profile", "turbo"};
        with_turbo.insert(with_turbo.end(), user.begin(), user.end());
        const auto turbo = run(app, with_turbo);
        require(turbo.resolution.profile == sp::Name::turbo && turbo.args == fast.args &&
                    win(turbo, app).value == "512" && a4(turbo).value == "48-63" && sp::applied(a4(turbo).status),
                "turbo is accepted and runs exactly the FAST levers");
        // another spelling of --profile (not a word) stays for the app.
        require(run(app, Args{"--profile=/tmp/x"}).args == Args{"--profile=/tmp/x"}, "non-word --profile untouched");
    }
    // the decode app does not see the spec spelling and vice versa
    require(win(run(hp::App::decode, Args{"--prefill-rows", "8"}), hp::App::decode).status == sp::Status::builtin,
            "decode engine ignores --prefill-rows");
    if (failures == 0) std::printf("qwen dense hip profile: PASS\n");
    return failures == 0 ? 0 : 1;
}
