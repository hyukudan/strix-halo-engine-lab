// Host test of the Qwen 27B dense HIP product word (include/strix/profiles/qwen_dense_hip.hpp): identity of the
// default column, pending FAST cell, explicit options win, --allow-pending applies, exact-window options.
#include "strix/profiles/qwen_dense_hip.hpp"

#include <cstdio>
#include <string>
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
        require(fast.args == user, "fast without allow-pending adds no option");
        require(fast.resolution.entries.size() == 1U && fast.resolution.entries[0].status == sp::Status::pending &&
                    fast.resolution.entries[0].value == exact.resolution.entries[0].value,
                "fast: the wide cell is pending, effective value equals exact");
        require(fast.resolution.entries[0].target == "64-row FAST wide windows (dense-q8a8-f16-v1)", "pending target reported");
        require(!sp::applied(fast.resolution.entries[0].status), "fast applies no lever without allow-pending");
        // allow-pending (flag and env) applies the runnable pending cell, in front of the user's options.
        Args allow{"--profile", "fast", "--allow-pending"};
        const auto applied = run(app, allow);
        require(applied.args == Args{opt, "64"}, "fast --allow-pending adds the 64-row option in front");
        require(applied.resolution.entries[0].status == sp::Status::unqualified, "allow-pending is stamped unqualified");
        const sp::Lookup allow_env = [](const char* key) -> const char* {
            return std::string(key) == std::string(hp::kAllowEnv) ? "1" : nullptr;
        };
        require(run(app, with_fast, allow_env).args.size() == user.size() + 2U, "allow-pending env applies the cell");
        // explicit option wins over the word and stays in the user's arguments.
        Args explicit_args{"--profile", "fast", "--allow-pending", opt, "16", "--gen", "8"};
        const auto won = run(app, explicit_args);
        require(won.args == Args{opt, "16", "--gen", "8"}, "explicit option wins, no duplicate from the profile");
        require(won.resolution.entries[0].status == sp::Status::off || won.resolution.entries[0].status == sp::Status::explicit_cli,
                "explicit option recorded as explicit");
        // exact-window options keep their meaning: the profile adds no window size.
        const char* exact_opt = app == hp::App::decode ? "--wide-exact" : "--prefill-exact";
        const auto ex = run(app, Args{"--profile", "fast", "--allow-pending", exact_opt, "64"});
        require(ex.args == Args{exact_opt, "64"}, "exact-window option suppresses the profile window");
        // another spelling of --profile (not a word) stays for the app.
        require(run(app, Args{"--profile=/tmp/x"}).args == Args{"--profile=/tmp/x"}, "non-word --profile untouched");
    }
    // the decode app does not see the spec spelling and vice versa
    require(run(hp::App::decode, Args{"--prefill-rows", "8"}).resolution.entries[0].status == sp::Status::builtin,
            "decode engine ignores --prefill-rows");
    bool refused = false;
    try { run(hp::App::spec, Args{"--profile", "turbo"}); } catch (const std::exception&) { refused = true; }
    require(refused, "turbo is not declared for this engine");
    if (failures == 0) std::printf("qwen dense hip profile: PASS\n");
    return failures == 0 ? 0 : 1;
}
