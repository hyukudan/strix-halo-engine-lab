// Server / CLI configuration parity gate (strix/serve_cli_parity.hpp): for every family built here, the same plain
// request is resolved through the CLI entry point and through the HTTP server entry point of the family's ONE resolver;
// the test fails on any difference that is not listed below with the reason serving requires it, and on any listed
// difference that no longer exists. `--print` prints both resolved configurations side by side (the report tools use it).
#include "strix/serve_cli_parity.hpp"

#include <cstring>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#if defined(STRIX_PARITY_GLM)
#include "glm_rank_defaults.hpp"
#endif
#if defined(STRIX_PARITY_MINIMAX)
#include "minimax/engine/mm_defaults.hpp"
#endif
#if defined(STRIX_PARITY_DS41)
#include "strix/deepseek41_pair_defaults.hpp"
#endif

namespace {

namespace sp = strix::parity;
int g_failures = 0;
bool g_print = false;

void check(bool ok, const std::string& what) {
    if (ok) return;
    ++g_failures;
    std::cerr << "FAIL: " << what << "\n";
}

void gate(const std::string& family, const sp::Config& cli, const sp::Config& server, std::span<const sp::Allowed> allowed) {
    const auto report = sp::compare(cli, server, allowed);
    if (g_print) {
        std::cout << "== " << family << ": " << cli.size() << " cli keys, " << server.size() << " server keys, "
                  << report.explained.size() << " serving-only, " << report.unexplained.size() << " unexplained\n";
        for (const auto& [k, v] : cli) {
            const auto it = server.find(k);
            std::cout << family << " " << k << " cli=" << v << " server=" << (it == server.end() ? std::string(sp::kAbsent) : it->second) << "\n";
        }
        for (const auto& [k, v] : server)
            if (!cli.contains(k)) std::cout << family << " " << k << " cli=" << sp::kAbsent << " server=" << v << "\n";
    }
    if (!report.ok() || g_print) std::cerr << sp::text(family, report);
    check(report.ok(), family + ": server and CLI resolve the same request to different values");
}

void helper_tests() {
    const sp::Config a{{"x", "1"}, {"y", "2"}, {"only_cli", "c"}}, b{{"x", "1"}, {"y", "3"}, {"only_server", "s"}};
    const sp::Allowed none[] = {{"unused", "never differs"}};
    auto r = sp::compare(a, a, {});
    check(r.ok() && r.explained.empty(), "identical configurations pass");
    r = sp::compare(a, b, {});
    check(!r.ok() && r.unexplained.size() == 3U, "three unexplained differences");
    const sp::Allowed all[] = {{"y", "serving reason"}, {"only_*", "entry point only"}};
    r = sp::compare(a, b, all);
    check(r.ok() && r.explained.size() == 3U && r.explained[0].reason == "entry point only", "allowances with reasons");
    r = sp::compare(a, a, none);
    check(!r.ok() && r.stale.size() == 1U, "an allowance that no longer differs fails");
    const sp::Allowed no_reason[] = {{"y", ""}};
    r = sp::compare(a, b, no_reason);
    check(!r.ok(), "an allowance without a reason does not count");
    const auto l = sp::from_lines("a=1\nb=2 (auto)\r\nnote\n\n");
    check(l.size() == 3U && l.at("a") == "1" && l.at("b") == "2 (auto)" && l.at("line.0") == "note", "from_lines");
    const std::vector<std::string> args{"--rank", "0", "--tp", "--rows", "8", "16"};
    const auto g = sp::from_args(args);
    check(g.size() == 3U && g.at("arg:--rank") == "0" && g.at("arg:--tp").empty() && g.at("arg:--rows") == "8 16", "from_args");
    check(sp::text("f", sp::compare(a, b, {})).find("f UNEXPLAINED y cli=2 server=3") != std::string::npos, "report text");
}

#if defined(STRIX_PARITY_GLM)
// GLM: strix-glm5-rank and strix-glm5-serve call rd::scan / rd::resolve; the server forwards its rank options through
// rd::serve_request_args and applies rd::serve_adjust (apps/glm_serve_main.cpp).
void glm() {
    namespace rd = strix::glm::rank_defaults;
    const rd::EnvGet none = [](const char*) -> const char* { return nullptr; };
    const std::vector<std::string> request{"--world", "2"};
    auto cli_args = request;
    cli_args.insert(cli_args.end(), {"--decode", "64"});  // the CLI's serving-shaped request: prompt + N decoded tokens
    const auto cli = rd::resolve(rd::scan(cli_args), none);
    auto server = rd::resolve(rd::scan(rd::serve_request_args(request, true)), none);
    rd::serve_adjust(server);
    static constexpr sp::Allowed allowed[] = {
        // OPEN (not yet removed): the CLI sizes the exchange with its startup admission (8192 rows when both ranks fit),
        // the server starts from the 4096-row fallback. Named here so it cannot grow silently.
        {"effective max-rows", "serve has no startup row admission: 4096-row exchange fallback (auto-4096-serve)"},
        {"effective max-rows-admission", "the startup row admission line exists on the CLI only"},
    };
    // "value (source)": the gate compares the value; the source label names the entry point's path to it.
    const auto values = [](sp::Config c) {
        for (auto& [k, v] : c)
            if (const auto at = v.rfind(" ("); at != std::string::npos && !v.empty() && v.back() == ')') v.erase(at);
        return c;
    };
    gate("glm", values(sp::from_lines(rd::print(cli))), values(sp::from_lines(rd::print(server))), allowed);
}
#endif

#if defined(STRIX_PARITY_MINIMAX)
// MiniMax: strix-minimax-rank and strix-minimax-serve call defaults::print_or_apply(app, ...) = parse_cli + effective_config.
void minimax() {
    namespace md = strix::minimax::defaults;
    const md::Lookup none = [](const char*) -> const char* { return nullptr; };
    const auto cli = sp::from_lines(md::effective_config(md::parse_cli("rank", {}), none));
    const auto server = sp::from_lines(md::effective_config(md::parse_cli("serve", {}), none));
    static constexpr sp::Allowed allowed[] = {
        {"app", "the entry point's own name"},
        {"session_reason", "text that quotes the app and transport names"},
        // OPEN (not yet removed): the server defaults to the device exchange, the CLI negotiates it.
        {"transport", "serve default usb4gpu (device exchange), rank default usb4-auto (negotiated)"},
        {"serve.*", "scheduler / speculation policy between concurrent requests; no CLI counterpart"},
    };
    gate("minimax", cli, server, allowed);
}
#endif

#if defined(STRIX_PARITY_DS41)
// DeepSeek41 pair: every app calls pair_defaults::resolve(App, ...) (Effective in the process).
void ds41() {
    namespace pd = strix::deepseek41::pair_defaults;
    const std::string model = "/md/ds41.gguf";
    const std::set<std::string> files{model}, dirs{"/md"};
    pd::Probe probe;
    probe.available_memory = [] { return std::uint64_t{64} << 30U; };
    probe.writable_space = [](const std::string&) { return std::uint64_t{512} << 30U; };
    probe.file = [&](const std::string& f) { return files.count(f) != 0; };
    probe.dir = [&](const std::string& d) { return dirs.count(d) != 0; };
    probe.list = [](const std::string&) { return std::vector<std::string>{}; };
    const auto config = [&](pd::App app) {
        const auto r = pd::resolve(app, {model, "--rank", "0", "--transport", "usb4auto"},
                                   [](const char*) -> const char* { return nullptr; }, probe, pd::Build{true});
        auto c = sp::from_args(std::span<const std::string>(r.args).subspan(1));
        for (const auto& [k, v] : r.set_env) c["env:" + k] = v;
        for (const auto& k : r.unset_env) c["env:" + k] = "<unset>";
        c["session.enabled"] = r.session.enabled ? "1" : "0";
        c["session.disk"] = r.session.disk ? "1" : "0";
        if (r.profile) c["profile"] = std::string(strix::profile::name(r.profile->profile));
        return c;
    };
    static constexpr sp::Allowed allowed[] = {
        {"arg:--engram-direct", "the resolver adds the flag for the CLI; the server has it as its built-in default"},
        {"arg:--tp-attention", "the resolver adds the flag for the CLI; the server has it as its built-in default"},
        {"arg:--peer-timeout", "serving waits for the peer rank; the CLI starts both ranks together"},
        {"arg:--slots", "request slots exist on the server only"},
        {"arg:--prefill-path", "the server prefills prompts on the device batch path; the CLI's prompt goes through its decode windows"},
        {"arg:--prefill-rows", "row budget of the server's device prefill path (auto)"},
    };
    gate("ds41", config(pd::App::rank), config(pd::App::serve), allowed);
}
#endif

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) g_print |= std::strcmp(argv[i], "--print") == 0;
    try {
        helper_tests();
#if defined(STRIX_PARITY_GLM)
        glm();
#endif
#if defined(STRIX_PARITY_MINIMAX)
        minimax();
#endif
#if defined(STRIX_PARITY_DS41)
        ds41();
#endif
    } catch (const std::exception& e) {
        std::cerr << "FAIL: exception: " << e.what() << "\n";
        return 1;
    }
    if (g_failures) std::cerr << g_failures << " failure(s)\n";
    else std::cout << "serve/cli parity: ok\n";
    return g_failures ? 1 : 0;
}
