// Server / CLI configuration parity gate (strix/serve_cli_parity.hpp): for every family built here, the same plain
// request is resolved through the CLI entry point and through the HTTP server entry point of the family's ONE resolver;
// the test fails on any difference that is not listed below with the reason serving requires it, and on any listed
// difference that no longer exists. `--print` prints both resolved configurations side by side (the report tools use it).
#include "strix/serve_cli_parity.hpp"
#include "strix/env_arms.hpp"

#include <span>

#include <cstring>
#include <functional>
#include <iostream>
#include <set>
#include <optional>
#include <string>
#include <vector>

#if defined(STRIX_PARITY_GLM)
#include "glm/load/glm_rank_defaults.hpp"
#endif
#if defined(STRIX_PARITY_MINIMAX)
#include "minimax/engine/mm_defaults.hpp"
#include "minimax/engine/mm_profile_table.hpp"
#endif
#if defined(STRIX_PARITY_DS41)
#include "strix/deepseek41_pair_defaults.hpp"
#include "strix/deepseek41_prefill.hpp"
#include "strix/ds41_prefill.hpp"
#include "strix/deepseek41_prefix_cache.hpp"
#endif
#if defined(STRIX_PARITY_GEMMA4)
#include "gemma4/engine/runtime.hpp"
#include <chrono>
#include <fstream>
#endif

namespace {

namespace sp = strix::parity;
int g_failures = 0;
bool g_print = false;
bool g_require_output_parity = false;
bool g_strict_gemma = false;

void check(bool ok, const std::string& what) {
    if (ok) return;
    ++g_failures;
    std::cerr << "FAIL: " << what << "\n";
}

void gate(const std::string& family, const sp::Config& cli, const sp::Config& server, std::span<const sp::Allowed> allowed) {
    const auto report = sp::compare(cli, server, allowed, g_require_output_parity);
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
    r = sp::compare(a, b, all, true);
    check(!r.ok() && r.unexplained.size() == 3U && r.stale.empty(), "strict parity refuses historical allowances");
    const sp::Config meta_cli{{"entry", "cli"}}, meta_server{{"entry", "http"}};
    const sp::Allowed metadata[] = {{"entry", "receipt label", "not read by execution"}};
    check(sp::compare(meta_cli, meta_server, metadata, true).ok(), "strict parity permits proven metadata");
    const sp::Allowed wildcard[] = {{"*", "scheduler", "some routes tested"}};
    check(!sp::compare(a, b, wildcard, true).ok(), "strict parity refuses wildcard evidence");
    const auto l = sp::from_lines("a=1\nb=2 (auto)\r\nnote\n\n");
    check(l.size() == 3U && l.at("a") == "1" && l.at("b") == "2 (auto)" && l.at("line.0") == "note", "from_lines");
    const std::vector<std::string> args{"--rank", "0", "--tp", "--rows", "8", "16"};
    const auto g = sp::from_args(args);
    check(g.size() == 3U && g.at("arg:--rank") == "0" && g.at("arg:--tp").empty() && g.at("arg:--rows") == "8 16", "from_args");
    check(sp::text("f", sp::compare(a, b, {})).find("f UNEXPLAINED y cli=2 server=3") != std::string::npos, "report text");
}

#if defined(STRIX_PARITY_GLM)
// GLM: strix-glm5-rank and strix-glm5-serve call rd::scan / rd::resolve; the server forwards its rank options through
// rd::resolve_serve, including normal decode and startup adjustment (apps/glm_serve_main.cpp).
void glm() {
    namespace rd = strix::glm::rank_defaults;
    const rd::EnvGet none = [](const char*) -> const char* { return nullptr; };
    const std::vector<std::string> request{"--world", "2"};
    // Nothing is allowed to differ: max-rows auto is admitted at load by the same function on both entry points
    // (apps/glm_auto_rows.hpp), so the server resolves the CLI's 8192-row candidate and its admission line.
    const std::span<const sp::Allowed> allowed{};
    // "value (source)": the gate compares the value; the source label names the entry point's path to it.
    const auto values = [](sp::Config c) {
        for (auto& [k, v] : c)
            if (const auto at = v.rfind(" ("); at != std::string::npos && !v.empty() && v.back() == ')') v.erase(at);
        return c;
    };
    for (const char* profile : {"exact", "fast", "turbo", "quality"})
    for (const bool complete : {false, true}) for (const bool opt_out : {false, true}) {
        auto profile_request = request;
        profile_request.insert(profile_request.end(), {"--profile", profile});
        rd::ModelFacts facts;
        facts.mtp_present = facts.mtp_complete = complete;
        facts.reason = complete ? "complete" : "missing mtp.safetensors";
        auto cli_args = profile_request;
        cli_args.insert(cli_args.end(), {"--decode", "64"});
        if (opt_out) cli_args.push_back("--no-mtp");
        const auto cli = rd::resolve(rd::scan(cli_args), none, facts);
        const auto server = rd::resolve_serve(profile_request, !opt_out, none, facts);
        const std::string expected = complete && !opt_out ? "1" : "0";
        check(cli.mtp_spec.value == expected && server.mtp_spec.value == expected &&
              cli.mtp.value == expected && server.mtp.value == expected && server.prefill == "gemv",
              "GLM complete/missing assets and opt-out select intended MTP/prefill route");
        const std::string_view expected_wmma = "0";  // no device or prompt known
        for (const auto* lever : {"pf_wmma", "mla_wmma", "kda_wmma", "attn_wmma"}) {
            const auto* cli_cell = cli.product->find(lever);
            const auto* serve_cell = server.product->find(lever);
            check(cli_cell && serve_cell && cli_cell->value == expected_wmma && serve_cell->value == expected_wmma,
                  "GLM rank and serve apply the same WMMA route for " + std::string(profile) + ": " + lever);
        }
        gate("glm", values(sp::from_lines(rd::print(cli))), values(sp::from_lines(rd::print(server))), allowed);
        gate("glm+profile-receipt", {{"profile", cli.product->json()}}, {{"profile", server.product->json()}}, {});
    }
    // The measured pair executions and each unmatched axis use the same product resolver: the TURBO literal rows
    // (8,108 / 31,258 tokens, capacity 36,864, 64 generated) and the FAST band row (2,049..8,192 tokens, any capacity,
    // bound to the run's own execution). TURBO nests FAST inside the band. EXACT never selects the bundle.
    rd::ModelFacts complete;
    complete.mtp_present = complete.mtp_complete = true;
    const auto verify_profile = [&](const char* profile, const rd::PrefillWork& work, const rd::EnvGet& env, bool on) {
        const std::vector<std::string> args{"--world", "2", "--profile", profile};
        const auto cli = rd::resolve(rd::scan(rd::serve_request_args(args, true)), env, complete, work);
        const auto serve = rd::resolve_serve(args, true, env, complete, work);
        for (const auto* lever : {"pf_wmma", "mla_wmma", "kda_wmma", "attn_wmma"}) {
            const auto* a = cli.product->find(lever);
            const auto* b = serve.product->find(lever);
            check(a && b && a->value == (on ? "1" : "0") && b->value == a->value,
                  std::string("GLM measured/fallback arithmetic agrees (") + profile + " prompt " +
                      std::to_string(work.prompt) + ") for " + lever);
        }
        const auto printed = rd::print(cli);
        const bool fast_row = printed.find(" fast=1 applies x1.370 noise 0.100 (GLM pair 8K FAST bundle") != std::string::npos;
        check(fast_row == (on && work.prompt >= 2049 && work.prompt <= 8192),  // TURBO nests the FAST cell in the band
              std::string("GLM FAST band row named in the route line (") + profile + " prompt " + std::to_string(work.prompt) + ")");
        gate("glm", values(sp::from_lines(printed)), values(sp::from_lines(rd::print(serve))), allowed);
        gate("glm+measured-receipt", {{"profile", cli.product->json()}}, {{"profile", serve.product->json()}}, {});
    };
    const auto verify = [&](const rd::PrefillWork& work, const rd::EnvGet& env, bool on) {
        verify_profile("turbo", work, env, on);
    };
    verify({true, 8108, 36864, 8192, 1, 64}, none, true);
    verify({true, 31258, 36864, 8192, 1, 64}, none, true);
    verify({false, 8108, 36864, 8192, 1, 64}, none, false);
    verify({true, 4096, 36864, 8192, 1, 64}, none, true);    // FAST band
    verify({true, 8109, 36864, 8192, 1, 64}, none, true);    // FAST band
    verify({true, 2048, 36864, 8192, 1, 64}, none, false);
    verify({true, 8193, 36864, 8192, 1, 64}, none, false);
    verify({true, 31258, 32768, 8192, 1, 64}, none, false);  // literal TURBO key: capacity
    verify({true, 31259, 36864, 8192, 1, 64}, none, false);
    verify({true, 8108, 36864, 4096, 1, 64}, none, false);
    verify({true, 8108, 36864, 8192, 2, 64}, none, false);
    verify({true, 8108, 36864, 8192, 1, 64}, [](const char* key) -> const char* {
        return std::string_view(key) == "STRIX_GLM_PF_KDA_WMMA" ? "0" : nullptr;
    }, false);
    verify({true, 8108, 36864, 8192, 1, 64}, [](const char* key) -> const char* {
        return std::string_view(key) == "STRIX_GLM_PF_ATTN_WMMA_T" ? "32" : nullptr;
    }, false);
    // FAST: the band row alone decides the prompt (capacity and generated are not keys of a band receipt).
    for (const std::uint64_t prompt : {2049ULL, 4096ULL, 8108ULL, 8192ULL})
        for (const std::uint64_t capacity : {16384ULL, 36864ULL})
            verify_profile("fast", {true, prompt, capacity, 8192, 1, 512}, none, true);
    for (const std::uint64_t prompt : {1ULL, 2048ULL, 8193ULL, 31258ULL})  // 32K stays TURBO
        verify_profile("fast", {true, prompt, 36864, 8192, 1, 64}, none, false);
    verify_profile("fast", {false, 8192, 36864, 8192, 1, 64}, none, false);  // pair not voted ready
    verify_profile("fast", {true, 8192, 36864, 4096, 1, 64}, none, false);   // unmeasured chunk
    verify_profile("fast", {true, 8192, 36864, 8192, 2, 64}, none, false);   // two lanes
    verify_profile("fast", {true, 8192, 36864, 8192, 1, 64}, [](const char* key) -> const char* {
        return std::string_view(key) == "STRIX_GLM_PF_WMMA" ? "0" : nullptr;
    }, false);                                                                 // explicit off wins
    for (const char* profile : {"exact", "quality"})
        verify_profile(profile, {true, 8192, 36864, 8192, 1, 64}, none, false);
    // Exercise the live entry points' work construction. A fresh served prompt
    // has one active lane even when the server holds a multi-slot arena.
    const std::vector<std::string> c1_args{"--world", "2", "--profile", "fast"};
    const auto c1_rank = rd::scan(rd::serve_request_args(c1_args, true));
    const auto c1_server = rd::scan(rd::serve_request_args({"--world", "2", "--profile", "fast", "--slots", "8"}, true));
    const auto c1_cli_config = rd::resolve(c1_rank, none, complete);
    const auto c1_serve_config = rd::resolve(c1_server, none, complete);
    const auto cli_prompt = rd::single_lane_prefill(true, 8192, 16384, 8192, 64);
    const auto serve_prompt = rd::serve_prefill_work(true, true, 8192, 16384, 8192, 64, true);
    check(rd::select_prefill_wmma(c1_rank, c1_cli_config, none, cli_prompt) &&
          rd::select_prefill_wmma(c1_server, c1_serve_config, none, serve_prompt),
          "GLM live FAST selection matches for one CLI request and a fresh served request in an eight-slot arena");
    for (const auto& declined : {rd::serve_prefill_work(false, true, 8192, 16384, 8192, 64, true),
                                 rd::serve_prefill_work(true, false, 8192, 16384, 8192, 64, true),
                                 rd::serve_prefill_work(true, true, 8192, 16384, 4096, 64, true)})
        check(!rd::select_prefill_wmma(c1_server, c1_serve_config, none, declined),
              "GLM serve declines the live FAST row without pair agreement, a fresh whole prompt and its measured chunk");
    // USB4 layout 2bq through the host route: CLI and serve resolve the same FAST band row on its own execution key.
    check(rd::select_prefill_wmma(c1_rank, c1_cli_config, none, rd::single_lane_prefill(true, 8192, 16384, 8192, 64, true, "usb4-host-2x1x40")) &&
          rd::select_prefill_wmma(c1_server, c1_serve_config, none, rd::serve_prefill_work(true, true, 8192, 16384, 8192, 64, true, "usb4-host-2x1x40")),
          "GLM FAST band row: CLI == serve on the measured 2bq host layout");
    const auto c1_turbo = rd::scan(rd::serve_request_args({"--world", "2", "--profile", "turbo"}, true));
    const auto c1_turbo_config = rd::resolve(c1_turbo, none, complete);
    check(!rd::select_prefill_wmma(c1_turbo, c1_turbo_config, none,
              rd::serve_prefill_work(true, true, 31258, 36864, 8192, 64, true)),
          "GLM serve does not select the CLI-only 32K TURBO receipt");
    // Scratch storage (full arena vs routed) is spelled in the execution key: the band row binds either (the gate and
    // the FAST timing ran on the full arena), the TURBO literal key names routed scratch only.
    verify_profile("fast", {true, 8192, 8448, 8192, 1, 64, false}, none, true);
    verify_profile("turbo", {true, 8192, 8448, 8192, 1, 64, false}, none, true);
    verify_profile("turbo", {true, 31258, 36864, 8192, 1, 64, false}, none, false);
    // Sampled MTP (row serve_sampled_mtp): both entry points resolve and admit it in rd::resolve, so the same request
    // gives the same route, the same receipt line and the same refusals (the command line reports it inapplicable).
    for (const char* profile : {"exact", "fast", "turbo", "quality"})
    for (const char* route : {"0", "coupled", "rejection"}) {
        const rd::EnvGet env = [route](const char* k) -> const char* {
            return std::string(k) == "STRIX_GLM_SERVE_SAMPLED_MTP" ? route : nullptr;
        };
        auto profile_request = request;
        profile_request.insert(profile_request.end(), {"--profile", profile});
        auto cli_args = profile_request;
        cli_args.insert(cli_args.end(), {"--decode", "64"});
        std::string cli_error, server_error;
        std::optional<rd::Effective> cli, server;
        try { cli = rd::resolve(rd::scan(cli_args), env); } catch (const std::invalid_argument& e) { cli_error = e.what(); }
        try { server = rd::resolve_serve(profile_request, true, env); } catch (const std::invalid_argument& e) { server_error = e.what(); }
        const bool admitted = std::string(route) != "rejection" || std::string(profile) == "fast" || std::string(profile) == "turbo";
        check(cli_error == server_error && admitted == cli_error.empty(), "GLM sampled MTP: same admission on CLI and server");
        if (!cli || !server) continue;
        // The receipt line is what both entry points print (and the server hashes into its hello); read it back.
        const auto receipt = [](const rd::Effective& e) {
            const auto s = rd::print(e);
            const auto at = s.find("effective serve-sampled-mtp=");
            return at == std::string::npos ? std::string() : s.substr(at, s.find('\n', at) - at);
        };
        const std::string want = std::string("effective serve-sampled-mtp=") + (std::string(route) == "0" ? "off" : route) + " (";
        check(receipt(*cli) == receipt(*server) && receipt(*server).rfind(want, 0) == 0 &&
              receipt(*cli).find("inapplicable") != std::string::npos,
              "GLM sampled MTP: same resolved route on CLI and server, in the receipt");
        gate("glm", values(sp::from_lines(rd::print(*cli))), values(sp::from_lines(rd::print(*server))), allowed);
    }
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
        {"app", "the entry point's own name", "effective_config receipt label, not an execution value"},
        {"session_reason", "explanation of the session_cache row (quotes the app and transport names)", "effective_config explanation only"},
        // Both entry points default to the host exchange (usb4-auto); on it the server admits its session cache across
        // requests (serve/usb4-auto is the gated session route), which a one-shot CLI run does not have.
        {"session_cache", "server session cache across requests on the gated host route; one-shot CLI keeps no session",
         "tools/minimax/session_cache_gate.sh on serve/usb4-auto: automatic and explicit ON output == OFF output"},
        {"serve.*", "scheduler / speculation policy between concurrent requests; no CLI counterpart"},
        // One host-path table (mm_defaults host_defaults()), two columns:
        {"host.STRIX_DIST_USB4_TIMEOUT_MS", "transport only: a request fails within a minute when the peer is gone"},
        {"host.STRIX_DIST_WAIT_TIMEOUT_MS", "transport only: a request fails within a minute when the peer is gone"},
        {"host.STRIX_DIST_USB4_TIMEOUT_MS.serving_reason", "the reason line exists on the server only"},
        {"host.STRIX_DIST_WAIT_TIMEOUT_MS.serving_reason", "the reason line exists on the server only"},
        // EXCEPTION for the user: exact prefill MoE table ring (1K -0.5 %), qualified with one engine owner only.
        {"host.STRIX_MM_PF_TABLE_RING", "not qualified with several sessions sharing the engine"},
        {"host.STRIX_MM_PF_TABLE_RING.serving_reason", "the reason line exists on the server only"},
    };
    gate("minimax", cli, server, allowed);
    check(cli.at("transport_capabilities") == "0" && server.at("transport_capabilities") == "0",
          "MiniMax's unqualified transport capabilities stay off by default for both entry points");
    const md::Lookup caps = [](const char* key) -> const char* {
        return std::string_view(key) == "STRIX_MM_USB4_CAPS" ? "streams,joint" : nullptr;
    };
    for (const auto* transport : {"usb4-auto", "usb4gpu", "usb4-gpu"}) {
        const auto c = sp::from_lines(md::effective_config(md::parse_cli("rank", {"model", "0", transport, "out"}), caps));
        const auto s = sp::from_lines(md::effective_config(md::parse_cli("serve", {"model", "--transport", transport}), caps));
        check(c.at("transport_capabilities") == "3" && s.at("transport_capabilities") == "3",
              "MiniMax rank and serve publish the same explicit USB4 capability mask");
    }
    for (const auto* transport : {"usb4:/dev/stream", "tcp-listen:3000"}) {
        bool refused = false;
        try { (void)md::transport_capabilities(transport, caps); }
        catch (const std::invalid_argument&) { refused = true; }
        check(refused, "MiniMax capabilities require a transport with startup capability agreement");
    }
    // FAST prefill profile rows (mm_profile_table.hpp): every row is printed by both entry points and resolves the
    // same with the bundle on, with an explicit OFF beating it, and on the FAST_I4 format (member not selected); the
    // same through the product word alone (STRIX_MM_PRODUCT_PROFILE=fast, as print_or_apply exports '--profile fast').
    const std::string word = md::kProductProfileEnv;
    for (const auto& keys : {std::map<std::string, std::string>{{word, "fast"}},
                             std::map<std::string, std::string>{{word, "fast"}, {"STRIX_MM_FAST_PREFILL", "0"}},
                             std::map<std::string, std::string>{{word, "fast"}, {"STRIX_MMX_MODEL_FORMAT", "fast_i4"}}}) {
        const md::Lookup env = [&keys](const char* k) -> const char* {
            const auto it = keys.find(k);
            return it == keys.end() ? nullptr : it->second.c_str();
        };
        const auto c = sp::from_lines(md::effective_config(md::parse_cli("rank", {}), env));
        const auto s = sp::from_lines(md::effective_config(md::parse_cli("serve", {}), env));
        const bool selected = keys.size() == 1U;
        const auto profile_line = [](const auto& lines) {
            const auto it = lines.find("prefill.fast_profile");
            return it == lines.end() ? std::string("absent") : it->second;
        };
        check(profile_line(c) == (selected ? "mm-fast-pf-v2" : "off") && profile_line(s) == profile_line(c),
              "minimax: --profile fast word: prefill.fast_profile=" + profile_line(c) + " (rank) / " + profile_line(s) + " (serve)");
        gate("minimax+word", c, s, allowed);
    }
    for (const auto& keys : {std::map<std::string, std::string>{{"STRIX_MM_FAST_PREFILL", "1"}},
                             std::map<std::string, std::string>{{"STRIX_MM_FAST_PREFILL", "1"}, {"STRIX_MM_PF_MSA_FLASH", "0"}},
                             std::map<std::string, std::string>{{"STRIX_MM_FAST_PREFILL", "1"}, {"STRIX_MMX_MODEL_FORMAT", "fast_i4"}},
                             std::map<std::string, std::string>{{"STRIX_MM_PF_MSA_FLASH", "2"}, {"STRIX_MM_PF_MSA_FLASH_LAYERS", "50-59"}}}) {
        const md::Lookup env = [&keys](const char* k) -> const char* {
            const auto it = keys.find(k);
            return it == keys.end() ? nullptr : it->second.c_str();
        };
        const auto c = sp::from_lines(md::effective_config(md::parse_cli("rank", {}), env));
        const auto s = sp::from_lines(md::effective_config(md::parse_cli("serve", {}), env));
        for (const auto& row : md::profile_table)
            check(c.count(std::string(row.env)) == 1 && s.count(std::string(row.env)) == 1,
                  "minimax: profile row " + std::string(row.env) + " printed by both entry points");
        for (const char* k : {"prefill.fast_profile", "prefill.fast_policies", "prefill.identity", "profile_fingerprint"})
            check(c.count(k) == 1 && s.count(k) == 1, std::string("minimax: ") + k + " printed by both entry points");
        gate("minimax+fast", c, s, allowed);
    }
    for (const auto* tier : {"exact", "fast", "turbo"}) {
        const md::Lookup env = [tier](const char* key) -> const char* {
            const std::string_view name(key);
            if (name == md::kProductProfileEnv) return tier;
            if (name == "STRIX_MM_PF_TABLE_RING") return "0";
            return nullptr;
        };
        for (const auto* transport : {"usb4-auto", "usb4gpu"}) {
            const auto profile_config = [&](const md::Cli& args) {
                const auto all = sp::from_lines(md::effective_config(args, env));
                sp::Config levers{{"transport", all.at("transport")},
                                  {"host.STRIX_MM_PF_TABLE_RING", all.at("host.STRIX_MM_PF_TABLE_RING")},
                                  {"profile_fingerprint", all.at("profile_fingerprint")}};
                for (const auto& row : md::profile_table)
                    levers.emplace(row.env, all.at(std::string(row.env)));
                for (const auto& [key, value] : all)
                    if (key.starts_with("prefill.") || key.starts_with("profile.")) levers.emplace(key, value);
                return levers;
            };
            const auto rank = profile_config(md::parse_cli("rank", {"model", "0", transport, "output"}));
            const auto serve = profile_config(md::parse_cli("serve", {"model", "--transport", transport}));
            check(serve.at("transport") == transport && serve.at("host.STRIX_MM_PF_TABLE_RING") == "0",
                  "MiniMax explicit transport and ring settings reach serving");
            gate(std::string("minimax+explicit+") + tier + '+' + transport, rank, serve, {});
        }
    }
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
    const auto config = [&](pd::App app, const char* profile) {
        const auto r = pd::resolve(app, {model, "--rank", "0", "--transport", "usb4auto", "--profile", profile},
                                   [](const char*) -> const char* { return nullptr; }, probe, pd::Build{true});
        auto c = sp::from_args(std::span<const std::string>(r.args).subspan(1));
        for (const auto& [k, v] : r.set_env) c["env:" + k] = v;
        for (const auto& k : r.unset_env) c["env:" + k] = "<unset>";
        c["session.enabled"] = r.session.enabled ? "1" : "0";
        c["session.disk"] = r.session.disk ? "1" : "0";
        if (r.profile) {
            c["profile"] = std::string(strix::profile::name(r.profile->profile));
            c["profile.fingerprint"] = strix::profile::hex_identity(r.profile->fingerprint());
            c["profile.receipt"] = r.profile->json();
            for (const auto& entry : r.profile->entries)
                if (entry.lever) c["profile." + std::string(entry.lever->name)] = entry.value;
        }
        return c;
    };
    static constexpr sp::Allowed allowed[] = {
        {"arg:--peer-timeout", "serving waits for the peer rank; the CLI starts both ranks together"},
        {"arg:--slots", "request slots exist on the server only"},
        {"arg:--prefill-path", "the server prefills prompts on the device batch path; the CLI's prompt goes through its decode windows"},
        {"arg:--prefill-rows", "row budget of the server's device prefill path (auto)"},
    };
    namespace pr = strix::profile;
    for (const char* profile : {"exact", "fast", "turbo", "quality"})
        gate(std::string("ds41+") + profile, config(pd::App::rank, profile), config(pd::App::serve, profile), allowed);
    const char* marker = pr::environment({}, pr::kPresetEnv);
    const auto previous = marker ? std::optional<std::string>(marker) : std::nullopt;
    for (const char* profile : {"exact", "fast", "turbo"}) {
        const auto native = config(pd::App::rank, profile);
        check(pr::export_environment(pr::kPresetEnv, "1"), "enable DeepSeek model preset marker");
        const auto preset = config(pd::App::serve, profile);
        strix::env_arms::set_env(pr::kPresetEnv, previous);
        gate(std::string("ds41+preset+") + profile, native, preset, allowed);
        check(preset.at("arg:--slots") == "8", "ds41: serve uses the resolved eight slots");
    }
    {
        pd::Probe with_draft = probe;
        with_draft.list = [](const std::string& dir) {
            if (dir != "/md") return std::vector<std::string>{};
            return std::vector<std::string>{"z-DSpark-MXFP4.gguf", "a-DSpark-MXFP4.gguf", "unrelated.gguf"};
        };
        const auto draft_args = [&](pd::App app, bool opt_out, bool explicit_draft = false) {
            std::vector<std::string> args{model, "--rank", "0", "--transport", "usb4auto", "--profile", "fast"};
            if (opt_out) args.emplace_back("--no-dspark");
            if (explicit_draft) args.insert(args.end(), {"--dspark", "/md/z-DSpark-MXFP4.gguf"});
            const auto resolved = pd::resolve(app, args, [](const char*) -> const char* { return nullptr; },
                                              with_draft, pd::Build{true});
            return sp::from_args(std::span<const std::string>(resolved.args).subspan(1));
        };
        const auto served = draft_args(pd::App::serve, false);
        const auto draft = served.find("arg:--dspark");
        check(draft != served.end() && draft->second == "/md/a-DSpark-MXFP4.gguf",
              "ds41: serve discovers the first eligible draft beside the target");
        check(!draft_args(pd::App::serve, true).contains("arg:--dspark"),
              "ds41: the server opt-out suppresses discovery");
        check(!draft_args(pd::App::rank, false).contains("arg:--dspark"),
              "ds41: automatic DSpark discovery applies only to serve");
        const auto explicit_cli = draft_args(pd::App::rank, false, true);
        const auto explicit_server = draft_args(pd::App::serve, false, true);
        const auto draft_option = [](const sp::Config& options) {
            const auto it = options.find("arg:--dspark");
            return sp::Config{{"draft", it == options.end() ? "" : it->second}};
        };
        gate("ds41/explicit-draft", draft_option(explicit_cli), draft_option(explicit_server), {});
        check(draft_option(explicit_server).at("draft") == "/md/z-DSpark-MXFP4.gguf",
              "ds41: explicit draft takes precedence over a discoverable draft");
    }
    namespace ds = strix::deepseek41;
    ds::PrefillOptions measured;
    measured.world = 2;
    measured.max_positions = 8192;
    measured.moe_chunk_rows = 4096;
    measured.row_split = false;
    measured.cone = false;
    measured.quality_head_rows = 64;
    std::uint32_t lanes = 1;
    const auto request = [&](pd::App app, const char* rank, std::uint64_t prompt) {
        const auto startup = pd::resolve(app, {model, "--rank", rank,
            "--transport", app == pd::App::prefill ? "usb4-auto" : "usb4auto", "--profile", "turbo"},
            [](const char*) -> const char* { return nullptr; }, probe, pd::Build{true});
        const auto result = pd::resolve_prefill_profile(*startup.profile, measured, prompt, lanes,
                                                       "gfx1151|AMD Radeon 8060S Graphics|fixture", true).value_or(*startup.profile);
        return sp::Config{{"d1_prefill", result.find("d1_prefill")->value},
                          {"decoder_replay", result.find("decoder_replay")->value},
                          {"fingerprint", pr::hex_identity(result.fingerprint())}};
    };
    measured.max_positions = 32768;
    for (const auto prompt : {4095U, 8128U, 8193U, 32768U, 32769U}) {
        const auto cli = request(pd::App::prefill, "0", prompt);
        const auto server = request(pd::App::serve, "1", prompt);
        gate("ds41+prompt+" + std::to_string(prompt), cli, server, {});
        const bool banded = prompt >= 4096U && prompt <= 32768U, replay = banded && prompt <= 8192U;
        check(cli.at("d1_prefill") == (banded ? "1" : "0") && cli.at("decoder_replay") == (replay ? "256" : "0"),
              "ds41: both ranks use D1 inside the measured bands, TURBO R256 only up to 8,192");
    }
    // The default server (eight slots, session RAM, no disk tier) sends a fresh whole text prompt to the same device
    // admission as the CLI; the slot count is not a scope. Pieces, image rows and disk tiers keep the previous route.
    check(pd::served_full_prompt(0U, 8128U, 8128U, false, false), "ds41: an eight-slot server enters the CLI admission");
    check(!pd::served_full_prompt(0U, 4096U, 8128U, false, false), "ds41: a cut piece keeps the previous route");
    check(!pd::served_full_prompt(256U, 8128U, 8128U, false, false), "ds41: a continuation keeps the previous route");
    check(!pd::served_full_prompt(0U, 8128U, 8128U, true, false), "ds41: image rows keep the previous route");
    check(!pd::served_full_prompt(0U, 8128U, 8128U, false, true), "ds41: a disk tier keeps the previous route");
    // The live lane count reaches the band rows (max_lanes 1): a pass of two sequences declines on both entries.
    lanes = 2;
    for (const auto prompt : {8128U, 32768U}) {
        const auto cli = request(pd::App::prefill, "0", prompt);
        const auto server = request(pd::App::serve, "1", prompt);
        gate("ds41+lanes2+" + std::to_string(prompt), cli, server, {});
        check(cli.at("d1_prefill") == "0" && cli.at("decoder_replay") == "0", "ds41: two lanes leave the measured bands");
    }
    lanes = 1;

    // The options each entry really builds. The pair CLI (apps/deepseek41_prefill_main.cpp options()): positions =
    // the prompt, no row cap, no drafter seed. The default server (slot_prefill_options): positions = --context, rows =
    // the auto capacity, and DSpark's 128-row seed. Same request -> same D1 / R256 cells (the seed rides on R256:
    // plan_bounded_replay computes the same rows for any seed <= R). Since the held-out re-score (gate_r) R256 is a
    // TURBO row up to 8,192 only, so a seeded server request without R256 (FAST, TURBO above 8,192) declines D1 and
    // keeps the previous prefill route while the CLI runs D1: a known served/CLI gap until a seeded D1 receipt exists.
    const auto cells = [&](const ds::PrefillOptions& o, const char* rank, std::uint64_t prompt, pd::App app,
                           const char* profile) -> sp::Config {
        const auto startup = pd::resolve(app, {model, "--rank", rank,
            "--transport", app == pd::App::prefill ? "usb4-auto" : "usb4auto", "--profile", profile},
            [](const char*) -> const char* { return nullptr; }, probe, pd::Build{true});
        const auto result = pd::resolve_prefill_profile(*startup.profile, o, prompt, 1U,
                                                       "gfx1151|AMD Radeon 8060S Graphics|fixture", true);
        if (!result) return {{"d1_prefill", "declined"}, {"decoder_replay", "declined"}};
        return {{"d1_prefill", result->find("d1_prefill")->value}, {"decoder_replay", result->find("decoder_replay")->value},
                {"profile.receipt", result->json()}};
    };
    for (const char* profile : {"exact", "fast", "turbo"})
        for (const auto prompt : {2048U, 8192U, 32768U}) {
            ds::PrefillOptions cli_o;  // the CLI defaults (row split, cone, head 1, window 8, chunk 1024, lanes 4)
            cli_o.world = 2;
            cli_o.max_positions = prompt;
            ds::PrefillOptions serve_o;
            serve_o.world = 2;
            serve_o.max_positions = 36864;
            serve_o.max_rows = 32768;
            serve_o.draft_rows = 128;
            const auto cli = cells(cli_o, "0", prompt, pd::App::prefill, profile);
            const auto server = cells(serve_o, "1", prompt, pd::App::serve, profile);
            const bool fast = std::string_view(profile) != "exact", banded = fast && prompt >= 4096U;
            const bool replay = banded && std::string_view(profile) == "turbo" && prompt <= 8192U;
            if (banded && !replay) {
                check(server.at("d1_prefill") == "declined" && cli.at("d1_prefill") == "1" && cli.at("decoder_replay") == "0",
                      std::string("ds41: without R256 the seeded server declines D1 (") + profile + ")");
                continue;
            }
            gate(std::string("ds41+request+") + profile + "+" + std::to_string(prompt), cli, server, {});
            check(server.at("d1_prefill") == (banded ? "1" : "0") && server.at("decoder_replay") == (replay ? "256" : "0"),
                  std::string("ds41: the seeded server request resolves the CLI cells (") + profile + ")");
        }
    {
        ds::PrefillOptions seeded;
        seeded.world = 2;
        seeded.max_positions = 36864;
        seeded.draft_rows = 257;  // a seed wider than R: the bounded plan refuses it, so the band declines
        check(cells(seeded, "0", 8192U, pd::App::serve, "fast").at("d1_prefill") == "declined",
              "ds41: a seed wider than the replay bound declines D1");
        for (const std::uint64_t total : {std::uint64_t{8192}, std::uint64_t{32768}}) {
            strix::ds41::SuffixReplayGeometry g0, g1;
            g0.draft_rows = 0;
            g1.draft_rows = 128;
            const auto a = strix::ds41::plan_bounded_replay(g0, 0U, total, 256U);
            const auto b = strix::ds41::plan_bounded_replay(g1, 0U, total, 256U);
            bool same = a.replay_row_layers == b.replay_row_layers && a.layers.size() == b.layers.size();
            for (std::size_t i = 0; same && i < a.layers.size(); ++i)
                same = a.layers[i].out_begin == b.layers[i].out_begin && a.layers[i].kv_begin == b.layers[i].kv_begin;
            check(same, "ds41: the bounded replay computes the same rows with and without the 128-row seed");
        }
    }
    // The served piece plan: a fresh prompt in the tier's band runs as one piece when the prefill holds it; everything
    // else keeps the default pieces (cache capture, a disk tier, image rows, a continuation, EXACT).
    {
        const auto band = [&](const char* profile) {
            return pd::whole_prompt_band(*pd::resolve(pd::App::serve, {model, "--rank", "0", "--transport", "usb4auto",
                "--profile", profile}, [](const char*) -> const char* { return nullptr; }, probe, pd::Build{true}).profile);
        };
        const auto fast = band("fast"), turbo = band("turbo"), exact = band("exact");
        check(fast.min == 4096U && fast.max == 32768U && turbo.min == fast.min && turbo.max == fast.max && exact.max == 0U,
              "ds41: the whole-prompt band is the D1 rows' [4096, 32768] in FAST and TURBO, none in EXACT");
        check(pd::whole_prompt_capacity(fast, 36864U) == 32768U && pd::whole_prompt_capacity(fast, 16384U) == 16384U &&
              pd::whole_prompt_capacity(exact, 36864U) == 0U, "ds41: the band's capacity request is bounded by the context");
        check(pd::served_piece_rows(16384U, 32768U, 32768U, fast, 0U, false, false, false) == 32768U,
              "ds41: a fresh 32,768-token FAST prompt is one piece (the CLI execution)");
        check(pd::served_piece_rows(16384U, 32768U, 32768U, exact, 0U, false, false, false) == 16384U,
              "ds41: EXACT keeps the measured 16,384-row pieces");
        check(pd::served_piece_rows(16384U, 16384U, 32768U, fast, 0U, false, false, false) == 16384U,
              "ds41: a prefill that cannot hold the prompt keeps the pieces");
        check(pd::served_piece_rows(16384U, 32768U, 32768U, fast, 0U, false, false, true) == 16384U &&
              pd::served_piece_rows(16384U, 32768U, 32768U, fast, 0U, false, true, false) == 16384U &&
              pd::served_piece_rows(16384U, 32768U, 32768U, fast, 0U, true, false, false) == 16384U &&
              pd::served_piece_rows(16384U, 32768U, 32768U, fast, 512U, false, false, false) == 16384U,
              "ds41: capture, a disk tier, image rows and a continuation keep the pieces");
        check(pd::served_piece_rows(16384U, 32768U, 40000U, fast, 0U, false, false, false) == 16384U &&
              pd::served_piece_rows(16384U, 32768U, 8192U, fast, 0U, false, false, false) == 16384U,
              "ds41: outside the band, and inside one piece already, nothing changes");
        // the served 32K prompt (30,975 tokens): the piece rows round up to 8 and the default plan is ONE piece to the end
        const auto rows = pd::served_piece_rows(16384U, 32768U, 30975U, fast, 0U, false, false, false);
        const auto plan = ds::select_prefix_prefill_plan(0U, 0U, 30975U, 1024U, rows, 9U, false, false, false, 0U, 0U);
        check(rows == 30976U && plan.ends == std::vector<std::uint32_t>{30975U} && plan.points.empty(),
              "ds41: an unaligned whole prompt is one piece (rows rounded to 8, plan_prefill_pieces accepts them)");
        check(pd::served_piece_rows(16384U, 32768U, 32767U, fast, 0U, false, false, false) == 32768U &&
              pd::served_piece_rows(16384U, 30968U, 30975U, fast, 0U, false, false, false) == 16384U,
              "ds41: the rounded piece must still fit the capacity");
    }

}
#endif

#if defined(STRIX_PARITY_GEMMA4)
// Gemma: strix-gemma4-decode calls parse_options, strix-gemma4-serve calls prepare_serve = parse_options + the serve
// additions (src/gemma4/engine/runtime.cpp). Same request words through both; every option and profile entry compared.
void gemma() {
    namespace e = strix::gemma4::engine;
    const auto isolated = [](auto&& resolve) {
        // Both native executables export the selected profile to the process. Start each arm with the
        // same environment so an exported CLI lever does not turn into an explicit HTTP override.
        struct ProfileEnvironment {
            // An unset switch is restored as unset: on POSIX an empty value would stay defined and count as an
            // explicit (and refused) setting at the next resolution.
            std::vector<std::pair<std::string, std::optional<std::string>>> saved;
            ProfileEnvironment() {
                for (const auto& lever : strix::gemma4::profile::levers()) {
                    if (lever.env.empty()) continue;
                    const std::string key(lever.env);
                    const char* raw = strix::profile::environment({}, key);
                    saved.emplace_back(key, raw ? std::optional<std::string>(raw) : std::nullopt);
                }
            }
            ~ProfileEnvironment() {
                for (const auto& [key, value] : saved) strix::env_arms::set_env(key, value);
            }
        } environment;
        return resolve();
    };
    const auto config = [](const e::Options& o) {
        sp::Config c;
        c["model"] = o.model.generic_string();
        c["rope"] = o.rope.generic_string();
        c["backend"] = o.backend;
        c["threads"] = std::to_string(o.threads);
        c["streams"] = std::to_string(o.streams);
        c["request.max_tokens"] = std::to_string(o.request.max_tokens);
        std::string stops;
        for (const auto id : o.request.stop_ids) stops += std::to_string(id) + ",";
        c["request.stop_ids"] = stops;
        c["request.temperature"] = std::to_string(o.request.sampling.temperature);
        c["request.top_p"] = std::to_string(o.request.sampling.top_p);
        c["request.top_k"] = std::to_string(o.request.sampling.top_k);
        c["request.seed"] = std::to_string(o.request.sampling.seed);
        c["request.dflash"] = std::to_string(static_cast<int>(o.request.dflash));
        c["request.draft_k"] = std::to_string(o.request.draft_k);
        c["draft"] = o.draft.generic_string();
        c["draft_mtp"] = o.draft_mtp.generic_string();
        c["profile"] = std::to_string(static_cast<int>(o.profile.name));
        c["profile.app"] = std::to_string(static_cast<unsigned>(o.profile.app));
        c["profile.receipt"] = o.profile.json();
        for (const auto& entry : o.profile.entries)
            if (entry.lever) c["profile." + std::string(entry.lever->name)] = entry.value;
        return c;
    };
    std::vector<std::string> words{"x", "--model", "/m/model.gguf", "--rope-tables", "/m/rope"};
    std::vector<char*> argv;
    for (auto& w : words) argv.push_back(w.data());
    const auto cli = isolated([&] { return config(e::parse_options(static_cast<int>(argv.size()), argv.data())); });
    const auto served = isolated([&] { return e::prepare_serve(static_cast<int>(argv.size()), argv.data()); });
    check(!served.exit_code, "gemma: prepare_serve accepts the CLI's request words");
    static constexpr sp::Allowed allowed[] = {
        // EXCEPTION for the user: the server sizes session-fixed choices (the EXL3 core identity) for the requests that
        // can decode together (STRIX_G4_SERVE_SLOTS, default 8); the CLI runs one stream unless --streams N.
        {"streams", "the server decodes up to STRIX_G4_SERVE_SLOTS requests together"},
    };
    if (!g_strict_gemma) gate("gemma", cli, config(served.options), allowed);

    // Explicitly match concurrency so the test compares only model/tier settings.
    for (const std::string model_name : {"gemma-4-e4b-it", "nemotron-3.5-bolt-e4b"})
    for (const std::string tier : {"exact", "fast", "turbo"}) {
        std::vector<std::string> tier_words = words;
        tier_words[2] = "/m/" + model_name + ".gguf";
        tier_words.insert(tier_words.end(), {"--profile", tier, "--streams", "1", "--draft", "off"});
        std::vector<char*> tier_argv;
        for (auto& word : tier_words) tier_argv.push_back(word.data());
        const auto native = isolated([&] { return e::parse_options(static_cast<int>(tier_argv.size()), tier_argv.data()); });
        const auto server = isolated([&] { return e::prepare_serve(static_cast<int>(tier_argv.size()), tier_argv.data()); });
        check(!server.exit_code, "gemma: serve accepts " + tier);
        check(server.deployment_slots == native.streams,
              model_name + " " + tier + ": the HTTP scheduler receives the explicitly matched stream count");
        check(native.profile.name == strix::profile::parse(tier), "gemma: native tier survives parsing");
        const auto* identity = native.profile.find("decode_identity");
        check(identity && identity->value == (tier == "exact" ? "g4-v1" : "g4-a8-v1"),
              "gemma: " + tier + " resolves its decode identity");
        const auto* attention = native.profile.find("decode_attention");
        check(attention && attention->value == "exact", "gemma: " + tier + " keeps qualified exact decode attention");
        gate(model_name + "/" + tier, config(native), config(server.options), {});
    }

    namespace fs = std::filesystem;
    const auto model = fs::temp_directory_path() /
        ("strix-gemma4-draft-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto drafter = model / "drafter-mtp";
    fs::create_directories(drafter);
    const auto expect_drafter = [&](const std::string& tier, const fs::path& model_path, const fs::path& expected) {
        std::vector<std::string> args{"x", "--model", model_path.string(), "--rope-tables", "rope", "--profile", tier};
        std::vector<char*> pointers;
        for (auto& arg : args) pointers.push_back(arg.data());
        const auto options = isolated([&] { return e::parse_options(static_cast<int>(pointers.size()), pointers.data()); });
        const auto server = isolated([&] { return e::prepare_serve(static_cast<int>(pointers.size()), pointers.data()); });
        check(!server.exit_code && options.draft_mtp == expected && server.options.draft_mtp == expected,
              "gemma " + tier + ": discover only a drafter with its weights file");
    };
    for (const auto* tier : {"exact", "fast", "turbo"}) expect_drafter(tier, model / "target.gguf", {});
    std::ofstream(drafter / "model.safetensors").put('x');
    for (const auto* tier : {"exact", "fast", "turbo"}) expect_drafter(tier, model / "target.gguf", drafter);
    fs::remove(drafter / "model.safetensors");
    fs::remove(drafter);
    fs::remove(model);
}
#endif

}  // namespace

int main(int argc, char** argv) {
    std::set<std::string_view> expected;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--print") g_print = true;
        else if (arg == "--require-output-parity") g_require_output_parity = true;
        else if (arg == "--strict-gemma") g_strict_gemma = true;
        else if (arg.starts_with("--expect-family=")) expected.emplace(arg.substr(16));
        else {
            std::cerr << "FAIL: unknown parity argument " << arg << "\n";
            return 1;
        }
    }
    try {
        helper_tests();
        if (g_strict_gemma) {
#if defined(STRIX_PARITY_GEMMA4)
            g_require_output_parity = true;
            gemma();
#else
            check(false, "Gemma resolver is required for strict parity");
#endif
            return g_failures ? 1 : 0;
        }
#if defined(STRIX_PARITY_GLM)
        std::cout << "included: glm (exact/fast/turbo/quality; complete/missing MTP assets, default/opt-out)\n";
        glm();
        expected.erase("glm");
#else
        std::cout << "skipped: glm (resolver not linked)\n";
#endif
#if defined(STRIX_PARITY_MINIMAX)
        std::cout << "included: minimax\n";
        minimax();
        expected.erase("minimax");
#else
        std::cout << "skipped: minimax (resolver not linked)\n";
#endif
#if defined(STRIX_PARITY_DS41)
        std::cout << "included: ds41\n";
        ds41();
        expected.erase("ds41");
#else
        std::cout << "skipped: ds41 (resolver not linked)\n";
#endif
#if defined(STRIX_PARITY_GEMMA4)
        std::cout << "included: gemma\n";
        gemma();
        expected.erase("gemma");
#else
        std::cout << "skipped: gemma (strix_gemma4_engine not built)\n";
#endif
        for (const auto family : expected) check(false, "expected parity resolver missing: " + std::string(family));
        std::cout << "qwen: strix-qwen-effective-config-tests (serve / CLI parity case)\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: exception: " << e.what() << "\n";
        return 1;
    }
    if (g_failures) std::cerr << g_failures << " failure(s)\n";
    else std::cout << "serve/cli parity: ok\n";
    return g_failures ? 1 : 0;
}
