#include "strix/deepseek41_pair_defaults.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// CPU-only equivalence of the two-node production defaults with the measured receipt, and of every off switch with
// the previous path (no device, no model: a fake filesystem).
//
// Receipt: the 2026-10-26 qualification run of the two-node production build (same code as this release),
// run by the node test scripts. Each app ran as
//   env LD_LIBRARY_PATH=... $EXTRA_ENV timeout L $BIN/$APP $MODELS --rank R --transport T --connect-timeout 60 ARGS
// with EXTRA_ENV and ARGS below ($RQ/ov, $X, $I/vocab are the lab's pack locations; here every pack is in
// <model dir>/packs). Gains of that build vs the previous production build: TTFT 1K
// 2.098 -> 1.954 s, 8K 11.090 -> 9.757 s; R1 28.873 -> 28.333 ms/token, R6 54.01 -> 52.70 ms, N8 71.84 -> 70.24 ms/step;
// serve C1 43.58 -> 48.00 tok/s; every run exact.
namespace {

namespace pd = strix::deepseek41::pair_defaults;

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

const std::string MD = "/lab/models/DeepSeek-V4.1-Flash-1017";
const std::string MODEL = MD + "/DeepSeek-V4.1-Flash-1017-Q2.gguf";
const std::string DRAFT = MD + "/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf";
const std::string PACKS = MD + "/packs";

std::string expand(std::string s) {
    const std::map<std::string, std::string> vars = {
        {"$RQ/ov", PACKS}, {"$X", PACKS}, {"$I/vocab", PACKS + "/vocab"}, {"$DRAFT", DRAFT}, {"$MD", MD}};
    for (const auto& [k, v] : vars)
        for (std::size_t p; (p = s.find(k)) != std::string::npos;) s.replace(p, k.size(), v);
    return s;
}
std::vector<std::string> words(const std::string& s) {
    std::istringstream in(expand(s));
    std::vector<std::string> out;
    for (std::string w; in >> w;) out.push_back(w);
    return out;
}

// node test scripts, verbatim
const std::string CU6 = "STRIX_DS41_OVERLAY=$RQ/ov/c-u6.gguf STRIX_DS41_CB4=1";
const std::string TRB = "STRIX_DS41_TRELLIS=$X/ours-K4 STRIX_DS41_TRELLIS_MIX=head=$X/ours-K5 STRIX_DS41_TRELLIS_SITES=all "
                        "STRIX_DS41_TRELLIS_DENSE_FULL=1 STRIX_DS41_TRELLIS_HEADOUT=1";

struct Receipt {
    pd::App app;
    std::string env;   // EXTRA_ENV (harness-only variables left out: STRIX_DS41_PREFIX_DIR = the turn's scratch dir)
    std::string args;  // --rank/--transport/--connect-timeout of the pair launcher + the production part of ARGS
    std::string transport;
};
// The bench-only part of ARGS (prompt, length, timing, check modes) is the workload, not the configuration; it is
// not in `args`. --layers 40 and --connect-timeout 60 are the apps' code defaults (unchanged).
const std::vector<Receipt> receipts = {
    // R1/R6: run ... "$CU6 $TRL" $R6 (R6 = --layers 40 --prompt-ids .. --generate 64 --engram-direct
    //   --no-gather --tp-attention --window-rows 6 --windows 32)
    {pd::App::rank, CU6 + " " + TRB, "--layers 40 --engram-direct --tp-attention", "usb4auto"},
    // N8: (N8 = --layers 40 --tp-attention --engram-direct --prompts .. --n 8 --generate 48
    //   --sync-start --mode batch)
    {pd::App::batch, CU6 + " " + TRB, "--layers 40 --tp-attention --engram-direct", "usb4auto"},
    // TTFT: run ... strix-deepseek41-prefill "$CU6 STRIX_DS41_EXPERT_TP=0 $E" $PR --layers 40 --timing
    {pd::App::prefill, CU6 + " STRIX_DS41_EXPERT_TP=0", "--layers 40", "usb4-auto"},
    // serve: the node serve launcher; --prefill-rows auto replaces the receipt's 2048 (the prefill-rows A/B: larger
    // pieces, same bits)
    {pd::App::serve, CU6 + " " + TRB,
     "--layers 40 --tp-attention --engram-direct --peer-timeout 120 --prefill-path device --slots 8 --context 4096 "
     "--prefill-rows auto --dspark $DRAFT --dspark-vocab auto:$I/vocab",
     "usb4auto"},
};
// Code defaults of the apps the receipt passed explicitly (unchanged by this resolver; serve's Args has tp = true,
// engram_direct = true).
std::map<std::string, std::string> code_defaults(pd::App app) {
    std::map<std::string, std::string> m = {{"--layers", "40"}, {"--connect-timeout", "60"}};
    if (app == pd::App::serve) m.insert({{"--context", "4096"}, {"--tp-attention", ""}, {"--engram-direct", ""}});
    return m;
}
// rank/batch/prefill have no value-less flags among the receipt flags but these
const std::set<std::string> switches = {"--tp-attention", "--engram-direct", "--no-tp-attention", "--engram-mapped", "--no-dspark"};

// flag -> value (last one wins, as the parsers), MODEL under "".
std::map<std::string, std::string> flags(const std::vector<std::string>& argv) {
    std::map<std::string, std::string> m;
    for (std::size_t i = 0; i < argv.size(); ++i) {
        if (argv[i].rfind("--", 0) != 0) m[""] = argv[i];
        else if (switches.count(argv[i])) m[argv[i]] = "";
        else if (i + 1 < argv.size()) m[argv[i]] = argv[i + 1], ++i;
    }
    return m;
}

struct World {
    std::map<std::string, std::string> env;
    std::set<std::string> files = {MODEL, DRAFT, PACKS + "/c-u6.gguf", PACKS + "/vocab/core48k.bin", PACKS + "/vocab/core32k.bin"};
    std::set<std::string> dirs = {MD, PACKS, PACKS + "/ours-K4", PACKS + "/ours-K5", PACKS + "/vocab"};
    pd::Build build{true};
    std::uint64_t memory = 64ULL << 30U, free_disk = 512ULL << 30U;

    pd::Resolved run(pd::App app, const std::vector<std::string>& args) const {
        pd::Probe p;
        p.available_memory = [this] { return memory; };
        p.writable_space = [this](const std::string&) { return free_disk; };
        p.file = [this](const std::string& f) { return files.count(f) != 0; };
        p.dir = [this](const std::string& d) { return dirs.count(d) != 0; };
        p.list = [this](const std::string& d) {
            std::vector<std::string> out;
            for (const auto& f : files)
                if (f.rfind(d + "/", 0) == 0 && f.find('/', d.size() + 1) == std::string::npos) out.push_back(f.substr(d.size() + 1));
            return out;
        };
        return pd::resolve(app, args, [this](const char* n) -> const char* {
            const auto it = env.find(n);
            return it == env.end() ? nullptr : it->second.c_str();
        }, p, build);
    }
};

std::map<std::string, std::string> env_of(const pd::Resolved& r) { return {r.set_env.begin(), r.set_env.end()}; }
std::map<std::string, std::string> env_of(const std::string& assignments) {
    std::map<std::string, std::string> m;
    for (const auto& w : words(assignments)) m[w.substr(0, w.find('='))] = w.substr(w.find('=') + 1);
    return m;
}
bool has_arg(const pd::Resolved& r, const std::string& a) { return std::find(r.args.begin(), r.args.end(), a) != r.args.end(); }
std::vector<std::string> plain(const Receipt& rc) { return {MODEL, "--rank", "0", "--transport", rc.transport}; }

void check_notices(pd::App app, const pd::Resolved& r, std::vector<std::string> expected, const std::string& reason,
                   const std::string& message) {
    if (app == pd::App::serve) expected.push_back("session cache: off (" + reason + ")");
    check(r.notices == expected, message + "\n" + pd::describe(app, r));
}
void test_equivalence() {
    const World w;
    for (const auto& rc : receipts) {
        const auto name = std::string(pd::app_name(rc.app));
        const auto r = w.run(rc.app, plain(rc));
        check_notices(rc.app, r, {}, "route not yet GPU-qualified: tp2-http-batch-dspark", name + ": plain invocation notices");
        check(r.unset_env.empty(), name + ": nothing unset");
        check(env_of(r) == env_of(rc.env), name + ": environment == receipt EXTRA_ENV\n" + pd::describe(rc.app, r));
        auto want = flags(words(MODEL + " --rank 0 --transport " + rc.transport + " --connect-timeout 60 " + rc.args));
        auto got = flags(r.args);
        for (const auto& [k, v] : code_defaults(rc.app)) {
            if (want.count(k) && want[k] == v && !got.count(k)) want.erase(k);
        }
        check(got == want, name + ": arguments == receipt argv (modulo unchanged code defaults)\n" + pd::describe(rc.app, r));
        check(r.args.front() == MODEL, name + ": MODEL stays the first argument");
    }
}

void test_off_switches() {
    for (const auto& rc : receipts) {
        const auto name = std::string(pd::app_name(rc.app));
        // master switch: the user's arguments untouched, every adopted code default =0
        {
            World w;
            w.env["STRIX_DS41_BASELINE"] = "1";
            const auto r = w.run(rc.app, plain(rc));
            check(r.args == plain(rc), name + ": BASELINE=1 leaves the arguments as given");
            check(r.set_env == pd::baseline_off_items(), name + ": BASELINE=1 sets exactly the adopted items =0");
            check_notices(rc.app, r, {}, "baseline switch", name + ": baseline notices");
            w.env["STRIX_DS41_SERVE_WARM"] = "1";  // an explicit value still wins
            check(env_of(w.run(rc.app, plain(rc))).count("STRIX_DS41_SERVE_WARM") == 0, name + ": BASELINE keeps explicit values");
        }
        {
            World w;
            w.env["STRIX_DS41_OVERLAY"] = "0";
            const auto r = w.run(rc.app, plain(rc));
            const auto e = env_of(r);
            check(!e.count("STRIX_DS41_OVERLAY") && !e.count("STRIX_DS41_CB4"), name + ": OVERLAY=0 no overlay / CB4");
            check(r.unset_env == std::vector<std::string>{"STRIX_DS41_OVERLAY"}, name + ": OVERLAY=0 is removed (not a path)");
            check(e.count("STRIX_DS41_TRELLIS") == (rc.app != pd::App::prefill ? 1U : 0U), name + ": OVERLAY=0 keeps the trellis item");
        }
        {
            World w;
            w.env["STRIX_DS41_TRELLIS"] = "0";
            const auto r = w.run(rc.app, plain(rc));
            for (const auto& [k, v] : r.set_env) check(k.rfind("STRIX_DS41_TRELLIS", 0) != 0, name + ": TRELLIS=0 sets no " + k);
            check(env_of(r).count("STRIX_DS41_OVERLAY") == 1U, name + ": TRELLIS=0 keeps the overlay item");
        }
        {
            World w;
            w.env["STRIX_DS41_PACKS"] = "0";
            const auto e = env_of(w.run(rc.app, plain(rc)));
            check(!e.count("STRIX_DS41_OVERLAY") && !e.count("STRIX_DS41_TRELLIS"), name + ": PACKS=0 no production packs");
        }
        {
            World w;  // an explicit environment value is never changed
            w.env["STRIX_DS41_TRELLIS_SITES"] = "shexp,head";
            w.env["STRIX_DS41_CB4"] = "0";
            const auto e = env_of(w.run(rc.app, plain(rc)));
            check(!e.count("STRIX_DS41_TRELLIS_SITES") && !e.count("STRIX_DS41_CB4"), name + ": explicit values win");
        }
    }
    for (const auto app : {pd::App::rank, pd::App::batch}) {
        const auto name = std::string(pd::app_name(app));
        const World w;
        auto args = std::vector<std::string>{MODEL, "--rank", "0", "--transport", "usb4auto", "--no-tp-attention", "--engram-mapped"};
        const auto r = w.run(app, args);
        check(!has_arg(r, "--tp-attention") && !has_arg(r, "--engram-direct"), name + ": --no-tp-attention / --engram-mapped");
        check(!has_arg(w.run(app, {MODEL, "--transport", "single"}), "--tp-attention"), name + ": one rank: no TP attention");
    }
    {
        const World w;
        const auto r = w.run(pd::App::serve, {MODEL, "--rank", "0", "--transport", "usb4auto", "--no-dspark"});
        check(!has_arg(r, "--dspark") && !has_arg(r, "--dspark-vocab"), "serve: --no-dspark");
        // explicit flags come after the defaults, so the parsers keep the user's value
        const auto s = w.run(pd::App::serve, {MODEL, "--rank", "0", "--transport", "usb4auto", "--slots", "4", "--prefill-path", "rows",
                                              "--dspark-vocab", "full", "--peer-timeout", "60", "--prefill-rows", "4096"});
        const auto f = flags(s.args);
        check(f.at("--slots") == "4" && f.at("--prefill-path") == "rows" && f.at("--dspark-vocab") == "full" &&
                  f.at("--peer-timeout") == "60" && f.at("--prefill-rows") == "4096",
              "serve: explicit flags win\n" + pd::describe(pd::App::serve, s));
        check(std::count(s.args.begin(), s.args.end(), "--dspark-vocab") == 1, "serve: an explicit --dspark-vocab is not doubled");
    }
}

void test_fallbacks() {
    {
        World w;  // no packs: the plain Q2 identity, one notice, nothing fails
        w.dirs = {MD};
        w.files = {MODEL};
        for (const auto& rc : receipts) {
            const auto r = w.run(rc.app, plain(rc));
            const auto e = env_of(r);
            check(!e.count("STRIX_DS41_OVERLAY") && !e.count("STRIX_DS41_TRELLIS"), std::string(pd::app_name(rc.app)) + ": no packs, no identity env");
            check(!r.notices.empty() && r.notices.front().find("set STRIX_DS41_PACKS") != std::string::npos,
                  std::string(pd::app_name(rc.app)) + ": no packs: notice");
        }
        const auto s = w.run(pd::App::serve, plain(receipts[3]));
        check(!has_arg(s, "--dspark") && has_arg(s, "--prefill-path"), "serve: no draft: DSpark off, the rest on");
    }
    {
        World w;  // trellis without the dense_r1 library: the overlay alone, one notice
        w.build.trellis_dense_r1 = false;
        const auto r = w.run(pd::App::rank, plain(receipts[0]));
        const auto e = env_of(r);
        check(e.count("STRIX_DS41_OVERLAY") && !e.count("STRIX_DS41_TRELLIS"), "rank: no dense_r1: overlay only");
        check_notices(pd::App::rank, r, {"deepseek41 defaults: trellis packs off (this build has no dense_r1 trellis library: CMake STRIX_DS41_TRELLIS_API=dense_r1)"}, "", "rank: no dense_r1: notice");
    }
    {
        World w;  // STRIX_DS41_PACKS points elsewhere
        const std::string other = "/srv/packs";
        w.env["STRIX_DS41_PACKS"] = other;
        w.files.insert({other + "/c-u6.gguf", other + "/vocab/core48k.bin", other + "/vocab/core32k.bin"});
        w.dirs.insert({other, other + "/ours-K4", other + "/ours-K5", other + "/vocab"});
        const auto r = w.run(pd::App::serve, plain(receipts[3]));
        const auto e = env_of(r);
        check(e.at("STRIX_DS41_OVERLAY") == other + "/c-u6.gguf" && e.at("STRIX_DS41_TRELLIS_MIX") == "head=" + other + "/ours-K5",
              "STRIX_DS41_PACKS is the pack directory");
        check(flags(r.args).at("--dspark-vocab") == "auto:" + other + "/vocab", "STRIX_DS41_PACKS holds the vocabulary");
    }
    {
        World w;
        w.env["STRIX_DS41_BATCH_ROWS16"] = "1";
        const auto r = w.run(pd::App::serve, plain(receipts[3]));
        check(!has_arg(r, "--dspark"), "serve: ROWS16: DSpark off");
        check_notices(pd::App::serve, r, {"deepseek41 defaults: serve DSpark off (not qualified with STRIX_DS41_BATCH_ROWS16=1)"},
                      "route not yet GPU-qualified: tp2-http-target", "serve: ROWS16 notices");
        const auto p = World().run(pd::App::serve, {MODEL, "--rank", "0", "--transport", "usb4auto", "--prefix-session-snapshots"});
        check(!has_arg(p, "--dspark"), "serve: session snapshots: DSpark off");
        const auto implicit = World().run(pd::App::serve, {MODEL, "--rank", "0", "--transport", "usb4auto", "--cache-profile", "agent"});
        check(has_arg(implicit, "--dspark"), "serve: implicit agent sessions never suppress auto DSpark");
        for (const auto veto : {"--no-prefix-session-snapshots", "--no-prefix-cache"}) {
            const auto disabled = World().run(pd::App::serve, {MODEL, "--rank", "0", "--transport", "usb4auto", veto, "--prefix-session-snapshots"});
            check(has_arg(disabled, "--dspark"), "serve: explicit session veto preserves auto DSpark");
        }
        const auto zero = World().run(pd::App::serve, {MODEL, "--rank", "0", "--transport", "usb4auto", "--prefix-cache-mb", "0", "--prefix-session-snapshots"});
        check(has_arg(zero, "--dspark"), "serve: numeric cache zero veto preserves auto DSpark");

        check_notices(pd::App::serve, p, {"deepseek41 defaults: serve DSpark off (--prefix-session-snapshots is not exact with --dspark)"},
                      "route not yet GPU-qualified: tp2-http-target", "serve: snapshot notices");
    }
    {
        const World w;
        const auto r = w.run(pd::App::rank, {MODEL, "--rank", "1", "--transport", "usb4auto", "--print-effective-config"});
        check(r.print_only && !has_arg(r, "--print-effective-config"), "--print-effective-config is consumed");
        const auto d = pd::describe(pd::App::rank, r);
        check(d.find("app strix-deepseek41-rank\n") == 0 && d.find("env STRIX_DS41_TRELLIS_SITES=all\n") != std::string::npos &&
                  d.find("arg --tp-attention\n") != std::string::npos,
              "describe lists env and args");
    }
}

void test_session_defaults() {
    World w; w.env["HOME"] = "cache-home";
    const std::vector<std::string> args{MODEL, "--transport", "usb4auto", "--no-dspark"};
    auto r = w.run(pd::App::serve, args);
    check(!r.session.enabled && !r.session.disk, "unqualified target serving defaults OFF");
    check(r.session.ram_bytes == w.memory / 10 && r.session.max_parked == 8, "RAM defaults");
    check(r.session.disk_bytes == 0 && r.session.directory == "cache-home/.cache/strix-engine/sessions", "disk defaults");
    const std::string session_lines = "session cache off\nsession ram_bytes 6871947673\nsession max_parked 8\n"
        "session disk off\nsession disk_bytes 0\nsession directory cache-home/.cache/strix-engine/sessions\n"
        "session reason route not yet GPU-qualified: tp2-http-target\n"
        "notice session cache: off (route not yet GPU-qualified: tp2-http-target)\n";
    check(pd::describe(pd::App::serve,r).ends_with(session_lines), "exact effective session lines");
    check(r.session.reason == "route not yet GPU-qualified: tp2-http-target", "target GPU admission gate");
    check(std::count(r.notices.begin(), r.notices.end(), "session cache: off (route not yet GPU-qualified: tp2-http-target)") == 1, "one qualification notice");
    auto force_on = args; force_on.insert(force_on.end(), {"--session-cache", "on"});
    check(!w.run(pd::App::serve, force_on).session.enabled, "on does not bypass qualification");
    const auto explicit_gate = w.run(pd::App::rank, {MODEL, "--session-save", "/tmp/gate", "--session-dspark", "--session-validate-only"});
    check(has_arg(explicit_gate, "--session-save") && has_arg(explicit_gate, "--session-dspark") && has_arg(explicit_gate, "--session-validate-only"), "explicit qualification phases preserved");
    auto off = args; off.insert(off.end(), {"--session-cache", "off"});
    check(!w.run(pd::App::serve, off).session.enabled, "CLI off switch");
    w.env["STRIX_SESSION_CACHE"] = "0";
    check(!w.run(pd::App::serve, args).session.enabled, "environment off switch");
    w.env.erase("STRIX_SESSION_CACHE");
    w.env["STRIX_DS41_BASELINE"] = "1";
    auto force = args; force.insert(force.end(), {"--session-cache", "on"});
    check(!w.run(pd::App::serve, force).session.enabled, "baseline forces sessions off");
    w.env.erase("STRIX_DS41_BASELINE");
    off = args; off.insert(off.end(), {"--session-disk", "off"});
    r = w.run(pd::App::serve, off);
    check(!r.session.enabled && !r.session.disk, "disk-only CLI off");
    w.env["STRIX_SESSION_DISK"] = "0";
    check(!w.run(pd::App::serve,args).session.disk, "disk-only env off");
    w.env.erase("STRIX_SESSION_DISK");
    auto zero_disk = args; zero_disk.insert(zero_disk.end(), {"--session-disk-bytes", "0"});
    check(!w.run(pd::App::serve,zero_disk).session.disk, "zero disk quota disables disk");
    const auto rank_off = w.run(pd::App::rank, {MODEL,"--session-save","/tmp/explicit","--session-dspark","--session-cache","off"});
    check(rank_off.args == std::vector<std::string>({MODEL,"--engram-direct"}), "rank off switch removes explicit session commands");
    const auto hash_off = w.run(pd::App::rank, {MODEL,"--session-hash-only","--session-dspark","--session-cache","off"});
    check(hash_off.args == std::vector<std::string>({MODEL,"--engram-direct"}), "rank off consumes value-free hash-only control");
    w.free_disk = 0; r = w.run(pd::App::serve,args);
    check(!r.session.enabled && !r.session.disk, "unqualified route stays off with unwritable disk");
    w.free_disk = 512ULL << 30U;
    w.memory = 1; r = w.run(pd::App::serve,args);
    check(!r.session.enabled && r.notices.size() == 1, "low memory is one-line fallback");
    w.memory = 64ULL << 30U;
    r = w.run(pd::App::serve,plain(receipts[3]));
    check(!r.session.enabled && r.session.reason.find("route not yet GPU-qualified: tp2-http-batch-dspark") != std::string::npos, "production batch DSpark unchanged");
    auto custom = args; custom.insert(custom.end(), {"--session-ram-bytes", "67108864", "--session-max-parked", "3",
        "--session-dir", "cache/sessions", "--session-disk-bytes", "268435456"});
    r = w.run(pd::App::serve,custom);
    check(!r.session.enabled && !r.session.disk && r.session.ram_bytes == 67108864 && r.session.max_parked == 3 &&
        r.session.disk_bytes == 268435456 && r.session.directory == "cache/sessions", "explicit budgets and directory");
    auto reordered = args; reordered.insert(reordered.end(), {"--dspark", DRAFT});
    check(!w.run(pd::App::serve,reordered).session.enabled, "DSpark after no-dspark stays fail closed");
    reordered.insert(reordered.end(), {"--no-dspark"});
    check(!w.run(pd::App::serve,reordered).session.enabled, "last no-dspark still requires target GPU qualification");
    w.env["XDG_CACHE_HOME"] = "/xdg";
    check(w.run(pd::App::serve,args).session.directory == "/xdg/strix-engine/sessions", "XDG directory");
    check(!w.run(pd::App::rank,plain(receipts[0])).session.enabled, "one-shot defaults never select session producer");
}

void test_session_off_settings() {
    for (const auto& rc : receipts) {
        World w;
        w.env["HOME"] = "cache-home";
        const auto args = plain(rc);
        const auto original = w.run(rc.app, args);
        const std::string reason = rc.app == pd::App::serve ? "route not yet GPU-qualified: tp2-http-batch-dspark" : "one-shot commands remain explicit";
        const std::string lines = "session cache off\nsession ram_bytes 6871947673\nsession max_parked 8\n"
            "session disk off\nsession disk_bytes 0\nsession directory cache-home/.cache/strix-engine/sessions\n"
            "session reason " + reason + "\n";
        const std::string notice = rc.app == pd::App::serve ? "notice session cache: off (" + reason + ")\n" : "";
        check(pd::describe(rc.app, original).ends_with(lines + notice), "exact default session inspection lines");
        for (const auto* value : {"0", "1", "false", "true", "on", "off", "other"}) {
            w.env["STRIX_BASELINE"] = value;
            check(pd::describe(rc.app, w.run(rc.app, args)) == pd::describe(rc.app, original), "generic baseline environment leaves resolution unchanged");
        }
        w.env.erase("STRIX_BASELINE");
        auto cli = args;
        cli.push_back("--baseline");
        const auto generic_cli = w.run(rc.app, cli);
        auto expected_args = original.args;
        expected_args.push_back("--baseline");
        check(generic_cli.args == expected_args && generic_cli.set_env == original.set_env && generic_cli.session.reason == original.session.reason,
              "generic baseline argument retains its original parser semantics");

        for (const auto* value : {"0", "1", "false", "true", "on", "off", "other"}) {
            w.env["STRIX_SESSION_CACHE"] = value;
            w.env["STRIX_SESSION_DISK"] = value;
            w.env["STRIX_SESSION_RAM_BYTES"] = "invalid";
            w.env["STRIX_SESSION_MAX_PARKED"] = "-1";
            w.env["STRIX_SESSION_DISK_BYTES"] = "999999999999999999999999999999";
            auto print_args = args;
            print_args.push_back("--print-effective-config");
            const auto off = w.run(rc.app, print_args);
            check(off.print_only && off.args == original.args && off.set_env == original.set_env && !off.session.enabled,
                  "off-path settings preserve execution and inspection");
            w.env["STRIX_DS41_BASELINE"] = "1";
            const auto baseline = w.run(rc.app, print_args);
            const std::string baseline_lines = "session cache off\nsession ram_bytes 6871947673\nsession max_parked 8\n"
                "session disk off\nsession disk_bytes 0\nsession directory cache-home/.cache/strix-engine/sessions\n"
                "session reason baseline switch\n";
            const std::string baseline_notice = rc.app == pd::App::serve ? "notice session cache: off (baseline switch)\n" : "";
            check(baseline.print_only && pd::describe(rc.app, baseline).ends_with(baseline_lines + baseline_notice), "exact baseline session inspection lines");
            check_notices(rc.app, baseline, {}, "baseline switch", "exact baseline session notice");
            w.env.erase("STRIX_DS41_BASELINE");
        }
        w.env["STRIX_DS41_BASELINE"] = "1";
        auto invalid_cli = args;
        invalid_cli.insert(invalid_cli.end(), {"--session-cache", "true", "--session-disk", "false", "--session-cache-mb", "999999999999999999999999",
            "--session-ram-bytes", "invalid", "--session-max-parked", "invalid", "--session-disk-bytes", "invalid"});
        check(!w.run(rc.app, invalid_cli).session.enabled, "baseline ignores malformed session CLI budgets");
    }
}

}  // namespace

int main() {
    test_equivalence();
    test_off_switches();
    test_fallbacks();
    test_session_defaults();
    test_session_off_settings();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("deepseek41 pair defaults: receipt equivalence (rank, batch, prefill, serve), off switches, fallbacks PASS\n");
    return 0;
}
