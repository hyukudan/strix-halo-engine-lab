// MiniMax M3.1 defaults-on resolver: the table of default-ON items, the baseline master switch and the canonical
// effective configuration (see mm_defaults.hpp).
#include "minimax/engine/mm_defaults.hpp"
#include "minimax/serve/minimax_sched.hpp"

#include <cstdio>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>

#ifndef STRIX_MM_GU4_SCALAR
#define STRIX_MM_GU4_SCALAR 1
#endif

namespace strix::minimax::defaults {
namespace {

// Measurement evidence for the adopted production defaults.
constexpr Item kItems[] = {
    {"serve_sched", "STRIX_MM_SERVE_SCHED", "interleave", "off", "Bounded serving policy; GPU qualification pending"},
    {"serve_service", "STRIX_MM_SERVE_SERVICE_MS", "250", "0", "Soft service opportunity target; GPU qualification pending"},
    {"serve_nest", "STRIX_MM_SERVE_NEST", "1", "0", "Nested short admission; extended resume-check GPU qualification pending"},
    {"serve_short_first", "STRIX_MM_SERVE_SHORT_FIRST", "1", "0", "Stable shortest-first admission, at most eight overtakes; qualification pending"},
    // decode
    {"moe_wide", "STRIX_MMX_MOE_WIDE", "2", "0",
     "Measurement: rows == alone, wide expert kernels at N2-4, N2 -3.0..-3.5 %, N4 -4..-6 %"},
    {"moe_gu_interleave", "STRIX_MMX_MOE_GU_INTERLEAVE", "1", "0",
     "Decode MoE measurement: bitwise (610 N/U/S cases, every row == alone), B8 gate/up interleave diverse U8-32 -1.5..-3 %, B1/B4 unchanged"},
    {"moe_single", "STRIX_MMX_MOE_SINGLE", "1", "0",
     "Decode MoE measurement: bitwise (610 N/U/S cases, every row == alone), B8 U>=25 singleton reader -3 %, B1/B4 unchanged"},
    {"moe_down_split", "STRIX_MMX_MOE_DOWN_SPLIT", "1", "0",
     "Decode MoE measurement: bitwise (610 N/U/S cases, every row == alone), B4 down partials + ordered finish, fresh-process "
     "OFF/ON/OFF/ON/OFF B/A U4 0.986/0.986, U8 0.993/0.989, U12 0.991/0.997, U16 0.986/1.000; B4 wide W4 Depth4 only"},
    {"index_conc", "STRIX_MM_INDEX_CONC", "1", "0",
     "Measurement: ids identical, 17/18 in-process block pairs faster, ~-0.2 ms/token"},
    {"attend_diet", "STRIX_MM_ATTEND_DIET", "1", "0",
     "Attention measurement t9: bitwise, v3 32 KB LDS vs 42 KB at sparse N2/W1 8K 83.9 -> 76.2, 32K 139.0 -> 131.6, "
     "128K 350.2 -> 340.8 us; only on those grids (1K N8, W8, 8K N4 slower stay at 42 KB)"},
    {"attend4_decode", "STRIX_MM_ATTEND4_DECODE", "1", "0",
     "Attention measurement t12: bitwise, v4 one head per wave vs v3-full at sparse N2/W1 8K 78.6 -> 75.2, 32K 132.2 -> "
     "128.3, 128K 346.4 -> 340.6 us; only on those grids (1K N1/N8, 8K N4 slower keep v3)"},
    {"xfer_crc_deferred", "STRIX_MMX_CRC", "deferred", "1",
     "Measurement: tokens identical in 7 processes, each deferred arm below its inline neighbours by 0.04-0.12 ms/token"},
    {"xb_wait", "STRIX_MM_XB_WAIT", "1", "0", "Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token"},
    {"xb_inplace", "STRIX_MM_XB_INPLACE", "1", "0", "Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token"},
    {"xb_publish", "STRIX_MM_XB_PUBLISH", "1", "0",
     "Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token (falls back without the v4/overlap cores)"},
    {"xb_postilp", "STRIX_MM_XB_POSTILP", "2", "0", "Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token"},
    {"xb_warm", "STRIX_MM_XB_WARM", "8", "0", "Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token"},
    {"dspark_attn", "STRIX_MM_DSPARK_ATTN", "2", "0",
     "Drafter measurement: bitwise, drafter attention B/A 0.964-0.975 / 0.918-0.924 / 0.815-0.819 at ctx 300/1000/4095"},
    {"draft_lin", "STRIX_MM_DRAFT_LIN", "opt", "lib", "Optimized draft linears (measurement evidence)"},
    {"linear_v11", "STRIX_MMX_LINEAR_V11", "1", "0",
     "Draft linear measurement: same bits as v10 (bitwise, ids == greedy 36/36), DSpark draft block -11.9 %"},
    // DSpark Markov chain (acceptance only, verify unchanged; C / |H| / top-C act on the cand form)
    {"markov_cand", "STRIX_MM_MARKOV", "cand", "full",
     "Markov measurement: cand C4096/H8192 + device exchange + top-C v2, verify exact, acceptance unchanged"},
    {"markov_cand_c", "STRIX_MM_MARKOV_C", "4096", "1024",
     "Markov measurement: C 4096 (offline approx1-K3: C4096/H8192 0 acceptance changes on 1857 anchors)"},
    {"markov_cand_h", "STRIX_MM_MARKOV_H", "8192", "2048",
     "Markov measurement: |H| 8192 (offline approx1-K3: C4096/H8192 0 acceptance changes on 1857 anchors)"},
    {"markov_topc_v2", "STRIX_MM_MARKOV_TOPC", "v2", "v1",
     "Markov measurement: top-C v2 = v1 canonical set / scores / picks (payload gates v1/v2 vs CPU)"},
    // prefill
    {"pf_core_v4", "STRIX_MMX_PREFILL_CORE", "v4", "lib", "Prefill core measurement: bitwise everywhere, whole call 0.34-0.58x lib"},
    {"pf_msa_batch", "STRIX_MM_PF_MSA_BATCH", "1", "0", "Prefill measurement: exact, pfall TTFT 1K -17 %, 8K -20 %"},
    {"pf_attend4", "STRIX_MM_PF_ATTEND4", "1", "0", "Prefill measurement: exact, pfall TTFT 1K -17 %, 8K -20 %"},
    {"pf_router_batch", "STRIX_MM_PF_ROUTER_BATCH", "1", "0", "Prefill measurement: exact, pfall TTFT 1K -17 %, 8K -20 %"},
    {"pf_mtfit", "STRIX_MMX_PREFILL_MTFIT", "1", "0", "Prefill measurement: exact, pfall TTFT 1K -17 %, 8K -20 %"},
    {"pf_msa_rework", "STRIX_MM_PF_MSA_REWORK", "3", "0", "Prefill attention measurement: exact, 8K B/A 0.963 / 0.957 / 0.972"},
    {"pf_idx_tile", "STRIX_MM_PF_IDX_TILE", "1", "0",
     "Prefill index measurement: exact (state/final rows identical), index scoring 8K -6.7 %, 32K -6.4 %; TTFT 8K -1.9 %, 32K -0.6 %, 1K neutral"},
    {"pf_idx_rows", "STRIX_MM_PF_IDX_ROWS", "64", "32",
     "Prefill index measurement: exact (state/final rows identical), Q64 vs Q32 index scoring 8K -35 %, 32K -28 %; TTFT 8K -1.5 %, 32K -1.8 %, 1K neutral"},
    {"pf_moe_pipe", "STRIX_MMX_PREFILL_PIPE", "1", "0", "Prefill measurement: exact, 8K B/A 0.973 / 0.935 / 0.924"},
    {"pf_kv_tile", "STRIX_MM_PF_KV_TILE", "auto", "off",
     "Prefill measurement: exact, 8K B/A 0.960 / 0.949 / 1.017; needs the Trellis tile hook, else a logged no-op"},
    {"pf_tx_uncached", "STRIX_MM_PF_TX_UNCACHED", "1", "0",
     "Prefill exchange measurement: exact, exchange-send 0.23 -> 0.002 s GPU, ~1.3 % (prefill module route)"},
    {"pf_async_start", "STRIX_MM_PF_ASYNC_START", "1", "0",
     "Grouped all-member >=2048-row gate (configurable); TP2 4096+3844 IDs identical, -2.3/-2.5 % with late fence; full state/workload qualification pending"},
    {"pf_late_src_fence", "STRIX_MM_PF_LATE_SRC_FENCE", "1", "0",
     "Admitted grouped async source lifetime; TP2 IDs identical, combined -2.3/-2.5 %; full state/workload qualification pending"},
    // transport (host path, bits unchanged)
    {"dist_crc_parallel", "STRIX_DIST_CRC_PARALLEL", "1", "0", "Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged"},
    {"dist_host_crc_overlap", "STRIX_DIST_HOST_CRC_OVERLAP", "1", "0", "Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged"},
    {"dist_host_chunk", "STRIX_DIST_HOST_CHUNK_BYTES", "1048576", "0", "Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged"},
};

// Route switches whose code default is the production route; printed as set or "default".
constexpr const char* kRoute[] = {"STRIX_MM_XFER", "STRIX_MM_PFX", "STRIX_MM_PF_INTERLEAVE"};

std::mutex g_mutex;
bool g_applied = false;
std::set<std::string> g_exported;

const char* process_env(const char* name) { return std::getenv(name); }

}  // namespace

std::span<const Item> items() { return kItems; }

std::size_t pf_async_min_rows(const char* value) {
    if (value == nullptr) return 2048;
    std::size_t rows = 0;
    const char* end = value + std::strlen(value);
    const auto parsed = std::from_chars(value, end, rows);
    if (parsed.ec != std::errc{} || parsed.ptr != end)
        throw std::invalid_argument("STRIX_MM_PF_ASYNC_MIN_ROWS: nonnegative row count required");
    return rows;
}

bool baseline(const Lookup& env) {
    const char* v = env(kBaselineEnv);
    return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
}

std::vector<Resolved> resolve(const Lookup& env) {
    const bool base = baseline(env);
    std::vector<Resolved> out;
    out.reserve(std::size(kItems));
    for (const Item& it : kItems) {
        Resolved r;
        r.item = &it;
        if (const char* v = env(it.env); v != nullptr) {
            r.value = v;
            r.source = Source::env;
        } else {
            r.value = base ? it.off : it.on;
            r.source = base ? Source::baseline : Source::on;
        }
        out.push_back(std::move(r));
    }
    return out;
}

void apply() {
    std::lock_guard<std::mutex> g(g_mutex);
    if (g_applied) return;
    g_applied = true;
    const Lookup env = process_env;
    for (const Resolved& r : resolve(env)) {
        if (r.source == Source::env) continue;
        (void)::setenv(r.item->env, r.value.c_str(), 0);
        g_exported.insert(r.item->env);
    }
    if (baseline(env))
        std::printf("minimax defaults: %s=1, the %zu optimization items are off (an explicit item value wins)\n", kBaselineEnv,
                    std::size(kItems));
}

bool from_defaults(const char* env) {
    std::lock_guard<std::mutex> g(g_mutex);
    return g_exported.count(env) != 0;
}

Cli parse_cli(const std::string& app, const std::vector<std::string>& args) {
    if (app != "rank" && app != "serve") throw std::invalid_argument("parse_cli: app must be rank or serve");
    Cli c;
    c.app = app;
    std::size_t i = 0;
    if (app == "rank") {  // MODEL_DIR RANK TRANSPORT OUT_DIR [options]
        if (args.size() > 2 && args[2].rfind("--", 0) != 0) c.transport = args[2];
        i = 4;
    } else {
        i = 1;  // MODEL_DIR [options]
    }
    for (; i < args.size(); ++i) {
        const std::string& a = args[i];
        const bool has = i + 1 < args.size();
        if (a == "--slots" && has) c.slots = static_cast<std::uint32_t>(std::stoul(args[++i]));
        else if (a == "--capacity" && has) c.capacity = std::stoull(args[++i]);
        else if (a == "--max-rows" && has) c.max_rows = static_cast<std::uint32_t>(std::stoul(args[++i]));
        else if (a == "--transport" && has && app == "serve") c.transport = args[++i];
    }
    return c;
}

namespace {
struct ServePolicy {
    serve::Sched sched;
    double service;
    double gap;
    bool short_first;
    bool nest;
};

// Pure host preflight, including defaults/baseline before apply() exports them.
ServePolicy serve_policy(const Lookup& env) {
    const auto resolved = resolve(env);
    const auto value = [&](const char* name) -> const char* {
        for (const auto& r : resolved)
            if (std::strcmp(r.item->env, name) == 0) return r.value.c_str();
        return env(name);
    };
    const auto flag = [&](const char* name) {
        const std::string v = value(name);
        if (v != "0" && v != "1") throw std::invalid_argument(std::string(name) + ": 0 | 1");
        return v == "1";
    };
    return {serve::sched_mode(value("STRIX_MM_SERVE_SCHED")),
            serve::service_ms(value("STRIX_MM_SERVE_SERVICE_MS")),
            serve::gap_ms(value("STRIX_MM_SERVE_GAP_MS")),
            flag("STRIX_MM_SERVE_SHORT_FIRST"), flag("STRIX_MM_SERVE_NEST")};
}
}  // namespace

std::string effective_config(const Cli& cli, const Lookup& env) {
    std::ostringstream o;
    o << "app=" << cli.app << "\n"
      << "transport=" << cli.transport << "\n"
      << "slots=" << cli.slots << "\n"
      << "capacity=" << cli.capacity << "\n"
      << "max_rows=" << cli.max_rows << "\n";
    for (const char* r : kRoute) {
        const char* v = env(r);
        o << r << "=" << (v != nullptr ? v : "default") << "\n";
    }
    o << "build.gu4_scalar=" << (STRIX_MM_GU4_SCALAR ? "on" : "off") << "\n";
    for (const Resolved& r : resolve(env)) o << r.item->env << "=" << r.value << "\n";
    o << "STRIX_MM_PF_ASYNC_MIN_ROWS=" << pf_async_min_rows(env("STRIX_MM_PF_ASYNC_MIN_ROWS")) << "\n";
    if (cli.app == "serve") {
        const auto policy = serve_policy(env);
        o << std::setprecision(std::numeric_limits<double>::max_digits10)
          << "serve.sched=" << (policy.sched == serve::Sched::off ? "off" : "interleave") << "\n"
          << "serve.service_ms=" << policy.service << "\n"
          << "serve.gap_ms=" << policy.gap << "\n"
          << "serve.short_first=" << (policy.short_first ? 1 : 0) << "\n"
          << "serve.nest=" << (policy.nest ? 1 : 0) << "\n";
    }
    return o.str();
}

bool print_or_apply(const std::string& app, int argc, char** argv) {
    std::vector<std::string> args;
    bool print = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], kPrintFlag) == 0) print = true;
        else args.emplace_back(argv[i]);
    }
    apply();
    (void)pf_async_min_rows(process_env("STRIX_MM_PF_ASYNC_MIN_ROWS"));
    if (app == "serve") (void)serve_policy(process_env);
    if (!print) return false;
    std::fputs(effective_config(parse_cli(app, args), process_env).c_str(), stdout);
    if (app == "serve") {
        const bool off = serve_policy(process_env).sched == serve::Sched::off;
        const char* profile = process_env("STRIX_MM_PROFILE");
        const char* identity = process_env("STRIX_MM_PREFILL_IDENTITY");
        const char* reason = off ? "sched_off" : profile != nullptr && std::atoi(profile) != 0 ? "profiling" :
                             identity != nullptr && std::strcmp(identity, "fp32") == 0 ? "fp32_prefill" : "";
        std::printf("serve.effective_sched=%s\nserve.fallback_reason=%s\nserve.max_unit_layers=dynamic\nserve.qualification=pending\n",
                    off ? "off" : *reason == 0 ? "interleave" : "unlimited", reason);
    }
    std::fflush(stdout);
    return true;
}

}  // namespace strix::minimax::defaults
