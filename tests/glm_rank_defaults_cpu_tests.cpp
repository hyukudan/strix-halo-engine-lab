// strix-glm5-rank defaults, CPU tests (no GPU, no peer): the plain default invocation must resolve to exactly the
// measured production receipt (route + flags + env), and every off switch / the baseline master switch must restore
// the old path while explicit settings win.
//   measured decode configuration: decode 64 + R8 batch, qualification passed
//   measured prefill configuration: hp host path, exact TTFT 1K/8K
#include "glm_rank_defaults.hpp"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace rd = strix::glm::rank_defaults;

namespace {

int failures = 0, checks = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        ++checks;                                                                 \
        if (!(c)) {                                                               \
            ++failures;                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);      \
        }                                                                         \
    } while (0)

using Env = std::map<std::string, std::string>;
using Args = std::vector<std::string>;

rd::EnvGet getter(const Env& env) {
    return [&env](const char* k) -> const char* {
        const auto it = env.find(k);
        return it == env.end() ? nullptr : it->second.c_str();
    };
}

rd::Effective resolve(const Args& args, const Env& env) { return rd::resolve(rd::scan(args), getter(env)); }

Args cat(Args a, const Args& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// Values only (the source tags differ by construction: the receipt passes explicitly what the default now resolves to).
bool same_values(const rd::Effective& a, const rd::Effective& b, std::string* why) {
    auto diff = [&](const std::string& k, const std::string& x, const std::string& y) {
        if (x == y) return false;
        *why += k + ": " + x + " != " + y + "; ";
        return true;
    };
    bool d = diff("identity", a.identity.value, b.identity.value);
    d = diff("mtp", a.mtp.value, b.mtp.value) || d;
    d = diff("prefill", a.prefill, b.prefill) || d;
    d = diff("prefill-requested", a.prefill_requested.value, b.prefill_requested.value) || d;
    d = diff("decode-xch", a.decode_xch.value, b.decode_xch.value) || d;
    d = diff("xch", a.xch, b.xch) || d;
    d = diff("max-rows", a.max_rows.value, b.max_rows.value) || d;
    for (const auto& it : rd::env_items()) d = diff(it.env, a.env.at(it.env).value, b.env.at(it.env).value) || d;
    return !d;
}

const Args kDeploy = {"--world", "2", "--rank", "0", "--transport", "usb4-auto"};
const Args kDecodeWork = {"--slots", "8", "--capacity", "2048", "--prompt", "w/q-chat.u32", "--decode", "64", "--batch", "8", "--steps", "16"};
const Args kTtftWork = {"--slots", "1", "--capacity", "33792", "--max-rows", "2048", "--pf-ttft", "w/q1k.u32,w/q8k.u32"};
// Measured command arguments (rank 0) and environment; STRIX_DIST_*_TIMEOUT_MS are the harness's
// watchdog limits (no route, not an optimization) and are excluded from the comparison by construction (not items).
const Args kReceiptD = {"--rank", "0", "--world", "2", "--transport", "usb4-auto", "--xch", "comm", "--mtp", "0"};
const Args kReceiptH = {"--rank", "0", "--world", "2", "--transport", "usb4-auto", "--mtp", "0"};
const Env kReceiptEnv = {{"STRIX_DIST_USB4_TIMEOUT_MS", "120000"}, {"STRIX_DIST_WAIT_TIMEOUT_MS", "120000"},
                         {"HSA_DISABLE_FRAGMENT_ALLOCATOR", "1"}};
const Env kHp = {{"STRIX_DIST_CRC_PARALLEL", "1"}, {"STRIX_DIST_HOST_CRC_OVERLAP", "1"}, {"STRIX_DIST_HOST_CHUNK_BYTES", "1048576"}};

Env merge(Env a, const Env& b) {
    for (const auto& [k, v] : b) a[k] = v;
    return a;
}

void test_plain_equals_receipt() {
    std::string why;
    // Measured decode / R8 configuration with the measured hp environment
    const auto plain = resolve(cat(kDeploy, kDecodeWork), {});
    const auto receipt = resolve(cat(kReceiptD, kDecodeWork), merge(kReceiptEnv, kHp));
    CHECK(same_values(plain, receipt, &why));
    if (!why.empty()) std::fprintf(stderr, "  decode receipt diff: %s\n", why.c_str());
    // what the receipt's log printed: identity r1b decode-xch xfer r1b-dense 7 r1b-hc 4, mtp 0, prefill gemv (requested auto)
    CHECK(plain.identity.value == "r1b" && plain.decode_xch.value == "xfer" && plain.mtp.value == "0");
    CHECK(plain.prefill == "gemv" && plain.prefill_requested.value == "auto");
    CHECK(plain.env.at("STRIX_GLM_R1B_DENSE").value == "7" && plain.env.at("STRIX_GLM_R1B_HC").value == "4");
    CHECK(!plain.baseline);
    // Measured TTFT configuration: the measurement mode resolves the same way with and without the receipt's flags/env
    why.clear();
    CHECK(same_values(resolve(cat(kDeploy, kTtftWork), {}), resolve(cat(kReceiptH, kTtftWork), merge(kReceiptEnv, kHp)), &why));
    if (!why.empty()) std::fprintf(stderr, "  ttft receipt diff: %s\n", why.c_str());
    // the old code's default (--mtp 1) sent the plain decode command to the row-by-row reference prefill
    CHECK(resolve(cat(cat(kDeploy, kDecodeWork), {"--mtp", "1"}), {}).prefill == "ref");
    // prefill chunk rows: default 4096; baseline 1024 (the old app default); explicit wins
    CHECK(plain.max_rows.value == "4096" && plain.max_rows.source == "default");
    CHECK(resolve(cat(kDeploy, kTtftWork), {}).max_rows.source == "explicit");
    CHECK(resolve(cat(cat(kDeploy, kDecodeWork), {"--max-rows", "1024"}), {}).max_rows.value == "1024");
    CHECK(resolve(cat(cat(kDeploy, kDecodeWork), {"--baseline"}), {}).max_rows.value == "1024");
}

void test_apply_env_default() {
    const auto plain = resolve(cat(kDeploy, kDecodeWork), {});
    Env set;
    rd::apply_env(plain, [&](const char* k, const char* v) { set[k] = v; });
    // the app sets exactly the items whose code default is the old path; engine defaults are reported only
    for (const auto& it : rd::env_items()) {
        if (it.app_sets) CHECK(set.count(it.env) == 1 && set[it.env] == it.on);
        else CHECK(set.count(it.env) == 0);
    }
    CHECK(set == merge(kHp, {{"HSA_DISABLE_FRAGMENT_ALLOCATOR", "1"}, {"STRIX_GLM_PF_PIPELINE", "2"},
                              {"STRIX_DIST_ASYNC_START", "1"}}));  // = the receipt's optimization env + the prefill overlap
    // an explicit value is never overwritten
    const auto expl = resolve(cat(kDeploy, kDecodeWork), {{"STRIX_DIST_CRC_PARALLEL", "0"}});
    Env set2;
    rd::apply_env(expl, [&](const char* k, const char* v) { set2[k] = v; });
    CHECK(set2.count("STRIX_DIST_CRC_PARALLEL") == 0);
}

void test_item_off_switches() {
    const Args plain = cat(kDeploy, kDecodeWork);
    const auto def = resolve(plain, {});
    for (const auto& it : rd::env_items()) {
        const auto e = resolve(plain, {{it.env, it.off}});
        CHECK(e.env.at(it.env).value == it.off && e.env.at(it.env).source == "explicit");
        for (const auto& other : rd::env_items())
            if (std::string(other.env) != it.env) CHECK(e.env.at(other.env).value == def.env.at(other.env).value);
        CHECK(e.identity.value == "r1b" && e.prefill == "gemv" && e.mtp.value == "0" && e.decode_xch.value == "xfer");
        Env set;
        rd::apply_env(e, [&](const char* k, const char* v) { set[k] = v; });
        CHECK(set.count(it.env) == 0);  // the process keeps the user's off value
    }
    // route items: flag and environment forms
    CHECK(resolve(cat(plain, {"--identity", "r1"}), {}).identity.value == "r1");
    CHECK(resolve(plain, {{"STRIX_GLM_IDENTITY", "r1"}}).identity.value == "r1");
    CHECK(resolve(cat(plain, {"--identity", "r1"}), {}).prefill == "gemv");  // prefill gemv (requested auto)
    CHECK(resolve(cat(plain, {"--identity", "r1b-all"}), {}).prefill == "ref");
    CHECK(resolve(cat(plain, {"--prefill", "ref"}), {}).prefill == "ref");
    CHECK(resolve(plain, {{"STRIX_GLM_PREFILL", "ref"}}).prefill == "ref");  // explicit reference prefill
    CHECK(resolve(cat(plain, {"--prefill", "gemv"}), {{"STRIX_GLM_PREFILL", "ref"}}).prefill == "gemv");  // flag wins
    const auto m1 = resolve(cat(plain, {"--mtp", "1"}), {});
    CHECK(m1.mtp.value == "1" && m1.prefill == "ref");
    CHECK(resolve(cat(plain, {"--decode-xch", "comm"}), {}).decode_xch.value == "comm");
    CHECK(resolve(plain, {{"STRIX_GLM_DECODE_XCH", "comm"}}).decode_xch.value == "comm");
    // auto-prefill exclusions (unchanged rules)
    CHECK(resolve(cat(plain, {"--spec", "1"}), {}).prefill == "ref");
    CHECK(resolve(cat(plain, {"--idcheck", "32"}), {}).prefill == "ref");
    CHECK(resolve(cat(plain, {"--aba", "96"}), {}).prefill == "ref");
    CHECK(resolve(cat(kDeploy, {"--mem-plan"}), {}).prefill == "ref");
    CHECK(resolve(cat(kDeploy, {"--g3", "t.u32", "--layers", "6"}), {}).prefill == "ref");
    CHECK(resolve(plain, {{"STRIX_GLM_SMOKE_HEAD", "1"}}).prefill == "ref");
}

void test_baseline() {
    const Args plain = cat(kDeploy, kDecodeWork);
    for (int form = 0; form < 2; ++form) {
        const auto b = form == 0 ? resolve(cat(plain, {"--baseline"}), {}) : resolve(plain, {{"STRIX_GLM_BASELINE", "1"}});
        CHECK(b.baseline);
        CHECK(b.identity.value == "r1" && b.mtp.value == "1" && b.prefill == "ref" && b.prefill_requested.value == "ref");
        CHECK(b.decode_xch.value == "comm");
        for (const auto& it : rd::env_items()) CHECK(b.env.at(it.env).value == (it.in_baseline ? it.off : it.on));
        Env set;
        rd::apply_env(b, [&](const char* k, const char* v) { set[k] = v; });
        for (const auto& it : rd::env_items()) {
            if (it.in_baseline) CHECK(set.count(it.env) == 1 && set[it.env] == it.off);
            else CHECK(set.count(it.env) == 0);
        }
    }
    CHECK(!resolve(plain, {{"STRIX_GLM_BASELINE", "0"}}).baseline);
    // explicit settings win under the baseline
    const auto bx = resolve(cat(plain, {"--baseline", "--identity", "r1b", "--mtp", "0", "--decode-xch", "xfer"}),
                            {{"STRIX_GLM_PREFILL", "auto"}, {"STRIX_GLM_HC_PRE_V", "7"}});
    CHECK(bx.identity.value == "r1b" && bx.mtp.value == "0" && bx.decode_xch.value == "xfer" && bx.prefill == "gemv");
    CHECK(bx.env.at("STRIX_GLM_HC_PRE_V").value == "7" && bx.env.at("STRIX_GLM_POST").value == "ref");
}

void test_scan_and_print() {
    // value-taking options never shadow flags: a prompt file named like an option is a value
    const auto r = rd::scan({"--prompt", "--baseline", "--world", "1", "--print-effective-config"});
    CHECK(!r.baseline_flag && r.world == 1 && r.print);
    const auto s = rd::print(resolve(cat(kDeploy, kDecodeWork), {}));
    CHECK(s.find("effective baseline=0\n") != std::string::npos);
    CHECK(s.find("effective prefill=gemv (requested auto, default)\n") != std::string::npos);
    CHECK(s.find("effective identity=r1b (default)\n") != std::string::npos);
    CHECK(s.find("effective env STRIX_DIST_CRC_PARALLEL=1 (default, item hp-crc-parallel)\n") != std::string::npos);
}

}  // namespace

int main() {
    test_plain_equals_receipt();
    test_apply_env_default();
    test_item_off_switches();
    test_baseline();
    test_scan_and_print();
    std::printf("glm rank defaults cpu tests: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
