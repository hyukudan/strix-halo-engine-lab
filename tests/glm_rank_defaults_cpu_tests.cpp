// strix-glm5-rank defaults, CPU tests (no GPU, no peer): the plain default invocation must resolve to exactly the
// measured production receipt (route + flags + env), and every off switch / the baseline master switch must restore
// the old path while explicit settings win.
//   measured decode configuration: decode 64 + R8 batch, qualification passed
//   measured prefill configuration: hp host path, exact TTFT 1K/8K
#include "glm/load/glm_rank_defaults.hpp"
#include "glm_rank_startup.hpp"
#include "glm/spec/glm_spec_merge.hpp"
#include "glm/engine/glm_mem.hpp"

#include <cstdio>
#include <barrier>
#include <cstring>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <limits>
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
    d = diff("split-frame", a.split_frame.value, b.split_frame.value) || d;
    d = diff("spec-arbiter", a.spec_arbiter ? "1" : "0", b.spec_arbiter ? "1" : "0") || d;
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
    CHECK(resolve(cat(cat(kDeploy, kDecodeWork), {"--mtp", "1", "--mtp-spec", "0"}), {}).prefill == "ref");
    // shared opt-out: --no-mtp is plain decode, like --mtp-spec 0
    CHECK(resolve(cat(cat(kDeploy, kDecodeWork), {"--no-mtp"}), {}).mtp_spec.value == "0");
    // Host-independent candidate; startup memory admission and the rank minimum may reduce it.
    CHECK(plain.max_rows.value == "8192" && plain.max_rows.source == "auto-8192");
    CHECK(resolve(cat(kDeploy, kTtftWork), {}).max_rows.source == "explicit");
    CHECK(resolve(cat(cat(kDeploy, kDecodeWork), {"--max-rows", "1024"}), {}).max_rows.value == "1024");
    CHECK(resolve(cat(cat(kDeploy, kDecodeWork), {"--baseline"}), {}).max_rows.value == "1024");
}

void test_apply_env_default() {
    CHECK(rd::env_items().size() == 44);  // one item per environment selector
    const auto normal = resolve({}, {});
    const auto baseline = resolve({"--baseline"}, {});
    for (const auto& [key, off] : Env{{"STRIX_GLM_PF_SEQPAR", "0"}, {"STRIX_GLM_PF_AUX_COMBINE", "0"},
                                     {"STRIX_GLM_PF_IN_TOKEN", "0"},
                                     {"STRIX_GLM_PF_SCRATCH", "full"}, {"STRIX_GLM_PF_ATTN_QWIN", "0"}}) {
        CHECK(baseline.env.at(key).value == off);
        CHECK(baseline.env.at(key).source == "baseline");
        CHECK(resolve({"--baseline"}, {{key, normal.env.at(key).value}}).env.at(key).value == normal.env.at(key).value);
    }
    CHECK(rd::print(normal).find("effective pf-seqpar=on") != std::string::npos);
    CHECK(rd::print(baseline).find("effective pf-seqpar=off") != std::string::npos);
    CHECK(rd::print(baseline).find("capacity-layout=replicated/full/qwin=0") != std::string::npos);
    for (const Args& args : {Args{"--prefill", "wmma"}, Args{"--max-rows", "1"}, Args{"--world", "1"}})
        CHECK(rd::print(resolve(args, {})).find("effective pf-seqpar=declined") != std::string::npos);
    const auto routed = resolve({}, {{"STRIX_GLM_PF_SCRATCH", "routed"}, {"STRIX_GLM_PF_ATTN_QWIN", "2048"}});
    CHECK(rd::print(routed).find("capacity-layout=replicated/routed/qwin=2048") != std::string::npos);
    CHECK(rd::print(routed).find("capacity_layout_reason=first-admitted-query-window") != std::string::npos);
    CHECK(rd::print(routed).find("admission and rank vote pending") != std::string::npos);
    CHECK(resolve({}, {}).env.at("STRIX_GLM_PF_DENSE_ROWS").value == "4");
    CHECK(resolve({"--baseline"}, {}).env.at("STRIX_GLM_PF_DENSE_ROWS").value == "1");
    // Dense rows v4 form selector: governed like the version selectors (default 0 = automatic, 0..7 accepted,
    // anything else refused by the resolver, before any device work).
    CHECK(resolve({}, {}).env.at("STRIX_GLM_PF_DENSE_FORM").value == "0");
    CHECK(resolve({"--baseline"}, {}).env.at("STRIX_GLM_PF_DENSE_FORM").value == "0");
    for (const char* v : {"0", "1", "4", "7"}) {
        const auto e = resolve({}, {{"STRIX_GLM_PF_DENSE_ROWS", "4"}, {"STRIX_GLM_PF_DENSE_FORM", v}});
        CHECK(e.env.at("STRIX_GLM_PF_DENSE_FORM").value == v && e.env.at("STRIX_GLM_PF_DENSE_FORM").source == "explicit");
        CHECK(rd::print(e).find(std::string("effective env STRIX_GLM_PF_DENSE_FORM=") + v + " (explicit, item pf-dense-form-auto; resolved form " + v) != std::string::npos);
    }
    for (const char* v : {"8", "-1", "07", "auto", "1 "}) {
        bool refused = false;
        try { (void)resolve({}, {{"STRIX_GLM_PF_DENSE_ROWS", "4"}, {"STRIX_GLM_PF_DENSE_FORM", v}}); }
        catch (const std::invalid_argument&) { refused = true; }
        CHECK(refused);
    }
    CHECK(rd::print(resolve({}, {})).find("effective env STRIX_GLM_PF_DENSE_FORM=0 (default, item pf-dense-form-auto; resolved form 0 (automatic per shape))\n") != std::string::npos);
    for (const char* v : {"1", "2", "3"})
        CHECK(resolve({}, {{"STRIX_GLM_PF_DENSE_ROWS", v}}).env.at("STRIX_GLM_PF_DENSE_ROWS").source == "explicit");
    CHECK(resolve({}, {}).env.at("STRIX_GLM_PF_MIXNORM").value == "1");
    CHECK(resolve({"--baseline"}, {}).env.at("STRIX_GLM_PF_MIXNORM").value == "0");
    for (const char* v : {"0", "1"})
        CHECK(resolve({}, {{"STRIX_GLM_PF_MIXNORM", v}}).env.at("STRIX_GLM_PF_MIXNORM").source == "explicit");
    CHECK(resolve({}, {}).env.at("STRIX_GLM_PF_MLA_GEMM").value == "3");
    CHECK(resolve({"--baseline"}, {}).env.at("STRIX_GLM_PF_MLA_GEMM").value == "2");
    for (const char* v : {"1", "2", "3"})
        CHECK(resolve({}, {{"STRIX_GLM_PF_MLA_GEMM", v}}).env.at("STRIX_GLM_PF_MLA_GEMM").source == "explicit");
    CHECK(resolve({}, {}).env.at("STRIX_GLM_PF_HC_EXPAND_BW").value == "1");
    CHECK(resolve({"--baseline"}, {}).env.at("STRIX_GLM_PF_HC_EXPAND_BW").value == "0");
    for (const char* v : {"0", "1"})
        CHECK(resolve({}, {{"STRIX_GLM_PF_HC_EXPAND_BW", v}}).env.at("STRIX_GLM_PF_HC_EXPAND_BW").source == "explicit");
    CHECK(resolve({}, {}).env.at("STRIX_GLM_PF_SUMEXP").value == "1");
    CHECK(resolve({"--baseline"}, {}).env.at("STRIX_GLM_PF_SUMEXP").value == "0");
    for (const char* v : {"0", "1"})
        CHECK(resolve({}, {{"STRIX_GLM_PF_SUMEXP", v}}).env.at("STRIX_GLM_PF_SUMEXP").source == "explicit");
    CHECK(resolve(cat(kDeploy, kDecodeWork), {}).env.at("STRIX_GLM_SPEC_XCH").value == "xfer");
    CHECK(resolve({"--baseline"}, {}).env.at("STRIX_GLM_SPEC_XCH").value == "comm");
    CHECK(resolve({}, {{"STRIX_GLM_SPEC_XCH", "comm"}}).env.at("STRIX_GLM_SPEC_XCH").source == "explicit");
    // shared Arbiter binding: default on, baseline off, explicit 0 off / 1 on, anything else refused at startup
    CHECK(resolve(cat(kDeploy, kDecodeWork), {}).spec_arbiter);
    CHECK(!resolve(cat(kDeploy, kDecodeWork), {{"STRIX_SPEC_ARBITER", "0"}}).spec_arbiter);
    CHECK(!resolve({"--baseline"}, {}).spec_arbiter);
    CHECK(resolve(cat(kDeploy, kDecodeWork), {{"STRIX_SPEC_ARBITER", "1"}}).spec_arbiter);
    CHECK(resolve({"--baseline"}, {{"STRIX_SPEC_ARBITER", "1"}}).spec_arbiter);
    CHECK(rd::print(resolve({}, {{"STRIX_SPEC_ARBITER", "1"}})).find("STRIX_SPEC_ARBITER=1") != std::string::npos);
    {
        bool threw = false;
        try { (void)resolve({}, {{"STRIX_SPEC_ARBITER", "2"}}); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }
    const auto plain = resolve(cat(kDeploy, kDecodeWork), {});
    Env set;
    rd::apply_env(plain, [&](const char* k, const char* v) { set[k] = v; });
    // the app sets exactly the items whose code default is the old path; engine defaults are reported only
    for (const auto& it : rd::env_items()) {
        if (it.app_sets) CHECK(set.count(it.env) == 1 && set[it.env] == it.on);
        else CHECK(set.count(it.env) == 0);
    }
    CHECK(set == merge(kHp, {{"HSA_DISABLE_FRAGMENT_ALLOCATOR", "1"}, {"STRIX_GLM_PF_PIPELINE", "2"},
                              {"STRIX_DIST_ASYNC_START", "1"}, {"STRIX_GLM_SPEC_KDA_TREE", "2"},
                              {"STRIX_GLM_PF_EXACT_X", "auto"}, {"STRIX_GLM_PF_EXACTMOE_V", "auto"},
                              {"STRIX_GLM_HC_NORM", "1"}}));
    // = the receipt's optimization env + the prefill overlap + the two-barrier KDA verify tree + the fused HC norm
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
    const auto m1 = resolve(cat(plain, {"--mtp", "1", "--mtp-spec", "0"}), {});
    CHECK(m1.mtp.value == "1" && m1.prefill == "ref");
    CHECK(resolve(cat(plain, {"--decode-xch", "comm"}), {}).decode_xch.value == "comm");
    CHECK(resolve(plain, {{"STRIX_GLM_DECODE_XCH", "comm"}}).decode_xch.value == "comm");
    {   // glm_split_frame (xport/split_frame.hpp): pending route stays OFF, one resolution for CLI and serve
        const auto sf = resolve(plain, {});
        CHECK(sf.split_frame.source == "default" && sf.split_policy.mode == strix::xport::SplitFramePolicy::Mode::off);
        CHECK(!sf.split_policy.on(1) && !sf.split_policy.on(3) && !sf.split_policy.on(4) && !sf.split_policy.on(8));
        CHECK(resolve(plain, {{"STRIX_XPORT_SPLIT_ROWS", "0"}}).split_policy.mode == strix::xport::SplitFramePolicy::Mode::off);
        const auto forced = resolve(plain, {{"STRIX_XPORT_SPLIT_ROWS", "1"}});
        CHECK(forced.split_frame.source == "explicit" && forced.split_policy.on(2) && !forced.split_policy.on(1));
        CHECK(resolve(cat(plain, {"--baseline"}), {}).split_policy.mode == strix::xport::SplitFramePolicy::Mode::off);
        // The engine splits only with the publishing producers on the fused boundary: the receipt follows that gate.
        for (const char* gate : {"STRIX_GLM_XB_PUBLISH", "STRIX_GLM_BOUNDARY"}) {
            const auto gated = resolve(plain, {{"STRIX_XPORT_SPLIT_ROWS", "1"}, {gate, "0"}});
            CHECK(gated.split_policy.mode == strix::xport::SplitFramePolicy::Mode::off && !gated.split_policy.on(2));
            CHECK(gated.split_frame.source == "needs-boundary-publish");
        }
        // Producer publish: engine built-in default (reported, not exported), baseline restores the copy + publish launch.
        CHECK(sf.env.at("STRIX_GLM_XB_PUBLISH").value == "1" && sf.env.at("STRIX_GLM_XB_PUBLISH").source == "default");
        CHECK(resolve(cat(plain, {"--baseline"}), {}).env.at("STRIX_GLM_XB_PUBLISH").value == "0");
        bool threw = false;
        try { (void)resolve(plain, {{"STRIX_XPORT_SPLIT_ROWS", "maybe"}}); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
        const rd::EnvGet none = [](const char*) -> const char* { return nullptr; };  // the serve resolver: same policy for the same request
        const auto served = rd::resolve_serve(plain, false, none);
        CHECK(served.split_frame.value == sf.split_frame.value && served.split_policy.mode == sf.split_policy.mode &&
              served.split_policy.crossover == sf.split_policy.crossover);
    }
    // auto-prefill exclusions (unchanged rules)
    CHECK(resolve(cat(plain, {"--spec", "1"}), {}).prefill == "ref");
    CHECK(resolve(cat(plain, {"--idcheck", "32"}), {}).prefill == "ref");
    CHECK(resolve(cat(plain, {"--aba", "96"}), {}).prefill == "ref");
    CHECK(resolve(cat(kDeploy, {"--mem-plan"}), {}).prefill == "gemv");
    CHECK(resolve(cat(kDeploy, {"--g3", "t.u32", "--layers", "6"}), {}).prefill == "ref");
    CHECK(resolve(plain, {{"STRIX_GLM_SMOKE_HEAD", "1"}}).prefill == "ref");
}

void test_baseline() {
    const Args plain = cat(kDeploy, kDecodeWork);
    for (int form = 0; form < 2; ++form) {
        const auto b = form == 0 ? resolve(cat(plain, {"--baseline"}), {}) : resolve(plain, {{"STRIX_GLM_BASELINE", "1"}});
        CHECK(b.baseline);
        CHECK(b.identity.value == "r1" && b.mtp.value == "0" && b.prefill == "ref" && b.prefill_requested.value == "ref");
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
    // exact prefill X2 core + MoE auto: literal defaults and off values (not only item-derived)
    const auto d = resolve(plain, {});
    CHECK(d.env.at("STRIX_GLM_PF_EXACT_X").value == "auto" && d.env.at("STRIX_GLM_PF_EXACTMOE_V").value == "auto");
    const auto bb = resolve(cat(plain, {"--baseline"}), {});
    CHECK(bb.env.at("STRIX_GLM_PF_EXACT_X").value == "0" && bb.env.at("STRIX_GLM_PF_EXACTMOE_V").value == "1");
    const auto b3 = resolve(cat(plain, {"--baseline"}), {{"STRIX_GLM_PF_EXACTMOE_V", "3"}});
    CHECK(b3.env.at("STRIX_GLM_PF_EXACTMOE_V").value == "3" && b3.env.at("STRIX_GLM_PF_EXACTMOE_V").source == "explicit");
    CHECK(b3.env.at("STRIX_GLM_PF_EXACT_X").value == "0");
    const auto p3 = resolve(plain, {{"STRIX_GLM_PF_EXACTMOE_V", "3"}, {"STRIX_GLM_PF_EXACT_X", "0"}});
    CHECK(p3.env.at("STRIX_GLM_PF_EXACTMOE_V").value == "3" && p3.env.at("STRIX_GLM_PF_EXACT_X").value == "0");
}

void test_mtp_default() {
    const auto dir = std::filesystem::temp_directory_path() /
        ("glm-mtp-header-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(dir);
    struct Cleanup {
        std::filesystem::path dir;
        ~Cleanup() {
            for (const auto& e : std::filesystem::directory_iterator(dir)) std::filesystem::remove(e.path());
            std::filesystem::remove(dir);
        }
    } cleanup{dir};
    CHECK(!rd::probe_model_mtp(dir).mtp_present);
    const auto names = rd::mtp_tensor_names();
    const auto roles = rd::mtp_tensor_roles();
    auto write_shard = [&](const char* file, std::size_t first, std::size_t last, const std::string& bad = "") {
        std::string json = "{";
        std::uint64_t offset = 0;
        for (std::size_t i = first; i < last; ++i) {
            if (i != first) json += ',';
            auto shape = roles[i].shape;
            auto dtype = roles[i].dtype;
            const bool corrupt = i == first;
            if (corrupt && bad == "norm") shape = {1};
            if (i == 2 && bad == "u8-norm") dtype = "U8";
            if (i == 5 && bad == "qa-norm") shape = {1};
            if (roles[i].trellis) shape[2] = 16 * (1 + i % 8);  // Exercise all integer bit widths.
            if (i == 16 && bad == "trellis-in") ++shape[0];
            if (i == 16 && bad == "trellis-bits") shape[2] = 72;
            std::uint64_t bytes = dtype == "U8" ? 1 : dtype == "F32" ? 4 : 2;
            std::string dims;
            for (const auto dim : shape) {
                bytes *= dim;
                if (!dims.empty()) dims += ',';
                dims += std::to_string(dim);
            }
            const auto end = offset + bytes;
            json += "\"" + names[i] + "\":{";
            if (!corrupt || bad != "dtype") json += "\"dtype\":\"" + dtype + "\",";
            json += "\"shape\":[" + dims + "],";
            json += "\"data_offsets\":[" + std::to_string(offset) + "," +
                    std::to_string(end + (corrupt && bad == "bytes" ? 1 : 0)) + "]}";
            offset = end;
        }
        json += '}';
        std::ofstream f(dir / file, std::ios::binary);
        const auto n = static_cast<std::uint64_t>(json.size());
        for (unsigned i = 0; i < 8; ++i) f.put(static_cast<char>((n >> (8 * i)) & 255));
        f.write(json.data(), static_cast<std::streamsize>(json.size()));
        // Payload bytes exist, but the probe reads only headers and stats the file size.
        // Seek over the large EXL3 payload: CPU tests inspect metadata, never its contents.
        if (offset != 0) {
            f.seekp(static_cast<std::streamoff>(offset - 1), std::ios::cur);
            f.put('\0');
        }
    };
    auto write = [&](bool complete) { write_shard("mtp.safetensors", 0, names.size() - (complete ? 0 : 1)); };
    write(false);
    auto facts = rd::probe_model_mtp(dir);
    CHECK(facts.mtp_present && !facts.mtp_complete);
    for (const auto& bad : {"norm", "bytes", "dtype", "u8-norm", "qa-norm", "trellis-in", "trellis-bits"}) {
        write_shard("mtp.safetensors", 0, names.size(), bad);
        const auto invalid = rd::probe_model_mtp(dir);
        CHECK(invalid.mtp_present && !invalid.mtp_complete);
        CHECK(invalid.reason.find(bad == std::string("u8-norm") ? "role dtype" :
                                  (bad == std::string("norm") || bad == std::string("qa-norm") ||
                                   bad == std::string("trellis-in") || bad == std::string("trellis-bits")) ? "role shape" :
                                  bad == std::string("bytes") ? "byte count mismatch" : "missing dtype") != std::string::npos);
    }
    // Indexed and unindexed shards form one inventory, including when layer 45 is entirely indexed.
    {
        std::ofstream index(dir / "model.safetensors.index.json");
        index << "{\"weight_map\":{\"" << names[0] << "\":\"indexed.safetensors\"}}";
    }
    const auto split = names.size() / 2;
    write_shard("indexed.safetensors", 0, split);
    write_shard("mtp.safetensors", split, names.size());
    const auto split_facts = rd::probe_model_mtp(dir);
    CHECK(split_facts.mtp_complete);
    CHECK((split_facts.mtp_files == std::vector<std::filesystem::path>{dir / "indexed.safetensors", dir / "mtp.safetensors"}));
    write_shard("mtp.safetensors", split - 1, names.size());
    facts = rd::probe_model_mtp(dir);
    CHECK(!facts.mtp_complete && facts.reason.find("tensor defined twice") != std::string::npos);
    write_shard("indexed.safetensors", 0, names.size());
    std::filesystem::remove(dir / "mtp.safetensors");
    facts = rd::probe_model_mtp(dir);
    CHECK(facts.mtp_present && facts.mtp_complete);
    std::filesystem::remove(dir / "indexed.safetensors");
    std::filesystem::remove(dir / "model.safetensors.index.json");
    write(true);
    facts = rd::probe_model_mtp(dir);
    CHECK(facts.mtp_present && facts.mtp_complete);
    // Discovery records every role tensor's payload span (role order, absolute offsets), still without reading payloads.
    CHECK(facts.mtp_spans.size() == roles.size() && facts.mtp_spans.front().name == names.front());
    CHECK(facts.mtp_spans.front().file == dir / "mtp.safetensors" && facts.mtp_spans.front().offset > 8);
    CHECK(facts.mtp_spans[1].offset == facts.mtp_spans[0].offset + facts.mtp_spans[0].bytes);
    CHECK(facts.mtp_digest.empty());
    auto resolve_head = [&](const Args& extra, const Env& env = Env{}) {
        return rd::resolve(rd::scan(cat(cat(kDeploy, kDecodeWork), extra)), getter(env), facts);
    };
    const auto on = resolve_head({});
    CHECK(on.mtp.value == "1" && on.spec.value == "1" && on.mtp_spec.value == "1");
    CHECK(on.prefill == "gemv" && on.kmax == 3 && on.adaptive);
    CHECK(rd::print(on).find("effective mtp-bootstrap=chunked") != std::string::npos);
    // pf_wmma per prompt (glm_pair_profile_table.hpp): the FAST band row 2,049..8,192 and the TURBO literal row at
    // 31,258, both for the measured pair execution only (8,192-row chunk, one lane, routed scratch).
    {
        struct Work {
            bool pair = true;
            std::uint64_t prompt = 8192, capacity = 36864;
            unsigned rows = rd::kTurboPrefillRows, lanes = 1;
        };
        const auto band = [&](const char* profile, Work w, const Args& extra = {}, const Env& env = Env{}) {
            const auto request = rd::scan(cat(cat(cat(kDeploy, kDecodeWork), {"--profile", profile}), extra));
            const auto e = rd::resolve(request, getter(env), facts);
            const rd::PrefillWork work{w.pair, w.prompt, w.capacity, w.rows, w.lanes, rd::kTurboPrefillGenerated, true};
            return rd::select_prefill_wmma(request, e, getter(env), work);
        };
        const auto prompt = [](std::uint64_t n) { Work w; w.prompt = n; return w; };
        CHECK(!band("fast", prompt(1020)));
        CHECK(band("fast", prompt(1024)));
        CHECK(band("fast", prompt(2048)));
        CHECK(band("fast", prompt(2049)));
        CHECK(band("fast", prompt(8192)));
        CHECK(!band("fast", prompt(8193)));
        CHECK(!band("fast", prompt(31258)));
        CHECK(!band("exact", prompt(8192)));
        CHECK(band("turbo", prompt(8192)));
        CHECK(band("turbo", prompt(31258)));
        CHECK(!band("turbo", prompt(32768)));
        CHECK(!band("turbo", prompt(16384)));
        Work w = prompt(8192);
        w.pair = false;  // the pair vote folds device, link and layout: any mismatch declines
        CHECK(!band("fast", w));
        w = prompt(8192);
        w.rows = 4096;
        CHECK(!band("fast", w));
        w = prompt(8192);
        w.lanes = 2;
        CHECK(!band("fast", w));
        CHECK(!band("fast", prompt(8192), {"--baseline"}));
        CHECK(!band("fast", prompt(8192), {}, {{"STRIX_GLM_PF_WMMA", "0"}}));
    }
    const auto off = resolve_head({"--mtp", "0"});
    std::string why;
    CHECK(same_values(off, resolve(cat(kReceiptD, kDecodeWork), merge(kReceiptEnv, kHp)), &why));
    CHECK(off.mtp_spec.value == "0" && off.spec.value == "0");
    CHECK(resolve_head({"--mtp-spec", "0"}).mtp.value == "0");
    CHECK(resolve_head({"--mtp-spec", "0", "--mtp", "1"}).mtp.value == "1");
    CHECK(resolve_head({"--baseline"}).mtp.value == "0");
    CHECK(resolve_head({"--baseline"}).mtp_spec.value == "0");
    CHECK(resolve_head({"--baseline", "--mtp-spec", "1"}).mtp_spec.value == "1");
    CHECK(resolve_head({}, {{"STRIX_GLM_MTP_SPEC", "0"}}).mtp_spec.value == "0");
    CHECK(resolve_head({"--mtp-spec", "1"}, {{"STRIX_GLM_MTP_SPEC", "0"}}).mtp_spec.value == "1");
    CHECK(resolve_head({"--mtp-spec", "0"}, {{"STRIX_GLM_MTP_SPEC", "1"}}).mtp_spec.value == "0");
    CHECK(resolve_head({"--mtp", "0", "--mtp-spec", "1"}).mtp_spec.value == "0");
    CHECK(resolve_head({"--mtp-k", "2", "--mtp-adaptive", "0"}).kmax == 2);
    CHECK(!resolve_head({"--mtp-adaptive", "0"}).adaptive);
    CHECK(resolve_head({"--mtp-prefill-gate", "p.u32"}).prefill == "ref");
    CHECK(resolve_head({"--mtp-prefill-gate", "p.u32"}).mtp_spec.value == "0");
    CHECK(rd::resolve(rd::scan({"--batch", "8"}), getter({}), facts).mtp_spec.value == "0");
    CHECK(rd::resolve(rd::scan({"--decode", "1"}), getter({}), {}).mtp_spec.value == "0");
    const rd::ModelFacts incomplete{true, false, "incomplete"};
    CHECK(rd::resolve(rd::scan({"--decode", "1"}), getter({}), incomplete).mtp.value == "0");
    CHECK(rd::resolve(rd::scan({"--decode", "1", "--mtp", "1"}), getter({}), incomplete).mtp.value == "0");
    CHECK(resolve_head({"--xch", "xfer"}).mtp_spec.value == "0");
    CHECK(resolve_head({"--xch", "xfer", "--mtp-spec", "1"}).mtp_spec.value == "0");
    CHECK(resolve_head({"--prefill", "wmma"}).mtp_spec.value == "0");
    CHECK(resolve_head({"--identity", "r1b-all"}).mtp_spec.value == "0");
    // Both local resolutions must enter the eligibility merge despite different head inventories.
    CHECK(on.mtp_spec_candidate && rd::resolve(rd::scan(cat(kDeploy, kDecodeWork)), getter({}), incomplete).mtp_spec_candidate);
    { std::ofstream f(dir / "mtp.safetensors", std::ios::binary); f << "bad"; }
    CHECK(!rd::probe_model_mtp(dir).mtp_complete);
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
    CHECK(s.find("effective env STRIX_GLM_PF_DENSE_ROWS=4 (default, item pf-dense-rows-v4; resolved v4)\n") != std::string::npos);
    CHECK(s.find("effective env STRIX_GLM_PF_MIXNORM=1 (default, item pf-mixnorm)\n") != std::string::npos);
    CHECK(s.find("effective env STRIX_GLM_PF_MLA_GEMM=3 (default, item pf-mla-gemm-v3; resolved v3)\n") != std::string::npos);
    CHECK(s.find("effective env STRIX_GLM_PF_SUMEXP=1 (default, item pf-sumexp)\n") != std::string::npos);
}

void test_max_rows_memory() {
    namespace mem = strix::glm::engine::mem;
    constexpr auto gib = mem::kGiB;
    const std::vector<mem::Term> terms = {{"physical plan", 80 * gib, 0}, {"first-forward reserve", gib, 0},
        {"pinned staging", gib, 0}, {"whole-tensor reads", 0, gib}, {"bounce", 0, gib},
        {"prefill path", 12 * gib, 0}, {"workspace and install", gib, 0}};
    mem::Snapshot snapshot;
    snapshot.meminfo_ok = snapshot.cma_free_ok = snapshot.zoneinfo_ok = snapshot.gpu = true;
    snapshot.cma_free = snapshot.cma_total = 13 * gib;
    snapshot.mem_available = 118 * gib;  // 105 GiB headroom = 97 GiB terms + 8 GiB margin
    snapshot.gtt_total = 200 * gib;
    mem::Margins margins;
    auto choose = [&](const Args& args, const Env& env, const mem::Snapshot& snap) {
        auto e = resolve(args, env);
        rd::select_max_rows(e, [&] { return mem::fits(terms, snap, margins); });
        return e;
    };
    const auto fits = choose({}, {}, snapshot);
    CHECK(fits.max_rows.value == "8192" && fits.max_rows.source == "auto-8192");
    mem::admit("candidate", terms, snapshot, margins);
    auto small_snapshot = snapshot;
    --small_snapshot.mem_available;
    const auto small = choose({}, {}, small_snapshot);
    CHECK(small.max_rows.value == "4096" && small.max_rows.source == "auto-4096-memory");
    // The old MemAvailable - CmaFree selector would accept these; the load admission refuses.
    for (unsigned failure = 0; failure < 5; ++failure) {
        auto snap = snapshot;
        auto costs = terms;
        if (failure == 0) { snap.cma_total += gib; snap.file_lru = gib; }  // file pages in CMA
        if (failure == 1) { snap.gtt_total = 96 * gib; }                  // GTT margin
        if (failure == 2) { costs[1].gtt += 1; }                        // forward reserve
        if (failure == 3) { costs[2].gtt += 1; }                        // staging
        if (failure == 4) { costs[4].host += 1; }                       // bounce
        auto e = resolve({}, {});
        rd::select_max_rows(e, [&] { return mem::fits(costs, snap, margins); });
        CHECK(e.max_rows.value == "4096");
        bool refused = false;
        try { mem::admit("candidate", costs, snap, margins); } catch (const mem::BudgetError&) { refused = true; }
        CHECK(refused);
        costs[0].gtt -= 8 * gib;  // the smaller row plan is admitted instead of failing startup
        CHECK(mem::fits(costs, snap, margins));
        mem::admit("fallback", costs, snap, margins);
    }
    CHECK(!mem::fits(terms, snapshot, margins, 102 * gib));  // later device-free guard
    auto missing = snapshot;
    missing.meminfo_ok = false;
    CHECK(choose({}, {}, missing).max_rows.value == "4096");
    missing = snapshot;
    missing.cma_free_ok = missing.zoneinfo_ok = false;
    CHECK(choose({}, {}, missing).max_rows.value == "4096");
    CHECK(!mem::fits({{"overflow", std::numeric_limits<std::uint64_t>::max(), 1}}, snapshot, margins));
    for (const auto& snap : {snapshot, small_snapshot, missing}) {
        CHECK(choose({"--max-rows", "1234"}, {}, snap).max_rows.value == "1234");
        CHECK(choose({"--baseline"}, {}, snap).max_rows.value == "1024");
        CHECK(choose({}, {{"STRIX_GLM_MAX_ROWS_AUTO", "0"}}, snap).max_rows.value == "4096");
    }
    const auto printed = rd::print(resolve({}, {}));
    CHECK(printed.find("effective max-rows=8192 (auto-8192)\n") != std::string::npos);
    CHECK(printed.find("startup-admission-and-rank-min") != std::string::npos);
    CHECK(printed.find("MemAvailable") == std::string::npos && printed.find("CmaFree") == std::string::npos);
}

// r94: the CLI pair votes the MTP drafter payload digest after the eligibility round (one extra paired round).
void test_drafter_digest_vote() {
    namespace spec = strix::glm::spec;
    auto pair = [&](const std::string& a, const std::string& b) {
        std::barrier barrier(2);
        spec::MergeRow payload[2];
        spec::MergeCtl control[2];
        std::string error[2];
        unsigned rounds[2] = {0, 0};
        const std::string digest[] = {a, b};
        auto rank = [&](unsigned r) {
            try {
                rd::agree_drafter_digest(digest[r], true, [&](auto* rows, unsigned n, auto& ctl) {
                    ++rounds[r];
                    payload[r] = rows[0];
                    control[r] = ctl;
                    barrier.arrive_and_wait();
                    spec::merge_rank_order({rows, n}, ctl, {&payload[1 - r], n}, control[1 - r], r);
                    barrier.arrive_and_wait();
                });
            } catch (const std::exception& e) { error[r] = e.what(); }
        };
        std::thread peer(rank, 1);
        rank(0);
        peer.join();
        CHECK(rounds[0] == 2 && rounds[1] == 2);
        CHECK(error[0].empty() == error[1].empty());
        return error[0].empty();
    };
    CHECK(pair("sha256:aa11", "sha256:aa11"));
    CHECK(!pair("sha256:aa11", "sha256:aa12"));
    CHECK(rd::drafter_digest_hash("sha256:aa11") != rd::drafter_digest_hash("sha256:aa12"));
    unsigned calls = 0;
    rd::agree_drafter_digest("x", false, [&](auto*, unsigned, auto&) { ++calls; });
    CHECK(calls == 0);  // no handshake or no eligible drafter: no round
}

void test_startup_agreement() {
    namespace spec = strix::glm::spec;
    using Fields = std::array<std::uint32_t, 8>;
    const Fields common = {1, 123, 4096, 64, 8192, 0, 1, 2166136261U};
    struct Frame { std::vector<unsigned char> bytes; unsigned kind; };
    auto frame = [](const spec::MergeRow* rows, unsigned n, const spec::MergeCtl& ctl) {
        Frame out{{}, ctl.kind};
        out.bytes.resize(sizeof(ctl) + n * sizeof(*rows));
        std::memcpy(out.bytes.data(), &ctl, sizeof(ctl));
        std::memcpy(out.bytes.data() + sizeof(ctl), rows, n * sizeof(*rows));
        return out;
    };
    // Fake Comm: real rank-order merge, two synchronous peers, captures every payload in sequence.
    auto pair = [&](const Fields& a, const Fields& b, bool auto0, bool auto1, bool expect_refusal, unsigned late0 = 0, unsigned late1 = 0) {
        std::barrier barrier(2);
        spec::MergeRow payload[2][8];
        spec::MergeCtl control[2];
        std::vector<Frame> wire[2];
        rd::StartupResult result[2]{};
        std::string error[2];
        const Fields fields[] = {a, b};
        const bool automatic[] = {auto0, auto1};
        const unsigned late[] = {late0, late1};
        auto rank = [&](unsigned r) {
            try {
                auto merge = [&](auto* rows, unsigned n, auto& ctl) {
                    wire[r].push_back(frame(rows, n, ctl));
                    std::copy_n(rows, n, payload[r]);
                    control[r] = ctl;
                    barrier.arrive_and_wait();
                    spec::merge_rank_order({rows, n}, ctl, {payload[1-r], n}, control[1-r], r);
                    barrier.arrive_and_wait();
                };
                result[r] = rd::startup_agreement(fields[r], automatic[r], true, merge);
                if (automatic[r]) result[r].rows = rd::recheck_rows(late[r] ? late[r] : result[r].rows, true, merge);
            } catch (const std::exception& e) { error[r] = e.what(); }
        };
        std::thread peer(rank, 1);
        rank(0);
        peer.join();
        for (unsigned r = 0; r < 2; ++r) {
            CHECK(wire[r].size() == (auto0 && auto1 && !expect_refusal ? 3U : 2U));  // fixed tags never shift
            CHECK(!error[r].empty() == expect_refusal);
            if (expect_refusal) CHECK(error[r].find("configurations or max-rows modes") != std::string::npos);
        }
        if (!expect_refusal) {
            CHECK(result[0].rows == result[1].rows);
            CHECK(result[0].rows == (auto0 ? std::min({a[2], b[2], late0 ? late0 : a[2], late1 ? late1 : b[2]}) : a[2]));
            CHECK(result[0].eligible == result[1].eligible);
        }
        return wire[0];
    };
    for (const auto& e : {resolve({}, {{"STRIX_GLM_MAX_ROWS_AUTO", "0"}}), resolve({"--max-rows", "8192"}, {}),
                          resolve({"--baseline"}, {}), resolve({"--decode", "64", "--prefill", "ref"}, {}),
                          resolve({"--decode", "64", "--identity", "r1b-all"}, {}), resolve({"--decode", "64", "--idcheck", "4"}, {})}) {
        auto fields = common;
        fields[2] = static_cast<unsigned>(std::stoul(e.max_rows.value));
        const auto wire = pair(fields, fields, rd::max_rows_auto(e), rd::max_rows_auto(e), false);
        // Literal encoding of the original eight-field eligibility message and its OR vote.
        spec::MergeRow legacy[8];
        for (unsigned i = 0; i < 8; ++i) legacy[i] = {static_cast<float>(fields[i]), fields[i], 0};
        spec::MergeCtl ctl;
        ctl.kind = spec::kEligibilityMerge;
        CHECK(wire[0].bytes == frame(legacy, 8, ctl).bytes);
        CHECK(wire[0].kind == spec::kEligibilityMerge);
        spec::MergeRow vote;
        ctl.kind = spec::kConfigurationMismatchMerge;
        CHECK(wire[1].bytes == frame(&vote, 1, ctl).bytes);
        auto different = fields;
        different[2] = fields[2] == 8192 ? 4096 : 8192;
        pair(fields, different, false, false, true);
    }
    for (unsigned a : {4096U, 8192U}) for (unsigned b : {4096U, 8192U}) {
        auto left = common, right = common;
        left[2] = a; right[2] = b;
        pair(left, right, true, true, false);
        pair(left, right, true, false, true);
        pair(left, right, false, true, true);
    }
    auto large = common;
    large[2] = 8192;
    pair(large, large, true, true, false, 4096, 0);  // later admission refuses on either rank
    pair(large, large, true, true, false, 0, 4096);
    auto ineligible = common;
    ineligible[0] = 0;
    pair(common, ineligible, false, false, false);
    // --tf on two comm ranks: config-only agreement on the fixed no-handshake rows (2 frames, rows unchanged),
    // and a FAST attention switch set on one rank only is refused on both ranks.
    {
        const auto request = rd::scan({"--world", "2", "--tf", "prompt.u32"});
        auto tf = rd::resolve(request, getter({}));
        rd::fallback_without_handshake(tf, request, getter({}));
        CHECK(!rd::max_rows_auto(tf) && tf.max_rows.value == "4096");
        auto fast = common;
        fast[1] = rd::kPfWmmaConfig | rd::kPfAttnWmmaConfig;
        fast[2] = 4096;
        pair(fast, fast, rd::max_rows_auto(tf), rd::max_rows_auto(tf), false);
        auto opt_out = fast;
        opt_out[1] = rd::kPfWmmaConfig;
        pair(fast, opt_out, false, false, true);
        auto kda = fast;  // the FAST KDA projections are on by default with the FAST route: one rank opting out is refused too
        kda[1] |= rd::kPfKdaWmmaConfig;
        pair(kda, kda, rd::max_rows_auto(tf), rd::max_rows_auto(tf), false);
        pair(kda, fast, false, false, true);
        auto mla = kda;  // the FAST MLA GEMMs (glm-pf-mla-wmma-v1, FAST default): a rank without them is refused
        mla[1] |= rd::kPfMlaWmmaConfig;
        pair(mla, mla, rd::max_rows_auto(tf), rd::max_rows_auto(tf), false);
        pair(mla, kda, false, false, true);
        auto idxproj = kda;  // the FAST indexer projections: one rank without them is refused
        idxproj[1] |= rd::kPfIdxProjWmmaConfig;
        pair(idxproj, idxproj, rd::max_rows_auto(tf), rd::max_rows_auto(tf), false);
        pair(idxproj, kda, false, false, true);
        auto router = kda;  // the FAST router product (opt-in): one rank with it and one without is refused
        router[1] |= rd::kPfRouterWmmaConfig;
        pair(router, router, rd::max_rows_auto(tf), rd::max_rows_auto(tf), false);
        pair(router, kda, false, false, true);
        CHECK(rd::kPfIdxProjWmmaConfig == (1U << 18));
        CHECK(rd::kPfRouterWmmaConfig == (1U << 20));
        auto both = idxproj;
        both[1] |= rd::kPfRouterWmmaConfig;
        pair(both, both, rd::max_rows_auto(tf), rd::max_rows_auto(tf), false);
        pair(both, idxproj, false, false, true);
        pair(both, router, false, false, true);
        CHECK((rd::kPfRouterWmmaConfig & (rd::kPfIdxProjWmmaConfig | rd::kPfKdaWmmaConfig | rd::kPfAttnWmmaConfig | rd::kPfWmmaConfig | rd::kFastI4Config |
                                          rd::kAutoRowsConfig | (3U << rd::kIdxWmmaShift))) == 0);
    }
    // External Xfer, including fixed and baseline modes, never enters this host handshake.
    for (const auto& extra : {Args{}, Args{"--baseline"}, Args{"--max-rows", "8192"}}) {
        auto e = resolve(cat({"--xch", "xfer"}, extra), {});
        unsigned exchanges = 0;
        rd::select_max_rows(e, [&] { ++exchanges; return true; });
        const bool enabled = rd::has_startup_handshake(2, "xfer", true, false);
        rd::startup_agreement(common, rd::max_rows_auto(e), enabled, [&](auto*, unsigned, auto&) { ++exchanges; });
        CHECK(exchanges == 0);
        if (extra.empty()) CHECK(e.max_rows.value == "4096" && e.max_rows.source == "auto-4096-xfer");
    }
    for (const auto& args : {Args{}, Args{"--baseline"}, Args{"--max-rows", "8192"}}) {
        const auto e = resolve(cat({"--xch", "xfer"}, args), {{"STRIX_GLM_MAX_ROWS_AUTO", "0"}});
        CHECK(!rd::max_rows_auto(e));
        unsigned calls = 0;
        rd::startup_agreement(common, false, false, [&](auto*, unsigned, auto&) { ++calls; });
        CHECK(calls == 0);
    }
    auto fallen = resolve({}, {});
    rd::select_max_rows(fallen, [] { return false; });
    rd::select_max_rows(fallen, [] { return true; });
    CHECK(fallen.max_rows.value == "4096");  // no promotion during the final admission recheck
    unsigned exchanges = 0;
    CHECK(rd::recheck_rows(4096, false, [&](auto*, unsigned, auto&) { ++exchanges; }) == 4096);
    CHECK(exchanges == 0);
    CHECK(!rd::has_startup_handshake(2, "comm", false, false));
    CHECK(!rd::has_startup_handshake(2, "comm", true, true));
    // --tf and --pf-ttft (both set reference_gate) add a config-only agreement on two comm ranks; other reference gates
    // and xfer/world 1 stay exchange-free, and the rows stay on the fixed fallback (no startup handshake).
    CHECK(rd::has_config_agreement(2, "comm", false, true, true));
    CHECK(rd::has_config_agreement(2, "comm", true, true, true));
    CHECK(rd::has_config_agreement(2, "comm", true, false, false));
    CHECK(!rd::has_config_agreement(2, "comm", true, true, false));
    CHECK(!rd::has_config_agreement(2, "xfer", false, true, true));
    CHECK(!rd::has_config_agreement(1, "comm", false, true, true));
    {
        rd::Effective pf;
        pf.max_rows = {"8192", "auto-8192"};
        const bool comm = true, handshake = rd::has_startup_handshake(2, "comm", false, true);
        CHECK(rd::has_config_agreement(2, "comm", false, true, true) && !handshake);
        rd::fallback_without_handshake(pf, comm, handshake);
        CHECK(pf.max_rows.value == "4096");
    }
}

void test_no_handshake_fallback() {
    const std::vector<Args> modes = {
        {}, {"--pf-ttft", "prompt.u32"}, {"--pf-gate", "prompt.u32"}, {"--pf-null-study", "prompt.u32"},
        {"--batch", "8"}, {"--g3", "prompt.u32"}, {"--idcheck", "4"},
        {"--ab", "1"}, {"--aba", "1"}, {"--g3-sweep"}, {"--aba-attn3", "1"}, {"--aba-xch", "1"},
        {"--g4", "8"}, {"--tf", "prompt.u32"}, {"--spec-oracle", "7"}, {"--spec-run", "7"},
        {"--state-in", "state.bin"}, {"--state-out", "state.bin"}, {"--mtp-prefill-gate", "prompt.u32"}
    };
    for (const auto& mode : modes) for (bool decode : {false, true}) {
        auto args = cat({"--world", "2"}, mode);
        if (decode) args = cat(args, {"--decode", "64"});
        const auto request = rd::scan(args);
        const bool handshake = rd::has_startup_handshake(request.world, request.xch, request.decode, request.reference_gate);
        for (unsigned rank : {0U, 1U}) {
            auto e = resolve(cat(args, {"--rank", std::to_string(rank)}), {});
            rd::fallback_without_handshake(e, true, handshake);
            if (handshake) {
                if (e.prefill == "ref") {
                    CHECK(e.max_rows.value == "4096" && e.max_rows.source == "auto-4096-ref");
                    CHECK(!rd::max_rows_auto(e));
                } else {
                    CHECK(e.max_rows.value == "8192" && e.max_rows.source == "auto-8192");
                    CHECK(rd::max_rows_auto(e));
                }
                continue;
            }
            CHECK(e.max_rows.value == "4096" && e.max_rows.source == "auto-4096-no-handshake");
            CHECK(!rd::max_rows_auto(e));  // Engine::Options::auto_rows disables admission/recheck rounds
            CHECK(rd::print(e).find("effective max-rows=4096 (auto-4096-no-handshake)") != std::string::npos);
            unsigned calls = 0;
            rd::select_max_rows(e, [&] { ++calls; return true; });
            rd::agree_max_rows(e, 8192);
            const auto rows = static_cast<unsigned>(std::stoul(e.max_rows.value));
            const auto result = rd::startup_agreement({0, 0, rows, 0, 0, 0, 0, 0}, rd::max_rows_auto(e), handshake,
                [&](auto*, unsigned, auto&) { ++calls; });
            CHECK(result.rows == 4096);
            CHECK(rd::recheck_rows(rows, rd::max_rows_auto(e), [&](auto*, unsigned, auto&) { ++calls; }) == 4096);
            CHECK(calls == 0);  // neither a memory probe nor a configuration exchange
            CHECK(e.max_rows.source == "auto-4096-no-handshake");
        }
    }
    for (const auto& e : {resolve({}, {{"STRIX_GLM_MAX_ROWS_AUTO", "0"}}), resolve({"--max-rows", "8192"}, {}),
                          resolve({"--baseline"}, {}), resolve({"--xch", "xfer"}, {})}) {
        auto kept = e;
        rd::fallback_without_handshake(kept, true, false);
        CHECK(rd::print(kept) == rd::print(e));
    }
    auto local = resolve({}, {});
    rd::fallback_without_handshake(local, false, false);  // single rank or load-only: no Comm
    CHECK(local.max_rows.value == "8192" && rd::max_rows_auto(local));
}

void test_effective_config_rows() {
    auto inspect = [](const Args& args, const Env& env = Env{}) {
        const auto request = rd::scan(cat(args, {"--print-effective-config"}));
        auto e = rd::resolve(request, getter(env));
        rd::fallback_without_handshake(e, request, getter(env));
        return rd::print(e);
    };
    for (const auto& args : {Args{"--pf-ttft", "prompt.u32"}, Args{"--pf-ttft", "prompt.u32", "--decode", "64"},
                             Args{"--batch", "8"}, Args{"--pf-gate", "prompt.u32"}}) {
        const auto printed = inspect(cat(kDeploy, args));
        CHECK(printed.find("effective max-rows=4096 (auto-4096-no-handshake)\n") != std::string::npos);
        CHECK(printed.find("max-rows-admission=") == std::string::npos);
        CHECK(inspect(cat(cat(kDeploy, args), {"--max-rows", "8192"})).find(
            "effective max-rows=8192 (explicit)\n") != std::string::npos);
    }
    // No Communicator exists in these inspection/execution routes. Use chunked prefill to
    // distinguish the no-handshake decision from the independent reference-prefill fallback.
    const Args chunked = {"--pf-ttft", "prompt.u32", "--prefill", "gemv"};
    for (const auto& printed : {inspect(cat({"--world", "1"}, chunked)),
                               inspect(cat({"--mem-plan"}, chunked)),
                               inspect(chunked, {{"STRIX_GLM_LOAD_TEST", "1"}})}) {
        CHECK(printed.find("effective max-rows=8192 (auto-8192)\n") != std::string::npos);
        CHECK(printed.find("max-rows-admission=startup-admission-and-rank-min") != std::string::npos);
    }
    CHECK(inspect({"--xch", "xfer"}).find("effective max-rows=4096 (auto-4096-xfer)\n") != std::string::npos);
}

void test_reference_prefill_rows() {
    rd::ModelFacts facts;
    facts.mtp_complete = true;
    const std::vector<std::pair<Args, Env>> routes = {
        {{"--prefill", "ref"}, {}}, {{}, {{"STRIX_GLM_PREFILL", "ref"}}},
        {{"--identity", "r1b-all"}, {}}, {{}, {{"STRIX_GLM_IDENTITY", "r1b-all"}}},
        {{"--idcheck", "4"}, {}}
    };
    for (const auto& [extra, env] : routes) for (bool complete : {false, true}) {
        const auto args = cat({"--decode", "64"}, extra);
        auto resolve_route = [&](const Args& more, const Env& overrides = Env{}) {
            return rd::resolve(rd::scan(cat(args, more)), getter(merge(env, overrides)),
                               complete ? facts : rd::ModelFacts{});
        };
        auto e = resolve_route({});
        CHECK(e.prefill == "ref");
        CHECK(e.max_rows.value == "4096" && e.max_rows.source == "auto-4096-ref");
        CHECK(!rd::max_rows_auto(e));
        rd::fallback_without_handshake(e, true, true);
        unsigned calls = 0;
        rd::select_max_rows(e, [&] { ++calls; return true; });
        rd::agree_max_rows(e, 8192);
        CHECK(rd::recheck_rows(4096, rd::max_rows_auto(e), [&](auto*, unsigned, auto&) { ++calls; }) == 4096);
        CHECK(calls == 0);  // no auto admission probe or additional row-count exchange
        const auto printed = rd::print(e);
        CHECK(printed.find("effective max-rows=4096 (auto-4096-ref)\n") != std::string::npos);
        CHECK(printed.find("max-rows-admission=") == std::string::npos);
        CHECK(resolve_route({"--max-rows", "8192"}).max_rows.source == "explicit");
        CHECK(resolve_route({"--max-rows", "8192"}).max_rows.value == "8192");
        CHECK(resolve_route({"--baseline"}).max_rows.value == "1024");
        CHECK(resolve_route({}, {{"STRIX_GLM_MAX_ROWS_AUTO", "0"}}).max_rows.source == "explicit");
        CHECK(resolve_route({"--xch", "xfer"}).max_rows.source == "auto-4096-xfer");
    }
    for (const auto& extra : {Args{}, Args{"--mtp-spec", "0"}}) {
        const auto e = rd::resolve(rd::scan(cat({"--decode", "64"}, extra)), getter({}), facts);
        CHECK(e.prefill == "gemv" && e.max_rows.value == "8192" && rd::max_rows_auto(e));
    }
}

void test_pf_variants() {
    // unknown prefill variant values fail at startup; the printer reports the resolved variant
    bool threw = false;
    try { (void)resolve(cat(kDeploy, kDecodeWork), {{"STRIX_GLM_PF_OBAR", "9"}}); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    // an explicit empty value is legal (not exported) and selects the default variant
    const auto se = rd::print(resolve(cat(kDeploy, kDecodeWork), {{"STRIX_GLM_PF_OBAR", ""}}));
    CHECK(se.find("effective env STRIX_GLM_PF_OBAR= (explicit, item pf-obar-v10; resolved v10, heads 64 runs v2)\n") != std::string::npos);
    const auto s10 = rd::print(resolve(cat(kDeploy, kDecodeWork), {}));
    CHECK(s10.find("effective env STRIX_GLM_PF_OBAR=10 (default, item pf-obar-v10; resolved v10, heads 64 runs v2)\n") != std::string::npos);
    const auto s7 = rd::print(resolve(cat(kDeploy, kDecodeWork), {{"STRIX_GLM_PF_OBAR", "4"}}));
    CHECK(s7.find("effective env STRIX_GLM_PF_OBAR=4 (explicit, item pf-obar-v10; resolved v4)\n") != std::string::npos);
    const auto sb = rd::print(resolve(cat(cat(kDeploy, kDecodeWork), {"--baseline"}), {}));
    CHECK(sb.find("STRIX_GLM_PF_OBAR=2 (baseline, item pf-obar-v10; resolved v2)\n") != std::string::npos);
}

// Every value --all-on exports for a GLM lever (data/profile/all_on.json, the rows whose engines name glm) must be a
// value the engine's own resolver accepts: the catalogue once shipped glm_sampled_mtp=1 while the resolver takes
// 0|coupled|rejection, which stopped every GLM all-on run at start-up. A refusal for another reason (a tier) is fine.
void test_all_on_values_accepted() {
    const auto path = std::filesystem::path(__FILE__).parent_path().parent_path() / "data" / "profile" / "all_on.json";
    std::ifstream in(path);
    CHECK(in.good());
    const auto field = [](const std::string& line, const std::string& key) {
        const std::string tag = "\"" + key + "\":\"";
        const auto at = line.find(tag);
        if (at == std::string::npos) return std::string();
        const auto from = at + tag.size();
        return line.substr(from, line.find('"', from) - from);
    };
    int rows = 0;
    for (std::string line; std::getline(in, line);) {
        if (line.find("\"engines\":[") == std::string::npos || line.find("\"glm\"") == std::string::npos) continue;
        const auto env = field(line, "env"), value = field(line, "value");
        if (env.empty() || value.empty()) continue;
        ++rows;
        try { (void)resolve(cat(kDeploy, kDecodeWork), {{env, value}}); }
        catch (const std::invalid_argument& x) { CHECK(std::string(x.what()).find("must be") == std::string::npos); }
    }
    CHECK(rows >= 1);
}

// Sampled MTP on the server: one typed row (off / coupled / rejection) resolved by rd::resolve for the command line and
// the server alike, printed in the effective receipt and admitted by tier there (no private switch). Written against
// names that existed before the row was resolved here: the receipt line and the refusals below fail on that tree.
void test_sampled_mtp_admission() {
    const auto line_of = [](const rd::Effective& e) {
        const auto s = rd::print(e);
        const auto at = s.find("effective serve-sampled-mtp=");
        return at == std::string::npos ? std::string() : s.substr(at, s.find('\n', at) - at);
    };
    const auto args = cat(kDeploy, kDecodeWork);
    const auto refused = [&](const std::vector<std::string>& a, std::map<std::string, std::string> env, const char* needle) {
        try { (void)resolve(a, env); } catch (const std::invalid_argument& x) { return std::string(x.what()).find(needle) != std::string::npos; }
        return false;
    };
    // default: off, builtin, in the receipt of both entry points; the rank command line names it inapplicable
    const auto def = resolve(args, {});
    CHECK(def.sampled_mtp.route == rd::SampledMtp::off && def.sampled_mtp.source == "builtin" && !def.product);
    CHECK(line_of(def).rfind("effective serve-sampled-mtp=off (builtin, row serve_sampled_mtp, identity none;", 0) == 0);
    CHECK(line_of(def).find("inapplicable there") != std::string::npos);
    CHECK(resolve(args, {{"STRIX_GLM_SERVE_SAMPLED_MTP", "0"}}).sampled_mtp.route == rd::SampledMtp::off);
    // coupled: any profile, also without one (resolved from the same table for the default column)
    for (const char* profile : {"", "exact", "quality", "fast", "turbo"}) {
        auto a = args;
        if (*profile) a = cat(a, {"--profile", profile});
        const auto e = resolve(a, {{"STRIX_GLM_SERVE_SAMPLED_MTP", "coupled"}});
        CHECK(e.sampled_mtp.route == rd::SampledMtp::coupled && e.sampled_mtp.source == "explicit");
        CHECK(e.sampled_mtp.acceptance() == strix::SampledAcceptance::coupled);
        CHECK(line_of(e).find("=coupled (explicit, row serve_sampled_mtp, identity glm-sampled-mtp-coupled-v1;") != std::string::npos);
    }
    // the previous 0|1 switch and unknown words are refused; the shared acceptance switch may not contradict the row
    CHECK(refused(args, {{"STRIX_GLM_SERVE_SAMPLED_MTP", "1"}}, "must be 0, coupled or rejection"));
    CHECK(refused(args, {{"STRIX_GLM_SERVE_SAMPLED_MTP", "yes"}}, "must be 0, coupled or rejection"));
    CHECK(refused(args, {{"STRIX_GLM_SERVE_SAMPLED_MTP", "coupled"}, {"STRIX_MTP_SAMPLED_ACCEPT", "rejection"}}, "conflicts with"));
    // rejection: refused without a profile and under exact / quality, admitted by fast / turbo as an engine override
    CHECK(refused(args, {{"STRIX_GLM_SERVE_SAMPLED_MTP", "rejection"}}, "only --profile fast or turbo"));
    CHECK(refused(cat(args, {"--profile", "quality"}), {{"STRIX_GLM_SERVE_SAMPLED_MTP", "rejection"}}, "only --profile fast or turbo"));
    const auto fast = resolve(cat(args, {"--profile", "fast"}), {{"STRIX_GLM_SERVE_SAMPLED_MTP", "rejection"}});
    CHECK(fast.sampled_mtp.route == rd::SampledMtp::rejection && fast.sampled_mtp.acceptance() == strix::SampledAcceptance::rejection);
    CHECK(fast.product && fast.product->engine_overrides.size() == 1 &&
          fast.product->engine_overrides[0] == "STRIX_GLM_SERVE_SAMPLED_MTP=rejection");
    // all-on record: the route that does not run says why (nullptr: it runs or the row is off)
    CHECK(rd::sampled_mtp_decline(def.sampled_mtp, false, false) == nullptr);
    const auto coupled = resolve(args, {{"STRIX_GLM_SERVE_SAMPLED_MTP", "coupled"}});
    CHECK(rd::sampled_mtp_decline(coupled.sampled_mtp, true, true) == nullptr);
    CHECK(std::string(rd::sampled_mtp_decline(coupled.sampled_mtp, false, true)) == "MTP self-speculation is off in this run");
    CHECK(std::string(rd::sampled_mtp_decline(coupled.sampled_mtp, true, false)) == "this server does not serve sampled requests");
    CHECK(line_of(fast).find("identity glm-sampled-mtp-rejection-v1;") != std::string::npos);
    // the two rules and off are distinct resolution identities; unset keeps the legacy (pre-row) fingerprint
    const auto fp = [&](const char* v) {
        std::map<std::string, std::string> env;
        if (v) env["STRIX_GLM_SERVE_SAMPLED_MTP"] = v;
        return resolve(cat(args, {"--profile", "fast"}), env).product->fingerprint();
    };
    CHECK(fp("coupled") != fp("rejection") && fp("coupled") != fp(nullptr) && fp("rejection") != fp(nullptr));
    const auto builtin = resolve(cat(args, {"--profile", "fast"}), {});
    const auto* row = builtin.product->find("serve_sampled_mtp");
    CHECK(row != nullptr && row->value == "0" && row->lever->omit_builtin_fingerprint);
    auto legacy = *builtin.product;
    std::erase_if(legacy.entries, [](const auto& entry) { return entry.lever->name == "serve_sampled_mtp"; });
    CHECK(legacy.fingerprint() == builtin.product->fingerprint());
    // the server's resolution path (rd::resolve_serve) resolves the same row
    const rd::EnvGet coupled_env = [](const char* k) -> const char* {
        return std::string(k) == "STRIX_GLM_SERVE_SAMPLED_MTP" ? "coupled" : nullptr;
    };
    CHECK(rd::resolve_serve(kDeploy, true, coupled_env).sampled_mtp.route == rd::SampledMtp::coupled);
}

}  // namespace

// r94: the drafter's content identity. Same payload under other file names / offsets -> same digest; one changed payload
// byte -> another digest; bytes outside the spans do not count; an incomplete inventory has none. The digest is named
// by the receipt and by the path-free print() the pair identity vote hashes.
void test_mtp_payload_digest() {
    const auto dir = std::filesystem::temp_directory_path() /
        ("glm-mtp-digest-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(dir);
    struct Cleanup {
        std::filesystem::path dir;
        ~Cleanup() {
            for (const auto& e : std::filesystem::directory_iterator(dir)) std::filesystem::remove(e.path());
            std::filesystem::remove(dir);
        }
    } cleanup{dir};
    const auto put = [&](const char* name, const std::string& bytes) {
        std::ofstream f(dir / name, std::ios::binary);
        f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    };
    std::string a(3000, '\0'), b(4000, 'x');
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<char>(i * 37 + 11);
    for (std::size_t i = 0; i < 3000 - 100; ++i) b[1000 + i] = a[100 + i];  // same span bytes, other file / offset
    put("a.safetensors", a);
    put("b.safetensors", b);
    rd::ModelFacts facts;
    facts.mtp_present = facts.mtp_complete = true;
    facts.mtp_spans = {{"t0", dir / "a.safetensors", 100, 1203}, {"t1", dir / "a.safetensors", 1303, 1697}};
    const auto digest = rd::mtp_payload_digest(facts);
    CHECK(digest.size() == 16);
    auto moved = facts;
    moved.mtp_spans = {{"t0", dir / "b.safetensors", 1000, 1203}, {"t1", dir / "b.safetensors", 2203, 1697}};
    CHECK(rd::mtp_payload_digest(moved) == digest);
    auto renamed = facts;
    renamed.mtp_spans[1].name = "t2";
    CHECK(rd::mtp_payload_digest(renamed) != digest);
    b[1000 + 1500] = static_cast<char>(b[1000 + 1500] ^ 1);  // one payload bit inside t1
    put("b.safetensors", b);
    CHECK(rd::mtp_payload_digest(moved) != digest);
    b[1000 + 1500] = static_cast<char>(b[1000 + 1500] ^ 1);
    b[10] = 'y';  // outside every span
    put("b.safetensors", b);
    CHECK(rd::mtp_payload_digest(moved) == digest);
    auto partial = facts;
    partial.mtp_complete = false;
    CHECK(rd::mtp_payload_digest(partial).empty());

    facts.mtp_files = {dir / "a.safetensors"};
    auto e = rd::resolve(rd::scan(cat(kDeploy, kDecodeWork)), getter(Env{}), facts);
    CHECK(e.mtp_spec.value == "1");
    e.head.mtp_digest = digest;
    CHECK(rd::mtp_decode_line(e).find(" payload=" + digest + "\n") != std::string::npos);
    auto other = e;
    other.head.mtp_digest = rd::mtp_payload_digest(renamed);
    const auto line = std::string("effective mtp-payload=") + digest + "\n";
    CHECK(rd::print(e, false).find(line) != std::string::npos);
    CHECK(rd::print(e, false) != rd::print(other, false));  // the identity vote tells the two drafters apart
    auto plain = rd::resolve(rd::scan(cat(cat(kDeploy, kDecodeWork), {"--mtp", "0"})), getter(Env{}), facts);
    plain.head.mtp_digest = digest;
    CHECK(plain.mtp_spec.value == "0" && rd::print(plain, false).find("effective mtp-payload=") == std::string::npos);
}

int main() {
    test_sampled_mtp_admission();
    test_all_on_values_accepted();
    test_pf_variants();
    test_mtp_default();
    test_mtp_payload_digest();
    test_plain_equals_receipt();
    test_apply_env_default();
    test_item_off_switches();
    test_baseline();
    test_scan_and_print();
    test_max_rows_memory();
    test_startup_agreement();
    test_drafter_digest_vote();
    test_no_handshake_fallback();
    test_effective_config_rows();
    test_reference_prefill_rows();
    std::printf("glm rank defaults cpu tests: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
