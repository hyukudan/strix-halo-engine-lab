// strix-glm5-rank defaults, CPU tests (no GPU, no peer): the plain default invocation must resolve to exactly the
// measured production receipt (route + flags + env), and every off switch / the baseline master switch must restore
// the old path while explicit settings win.
//   measured decode configuration: decode 64 + R8 batch, qualification passed
//   measured prefill configuration: hp host path, exact TTFT 1K/8K
#include "glm_rank_defaults.hpp"

#include <cstdio>
#include <chrono>
#include <filesystem>
#include <fstream>
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
    CHECK(resolve(cat(cat(kDeploy, kDecodeWork), {"--mtp", "1", "--mtp-spec", "0"}), {}).prefill == "ref");
    // prefill chunk rows: default 4096; baseline 1024 (the old app default); explicit wins
    CHECK(plain.max_rows.value == "4096" && plain.max_rows.source == "default");
    CHECK(resolve(cat(kDeploy, kTtftWork), {}).max_rows.source == "explicit");
    CHECK(resolve(cat(cat(kDeploy, kDecodeWork), {"--max-rows", "1024"}), {}).max_rows.value == "1024");
    CHECK(resolve(cat(cat(kDeploy, kDecodeWork), {"--baseline"}), {}).max_rows.value == "1024");
}

void test_apply_env_default() {
    CHECK(rd::env_items().size() == 20);
    CHECK(resolve(cat(kDeploy, kDecodeWork), {}).env.at("STRIX_GLM_SPEC_XCH").value == "xfer");
    CHECK(resolve({"--baseline"}, {}).env.at("STRIX_GLM_SPEC_XCH").value == "comm");
    CHECK(resolve({}, {{"STRIX_GLM_SPEC_XCH", "comm"}}).env.at("STRIX_GLM_SPEC_XCH").source == "explicit");
    const auto plain = resolve(cat(kDeploy, kDecodeWork), {});
    Env set;
    rd::apply_env(plain, [&](const char* k, const char* v) { set[k] = v; });
    // the app sets exactly the items whose code default is the old path; engine defaults are reported only
    for (const auto& it : rd::env_items()) {
        if (it.app_sets) CHECK(set.count(it.env) == 1 && set[it.env] == it.on);
        else CHECK(set.count(it.env) == 0);
    }
    CHECK(set == merge(kHp, {{"HSA_DISABLE_FRAGMENT_ALLOCATOR", "1"}, {"STRIX_GLM_PF_PIPELINE", "2"},
                              {"STRIX_DIST_ASYNC_START", "1"}, {"STRIX_GLM_SPEC_KDA_TREE", "2"}}));
    // = the receipt's optimization env + the prefill overlap + the two-barrier KDA verify tree
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
    CHECK(rd::probe_model_mtp(dir).mtp_complete);
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
    auto resolve_head = [&](const Args& extra, const Env& env = Env{}) {
        return rd::resolve(rd::scan(cat(cat(kDeploy, kDecodeWork), extra)), getter(env), facts);
    };
    const auto on = resolve_head({});
    CHECK(on.mtp.value == "1" && on.spec.value == "1" && on.mtp_spec.value == "1");
    CHECK(on.prefill == "gemv" && on.kmax == 3 && on.adaptive);
    CHECK(rd::print(on).find("effective mtp-bootstrap=chunked") != std::string::npos);
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
}

}  // namespace

int main() {
    test_mtp_default();
    test_plain_equals_receipt();
    test_apply_env_default();
    test_item_off_switches();
    test_baseline();
    test_scan_and_print();
    std::printf("glm rank defaults cpu tests: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
