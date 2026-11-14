// DS41 prefill resident adapter (apps/deepseek41_prefill_resident.hpp), host only:
//  1. every "STRIX_DS41_*" literal in the prefill sources is in the lever table (undeclared = reload, so a lever
//     nobody classified can never be switched in place by mistake); the table is sorted and has no duplicates;
//  2. a fake prefill whose result depends on a converted lever (strix::lever_cache) and on a construct lever
//     (apply_levers) gives, inside ONE process, per arm the same logits as a fresh process of that arm, for every
//     position in the ABAB order; a reload-class lever is refused instead of measured; cold runs are marked.

#include "deepseek41_prefill_resident.hpp"

#include "strix/lever_cache.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace dr = strix::deepseek41::resident;
namespace rp = strix::serve::resident;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

#ifdef _WIN32
void set_env(const char* k, const char* v) { _putenv_s(k, v); }
void unset_env(const char* k) { _putenv_s(k, ""); }
#else
void set_env(const char* k, const char* v) { setenv(k, v, 1); }
void unset_env(const char* k) { unsetenv(k); }
#endif

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void table_covers_sources() {
    std::set<std::string> table;
    std::string previous;
    for (const auto& l : strix::deepseek41::kPrefillLevers) {
        const std::string name(l.name);
        check(previous.empty() || previous < name, "table sorted and unique at " + name);
        previous = name;
        table.insert(name);
    }
    const fs::path root = STRIX_SOURCE_DIR;
    const std::regex literal("\"(STRIX_DS41_[A-Z0-9_]+)\"");
    std::size_t files = 0, names = 0;
    for (const std::string_view rel : strix::deepseek41::kPrefillLeverSources) {
        const fs::path p = root / fs::path(std::string(rel));
        check(fs::exists(p), "lever source exists: " + std::string(rel));
        const std::string text = read_file(p);
        ++files;
        for (std::sregex_iterator it(text.begin(), text.end(), literal), end; it != end; ++it) {
            ++names;
            const std::string name = (*it)[1].str();
            check(table.count(name) == 1, "lever " + name + " (read in " + std::string(rel) + ") is in kPrefillLevers");
        }
    }
    check(files >= 10 && names >= 100, "scanned the prefill sources (" + std::to_string(files) + " files, " +
                                           std::to_string(names) + " reads)");
    // the adapter's runner table maps every row; a name outside the table is reload
    const rp::LeverTable t = dr::lever_table();
    check(t.size() == table.size(), "runner table has every row");
    check(rp::lever_class(t, "STRIX_DS41_NOT_A_LEVER") == rp::LeverClass::reload, "undeclared lever is reload");
}

// ---- fake engine -------------------------------------------------------------------------------------------

// A converted lever (was a static const): which kernel form the fake uses.
strix::lever_cache<int> g_form{[] {
    const char* v = std::getenv("STRIX_DS41_HC_SITE_FAST");
    return v != nullptr && v[0] == '0' ? 0 : 1;
}};

struct FakeResult {
    std::vector<float> logits;
    std::uint32_t next_token = 0;
    double seconds = 0, engram_seconds = 0, head_seconds = 0;
};

struct FakePrefill {
    int construct_value = 0;  // a per-object switch, re-resolved by apply_levers (as DevicePrefill)
    int applies = 0;
    static int resolve_construct() {
        const char* v = std::getenv("FAKE_CONSTRUCT");
        return v != nullptr ? std::atoi(v) : 0;
    }
    FakePrefill() : construct_value(resolve_construct()) {}
    void apply_levers() {
        construct_value = resolve_construct();
        ++applies;
    }
    FakeResult run(std::span<const std::uint32_t> rows) {
        FakeResult r;
        const int form = g_form();
        float acc = 0;
        for (const std::uint32_t id : rows) acc += static_cast<float>(id % 97U);
        r.logits = {acc, static_cast<float>(form), static_cast<float>(construct_value), static_cast<float>(rows.size())};
        r.next_token = static_cast<std::uint32_t>(acc) + static_cast<std::uint32_t>(form) * 1000U;
        r.seconds = 0.001;
        return r;
    }
};

struct Receipt {
    std::string arm, logits_fnv, error;
    bool cold = false, warmup = false;
};

std::vector<Receipt> receipts(const fs::path& out) {
    std::vector<fs::path> files;  // run-<seq 5 digits>-<arm>-<prompt>.json: name order is run order
    for (const auto& e : fs::directory_iterator(out))
        if (e.path().filename().string().rfind("run-", 0) == 0) files.push_back(e.path());
    std::sort(files.begin(), files.end());
    std::vector<Receipt> v;
    for (const auto& f : files) {
        const auto j = strix::serve::parse_json(read_file(f));
        Receipt r;
        r.arm = j.find("arm")->as_string();
        if (const auto* x = j.find("logits_fnv"); x != nullptr && x->is_string()) r.logits_fnv = x->as_string();
        if (const auto* x = j.find("error"); x != nullptr && x->is_string()) r.error = x->as_string();
        r.cold = j.find("cold_run")->as_bool();
        r.warmup = j.find("warmup")->as_bool();
        v.push_back(r);
    }
    return v;
}

std::string plan_text(const fs::path& prompt, const std::string& arms) {
    return R"({"engine":"deepseek41-prefill","order":"abab","repeats":3,"warmup":0,"steps":1,
      "prompts":[{"name":"p","path":")" + prompt.generic_string() + R"("},{"name":"q","path":")" +
           prompt.generic_string() + R"("}],"arms":[)" + arms + "]}";
}

void fake_plan_matches_fresh() {
    const fs::path dir = fs::temp_directory_path() / ("strix-ds41-resident-test-" + std::to_string(std::rand()));
    fs::create_directories(dir);
    const fs::path prompt = dir / "p.csv";
    {
        std::ofstream p(prompt);
        p << "5,17,301,42,9,";
    }
    check(dr::read_csv_prompt(prompt.string()).size() == 5, "csv prompt read like --prompt");
    unset_env("STRIX_DS41_HC_SITE_FAST");
    unset_env("FAKE_CONSTRUCT");
    const std::string arm_a = R"({"name":"A","env":{"STRIX_DS41_HC_SITE_FAST":"0"},"prompts":["p","q"]})";
    const std::string arm_b = R"({"name":"B","env":{"STRIX_DS41_HC_SITE_FAST":"1"},"prompts":["p","q"]})";
    const auto run = [&](const std::string& arms, const std::string& tag) {
        const fs::path planf = dir / (tag + ".json");
        {
            std::ofstream f(planf);
            f << plan_text(prompt, arms);
        }
        FakePrefill pre;
        dr::Wiring w;
        w.capacity = 64;
        w.print = false;
        const auto oc = dr::run_plan_file(pre, w, planf.string(), (dir / tag).string(), 0.0);
        if (tag != "construct") check(pre.applies > 0, tag + ": apply_levers called on arm switch");
        (void)oc;
        return receipts(dir / tag);
    };
    // fresh processes of each arm alone (the cache's first read after the arm's environment: the fresh value)
    const auto fa = run(arm_a, "fresh-a");
    const auto fb = run(arm_b, "fresh-b");
    const auto res = run(arm_a + "," + arm_b, "resident");
    check(fa.size() == 6 && fb.size() == 6 && res.size() == 12, "run counts (6, 6, 12)");
    std::map<std::string, std::string> fresh = {{"A", fa.empty() ? "" : fa[0].logits_fnv}, {"B", fb.empty() ? "" : fb[0].logits_fnv}};
    check(!fresh["A"].empty() && fresh["A"] != fresh["B"], "the lever changes the result");
    std::size_t cold = 0;
    for (const auto& r : res) {
        check(r.error.empty(), "resident run without error: " + r.error);
        check(r.logits_fnv == fresh[r.arm], "resident " + r.arm + " logits equal the fresh process of " + r.arm);
        cold += r.cold ? 1U : 0U;
    }
    // ABAB with two prompts per occurrence: each occurrence is cold then warm
    check(cold == 6, "one cold run per arm occurrence (" + std::to_string(cold) + ")");
    for (std::size_t i = 0; i + 1 < res.size(); i += 2) check(res[i].cold && !res[i + 1].cold, "cold, then warm");
    // construct lever via apply_levers
    const auto con = run(R"({"name":"C0","env":{"FAKE_CONSTRUCT":"0"},"prompts":["p"]},{"name":"C5","env":{"FAKE_CONSTRUCT":"5"},"prompts":["p"]})",
                         "construct");
    // FAKE_CONSTRUCT is not in the table -> reload: refused, never measured with a stale value
    bool refused = !con.empty();
    for (const auto& r : con) refused = refused && (r.arm == "C0" || r.error.find("reload") != std::string::npos);
    check(refused, "an undeclared lever is refused as reload");
    check(std::getenv("STRIX_DS41_HC_SITE_FAST") == nullptr || std::getenv("STRIX_DS41_HC_SITE_FAST")[0] == 0,
          "the runner restores the launch environment");
    std::error_code ec;
    fs::remove_all(dir, ec);
}

}  // namespace

int main() try {
    table_covers_sources();
    fake_plan_matches_fresh();
    if (failures == 0) std::printf("deepseek41_prefill_resident_cpu_tests: ok\n");
    return failures == 0 ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
    return 1;
}
