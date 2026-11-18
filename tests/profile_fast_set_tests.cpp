// One shared host test of the product tier word over every engine's profile table: `--profile fast` must mean
// "every qualified FAST lever ON" (and every lever selected for fast ahead of its receipt, stamped so). It fails for
// any engine where
//   - a row qualified (ready) or selected (provisional) for fast is not in the resolved FAST set,
//   - a row is marked qualified for fast but its fast cell is empty (the qualified lever would never run),
//   - fast resolves to the same set as exact while such a row exists,
//   - the exact column enables a bit-changing row, or a qualified fast cell carries no receipt.
// Tables: DeepSeek V4.1 pair, GLM pair, MiniMax pair (+ its FAST bundle switch), Qwen Flash-Next HIP and CUDA,
// CUDA Qwen dense (per-regime qualification rows). Host-only.
#include "strix/deepseek41_pair_profile_table.hpp"
#include "strix/engine_profile.hpp"
#include "strix/glm_pair_profile_table.hpp"
#include "strix/profiles/qwen.hpp"
#include "strix/profiles/qwen_cuda.hpp"
#include "strix/profiles/qwen_dense_cuda.hpp"
#include "minimax/engine/mm_defaults.hpp"
#include "minimax/engine/mm_profile_table.hpp"

#include <cstdio>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace p = strix::profile;
constexpr auto kExact = static_cast<unsigned>(p::Name::standard);
constexpr auto kFast = static_cast<unsigned>(p::Name::fast);
int failures = 0;

void require(bool ok, const std::string& what) {
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
}

p::Resolution run(const p::Engine& engine, p::Name profile) {
    p::Inputs in;
    in.profile = profile;
    in.env = [](const char*) -> const char* { return nullptr; };
    return p::resolve(engine, in);
}

std::string applied_set(const p::Resolution& r) {
    std::string out;
    for (const auto& e : r.entries)
        if (p::applied(e.status)) out += (out.empty() ? "" : ",") + std::string(e.lever->name) + "=" + e.value;
    return out.empty() ? "(none)" : out;
}

// A fast cell the resolver must apply with no environment and no --allow-pending.
bool fast_cell_live(const p::Lever& l) {
    const auto v = l.value[kFast];
    return !v.empty() && v != l.builtin && (l.ready[kFast] || (l.provisional[kFast] && l.runnable[kFast]));
}

// extra_fast: the engine's fast profile also selects a FAST bundle outside these rows (checked by the caller).
void check_table(std::string_view tag, std::span<const p::Lever> rows, bool extra_fast = false) {
    const p::Engine engine{tag, "", rows};
    const auto exact = run(engine, p::Name::standard);
    const auto fast = run(engine, p::Name::fast);
    bool any = extra_fast;
    for (const auto& l : rows) {
        const std::string where = std::string(tag) + "/" + std::string(l.name);
        if (l.kind != p::Kind::exact)
            require(l.value[kExact].empty() || l.value[kExact] == l.builtin, where + ": exact column enables a bit-changing value");
        if (l.ready[kFast] && l.kind != p::Kind::exact && !l.provisional[kFast])
            require(!l.value[kFast].empty() || !l.runnable[kFast],
                    where + ": marked qualified for fast but the fast cell is empty (the lever never runs under fast)");
        if (!fast_cell_live(l)) continue;
        any = true;
        const auto* entry = fast.find(l.name);
        require(entry != nullptr && p::applied(entry->status) && entry->value == l.value[kFast],
                where + ": qualified/selected FAST cell is not in the resolved fast set");
        if (l.ready[kFast]) require(!l.source.empty(), where + ": qualified FAST cell without a receipt");
    }
    for (const auto& e : exact.entries)
        require(!p::applied(e.status), std::string(tag) + "/" + std::string(e.lever->name) + ": exact applies a lever");
    if (any && !extra_fast)
        require(fast.fingerprint() != exact.fingerprint() && fast.profile_applied(),
                std::string(tag) + ": fast resolves to the exact set while a FAST row is qualified or selected");
    std::printf("%-22s fast = %s [%s]\n", std::string(tag).c_str(), applied_set(fast).c_str(), fast.qualification().c_str());
}

void tier_words() {
    require(p::parse("exact") == p::Name::standard && p::parse("default") == p::Name::standard, "exact is the default column");
    require(p::is_name("exact") && p::is_name("fast") && p::is_name("turbo"), "tier words exact|fast|turbo");
    std::vector<std::string> args{"m.gguf", "--profile", "exact"};
    p::Inputs in;
    p::take_options(strix::glm::rank_defaults::product_profile_engine, args, in);
    require(args.size() == 1U && in.profile == p::Name::standard, "--profile exact is consumed as the default column");
}

void minimax_bundle() {
    namespace md = strix::minimax::defaults;
    std::map<std::string, std::string> env{{md::kProductProfileEnv, "fast"}};
    const auto lookup = [&env](const char* k) -> const char* {
        const auto it = env.find(k);
        return it == env.end() ? nullptr : it->second.c_str();
    };
    require(md::fast_profile_selects_bundle(lookup), "minimax: fast selects the qualified bundle mm-fast-pf-v2");
    env[md::kFastBundle.env] = "0";
    require(!md::fast_profile_selects_bundle(lookup), "minimax: an explicit bundle switch 0 wins");
    env.erase(md::kFastBundle.env);
    env[md::kProductProfileEnv] = "exact";
    require(!md::fast_profile_selects_bundle(lookup), "minimax: exact selects no bundle");
}

void qwen_dense_regimes() {
    namespace qd = strix::qwen35::profile;
    for (const auto& q : qd::qualified) {
        const qd::Regime regime{q.weights, q.route, q.min_prompt_tokens};
        const auto fast = qd::resolve(p::Name::fast, regime, {});
        const auto* e = fast->find(q.lever);
        require(e != nullptr && p::applied(e->status), "qwen35-cuda-dense/" + std::string(q.lever) + ": qualified regime not in the fast set");
        const auto exact = qd::resolve(p::Name::standard, regime, {});
        for (const auto& x : exact->resolution.entries)
            require(!p::applied(x.status), "qwen35-cuda-dense/" + std::string(x.lever->name) + ": exact applies a lever");
    }
    check_table("qwen35-cuda-dense(all)", std::span<const p::Lever>(qd::table), true);
}
}  // namespace

int main() {
    tier_words();
    check_table("deepseek41-pair", strix::deepseek41::pair_defaults::profile_table);
    check_table("glm-pair", strix::glm::rank_defaults::product_profile_table);
    minimax_bundle();
    check_table("minimax-pair", strix::minimax::defaults::profile_table, /*extra_fast=*/true);
    check_table("qwen-hip", strix::qwen::profile::table);
    check_table("qwen-cuda", strix::qwen::cuda_profile::table);
    qwen_dense_regimes();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("profile fast set: all engines PASS\n");
    return 0;
}
