// One shared host test of the product tier word over every engine's profile table: `--profile fast` must mean
// "every qualified FAST lever ON" (and every lever selected for fast ahead of its receipt, stamped so). It fails for
// any engine where
//   - a row qualified (ready) or selected (provisional) for fast is not in the resolved FAST set,
//   - a row is marked qualified for fast but its fast cell is empty (the qualified lever would never run),
//   - fast resolves to the same set as exact while such a row exists,
//   - the exact column enables a bit-changing row, or a qualified fast cell carries no receipt,
//   - a FAST bundle qualified for the word (fast_profile::Bundle::profile_receipt) has a member whose fast cell is not
//     qualified with the member's value (the gated lever never runs under the word: the original MiniMax defect),
//   - a FAST gate ledger receipt (strix::profile::FastGate) names a value the resolved fast set does not apply, or a
//     bit-changing fast cell is marked qualified without a ledger receipt or a word-qualified bundle.
// Negative fixtures prove the checker fails on every form of that defect (bundle member and single lever). MiniMax is also checked through its resolver
// (prefill_policy / profile_resolution / prefill_identity under STRIX_MM_PRODUCT_PROFILE alone: no export).
// Tables: DeepSeek V4.1 pair, GLM pair, MiniMax pair, Qwen Flash-Next HIP and CUDA, CUDA Qwen dense (per-regime
// qualification rows). Host-only.
#include "strix/deepseek41_pair_profile_table.hpp"
#include "strix/ds41_fast_profile.hpp"
#include "strix/fast_profile.hpp"
#include "strix/engine_profile.hpp"
#include "strix/glm_pair_profile_table.hpp"
#include "strix/profiles/ds41_rtx.hpp"
#include "strix/profiles/qwen.hpp"
#include "strix/profiles/qwen_cuda.hpp"
#include "strix/profiles/qwen_dense_cuda.hpp"
#include "strix/profiles/qwen_dense_hip.hpp"
#include "minimax/engine/mm_defaults.hpp"
#include "minimax/engine/mm_profile_table.hpp"

#include <cstdint>
#include <cstdio>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
namespace p = strix::profile;
constexpr auto kExact = static_cast<unsigned>(p::Name::standard);
constexpr auto kFast = static_cast<unsigned>(p::Name::fast);
int failures = 0;
bool fixture = false;  // a negative fixture runs: its failures are expected and counted, not reported

void require(bool ok, const std::string& what) {
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "%s %s\n", fixture ? "(expected) fail" : "FAIL", what.c_str());
}

// Runs `check` on a deliberately broken input; the checker must report at least one failure.
template <class Check>
void must_fail(const std::string& what, const Check& check) {
    const int before = failures;
    fixture = true;
    check();
    fixture = false;
    const bool caught = failures > before;
    failures = before;
    require(caught, "negative fixture not caught: " + what);
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
// `ladder`: the engine's ladder rows (incumbent grants keep the cells that ran before the ladder rule).
void check_table(std::string_view tag, std::span<const p::Lever> rows, bool extra_fast = false,
                 std::span<const p::LadderRow> ladder = {}) {
    const p::Engine engine{tag, "", rows, false, {}, {}, {}, ladder};
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
    if (!fixture)
        std::printf("%-22s fast = %s [%s]\n", std::string(tag).c_str(), applied_set(fast).c_str(), fast.qualification().c_str());
}

// A FAST bundle qualified for the word: each member is a qualified fast cell carrying the member's value, and the
// resolved fast set applies it. A bundle without profile_receipt is a switch only (nothing to check here).
void check_bundle(std::string_view tag, std::span<const p::Lever> rows, const strix::fast_profile::Bundle& bundle,
                  std::span<const p::LadderRow> ladder = {}) {
    if (bundle.profile_receipt == nullptr) {
        if (!fixture) std::printf("%-22s bundle %s: switch only (no profile receipt)\n", std::string(tag).c_str(), bundle.profile);
        return;
    }
    const p::Engine engine{tag, "", rows, false, {}, {}, {}, ladder};
    const auto fast = run(engine, p::Name::fast);
    for (const auto& m : bundle.members) {
        const std::string where = std::string(tag) + "/" + bundle.profile + "/" + m.env;
        const p::Lever* row = nullptr;
        for (const auto& l : rows) if (l.env == m.env) row = &l;
        require(row != nullptr, where + ": member of a word-qualified bundle has no profile row");
        if (row == nullptr) continue;
        require(row->kind != p::Kind::exact && row->ready[kFast] && !row->provisional[kFast] && row->runnable[kFast] &&
                    row->value[kFast] == m.value,
                where + ": gated member's fast cell is empty or not qualified with the bundle value (never runs under the word)");
        const p::Entry* entry = nullptr;
        for (const auto& e : fast.entries) if (e.lever == row) entry = &e;
        require(entry != nullptr && p::applied(entry->status) && entry->value == m.value,
                where + ": gated member is not in the resolved fast set");
    }
    if (!fixture) std::printf("%-22s bundle %s: word-qualified (%s)\n", std::string(tag).c_str(), bundle.profile, bundle.profile_receipt);
}

// Rows qualified for the word by a bundle receipt (empty when the bundle is a switch only).
std::vector<std::string_view> bundle_rows(std::span<const p::Lever> rows, const strix::fast_profile::Bundle& bundle) {
    std::vector<std::string_view> out;
    if (bundle.profile_receipt == nullptr) return out;
    for (const auto& m : bundle.members)
        for (const auto& l : rows)
            if (l.env == m.env) out.push_back(l.name);
    return out;
}

// The FAST gate ledger (strix::profile::FastGate): every receipted value is in the resolved fast set and every
// qualified bit-changing fast cell has a receipt. An engine adds a `fast_gates` ledger beside its table when a single
// lever passes its gate; today every qualified bit-changing FAST cell is a member of a word-qualified bundle.
void check_gates(std::string_view tag, std::span<const p::Lever> rows, std::span<const p::FastGate> ledger,
                 std::span<const std::string_view> covered = {}) {
    const p::Engine engine{tag, "", rows};
    const auto defect = p::fast_gate_defect(engine, ledger, covered);
    require(defect.empty(), defect);
    if (!fixture)
        std::printf("%-22s FAST gate ledger: %zu receipt(s), %zu bundle row(s)\n", std::string(tag).c_str(), ledger.size(),
                    covered.size());
}

// The checker itself: both forms of the original MiniMax defect must fail.
void negative_fixtures() {
    namespace md = strix::minimax::defaults;
    auto rows = md::profile_table;  // copy; mutate the flash member back to its pre-fix form
    for (auto& l : rows) {
        if (l.env != md::kFastMembers[0].env) continue;
        l.value[kFast] = "";
        l.ready[kFast] = false;
    }
    must_fail("gated bundle member with an empty fast cell, ready[fast]=false",
              [&] { check_bundle("minimax-fixture-a", rows, md::kFastBundle); });
    auto rows_b = md::profile_table;  // marked qualified but emptied
    for (auto& l : rows_b)
        if (l.env == md::kFastMembers[1].env) l.value[kFast] = "";
    must_fail("qualified fast row with an empty fast cell", [&] { check_table("minimax-fixture-b", rows_b); });
    must_fail("gated bundle member with an empty qualified fast cell",
              [&] { check_bundle("minimax-fixture-b", rows_b, md::kFastBundle); });
    // The single-lever form (no bundle): the gate receipt is in the ledger, the fast cell is empty, ready[fast]=false.
    auto rows_c = md::profile_table;
    for (auto& l : rows_c)
        if (l.name == "router_wmma") {
            l.value[kFast] = "";
            l.ready[kFast] = false;
        }
    const p::FastGate gate_c[] = {{"router_wmma", "1", "fixture receipt"}};
    const std::string_view flash_row[] = {"msa_flash"};
    must_fail("gated single lever outside a bundle: empty fast cell, ready[fast]=false",
              [&] { check_gates("minimax-fixture-c", rows_c, gate_c, flash_row); });
    // The converse: a bit-changing fast cell marked qualified with no ledger receipt and no word-qualified bundle.
    must_fail("qualified bit-changing fast cell without a FAST gate receipt",
              [&] { check_gates("minimax-fixture-d", md::profile_table, {}); });
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
    const auto status = [](const p::Resolution& r, std::string_view lever) {
        const auto* e = r.find(lever);
        return e == nullptr ? std::string("absent") : std::string(p::status_name(e->status)) + ":" + e->value;
    };
    // The word alone (no switch exported anywhere): the resolver runs the bundle.
    require(md::fast_profile_selects_bundle(lookup), "minimax: fast selects the qualified bundle mm-fast-pf-v2");
    const auto pf = md::prefill_policy(lookup);
    require(pf.msa_flash == 1 && pf.router_wmma && pf.bundle_from_profile,
            "minimax: STRIX_MM_PRODUCT_PROFILE=fast alone runs msa_flash=1 + router_wmma (resolver, not an export)");
    const auto fast_id = md::prefill_identity(lookup);
    const auto fast_res = md::profile_resolution(lookup);
    require(status(fast_res, "msa_flash") == "profile:1" && status(fast_res, "router_wmma") == "profile:1",
            "minimax: the word's members keep the profile status (got " + status(fast_res, "msa_flash") + ", " +
                status(fast_res, "router_wmma") + ")");
    {   // the same run and prefill identity as the switch
        std::map<std::string, std::string> sw{{md::kFastBundle.env, "1"}};
        const auto sw_lookup = [&sw](const char* k) -> const char* {
            const auto it = sw.find(k);
            return it == sw.end() ? nullptr : it->second.c_str();
        };
        auto spf = md::prefill_policy(sw_lookup);
        spf.canonical.erase(md::kFastBundle.env);  // the switch itself is the only difference (requested provenance)
        require(md::prefill_identity(sw_lookup) == fast_id && spf.policies == pf.policies && spf.canonical == pf.canonical,
                "minimax: --profile fast = STRIX_MM_FAST_PREFILL=1 (identity " + fast_id + ")");
    }
    // An explicit switch 0 wins: nothing runs and the resolution says so (no applied member).
    env[md::kFastBundle.env] = "0";
    require(!md::fast_profile_selects_bundle(lookup), "minimax: an explicit bundle switch 0 wins");
    const auto off = md::prefill_policy(lookup);
    const auto off_res = md::profile_resolution(lookup);
    require(off.msa_flash == 0 && !off.router_wmma && status(off_res, "msa_flash") == "off:0" &&
                status(off_res, "router_wmma") == "off:0",
            "minimax: switch 0 under fast resolves the members off (got " + status(off_res, "msa_flash") + ", " +
                status(off_res, "router_wmma") + ")");
    env.erase(md::kFastBundle.env);
    // FAST_I4 admission: the flash member is not admitted; the resolution must not claim it.
    env["STRIX_MMX_MODEL_FORMAT"] = "fast_i4";
    const auto i4 = md::prefill_policy(lookup);
    require(i4.msa_flash == 0 && !i4.router_wmma && status(md::profile_resolution(lookup), "msa_flash") == "off:0",
            "minimax: fast on fast_i4 keeps the bundle's admission (flash and router not run, resolved off)");
    env.erase("STRIX_MMX_MODEL_FORMAT");
    // The exact baseline refuses the word (before the word selected the bundle this combination ran exact).
    env[md::kBaselineEnv] = "1";
    bool refused = false;
    try { (void)md::prefill_policy(lookup); } catch (const std::invalid_argument&) { refused = true; }
    require(refused, "minimax: --baseline --profile fast is refused");
    env.erase(md::kBaselineEnv);
    // exact / no word: no bundle, the legacy identity.
    env[md::kProductProfileEnv] = "exact";
    require(!md::fast_profile_selects_bundle(lookup) && md::prefill_identity(lookup) != fast_id &&
                md::prefill_policy(lookup).msa_flash == 0,
            "minimax: exact selects no bundle");
    const auto exact_fp = md::profile_resolution(lookup).fingerprint();
    const auto exact_id = md::prefill_identity(lookup);
    env.clear();
    require(md::prefill_identity(lookup) == exact_id && md::profile_resolution(lookup).fingerprint() == exact_fp,
            "minimax: no word == exact (identity and fingerprint)");
}

void qwen_dense_regimes() {
    namespace qd = strix::qwen35::profile;
    for (const auto& q : qd::qualified) {
        const qd::Regime regime{q.weights, q.route, q.gate.min_prompt_tokens};
        const auto fast = qd::resolve(p::Name::fast, regime, {});
        const auto* e = fast->find(q.gate.lever);
        require(e != nullptr && p::applied(e->status), "qwen35-cuda-dense/" + std::string(q.gate.lever) + ": qualified regime not in the fast set");
        const auto exact = qd::resolve(p::Name::standard, regime, {});
        for (const auto& x : exact->resolution.entries)
            require(!p::applied(x.status), "qwen35-cuda-dense/" + std::string(x.lever->name) + ": exact applies a lever");
    }
    check_table("qwen35-cuda-dense(all)", std::span<const p::Lever>(qd::table), true);
}

// Shared regime ledger row (strix::profile::RegimeGate): range bounds inclusive, unknown prompt (0) outside a row whose
// lower bound is 1, empty value = any value of the cell, the range text the engines print.
void regime_gate_rows() {
    constexpr p::RegimeGate bounded{"lev", "7", 1, 8192, "r"};
    constexpr p::RegimeGate any_value{"lev", "", 0, p::kNoPromptBound, "r"};
    static_assert(p::regime_covers(bounded, "lev", "7", 1) && p::regime_covers(bounded, "lev", "7", 8192));
    static_assert(!p::regime_covers(bounded, "lev", "7", 0) && !p::regime_covers(bounded, "lev", "7", 8193));
    static_assert(!p::regime_covers(bounded, "lev", "8", 100) && !p::regime_covers(bounded, "other", "7", 100));
    static_assert(p::regime_covers(any_value, "lev", "8", 0) && p::regime_covers(any_value, "lev", "", p::kNoPromptBound));
    const p::RegimeGate rows[] = {bounded, {"lev", "7", 8193, p::kNoPromptBound, "r2"}};
    require(p::regime_gate(rows, "lev", "7", 9000) == &rows[1] && p::regime_gate(rows, "lev", "7", 0) == nullptr,
            "regime ledger: first covering row");
    require(p::regime_range_text(bounded) == " / prompt <= 8192" && p::regime_range_text({"lev", "7", 2, 8192, "r"}) == " / prompt 2..8192" &&
                p::regime_range_text(any_value).empty() &&
                p::regime_range_text(rows[1]) == " / prompt >= 8193" &&
                p::regime_range_text({"lev", "7", 0, 4096, "r"}) == " / prompt <= 4096",
            "regime ledger: range text");
}

// Qwen Flash-Next HIP regime ledger (strix::qwen::profile::fast_regimes): inside a row's prompt range the fast cell is
// qualified (status set, same value); outside it (and with the prompt unknown) the table's provisional flag is
// unchanged; exact applies nothing; an explicit off switch still wins inside the regime.
void qwen_hip_regimes() {
    namespace qh = strix::qwen::profile;
    const std::span<const p::RegimeGate> rows(qh::fast_regimes);
    const auto status_at = [&](p::Name name, std::uint64_t tokens, std::string_view lever, const char* env_key = nullptr,
                               const char* env_value = nullptr) {
        qh::Inputs in;
        in.env = [=](const char* key) -> const char* { return env_key && std::string_view(key) == env_key ? env_value : nullptr; };
        in.max_prompt_tokens = tokens;
        in.regimes = rows;
        const auto r = qh::resolve(name, in);
        const auto* e = r.find(lever);
        return std::make_pair(e ? e->status : p::Status::builtin, e ? e->value : std::string());
    };
    for (const auto& q : rows) {
        const std::string where = "qwen-hip/" + std::string(q.lever);
        const p::Lever* lever = nullptr;
        for (const auto& l : qh::table) if (l.name == q.lever) lever = &l;
        require(lever != nullptr && !lever->value[kFast].empty() && lever->runnable[kFast], where + ": regime row names no runnable fast cell");
        require(!q.receipt.empty() && q.min_prompt_tokens >= 1U && q.max_prompt_tokens >= q.min_prompt_tokens,
                where + ": regime row without receipt or range (an unknown prompt must stay outside)");
        if (!lever) continue;
        require(q.value == lever->value[kFast], where + ": regime row value is not the fast cell");
        for (const auto tokens : {q.min_prompt_tokens, q.max_prompt_tokens}) {
            const auto [status, value] = status_at(p::Name::fast, tokens, q.lever);
            require(status == p::Status::set && value == lever->value[kFast], where + ": not qualified inside its regime");
            require(!p::applied(status_at(p::Name::standard, tokens, q.lever).first), where + ": exact applies it in the regime");
        }
        for (const std::uint64_t tokens : {std::uint64_t{0}, q.max_prompt_tokens + 1U, std::uint64_t{32460}})
            require(status_at(p::Name::fast, tokens, q.lever).first == (lever->provisional[kFast] ? p::Status::provisional : p::Status::pending),
                    where + ": outside its regime the table flags changed");
        if (!lever->env.empty() && !lever->off.empty()) {
            const std::string key(lever->env), off(lever->off);
            require(status_at(p::Name::fast, q.max_prompt_tokens, q.lever, key.c_str(), off.c_str()).first == p::Status::off,
                    where + ": explicit off lost inside the regime");
        }
    }
    std::printf("%-22s regimes: %zu rows qualified up to %llu prompt tokens\n", "qwen-hip", rows.size(),
                static_cast<unsigned long long>(rows.empty() ? 0U : rows.front().max_prompt_tokens));
}
}  // namespace

int main() {
    tier_words();
    negative_fixtures();
    check_table("deepseek41-pair", strix::deepseek41::pair_defaults::profile_table, false,
                strix::deepseek41::pair_defaults::ladder_rows);
    check_bundle("deepseek41-pair", strix::deepseek41::pair_defaults::profile_table, strix::ds41::fast_profile::kBundle,
                 strix::deepseek41::pair_defaults::ladder_rows);
    check_table("glm-pair", strix::glm::rank_defaults::product_profile_table, false,
                strix::glm::rank_defaults::product_ladder_rows);
    check_table("deepseek41-rtx", strix::ds41::rtx::profile::shared_rows());
    minimax_bundle();
    check_table("minimax-pair", strix::minimax::defaults::profile_table, false, strix::minimax::defaults::ladder_rows);
    check_bundle("minimax-pair", strix::minimax::defaults::profile_table, strix::minimax::defaults::kFastBundle,
                 strix::minimax::defaults::ladder_rows);
    check_table("qwen-hip", strix::qwen::profile::table, false, strix::qwen::profile::ladder_rows);
    check_table("qwen-cuda", strix::qwen::cuda_profile::table, false, strix::qwen::cuda_profile::ladder_rows);
    check_table("qwen35-hip-dense", strix::qwen35::hip_profile::table, false, strix::qwen35::hip_profile::kLadderRows);
    qwen_dense_regimes();
    regime_gate_rows();
    qwen_hip_regimes();
    {
        namespace dp = strix::deepseek41::pair_defaults;
        namespace md = strix::minimax::defaults;
        const auto ds41_rows = bundle_rows(dp::profile_table, strix::ds41::fast_profile::kBundle);
        const auto mm_rows = bundle_rows(md::profile_table, md::kFastBundle);
        check_gates("deepseek41-pair", dp::profile_table, dp::fast_gates, ds41_rows);
        check_gates("glm-pair", strix::glm::rank_defaults::product_profile_table, {});
        check_gates("deepseek41-rtx", strix::ds41::rtx::profile::shared_rows(), {});
        check_gates("minimax-pair", md::profile_table, {}, mm_rows);
        check_gates("qwen-hip", strix::qwen::profile::table, {});
        check_gates("qwen-cuda", strix::qwen::cuda_profile::table, {});
        check_gates("qwen35-hip-dense", strix::qwen35::hip_profile::table, strix::qwen35::hip_profile::kFastGates);
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("profile fast set: all engines PASS\n");
    return 0;
}
