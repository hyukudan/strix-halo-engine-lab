// Ladder rule (strix/engine_profile.hpp): TURBO is never slower than FAST, FAST never slower than EXACT, in any
// regime. Fixtures check the verdicts and the route line; the walk resolves every lossy lever of every registered engine
// at every probe regime of its ladder rows and fails when a regime would run a cell that is not measured faster than
// the cell below it (or a ladder row is malformed).
#include "strix/engine_profile.hpp"
#include "strix/cuda/qwen_options.hpp"
#include "strix/glm_pair_profile_table.hpp"
#include "strix/deepseek41_pair_profile_table.hpp"
#include "strix/model_registry.hpp"
#include "strix/profiles/ds41_rtx.hpp"
#include "strix/profiles/gemma4.hpp"
#include "strix/profiles/qwen_cuda.hpp"
#include "strix/profiles/qwen_dense_cuda.hpp"
#include "strix/profiles/qwen_dense_hip.hpp"
#include "strix/profiles/hy4.hpp"
#include "strix/profiles/inkling2_mini.hpp"
#include "minimax/engine/mm_defaults.hpp"
#include "cuda/models/glm/producer/profile.hpp"

#include <cstdio>
#include <deque>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace p = strix::profile;

namespace {

int failures = 0;
void require(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

constexpr std::string_view kLever = "fixture-w4a4@1";
// FAST cell "a8", TURBO cell "a4": both lossy, both ready.
constexpr std::array<p::Lever, 1> kRows{{
    {kLever, "STRIX_FIXTURE_W4A4", "", {"", "a8", "", "a4"}, "off", p::Kind::bit_changing,
     {true, true, false, true}, {true, true, false, true}, "", "off", "fixture receipts", false, {}, false, {},
     {{{"off", "v1"}, {"a8", "v1"}, {"a4", "v1"}, {"a6", "v1"}}}},
}};

p::Resolution resolve(const p::Engine& engine, p::Name tier, p::Regime regime = {}) {
    p::Inputs in;
    in.profile = tier;
    in.regime = regime;
    in.env = [](const char*) -> const char* { return nullptr; };
    return p::resolve(engine, in);
}

// The execution every fixture regime runs outside the table (Regime::execution; the helpers below set it).
constexpr std::string_view kExec = "fixture-exec";
// Rows hold views: the spellings built here live for the whole test.
std::string_view keep(std::string text) {
    static std::deque<std::string> pool;
    return pool.emplace_back(std::move(text));
}
// The reference execution a timing row was measured over, spelled as LadderArms::over() spells it: `levers` is the
// table's "lever=value" list; the version, the execution and the candidate (the row's own lever and value) are added.
// Every "lever=value" item without an identity gets the fixture identity {v1} (the fixture tables declare v1 for every
// value they run).
std::string identified(std::string_view items) {
    std::string out;
    while (!items.empty()) {
        const auto comma = items.find(',');
        const auto item = items.substr(0, comma);
        if (!out.empty()) out += ",";
        out += item;
        if (item.find('{') == std::string_view::npos) out += "{v1}";
        if (comma == std::string_view::npos) break;
        items.remove_prefix(comma + 1U);
    }
    return out;
}
p::LadderRow measured_over(p::LadderRow row, std::string_view levers, std::string_view execution = kExec) {
    row.over = keep(std::string(p::kLadderArmVersion) + ";" + std::string(execution) + ";" + identified(levers) +
                    ";candidate=" + std::string(row.lever) + "=" + std::string(row.value) + "{v1};tier=" + p::name(row.tier));
    return row;
}
// The references of the one-lever fixture (LadderArms::over spelling): a FAST row is measured over EXACT (the builtin
// off), a TURBO row over the FAST arm where its FAST cell runs.
constexpr std::string_view kExactArm = "fixture-w4a4@1=off";
constexpr std::string_view kFastArm = "fixture-w4a4@1=a8";

void verdicts() {
    const p::LadderRow fast_pf = measured_over({kLever, "a8", p::Name::fast, p::Phase::prefill, "dev", 1, 8192, 1, 1, 1.20,
                                                    0.02, "r/fast-pf"}, kExactArm);
    const p::LadderRow turbo_pf = measured_over({kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 2049, 4096, 1,
                                                     1, 1.30, 0.02, "r/turbo-pf-4k"}, kFastArm);
    const p::LadderRow turbo_pf_noise = measured_over({kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 4097,
                                                           8192, 1, 1, 1.003, 0.02, "r/turbo-pf-8k"}, kFastArm);
    // Measured where FAST is not measured (decode at 8-16 lanes): over the builtin FAST arm, which runs there.
    const p::LadderRow turbo_dec_slow = measured_over({kLever, "a4", p::Name::turbo, p::Phase::decode, "dev", 1, 65536, 8,
                                                           16, 0.95, 0.02, "r/turbo-dec"}, kExactArm);
    const p::LadderRow fast_dec_slow = measured_over({kLever, "a8", p::Name::fast, p::Phase::decode, "dev", 1, 65536, 1,
                                                          1, 0.97, 0.01, "r/fast-dec"}, kExactArm);
    const p::LadderRow turbo_dec_b1 = measured_over({kLever, "a4", p::Name::turbo, p::Phase::decode, "dev", 1, 65536,
                                                         1, 1, 1.25, 0.02, "r/turbo-dec-b1"}, kFastArm);
    const std::array rows{fast_pf, turbo_pf, turbo_pf_noise, turbo_dec_slow, fast_dec_slow, turbo_dec_b1};
    const p::Engine engine{"fixture", "", kRows, false, "fixture-turbo-v1", {}, {}, rows};
    require(p::ladder_defect(engine).empty(), "fixture ladder is well formed: " + p::ladder_defect(engine));

    static_assert(p::ladder_gain(1.008, 0.002), "a measured sub-percent gain is enough");
    static_assert(!p::ladder_gain(1.015, 0.02) && p::ladder_gain(1.021, 0.02), "stated margin");
    static_assert(!p::ladder_gain(std::numeric_limits<double>::infinity(), 0.02));
    static_assert(!p::ladder_gain(1.2, std::numeric_limits<double>::quiet_NaN()));
    const auto check = [&](p::Name tier, p::Regime regime, std::string_view value, std::string_view route) {
        const auto actual = resolve(engine, tier, regime);
        const auto* entry = actual.find(kLever);
        require(entry && entry->value == value, "effective production value " + std::string(value));
        if (route.empty()) return;
        require(entry && entry->ladder.find(route) != std::string::npos &&
                    actual.describe().find(entry->ladder) != std::string::npos &&
                    actual.json().find(entry->ladder) != std::string::npos, "route in CLI and server receipts");
        const auto exported = p::exports(actual, std::set<std::string>{});
        const auto setting = std::ranges::find_if(exported, [](const auto& item) {
            return item.first == "STRIX_FIXTURE_W4A4";
        });
        require(value == "off" ? setting == exported.end() : setting != exported.end() && setting->second == value,
                "only the effective value is exported");
    };
    check(p::Name::turbo, {p::Phase::prefill, 4096, 1, "dev", "fixture-exec"}, "a4", "applies x1.300");
    check(p::Name::turbo, {p::Phase::prefill, 2049, 1, "dev", "fixture-exec"}, "a4", "-> turbo=a4");
    check(p::Name::turbo, {p::Phase::prefill, 2048, 1, "dev", "fixture-exec"}, "a8", "-> fast=a8");
    check(p::Name::turbo, {p::Phase::prefill, 4097, 1, "dev", "fixture-exec"}, "a8", "within noise");
    check(p::Name::turbo, {p::Phase::prefill, 8193, 1, "dev", "fixture-exec"}, "off", "unmeasured regime");
    check(p::Name::turbo, {p::Phase::prefill, 4096, 1, "other", "fixture-exec"}, "off", "unmeasured regime");
    check(p::Name::turbo, {p::Phase::prefill, 4096, 2, "dev", "fixture-exec"}, "off", "unmeasured regime");
    check(p::Name::turbo, {p::Phase::prefill, 0, 1, "dev", "fixture-exec"}, "off", "unmeasured regime");
    check(p::Name::turbo, {p::Phase::decode, 4096, 8, "dev", "fixture-exec"}, "off", "slower");
    // FAST is measured slower in decode at one lane, so the FAST arm that runs there is not the arm the TURBO row was
    // measured over: the row is stale and the cell does not run.
    check(p::Name::turbo, {p::Phase::decode, 4096, 1, "dev", "fixture-exec"}, "off", "unmeasured regime");
    check(p::Name::fast, {p::Phase::prefill, 4096, 1, "dev", "fixture-exec"}, "a8", "-> fast=a8");
    check(p::Name::standard, {p::Phase::prefill, 4096, 1, "dev", "fixture-exec"}, "off", "");

    // A refused nonempty TURBO cell leaves the measured FAST cell and its exported status intact.
    auto refused_rows = kRows;
    refused_rows[0].ready[3] = false;
    const p::Engine refused{"refused", "", refused_rows, false, "refused-v1", {}, {}, rows};
    const p::Regime measured{p::Phase::prefill, 4096, 1, "dev", "fixture-exec"};
    const auto fast_run = resolve(refused, p::Name::fast, measured);
    const auto exported = [](const p::Resolution& run) {
        const auto env = p::exports(run, std::set<std::string>{});
        return std::ranges::find(env, std::pair<std::string, std::string>{"STRIX_FIXTURE_W4A4", "a8"}) != env.end();
    };
    require(exported(fast_run), "measured FAST fixture exports a8");
    for (const bool runnable : {true, false}) {
        refused_rows[0].runnable[3] = runnable;
        const auto turbo_run = resolve(refused, p::Name::turbo, measured);
        require(turbo_run.find(kLever)->value == "a8" &&
                    turbo_run.find(kLever)->status == fast_run.find(kLever)->status &&
                    turbo_run.find(kLever)->ladder.find("turbo=a4 refused") != std::string::npos &&
                    exported(turbo_run),
                "pending/unavailable TURBO replacement keeps the FAST export and records the refusal");
    }

    // TURBO nests FAST (review r75 P1-a): a same-value TURBO cell its column refuses (pending, runnable or not) keeps the
    // FAST cell, its status and its export; it never drops the FAST member to the builtin.
    refused_rows[0].value[3] = "a8";
    for (const bool runnable : {true, false}) {
        refused_rows[0].runnable[3] = runnable;
        const auto same = resolve(refused, p::Name::turbo, measured);
        require(same.find(kLever)->value == "a8" && same.find(kLever)->status == fast_run.find(kLever)->status &&
                    exported(same),
                "a same-value TURBO binding unavailable in this composition keeps the FAST cell");
    }

    // An explicit setting is the user's own: it runs as set in every regime.
    p::Inputs in;
    in.profile = p::Name::turbo;
    in.env = [](const char* key) -> const char* {
        return std::string_view(key) == "STRIX_FIXTURE_W4A4" ? "a4" : nullptr;
    };
    const auto opted = p::resolve(engine, in);
    require(opted.find(kLever)->value == "a4", "explicit environment wins in an unmeasured regime");
    in.cli.emplace_back(kLever, "a8");
    require(p::resolve(engine, in).find(kLever)->value == "a8", "explicit CLI beats explicit environment");

    // An engine without ladder rows: every lossy cell resolves to the builtin; an exact lever keeps its cell.
    const p::Engine bare{"bare", "", kRows, false, "bare-turbo-v1"};
    require(resolve(bare, p::Name::turbo).find(kLever)->value == "off",
            "no ladder rows: lossy cells do not apply");
    constexpr std::array<p::Lever, 1> exact_rows{{
        {"sched", "STRIX_FIXTURE_SCHED", "", {"", "2", "", ""}, "1", p::Kind::exact, {true, true, false, true},
         {true, true, false, true}, "", "", "fixture scheduling receipt", false, {}, false, {}, {{{"1", "v1"}, {"2", "v1"}}}},
    }};
    const p::Engine scheduling{"sched", "", exact_rows, false, "sched-turbo-v1"};
    require(resolve(scheduling, p::Name::turbo).find("sched")->value == "2",
            "exact (scheduling) levers are outside the ladder");
}

void defects() {
    const auto defect = [](p::LadderRow row) {
        const std::array rows{row};
        const p::Engine engine{"fixture", "", kRows, false, "fixture-turbo-v1", {}, {}, rows};
        return p::ladder_defect(engine);
    };
    const p::LadderRow good = measured_over({kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 1, 4096, 1, 1, 1.3, 0.02,
                                             "r"}, kFastArm);
    require(defect(good).empty(), "good row");
    auto row = good;
    row.value = {};
    require(!defect(row).empty(), "timing row naming no value (it would match every value)");
    row = good;
    for (const std::string_view unbound : {std::string_view{}, std::string_view{"fixture-w4a4@1=a8"},
                                           std::string_view{"ladder-arm/v1;fixture-w4a4@1=a8"},
                                           std::string_view{"ladder-arm/v2;fixture-exec;fixture-w4a4@1=a8;candidate=fixture-w4a4@1=a4"},
                                           std::string_view{"ladder-arm/v3"},
                                           std::string_view{"ladder-arm/v4;;fixture-w4a4@1=a8{v1};candidate=fixture-w4a4@1=a4{v1}"},
                                           std::string_view{"ladder-arm/v3;fixture-exec;fixture-w4a4@1=a8"}}) {
        row.over = unbound;
        require(!defect(row).empty(), "timing row without a versioned reference: " + std::string(unbound));
    }
    row = good;
    row.lever = "missing";
    require(!defect(row).empty(), "row naming no lever");
    row = good;
    row.value = "a8";
    require(!defect(row).empty(), "row value that is not the tier's cell");
    row = good;
    row.tier = p::Name::standard;
    require(!defect(row).empty(), "exact column row");
    row = good;
    row.receipt = "";
    require(!defect(row).empty(), "row without receipt");
    row = good;
    row.device = "";
    require(!defect(row).empty(), "row without device");
    row = good;
    row.speedup = 0.0;
    require(!defect(row).empty(), "unmeasured speedup");
    row = good;
    row.noise = std::numeric_limits<double>::quiet_NaN();
    require(!defect(row).empty(), "NaN noise");
    row = good;
    row.min_context = 0;
    require(!defect(row).empty(), "band covering the unknown context");
    row = good;
    row.min_lanes = 4;
    row.max_lanes = 2;
    require(!defect(row).empty(), "unordered lane band");
}

// The defect of one shipped ordinary timing row (empty: none). Resolved at its band corner on the execution it names,
// its cell must be eligible and judged over exactly the reference the row names; an entry alone proves nothing (a cell
// whose column admission is off has an entry with no judgment).
std::string shipped_row_defect(const p::Engine& engine, const p::LadderRow& row) {
    const auto actual = resolve(engine, row.tier, {row.phase, row.min_context, row.min_lanes, row.device,
                                                   p::ladder_reference_execution(row.over)});
    const auto* entry = actual.find(row.lever);
    if (entry == nullptr) return "no entry for the lever";
    const std::string step = " " + std::string(p::name(row.tier)) + "=" + std::string(row.value) + " ";
    if (entry->ladder.find(step) == std::string::npos) return "the cell was not judged in its regime: " + entry->ladder;
    if (entry->ladder_reference != row.over) return "judged over " + entry->ladder_reference + ", not the row's reference";
    if (entry->ladder.find(std::string(row.receipt)) == std::string::npos) return "the row did not decide: " + entry->ladder;
    return {};
}

// Controls of shipped_row_defect: a judged row passes; a mismatched reference and a row whose cell is never judged
// (column admission off) are both reported.
void shipped_rows() {
    const auto good = measured_over({kLever, "a8", p::Name::fast, p::Phase::prefill, "dev", 1, 4096, 1, 1, 1.2, 0.02,
                                     "r/shipped"}, kExactArm);
    const std::array rows{good};
    const p::Engine engine{"shipped", "", kRows, false, "shipped-v1", {}, {}, rows};
    require(shipped_row_defect(engine, good).empty(), "a judged shipped row: " + shipped_row_defect(engine, good));
    const auto wrong = measured_over(good, "fixture-w4a4@1=a6");
    const std::array wrong_rows{wrong};
    const p::Engine mismatched{"shipped", "", kRows, false, "shipped-v1", {}, {}, wrong_rows};
    require(!shipped_row_defect(mismatched, wrong).empty(), "a row naming a reference that never runs is reported");
    auto closed = kRows;
    closed[0].ready[1] = false;
    closed[0].runnable[1] = false;
    const p::Engine skipped{"shipped", "", closed, false, "shipped-v1", {}, {}, rows};
    require(!shipped_row_defect(skipped, good).empty(), "a row whose cell is never judged is reported");
}

// Review r61 P1: a receipt binds the execution outside the table and the implementation identities of both executions.
void execution_binding() {
    const p::LadderExecution base{"cuda", "product-a", {8192, 16384, 1, 2048, 2048, 1, 0, 128}, "weights=x"};
    const auto spelled = p::ladder_execution(base);
    require(!spelled.empty() && spelled.find("product-a") != std::string::npos, "execution spelling: " + spelled);
    const auto differs = [&](auto change, const char* what) {
        auto other = base;
        change(other);
        require(p::ladder_execution(other) != spelled, std::string("execution spelling ignores ") + what);
    };
    differs([](p::LadderExecution& e) { e.device = "product-b"; }, "the device product");
    differs([](p::LadderExecution& e) { e.backend = "hip"; }, "the backend");
    differs([](p::LadderExecution& e) { e.workload.prompt_tokens = 4096; }, "the prompt");
    differs([](p::LadderExecution& e) { e.workload.prefill_chunk = 8192; }, "the prefill chunk");
    differs([](p::LadderExecution& e) { e.workload.decode_rows = 4; }, "the decode rows");
    differs([](p::LadderExecution& e) { e.settings = "weights=y"; }, "the settings");
    for (const auto unknown : {+[](p::LadderExecution& e) { e.device = {}; },
                               +[](p::LadderExecution& e) { e.workload.prefill_chunk = 0; },
                               +[](p::LadderExecution& e) { e.workload.context_capacity = 0; }}) {
        auto other = base;
        unknown(other);
        require(p::ladder_execution(other).empty(), "an incomplete execution is not spelled");
    }
    // Review r63 P1 #1: the declared implementation identity of every value (Lever::identity, exact value match, no
    // prefix rule) is part of the key. A receipt kept fixed stops deciding when the candidate's identity changes, when a
    // reference-only identity changes, and when the run spells a value the table does not identify.
    static_assert(p::declared_identity({{{"on", "impl-v1"}, {"o", "x-v1"}}}, "on") == "impl-v1" &&
                  p::declared_identity({{{"on", "impl-v1"}}}, "o").empty() &&
                  p::declared_identity({{{"on", "impl-v1"}}}, "on (impl-v1)").empty() &&
                  p::declared_identity({}, "on").empty() && !p::identity_token("a b") && !p::identity_token("{x}"));
    constexpr std::string_view kSpelled =
        "ladder-arm/v4;fixture-exec;fixture-w4a4@1=off{v1};candidate=fixture-w4a4@1=a8{fixture-impl-v1};tier=fast";
    p::LadderRow timed{kLever, "a8", p::Name::fast, p::Phase::prefill, "dev", 1, 8192, 1, 1, 1.2, 0.02, "r/impl"};
    timed.over = kSpelled;
    const std::array rows{timed};
    const p::Regime at4k{p::Phase::prefill, 4096, 1, "dev", kExec};
    auto table = kRows;
    table[0].identity = {{{"off", "v1"}, {"a8", "fixture-impl-v1"}, {"a4", "v1"}}};
    const p::Engine v1{"identity", "", table, false, "identity-v1", {}, {}, rows};
    const auto applied = resolve(v1, p::Name::fast, at4k);
    require(applied.find(kLever)->value == "a8" && applied.find(kLever)->ladder_reference == kSpelled,
            "a receipt over the declared implementation applies: " + applied.find(kLever)->ladder_reference);
    table[0].identity = {{{"off", "v1"}, {"a8", "fixture-impl-v2"}, {"a4", "v1"}}};  // candidate identity v2
    const p::Engine v2{"identity", "", table, false, "identity-v1", {}, {}, rows};
    require(resolve(v2, p::Name::fast, at4k).find(kLever)->value == "off", "a new candidate identity retires the receipt");
    table[0].identity = {{{"off", "v2"}, {"a8", "fixture-impl-v1"}, {"a4", "v1"}}};  // reference-only identity v2
    const p::Engine ref2{"identity", "", table, false, "identity-v1", {}, {}, rows};
    require(resolve(ref2, p::Name::fast, at4k).find(kLever)->value == "off", "a new reference identity retires the receipt");
    // The target text never identifies a value (no display-target rule): the same receipt with the identity only in the
    // target does not decide.
    table[0].identity = {{{"off", "v1"}, {"a8", "v1"}, {"a4", "v1"}}};
    table[0].target = "a8 (fixture-impl-v1)";
    const p::Engine bare{"identity", "", table, false, "identity-v1", {}, {}, rows};
    require(resolve(bare, p::Name::fast, at4k).find(kLever)->value == "off", "a target is not an identity");
    // A table read with timing receipts must identify every value it can run (load-time error, never a default).
    table[0].identity = {{{"off", "v1"}, {"a4", "v1"}}};
    const p::Engine missing{"identity", "", table, false, "identity-v1", {}, {}, rows};
    require(!p::ladder_defect(missing).empty(), "an unidentified value is a ladder defect");
    bool threw = false;
    try { static_cast<void>(resolve(missing, p::Name::fast, at4k)); } catch (const std::logic_error&) { threw = true; }
    require(threw, "an unidentified value refuses the resolution");
}

// Walk all declared rows, including currently pending candidates. Marking the candidate ready here makes this
// test fail if a first binding can export the value before a timing receipt is registered.
void walk(const p::Engine& registered) {
    require(p::ladder_defect(registered).empty(), p::ladder_defect(registered));
    // Every shipped timing row is judged at its own band on its own execution: a row left behind by a changed builtin,
    // schedule, lower cell or column admission would silently stop deciding.
    // Every shipped grant names the identity its table declares for the granted value today (review r75 P1-b).
    for (const auto& row : registered.ladder) {
        if (!row.incumbent) continue;
        for (const auto& lever : registered.table)
            if (lever.name == row.lever)
                require(row.identity == p::declared_identity(lever.identity, row.value.empty()
                                                                                  ? lever.value[p::value_column(lever, row.tier)]
                                                                                  : row.value),
                        std::string(registered.tag) + "/" + std::string(row.lever) + " grant identity '" +
                            std::string(row.identity) + "' is not the declared identity of its value");
    }
    for (const auto& row : registered.ladder) {
        if (row.incumbent || !row.joint.empty()) continue;
        const auto defect = shipped_row_defect(registered, row);
        require(defect.empty(), std::string(registered.tag) + "/" + std::string(row.lever) + " shipped timing row (" +
                                    std::string(row.receipt) + "): " + defect);
    }
    std::vector<p::Lever> rows(registered.table.begin(), registered.table.end());
    auto engine = registered;
    engine.table = rows;
    p::Inputs implicit;
    implicit.env = [](const char*) -> const char* { return nullptr; };
    const auto exact = resolve(engine, p::Name::standard);
    require(p::resolve(engine, implicit).fingerprint() == exact.fingerprint(), "implicit native default equals EXACT");
    for (const auto& entry : exact.entries) {
        const auto wanted = entry.lever->value[0];
        const auto baseline = !wanted.empty() && entry.lever->ready[0] ? wanted : entry.lever->builtin;
        require(entry.value == baseline && entry.ladder.empty(), "EXACT retains the registered value");
    }
    for (auto& row : rows) {
        if (row.kind == p::Kind::exact) continue;
        row.ready.fill(true);
        row.provisional.fill(false);
    }
    auto probes = p::ladder_probes(engine);
    probes.push_back({p::Phase::prefill, 0, 1, {}});
    for (const auto& regime : probes) {
        for (const auto tier : {p::Name::fast, p::Name::turbo}) {
            if (tier == p::Name::turbo && engine.turbo_version.empty()) continue;
            const auto actual = resolve(engine, tier, regime);
            for (const auto& entry : actual.entries) {
                if (entry.lever->kind == p::Kind::exact || (entry.value == entry.lever->builtin && !p::applied(entry.status))) continue;
                require(entry.ladder.find("applies") != std::string::npos ||
                            entry.ladder.find("incumbent") != std::string::npos,
                        std::string(engine.tag) + "/" + std::string(entry.lever->name) + " applied without the ladder");
            }
        }
    }
}

void cuda_cli_server() {
    namespace qo = strix::cuda::qwen_options;
    p::LadderRow timing{"qsa_mma", "1", p::Name::fast, p::Phase::prefill, "fixture", 4096, 4096,
                        1, 1, 1.02, 0.005, "fixture-qsa"};
    qo::Request request;
    request.engine.model_gguf = "model.gguf";
    request.engine.tables_dir = "tables";
    request.engine.rope_table = "rope";
    request.inputs.env = [](const char*) -> const char* { return nullptr; };
    request.profile = p::Name::fast;
    // The reference the adapter runs under the cell (LadderArms::over): the row is measured over it.
    const auto reference_of = [&](std::string_view lever) {
        const auto resolved = qo::resolve(request, qo::Entry::cli);  // the entry lives in this resolution
        const auto* entry = resolved.resolution.find(lever);
        return entry ? entry->ladder_reference : std::string();
    };
    request.inputs.regime = {p::Phase::prefill, 4096, 1, "fixture", "fixture-exec"};
    const std::string qsa_reference = reference_of("qsa_mma");
    require(qsa_reference.starts_with("ladder-arm/v4;"), "Qwen CUDA reports the reference: " + qsa_reference);
    timing.over = qsa_reference;
    request.inputs.ladder = std::span(&timing, 1);
    for (const auto context : {4096U, 4097U}) {
        request.inputs.regime = {p::Phase::prefill, context, 1, "fixture", "fixture-exec"};
        const auto cli = qo::resolve(request, qo::Entry::cli);
        const auto server = qo::resolve(request, qo::Entry::serve);
        // Inside the timing band the row decides; outside it the incumbent grant keeps the cell main selected.
        require(cli.choice.qsa_mma && server.choice.qsa_mma == cli.choice.qsa_mma &&
                    cli.resolution.json().find(context == 4096U ? "applies x1.020" : "incumbent") != std::string::npos,
                "Qwen CUDA CLI and server select the same measured or incumbent kernel form");
        require(cli.resolution.json() == server.resolution.json() &&
                    server.resolution.json().find("strix-ladder/v1 qsa_mma") != std::string::npos,
                "Qwen CUDA CLI and server emit the same decision");
    }
    // Review r75 P2: a receipt that observably changes the decision on BOTH entries. The same complete key measured
    // slower turns the incumbent form off inside its band on the CLI and on the server alike (a server that ignored the
    // receipt would keep the grant), and leaves the grant outside the band.
    p::LadderRow slower = timing;
    slower.speedup = 0.90;
    slower.receipt = "fixture-qsa-slower";
    request.inputs.ladder = std::span(&slower, 1);
    for (const auto context : {4096U, 4097U}) {
        request.inputs.regime = {p::Phase::prefill, context, 1, "fixture", "fixture-exec"};
        const bool inside = context == 4096U;
        for (const auto entry : {qo::Entry::cli, qo::Entry::serve}) {
            const auto r = qo::resolve(request, entry);
            require(r.choice.qsa_mma == !inside &&
                        r.resolution.json().find(inside ? "slower x0.900" : "incumbent") != std::string::npos,
                    std::string(entry == qo::Entry::cli ? "CLI" : "server") +
                        ": a slower receipt of the complete key turns the form off inside its band only");
        }
    }
    // Provide evidence for a non-incumbent row through the native options adapter.
    const p::RegimeGate gate{"moe_route_bf16", "1", 4096, 4097, "fixture gate"};
    p::LadderRow candidate{"moe_route_bf16", "1", p::Name::fast, p::Phase::prefill, "fixture", 4096, 4096,
                           1, 1, 1.02, 0.005, "fixture route storage"};
    request.inputs.cli = {{"expert_fast", "off"}};
    request.inputs.regimes = std::span(&gate, 1);
    request.inputs.ladder = {};
    request.inputs.max_prompt_tokens = 4096;
    request.inputs.regime = {p::Phase::prefill, 4096, 1, "fixture", "fixture-exec"};
    const std::string route_reference = reference_of("moe_route_bf16");
    // The explicit switch is spelled as the lever's off value; the lever runs after the judged cell, at the tier below.
    require(route_reference.find(",expert_fast=0{") != std::string::npos,
            "the explicit expert_fast=off is part of the reference as its off value: " + route_reference);
    candidate.over = route_reference;
    request.inputs.ladder = std::span(&candidate, 1);
    const auto profile_selects_storage = [](const qo::Resolved& r) {
        const auto* entry = r.resolution.find("moe_route_bf16");
        return entry && entry->value == "1" && entry->status == p::Status::set;
    };
    require(profile_selects_storage(qo::resolve(request, qo::Entry::cli)) &&
            profile_selects_storage(qo::resolve(request, qo::Entry::serve)),
            "a measured non-incumbent candidate runs with the same known prompt on both entries");
    ++request.inputs.regime.context;
    require(!profile_selects_storage(qo::resolve(request, qo::Entry::cli)) &&
                !profile_selects_storage(qo::resolve(request, qo::Entry::serve)),
            "the neighboring context declines the candidate on both entries");
    request.inputs.regime = p::startup_prefill_regime("fixture");
    const auto startup = qo::resolve(request, qo::Entry::serve);
    require(!profile_selects_storage(startup) && startup.choice.qsa_mma,
            "startup declines the non-incumbent timing row and retains the incumbent form");
}

void custom_adapters() {
    constexpr p::Regime regime{p::Phase::prefill, 4096, 1, "dev", "fixture-exec"};
    const p::LadderRow sample = measured_over({kLever, "a8", p::Name::fast, p::Phase::prefill, "dev", 4096, 4096, 1, 1,
                                                   1.008, 0.002, "r/fast"}, kExactArm);
    for (unsigned probe = 0; probe < 4; ++probe) {
        auto row = sample;
        switch (probe) {
        case 1: row.receipt = {}; break;
        case 2: row.speedup = std::numeric_limits<double>::infinity(); break;
        case 3: row.noise = std::numeric_limits<double>::quiet_NaN(); break;
        default: break;
        }
        strix::qwen::profile::Inputs in;
        in.env = [](const char*) -> const char* { return nullptr; };
        in.regime = regime;
        in.ladder = std::span(&row, 1);
        const auto qwen = strix::qwen::profile::resolve(p::Name::fast, in, kRows);
        const auto* cell = qwen.find(kLever);
        require(cell && cell->value == (probe == 0 ? "a8" : "off") &&
                    qwen.json().find("strix-ladder/v1") != std::string::npos, "Qwen native selection checks timing");
    }

    namespace g = strix::gemma4::profile;
    g::Lever gemma{.name = kLever, .value = {"", "a8", ""}, .builtin = "off", .ready = {true, true, true},
                   .kind = p::Kind::bit_changing, .apps = unsigned(g::App::engine), .identity = {{{"off", "v1"}, {"a8", "v1"}}}};
    const auto g4 = g::resolve(p::Name::fast, g::App::engine, {}, std::span(&gemma, 1));
    require(g4.find(kLever)->value == "off", "Gemma cannot select a lossy unmeasured row");

    namespace r = strix::ds41::rtx::profile;
    r::Lever rtx{.name = kLever, .value = {"", "a8", ""}, .builtin = "off", .ready = {true, true, true},
                 .kind = p::Kind::bit_changing, .apps = unsigned(r::App::replica), .source = "fixture", .identity = {{{"off", "v1"}, {"a8", "v1"}}}};
    const auto ds = r::resolve(p::Name::fast, false, r::App::replica, {}, std::span(&rtx, 1));
    require(ds.find(kLever)->value == "off", "DeepSeek RTX cannot select a lossy unmeasured row");

    // The shared ladder is also instantiated for three-column adapters. TURBO
    // is not a valid column there, even if a caller bypasses native selection.
    require(p::value_column(rtx, p::Name::fast) == 1U, "legacy FAST column");
    require(p::value_column(rtx, p::Name::quality) == 2U, "legacy QUALITY column");
    bool rejected = false;
    try {
        (void)p::select_profile_cell(rtx, {}, p::Name::turbo, regime,
                                     [](unsigned) { return p::Status::set; }, false, {});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "three-column ladder refuses absent TURBO column");

    namespace glm = strix::glm::rtxpf::profile;
    glm::Lever candidate{.name = kLever, .env = "", .value = {"", "a8", ""}, .builtin = "off",
                         .ready = {true, true, true}, .kind = glm::Kind::bit_changing,
                         .apps = unsigned(glm::App::decode_service), .source = "fixture", .identity = {{{"off", "v1"}, {"a8", "v1"}}}};
    const auto gpu = glm::resolve(glm::Name::fast, false, glm::App::decode_service, {}, std::span(&candidate, 1));
    require(gpu.find(kLever)->value == "off", "GLM RTX cannot select a lossy unmeasured row");
}

// Incumbent preservation: every production entry point resolves FAST / TURBO (and refuses) exactly as before the
// ladder rule. One line per resolved entry: "<entry point> <tier> <regime> <lever>=<value> s<status>". The golden
// (tests/golden/profile_selections.txt) pins incumbent execution values and their receipt statuses.
// Additions and intended receipt changes are reviewed when the golden is updated.
// The individual adapter suites exercise --allow-pending separately.
template <class Res>
void lines(std::vector<std::string>& out, const std::string& where, const Res& res) {
    for (const auto& e : res.entries)
        if (e.lever != nullptr)
            out.push_back(where + " " + std::string(e.lever->name) + "=" + std::string(e.value) + " s" +
                          std::to_string(static_cast<int>(e.status)));
}
template <class F>
void guarded(std::vector<std::string>& out, const std::string& where, F&& f) {
    try {
        f();
    } catch (const std::exception& error) {
        out.push_back(where + " throws " + error.what());
    }
}

std::vector<std::string> selections() {
    namespace q = strix::qwen;
    namespace qd = strix::qwen35;
    namespace mm = strix::minimax::defaults;
    const auto none = [](const char*) -> const char* { return nullptr; };
    const auto tier = [](int t) { return std::string(t == 1 ? "fast" : "turbo"); };
    std::vector<std::string> out;
    const p::Engine engines[] = {
        {"deepseek41-pair", "", strix::deepseek41::pair_defaults::profile_table, false, "ds41-turbo-v1", {}, {},
         strix::deepseek41::pair_defaults::ladder_rows},
        {"deepseek41-rtx", "", strix::ds41::rtx::profile::shared_rows()},
        strix::glm::rank_defaults::product_profile_engine,
        mm::profile_engine(),
        {"qwen35", "", qd::profile::table, false, qd::profile::kTurboVersion},
        qd::hip_profile::engine(qd::hip_profile::App::decode),
        qd::hip_profile::engine(qd::hip_profile::App::spec),
        strix::hy4::profile::engine(),
        strix::inkling2::kMiniProfile,
    };
    for (const auto& engine : engines)
        for (int t : {1, 3}) {
            const auto where = "A " + std::string(engine.tag) + " " + tier(t);
            guarded(out, where, [&] {
                p::Inputs in;
                in.profile = static_cast<p::Name>(t);
                in.env = none;
                lines(out, where, p::resolve(engine, in));
            });
        }
    for (int t : {1, 3})
        for (std::uint64_t tokens : {0ULL, 1ULL, 4096ULL, 8192ULL, 8193ULL, 32768ULL}) {
            const auto where = "B qwen-hip " + tier(t) + " ctx=" + std::to_string(tokens);
            guarded(out, where, [&] {
                q::profile::Inputs in;
                in.env = none;
                in.max_prompt_tokens = tokens;
                in.regimes = q::profile::fast_regimes;
                in.regime = {p::Phase::prefill, tokens, 1, "gfx1151", "fixture-exec"};
                lines(out, where, q::profile::resolve(static_cast<p::Name>(t), in));
            });
            const auto cuda = "B qwen-cuda " + tier(t);
            guarded(out, cuda, [&] {
                q::profile::Inputs in;
                in.env = none;
                lines(out, cuda, q::cuda_profile::resolve(static_cast<p::Name>(t), in));
            });
        }
    for (int t : {1, 3})
        for (int w = 0; w < 3; ++w)
            for (int r = 0; r < 2; ++r)
                for (std::uint64_t tokens : {0ULL, 4096ULL, 8191ULL, 8192ULL, 32768ULL}) {
                    const auto where = "C qwen35-cuda-dense " + tier(t) + " w" + std::to_string(w) + " r" +
                                       std::to_string(r) + " ctx=" + std::to_string(tokens);
                    guarded(out, where, [&] {
                        qd::profile::Regime regime;
                        regime.weights = static_cast<qd::profile::Weights>(w);
                        regime.route = static_cast<qd::profile::Route>(r);
                        regime.prompt_tokens = tokens;
                        regime.device = "sm_120";
                        regime.arch = "sm_120";
                        lines(out, where, qd::profile::resolve(static_cast<p::Name>(t), regime, {})->resolution);
                    });
                }
    for (int t : {1, 3}) {
        namespace g = strix::gemma4::profile;
        for (auto app : {g::App::engine, g::App::decode_bench, g::App::prefill_bench}) {
            const auto where = "D gemma4 " + tier(t) + " app" + std::to_string(static_cast<unsigned>(app));
            guarded(out, where, [&] { lines(out, where, g::resolve(static_cast<p::Name>(t), app, {})); });
        }
        namespace r = strix::ds41::rtx::profile;
        for (auto app : {r::App::replica, r::App::prefill_service}) {
            const auto where = "E ds41-rtx " + tier(t) + " app" + std::to_string(static_cast<unsigned>(app));
            guarded(out, where, [&] { lines(out, where, r::resolve(static_cast<p::Name>(t), false, app, {})); });
        }
    }
    {
        namespace gl = strix::glm::rtxpf::profile;
        guarded(out, "F glm-rtx fast", [&] {
            lines(out, "F glm-rtx fast", gl::resolve(gl::Name::fast, false, gl::App::decode_service, {}));
        });
    }
    for (int t : {1, 3})
        for (int d = 0; d < 2; ++d)
            for (int topo = 0; topo < 2; ++topo)
                for (std::uint64_t tokens : {0ULL, 4096ULL, 8192ULL, 32768ULL}) {
                    const auto where = "G hy4 " + tier(t) + " d" + std::to_string(d) + " t" + std::to_string(topo) +
                                       " ctx=" + std::to_string(tokens);
                    guarded(out, where, [&] {
                        const strix::hy4::profile::Regime regime{static_cast<strix::hy4::profile::Device>(d),
                                                                 static_cast<strix::hy4::profile::Topology>(topo), tokens};
                        p::Inputs in;
                        in.profile = static_cast<p::Name>(t);
                        in.env = none;
                        lines(out, where, strix::hy4::profile::resolve(regime, in)->resolution);
                    });
                }
    for (const char* word : {"fast", "turbo"}) {
        const auto where = std::string("H minimax ") + word;
        guarded(out, where, [&] {
            std::map<std::string, std::string> env{{mm::kProductProfileEnv, word}};
            const auto lookup = [&env](const char* k) -> const char* {
                const auto it = env.find(k);
                return it == env.end() ? nullptr : it->second.c_str();
            };
            const auto policy = mm::prefill_policy(lookup);
            out.push_back(where + " msa_flash=" + std::to_string(policy.msa_flash) + " router_wmma=" +
                          std::to_string(int(policy.router_wmma)) + " bundle=" +
                          std::to_string(int(mm::fast_profile_selects_bundle(lookup))) + " id=" +
                          mm::prefill_identity(lookup));
            lines(out, where, mm::profile_resolution(lookup));
        });
    }
    return out;
}

void incumbents(const char* golden, bool write) {
    const auto now = selections();
    if (write) {
        std::ofstream file(golden, std::ios::binary);
        for (const auto& line : now) file << line << '\n';
        return;
    }
    std::ifstream file(golden, std::ios::binary);
    require(static_cast<bool>(file), std::string("golden selections not readable: ") + golden);
    std::vector<std::string> before;
    for (std::string line; std::getline(file, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) before.push_back(line);
    }
    require(before.size() == now.size(),
            "selection count " + std::to_string(now.size()) + " != golden " + std::to_string(before.size()));
    for (std::size_t i = 0; i < before.size() && i < now.size(); ++i)
        require(before[i] == now[i], "selection moved: '" + before[i] + "' -> '" + now[i] + "'");
}
// TURBO nests FAST: no TURBO column of any engine table leaves a FAST member off (a TURBO cell spelling the builtin
// over a FAST value). An empty TURBO cell runs the FAST cell. Beyond the spelling, the RESOLVED cells nest in every
// probed regime with the production admission: wherever FAST runs a non-builtin value, TURBO runs it too or runs an
// admitted replacement of its own; a TURBO proposal its column refuses (pending, not runnable) never resolves the
// cell below FAST (review r75: the MiniMax TURBO flash proposal dropped the FAST flash member).
void turbo_nests_fast(std::span<const p::Engine> engines) {
    for (const auto& engine : engines) {
        for (const auto& lever : engine.table) {
            const auto fast = lever.value[static_cast<unsigned>(p::Name::fast)];
            const auto turbo = lever.value[static_cast<unsigned>(p::Name::turbo)];
            require(fast.empty() || turbo.empty() || turbo == fast || turbo != lever.builtin,
                    std::string(engine.tag) + "/" + std::string(lever.name) + ": a TURBO cell leaves a FAST member off");
        }
        if (engine.turbo_version.empty()) continue;
        auto probes = p::ladder_probes(engine);
        probes.push_back({});
        for (const auto& regime : probes) {
            const auto f = resolve(engine, p::Name::fast, regime);
            const auto t = resolve(engine, p::Name::turbo, regime);
            require(f.entries.size() == t.entries.size(), std::string(engine.tag) + ": FAST and TURBO entry counts");
            for (std::size_t i = 0; i < f.entries.size() && i < t.entries.size(); ++i) {
                const auto& fe = f.entries[i];
                const auto& te = t.entries[i];
                if (!p::applied(fe.status) || fe.value == fe.lever->builtin) continue;
                require(p::applied(te.status) && te.value != te.lever->builtin,
                        std::string(engine.tag) + "/" + std::string(fe.lever->name) + ": TURBO resolves '" + te.value +
                            "' (status " + p::status_name(te.status) + ") where FAST runs '" + fe.value + "'");
            }
        }
    }
}

// A paired decision is admitted only by a row measured on the pair (LadderRow::joint): single-cell rows of either half
// never read it, and the joint row is invisible to the plain judge.
void paired_rows() {
    const p::Regime regime{p::Phase::prefill, 8192, 1, "dev", "fixture-exec"};
    p::LadderRow joint{"shadow", "fp8", p::Name::turbo, p::Phase::prefill, "dev", 8192, 8192, 1, 1, 1.10, 0.01, "pair receipt"};
    joint.joint = "reader";
    const p::JointLadderScope scope{"device-id", "fast@1", "candidate-base@1", "shadow@1", "reader@1",
                                    {8192, 32768, 1, 2048, 16, 1, 0, 128}, "on"};
    joint.joint_scope = scope;
    p::LadderRow plain = joint;
    plain.joint = {};
    plain.receipt = "shadow alone";
    const p::LadderRow both[] = {joint, plain};
    require(p::ladder_judge_pair(std::span(&plain, 1), "shadow", "fp8", p::Name::turbo, regime, "reader", scope).rung !=
                p::Rung::applies, "a single-cell row admitted the pair");
    require(p::ladder_judge(std::span(&joint, 1), "shadow", "fp8", p::Name::turbo, regime, regime.execution, "v1").rung != p::Rung::applies,
            "a joint row decided the plain cell");
    const auto verdict = p::ladder_judge_pair(both, "shadow", "fp8", p::Name::turbo, regime, "reader", scope);
    require(verdict.rung == p::Rung::applies && verdict.row == &both[0], "the joint row decides the pair");
    require(p::ladder_judge_pair(both, "shadow", "fp8", p::Name::turbo, regime, "other", scope).rung != p::Rung::applies,
            "a joint row decided another partner");
    p::LadderRow slow = joint;
    slow.speedup = 0.97;
    require(p::ladder_judge_pair(std::span(&slow, 1), "shadow", "fp8", p::Name::turbo, regime, "reader", scope).rung == p::Rung::slower,
            "a losing joint row must not admit the pair");
}
// A new gate row (a wider prompt band for a qualified cell) is not an incumbent grant: the cell keeps the tier below
// beyond the frozen band until a matching timing row is registered.
void hy4_new_quality_rows() {
    namespace h = strix::hy4::profile;
    const auto none = [](const char*) -> const char* { return nullptr; };
    p::Inputs in;
    in.profile = p::Name::fast;
    in.env = none;
    // No timing row prices another width inside the gated window class: the 8K greedy execution keeps the frozen 4096
    // grant in FAST and TURBO (r89 on #1310).
    const h::Regime measured{h::Device::gfx1151, h::Topology::one_node, 8192, "hip/rank/single/full/greedy/8384/64"};
    for (const auto tier : {p::Name::fast, p::Name::turbo}) {
        in.profile = tier;
        const auto resolution = h::resolve(measured, in);
        const auto* rows = resolution->find("prompt_rows");
        require(rows->value == "4096" && rows->status == p::Status::set,
                "Hy4 keeps the incumbent 4096 width without a matched timing row");
    }
    in.profile = p::Name::fast;
    const h::Regime beyond{h::Device::gfx1151, h::Topology::one_node, 40000, "fixture-exec"};
    const h::Regime inside{h::Device::gfx1151, h::Topology::one_node, 32568, "fixture-exec"};
    require(h::resolve(inside, in)->find("prompt_rows")->value == "4096", "frozen grant: the qualified band still runs");
    std::vector<h::Qualified> rows(std::begin(h::qualified), std::end(h::qualified));
    rows.push_back({h::Device::gfx1151, h::Topology::one_node, {"prompt_rows", "4096", 0, 65536, "fixture wider gate row"}});
    rows.push_back({h::Device::gfx1151, h::Topology::one_node, {"pf_mha", "1", 0, 65536, "fixture wider gate row"}});
    const auto wide = h::resolve(beyond, in, rows);
    require(wide->find("prompt_rows")->value == "0" && wide->find("pf_mha")->value == "0",
            "a new quality row beyond the frozen band must leave the cell at the lower tier without a timing row");
    require(h::resolve(inside, in, rows)->find("prompt_rows")->value == "4096", "the frozen band keeps its cell");
    // Exercise the same adapter as CLI and serve, including its input shape and topology checks.
    h::Measured timings[]{{h::Topology::one_node,
        {"prompt_rows", "4096", p::Name::fast, p::Phase::prefill, "gfx1151", 40000, 65536, 1, 1, 1.05, 0.01,
         "fixture timing"}}};
    // Measured over the reference the adapter runs there (LadderArms::over).
    const std::string hy4_reference = h::resolve(beyond, in, rows)->find("prompt_rows")->ladder_reference;
    require(hy4_reference.starts_with("ladder-arm/v4;"), "Hy4 reports the reference: " + hy4_reference);
    timings[0].timing.over = hy4_reference;
    const auto resolved = h::resolve(beyond, in, rows, h::incumbent_grants, timings);
    require(resolved->find("prompt_rows")->value == "4096", "a matching timing row admits the new band");
    auto switched = in;
    switched.env = [](const char* key) -> const char* {
        return std::string_view(key) == "STRIX_HY4_PF_A8_PRODUCER" ? "off" : nullptr;
    };
    require(h::resolve(beyond, switched, rows, h::incumbent_grants, timings)->find("prompt_rows")->value == "4096",
            "legacy flag spelling with the same runtime value keeps the timing match");
    switched.env = [](const char* key) -> const char* {
        return std::string_view(key) == "STRIX_HY4_PF_A8_PRODUCER" ? "0legacy" : nullptr;
    };
    require(h::resolve(beyond, switched, rows, h::incumbent_grants, timings)->find("prompt_rows")->value == "0",
            "changing the effective Hy4 flag value invalidates the timing match");
    auto other = beyond;
    --other.prompt_tokens;
    require(h::resolve(other, in, rows, h::incumbent_grants, timings)->find("prompt_rows")->value == "0",
            "neighboring prompt declines the same measured row");
    other = beyond;
    other.topology = h::Topology::pair;
    rows.push_back({h::Device::gfx1151, h::Topology::pair, {"prompt_rows", "4096", 0, 65536, "fixture wider pair gate"}});
    require(h::resolve(other, in, rows, h::incumbent_grants, timings)->find("prompt_rows")->value == "0",
            "pair topology declines the one-node measured row");
}

// Ordinary timing rows decide only the comparison they measured (LadderRow::over), --allow-pending never runs a measured
// loser, and a startup resolution (context 0) runs grants only.
void bound_evidence() {
    const auto run = [](const p::Engine& engine, p::Name tier, p::Regime regime, p::Pairs cli = {},
                        bool allow_pending = false) {
        p::Inputs in;
        in.profile = tier;
        in.regime = regime;
        in.cli = std::move(cli);
        in.allow_pending = allow_pending;
        in.env = [](const char*) -> const char* { return nullptr; };
        return p::resolve(engine, in);
    };
    const auto value_of = [](const p::Resolution& res, std::string_view lever) {
        const auto* entry = res.find(lever);
        return entry ? entry->value : std::string("<none>");
    };
    const auto route_of = [](const p::Resolution& res, std::string_view lever) {
        const auto* entry = res.find(lever);
        return entry ? entry->ladder : std::string();
    };
    const p::Regime at4k{p::Phase::prefill, 4096, 1, "dev", "fixture-exec"};
    const p::LadderRow fast_a8 = measured_over({kLever, "a8", p::Name::fast, p::Phase::prefill, "dev", 1, 8192, 1, 1, 1.20,
                                                    0.02, "r/a8"}, kExactArm);
    const p::LadderRow turbo_a4 = measured_over({kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 1, 8192, 1, 1,
                                                     1.30, 0.02, "r/a4-over-a8"}, kFastArm);
    {
        // The review case: a new FAST value qualifies and gets its own FAST record; the TURBO record (denominator a8)
        // is left untouched. It must not decide over the new FAST arm.
        const std::array rows{fast_a8, turbo_a4};
        const p::Engine before{"bound", "", kRows, false, "bound-v1", {}, {}, rows};
        require(value_of(run(before, p::Name::turbo, at4k), kLever) == "a4", "TURBO row over the running FAST arm applies");
        auto requalified = kRows;
        requalified[0].value[1] = "a6";
        const p::LadderRow fast_a6 = measured_over({kLever, "a6", p::Name::fast, p::Phase::prefill, "dev", 1, 8192, 1, 1,
                                                        1.15, 0.02, "r/a6"}, kExactArm);
        const std::array after_rows{fast_a6, turbo_a4};
        const p::Engine after{"bound", "", requalified, false, "bound-v1", {}, {}, after_rows};
        const auto res = run(after, p::Name::turbo, at4k);
        require(value_of(res, kLever) == "a6", "stale TURBO row (measured over a8) keeps the new FAST cell a6");
        require(route_of(res, kLever).find("unmeasured regime") != std::string::npos,
                "the route names the stale evidence: " + route_of(res, kLever));
        // Re-timed over the new FAST arm, the TURBO cell runs again.
        const std::array retimed_rows{fast_a6, measured_over({kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 1,
                                                                  8192, 1, 1, 1.25, 0.02, "r/a4-over-a6"},
                                                                 "fixture-w4a4@1=a6")};
        const p::Engine retimed{"bound", "", requalified, false, "bound-v1", {}, {}, retimed_rows};
        require(value_of(run(retimed, p::Name::turbo, at4k), kLever) == "a4", "TURBO row re-timed over a6 applies");
    }
    {
        // A later incumbent grant is part of the execution an earlier timed cell extends (review r61): the FAST row of
        // the first lever measured with the later lever off does not decide where the grant turns it on, and a
        // combination measured to lose keeps the incumbent execution.
        constexpr std::string_view kOther = "fixture-other@1";
        std::array<p::Lever, 2> table{kRows[0], kRows[0]};
        table[1].name = kOther;
        table[1].env = "STRIX_FIXTURE_OTHER";
        table[1].value = {"", "on", "", ""};
        table[1].identity = {{{"off", "v1"}, {"on", "v1"}}};
        const auto grant = p::incumbent_cell(kOther, "on", p::Name::fast, "r/other-grant", "v1");
        const auto fast_row = [](double speedup, std::string_view over, std::string_view receipt) {
            return measured_over({kLever, "a8", p::Name::fast, p::Phase::prefill, "dev", 1, 8192, 1, 1, speedup, 0.02,
                                  receipt}, over);
        };
        const std::array isolated{fast_row(1.20, "fixture-w4a4@1=off,fixture-other@1=off", "r/a8-without-other"), grant};
        const p::Engine without{"later-grant", "", table, false, "later-grant-v1", {}, {}, isolated};
        const auto stale = run(without, p::Name::fast, at4k);
        require(value_of(stale, kLever) == "off" && value_of(stale, kOther) == "on" &&
                    route_of(stale, kLever).find("unmeasured regime") != std::string::npos &&
                    stale.find(kLever)->ladder_reference ==
                        "ladder-arm/v4;fixture-exec;fixture-w4a4@1=off{v1},fixture-other@1=on{v1};candidate=fixture-w4a4@1=a8{v1};tier=fast",
                "a row measured without the later grant does not decide the combined arm: " + route_of(stale, kLever));
        const std::array losing{fast_row(0.95, "fixture-w4a4@1=off,fixture-other@1=on", "r/a8-with-other-loses"),
                                isolated[0], grant};
        const p::Engine loses{"later-grant", "", table, false, "later-grant-v1", {}, {}, losing};
        const auto lost = run(loses, p::Name::fast, at4k);
        require(value_of(lost, kLever) == "off" && value_of(lost, kOther) == "on" &&
                    route_of(lost, kLever).find("slower") != std::string::npos,
                "a combined arm measured to lose keeps the incumbent FAST arm: " + route_of(lost, kLever));
        const std::array combined{fast_row(1.10, "fixture-w4a4@1=off,fixture-other@1=on", "r/a8-with-other"), grant,
                                  measured_over({kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 1, 8192, 1, 1,
                                                 1.30, 0.02, "r/a4-over-both"}, "fixture-w4a4@1=a8,fixture-other@1=on")};
        const p::Engine engine{"later-grant", "", table, false, "later-grant-v1", {}, {}, combined};
        const auto plain = run(engine, p::Name::turbo, at4k);
        require(value_of(plain, kLever) == "a4" && value_of(plain, kOther) == "on",
                "rows measured over the complete arms apply: " + route_of(plain, kLever));
        // An explicit setting of the later lever changes both executions: neither row decides.
        const auto overridden = run(engine, p::Name::turbo, at4k, {{std::string(kOther), "off"}});
        require(value_of(overridden, kLever) == "off",
                "an override of a later lever makes the rows stale: " + route_of(overridden, kLever));
        // A repeated option resolves to its last value in the reference and in the execution alike (review r61).
        const auto repeated = run(engine, p::Name::turbo, at4k, {{std::string(kOther), "off"}, {std::string(kOther), "on"}});
        require(value_of(repeated, kLever) == "a4" && value_of(repeated, kOther) == "on",
                "the last value of a repeated option is the one timed and run: " + route_of(repeated, kLever));
        const auto repeated_off = run(engine, p::Name::turbo, at4k, {{std::string(kOther), "on"}, {std::string(kOther), "off"}});
        require(value_of(repeated_off, kLever) == "off" && value_of(repeated_off, kOther) == "off",
                "a repeated option ending in off is spelled off: " + route_of(repeated_off, kLever));
        // The same receipt on another execution (device product, workload or setting) or an unknown one decides nothing.
        for (const std::string_view execution : {std::string_view{"fixture-exec-other-device"}, std::string_view{}}) {
            auto elsewhere = at4k;
            elsewhere.execution = execution;
            const auto other = run(engine, p::Name::turbo, elsewhere);
            require(value_of(other, kLever) == "off" && value_of(other, kOther) == "on",
                    "a receipt never crosses executions: " + std::string(execution));
        }
    }
    {
        // --allow-pending runs only cells no measurement decides: measured slower / within noise stay below.
        for (const double speedup : {0.98, 1.005}) {
            const std::array rows{measured_over({kLever, "a8", p::Name::fast, p::Phase::prefill, "dev", 1, 4096, 1, 1, speedup,
                                                 0.01, "r/a8-loser"}, kExactArm)};
            const p::Engine engine{"bench", "", kRows, false, "bench-v1", {}, {}, rows};
            const auto res = run(engine, p::Name::fast, at4k, {}, true);
            require(value_of(res, kLever) == "off" &&
                        route_of(res, kLever).find("unqualified benchmark") == std::string::npos,
                    "allow-pending never runs a measured FAST loser x" + std::to_string(speedup));
            const auto unmeasured = run(engine, p::Name::fast, {p::Phase::prefill, 8192, 1, "dev", "fixture-exec"}, {}, true);
            require(value_of(unmeasured, kLever) == "a8" &&
                        route_of(unmeasured, kLever).find("unqualified benchmark") != std::string::npos,
                    "allow-pending runs the unmeasured regime as a benchmark arm");
        }
    }
    {
        // Startup (context 0): no timing band holds an unknown regime, so a grant keeps its cell (timed or not) and an
        // ungranted timed cell stays below; once the prompt is known the rows decide.
        const std::array granted{p::incumbent_cell(kLever, "a8", p::Name::fast, "r/grant", "v1")};
        const p::Engine grant_only{"grant", "", kRows, false, "grant-v1", {}, {}, granted};
        const p::Regime startup = p::startup_prefill_regime("dev");
        require(value_of(run(grant_only, p::Name::fast, startup), kLever) == "a8", "untimed grant keeps its cell at startup");
        const std::array timed_grant{p::incumbent_cell(kLever, "a8", p::Name::fast, "r/grant", "v1"), fast_a8};
        const p::Engine timed{"grant", "", kRows, false, "grant-v1", {}, {}, timed_grant};
        require(value_of(run(timed, p::Name::fast, startup), kLever) == "a8", "timed grant keeps its cell at startup");
        const std::array timed_only{fast_a8};
        const p::Engine ungranted{"grant", "", kRows, false, "grant-v1", {}, {}, timed_only};
        require(value_of(run(ungranted, p::Name::fast, startup), kLever) == "off", "ungranted timed cell stays below at startup");
        // Review r75 P1-b: a grant binds the implementation identity it was given to. The same grant over a row whose
        // declared identity of the granted value changed (no new receipt) grants nothing: the cell resolves below.
        auto bumped = kRows;
        for (auto& item : bumped[0].identity)
            if (item.value == "a8") item.identity = "v2";
        const p::Engine stale{"grant", "", bumped, false, "grant-v1", {}, {}, granted};
        require(value_of(run(stale, p::Name::fast, startup), kLever) == "off",
                "r75 P1-b a fixed grant does not admit a changed implementation identity");
        require(p::ladder_judge(granted, kLever, "a8", p::Name::fast, startup, {}, "v1").rung == p::Rung::incumbent &&
                    p::ladder_judge(granted, kLever, "a8", p::Name::fast, startup, {}, "v2").rung == p::Rung::unmeasured &&
                    p::ladder_judge(granted, kLever, "a8", p::Name::fast, startup, {}, {}).rung == p::Rung::unmeasured,
                "r75 P1-b a grant matches on its identity only");
        require(value_of(run(ungranted, p::Name::fast, at4k), kLever) == "a8", "the row decides once the prompt is known");
        require(value_of(run(timed, p::Name::fast, {p::Phase::prefill, 16384, 1, "dev", "fixture-exec"}), kLever) == "a8",
                "outside the timed bands the grant keeps its cell");
    }
    {
        // r51: the reference names every lever with its effective value, builtins included. A FAST row measured over
        // the builtin off stops deciding once the builtin itself is another execution.
        const std::array rows{fast_a8};
        const p::Engine engine{"builtin", "", kRows, false, "builtin-v1", {}, {}, rows};
        require(value_of(run(engine, p::Name::fast, at4k), kLever) == "a8", "FAST row over the builtin off applies");
        auto rebuilt = kRows;
        rebuilt[0].builtin = "a2";
        rebuilt[0].identity = {{{"a2", "v1"}, {"off", "v1"}, {"a8", "v1"}, {"a4", "v1"}}};
        const p::Engine changed{"builtin", "", rebuilt, false, "builtin-v1", {}, {}, rows};
        const auto res = run(changed, p::Name::fast, at4k);
        require(value_of(res, kLever) == "a2" &&
                    route_of(res, kLever).find("unmeasured regime") != std::string::npos,
                "a changed builtin makes the FAST receipt stale: " + route_of(res, kLever));
    }
    {
        // An exact schedule lever is part of the reference: overriding it changes the timing denominator.
        constexpr std::string_view kSchedule = "fixture-schedule@1";
        std::array<p::Lever, 2> table{kRows[0], kRows[0]};
        table[1].name = kSchedule;
        table[1].env = "STRIX_FIXTURE_SCHEDULE";
        table[1].value = {"", "", "", ""};
        table[1].builtin = "1";
        table[1].off = {};
        table[1].kind = p::Kind::exact;
        table[1].identity = {{{"1", "v1"}, {"2", "v1"}}};
        const std::array rows{measured_over(fast_a8, "fixture-w4a4@1=off,fixture-schedule@1=1")};
        const p::Engine engine{"schedule", "", table, false, "schedule-v1", {}, {}, rows};
        require(value_of(run(engine, p::Name::fast, at4k), kLever) == "a8", "FAST row over the default schedule applies");
        const auto res = run(engine, p::Name::fast, at4k, {{std::string(kSchedule), "2"}});
        require(value_of(res, kLever) == "off" && value_of(res, kSchedule) == "2",
                "an exact schedule override makes the FAST receipt stale: " + route_of(res, kLever));
        require(res.find(kLever)->ladder_reference ==
                    "ladder-arm/v4;fixture-exec;fixture-w4a4@1=off{v1},fixture-schedule@1=2{v1};candidate=fixture-w4a4@1=a8{v1};tier=fast",
                "the reference spells the override: " + res.find(kLever)->ladder_reference);
        // A tier-specific exact schedule is part of the tier's incumbent execution (review r61): the row timed under
        // the default schedule does not decide where FAST runs schedule 2, a row timed under schedule 2 does.
        table[1].value = {"", "2", "", ""};
        const p::Engine tiered{"schedule", "", table, false, "schedule-v1", {}, {}, rows};
        const auto at_two = run(tiered, p::Name::fast, at4k);
        require(value_of(at_two, kSchedule) == "2" && value_of(at_two, kLever) == "off",
                "a row timed under another schedule does not decide the FAST schedule: " + route_of(at_two, kLever));
        const std::array timed_two{measured_over(fast_a8, "fixture-w4a4@1=off,fixture-schedule@1=2")};
        const p::Engine retimed{"schedule", "", table, false, "schedule-v1", {}, {}, timed_two};
        require(value_of(run(retimed, p::Name::fast, at4k), kLever) == "a8", "a row timed under the FAST schedule applies");
    }
    {
        // A timing row names its value: an empty value covers no cell (it would also match a6).
        p::LadderRow any = fast_a8;
        any.value = {};
        const std::array rows{any};
        const p::Engine engine{"any", "", kRows, false, "any-v1", {}, {}, rows};
        require(value_of(run(engine, p::Name::fast, at4k), kLever) == "off" && !p::ladder_defect(engine).empty(),
                "an ordinary row without a value never decides and is a defect");
    }
    {
        // A TURBO row explicitly measured over the builtin off applies when the FAST a8 candidate is measured slower
        // and FAST runs off (the reference that actually runs; no projected-column recheck).
        const std::array rows{measured_over({kLever, "a8", p::Name::fast, p::Phase::prefill, "dev", 1, 8192, 1, 1, 0.95,
                                             0.01, "r/a8-loses"}, kExactArm),
                              measured_over({kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 1, 8192, 1, 1, 1.30,
                                             0.02, "r/a4-over-off"}, kExactArm)};
        const p::Engine engine{"over-off", "", kRows, false, "over-off-v1", {}, {}, rows};
        require(value_of(run(engine, p::Name::fast, at4k), kLever) == "off", "FAST a8 measured slower runs off");
        require(value_of(run(engine, p::Name::turbo, at4k), kLever) == "a4", "TURBO row over the running off applies");
    }
    {
        // Two isolated TURBO wins over the same FAST arm do not admit their unmeasured combination: cells of one tier
        // compose in table order, so the second cell decides only on a row measured over the first.
        constexpr std::string_view kSecond = "fixture-second@1";
        std::array<p::Lever, 2> table{kRows[0], kRows[0]};
        table[0].value = {"", "", "", "a4"};
        table[1].name = kSecond;
        table[1].env = "STRIX_FIXTURE_SECOND";
        table[1].value = {"", "", "", "b4"};
        table[1].identity = {{{"off", "v1"}, {"b4", "v1"}}};
        constexpr std::string_view kBase = "fixture-w4a4@1=off,fixture-second@1=off";
        constexpr std::string_view kWithFirst = "fixture-w4a4@1=a4,fixture-second@1=off";
        const auto turbo_row = [](std::string_view lever, std::string_view value, double speedup, std::string_view over,
                                  std::string_view receipt) {
            return measured_over({lever, value, p::Name::turbo, p::Phase::prefill, "dev", 4096, 4096, 1, 1, speedup, 0.02,
                                  receipt}, over);
        };
        const std::array isolated{turbo_row(kLever, "a4", 1.10, kBase, "r/a4-alone"),
                                  turbo_row(kSecond, "b4", 1.10, kBase, "r/b4-alone")};
        const p::Engine alone{"compose", "", table, false, "compose-v1", {}, {}, isolated};
        const auto two = run(alone, p::Name::turbo, at4k);
        require(value_of(two, kLever) == "a4" && value_of(two, kSecond) == "off" &&
                    route_of(two, kSecond).find("unmeasured regime") != std::string::npos,
                "two isolated wins: only the first measured cell runs: " + route_of(two, kSecond));
        const std::array losing{isolated[0], isolated[1], turbo_row(kSecond, "b4", 0.95, kWithFirst, "r/b4-joint-loses")};
        const p::Engine joint_loses{"compose", "", table, false, "compose-v1", {}, {}, losing};
        const auto lost = run(joint_loses, p::Name::turbo, at4k);
        require(value_of(lost, kLever) == "a4" && value_of(lost, kSecond) == "off" &&
                    route_of(lost, kSecond).find("slower") != std::string::npos,
                "a losing joint result keeps the combination off: " + route_of(lost, kSecond));
        const std::array winning{isolated[0], turbo_row(kSecond, "b4", 1.05, kWithFirst, "r/b4-joint-wins")};
        const p::Engine joint_wins{"compose", "", table, false, "compose-v1", {}, {}, winning};
        const auto both = run(joint_wins, p::Name::turbo, at4k);
        require(value_of(both, kLever) == "a4" && value_of(both, kSecond) == "b4",
                "a combination measured over the first cell runs both");
    }
    {
        // An --allow-pending benchmark cell runs, so it is part of the reference of every later cell: a FAST row
        // measured over the default execution (the pending cell at its builtin) does not qualify the later cell there.
        constexpr std::string_view kLater = "fixture-later@1";
        std::array<p::Lever, 2> table{kRows[0], kRows[0]};
        table[0].value = {"", "a8", "", ""};
        table[0].ready = {true, false, false, true};
        table[0].target = "fixture pending";
        table[1].name = kLater;
        table[1].env = "STRIX_FIXTURE_LATER";
        table[1].value = {"", "b8", "", ""};
        table[1].identity = {{{"off", "v1"}, {"b8", "v1"}, {"b4", "v1"}}};
        const std::array rows{measured_over({kLater, "b8", p::Name::fast, p::Phase::prefill, "dev", 4096, 4096, 1, 1,
                                             1.10, 0.02, "r/b8-over-default"},
                                            "fixture-w4a4@1=off,fixture-later@1=off")};
        const p::Engine engine{"benchmark-arm", "", table, false, "benchmark-arm-v1", {}, {}, rows};
        const auto plain = run(engine, p::Name::fast, at4k);
        require(value_of(plain, kLever) == "off" && plain.find(kLater)->status == p::Status::set &&
                    value_of(plain, kLater) == "b8",
                "the row over the default execution qualifies the later cell: " + route_of(plain, kLater));
        const auto bench = run(engine, p::Name::fast, at4k, {}, true);
        require(value_of(bench, kLever) == "a8" && bench.find(kLever)->status == p::Status::unqualified,
                "--allow-pending runs the pending cell as an unqualified benchmark");
        require(bench.find(kLater)->ladder_reference ==
                        "ladder-arm/v4;fixture-exec;fixture-w4a4@1=a8{v1},fixture-later@1=off{v1};candidate=fixture-later@1=b8{v1};tier=fast" &&
                    bench.find(kLater)->status != p::Status::set,
                "the benchmark cell is part of the later cell's reference: " + route_of(bench, kLater));
        // TURBO falls back to the FAST cells the arms resolved, the unqualified benchmark cell included (review r61):
        // a TURBO receipt measured over that cell decides, one measured over the dropped fallback does not.
        table[1].value = {"", "b8", "", "b4"};
        const auto turbo_row = [](std::string_view over, std::string_view receipt) {
            return measured_over({kLater, "b4", p::Name::turbo, p::Phase::prefill, "dev", 4096, 4096, 1, 1, 1.10, 0.02,
                                  receipt}, over);
        };
        const std::array over_bench{rows[0], turbo_row("fixture-w4a4@1=a8,fixture-later@1=b8", "r/b4-over-bench")};
        const p::Engine with_bench{"benchmark-arm", "", table, false, "benchmark-arm-v1", {}, {}, over_bench};
        const auto bench_turbo = run(with_bench, p::Name::turbo, at4k, {}, true);
        require(value_of(bench_turbo, kLever) == "a8" && bench_turbo.find(kLever)->status == p::Status::unqualified &&
                    value_of(bench_turbo, kLater) == "b4" &&
                    bench_turbo.find(kLater)->ladder_reference ==
                        "ladder-arm/v4;fixture-exec;fixture-w4a4@1=a8{v1},fixture-later@1=b8{v1};candidate=fixture-later@1=b4{v1};tier=turbo",
                "TURBO exports the benchmark cell its receipt was timed over: " + route_of(bench_turbo, kLater));
        const std::array over_dropped{rows[0], turbo_row("fixture-w4a4@1=off,fixture-later@1=b8", "r/b4-over-off")};
        const p::Engine with_dropped{"benchmark-arm", "", table, false, "benchmark-arm-v1", {}, {}, over_dropped};
        const auto mismatch = run(with_dropped, p::Name::turbo, at4k, {}, true);
        require(value_of(mismatch, kLever) == "a8" && value_of(mismatch, kLater) == "b8",
                "a receipt over a fallback the run does not export decides nothing: " + route_of(mismatch, kLater));
    }
}
// ---- Review r63 D: the receipt matching holds by construction, proven by enumeration over every engine table ----

// The shared row view of a custom adapter table (Gemma 4, GLM RTX producer): the lever data the shared ladder reads.
template <class Row>
std::vector<p::Lever> shared_view(std::span<const Row> table) {
    std::vector<p::Lever> out;
    for (const auto& l : table) {
        p::Lever row{};
        row.name = l.name;
        row.env = l.env;
        for (std::size_t c = 0; c < l.value.size() && c < row.value.size(); ++c) {
            row.value[c] = l.value[c];
            row.ready[c] = l.ready[c];
            row.runnable[c] = l.runnable[c];
        }
        row.builtin = l.builtin;
        row.kind = l.kind == decltype(l.kind)::exact ? p::Kind::exact : p::Kind::bit_changing;
        if constexpr (requires { l.off; }) row.off = l.off;
        row.identity = l.identity;
        out.push_back(row);
    }
    return out;
}

// Every entry of every tier: value, status, route and reference (what a receipt could change).
std::string spelled_resolution(const p::Engine& engine, const p::Regime& regime, std::optional<p::Name> only = {}) {
    std::string out;
    for (const auto tier : {p::Name::standard, p::Name::fast, p::Name::turbo}) {
        if ((tier == p::Name::turbo && engine.turbo_version.empty()) || (only && tier != *only)) continue;
        const auto res = resolve(engine, tier, regime);
        for (const auto& e : res.entries)
            out += std::string(e.lever->name) + "=" + e.value + "/" + p::status_name(e.status) + "/" + e.ladder + "/" +
                   e.ladder_reference + ";";
        out += "|";
    }
    return out;
}

constexpr p::LadderExecution kEnumExecution{"hip", "product-a", {8192, 16384, 1, 2048, 2048, 1, 0, 128}, "settings-a"};

// The execution segment of `over` replaced by `execution`.
std::string with_execution(std::string_view over, std::string_view execution) {
    const auto first = over.find(';');
    const auto second = over.find(';', first + 1U);
    return std::string(over.substr(0, first + 1U)) + std::string(execution) + std::string(over.substr(second));
}

struct EnumerationCount {
    std::size_t engines = 0, cells = 0, mutations = 0;
};

// For every judged cell (FAST and TURBO) of `registered`: a decisive receipt over its complete key (a gain where the cell
// does not already run; a loss where an incumbent grant runs it) must decide it, and the same receipt with any ONE key
// field changed (lever, value, tier, phase, device class, band, version, every field of the execution, every lever's
// spelled value or identity, the candidate's identity, the partial spelling) must leave EVERY cell of EVERY tier exactly
// as with no receipt at all. The run side too: the receipt kept fixed while the run changes one execution field or one
// declared identity (candidate or reference-only) resolves as that run does with no receipt.
void enumerate(const p::Engine& registered, EnumerationCount& count) {
    const std::string tag(registered.tag);
    ++count.engines;
    for (const auto& lever : registered.table)
        require(p::identity_defect(lever).empty(), tag + ": undeclared identity " + p::identity_defect(lever));
    std::vector<p::Lever> rows(registered.table.begin(), registered.table.end());
    for (auto& row : rows)
        if (p::ladder_judged(row)) {
            row.ready.fill(true);
            row.runnable.fill(true);
            row.provisional.fill(false);
        }
    std::vector<p::LadderRow> grants;
    for (const auto& row : registered.ladder)
        if (row.incumbent) grants.push_back(row);
    const std::string execution = p::ladder_execution(kEnumExecution);
    const p::Regime regime{p::Phase::prefill, 8192, 1, "dev-class", execution};
    auto engine = registered;
    engine.table = rows;
    engine.ladder = grants;
    const auto none = spelled_resolution(engine, regime);
    // Review r75 P1-b: a grant whose identity is not the one the table declares for its value grants nothing: every cell
    // resolves exactly as with that grant removed.
    for (std::size_t g = 0; g < grants.size(); ++g) {
        std::vector<p::LadderRow> with = grants, without;
        with[g].identity = keep(std::string(grants[g].identity) + "-changed");
        for (std::size_t i = 0; i < grants.size(); ++i)
            if (i != g) without.push_back(grants[i]);
        auto changed = engine, removed = engine;
        changed.ladder = with;
        removed.ladder = without;
        require(spelled_resolution(changed, regime) == spelled_resolution(removed, regime),
                tag + "/" + std::string(grants[g].lever) + ": a grant of another identity still decides");
        ++count.mutations;
    }
    const auto admit = [](const p::Lever& l, unsigned c) { return p::column_admission(l, c, false, false); };
    const auto arms = p::ladder_arms(std::span<const p::Lever>(rows), std::span<const p::LadderRow>(grants), regime, admit);
    const auto outcome = [&](const p::LadderRow& r, const p::Engine& base, const p::Regime& run) {
        std::vector<p::LadderRow> v = grants;
        v.push_back(r);
        auto e = base;
        e.ladder = v;
        return spelled_resolution(e, run);
    };
    for (const auto tier : {p::Name::fast, p::Name::turbo}) {
        if (tier == p::Name::turbo && registered.turbo_version.empty()) continue;
        const auto t = p::LadderArms::slot(tier), below = p::LadderArms::slot(p::tier_below(tier));
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto& lever = rows[i];
            if (!p::ladder_judged(lever)) continue;
            const auto column = p::value_column(lever, tier);
            if (tier == p::Name::turbo && column == static_cast<unsigned>(p::Name::fast)) continue;
            const std::string wanted(lever.value[column]);
            const std::string lower = arms.values[below][i];
            if (wanted.empty() || wanted == lower) continue;
            const std::string where = tag + "/" + std::string(lever.name) + "=" + wanted + " " + p::name(tier);
            const bool granted = arms.values[t][i] == wanted;
            const auto reference = arms.over(tier, lever.name, wanted);
            require(!reference.empty(), where + ": the cell's complete key is spelled");
            if (reference.empty()) continue;
            p::LadderRow receipt{keep(std::string(lever.name)), keep(wanted), tier, regime.phase, regime.device, 1,
                                 p::kNoContextBound, 1, p::kNoLaneBound, granted ? 0.5 : 1.5, 0.01, "enum/receipt"};
            receipt.over = keep(reference);
            {
                std::vector<p::LadderRow> v = grants;
                v.push_back(receipt);
                auto e = engine;
                e.ladder = v;
                const auto decided = resolve(e, tier, regime);
                require(decided.find(lever.name)->value == (granted ? lower : wanted),
                        where + ": the decisive receipt decides (" + decided.find(lever.name)->ladder + ")");
            }
            ++count.cells;
            const auto inert = [&](const p::LadderRow& r, const std::string& what) {
                ++count.mutations;
                require(outcome(r, engine, regime) == none, where + ": a receipt with " + what + " changed a cell");
            };
            auto r = receipt;
            r.lever = keep(std::string(lever.name) + "-other");
            inert(r, "another lever");
            r = receipt;
            r.value = keep(wanted + "-other");
            inert(r, "another value");
            // Another tier: the cell under test resolves as with no receipt. (The receipt may still be a complete key of
            // the other tier's cell when both tiers compare the very same pair of executions: that is not this cell.)
            r = receipt;
            r.tier = tier == p::Name::fast ? p::Name::turbo : p::Name::fast;
            ++count.mutations;
            {
                std::vector<p::LadderRow> v = grants;
                v.push_back(r);
                auto e = engine;
                e.ladder = v;
                require(spelled_resolution(e, regime, tier) == spelled_resolution(engine, regime, tier),
                        where + ": a receipt of another tier changed the cell");
            }
            r = receipt;
            r.phase = p::Phase::decode;
            inert(r, "another phase");
            r = receipt;
            r.device = "dev-other";
            inert(r, "another device class");
            r = receipt;
            r.min_context = regime.context + 1U;
            inert(r, "a context band outside the run");
            r = receipt;
            r.min_lanes = regime.lanes + 1U;
            inert(r, "a lane band outside the run");
            r = receipt;
            r.over = keep("ladder-arm/v3" + reference.substr(p::kLadderArmVersion.size()));
            inert(r, "another spelling version");
            r = receipt;
            r.over = keep(std::string(p::kLadderArmVersion) + ";" + execution + reference.substr(reference.find(";candidate=")));
            inert(r, "a partial key (execution and candidate only)");
            r = receipt;
            const auto tier_at = reference.rfind(";tier=");
            r.over = keep(reference.substr(0, tier_at - 1U) + "-other}" + reference.substr(tier_at));
            inert(r, "another candidate identity");
            const std::array<void (*)(p::LadderExecution&), 11> fields{
                +[](p::LadderExecution& e) { e.backend = "cuda"; },
                +[](p::LadderExecution& e) { e.device = "product-b"; },
                +[](p::LadderExecution& e) { e.workload.prompt_tokens = 4096; },
                +[](p::LadderExecution& e) { e.workload.context_capacity = 32768; },
                +[](p::LadderExecution& e) { e.workload.lanes = 2; },
                +[](p::LadderExecution& e) { e.workload.prefill_chunk = 4096; },
                +[](p::LadderExecution& e) { e.workload.prefill_rows = 1024; },
                +[](p::LadderExecution& e) { e.workload.decode_rows = 2; },
                +[](p::LadderExecution& e) { e.workload.verify_rows = 4; },
                +[](p::LadderExecution& e) { e.workload.generated_tokens = 256; },
                +[](p::LadderExecution& e) { e.settings = "settings-b"; }};
            for (std::size_t f = 0; f < fields.size(); ++f) {
                auto other = kEnumExecution;
                fields[f](other);
                const auto spelled = p::ladder_execution(other);
                r = receipt;
                r.over = keep(with_execution(reference, spelled));
                inert(r, "execution field " + std::to_string(f) + " changed");
                // Run side: the receipt fixed, a run on the other execution resolves as with no receipt there.
                auto run = regime;
                run.execution = spelled;
                ++count.mutations;
                require(outcome(receipt, engine, run) == spelled_resolution(engine, run),
                        where + ": the receipt decided a run on another execution (field " + std::to_string(f) + ")");
            }
            for (std::size_t j = 0; j < rows.size(); ++j) {
                // The value lever j is spelled with in this key, and its declared identity entry.
                const std::string spelled = j < i ? arms.values[t][j] : j == i ? arms.values[below][j] : arms.incumbent[t][j];
                auto renamed = arms;
                for (auto& entry : renamed.identities[j])
                    if (entry.value == spelled) entry.identity = keep(std::string(entry.identity) + "-other");
                r = receipt;
                r.over = keep(renamed.over(tier, lever.name, wanted));
                inert(r, "lever " + std::string(rows[j].name) + " at another identity");
                auto revalued = arms;
                for (auto& entry : revalued.identities[j])
                    if (entry.value == spelled) entry.value = "zz-other";
                auto& slot_value = j < i ? revalued.values[t][j] : j == i ? revalued.values[below][j] : revalued.incumbent[t][j];
                slot_value = "zz-other";
                r = receipt;
                r.over = keep(revalued.over(tier, lever.name, wanted));
                inert(r, "lever " + std::string(rows[j].name) + " at another value");
                // Run side: the table declares another identity for that spelled value (a new implementation).
                auto table = rows;
                for (auto& entry : table[j].identity)
                    if (entry.value == spelled) entry.identity = keep(std::string(entry.identity) + "-v2");
                auto changed = engine;
                changed.table = table;
                ++count.mutations;
                require(outcome(receipt, changed, regime) == spelled_resolution(changed, regime),
                        where + ": the receipt decided after lever " + std::string(rows[j].name) + " changed identity");
            }
        }
    }
}

// Review r63 reproductions on the shipped Qwen dense CUDA table (the shared resolver every adapter reads).
void r63_reproductions() {
    namespace qd = strix::qwen35;
    std::vector<p::Lever> rows(std::begin(qd::profile::table), std::end(qd::profile::table));
    // Every lossy or timed column admitted, as the adapter admits it from its quality evidence (qualified FAST cells,
    // TURBO gates): what is left to decide is the timing.
    for (auto& row : rows)
        if (p::ladder_judged(row)) {
            row.ready.fill(true);
            row.runnable.fill(true);
            row.provisional.fill(false);
        }
    const auto at = [&](std::string_view name) {
        for (std::size_t i = 0; i < rows.size(); ++i)
            if (rows[i].name == name) return i;
        return rows.size();
    };
    const auto spelled = [](std::uint32_t generated, std::string_view product) {
        auto e = kEnumExecution;
        e.backend = "cuda";
        e.device = product;
        e.workload.generated_tokens = generated;
        return p::ladder_execution(e);
    };
    const auto exec128 = spelled(128, "product-a"), exec256 = spelled(256, "product-a"), other = spelled(128, "product-b");
    const p::Regime run128{p::Phase::prefill, 8192, 1, "sm_120", exec128};
    const auto admit = [](const p::Lever& l, unsigned c) { return p::column_admission(l, c, false, false); };
    const auto with_cli = [&](const p::Engine& engine, p::Name tier, p::Pairs cli) {
        p::Inputs in;
        in.profile = tier;
        in.regime = run128;
        in.env = [](const char*) -> const char* { return nullptr; };
        in.cli = std::move(cli);
        return p::resolve(engine, in);
    };
    // (1) A receipt keeps deciding only the implementation it names: dense_activation=on measured at identity v1.
    {
        const std::vector<p::LadderRow> none;
        const auto arms = p::ladder_arms(std::span<const p::Lever>(rows), std::span<const p::LadderRow>(none), run128, admit);
        p::LadderRow gain{"dense_activation", "on", p::Name::turbo, p::Phase::prefill, "sm_120", 1, p::kNoContextBound, 1,
                          p::kNoLaneBound, 1.2, 0.01, "r63/dense-v1"};
        gain.over = keep(arms.over(p::Name::turbo, "dense_activation", "on"));
        const std::array ladder{gain};
        p::Engine engine{"qwen35", "", rows, false, qd::profile::kTurboVersion, {}, {}, ladder};
        require(resolve(engine, p::Name::turbo, run128).find("dense_activation")->value == "on", "r63 (1) the v1 receipt decides v1");
        auto v2 = rows;
        for (auto& entry : v2[at("dense_activation")].identity)
            if (entry.value == "on") entry.identity = "dense-window-fp4-fp8-v2";
        engine.table = v2;
        require(resolve(engine, p::Name::turbo, run128).find("dense_activation")->value == "off",
                "r63 (1) the v1 timing row does not decide the v2 implementation");
        // The reference side: an explicit prefill_gdn=chunk16 is another implementation than the declared chunk16tc,
        // and an explicit value the table does not identify spells no key at all.
        engine.table = rows;
        require(with_cli(engine, p::Name::turbo, {{"prefill_gdn", "chunk16"}}).find("dense_activation")->value == "off",
                "r63 (1) an explicit chunk16 reference does not reuse the chunk16tc receipt");
        const auto unknown = with_cli(engine, p::Name::turbo, {{"prefill_gdn", "chunk8"}});
        require(unknown.find("dense_activation")->value == "off" && unknown.find("dense_activation")->ladder_reference.empty(),
                "r63 (1) an unidentified explicit value spells no key");
    }
    // (3) An out-of-scope timing row never revokes the frozen FAST grant prefill_arith=fast4.
    {
        const std::vector<p::LadderRow> grant_only{p::incumbent_cell("prefill_arith", "fast4", p::Name::fast, "r63/frozen", "nvfp4-tile-mma-f32-v1")};
        const auto arms = p::ladder_arms(std::span<const p::Lever>(rows), std::span<const p::LadderRow>(grant_only), run128, admit);
        for (const double speedup : {1.2, 0.8}) {
            p::LadderRow timed{"prefill_arith", "fast4", p::Name::fast, p::Phase::prefill, "sm_120", 1, p::kNoContextBound,
                               1, p::kNoLaneBound, speedup, 0.01, "r63/n-gen-128"};
            timed.over = keep(arms.over(p::Name::fast, "prefill_arith", "fast4"));
            const std::vector<p::LadderRow> ladder{grant_only[0], timed};
            const p::Engine engine{"qwen35", "", rows, false, qd::profile::kTurboVersion, {}, {}, ladder};
            const auto value = [&](std::string_view execution) {
                return resolve(engine, p::Name::fast, {p::Phase::prefill, 8192, 1, "sm_120", execution}).find("prefill_arith")->value;
            };
            require(value(exec128) == (speedup > 1.0 ? "fast4" : "exact"), "r63 (3) the matching execution is decided by its row");
            require(value(exec256) == "fast4", "r63 (3) a second generation length keeps the frozen grant: " + std::string(value(exec256)));
            require(value(other) == "fast4", "r63 (3) a second product keeps the frozen grant");
            require(value({}) == "fast4", "r63 (3) an unknown execution keeps the frozen grant");
        }
    }
    // (2) The drafter schedule (timed exact lever) is judged over the complete lower execution: a receipt measured over
    // the FAST split_k=fast does not admit it under an explicit split_k=exact, and the truncated spelling never does.
    {
        const std::vector<p::LadderRow> none;
        const auto arms = p::ladder_arms(std::span<const p::Lever>(rows), std::span<const p::LadderRow>(none), run128, admit);
        p::LadderRow gain{"drafter_window", "15", p::Name::turbo, p::Phase::prefill, "sm_120", 1, p::kNoContextBound, 1,
                          p::kNoLaneBound, 1.2, 0.01, "r63/drafter"};
        // split_k runs its FAST value through its incumbent grant (as in the adapter), so the schedule's key spells it.
        const auto split_grant = p::incumbent_cell("split_k", "fast", p::Name::fast, "r63/split-k-grant", "nvfp4-splitk-fast-v1");
        const std::vector<p::LadderRow> granted{split_grant};
        const auto with_split = p::ladder_arms(std::span<const p::Lever>(rows), std::span<const p::LadderRow>(granted),
                                               run128, admit);
        gain.over = keep(with_split.over(p::Name::turbo, "drafter_window", "15"));
        std::array ladder{split_grant, gain};
        p::Engine engine{"qwen35", "", rows, false, qd::profile::kTurboVersion, {}, {}, ladder};
        require(resolve(engine, p::Name::turbo, run128).find("drafter_window")->value == "15", "r63 (2) the full key decides");
        require(with_cli(engine, p::Name::turbo, {{"split_k", "exact"}}).find("drafter_window")->value == "7",
                "r63 (2) another companion setting falls back");
        require(with_cli(engine, p::Name::turbo, {{"dense_activation", "on"}}).find("drafter_window")->value == "7",
                "r63 (2) an explicit dense_activation=on falls back");
        ladder[1].over = keep(std::string(p::kLadderArmVersion) + ";" + exec128 +
                              ";drafter_window=7{chain-window-7-v1};candidate=drafter_window=15{" +
                              std::string(qd::profile::kDraftIdentity) + "}");
        require(resolve(engine, p::Name::turbo, run128).find("drafter_window")->value == "7", "r63 (2) the truncated receipt is rejected");
        // Without a timing receipt the schedule never runs: no arm-less judgement counts as incumbent.
        engine.ladder = std::span<const p::LadderRow>(&ladder[0], 1);
        require(resolve(engine, p::Name::turbo, run128).find("drafter_window")->value == "7", "r63 (2) no receipt, no schedule change");
    }
}

// MiniMax IU4 layer set: the TURBO cell is a candidate (no timing row: the builtin runs, also under --allow-pending);
// an explicit positive layer set reaches the prefill identity; the legacy format keeps its layers 38-59.
void minimax_i4_binding() {
    namespace mm = strix::minimax::defaults;
    p::Inputs in;
    in.profile = p::Name::turbo;
    in.allow_pending = true;
    in.regime = {p::Phase::prefill, 8192, 1, "gfx1151", "fixture-exec"};
    in.env = [](const char*) -> const char* { return nullptr; };
    const auto r = p::resolve(mm::profile_engine(), in);
    const auto* layers = r.find("iu4_layers");
    require(layers && layers->lever->runnable[3], "MiniMax IU4 mask has a runnable TURBO binding");
    require(layers && layers->value == "none", "MiniMax IU4 remains off in the unmeasured regime");
    const auto explicit_layers = [](const char* k) -> const char* {
        return std::string_view(k) == "STRIX_MM_IU4_LAYERS" ? "43-59" : nullptr;
    };
    require(mm::prefill_identity(explicit_layers).find("moe_prefill_format=fast_i4:layers=43-59") != std::string::npos,
            "an explicit MiniMax IU4 mask reaches the execution identity");
    const auto run = [](const std::map<std::string, std::string>& env) {
        return mm::prefill_policy([&](const char* k) -> const char* {
            const auto it = env.find(k);
            return it == env.end() ? nullptr : it->second.c_str();
        });
    };
    const auto legacy = run({{"STRIX_MMX_MODEL_FORMAT", "fast_i4"}});
    require(legacy.fast_i4 && legacy.iu4_layers.count() == 22 && !legacy.iu4_layers.test(37) && legacy.iu4_layers.test(38),
            "MiniMax legacy IU4 mask retains layers 38-59");
    require(mm::prefill_identity([](const char* k) -> const char* {
                return std::string_view(k) == "STRIX_MMX_MODEL_FORMAT" ? "fast_i4" : nullptr;
            }).find("moe_prefill_format=fast_i4") != std::string::npos,
            "the legacy format keeps its identity");
    const auto selected = run({{"STRIX_MMX_MODEL_FORMAT", "fast_i4"}, {"STRIX_MMX_FAST_I4_EXL3_LAYERS", "none"},
                               {"STRIX_MM_IU4_LAYERS", "43,58-59"}});
    require(selected.fast_i4 && selected.iu4_layers.count() == 3 && selected.iu4_layers.test(43) && selected.iu4_layers.test(59),
            "MiniMax explicit positive mask overrides the legacy exclusion mask");
    const auto off = run({{"STRIX_MMX_MODEL_FORMAT", "exl3"}, {"STRIX_MM_IU4_LAYERS", "38-59"}});
    require(!off.fast_i4 && off.iu4_layers.count() == 0, "MiniMax explicit EXL3 refuses an IU4 mask");
    require(!run({{"STRIX_MM_IU4_LAYERS", "none"}}).fast_i4, "an empty explicit mask leaves EXL3 selected");
    require(!run({{"STRIX_MMX_MODEL_FORMAT", "fast_i4"}, {"STRIX_MM_IU4_LAYERS", "none"}}).fast_i4,
            "an empty explicit mask also disables the legacy IU4 format");
    require(!run({}).fast_i4 && run({}).iu4_layers.count() == 0, "no setting: EXL3");
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 3) return 2;
    if (argc > 3 && std::string_view(argv[3]) == "--write") {
        incumbents(argv[2], true);
        return 0;
    }
    incumbents(argv[2], false);
    verdicts();
    bound_evidence();
    defects();
    shipped_rows();
    execution_binding();
    minimax_i4_binding();
    custom_adapters();
    cuda_cli_server();
    namespace q = strix::qwen;
    namespace qd = strix::qwen35;
    namespace mm = strix::minimax::defaults;
    const p::Engine engines[] = {
        {"qwen", "", q::profile::levers(), false, q::profile::kTurboVersion, {}, {}, q::profile::ladder_rows},
        {"qwen-cuda", "", q::cuda_profile::levers(), false, q::profile::kTurboVersion, {}, {}, q::cuda_profile::ladder_rows},
        {"deepseek41-pair", "", strix::deepseek41::pair_defaults::profile_table, false, "ds41-turbo-v1", {}, {},
         strix::deepseek41::pair_defaults::ladder_rows},
        {"deepseek41-rtx", "", strix::ds41::rtx::profile::shared_rows()},
        strix::glm::rank_defaults::product_profile_engine,
        mm::profile_engine(),
        {"qwen35", "", qd::profile::table, false, qd::profile::kTurboVersion},
        qd::hip_profile::engine(qd::hip_profile::App::decode),
        qd::hip_profile::engine(qd::hip_profile::App::spec),
        strix::hy4::profile::engine(),
        strix::inkling2::kMiniProfile,
    };
    for (const auto& registered : strix::model_registry::load(argv[1])) {
        const auto tag = registered.profile_tag;
        require(tag == "gemma4" || std::ranges::any_of(engines, [&](const auto& engine) {
                    return engine.tag == tag || (tag == "hy4" && engine.tag == "hy4-hip");
                }), "profile registry entry must be covered: " + tag);
    }
    turbo_nests_fast(engines);
    paired_rows();
    hy4_new_quality_rows();
    for (const auto& engine : engines) walk(engine);
    // Review r63 D: every engine table the ladder reads, the custom adapters' tables through their shared row view.
    EnumerationCount count;
    const auto each = [&](const p::Engine& engine) {
        try {
            enumerate(engine, count);
        } catch (const std::exception& error) {
            require(false, std::string(engine.tag) + " enumeration: " + error.what());
        }
    };
    for (const auto& engine : engines) each(engine);
    const auto gemma_rows = shared_view(strix::gemma4::profile::levers());
    each({"gemma4", "", gemma_rows, false, {}});
    const auto glm_rtx_rows = shared_view(strix::glm::rtxpf::profile::levers());
    each({"glm-rtx", "", glm_rtx_rows, false, {}});
    require(count.engines == std::size(engines) + 2U && count.cells > 0U, "enumeration covered every engine table");
    std::printf("profile ladder enumeration: %zu engine tables, %zu judged cells, %zu one-field mutations" "\n",
                count.engines, count.cells, count.mutations);
    r63_reproductions();
    if (failures) return 1;
    std::puts("profile ladder tests: PASS");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
